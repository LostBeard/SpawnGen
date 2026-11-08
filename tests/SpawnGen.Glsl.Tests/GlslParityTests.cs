using System.IO.Compression;
using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.OpenCL;
using SpawnGen.Glsl;

namespace SpawnGen.Glsl.Tests;

/// <summary>
/// Shaders compiled to ILGPU kernels against ComfyUI's own GLSL renderer (Fixtures/make_glsl.py: OpenGL ES through ANGLE on the RX 590). SpawnGen's test
/// shaders (Fixtures/*.frag) run always on the CPU accelerator; ComfyUI's blueprint shaders are read from a ComfyUI checkout's blueprints folder
/// (SPAWNGEN_COMFY_BLUEPRINTS), since they are not stored here.
/// </summary>
public class GlslParityTests
{
    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures");

    static JsonElement Fixture()
    {
        using var gz = new GZipStream(File.OpenRead(Path.Combine(Dir, "glsl.json.gz")), CompressionMode.Decompress);
        return JsonDocument.Parse(gz).RootElement.Clone();
    }

    static float[] F32(string b64)
    {
        var bytes = Convert.FromBase64String(b64);
        var f = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, f, 0, bytes.Length);
        return f;
    }

    public static IEnumerable<string> OwnCases() => ["own_features_math", "own_features_control", "own_features_multipass"];
    public static IEnumerable<string> BlueprintCases() =>
    [
        "brightness_contrast", "chromatic_aberration", "color_adjustment", "color_balance", "color_balance_raw", "color_curves", "edge_preserving_blur",
        "film_grain_smooth", "film_grain_grainy", "glow", "glow_screen", "hue_saturation", "hue_saturation_reds", "hue_saturation_colorize",
        "blur_gaussian", "blur_box", "blur_radial", "image_channels", "image_levels", "sharpen", "unsharp_mask",
    ];

    [TestCaseSource(nameof(OwnCases))]
    public async Task Own_shader_matches_ComfyUI_on_the_CPU_accelerator(string name)
    {
        using var context = Context.Create(b => b.CPU());
        using var acc = context.CreateCPUAccelerator(0);
        await Check(acc, name, File.ReadAllText(Path.Combine(Dir, name["own_".Length..] + ".frag")));
    }

    [TestCaseSource(nameof(BlueprintCases))]
    [Explicit("Needs SPAWNGEN_COMFY_BLUEPRINTS: ComfyUI's blueprints folder (its GLSL shaders are ComfyUI's and are not stored here).")]
    public async Task Blueprint_shader_matches_ComfyUI_on_the_CPU_accelerator(string name)
    {
        using var context = Context.Create(b => b.CPU());
        using var acc = context.CreateCPUAccelerator(0);
        await Check(acc, name, BlueprintShader(name));
    }

    [TestCaseSource(nameof(BlueprintCases))]
    [Explicit("Needs SPAWNGEN_COMFY_BLUEPRINTS and an OpenCL GPU.")]
    public async Task Blueprint_shader_matches_ComfyUI_on_OpenCL(string name)
    {
        using var context = Context.Create(b => b.OpenCL());
        using var acc = context.Devices.First(d => d.AcceleratorType == AcceleratorType.OpenCL).CreateAccelerator(context);
        TestContext.Out.WriteLine(acc.Name);
        await Check(acc, name, BlueprintShader(name));
    }

    [TestCaseSource(nameof(OwnCases))]
    [Explicit("Needs an OpenCL GPU.")]
    public async Task Own_shader_matches_ComfyUI_on_OpenCL(string name)
    {
        using var context = Context.Create(b => b.OpenCL());
        using var acc = context.Devices.First(d => d.AcceleratorType == AcceleratorType.OpenCL).CreateAccelerator(context);
        await Check(acc, name, File.ReadAllText(Path.Combine(Dir, name["own_".Length..] + ".frag")));
    }

    static string BlueprintShader(string caseName)
    {
        var folder = Environment.GetEnvironmentVariable("SPAWNGEN_COMFY_BLUEPRINTS");
        if (string.IsNullOrEmpty(folder)) Assert.Ignore("SPAWNGEN_COMFY_BLUEPRINTS is not set.");
        var blueprint = Fixture().GetProperty("cases").EnumerateArray().First(c => c.GetProperty("name").GetString() == caseName).GetProperty("shader").GetString()!;
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(folder!, blueprint + ".json")));
        string? found = null;
        void Walk(JsonElement e)
        {
            if (found is not null) return;
            if (e.ValueKind == JsonValueKind.Object)
            {
                if (e.TryGetProperty("type", out var t) && t.ValueKind == JsonValueKind.String && t.GetString() == "GLSLShader" && e.TryGetProperty("widgets_values", out var w))
                    found = w[0].GetString();
                foreach (var p in e.EnumerateObject()) Walk(p.Value);
            }
            else if (e.ValueKind == JsonValueKind.Array) foreach (var x in e.EnumerateArray()) Walk(x);
        }
        Walk(doc.RootElement);
        return found ?? throw new InvalidOperationException($"No GLSL shader in {blueprint}.");
    }

    static async Task Check(Accelerator acc, string name, string source)
    {
        var fx = Fixture();
        int w = fx.GetProperty("width").GetInt32(), h = fx.GetProperty("height").GetInt32();
        var picture = new ShaderImage(F32(fx.GetProperty("picture").GetString()!), w, h, 3);
        var curves = fx.GetProperty("curves").EnumerateArray().Select(c => F32(c.GetString()!)).ToArray();
        var c = fx.GetProperty("cases").EnumerateArray().First(c => c.GetProperty("name").GetString() == name);
        var floats = c.GetProperty("floats").EnumerateArray().Select(v => v.GetSingle()).ToArray();
        var ints = c.GetProperty("ints").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        var bools = c.GetProperty("bools").EnumerateArray().Select(v => v.GetBoolean()).ToArray();
        var expected = c.GetProperty("outputs").EnumerateArray().Select(o => F32(o.GetString()!)).ToArray();

        var shader = GlslShader.Compile(source);
        var outputs = await shader.RunAsync(acc, w, h, [picture], floats, ints, bools, c.GetProperty("curves").GetBoolean() ? curves : []);
        double worst = 0;
        for (int k = 0; k < expected.Length; k++)
        {
            // ComfyUI pads the outputs a shader does not name with black.
            var ours = k < outputs.Length ? outputs[k] : new float[w * h * 4];
            double max = 0, sum = 0;
            for (int i = 0; i < ours.Length; i++) { double d = Math.Abs(ours[i] - expected[k][i]); sum += d; max = Math.Max(max, d); }
            TestContext.Out.WriteLine($"{name} output {k}: largest difference {max * 255:F3}/255, mean {sum / ours.Length * 255:F4}/255");
            worst = Math.Max(worst, max);
        }
        Assert.That(worst * 255, Is.LessThan(2.0), "largest difference in 1/255 steps");
    }
}
