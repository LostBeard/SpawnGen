using System.Text.Json;
using ILGPU;
using ILGPU.Runtime.CPU;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>Curve (ComfyUI's CURVE) against ComfyUI's own CurveInput (Fixtures/curves, make_curves.py), and the GLSL Shader node on the CPU accelerator.</summary>
public class GlslShaderNodeTests
{
    [Test]
    public void Curves_give_ComfyUIs_lookup_tables()
    {
        var cases = JsonDocument.Parse(File.ReadAllText(Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "curves", "curves.json"))).RootElement;
        int n = 0;
        foreach (var c in cases.EnumerateArray())
        {
            var curve = Curve.FromRaw(JsonLiteral.FromJson(System.Text.Json.Nodes.JsonNode.Parse(c.GetProperty("curve").GetRawText())));
            var expected = c.GetProperty("lut").EnumerateArray().Select(v => v.GetDouble()).ToArray();
            Assert.That(curve.ToLut(256), Is.EqualTo(expected).Within(1e-12), $"curve {n++}: {curve}");
        }
    }

    sealed class Services(IResultStore store, MlRuntime runtime) : IServiceProvider
    {
        public object? GetService(Type t) => t == typeof(IResultStore) ? store : t == typeof(MlRuntime) ? runtime : null;
    }

    static NodeContext Context(IServiceProvider services) =>
        (NodeContext)Activator.CreateInstance(typeof(NodeContext), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, null,
            ["1", "GLSLShader", services, CancellationToken.None, null, new Lazy<string>(() => "{}"), new Dictionary<string, string>(), null, null, 0], null)!;

    const string Shader = """
        #version 300 es
        precision highp float;
        uniform sampler2D u_image0;
        uniform sampler2D u_image1;
        uniform sampler2D u_curve0;
        uniform float u_float0;
        uniform int u_int0;
        uniform bool u_bool0;
        uniform vec2 u_resolution;
        in vec2 v_texCoord;
        layout(location = 0) out vec4 fragColor0;
        layout(location = 2) out vec4 fragColor2;
        void main() {
            vec4 a = texture(u_image0, v_texCoord);
            vec4 b = texture(u_image1, v_texCoord);
            float lut = texelFetch(u_curve0, ivec2(255, 0), 0).r;
            fragColor0 = vec4(a.r * u_float0, b.g, lut, u_bool0 ? 1.0 : 0.5);
            fragColor2 = vec4(float(u_int0) / 10.0, u_resolution / 100.0, 1.0);
        }
        """;

    [Test]
    public async Task The_node_renders_every_picture_of_the_batch_with_its_uniforms_on_the_CPU_accelerator()
    {
        using var ilgpu = ILGPU.Context.Create(b => b.CPU());
        using var acc = ilgpu.CreateCPUAccelerator(0);
        using var runtime = new MlRuntime(new FolderModelSource(Path.GetTempPath()), acc);
        var services = new Services(new MemoryResultStore(), runtime);
        var a = new ImageBatch(2, 2, 3, 3);
        for (int i = 0; i < a.Data.Length; i++) a.Data[i] = (i % 7) / 7f;
        var b = new ImageBatch(2, 2, 3, 4);
        for (int i = 0; i < b.Data.Length; i++) b.Data[i] = 0.25f;
        var curve = new Curve([(0.0, 0.0), (1.0, 0.75)], "linear");

        var (o0, o1, o2, o3, ui) = await new GLSLShader().ExecuteAsync(Context(services), services, Shader, new DynamicChoice("from_input", new Dictionary<string, object?>()),
            [a, b], [2.0], [7], [true], [curve]);
        Assert.That(o0.Shape, Is.EqualTo(new[] { 2, 2, 3, 4 }), "RGBA at the first image's size, one per picture");
        for (int p = 0; p < 2 * 2 * 3; p++)
        {
            Assert.That(o0.Data[p * 4], Is.EqualTo(a.Data[p * 3] * 2f).Within(1e-6), "u_image0 at the pixel centre, times u_float0");
            Assert.That(o0.Data[p * 4 + 1], Is.EqualTo(0.25f).Within(1e-6), "u_image1");
            Assert.That(o0.Data[p * 4 + 2], Is.EqualTo(0.75f).Within(1e-6), "the curve's last entry");
            Assert.That(o0.Data[p * 4 + 3], Is.EqualTo(1f), "u_bool0");
            Assert.That(o2.Data[(p * 4)..(p * 4 + 4)], Is.EqualTo(new[] { 0.7f, 0.03f, 0.02f, 1f }).Within(1e-6), "fragColor2: u_int0 and u_resolution");
        }
        Assert.That(o1.Data.Concat(o3.Data), Is.All.EqualTo(0f), "outputs the shader does not write are black");
        Assert.That(((List<object>)ui["images"]!).Count, Is.EqualTo(2), "a preview of each picture of the first output");

        var (custom, _, _, _, _) = await new GLSLShader().ExecuteAsync(Context(services), services, Shader,
            new DynamicChoice("custom", new Dictionary<string, object?> { ["width"] = 5L, ["height"] = 4L }), [a], [], [], [], []);
        Assert.That(custom.Shape, Is.EqualTo(new[] { 2, 4, 5, 4 }), "custom size; missing uniforms read 0");

        var bad = """
            #version 300 es
            precision highp float;
            out vec4 fragColor0;
            void main() { float x = 1; fragColor0 = vec4(x); }
            """;
        Assert.That(() => new GLSLShader().ExecuteAsync(Context(services), services, bad, new DynamicChoice("from_input", new Dictionary<string, object?>()), [a], [], [], [], []),
            Throws.InvalidOperationException.With.Message.Contains("ERROR: 0:4:").And.Message.Contains("cannot convert from 'int' to 'float'"));
    }
}
