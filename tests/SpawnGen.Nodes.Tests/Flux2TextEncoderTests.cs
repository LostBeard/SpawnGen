using System.Text.Json;
using SpawnGen.Engine.Nodes;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// FLUX.2's text encoder (Load CLIP type "flux2", ComfyUI's qwen_3_4b.safetensors) against ComfyUI's CLIP Text Encode (bef11fc, CPU, float32;
/// Fixtures/flux2_text: the first 34 of the 512 rows for three prompts, and the attention mask's real-token counts).
/// </summary>
public class Flux2TextEncoderTests
{
    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "flux2_text");
    static JsonElement Cases => JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "cases.json"))).RootElement.GetProperty("cases");

    static float[] Raw(string name)
    {
        var b = File.ReadAllBytes(Path.Combine(Dir, name));
        var d = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, d, 0, b.Length);
        return d;
    }

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }

    static double RelRms(ReadOnlySpan<float> ours, ReadOnlySpan<float> theirs)
    {
        double num = 0, den = 0;
        for (int i = 0; i < ours.Length; i++) { double d = ours[i] - theirs[i]; num += d * d; den += (double)theirs[i] * theirs[i]; }
        return Math.Sqrt(num / den);
    }

    [Test, Explicit("Needs SPAWNGEN_MODELS_ROOT with text_encoders/qwen_3_4b.safetensors (Comfy-Org's) and a GPU; the Qwen tokenizer downloads on first use.")]
    public async Task The_conditioning_is_ComfyUIs()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        ModelFolders.Scan(root!);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var services = new Services(runtime);
        var context = (NodeContext)Activator.CreateInstance(typeof(NodeContext), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, null,
            ["1", "X", services, CancellationToken.None, null, new Lazy<string>(() => "{}"), new Dictionary<string, string>(), null, null, 0], null)!;
        var clip = await new CLIPLoader().ExecuteAsync("qwen_3_4b.safetensors", "flux2", context, services);
        var model = (Flux2TextEncoderModel)clip.Encoder;
        var report = new List<string>();
        foreach (var c in Cases.EnumerateObject())
        {
            var prompt = c.Value.GetProperty("prompt").GetString()!;
            int tokens = c.Value.GetProperty("tokens").GetInt32();
            var (_, real) = await model.TokenizeAsync(prompt, CancellationToken.None);
            Assert.That(real, Is.EqualTo(tokens), $"{c.Name}: ComfyUI's attention mask counts {tokens} real tokens");
            var encoding = await model.EncodePromptAsync(prompt, CancellationToken.None);
            Assert.That(encoding.Shape, Is.EqualTo(new[] { 1, 512, 7680 }));
            var theirs = Raw(c.Value.GetProperty("file").GetString()!);
            double realRows = RelRms(encoding.Context.AsSpan(0, tokens * 7680), theirs.AsSpan(0, tokens * 7680));
            double padRows = RelRms(encoding.Context.AsSpan(tokens * 7680, (34 - tokens) * 7680), theirs.AsSpan(tokens * 7680, (34 - tokens) * 7680));
            report.Add($"{c.Name}: {tokens} tokens, real rows {realRows:E2}, padding rows {padRows:E2}");
            // Measured on the RX 590 (OpenCL), 2026-11-14: real rows 1.3e-6, padding rows 2e-5 .. 8e-5 relative RMS (both float32).
            Assert.That(realRows, Is.LessThan(1e-4), c.Name);
            Assert.That(padRows, Is.LessThan(1e-3), c.Name);
        }
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: " + string.Join("; ", report));
    }
}
