using System.Text.Json;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// ComfyUI's split FLUX.2 Klein graph against ComfyUI (bef11fc, CPU float32, flux-2-klein-4b.safetensors in bf16; Fixtures/flux2_klein): the
/// model's own schedules, one model call, and the distilled template graph end to end. SpawnGen runs the fp8 file on an 8 GB card (the bf16
/// transformer is 7.8 GB), so model outputs are compared with the fp8 file's tolerance (the library measures it at 2.7e-2 per matrix).
/// </summary>
public class Flux2KleinTests
{
    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "flux2_klein");
    static JsonElement Cases => JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "cases.json"))).RootElement;

    static float[] Raw(string name)
    {
        var b = File.ReadAllBytes(Path.Combine(Dir, name));
        var d = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, d, 0, b.Length);
        return d;
    }

    static double RelRms(float[] ours, float[] theirs)
    {
        double num = 0, den = 0;
        for (int i = 0; i < ours.Length; i++) { double d = ours[i] - theirs[i]; num += d * d; den += (double)theirs[i] * theirs[i]; }
        return Math.Sqrt(num / den);
    }

    sealed class Capture : IExecutionObserver
    {
        public Dictionary<string, NodeOutputs> Outputs { get; } = [];
        public void OnNodeExecuted(string nodeId, NodeOutputs outputs) => Outputs[nodeId] = outputs;
    }

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }

    [Test]
    public void The_models_own_schedules_are_ComfyUIs()
    {
        // ComfyUI's BasicScheduler on Klein: a flow with the shift e^2.02 (mu 2.02) at every size.
        var model = new ModelHandle(new Flux2DiffusionModel(null!, "flux-2-klein-4b.safetensors", "", ""));
        foreach (var c in Cases.GetProperty("basic").EnumerateObject())
        {
            var parts = c.Name.Split('_');
            var expected = c.Value.EnumerateArray().Select(v => (float)v.GetDouble()).ToArray();
            var ours = new BasicScheduler().Execute(model, parts[0], int.Parse(parts[1]), 1.0).Values;
            Assert.That(ours, Is.EqualTo(expected).Within(2e-6), c.Name);
        }
    }

    [Test]
    public void Load_Diffusion_Model_tells_FLUX2_by_its_shared_modulation()
    {
        Assert.That(Flux2DiffusionModel.MarkerTensor, Is.EqualTo("double_stream_modulation_img.lin.weight"));
        Assert.That(new Flux2DiffusionModel(null!, "x", "", "").LatentFormat.Channels, Is.EqualTo(128));
    }

    /// <summary>The distilled template graph at 256 x 256, 2 steps, seed 42, as ComfyUI ran it; the final latent and the picture against ComfyUI's.</summary>
    [Test, Explicit("Needs SPAWNGEN_MODELS_ROOT with diffusion_models/flux-2-klein-4b-fp8.safetensors, text_encoders/qwen_3_4b.safetensors, vae/flux2-vae.safetensors and a GPU.")]
    public async Task The_distilled_template_graph_matches_ComfyUI()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        ModelFolders.Scan(root!);
        var outDir = Path.Combine(TestContext.CurrentContext.WorkDirectory, "flux2_klein_out", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(outDir);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly);
        var executor = new PromptExecutor(registry, new Services(runtime, new OutputOptions { Directory = outDir, TempDirectory = outDir }));
        var g = new PromptGraph();
        g.Add("unet", "UNETLoader", ("unet_name", "flux-2-klein-4b-fp8.safetensors"), ("weight_dtype", "default"));
        g.Add("clip", "CLIPLoader", ("clip_name", "qwen_3_4b.safetensors"), ("type", "flux2"), ("device", "default"));
        g.Add("vae", "VAELoader", ("vae_name", "flux2-vae.safetensors"));
        g.Add("pos", "CLIPTextEncode", ("clip", PromptInput.Link("clip")), ("text", "a red apple on a wooden table"));
        g.Add("neg", "ConditioningZeroOut", ("conditioning", PromptInput.Link("pos")));
        g.Add("guider", "CFGGuider", ("model", PromptInput.Link("unet")), ("positive", PromptInput.Link("pos")), ("negative", PromptInput.Link("neg")), ("cfg", 1.0));
        g.Add("sampler", "KSamplerSelect", ("sampler_name", "euler"));
        g.Add("sigmas", "Flux2Scheduler", ("steps", 2L), ("width", 256L), ("height", 256L));
        g.Add("latent", "EmptyFlux2LatentImage", ("width", 256L), ("height", 256L), ("batch_size", 1L));
        g.Add("noise", "RandomNoise", ("noise_seed", 42L));
        g.Add("run", "SamplerCustomAdvanced", ("noise", PromptInput.Link("noise")), ("guider", PromptInput.Link("guider")), ("sampler", PromptInput.Link("sampler")),
            ("sigmas", PromptInput.Link("sigmas")), ("latent_image", PromptInput.Link("latent")));
        g.Add("dec", "VAEDecode", ("samples", PromptInput.Link("run")), ("vae", PromptInput.Link("vae")));
        g.Add("save", "SaveImage", ("images", PromptInput.Link("dec")), ("filename_prefix", "klein"));
        var clock = System.Diagnostics.Stopwatch.StartNew();
        var capture = new Capture();
        await executor.ExecuteAsync(g, null, capture);
        var seconds = clock.Elapsed.TotalSeconds;
        var latent = (LatentBatch)capture.Outputs["run"].Values[0]!;
        Assert.That(latent.Shape, Is.EqualTo(new[] { 1, 128, 16, 16 }));
        double latentErr = RelRms(latent.Data, Raw("klein_out_samples_f32_1x128x16x16.raw"));
        var ours = PngReader.Decode(File.ReadAllBytes(Directory.GetFiles(outDir, "klein_*.png").Single()));
        var theirs = PngReader.Decode(File.ReadAllBytes(Path.Combine(Dir, "comfyui_picture.png")));
        double mse = 0;
        for (int i = 0; i < ours.Data.Length; i++) { double d = ours.Data[i] - theirs.Data[i]; mse += d * d; }
        double psnr = 10 * Math.Log10(ours.Data.Length / mse);
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: {seconds:F1} s; final latent {latentErr:E2} relative RMS vs ComfyUI's (bf16 there, fp8 here); picture {psnr:F1} dB; {outDir}");
        // Measured on the RX 590 (OpenCL, ML 5.8.17-local.2, fp8 file): final latent 6.8e-2 relative RMS, picture 36.9 dB from ComfyUI's bf16 run.
        Assert.That(latentErr, Is.LessThan(0.1), "the fp8 file's rounding, not a different picture");
        Assert.That(psnr, Is.GreaterThan(33));
    }
}
