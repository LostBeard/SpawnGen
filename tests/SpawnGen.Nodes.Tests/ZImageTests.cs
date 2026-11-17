using System.Text.Json;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// ComfyUI's Z-Image Turbo template graph (image_z_image_turbo) against ComfyUI (bef11fc, CPU float32; Fixtures/zimage, the dumps sent to
/// SpawnDev.ILGPU.ML): the model's own schedule, the lumina2 conditioning, two model calls, and the graph at 256 x 256, 2 steps, seed 42.
/// </summary>
public class ZImageTests
{
    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "zimage");

    static float[] Raw(string name)
    {
        var b = File.ReadAllBytes(Path.Combine(Dir, name));
        var d = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, d, 0, b.Length);
        return d;
    }

    static float[] Sigmas(string name) =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, name))).RootElement.GetProperty("sigmas").EnumerateArray().Select(v => (float)v.GetDouble()).ToArray();

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

    static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly);

    [Test]
    public void The_models_own_schedule_and_AuraFlow_3_are_ComfyUIs()
    {
        var model = new ModelHandle(new ZImageDiffusionModel(null!, "z_image_turbo_bf16.safetensors", "", ""));
        Assert.That(new BasicScheduler().Execute(model, "simple", 8, 1.0).Values, Is.EqualTo(Sigmas("z_sigmas_simple8_default.json")).Within(2e-6));
        var aura = new ModelSamplingAuraFlow().Execute(model, 3.0);
        Assert.That(new BasicScheduler().Execute(aura, "simple", 8, 1.0).Values, Is.EqualTo(Sigmas("z_sigmas_simple8_auraflow3.json")).Within(2e-6));
        Assert.That(model.LatentFormat.Channels, Is.EqualTo(16));
    }

    [Test]
    public void The_latent_goes_into_the_model_as_ae_space()
    {
        // (LATENT - 0.1159) x 0.3611, per FLUX.1's ae (measured from ComfyUI's model calls).
        var latent = new float[16];
        latent[3] = 1.1159f;
        var model = LatentFormat.Flux1.ProcessIn(latent);
        Assert.That(model[3], Is.EqualTo(0.3611f).Within(1e-6));
        Assert.That(model[0], Is.EqualTo(-0.1159f * 0.3611f).Within(1e-6));
    }

    static string? Root()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        ModelFolders.Scan(root!);
        return root;
    }

    const string Needs = "Needs SPAWNGEN_MODELS_ROOT with diffusion_models/z_image_turbo_bf16.safetensors, text_encoders/qwen_3_4b.safetensors, vae/ae.safetensors and a GPU.";

    [Test, Explicit(Needs)]
    public async Task The_lumina2_conditioning_is_ComfyUIs()
    {
        var root = Root();
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var executor = new PromptExecutor(Registry(), new Services(runtime));
        var g = new PromptGraph();
        g.Add("clip", "CLIPLoader", ("clip_name", "qwen_3_4b.safetensors"), ("type", "lumina2"), ("device", "default"));
        g.Add("pos", "CLIPTextEncode", ("clip", PromptInput.Link("clip")), ("text", "a red apple on a wooden table"));
        g.Add("empty", "CLIPTextEncode", ("clip", PromptInput.Link("clip")), ("text", ""));
        g.Add("z", "ConditioningCombine", ("conditioning_1", PromptInput.Link("pos")), ("conditioning_2", PromptInput.Link("empty")));
        g.Add("show", "PreviewAny", ("source", PromptInput.Link("z")));
        var capture = new Capture();
        await executor.ExecuteAsync(g, null, capture);
        var pos = ((Conditioning)capture.Outputs["pos"].Values[0]!).Entries[0];
        var empty = ((Conditioning)capture.Outputs["empty"].Values[0]!).Entries[0];
        Assert.That(pos.Shape, Is.EqualTo(new[] { 1, 15, 2560 }));
        Assert.That(empty.Shape, Is.EqualTo(new[] { 1, 8, 2560 }));
        double ePos = RelRms(pos.Embeddings, Raw("z_cond_0_cond_f32_1x15x2560.raw")), eEmpty = RelRms(empty.Embeddings, Raw("z_cond_empty_0_cond_f32_1x8x2560.raw"));
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: prompt {ePos:E2}, empty {eEmpty:E2} relative RMS vs ComfyUI");
        Assert.That(ePos, Is.LessThan(1e-2));
        Assert.That(eEmpty, Is.LessThan(1e-2));
    }

    [Test, Explicit(Needs)]
    public async Task Both_model_calls_give_ComfyUIs_denoised_estimate()
    {
        var root = Root();
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var services = new Services(runtime);
        var context = (NodeContext)Activator.CreateInstance(typeof(NodeContext), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, null,
            ["1", "UNETLoader", services, CancellationToken.None, null, new Lazy<string>(() => "{}"), new Dictionary<string, string>(), null, null, 0], null)!;
        var handle = await new UNETLoader().ExecuteAsync("z_image_turbo_bf16.safetensors", "default", context, services);
        Assert.That(handle.Model, Is.InstanceOf<ZImageDiffusionModel>());
        using var run = handle.Model.BeginRun(runtime.Accelerator);
        var entry = new CondEntry(Raw("z_cond_0_cond_f32_1x15x2560.raw"), [1, 15, 2560]);
        var report = new List<string>();
        for (int call = 0; call < 2; call++)
        {
            var x = Raw($"z_call_{call}_input_f32_1x16x32x32.raw");
            float sigma = Raw($"z_call_{call}_timestep_f32_1.raw")[0];
            var clock = System.Diagnostics.Stopwatch.StartNew();
            var v = await run.PredictAsync(x, 32, 32, sigma, entry, null, UNetOptions.None, null, CancellationToken.None);
            var denoised = x.Select((xi, i) => xi - sigma * v[i]).ToArray();
            double err = RelRms(denoised, Raw($"z_call_{call}_output_f32_1x16x32x32.raw"));
            report.Add($"call {call} (sigma {sigma:F4}): {err:E2} in {clock.Elapsed.TotalSeconds:F1} s");
            Assert.That(err, Is.LessThan(2e-2), $"call {call}");
        }
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name} ({runtime.Accelerator.MemorySize >> 20} MB): " + string.Join("; ", report));
    }

    /// <summary>The template graph (its 1024 x 1024 and 8 steps cut to 256 x 256 and 2) as ComfyUI ran it: the final latent and the picture.</summary>
    [Test, Explicit(Needs)]
    public async Task The_template_graph_matches_ComfyUI()
    {
        var root = Root();
        var outDir = Path.Combine(TestContext.CurrentContext.WorkDirectory, "zimage_out", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(outDir);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var executor = new PromptExecutor(Registry(), new Services(runtime, new OutputOptions { Directory = outDir, TempDirectory = outDir }));
        var g = new PromptGraph();
        g.Add("unet", "UNETLoader", ("unet_name", "z_image_turbo_bf16.safetensors"), ("weight_dtype", "default"));
        g.Add("clip", "CLIPLoader", ("clip_name", "qwen_3_4b.safetensors"), ("type", "lumina2"), ("device", "default"));
        g.Add("vae", "VAELoader", ("vae_name", "ae.safetensors"));
        g.Add("pos", "CLIPTextEncode", ("clip", PromptInput.Link("clip")), ("text", "a red apple on a wooden table"));
        g.Add("neg", "ConditioningZeroOut", ("conditioning", PromptInput.Link("pos")));
        g.Add("msa", "ModelSamplingAuraFlow", ("model", PromptInput.Link("unet")), ("shift", 3.0));
        g.Add("latent", "EmptySD3LatentImage", ("width", 256L), ("height", 256L), ("batch_size", 1L));
        g.Add("ks", "KSampler", ("model", PromptInput.Link("msa")), ("positive", PromptInput.Link("pos")), ("negative", PromptInput.Link("neg")),
            ("latent_image", PromptInput.Link("latent")), ("seed", 42L), ("steps", 2L), ("cfg", 1.0), ("sampler_name", "res_multistep"),
            ("scheduler", "simple"), ("denoise", 1.0));
        g.Add("dec", "VAEDecode", ("samples", PromptInput.Link("ks")), ("vae", PromptInput.Link("vae")));
        g.Add("save", "SaveImage", ("images", PromptInput.Link("dec")), ("filename_prefix", "zimage"));
        var capture = new Capture();
        var clock = System.Diagnostics.Stopwatch.StartNew();
        await executor.ExecuteAsync(g, null, capture);
        var seconds = clock.Elapsed.TotalSeconds;
        var latent = (LatentBatch)capture.Outputs["ks"].Values[0]!;
        Assert.That(latent.Shape, Is.EqualTo(new[] { 1, 16, 32, 32 }));
        double latentErr = RelRms(latent.Data, Raw("z_out_samples_f32_1x16x32x32.raw"));
        var ours = PngReader.Decode(File.ReadAllBytes(Directory.GetFiles(outDir, "zimage_*.png").Single()));
        var theirs = PngReader.Decode(File.ReadAllBytes(Path.Combine(Dir, "comfyui_picture_256.png")));
        double mse = 0;
        for (int i = 0; i < ours.Data.Length; i++) { double d = ours.Data[i] - theirs.Data[i]; mse += d * d; }
        double psnr = 10 * Math.Log10(ours.Data.Length / mse);
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: {seconds:F1} s; final latent {latentErr:E2} relative RMS vs ComfyUI's; picture {psnr:F1} dB; {outDir}");
        Assert.That(latentErr, Is.LessThan(5e-2));
        Assert.That(psnr, Is.GreaterThan(30));
    }
}
