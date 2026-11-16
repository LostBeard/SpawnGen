using System.Text.Json;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// ComfyUI's FLUX.1 dev graph from its all-in-one checkpoint (flux1-dev-fp8.safetensors) against ComfyUI (bef11fc, CPU float32; Fixtures/flux1, the
/// dumps sent to SpawnDev.ILGPU.ML): the model's own schedule, the conditioning (T5 rows and the pooled CLIP-L vector), two model calls with
/// FluxGuidance 3.5, and the graph at 256 x 256, 2 steps, seed 42 (Load Checkpoint, CLIP Text Encode, FluxGuidance, BasicGuider, euler).
/// </summary>
public class Flux1Tests
{
    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "flux1");

    static float[] Raw(string name)
    {
        var b = File.ReadAllBytes(Path.Combine(Dir, name));
        var d = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, d, 0, b.Length);
        return d;
    }

    static float[] Sigmas(string name) =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, name))).RootElement.GetProperty("sigmas").EnumerateArray().Select(v => (float)v.GetDouble()).ToArray();

    static double RelRms(ReadOnlySpan<float> ours, ReadOnlySpan<float> theirs)
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
    public void The_models_own_schedule_is_ComfyUIs()
    {
        // ComfyUI's FLUX.1 schedule without a ModelSamplingFlux node: the flow with mu 1.15 (= ModelSamplingFlux at 1024 x 1024).
        var model = new ModelHandle(new Flux1DiffusionModel(null!, "flux1-dev-fp8.safetensors", "", ""));
        Assert.That(new BasicScheduler().Execute(model, "simple", 20, 1.0).Values, Is.EqualTo(Sigmas("flux1_sigmas_simple20_default.json")).Within(2e-6));
        Assert.That(new BasicScheduler().Execute(model, "simple", 2, 1.0).Values, Is.EqualTo(Sigmas("flux1_sigmas_run.json")).Within(2e-6));
        Assert.That(model.LatentFormat.Channels, Is.EqualTo(16));
    }

    [Test]
    public void Load_Checkpoint_tells_an_all_in_one_FLUX1_file_by_its_tensors()
    {
        Assert.That(Flux1Checkpoint.Is(["model.diffusion_model.double_blocks.0.img_attn.qkv.weight", "text_encoders.t5xxl.transformer.shared.weight"]), Is.True);
        Assert.That(Flux1Checkpoint.Is(["model.diffusion_model.double_blocks.0.img_attn.qkv.weight"]), Is.False, "a transformer alone is Load Diffusion Model's");
        Assert.That(Flux1Checkpoint.Is(["model.diffusion_model.input_blocks.0.0.weight"]), Is.False, "an SD checkpoint");
    }

    static string? Root()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        ModelFolders.Scan(root!);
        CheckpointCatalog.ScanCheckpointFiles(root!);
        return root;
    }

    const string Needs = "Needs SPAWNGEN_MODELS_ROOT with checkpoints/flux1-dev-fp8.safetensors (ComfyUI's all-in-one file) and a GPU.";

    [Test, Explicit(Needs)]
    public async Task The_conditioning_is_ComfyUIs()
    {
        var root = Root();
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var executor = new PromptExecutor(Registry(), new Services(runtime));
        var g = new PromptGraph();
        g.Add("ckpt", "CheckpointLoaderSimple", ("ckpt_name", "flux1-dev-fp8.safetensors"));
        g.Add("pos", "CLIPTextEncode", ("clip", PromptInput.Link("ckpt", 1)), ("text", "a red apple on a wooden table"));
        g.Add("empty", "CLIPTextEncode", ("clip", PromptInput.Link("ckpt", 1)), ("text", ""));
        g.Add("z", "ConditioningCombine", ("conditioning_1", PromptInput.Link("pos")), ("conditioning_2", PromptInput.Link("empty")));
        g.Add("show", "PreviewAny", ("source", PromptInput.Link("z")));
        var capture = new Capture();
        var clock = System.Diagnostics.Stopwatch.StartNew();
        await executor.ExecuteAsync(g, null, capture);
        var pos = ((Conditioning)capture.Outputs["pos"].Values[0]!).Entries[0];
        var empty = ((Conditioning)capture.Outputs["empty"].Values[0]!).Entries[0];
        Assert.That(pos.Shape, Is.EqualTo(new[] { 1, 256, 4096 }));
        var theirs = Raw("flux1_cond_0_cond_f32_1x256x4096.raw");
        double t5All = RelRms(pos.Embeddings, theirs), t5Prompt = RelRms(pos.Embeddings.AsSpan(0, 16 * 4096), theirs.AsSpan(0, 16 * 4096));
        double t5Empty = RelRms(empty.Embeddings.AsSpan(0, 32 * 4096), Raw("flux1_cond_empty_rows0to31_f32_1x32x4096.raw"));
        double pooled = RelRms(pos.Pooled!, Raw("flux1_cond_0_pooled_output_f32_1x768.raw")), pooledEmpty = RelRms(empty.Pooled!, Raw("flux1_cond_empty_0_pooled_output_f32_1x768.raw"));
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: {clock.Elapsed.TotalSeconds:F1} s; T5 prompt rows {t5Prompt:E2}, all 256 rows {t5All:E2}, empty's first 32 rows {t5Empty:E2}; "
            + $"pooled CLIP-L {pooled:E2}, empty {pooledEmpty:E2} (relative RMS vs ComfyUI)");
        Assert.That(t5Prompt, Is.LessThan(2e-2));
        Assert.That(t5All, Is.LessThan(3e-2));
        Assert.That(t5Empty, Is.LessThan(3e-2));
        Assert.That(pooled, Is.LessThan(1e-2));
        Assert.That(pooledEmpty, Is.LessThan(1e-2));
    }

    [Test, Explicit(Needs)]
    public async Task Both_model_calls_give_ComfyUIs_denoised_estimate()
    {
        var root = Root();
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var services = new Services(runtime);
        var context = (NodeContext)Activator.CreateInstance(typeof(NodeContext), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, null,
            ["1", "CheckpointLoaderSimple", services, CancellationToken.None, null, new Lazy<string>(() => "{}"), new Dictionary<string, string>(), null, null, 0], null)!;
        var (model, _, _) = await new CheckpointLoaderSimple().ExecuteAsync("flux1-dev-fp8.safetensors", context, services);
        Assert.That(model.Model, Is.InstanceOf<Flux1DiffusionModel>());
        using var run = model.Model.BeginRun(runtime.Accelerator);
        var entry = new CondEntry(Raw("flux1_cond_0_cond_f32_1x256x4096.raw"), [1, 256, 4096]) { Pooled = Raw("flux1_cond_0_pooled_output_f32_1x768.raw"), Guidance = 3.5 };
        var report = new List<string>();
        for (int call = 0; call < 2; call++)
        {
            var x = Raw($"flux1_call_{call}_input_f32_1x16x32x32.raw");
            float sigma = Raw($"flux1_call_{call}_timestep_f32_1.raw")[0];
            var clock = System.Diagnostics.Stopwatch.StartNew();
            var v = await run.PredictAsync(x, 32, 32, sigma, entry, null, UNetOptions.None, null, CancellationToken.None);
            var denoised = x.Select((xi, i) => xi - sigma * v[i]).ToArray();
            double err = RelRms(denoised, Raw($"flux1_call_{call}_output_f32_1x16x32x32.raw"));
            report.Add($"call {call} (sigma {sigma:F4}): {err:E2} in {clock.Elapsed.TotalSeconds:F1} s");
            Assert.That(err, Is.LessThan(2e-2), $"call {call}");
        }
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name} ({runtime.Accelerator.MemorySize >> 20} MB): " + string.Join("; ", report));
    }

    /// <summary>The checkpoint graph ComfyUI ran for the dumps (256 x 256, 2 steps, seed 42): the final latent and the picture.</summary>
    [Test, Explicit(Needs)]
    public async Task The_checkpoint_graph_matches_ComfyUI()
    {
        var root = Root();
        var outDir = Path.Combine(TestContext.CurrentContext.WorkDirectory, "flux1_out", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(outDir);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var executor = new PromptExecutor(Registry(), new Services(runtime, new OutputOptions { Directory = outDir, TempDirectory = outDir }));
        var g = new PromptGraph();
        g.Add("ckpt", "CheckpointLoaderSimple", ("ckpt_name", "flux1-dev-fp8.safetensors"));
        g.Add("pos", "CLIPTextEncode", ("clip", PromptInput.Link("ckpt", 1)), ("text", "a red apple on a wooden table"));
        g.Add("guid", "FluxGuidance", ("conditioning", PromptInput.Link("pos")), ("guidance", 3.5));
        g.Add("guider", "BasicGuider", ("model", PromptInput.Link("ckpt", 0)), ("conditioning", PromptInput.Link("guid")));
        g.Add("sampler", "KSamplerSelect", ("sampler_name", "euler"));
        g.Add("sig", "BasicScheduler", ("model", PromptInput.Link("ckpt", 0)), ("scheduler", "simple"), ("steps", 2L), ("denoise", 1.0));
        g.Add("latent", "EmptySD3LatentImage", ("width", 256L), ("height", 256L), ("batch_size", 1L));
        g.Add("noise", "RandomNoise", ("noise_seed", 42L));
        g.Add("run", "SamplerCustomAdvanced", ("noise", PromptInput.Link("noise")), ("guider", PromptInput.Link("guider")), ("sampler", PromptInput.Link("sampler")),
            ("sigmas", PromptInput.Link("sig")), ("latent_image", PromptInput.Link("latent")));
        g.Add("dec", "VAEDecode", ("samples", PromptInput.Link("run")), ("vae", PromptInput.Link("ckpt", 2)));
        g.Add("save", "SaveImage", ("images", PromptInput.Link("dec")), ("filename_prefix", "flux1"));
        var capture = new Capture();
        var clock = System.Diagnostics.Stopwatch.StartNew();
        await executor.ExecuteAsync(g, null, capture);
        var seconds = clock.Elapsed.TotalSeconds;
        var latent = (LatentBatch)capture.Outputs["run"].Values[0]!;
        Assert.That(latent.Shape, Is.EqualTo(new[] { 1, 16, 32, 32 }));
        double latentErr = RelRms(latent.Data, Raw("flux1_out_samples_f32_1x16x32x32.raw"));
        var ours = PngReader.Decode(File.ReadAllBytes(Directory.GetFiles(outDir, "flux1_*.png").Single()));
        var theirs = PngReader.Decode(File.ReadAllBytes(Path.Combine(Dir, "comfyui_picture_256.png")));
        double mse = 0;
        for (int i = 0; i < ours.Data.Length; i++) { double d = ours.Data[i] - theirs.Data[i]; mse += d * d; }
        double psnr = 10 * Math.Log10(ours.Data.Length / mse);
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: {seconds:F1} s; final latent {latentErr:E2} relative RMS vs ComfyUI's; picture {psnr:F1} dB; {outDir}");
        Assert.That(latentErr, Is.LessThan(5e-2));
        Assert.That(psnr, Is.GreaterThan(30));
    }
}
