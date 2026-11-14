using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// ComfyUI's split FLUX.1 graph from its standalone files against ComfyUI (bef11fc, CPU float32; Fixtures/flux1_split): Load Diffusion Model with
/// FLUX.1 Krea dev (fp8_scaled), DualCLIPLoader type flux (clip_l + t5xxl_fp8_e4m3fn_scaled), Load VAE (ae); the Krea template's graph (no
/// FluxGuidance: ComfyUI passes 3.5) at 256 x 256, 2 steps, seed 42.
/// </summary>
/// <remarks>
/// Two known gaps, both in SpawnDev.ILGPU.ML 5.8.18-local.2's CHANGELOG: the scaled 8-bit transformer is ~1.3e-3 from ComfyUI (an ILGPU float to
/// bfloat16 rounding bug in kernels, fixed in SpawnDev.ILGPU 5.4.33); the scaled T5 runs exact since ML -local.3 (6.5e-6). The bounds below hold today's figures; they tighten when those land. ComfyUI's own Linear on this file is plain fp32
/// (measured one layer at 5e-8: _DevComms 2026-11-14-1000).
/// </remarks>
public class Flux1SplitTests
{
    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "flux1_split");

    static float[] Raw(string name)
    {
        var b = File.ReadAllBytes(Path.Combine(Dir, name));
        var d = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, d, 0, b.Length);
        return d;
    }

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

    static string? Root()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        ModelFolders.Scan(root!);
        return root;
    }

    const string Needs = "Needs SPAWNGEN_MODELS_ROOT with diffusion_models/flux1-krea-dev_fp8_scaled.safetensors, text_encoders/clip_l.safetensors and "
        + "t5xxl_fp8_e4m3fn_scaled.safetensors, vae/ae.safetensors, and a GPU.";

    [Test, Explicit(Needs)]
    public async Task The_dual_CLIP_conditioning_is_ComfyUIs()
    {
        var root = Root();
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var executor = new PromptExecutor(Registry(), new Services(runtime));
        var g = new PromptGraph();
        g.Add("clip", "DualCLIPLoader", ("clip_name1", "clip_l.safetensors"), ("clip_name2", "t5xxl_fp8_e4m3fn_scaled.safetensors"), ("type", "flux"), ("device", "default"));
        g.Add("pos", "CLIPTextEncode", ("clip", PromptInput.Link("clip")), ("text", "a red apple on a wooden table"));
        g.Add("empty", "CLIPTextEncode", ("clip", PromptInput.Link("clip")), ("text", ""));
        g.Add("z", "ConditioningCombine", ("conditioning_1", PromptInput.Link("pos")), ("conditioning_2", PromptInput.Link("empty")));
        g.Add("show", "PreviewAny", ("source", PromptInput.Link("z")));
        var capture = new Capture();
        await executor.ExecuteAsync(g, null, capture);
        var pos = ((Conditioning)capture.Outputs["pos"].Values[0]!).Entries[0];
        var empty = ((Conditioning)capture.Outputs["empty"].Values[0]!).Entries[0];
        var theirs = Raw("f1s_cond_0_cond_f32_1x256x4096.raw");
        double t5Prompt = RelRms(pos.Embeddings.AsSpan(0, 16 * 4096), theirs.AsSpan(0, 16 * 4096)), t5All = RelRms(pos.Embeddings, theirs);
        double t5Empty = RelRms(empty.Embeddings.AsSpan(0, 32 * 4096), Raw("f1s_cond_empty_rows0to31_f32_1x32x4096.raw"));
        double pooled = RelRms(pos.Pooled!, Raw("f1s_cond_0_pooled_output_f32_1x768.raw")), pooledEmpty = RelRms(empty.Pooled!, Raw("f1s_cond_empty_0_pooled_output_f32_1x768.raw"));
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: T5 (fp8_scaled) prompt rows {t5Prompt:E2}, all 256 {t5All:E2}, empty {t5Empty:E2}; clip_l pooled {pooled:E2}, empty {pooledEmpty:E2}");
        Assert.That(t5Prompt, Is.LessThan(1e-4), "6.5e-6 on the RX 590 with ML -local.3 (the exact scaled path)");
        Assert.That(t5All, Is.LessThan(1e-4));
        Assert.That(t5Empty, Is.LessThan(1e-4));
        Assert.That(pooled, Is.LessThan(1e-3));
        Assert.That(pooledEmpty, Is.LessThan(1e-3));
    }

    [Test, Explicit(Needs)]
    public async Task Both_Krea_model_calls_give_ComfyUIs_denoised_estimate()
    {
        var root = Root();
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var services = new Services(runtime);
        var context = (NodeContext)Activator.CreateInstance(typeof(NodeContext), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, null,
            ["1", "UNETLoader", services, CancellationToken.None, null, new Lazy<string>(() => "{}"), new Dictionary<string, string>(), null, null, 0], null)!;
        var handle = await new UNETLoader().ExecuteAsync("flux1-krea-dev_fp8_scaled.safetensors", "default", context, services);
        Assert.That(handle.Model, Is.InstanceOf<Flux1DiffusionModel>());
        using var run = handle.Model.BeginRun(runtime.Accelerator);
        // No FluxGuidance in the Krea template: ComfyUI passed guidance 3.5 (measured), the model's default here too.
        var entry = new CondEntry(Raw("f1s_cond_0_cond_f32_1x256x4096.raw"), [1, 256, 4096]) { Pooled = Raw("f1s_call_0_c_y_f32_1x768.raw") };
        var report = new List<string>();
        for (int call = 0; call < 2; call++)
        {
            var x = Raw($"f1s_call_{call}_input_f32_1x16x32x32.raw");
            float sigma = Raw($"f1s_call_{call}_timestep_f32_1.raw")[0];
            var v = await run.PredictAsync(x, 32, 32, sigma, entry, null, UNetOptions.None, null, CancellationToken.None);
            var denoised = x.Select((xi, i) => xi - sigma * v[i]).ToArray();
            double err = RelRms(denoised, Raw($"f1s_call_{call}_output_f32_1x16x32x32.raw"));
            report.Add($"call {call} (sigma {sigma:F4}): {err:E2}");
            Assert.That(err, Is.LessThan(1e-2), $"call {call}: the scaled transformer's known gap is ~1.3e-3");
        }
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: " + string.Join("; ", report));
    }

    [Test, Explicit(Needs)]
    public async Task The_Krea_template_graph_matches_ComfyUI()
    {
        var root = Root();
        var outDir = Path.Combine(TestContext.CurrentContext.WorkDirectory, "flux1_split_out", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(outDir);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var executor = new PromptExecutor(Registry(), new Services(runtime, new OutputOptions { Directory = outDir, TempDirectory = outDir }));
        var g = new PromptGraph();
        g.Add("unet", "UNETLoader", ("unet_name", "flux1-krea-dev_fp8_scaled.safetensors"), ("weight_dtype", "default"));
        g.Add("clip", "DualCLIPLoader", ("clip_name1", "clip_l.safetensors"), ("clip_name2", "t5xxl_fp8_e4m3fn_scaled.safetensors"), ("type", "flux"), ("device", "default"));
        g.Add("vae", "VAELoader", ("vae_name", "ae.safetensors"));
        g.Add("pos", "CLIPTextEncode", ("clip", PromptInput.Link("clip")), ("text", "a red apple on a wooden table"));
        g.Add("neg", "ConditioningZeroOut", ("conditioning", PromptInput.Link("pos")));
        g.Add("latent", "EmptySD3LatentImage", ("width", 256L), ("height", 256L), ("batch_size", 1L));
        g.Add("ks", "KSampler", ("model", PromptInput.Link("unet")), ("positive", PromptInput.Link("pos")), ("negative", PromptInput.Link("neg")),
            ("latent_image", PromptInput.Link("latent")), ("seed", 42L), ("steps", 2L), ("cfg", 1.0), ("sampler_name", "euler"), ("scheduler", "simple"), ("denoise", 1.0));
        g.Add("dec", "VAEDecode", ("samples", PromptInput.Link("ks")), ("vae", PromptInput.Link("vae")));
        g.Add("save", "SaveImage", ("images", PromptInput.Link("dec")), ("filename_prefix", "krea"));
        var capture = new Capture();
        var clock = System.Diagnostics.Stopwatch.StartNew();
        await executor.ExecuteAsync(g, null, capture);
        var seconds = clock.Elapsed.TotalSeconds;
        var latent = (LatentBatch)capture.Outputs["ks"].Values[0]!;
        double latentErr = RelRms(latent.Data, Raw("f1s_out_samples_f32_1x16x32x32.raw"));
        var ours = PngReader.Decode(File.ReadAllBytes(Directory.GetFiles(outDir, "krea_*.png").Single()));
        var theirs = PngReader.Decode(File.ReadAllBytes(Path.Combine(Dir, "comfyui_picture_256.png")));
        double mse = 0;
        for (int i = 0; i < ours.Data.Length; i++) { double d = ours.Data[i] - theirs.Data[i]; mse += d * d; }
        double psnr = 10 * Math.Log10(ours.Data.Length / mse);
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: {seconds:F1} s; final latent {latentErr:E2} relative RMS vs ComfyUI's; picture {psnr:F1} dB; {outDir}");
        Assert.That(latentErr, Is.LessThan(5e-2));
        Assert.That(psnr, Is.GreaterThan(30));
    }
}
