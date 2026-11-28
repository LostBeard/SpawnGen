using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// ComfyUI's Anima preview template graph (image_anima_preview) against ComfyUI (CPU float32, 2026-11-27, run 1: Fixtures/anima, made by
/// queue_run1.py there): the model's schedule, the Qwen3 + T5 conditioning, two model calls (batch rows [negative, positive]) and the graph at
/// 512 x 512, 4 steps, cfg 4, er_sde / simple, seed 42.
/// </summary>
public class AnimaTests
{
    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "anima");

    static float[] Raw(string name)
    {
        var b = File.ReadAllBytes(Path.Combine(Dir, name));
        var d = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, d, 0, b.Length);
        return d;
    }

    /// <summary>The T5 ids ComfyUI's encoder put in the conditioning (the probe wrote the int32 ids as float32).</summary>
    static int[] T5Ids(string name) => Raw(name).Select(v => (int)v).ToArray();

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

    const string Positive = "masterpiece, best quality, score_7, safe, anime, a girl with a red umbrella standing in the rain";
    const string Negative = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts";

    [Test]
    public void The_models_schedule_and_first_timestep_are_ComfyUIs()
    {
        var model = new ModelHandle(new AnimaDiffusionModel(null!, "anima-preview3-base.safetensors", "", ""));
        // simple, 4 steps: the shift-3 flow (ComfyUI's sample_sigmas in the call dumps)
        Assert.That(new BasicScheduler().Execute(model, "simple", 4, 1.0).Values, Is.EqualTo(Raw("call_0_c_transformer_options_sample_sigmas_f32_5.raw")).Within(2e-6));
        // The DiT's timestep is sigma itself. Run 6 (ManualSigmas 1, 0.99999, 0.99997, 0.99996, 0.9999, 0.999, 0.99, 0.5 with euler) passed
        // exactly those; run 1's first call at 0.9999667 under sigma 1 is er_sde's own first evaluation point, not the model's mapping.
        foreach (var sigma in new[] { 1.0, 0.99999, 0.9999, 0.999, 0.99, 0.9, 0.5 })
            Assert.That(model.Model.Sampling.Timestep(sigma), Is.EqualTo(sigma).Within(1e-7), $"sigma {sigma}");
        Assert.That(model.LatentFormat.Channels, Is.EqualTo(16));
    }

    /// <summary>
    /// Run 7 (every sampler on sigmas 1, 0.5, 0, cfg 1, Anima at shift 3) and run 8 (er_sde at shifts 1 and 6): 15 samplers call a flow model first at
    /// the sigma for percent 1e-4, the other 30 at 1. A stand-in model records the sigma of each sampler's first call.
    /// </summary>
    [TestCase(3.0, 0.99996668f)]
    [TestCase(1.0, 0.9999f)]
    [TestCase(6.0, 0.99998333f)]
    public async Task Fifteen_samplers_start_a_flow_model_at_percent_1e_4(double shift, float expected)
    {
        var sampling = new ModelSampling(ModelSigmas.Flow(ModelSigmas.Sd3Shift(shift), 1), PredictionType.Const);
        foreach (var sampler in Samplers.Names)
        {
            float? first = null;
            Samplers.DenoiseFn model = (x, sigma) => { first ??= sigma; return Task.FromResult(new float[x.Length]); };
            await Samplers.RunAsync(sampler, new float[16], [1f, 0.5f, 0f], model, new Samplers.NoiseSource(1, 1, 4, 4), sampling: sampling);
            float want = Samplers.OffsetsFirstSigma.Contains(sampler) ? expected : 1f;
            Assert.That(first, Is.EqualTo(want).Within(2e-7), $"{sampler} at shift {shift}");
        }
        // exactly ComfyUI's 15 (run 7)
        Assert.That(Samplers.OffsetsFirstSigma, Has.Count.EqualTo(15));
    }

    /// <summary>
    /// One er_sde step on ComfyUI's own first prediction (call_0's outputs, rows [negative, positive], cfg 4) from ComfyUI's noise (seed 42): the
    /// latent the sampler hands the model next must be ComfyUI's call_1 input. Localizes the sampler from the model.
    /// </summary>
    [Test]
    public async Task One_er_sde_step_lands_where_ComfyUIs_did()
    {
        var sampling = new ModelSampling(ModelSigmas.Flow(ModelSigmas.Sd3Shift(3.0), 1), PredictionType.Const);
        const int n = 16 * 64 * 64;
        const float cfg = 4f;
        var out0 = Raw("call_0_output_f32_2x16x1x64x64.raw");
        var guided = new float[n];
        for (int i = 0; i < n; i++) guided[i] = out0[i] + cfg * (out0[n + i] - out0[i]);
        var x0 = Raw("call_0_input_f32_2x16x1x64x64.raw")[..n];
        float[]? second = null;
        float firstSigma = 0, secondSigma = 0;
        int calls = 0;
        Samplers.DenoiseFn model = (x, sigma) =>
        {
            if (++calls == 1) { firstSigma = sigma; return Task.FromResult(guided); }
            if (second is null) { second = (float[])x.Clone(); secondSigma = sigma; }
            return Task.FromResult(new float[n]);
        };
        // the sampler's own noise is the seed + 1 stream (KSampler does that: SamplingCore; run 9)
        await Samplers.RunAsync("er_sde", x0, Raw("call_0_c_transformer_options_sample_sigmas_f32_5.raw"), model, new Samplers.NoiseSource(42 + 1, 16, 64, 64), sampling: sampling);
        var want = Raw("call_1_input_f32_2x16x1x64x64.raw")[..n];
        double err = RelRms(second!, want);
        TestContext.Out.WriteLine($"first call sigma {firstSigma:F7}, second {secondSigma:F7}; second input vs ComfyUI's call_1 input: {err:E2} relative RMS");
        Assert.That(firstSigma, Is.EqualTo(Raw("call_0_timestep_f32_2.raw")[0]).Within(2e-7));
        Assert.That(err, Is.LessThan(1e-5));
    }

    /// <summary>
    /// Run 9 (Fixtures/anima/noise, queue_run9.py): 21 noise-adding samplers, seed 7, sigmas 1, 0.5, 0, cfg 1, Anima at 128 x 128. Each sampler runs on
    /// ComfyUI's first prediction; the latent it hands the model next must be ComfyUI's second call's input. Which seed's stream gives it shows which
    /// noise stream ComfyUI's sampler drew from.
    /// </summary>
    [Test]
    public async Task Each_noisy_samplers_second_call_is_ComfyUIs()
    {
        var sampling = new ModelSampling(ModelSigmas.Flow(ModelSigmas.Sd3Shift(3.0), 1), PredictionType.Const);
        string dir = Path.Combine(Dir, "noise");
        float[] Load(string pattern) { var b = File.ReadAllBytes(Directory.GetFiles(dir, pattern).Single()); var d = new float[b.Length / 4]; Buffer.BlockCopy(b, 0, d, 0, b.Length); return d; }
        var samplers = Directory.GetFiles(dir, "s_*_0_input_*.raw").Select(f => Path.GetFileName(f)[2..^"_0_input_f32_1x16x1x16x16.raw".Length]).Order().ToList();
        var report = new List<string>();
        foreach (var sampler in samplers)
        {
            var x0 = Load($"s_{sampler}_0_input_*.raw");
            var out0 = Load($"s_{sampler}_0_output_*.raw");
            var want = Load($"s_{sampler}_1_input_*.raw");
            var found = new List<string>();
            foreach (int offset in new[] { 0, 1 })
            {
                float[]? second = null;
                int calls = 0;
                Samplers.DenoiseFn model = (x, sigma) =>
                {
                    if (++calls == 1) return Task.FromResult(out0);
                    second ??= (float[])x.Clone();
                    return Task.FromResult(new float[x.Length]);
                };
                try
                {
                    await Samplers.RunAsync(sampler, x0, [1f, 0.5f, 0f], model, new Samplers.NoiseSource(7 + offset, 16, 16, 16), sampling: sampling);
                    double err = second is null ? double.NaN : RelRms(second, want);
                    found.Add($"seed+{offset}: {err:E1}");
                }
                catch (Exception e) { found.Add($"seed+{offset}: {e.GetType().Name} {e.StackTrace?.Split('\n').FirstOrDefault()?.Trim()}"); }
            }
            report.Add($"{sampler}: {string.Join(", ", found)}");
        }
        TestContext.Out.WriteLine(string.Join("\n", report));
        // Settled: these draw from seed + 1 and land on ComfyUI's input (~1.5e-7). Not settled here: dpm_2_ancestral (its second call is a stage
        // before any noise), the cfg_pp variants (they need the unconditional estimate, which this replay does not give), dpmpp_2s_ancestral (NaN from
        // sigma 1 on a flow model), and the Brownian-tree samplers dpmpp_sde / 2m_sde / 3m_sde (SpawnGen's Brownian path is a stand-in for torchsde's
        // tree and matches neither stream): docs/PLANS.md.
        string[] settled = ["ddpm", "er_sde", "euler_ancestral", "exp_heun_2_x0_sde", "lcm", "res_multistep_ancestral", "sa_solver", "sa_solver_pece", "seeds_2", "seeds_3"];
        foreach (var s in settled)
        {
            var line = report.Single(r => r.StartsWith(s + ":", StringComparison.Ordinal));
            var plus1 = double.Parse(line.Split("seed+1: ")[1].Split(',')[0], System.Globalization.CultureInfo.InvariantCulture);
            Assert.That(plus1, Is.LessThan(1e-5), s);
        }
    }

    static string? Root()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        ModelFolders.Scan(root!);
        return root;
    }

    const string Needs = "Needs SPAWNGEN_MODELS_ROOT with diffusion_models/anima-preview3-base.safetensors, text_encoders/qwen_3_06b_base.safetensors, vae/qwen_image_vae.safetensors and a GPU.";

    [Test, Explicit(Needs)]
    public async Task The_conditioning_is_ComfyUIs()
    {
        var root = Root();
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var executor = new PromptExecutor(Registry(), new Services(runtime));
        var g = new PromptGraph();
        g.Add("clip", "CLIPLoader", ("clip_name", "qwen_3_06b_base.safetensors"), ("type", "stable_diffusion"), ("device", "default"));
        g.Add("pos", "CLIPTextEncode", ("clip", PromptInput.Link("clip")), ("text", Positive));
        g.Add("neg", "CLIPTextEncode", ("clip", PromptInput.Link("clip")), ("text", Negative));
        g.Add("z", "ConditioningCombine", ("conditioning_1", PromptInput.Link("pos")), ("conditioning_2", PromptInput.Link("neg")));
        g.Add("show", "PreviewAny", ("source", PromptInput.Link("z")));
        var capture = new Capture();
        await executor.ExecuteAsync(g, null, capture);
        var pos = ((Conditioning)capture.Outputs["pos"].Values[0]!).Entries[0];
        var neg = ((Conditioning)capture.Outputs["neg"].Values[0]!).Entries[0];
        Assert.That(pos.Shape, Is.EqualTo(new[] { 1, 24, 1024 }));
        Assert.That(neg.Shape, Is.EqualTo(new[] { 1, 23, 1024 }));
        Assert.That(pos.T5Ids, Is.EqualTo(T5Ids("pos_0_t5xxl_ids_f32_26.raw")));
        Assert.That(neg.T5Ids, Is.EqualTo(T5Ids("neg_0_t5xxl_ids_f32_27.raw")));
        double ePos = RelRms(pos.Embeddings, Raw("pos_0_cond_f32_1x24x1024.raw")), eNeg = RelRms(neg.Embeddings, Raw("neg_0_cond_f32_1x23x1024.raw"));
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: positive {ePos:E2}, negative {eNeg:E2} relative RMS vs ComfyUI");
        Assert.That(ePos, Is.LessThan(1e-3));
        Assert.That(eNeg, Is.LessThan(1e-3));
    }

    [Test, Explicit(Needs)]
    public async Task Both_model_calls_give_ComfyUIs_denoised_estimate()
    {
        var root = Root();
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var services = new Services(runtime);
        var context = (NodeContext)Activator.CreateInstance(typeof(NodeContext), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, null,
            ["1", "UNETLoader", services, CancellationToken.None, null, new Lazy<string>(() => "{}"), new Dictionary<string, string>(), null, null, 0], null)!;
        var handle = await new UNETLoader().ExecuteAsync("anima-preview3-base.safetensors", "default", context, services);
        Assert.That(handle.Model, Is.InstanceOf<AnimaDiffusionModel>());
        using var run = handle.Model.BeginRun(runtime.Accelerator);
        // ComfyUI's batch rows: 0 = the negative, 1 = the positive (measured: c_crossattn row 1 is the positive's adapter output)
        var entries = new[]
        {
            new CondEntry(Raw("neg_0_cond_f32_1x23x1024.raw"), [1, 23, 1024]) { T5Ids = T5Ids("neg_0_t5xxl_ids_f32_27.raw") },
            new CondEntry(Raw("pos_0_cond_f32_1x24x1024.raw"), [1, 24, 1024]) { T5Ids = T5Ids("pos_0_t5xxl_ids_f32_26.raw") },
        };
        const int n = 16 * 64 * 64;
        var report = new List<string>();
        for (int call = 0; call < 2; call++)
        {
            var x = Raw($"call_{call}_input_f32_2x16x1x64x64.raw");
            var want = Raw($"call_{call}_output_f32_2x16x1x64x64.raw");
            var t = Raw($"call_{call}_timestep_f32_2.raw");
            for (int row = 0; row < 2; row++)
            {
                var xr = x[(row * n)..((row + 1) * n)];
                var clock = System.Diagnostics.Stopwatch.StartNew();
                var v = await run.PredictAsync(xr, 64, 64, t[row], entries[row], null, UNetOptions.None, null, CancellationToken.None);
                var denoised = xr.Select((xi, i) => xi - t[row] * v[i]).ToArray();
                double err = RelRms(denoised, want[(row * n)..((row + 1) * n)]);
                report.Add($"call {call} row {row} (sigma {t[row]:F7}): {err:E2} in {clock.Elapsed.TotalSeconds:F2} s");
                Assert.That(err, Is.LessThan(1e-3), $"call {call} row {row}");
            }
        }
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name} ({runtime.Accelerator.MemorySize >> 20} MB): " + string.Join("; ", report));
    }

    const string NeedsLLLite = "Needs SPAWNGEN_MODELS_ROOT with diffusion_models/anima-base-v1.0.safetensors, model_patches/anima-lllite-inpainting-v2.safetensors and a GPU.";

    /// <summary>
    /// Apply Anima LLLite against ComfyUI (runs 4 / 5: Fixtures/anima/lllite, queue_run4.py / queue_run5.py): anima-base-v1.0 with
    /// anima-lllite-inpainting-v2 on compositor-a.png and a Painter rectangle mask, cfg 1; both model calls through Load Model Patch -> Apply Anima
    /// LLLite -> the Anima run, and the same calls WITHOUT the patch against run 5 (the control, which also shows the patch matters).
    /// </summary>
    [Test, Explicit(NeedsLLLite)]
    public async Task Apply_Anima_LLLite_gives_ComfyUIs_model_calls()
    {
        var root = Root();
        string dir = Path.Combine(Dir, "lllite");
        float[] L(string pattern) { var b = File.ReadAllBytes(Directory.GetFiles(dir, pattern).Single()); var d = new float[b.Length / 4]; Buffer.BlockCopy(b, 0, d, 0, b.Length); return d; }
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var services = new Services(runtime);
        var context = (NodeContext)Activator.CreateInstance(typeof(NodeContext), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, null,
            ["1", "UNETLoader", services, CancellationToken.None, null, new Lazy<string>(() => "{}"), new Dictionary<string, string>(), null, null, 0], null)!;
        var model = await new UNETLoader().ExecuteAsync("anima-base-v1.0.safetensors", "default", context, services);
        var patch = new ModelPatchLoader().Execute("anima-lllite-inpainting-v2.safetensors");
        var picture = ImageDecoder.Decode(File.ReadAllBytes(Path.Combine(dir, "compositor-a.png")));
        ImageDecoder.Decode(File.ReadAllBytes(Path.Combine(dir, "painter-lllite-rect.png")), out var alpha);
        var mask = new MaskBatch(1, 512, 512, alpha);   // Painter's MASK is the painting's alpha (PainterTests)
        var patched = new AnimaLLLiteApply().Execute(model, patch, picture, 1.0, 0.0, 1.0, mask);
        var entry = new CondEntry(L("with_pos_0_cond_*.raw"), [1, 4, 1024]) { T5Ids = L("with_pos_0_t5xxl_ids_*.raw").Select(v => (int)v).ToArray() };
        using var run = model.Model.BeginRun(runtime.Accelerator);
        var report = new List<string>();
        foreach (var (tag, handle, limit) in new[] { ("with", patched, 1e-3), ("without", model, 1e-3) })
            for (int call = 0; call < 2; call++)
            {
                var x = L($"{tag}_call_{call}_input_*.raw");
                float sigma = L($"{tag}_call_{call}_timestep_*.raw")[0];
                var options = (handle.Patches.UNet ?? UNetOptions.None).At(sigma);
                var v = await run.PredictAsync(x, 64, 64, sigma, entry, null, options, null, CancellationToken.None);
                var denoised = x.Select((xi, i) => xi - sigma * v[i]).ToArray();
                double err = RelRms(denoised, L($"{tag}_call_{call}_output_*.raw"));
                report.Add($"{tag} LLLite call {call} (sigma {sigma:F5}): {err:E2}");
                Assert.That(err, Is.LessThan(limit), $"{tag} call {call}");
            }
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: " + string.Join("; ", report));
    }

    /// <summary>The preview template graph (its 1024 x 1024 and 30 steps cut to 512 x 512 and 4) as ComfyUI ran it: the final latent and the picture.</summary>
    [Test, Explicit(Needs)]
    public async Task The_template_graph_matches_ComfyUI()
    {
        var root = Root();
        var outDir = Path.Combine(TestContext.CurrentContext.WorkDirectory, "anima_out", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(outDir);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var executor = new PromptExecutor(Registry(), new Services(runtime, new OutputOptions { Directory = outDir, TempDirectory = outDir }));
        var g = new PromptGraph();
        g.Add("unet", "UNETLoader", ("unet_name", "anima-preview3-base.safetensors"), ("weight_dtype", "default"));
        g.Add("clip", "CLIPLoader", ("clip_name", "qwen_3_06b_base.safetensors"), ("type", "stable_diffusion"), ("device", "default"));
        g.Add("vae", "VAELoader", ("vae_name", "qwen_image_vae.safetensors"));
        g.Add("pos", "CLIPTextEncode", ("clip", PromptInput.Link("clip")), ("text", Positive));
        g.Add("neg", "CLIPTextEncode", ("clip", PromptInput.Link("clip")), ("text", Negative));
        g.Add("latent", "EmptyLatentImage", ("width", 512L), ("height", 512L), ("batch_size", 1L));
        g.Add("ks", "KSampler", ("model", PromptInput.Link("unet")), ("positive", PromptInput.Link("pos")), ("negative", PromptInput.Link("neg")),
            ("latent_image", PromptInput.Link("latent")), ("seed", 42L), ("steps", 4L), ("cfg", 4.0), ("sampler_name", "er_sde"),
            ("scheduler", "simple"), ("denoise", 1.0));
        g.Add("dec", "VAEDecode", ("samples", PromptInput.Link("ks")), ("vae", PromptInput.Link("vae")));
        g.Add("save", "SaveImage", ("images", PromptInput.Link("dec")), ("filename_prefix", "anima"));
        var capture = new Capture();
        var clock = System.Diagnostics.Stopwatch.StartNew();
        await executor.ExecuteAsync(g, null, capture);
        var seconds = clock.Elapsed.TotalSeconds;
        var latent = (LatentBatch)capture.Outputs["ks"].Values[0]!;
        Assert.That(latent.Shape, Is.EqualTo(new[] { 1, 16, 1, 64, 64 }));
        double latentErr = RelRms(latent.Data, Raw("out_samples_f32_1x16x1x64x64.raw"));
        var ours = PngReader.Decode(File.ReadAllBytes(Directory.GetFiles(outDir, "anima_*.png").Single()));
        var theirs = PngReader.Decode(File.ReadAllBytes(Path.Combine(Dir, "comfyui_picture_512.png")));
        double mse = 0;
        for (int i = 0; i < ours.Data.Length; i++) { double d = ours.Data[i] - theirs.Data[i]; mse += d * d; }
        double psnr = 10 * Math.Log10(ours.Data.Length / mse);
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: {seconds:F1} s; final latent {latentErr:E2} relative RMS vs ComfyUI's; picture {psnr:F1} dB; {outDir}");
        Assert.That(latentErr, Is.LessThan(1e-2));
        Assert.That(psnr, Is.GreaterThan(35));
    }
}
