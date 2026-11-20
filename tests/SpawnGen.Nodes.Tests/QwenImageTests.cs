using System.Text.Json;
using SpawnDev.ILGPU.ML.Wan;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// ComfyUI's Qwen-Image template graph (image_qwen_image, its default path: no Lightning LoRA) against ComfyUI (bef11fc, CPU float32;
/// Fixtures/qwenimage, the dumps sent to SpawnDev.ILGPU.ML on 2026-11-14): the schedules, the latent space, the qwen_image conditioning, two model
/// calls, the VAE, and the graph at 256 x 256, 2 steps, cfg 4, seed 42, prompt "a red apple on a wooden table" with the empty prompt as negative.
/// </summary>
public class QwenImageTests
{
    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "qwenimage");

    static float[] Raw(string name)
    {
        var b = File.ReadAllBytes(Path.Combine(Dir, name));
        var d = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, d, 0, b.Length);
        return d;
    }

    static float[] Sigmas(string name) =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, name))).RootElement.GetProperty("sigmas").EnumerateArray().Select(v => (float)v.GetDouble()).ToArray();

    static int[] Ids(string name) =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, name))).RootElement.GetProperty("calls")[0].GetProperty("ids").EnumerateArray().Select(v => v.GetInt32()).ToArray();

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
    public void The_models_own_schedule_and_AuraFlow_3_1_are_ComfyUIs()
    {
        var model = new ModelHandle(new QwenImageDiffusionModel(null!, "qwen_image_fp8_e4m3fn.safetensors", "", ""));
        Assert.That(new BasicScheduler().Execute(model, "simple", 10, 1.0).Values, Is.EqualTo(Sigmas("qi_default_sigmas_simple10.json")).Within(2e-6));
        Assert.That(new BasicScheduler().Execute(model, "simple", 2, 1.0).Values, Is.EqualTo(Sigmas("qi_default_sigmas_simple2.json")).Within(2e-6));
        var aura = new ModelSamplingAuraFlow().Execute(model, 3.1);
        Assert.That(new BasicScheduler().Execute(aura, "simple", 20, 1.0).Values, Is.EqualTo(Sigmas("qi_sigmas_simple20_auraflow3.1.json")).Within(2e-6));
        Assert.That(new BasicScheduler().Execute(aura, "simple", 2, 1.0).Values, Is.EqualTo(Sigmas("qi_sigmas_simple2_auraflow3.1.json")).Within(2e-6));
    }

    [Test]
    public void The_latent_goes_into_the_model_normalised_per_channel()
    {
        // (LATENT - mean) / std per channel, one frame (Wan 2.1's VAE statistics); the GPU test checks the numbers against the library's VAE.
        var f = LatentFormat.Wan21;
        Assert.That(f.Channels, Is.EqualTo(16));
        Assert.That(f.VideoShaped, Is.True);
        var latent = new float[16 * 4];
        for (int c = 0; c < 16; c++) for (int p = 0; p < 4; p++) latent[c * 4 + p] = f.LatentsMean![c] + f.LatentsStd![c] * (p + 1);
        var model = f.ProcessIn(latent);
        for (int c = 0; c < 16; c++) for (int p = 0; p < 4; p++) Assert.That(model[c * 4 + p], Is.EqualTo(p + 1f).Within(1e-5), $"channel {c}");
    }

    [Test]
    public void An_empty_picture_latent_becomes_one_video_frame()
    {
        // Measured: Empty SD3 Latent Image [1, 16, 32, 32] sampled with Qwen-Image gives a LATENT [1, 16, 1, 32, 32].
        var empty = new LatentBatch(1, 16, 32, 32);
        var shaped = SamplingCore.ToModelChannels(empty, LatentFormat.Wan21);
        Assert.That(shaped.Shape, Is.EqualTo(new[] { 1, 16, 1, 32, 32 }));
        // Other families keep a picture latent as it is; a latent with content is never reshaped.
        Assert.That(SamplingCore.ToModelChannels(empty, LatentFormat.Flux1).Shape, Is.EqualTo(new[] { 1, 16, 32, 32 }));
        var content = new LatentBatch(1, 16, 32, 32);
        content.Data[5] = 1f;
        Assert.That(SamplingCore.ToModelChannels(content, LatentFormat.Wan21).Shape, Is.EqualTo(new[] { 1, 16, 32, 32 }));
    }

    [Test]
    public void The_loaders_recognise_ComfyUIs_files_by_their_tensors()
    {
        string[] bare = ["img_in.weight", "txt_norm.weight", "transformer_blocks.0.attn.to_q.weight"];
        Assert.That(QwenImageDiffusionModel.Is(bare), Is.True);
        Assert.That(QwenImageDiffusionModel.Is([.. bare.Select(n => "model.diffusion_model." + n)]), Is.True, "the 2509 edit file's prefix");
        Assert.That(QwenImageDiffusionModel.Is(["img_in.weight", "txt_in.weight", "double_blocks.0.img_attn.qkv.weight"]), Is.False, "FLUX.1");
        string[] wanVae = ["conv1.weight", "conv2.weight", "encoder.conv1.weight", "decoder.conv1.weight"];
        Assert.That(WanComfyVaeModel.Is(wanVae), Is.True);
        Assert.That(WanComfyVaeModel.Is([.. wanVae, "latents_mean"]), Is.False, "the library's own layout");
        Assert.That(WanComfyVaeModel.Is(["encoder.conv_in.weight", "decoder.conv_in.weight"]), Is.False, "an SD VAE");
    }

    static string? Root()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        ModelFolders.Scan(root!);
        return root;
    }

    static NodeContext Context(Services services, string classType) =>
        (NodeContext)Activator.CreateInstance(typeof(NodeContext), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, null,
            ["1", classType, services, CancellationToken.None, null, new Lazy<string>(() => "{}"), new Dictionary<string, string>(), null, null, 0], null)!;

    const string Needs = "Needs SPAWNGEN_MODELS_ROOT with diffusion_models/qwen_image_fp8_e4m3fn.safetensors, text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors, "
        + "vae/qwen_image_vae.safetensors, and a GPU.";

    [Test, Explicit(Needs)]
    public async Task The_qwen_image_conditioning_is_ComfyUIs()
    {
        var root = Root();
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var executor = new PromptExecutor(Registry(), new Services(runtime));
        var g = new PromptGraph();
        g.Add("clip", "CLIPLoader", ("clip_name", "qwen_2.5_vl_7b_fp8_scaled.safetensors"), ("type", "qwen_image"), ("device", "default"));
        g.Add("pos", "CLIPTextEncode", ("clip", PromptInput.Link("clip")), ("text", "a red apple on a wooden table"));
        g.Add("empty", "CLIPTextEncode", ("clip", PromptInput.Link("clip")), ("text", ""));
        g.Add("z", "ConditioningCombine", ("conditioning_1", PromptInput.Link("pos")), ("conditioning_2", PromptInput.Link("empty")));
        g.Add("show", "PreviewAny", ("source", PromptInput.Link("z")));
        var capture = new Capture();
        var clock = System.Diagnostics.Stopwatch.StartNew();
        await executor.ExecuteAsync(g, null, capture);
        var pos = ((Conditioning)capture.Outputs["pos"].Values[0]!).Entries[0];
        var empty = ((Conditioning)capture.Outputs["empty"].Values[0]!).Entries[0];
        // The ids ComfyUI fed its embedding table, ours the same.
        var clipServices = new Services(runtime);
        var clip = await new CLIPLoader().ExecuteAsync("qwen_2.5_vl_7b_fp8_scaled.safetensors", "qwen_image", Context(clipServices, "CLIPLoader"), clipServices);
        var encoder = (QwenImageTextEncoderModel)clip.Encoder;
        Assert.That(await encoder.TokenizeAsync("a red apple on a wooden table", CancellationToken.None), Is.EqualTo(Ids("qi_ids_prompt.json")));
        Assert.That(await encoder.TokenizeAsync("", CancellationToken.None), Is.EqualTo(Ids("qi_ids_empty.json")));
        Assert.That(pos.Shape, Is.EqualTo(new[] { 1, 12, 3584 }));
        Assert.That(empty.Shape, Is.EqualTo(new[] { 1, 5, 3584 }));
        double ePos = RelRms(pos.Embeddings, Raw("qi_cond_0_cond_f32_1x12x3584.raw")), eEmpty = RelRms(empty.Embeddings, Raw("qi_cond_empty_0_cond_f32_1x5x3584.raw"));
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: prompt {ePos:E2}, empty {eEmpty:E2} relative RMS vs ComfyUI; {clock.Elapsed.TotalSeconds:F1} s");
        Assert.That(ePos, Is.LessThan(1e-4), "4.4e-6 on the RX 590");
        Assert.That(eEmpty, Is.LessThan(1e-4), "7.9e-6 on the RX 590");
    }

    [Test, Explicit(Needs)]
    public async Task Both_model_calls_give_ComfyUIs_denoised_estimate()
    {
        var root = Root();
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var services = new Services(runtime);
        var handle = await new UNETLoader().ExecuteAsync("qwen_image_fp8_e4m3fn.safetensors", "default", Context(services, "UNETLoader"), services);
        Assert.That(handle.Model, Is.InstanceOf<QwenImageDiffusionModel>());
        using var run = handle.Model.BeginRun(runtime.Accelerator);
        var report = new List<string>();
        foreach (var (call, rows) in new[] { (0, 12), (1, 5) })
        {
            // The wrapper's view: the model-space latent [1, 16, 1, 32, 32], the call's own text rows (cond and uncond run apart).
            var entry = new CondEntry(Raw($"qi_call_{call}_c_c_crossattn_f32_1x{rows}x3584.raw"), [1, rows, 3584]);
            var x = Raw($"qi_call_{call}_input_f32_1x16x1x32x32.raw");
            float sigma = Raw($"qi_call_{call}_timestep_f32_1.raw")[0];
            var clock = System.Diagnostics.Stopwatch.StartNew();
            var v = await run.PredictAsync(x, 32, 32, sigma, entry, null, UNetOptions.None, null, CancellationToken.None);
            var denoised = x.Select((xi, i) => xi - sigma * v[i]).ToArray();
            double err = RelRms(denoised, Raw($"qi_call_{call}_output_f32_1x16x1x32x32.raw"));
            report.Add($"call {call} ({rows} rows, sigma {sigma:F4}): {err:E2} in {clock.Elapsed.TotalSeconds:F1} s");
            Assert.That(err, Is.LessThan(1e-3), $"call {call}: 1.7e-4 / 2.0e-4 on the RX 590");
        }
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name} ({runtime.Accelerator.MemorySize >> 20} MB): " + string.Join("; ", report));
    }

    [Test, Explicit(Needs)]
    public async Task The_VAE_matches_ComfyUIs_and_its_statistics_are_LatentFormat_Wan21s()
    {
        var root = Root();
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var services = new Services(runtime);
        var vae = await new VAELoader().ExecuteAsync("qwen_image_vae.safetensors", Context(services, "VAELoader"), services);
        Assert.That(vae.Vae, Is.InstanceOf<WanComfyVaeModel>());
        var picture = new ImageBatch(1, 192, 256, 3);
        Raw("f2_img_1x192x256x3.raw").CopyTo(picture.Data, 0);
        var latent = await new VAEEncode().ExecuteAsync(picture, vae, Context(services, "VAEEncode"), services);
        Assert.That(latent.Shape, Is.EqualTo(new[] { 1, 16, 1, 24, 32 }));
        double enc = RelRms(latent.Data, Raw("qi_enc_256x192_samples_f32_1x16x1x24x32.raw"));
        var comfyLatent = new LatentBatch(1, 16, 1, 24, 32, Raw("qi_enc_256x192_samples_f32_1x16x1x24x32.raw"));
        var decoded = await new VAEDecode().ExecuteAsync(comfyLatent, vae, Context(services, "VAEDecode"), services);
        await decoded.ToHostAsync(); // VAE Decode may leave the pictures on the GPU
        Assert.That(new[] { decoded.Batch, decoded.Height, decoded.Width }, Is.EqualTo(new[] { 1, 192, 256 }));
        double dec = RelRms(decoded.Data, Raw("qi_dec_of_enc_256x192_images_f32_1x192x256x3.raw"));
        // The constants the model's latent space uses are the VAE file's own.
        var (vaeRepo, vaeFile) = ModelFolders.Locate(ModelFolders.Vae, "qwen_image_vae.safetensors");
        using var weights = await Wan22Pipeline.LoadWeightsAsync(runtime.Accelerator, runtime.ModelSource, vaeRepo, vaeFile);
        using var wan = new WanVae(runtime.Accelerator, weights);
        var (mean, std) = await wan.GetLatentNormalizationAsync();
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: encode {enc:E2}, decode {dec:E2} relative RMS vs ComfyUI");
        Assert.That(mean, Is.EqualTo(LatentFormat.Wan21.LatentsMean).Within(1e-4));
        Assert.That(std, Is.EqualTo(LatentFormat.Wan21.LatentsStd).Within(1e-4));
        Assert.That(enc, Is.LessThan(1e-5), "8.0e-7 on the RX 590");
        Assert.That(dec, Is.LessThan(1e-5), "3.4e-7 on the RX 590");
    }

    /// <summary>The template graph (its 1328 x 1328 and 20 steps cut to 256 x 256 and 2) as ComfyUI ran it: the final latent and the picture.</summary>
    [Test, Explicit(Needs)]
    public async Task The_template_graph_matches_ComfyUI()
    {
        var root = Root();
        var outDir = Path.Combine(TestContext.CurrentContext.WorkDirectory, "qwenimage_out", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(outDir);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var executor = new PromptExecutor(Registry(), new Services(runtime, new OutputOptions { Directory = outDir, TempDirectory = outDir }));
        var g = new PromptGraph();
        g.Add("unet", "UNETLoader", ("unet_name", "qwen_image_fp8_e4m3fn.safetensors"), ("weight_dtype", "default"));
        g.Add("clip", "CLIPLoader", ("clip_name", "qwen_2.5_vl_7b_fp8_scaled.safetensors"), ("type", "qwen_image"), ("device", "default"));
        g.Add("vae", "VAELoader", ("vae_name", "qwen_image_vae.safetensors"));
        g.Add("pos", "CLIPTextEncode", ("clip", PromptInput.Link("clip")), ("text", "a red apple on a wooden table"));
        g.Add("neg", "CLIPTextEncode", ("clip", PromptInput.Link("clip")), ("text", ""));
        g.Add("msa", "ModelSamplingAuraFlow", ("model", PromptInput.Link("unet")), ("shift", 3.1));
        g.Add("latent", "EmptySD3LatentImage", ("width", 256L), ("height", 256L), ("batch_size", 1L));
        g.Add("ks", "KSampler", ("model", PromptInput.Link("msa")), ("positive", PromptInput.Link("pos")), ("negative", PromptInput.Link("neg")),
            ("latent_image", PromptInput.Link("latent")), ("seed", 42L), ("steps", 2L), ("cfg", 4.0), ("sampler_name", "euler"),
            ("scheduler", "simple"), ("denoise", 1.0));
        g.Add("dec", "VAEDecode", ("samples", PromptInput.Link("ks")), ("vae", PromptInput.Link("vae")));
        g.Add("save", "SaveImage", ("images", PromptInput.Link("dec")), ("filename_prefix", "qwenimage"));
        var capture = new Capture();
        var clock = System.Diagnostics.Stopwatch.StartNew();
        await executor.ExecuteAsync(g, null, capture);
        var seconds = clock.Elapsed.TotalSeconds;
        var latent = (LatentBatch)capture.Outputs["ks"].Values[0]!;
        Assert.That(latent.Shape, Is.EqualTo(new[] { 1, 16, 1, 32, 32 }), "ComfyUI's LATENT is 5-D (measured)");
        double latentErr = RelRms(latent.Data, Raw("qi_out_samples_f32_1x16x1x32x32.raw"));
        var ours = PngReader.Decode(File.ReadAllBytes(Directory.GetFiles(outDir, "qwenimage_*.png").Single()));
        var theirs = PngReader.Decode(File.ReadAllBytes(Path.Combine(Dir, "comfyui_picture_256.png")));
        double mse = 0;
        for (int i = 0; i < ours.Data.Length; i++) { double d = ours.Data[i] - theirs.Data[i]; mse += d * d; }
        double psnr = 10 * Math.Log10(ours.Data.Length / mse);
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: {seconds:F1} s; final latent {latentErr:E2} relative RMS vs ComfyUI's; picture {psnr:F1} dB; {outDir}");
        Assert.That(latentErr, Is.LessThan(1e-3), "1.3e-4 on the RX 590");
        Assert.That(psnr, Is.GreaterThan(60), "72.5 dB on the RX 590");
    }
}
