using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Qwen-Image Edit 2511 against ComfyUI (bef11fc, CPU float32; Fixtures/qwenedit2511, the dumps sent to SpawnDev.ILGPU.ML on 2026-11-16):
/// image_qwen_image_edit_2511's default path (ModelSamplingAuraFlow 3.1, CFGNorm 1, FluxKontextMultiReferenceLatentMethod "index_timestep_zero" on
/// both prompts, euler / simple, cfg 4) with picture A (and A + B), seed 42. The text encode is 2509's (the same node and files: ComfyUI's
/// conditioning is bit-identical), so the text rows and reference latents come from Fixtures/qwenedit.
/// </summary>
public class QwenImageEdit2511Tests
{
    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "qwenedit2511");
    static string Dir2509 => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "qwenedit");

    static float[] Raw(string dir, string name)
    {
        var b = File.ReadAllBytes(Path.Combine(dir, name));
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

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }

    static NodeContext Context(Services services, string classType) =>
        (NodeContext)Activator.CreateInstance(typeof(NodeContext), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, null,
            ["1", classType, services, CancellationToken.None, null, new Lazy<string>(() => "{}"), new Dictionary<string, string>(), null, null, 0], null)!;

    static string? Root()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        ModelFolders.Scan(root!);
        return root;
    }

    [Test]
    public void The_reference_method_sets_the_timestep_zero_modulation()
    {
        var model = new QwenImageDiffusionModel(null!, "qwen_image_edit_2511_fp8mixed.safetensors", "", "");
        var entry = new CondEntry([0f], [1, 1, 3584]);
        Assert.That(model.ReferencesAtTimestepZero(entry), Is.False, "no method: the file's own choice (none opened yet: off)");
        var conditioning = new FluxKontextMultiReferenceLatentMethod().Execute(new Conditioning([entry]), "index_timestep_zero");
        Assert.That(model.ReferencesAtTimestepZero(conditioning.Entries[0]), Is.True);
        var offset = new FluxKontextMultiReferenceLatentMethod().Execute(new Conditioning([entry]), "offset");
        Assert.Throws<NotSupportedException>(() => model.ReferencesAtTimestepZero(offset.Entries[0]));
    }

    const string Needs = "Needs SPAWNGEN_MODELS_ROOT with diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors and a GPU.";

    /// <remarks>
    /// The one-picture sigma-1 calls sit near 1e-3 (2509's are 3e-4): float32 conditioning in ComfyUI's own run, not a difference in the math.
    /// SpawnDev.ILGPU.ML's float64 rebuild of block 0's attention (from the file's weights and ComfyUI's inputs, ML master 53321738) matches
    /// ComfyUI's text output to 7.7e-7 but its image output only to 1.1e-3, and the same rebuild in float32 moves the image rows by 9.9e-4: the
    /// image queries' softmax is very sharp there.
    /// </remarks>
    [Test, Explicit(Needs)]
    public async Task The_2511_model_calls_give_ComfyUIs_denoised_estimate()
    {
        var root = Root();
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var services = new Services(runtime);
        var handle = await new UNETLoader().ExecuteAsync("qwen_image_edit_2511_fp8mixed.safetensors", "default", Context(services, "UNETLoader"), services);
        Assert.That(handle.Model, Is.InstanceOf<QwenImageDiffusionModel>());
        using var run = handle.Model.BeginRun(runtime.Accelerator);
        var refA = new LatentBatch(1, 16, 1, 111, 148, Raw(Dir2509, "qe_te1_cond_0_reference_latents_0_f32_1x16x1x111x148.raw"));
        var refB = new LatentBatch(1, 16, 1, 99, 165, Raw(Dir2509, "qe_te2_cond_0_reference_latents_1_f32_1x16x1x99x165.raw"));
        var report = new List<string>();
        foreach (var (name, call, text, rows, refs) in new[]
        {
            ("q11_1", 0, "qe_te1_cond_0_cond_f32_1x212x3584.raw", 212, new[] { refA }),
            ("q11_1", 1, "qe_te1_neg_cond_0_cond_f32_1x204x3584.raw", 204, new[] { refA }),
            ("q11_1", 2, "qe_te1_cond_0_cond_f32_1x212x3584.raw", 212, new[] { refA }),
            ("q11_1", 3, "qe_te1_neg_cond_0_cond_f32_1x204x3584.raw", 204, new[] { refA }),
            ("q11_2", 0, "qe_te2_cond_0_cond_f32_1x417x3584.raw", 417, new[] { refA, refB }),
            ("q11_2", 1, "qe_te2_neg_cond_0_cond_f32_1x409x3584.raw", 409, new[] { refA, refB }),
        })
        {
            var entry = new CondEntry(Raw(Dir2509, text), [1, rows, 3584]) { ReferenceLatents = refs, ReferenceLatentsMethod = "index_timestep_zero" };
            var x = Raw(Dir, $"{name}_call_{call}_input_f32_1x16x1x24x32.raw");
            float sigma = call < 2 ? 1f : Raw(Dir, $"q11_1_call_{call}_timestep_f32_1.raw")[0];
            var clock = System.Diagnostics.Stopwatch.StartNew();
            var v = await run.PredictAsync(x, 24, 32, sigma, entry, null, UNetOptions.None, null, CancellationToken.None);
            double err = RelRms([.. x.Select((xi, i) => xi - sigma * v[i])], Raw(Dir, $"{name}_call_{call}_output_f32_1x16x1x24x32.raw"));
            report.Add($"{name} call {call} (sigma {sigma:G7}, {refs.Length} reference(s)): {err:E2} in {clock.Elapsed.TotalSeconds:F1} s");
            Assert.That(err, Is.LessThan(2e-3), string.Join("; ", report));   // 1.8e-4 to 9.7e-4 on the RX 590
        }
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name} ({runtime.Accelerator.MemorySize >> 20} MB): " + string.Join("; ", report));
    }

    /// <summary>The template's default path chained on ComfyUI's float picture A (Load Image reads PNGs at 8 bits), 2 steps.</summary>
    [Test, Explicit(Needs + " Also text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors and vae/qwen_image_vae.safetensors.")]
    public async Task The_2511_template_graph_matches_ComfyUI()
    {
        var root = Root();
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var services = new Services(runtime);
        NodeContext Ctx(string type) => Context(services, type);
        var a = new ImageBatch(1, 192, 256, 3);
        Raw(Dir2509, "f2_img_1x192x256x3.raw").CopyTo(a.Data, 0);
        var clock = System.Diagnostics.Stopwatch.StartNew();
        var unet = await new UNETLoader().ExecuteAsync("qwen_image_edit_2511_fp8mixed.safetensors", "default", Ctx("UNETLoader"), services);
        var clip = await new CLIPLoader().ExecuteAsync("qwen_2.5_vl_7b_fp8_scaled.safetensors", "qwen_image", Ctx("CLIPLoader"), services);
        var vae = await new VAELoader().ExecuteAsync("qwen_image_vae.safetensors", Ctx("VAELoader"), services);
        var method = new FluxKontextMultiReferenceLatentMethod();
        var pos = method.Execute(await new TextEncodeQwenImageEditPlus().ExecuteAsync(clip, Ctx("TextEncodeQwenImageEditPlus"), "Turn the picture into a watercolor painting", vae, a), "index_timestep_zero");
        var neg = method.Execute(await new TextEncodeQwenImageEditPlus().ExecuteAsync(clip, Ctx("TextEncodeQwenImageEditPlus"), "", vae, a), "index_timestep_zero");
        var model = new CFGNorm().Execute(new ModelSamplingAuraFlow().Execute(unet, 3.1), 1.0);
        var latentIn = await new VAEEncode().ExecuteAsync(a, vae, Ctx("VAEEncode"), services);
        var latent = await new KSampler().ExecuteAsync(model, pos, neg, latentIn, Ctx("KSampler"), services, 42, 2, 4.0, "euler", "simple", 1.0);
        Assert.That(latent.Shape, Is.EqualTo(new[] { 1, 16, 1, 24, 32 }));
        double latentErr = RelRms(latent.Data, Raw(Dir, "q11_1_out_samples_f32_1x16x1x24x32.raw"));
        var picture = await new VAEDecode().ExecuteAsync(latent, vae, Ctx("VAEDecode"), services);
        await picture.ToHostAsync(); // VAE Decode may leave the pictures on the GPU
        var seconds = clock.Elapsed.TotalSeconds;
        var theirs = PngReader.Decode(File.ReadAllBytes(Path.Combine(Dir, "comfyui_picture_2511_256x192.png")));
        double mse = 0;
        for (int i = 0; i < theirs.Data.Length; i++)
        {
            double d = Math.Clamp(Math.Floor(picture.Data[i] * 255.0 + 0.5), 0, 255) / 255.0 - theirs.Data[i];
            mse += d * d;
        }
        double psnr = 10 * Math.Log10(theirs.Data.Length / mse);
        var outFile = Path.Combine(TestContext.CurrentContext.WorkDirectory, $"qwenedit2511_{Guid.NewGuid():N}_f32_192x256x3.raw");
        File.WriteAllBytes(outFile, System.Runtime.InteropServices.MemoryMarshal.AsBytes(picture.Data.AsSpan()).ToArray());
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: {seconds:F1} s; final latent {latentErr:E2} relative RMS vs ComfyUI's; picture {psnr:F1} dB; {outFile}");
        Assert.That(latentErr, Is.LessThan(1e-2), "3.3e-3 on the RX 590");
        Assert.That(psnr, Is.GreaterThan(46), "52.4 dB on the RX 590");
    }
}
