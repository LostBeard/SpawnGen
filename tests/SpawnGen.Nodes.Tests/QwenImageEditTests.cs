using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Qwen-Image Edit 2509's TextEncodeQwenImageEditPlus against ComfyUI (bef11fc, CPU float32; Fixtures/qwenedit, the dumps sent to SpawnDev.ILGPU.ML
/// on 2026-11-15): picture A (256 x 192) alone and with picture B (200 x 120), prompt "Turn the picture into a watercolor painting".
/// </summary>
public class QwenImageEditTests
{
    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "qwenedit");

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

    static ImageBatch Picture(string name, int height, int width)
    {
        var p = new ImageBatch(1, height, width, 3);
        Raw(name).CopyTo(p.Data, 0);
        return p;
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
    public void The_vision_and_reference_sizes_are_ComfyUIs()
    {
        // Measured from ComfyUI's vision grids ([1, 24, 32], [1, 22, 36]) and reference latents ([.., 111, 148], [.., 99, 165]).
        Assert.That(TextEncodeQwenImageEditPlus.VisionSize(192, 256), Is.EqualTo(((333, 443), (336, 448))));
        Assert.That(TextEncodeQwenImageEditPlus.VisionSize(120, 200), Is.EqualTo(((297, 496), (308, 504))));
        Assert.That(TextEncodeQwenImageEditPlus.ReferenceSize(192, 256), Is.EqualTo((888, 1184)));
        Assert.That(TextEncodeQwenImageEditPlus.ReferenceSize(120, 200), Is.EqualTo((792, 1320)));
    }

    [Test]
    public void The_vision_picture_is_ComfyUIs()
    {
        // ComfyUI's picture as its vision tower received it (the patch rows put back in place, un-normalised): area, then bilinear, no 8-bit step.
        foreach (var (pic, h, w, want, wh, ww) in new[] { ("f2_img_1x192x256x3.raw", 192, 256, "qe_vision_picture_A_f32_336x448x3.raw", 336, 448),
            ("f2_img_1x120x200x3.raw", 120, 200, "qe_vision_picture_B_f32_308x504x3.raw", 308, 504) })
        {
            var ours = TextEncodeQwenImageEditPlus.VisionPicture(Picture(pic, h, w));
            Assert.That((ours.Height, ours.Width), Is.EqualTo((wh, ww)));
            var theirs = Raw(want);
            double worst = 0;
            for (int i = 0; i < theirs.Length; i++) worst = Math.Max(worst, Math.Abs(ours.Data[i] - theirs[i]));
            Assert.That(worst, Is.LessThan(1e-5), $"{w} x {h}: largest difference {worst:E2} (1.3e-6 measured in float64)");
        }
    }

    static int[] Ids(string name) =>
        System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, name))).RootElement.GetProperty("calls")[0].GetProperty("ids").EnumerateArray().Select(v => v.GetInt32()).ToArray();

    const string NeedsAll = "Needs SPAWNGEN_MODELS_ROOT with text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors, vae/qwen_image_vae.safetensors and a GPU.";

    /// <summary>The node: A alone (the prompt and the empty one) and A with B; ids, conditioning and reference latents against ComfyUI's.</summary>
    [Test, Explicit(NeedsAll)]
    public async Task The_edit_conditioning_is_ComfyUIs()
    {
        var root = Root();
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var services = new Services(runtime);
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly);
        Assert.That(registry.GetSchema("TextEncodeQwenImageEditPlus")!.DisplayName, Is.EqualTo("Text Encode Qwen Image Edit Plus"));
        // The ids, the image-pad ids taken out (ComfyUI's embedding table never sees them: vision start and end sit together).
        var clip = await new CLIPLoader().ExecuteAsync("qwen_2.5_vl_7b_fp8_scaled.safetensors", "qwen_image", Context(services, "CLIPLoader"), services);
        var encoder = (QwenImageTextEncoderModel)clip.Encoder;
        const string Prompt = "Turn the picture into a watercolor painting";
        foreach (var (text, grids, file) in new[] { (Prompt, new[] { (24, 32) }, "qe_te1_ids.json"), ("", new[] { (24, 32) }, "qe_te1_neg_ids.json"),
            (Prompt, new[] { (24, 32), (22, 36) }, "qe_te2_ids.json") })
        {
            var (ids, starts) = await encoder.TokenizeEditAsync(text, grids, CancellationToken.None);
            Assert.That(ids.Where(i => i != QwenImageTextEncoderModel.ImagePad), Is.EqualTo(Ids(file)), file);
            Assert.That(ids.Take(QwenImageTextEncoderModel.EditDropped).Last(), Is.EqualTo(198), "the dropped rows end with <|im_start|>user\n");
            Assert.That(starts[0], Is.EqualTo(70));
        }
        var a = Picture("f2_img_1x192x256x3.raw", 192, 256);
        var b = Picture("f2_img_1x120x200x3.raw", 120, 200);
        var node = new TextEncodeQwenImageEditPlus();
        var clock = System.Diagnostics.Stopwatch.StartNew();
        var vae = await new VAELoader().ExecuteAsync("qwen_image_vae.safetensors", Context(services, "VAELoader"), services);
        var one = (await node.ExecuteAsync(clip, Context(services, "TextEncodeQwenImageEditPlus"), Prompt, vae, a)).Entries.Single();
        var oneNeg = (await node.ExecuteAsync(clip, Context(services, "TextEncodeQwenImageEditPlus"), "", vae, a)).Entries.Single();
        var two = (await node.ExecuteAsync(clip, Context(services, "TextEncodeQwenImageEditPlus"), Prompt, vae, a, b)).Entries.Single();
        Assert.That(one.Shape, Is.EqualTo(new[] { 1, 212, 3584 }));
        Assert.That(oneNeg.Shape, Is.EqualTo(new[] { 1, 204, 3584 }));
        Assert.That(two.Shape, Is.EqualTo(new[] { 1, 417, 3584 }));
        double e1 = RelRms(one.Embeddings, Raw("qe_te1_cond_0_cond_f32_1x212x3584.raw")), e1n = RelRms(oneNeg.Embeddings, Raw("qe_te1_neg_cond_0_cond_f32_1x204x3584.raw")),
            e2 = RelRms(two.Embeddings, Raw("qe_te2_cond_0_cond_f32_1x417x3584.raw"));
        Assert.That(one.ReferenceLatents!.Select(r => r.Shape), Is.EqualTo(new[] { new[] { 1, 16, 1, 111, 148 } }));
        Assert.That(two.ReferenceLatents!.Select(r => r.Shape), Is.EqualTo(new[] { new[] { 1, 16, 1, 111, 148 }, new[] { 1, 16, 1, 99, 165 } }));
        double r1 = RelRms(one.ReferenceLatents![0].Data, Raw("qe_te1_cond_0_reference_latents_0_f32_1x16x1x111x148.raw")),
            r2 = RelRms(two.ReferenceLatents![1].Data, Raw("qe_te2_cond_0_reference_latents_1_f32_1x16x1x99x165.raw"));
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: one picture {e1:E2} (negative {e1n:E2}), two pictures {e2:E2}; reference latents A {r1:E2}, B {r2:E2}; {clock.Elapsed.TotalSeconds:F1} s");
        Assert.That(e1, Is.LessThan(3e-4), "4.4e-5 on the RX 590");
        Assert.That(e1n, Is.LessThan(3e-4), "4.5e-5 on the RX 590");
        Assert.That(e2, Is.LessThan(3e-4), "7.3e-5 on the RX 590");
        Assert.That(r1, Is.LessThan(1e-5), "1.2e-6 on the RX 590");
        Assert.That(r2, Is.LessThan(1e-5), "1.4e-6 on the RX 590");
    }


    const string NeedsModel = "Needs SPAWNGEN_MODELS_ROOT with diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors and a GPU.";

    /// <summary>
    /// The 2509 transformer's calls with reference latents as ComfyUI made them (graph1: picture A; graph2: A and B), from the conditioning's text
    /// rows and raw reference latents (the run normalises them as ComfyUI's model receives them).
    /// </summary>
    [Test, Explicit(NeedsModel)]
    public async Task The_edit_model_calls_give_ComfyUIs_denoised_estimate()
    {
        var root = Root();
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var services = new Services(runtime);
        var handle = await new UNETLoader().ExecuteAsync("qwen_image_edit_2509_fp8_e4m3fn.safetensors", "default", Context(services, "UNETLoader"), services);
        Assert.That(handle.Model, Is.InstanceOf<QwenImageDiffusionModel>());
        using var run = handle.Model.BeginRun(runtime.Accelerator);
        var refA = new LatentBatch(1, 16, 1, 111, 148, Raw("qe_te1_cond_0_reference_latents_0_f32_1x16x1x111x148.raw"));
        var refB = new LatentBatch(1, 16, 1, 99, 165, Raw("qe_te2_cond_0_reference_latents_1_f32_1x16x1x99x165.raw"));
        var report = new List<string>();
        foreach (var (graph, call, text, rows, refs, sigma) in new[]
        {
            ("graph1", 2, "qe_te1_cond_0_cond_f32_1x212x3584.raw", 212, new[] { refA }, 0.75f),
            ("graph1", 3, "qe_te1_neg_cond_0_cond_f32_1x204x3584.raw", 204, new[] { refA }, 0.75f),
            ("graph1", 0, "qe_te1_cond_0_cond_f32_1x212x3584.raw", 212, new[] { refA }, 1f),
            ("graph1", 1, "qe_te1_neg_cond_0_cond_f32_1x204x3584.raw", 204, new[] { refA }, 1f),
            ("graph2", 0, "qe_te2_cond_0_cond_f32_1x417x3584.raw", 417, new[] { refA, refB }, 1f),
            ("graph2", 1, "qe_te2_neg_cond_0_cond_f32_1x409x3584.raw", 409, new[] { refA, refB }, 1f),
        })
        {
            var entry = new CondEntry(Raw(text), [1, rows, 3584]) { ReferenceLatents = refs };
            var x = Raw($"qe_{graph}_call_{call}_input_f32_1x16x1x24x32.raw");
            var clock = System.Diagnostics.Stopwatch.StartNew();
            var v = await run.PredictAsync(x, 24, 32, sigma, entry, null, UNetOptions.None, null, CancellationToken.None);
            var denoised = x.Select((xi, i) => xi - sigma * v[i]).ToArray();
            double err = RelRms(denoised, Raw($"qe_{graph}_call_{call}_output_f32_1x16x1x24x32.raw"));
            report.Add($"{graph} call {call} (sigma {sigma}, {refs.Length} reference(s), {rows} text rows): {err:E2} in {clock.Elapsed.TotalSeconds:F1} s");
            Assert.That(err, Is.LessThan(2e-3), string.Join("; ", report));   // 1.7e-4 to 7.9e-4 on the RX 590
        }
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name} ({runtime.Accelerator.MemorySize >> 20} MB): " + string.Join("; ", report));
    }

    /// <summary>
    /// ComfyUI's image_qwen_image_edit_2509 template on its default path (Lightning LoRA off: AuraFlow 3, CFGNorm 1, euler / simple, cfg 4) with
    /// picture A at 256 x 192 fed directly (FluxKontextImageScale left out, as in ComfyUI's run), cut to 2 steps, seed 42: the final latent and
    /// the picture against ComfyUI's graph1.
    /// </summary>
    /// <remarks>
    /// Measured on the RX 590 (2026-11-15): latent_image 8.0e-7; the four calls' inputs 1.8e-7 (the noise) and 3.3e-5 (after the first step) from
    /// ComfyUI's, their denoised estimates 3.6e-4, 3.2e-4, 9.0e-4, 4.7e-3; final latent 9.1e-3, picture 48.8 dB. The negative's call at sigma 0.75
    /// is ~140x sensitive to its input there: with ComfyUI's x2 it is 7.9e-4 from ComfyUI's, with ours (3.3e-5 away) 4.7e-3, with our text and
    /// ComfyUI's x2 8.0e-4. So the final latent's 9e-3 is that sensitivity acting on the first step's rounding, not a fault of a part; each part
    /// is pinned by its own test (The_edit_model_calls_give_ComfyUIs_denoised_estimate, The_edit_conditioning_is_ComfyUIs).
    /// The nodes are chained directly on ComfyUI's float picture A: Load Image cannot carry it exactly (it reads a PNG at 8 bits, as ComfyUI's
    /// does, 2e-3 to 3.5e-3 off), and ComfyUI's run took the float picture.
    /// </remarks>
    [Test, Explicit(NeedsModel + " Also text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors and vae/qwen_image_vae.safetensors.")]
    public async Task The_2509_template_graph_matches_ComfyUI()
    {
        var root = Root();
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var services = new Services(runtime);
        NodeContext Ctx(string type) => Context(services, type);
        const string Prompt = "Turn the picture into a watercolor painting";
        var a = Picture("f2_img_1x192x256x3.raw", 192, 256);
        var clock = System.Diagnostics.Stopwatch.StartNew();
        var unet = await new UNETLoader().ExecuteAsync("qwen_image_edit_2509_fp8_e4m3fn.safetensors", "default", Ctx("UNETLoader"), services);
        var clip = await new CLIPLoader().ExecuteAsync("qwen_2.5_vl_7b_fp8_scaled.safetensors", "qwen_image", Ctx("CLIPLoader"), services);
        var vae = await new VAELoader().ExecuteAsync("qwen_image_vae.safetensors", Ctx("VAELoader"), services);
        var pos = await new TextEncodeQwenImageEditPlus().ExecuteAsync(clip, Ctx("TextEncodeQwenImageEditPlus"), Prompt, vae, a);
        var neg = await new TextEncodeQwenImageEditPlus().ExecuteAsync(clip, Ctx("TextEncodeQwenImageEditPlus"), "", vae, a);
        var recorder = new RecordingModel(unet.Model);
        var model = new CFGNorm().Execute(new ModelSamplingAuraFlow().Execute(new ModelHandle(recorder, unet.Sampling), 3.0), 1.0);
        var latentIn = await new VAEEncode().ExecuteAsync(a, vae, Ctx("VAEEncode"), services);
        double inErr = RelRms(latentIn.Data, Raw("qe_graph1_latent_in_samples_f32_1x16x1x24x32.raw"));
        var latent = await new KSampler().ExecuteAsync(model, pos, neg, latentIn, Ctx("KSampler"), services, 42, 2, 4.0, "euler", "simple", 1.0);
        Assert.That(latent.Shape, Is.EqualTo(new[] { 1, 16, 1, 24, 32 }));
        double latentErr = RelRms(latent.Data, Raw("qe_graph1_out_samples_f32_1x16x1x24x32.raw"));
        var picture = await new VAEDecode().ExecuteAsync(latent, vae, Ctx("VAEDecode"), services);
        await picture.ToHostAsync(); // VAE Decode may leave the pictures on the GPU
        var seconds = clock.Elapsed.TotalSeconds;
        // Each call against ComfyUI's (its four calls: prompt and negative at sigma 1, then at 0.75): the input, and the denoised estimate.
        var calls = new List<string>();
        for (int i = 0; i < recorder.Calls.Count && i < 4; i++)
        {
            var (inp, t, rows, v) = recorder.Calls[i];
            var den = inp.Select((xi, j) => xi - t * v[j]).ToArray();
            calls.Add($"call {i} (t {t}, {rows} rows): input {RelRms(inp, Raw($"qe_graph1_call_{i}_input_f32_1x16x1x24x32.raw")):E2}, denoised {RelRms(den, Raw($"qe_graph1_call_{i}_output_f32_1x16x1x24x32.raw")):E2}");
        }
        TestContext.Out.WriteLine(string.Join("; ", calls));
        var theirs = PngReader.Decode(File.ReadAllBytes(Path.Combine(Dir, "comfyui_picture_graph1_256x192.png")));
        double mse = 0;
        for (int i = 0; i < theirs.Data.Length; i++)
        {
            double ours8 = Math.Clamp(Math.Floor(picture.Data[i] * 255.0 + 0.5), 0, 255) / 255.0, d = ours8 - theirs.Data[i];
            mse += d * d;
        }
        double psnr = 10 * Math.Log10(theirs.Data.Length / mse);
        var outFile = Path.Combine(TestContext.CurrentContext.WorkDirectory, $"qwenedit_graph1_{Guid.NewGuid():N}_f32_192x256x3.raw");
        File.WriteAllBytes(outFile, System.Runtime.InteropServices.MemoryMarshal.AsBytes(picture.Data.AsSpan()).ToArray());
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: {seconds:F1} s; latent_image {inErr:E2}; final latent {latentErr:E2} relative RMS vs ComfyUI's; picture {psnr:F1} dB; {outFile}");
        Assert.That(inErr, Is.LessThan(1e-5), "8.0e-7 on the RX 590");
        Assert.That(latentErr, Is.LessThan(2e-2), "9.1e-3 on the RX 590 (see remarks)");
        Assert.That(psnr, Is.GreaterThan(44), "48.8 dB on the RX 590");
    }


    /// <summary>A model that records every call (input, timestep, text rows, velocity) of the one it wraps.</summary>
    sealed class RecordingModel(IDiffusionModel inner) : IDiffusionModel
    {
        public List<(float[] Input, float Timestep, int TextRows, float[] Output)> Calls { get; } = [];
        public string Name => inner.Name;
        public SpawnDev.ILGPU.ML.StableDiffusion.SdFamily Family => inner.Family;
        public LatentFormat LatentFormat => inner.LatentFormat;
        public ModelSampling Sampling => inner.Sampling;
        public SpawnDev.ILGPU.ML.StableDiffusion.SdUNet? UNet => inner.UNet;
        public Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) => inner.UseWeightsAsync(weights, cancellationToken);
        public UNetRun BeginRun(ILGPU.Runtime.Accelerator accelerator) => new Run(this, inner.BeginRun(accelerator));

        sealed class Run(RecordingModel owner, UNetRun run) : UNetRun
        {
            public override async Task<float[]> PredictAsync(float[] input, int h, int w, float timestep, CondEntry entry, IReadOnlyList<ControlCall>? controls,
                UNetOptions options, float[]? concat, CancellationToken cancellationToken)
            {
                run.Frames = Frames;
                run.FirstFrameClean = FirstFrameClean;
                var output = await run.PredictAsync(input, h, w, timestep, entry, controls, options, concat, cancellationToken);
                owner.Calls.Add(((float[])input.Clone(), timestep, entry.Shape[^2], (float[])output.Clone()));
                return output;
            }

            public override void Dispose() => run.Dispose();
        }
    }


    const string NeedsVae = "Needs SPAWNGEN_MODELS_ROOT with vae/qwen_image_vae.safetensors and a GPU.";

    /// <summary>
    /// Which resampler ComfyUI uses for the reference latent's picture (about 1 MP: each side times sqrt(1024^2 / (w h)), rounded to a multiple of
    /// 8; A -> 1184 x 888, B -> 1320 x 792, measured from the latents' sizes): each candidate encoded with the same VAE (8e-7 from ComfyUI's) and
    /// compared with ComfyUI's reference latent. Prints every candidate; asserts the node's own choice is the best and within 1e-4.
    /// </summary>
    [Test, Explicit(NeedsVae)]
    public async Task The_reference_latent_resize_is_ComfyUIs()
    {
        var root = Root();
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var services = new Services(runtime);
        var vae = await new VAELoader().ExecuteAsync("qwen_image_vae.safetensors", Context(services, "VAELoader"), services);
        var report = new List<string>();
        foreach (var (pic, h, w, refFile, lh, lw) in new[]
        {
            ("f2_img_1x192x256x3.raw", 192, 256, "qe_te1_cond_0_reference_latents_0_f32_1x16x1x111x148.raw", 111, 148),
            ("f2_img_1x120x200x3.raw", 120, 200, "qe_te2_cond_0_reference_latents_1_f32_1x16x1x99x165.raw", 99, 165),
        })
        {
            var picture = Picture(pic, h, w);
            var (rh, rw) = TextEncodeQwenImageEditPlus.ReferenceSize(h, w);
            Assert.That((rh, rw), Is.EqualTo((lh * 8, lw * 8)), $"{w} x {h}: the reference picture's size");
            var want = Raw(refFile);
            var errors = new Dictionary<string, double>();
            foreach (var method in new[] { "area", "bilinear", "bicubic", "lanczos", "nearest-exact" })
            {
                var latent = await new VAEEncode().ExecuteAsync(CommonUpscale.Resize(picture, rw, rh, method), vae, Context(services, "VAEEncode"), services);
                errors[method] = RelRms(latent.Data, want);
            }
            report.Add($"{w} x {h} -> {rw} x {rh}: " + string.Join(", ", errors.OrderBy(e => e.Value).Select(e => $"{e.Key} {e.Value:E2}")));
            var best = errors.OrderBy(e => e.Value).First();
            Assert.That(best.Key, Is.EqualTo(TextEncodeQwenImageEditPlus.ReferenceMethod), string.Join("; ", report));
            Assert.That(best.Value, Is.LessThan(1e-4), string.Join("; ", report));
        }
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: " + string.Join("; ", report));
    }
}
