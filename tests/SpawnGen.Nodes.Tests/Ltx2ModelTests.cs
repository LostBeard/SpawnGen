using System.Text.Json;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// LTX-2 19B (SpawnDev.ILGPU.ML's port) through SpawnGen's nodes against ComfyUI (bef11fc, CPU float32; Fixtures/ltx2, the runs sent to
/// SpawnDev.ILGPU.ML on 2026-11-17): the video_ltx2_t2v template's prompts and models (ltx-2-19b-dev-fp8, gemma_3_12B_it_fp4_mixed), one sampling
/// pass at 64 x 64, 9 frames, 25 fps, seed 10, cfg 4, euler, LTXVScheduler 2 steps.
/// </summary>
public class Ltx2ModelTests
{
    const string Checkpoint = "ltx-2-19b-dev-fp8.safetensors", Gemma = "gemma_3_12B_it_fp4_mixed.safetensors";

    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "ltx2");

    static float[] Raw(string name)
    {
        var b = File.ReadAllBytes(Path.Combine(Dir, name));
        var d = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, d, 0, b.Length);
        return d;
    }

    static JsonElement Prompt => JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "ltx2_prompt.json"))).RootElement;

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

    const string Needs = "Needs SPAWNGEN_MODELS_ROOT with checkpoints/ltx-2-19b-dev-fp8.safetensors and text_encoders/gemma_3_12B_it_fp4_mixed.safetensors, and a GPU.";

    static (MlRuntime Runtime, Services Services) Runtime()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        ModelFolders.Scan(root!);
        CheckpointCatalog.ScanCheckpointFiles(root!);   // checkpoints/*.safetensors and loras/, as the server does at start
        CheckpointCatalog.ScanLoraFiles(root!);
        var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        return (runtime, new Services(runtime));
    }

    static Task<ClipHandle> ClipAsync(Services services) =>
        new LTXAVTextEncoderLoader().ExecuteAsync(Gemma, Checkpoint, "default", services);

    /// <summary>The prompt as ComfyUI feeds Gemma: BOS, then the file's own SentencePiece ids (116 real ids measured).</summary>
    [Test, Explicit(Needs)]
    public async Task The_prompt_ids_are_ComfyUIs()
    {
        var (runtime, services) = Runtime();
        using var _r = runtime;
        var model = (Ltx2TextModel)(await ClipAsync(services)).Encoder;
        var ids = await model.TokenizeAsync(Prompt.GetProperty("positive").GetString()!, CancellationToken.None);
        Assert.That(ids, Is.EqualTo(Prompt.GetProperty("positive_real_ids").EnumerateArray().Select(v => v.GetInt32()).ToArray()));
    }

    /// <summary>CLIP Text Encode on Load LTXV Audio Text Encoder: the conditioning [1, 1024, 7680] against ComfyUI's (every 16th row compared).</summary>
    [Test, Explicit(Needs)]
    public async Task The_conditioning_is_ComfyUIs()
    {
        var (runtime, services) = Runtime();
        using var _r = runtime;
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var cond = await new CLIPTextEncode().ExecuteAsync(Prompt.GetProperty("positive").GetString()!, await ClipAsync(services), Context(services, "CLIPTextEncode"));
        var e = cond.Entries.Single();
        Assert.That(e.Shape, Is.EqualTo(new[] { 1, 1024, 7680 }));
        var every16 = Enumerable.Range(0, 64).SelectMany(r => e.Embeddings.AsSpan(r * 16 * 7680, 7680).ToArray()).ToArray();
        double err = RelRms(every16, Raw("ltx2_te_pos_every16_f32_64x7680.raw"));
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: conditioning {err:E2} relative RMS vs ComfyUI, {sw.Elapsed.TotalSeconds:0} s");
        Assert.That(err, Is.LessThan(1e-4));   // measured 2.9e-6 on the RX 590
    }

    /// <summary>The template's first pass as one SpawnGen graph (nodes chained directly), against ComfyUI's final latents, frames and audio.</summary>
    [Test, Explicit(Needs)]
    public async Task The_template_graph_matches_ComfyUI()
    {
        var (runtime, services) = Runtime();
        using var _r = runtime;
        NodeContext Ctx(string type) => Context(services, type);
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var clip = await ClipAsync(services);
        var pos = await new CLIPTextEncode().ExecuteAsync(Prompt.GetProperty("positive").GetString()!, clip, Ctx("CLIPTextEncode"));
        var neg = await new CLIPTextEncode().ExecuteAsync(Prompt.GetProperty("negative").GetString()!, clip, Ctx("CLIPTextEncode"));
        var (positive, negative) = new LTXVConditioning().Execute(pos, neg, 25);
        double encodeSeconds = sw.Elapsed.TotalSeconds;
        var (model, _, vae) = await new CheckpointLoaderSimple().ExecuteAsync(Checkpoint, Ctx("CheckpointLoaderSimple"), services);
        var audioVae = await new LTXVAudioVAELoader().ExecuteAsync(Checkpoint, services);
        var av = new LTXVConcatAVLatent().Execute(new EmptyLTXVLatentVideo().Execute(64, 64, 9, 1), new LTXVEmptyLatentAudio().Execute(9, 25, 1, audioVae));
        var sigmas = new LTXVScheduler().Execute(2, 2.05, 0.95, true, 0.1, av);
        var guider = new CFGGuider().Execute(model, positive, negative, 4.0);
        var (output, _) = await new SamplerCustomAdvanced().ExecuteAsync(new RandomNoise().Execute(10), guider, new KSamplerSelect().Execute("euler"), sigmas, av,
            Ctx("SamplerCustomAdvanced"), services);
        double sampleSeconds = sw.Elapsed.TotalSeconds - encodeSeconds;
        var (video, audio) = new LTXVSeparateAVLatent().Execute(output);
        double videoErr = RelRms(video.Data, Raw("ltx2_dec_video_latent_samples_f32_1x128x2x2x2.raw")),
            audioErr = RelRms(audio.Data, Raw("ltx2_dec_audio_latent_samples_f32_1x8x9x16.raw"));
        var frames = await new VAEDecode().ExecuteAsync(video, vae, Ctx("VAEDecode"), services);
        var clipOut = await new LTXVAudioVAEDecode().ExecuteAsync(audio, audioVae, Ctx("LTXVAudioVAEDecode"));
        Assert.That((frames.Batch, frames.Height, frames.Width), Is.EqualTo((9, 64, 64)));
        Assert.That((clipOut.ChannelCount, clipOut.Length, clipOut.SampleRate), Is.EqualTo((2, 7920, 24000)));
        double framesErr = RelRms(frames.Data, Raw("ltx2_dec_frames_images_f32_9x64x64x3.raw")),
            waveErr = RelRms([.. clipOut.Channels[0], .. clipOut.Channels[1]], Raw("ltx2_dec_audio_waveform_f32_1x2x7920.raw"));
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: final latent video {videoErr:E2}, audio {audioErr:E2}; frames {framesErr:E2}, waveform {waveErr:E2} "
            + $"(relative RMS vs ComfyUI); encode {encodeSeconds:0} s, sample {sampleSeconds:0} s, total {sw.Elapsed.TotalSeconds:0} s");
        // Measured on the RX 590 (OpenCL): 2.5e-5, 8.0e-6, 2.7e-5, 1.2e-4
        Assert.That(videoErr, Is.LessThan(3e-4));
        Assert.That(audioErr, Is.LessThan(3e-4));
        Assert.That(framesErr, Is.LessThan(3e-4));
        Assert.That(waveErr, Is.LessThan(1e-3));
    }

    /// <summary>LTXVLatentUpsampler (x2 spatial upscaler + the checkpoint VAE's statistics) on ComfyUI's input, against its output.</summary>
    [Test, Explicit(Needs + " Also latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors.")]
    public async Task The_latent_upsampler_matches_ComfyUI()
    {
        var (runtime, services) = Runtime();
        using var _r = runtime;
        var (_, _, vae) = await new CheckpointLoaderSimple().ExecuteAsync(Checkpoint, Context(services, "CheckpointLoaderSimple"), services);
        var input = new LatentBatch([1, 128, 2, 2, 2], Raw("ltx2_up_in_samples_f32_1x128x2x2x2.raw")) { Type = "audio" };
        var output = await new LTXVLatentUpsampler().ExecuteAsync(input, new LatentUpscaleModelLoader().Execute("ltx-2-spatial-upscaler-x2-1.0.safetensors"), vae);
        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 128, 2, 4, 4 }));
        Assert.That(output.Type, Is.EqualTo("audio"), "the keys are kept (measured)");
        double err = RelRms(output.Data, Raw("ltx2_up_out_samples_f32_1x128x2x4x4.raw"));
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: upsampled latent {err:E2} relative RMS vs ComfyUI");
        Assert.That(err, Is.LessThan(1e-4));
    }

    /// <summary>VAE Encode with LTX-2's VAE: one 192 x 256 picture to [1, 128, 1, 6, 8], against ComfyUI's.</summary>
    [Test, Explicit(Needs)]
    public async Task VAE_Encode_matches_ComfyUI()
    {
        var (runtime, services) = Runtime();
        using var _r = runtime;
        var (_, _, vae) = await new CheckpointLoaderSimple().ExecuteAsync(Checkpoint, Context(services, "CheckpointLoaderSimple"), services);
        var picture = new ImageBatch(1, 192, 256, 3, Raw("ltx2_venc_input_f32_1x192x256x3.raw"));
        var latent = await new VAEEncode().ExecuteAsync(picture, vae, Context(services, "VAEEncode"), services);
        Assert.That(latent.Shape, Is.EqualTo(new[] { 1, 128, 1, 6, 8 }));
        double err = RelRms(latent.Data, Raw("ltx2_venc_samples_f32_1x128x1x6x8.raw"));
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: encoded latent {err:E2} relative RMS vs ComfyUI");
        Assert.That(err, Is.LessThan(1e-4));
    }

    /// <summary>
    /// The template's second-pass model (Load LoRA Model Only, the distilled LoRA at 1, CFGGuider cfg 1) on the first pass's setup, with SUBSETS of
    /// the LoRA (ComfyUI cannot bake the whole one on a 32 GB CPU): block 0 + the top-level layers (bf16 in the checkpoint: ComfyUI's float W + B A,
    /// tight) and block 1 alone (fp8: ComfyUI re-rounds W + B A stochastically; the run setting ltx2.comfy_fp8_lora = true repeats that, and the
    /// default exact float form sits further off).
    /// The subset files are local copies of the LoRA's tensors (_scratch/probe/ltx2), not shipped.
    /// </summary>
    [Test, Explicit(Needs + " Also loras/ltx-2-19b-distilled-lora-384-block0-top.safetensors and -block1.safetensors (subsets of the distilled LoRA).")]
    public async Task The_distilled_LoRA_matches_ComfyUI()
    {
        var (runtime, services) = Runtime();
        using var _r = runtime;
        NodeContext Ctx(string type) => Context(services, type);
        var clip = await ClipAsync(services);
        var pos = await new CLIPTextEncode().ExecuteAsync(Prompt.GetProperty("positive").GetString()!, clip, Ctx("CLIPTextEncode"));
        var neg = await new CLIPTextEncode().ExecuteAsync(Prompt.GetProperty("negative").GetString()!, clip, Ctx("CLIPTextEncode"));
        var (positive, negative) = new LTXVConditioning().Execute(pos, neg, 25);
        var (model, _, _) = await new CheckpointLoaderSimple().ExecuteAsync(Checkpoint, Ctx("CheckpointLoaderSimple"), services);
        var audioVae = await new LTXVAudioVAELoader().ExecuteAsync(Checkpoint, services);
        var report = new List<string>();
        // block 1 twice: with the run setting ltx2.comfy_fp8_lora = true ComfyUI's seeded re-rounding of the patched 8-bit weights (its result),
        // then false (the default) the exact float form, which must sit clearly further off (it proves the setting reaches the model).
        foreach (var (subset, prefix, comfyFp8, low, high) in new[] { ("block0-top", "bf16sub", true, 0.0, 1e-4), ("block1", "fp8sub", true, 0.0, 1e-4), ("block1", "fp8sub", false, 1e-3, 5e-2) })
        {
            var lora = new LoraLoaderModelOnly().Execute(model, $"ltx-2-19b-distilled-lora-384-{subset}.safetensors", 1.0);
            var av = new LTXVConcatAVLatent().Execute(new EmptyLTXVLatentVideo().Execute(64, 64, 9, 1), new LTXVEmptyLatentAudio().Execute(9, 25, 1, audioVae));
            var sampler = Ctx("SamplerCustomAdvanced");
            typeof(NodeContext).GetProperty(nameof(NodeContext.Settings))!.SetValue(sampler, new Dictionary<string, string> { [RunSettings.Ltx2ComfyFp8Lora] = comfyFp8 ? "true" : "false" });
            var (output, _) = await new SamplerCustomAdvanced().ExecuteAsync(new RandomNoise().Execute(10), new CFGGuider().Execute(lora, positive, negative, 1.0),
                new KSamplerSelect().Execute("euler"), new LTXVScheduler().Execute(2, 2.05, 0.95, true, 0.1, av), av, sampler, services);
            var (video, audio) = new LTXVSeparateAVLatent().Execute(output);
            double v = RelRms(video.Data, Raw($"ltx2_lora_{prefix}_out_video_f32_1x128x2x2x2.raw")), a = RelRms(audio.Data, Raw($"ltx2_lora_{prefix}_out_audio_f32_1x8x9x16.raw"));
            report.Add($"{subset} ({(comfyFp8 ? "ComfyUI's fp8 rounding" : "exact")}): final latent video {v:E2}, audio {a:E2}");
            TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: {report[^1]} (relative RMS vs ComfyUI)");
            Assert.That(Math.Max(v, a), Is.InRange(low, high), string.Join("; ", report));
        }
    }

    /// <summary>
    /// LTXVImgToVideoInplace against ComfyUI: the 192 x 256 picture into a 17-frame latent at its own size (strength 1 and 0.5), into 9-frame
    /// latents of other sizes (320 x 320: center crop; 96 x 128: plain resize), and bypassed. Latent frame 0 and the per-frame noise mask compared.
    /// </summary>
    [Test, Explicit(Needs)]
    public async Task LTXVImgToVideoInplace_matches_ComfyUI()
    {
        var (runtime, services) = Runtime();
        using var _r = runtime;
        var (_, _, vae) = await new CheckpointLoaderSimple().ExecuteAsync(Checkpoint, Context(services, "CheckpointLoaderSimple"), services);
        var picture = new ImageBatch(1, 192, 256, 3, Raw("ltx2_venc_input_f32_1x192x256x3.raw"));
        var report = new List<string>();
        foreach (var (w, h, n, strength, file) in new[]
        {
            (256, 192, 17, 1.0, "inplace_s10_on_samples_f32_1x128x3x6x8.raw"), (256, 192, 17, 0.5, "inplace_s05_on_samples_f32_1x128x3x6x8.raw"),
            (320, 320, 9, 1.0, "inplace_resize_320x320_samples_f32_1x128x2x10x10.raw"), (128, 96, 9, 1.0, "inplace_resize_128x96_samples_f32_1x128x2x3x4.raw"),
        })
        {
            var empty = new EmptyLTXVLatentVideo().Execute(w, h, n, 1);
            var latent = await new LTXVImgToVideoInplace().ExecuteAsync(vae, picture, empty, strength, false);
            double err = RelRms(latent.Data, Raw(file));
            report.Add($"{w}x{h} strength {strength}: {err:E2}");
            Assert.That(err, Is.LessThan(1e-4), string.Join("; ", report));
            Assert.That(latent.NoiseMask!.Data, Is.EqualTo(new[] { (float)(1 - strength) }.Concat(Enumerable.Repeat(1f, latent.Frames - 1))));
            Assert.That((latent.Type, latent.DownscaleRatioSpacial), Is.EqualTo(((string?)null, (int?)null)), "measured: only samples and noise_mask");
        }
        var bypassed = new EmptyLTXVLatentVideo().Execute(256, 192, 17, 1);
        Assert.That(await new LTXVImgToVideoInplace().ExecuteAsync(vae, picture, bypassed, 1.0, true), Is.SameAs(bypassed));
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: " + string.Join("; ", report) + " (relative RMS vs ComfyUI)");
    }

    [Test]
    public void LTXVPreprocess_maps_compression_to_the_measured_QP()
    {
        var picture = new ImageBatch(1, 2, 2, 3, [.. Enumerable.Range(0, 12).Select(i => i / 12f)]);
        Assert.That(new LTXVPreprocess().Execute(picture, 0), Is.SameAs(picture), "0 passes the picture through (measured)");
        foreach (var (c, qp) in new[] { (1, 11), (15, 11), (20, 14), (33, 27), (35, 29), (40, 34), (45, 38), (50, 42), (51, 47), (100, 47) })
            Assert.That(LTXVPreprocess.Qp(c), Is.EqualTo(qp), $"compression {c}");
    }

    /// <summary>The node at the template's compression against ComfyUI's picture: close, not exact (a different H.264 encoder).</summary>
    [Test, Explicit("Windows: Media Foundation H.264.")]
    public void LTXVPreprocess_at_33_is_close_to_ComfyUI()
    {
        if (!OperatingSystem.IsWindows()) Assert.Ignore("Media Foundation H.264 is Windows-only.");
        var picture = new ImageBatch(1, 192, 256, 3, Raw("ltx2_venc_input_f32_1x192x256x3.raw"));
        var theirs = Raw("ltxv_preprocess_33_f32_1x192x256x3.raw");
        var ours = new LTXVPreprocess().Execute(picture, 33);
        double err = RelRms(ours.Data, theirs), comfyChange = RelRms(theirs, picture.Data);
        TestContext.Out.WriteLine($"LTXV Preprocess 33: {err:E2} relative RMS from ComfyUI's picture (ComfyUI's own change {comfyChange:E2})");
        Assert.That(ours.Data.All(v => Math.Abs(v * 255 - Math.Round(v * 255)) < 1e-3), "values on the 1 / 255 grid, as ComfyUI's");
        Assert.That(err, Is.LessThan(0.75 * comfyChange), "closer to ComfyUI's picture than to the untouched one by a clear margin");
    }

    /// <summary>
    /// Experiment (2026-11-18): Media Foundation H.264 at each constant QP against ComfyUI's LTXV Preprocess at img_compression 33, 35 and 100
    /// (libx264), to choose the compression -> QP mapping. Prints the relative RMS error per QP; asserts only that some QP comes close.
    /// </summary>
    [Test, Explicit("Windows; prints the H.264 QP sweep against ComfyUI's LTXV Preprocess.")]
    [System.Runtime.Versioning.SupportedOSPlatform("windows")]
    public void LTXVPreprocess_QP_sweep_against_ComfyUI()
    {
        if (!OperatingSystem.IsWindows()) Assert.Ignore("Media Foundation H.264 is Windows-only.");
        var picture = new ImageBatch(1, 192, 256, 3, Raw("ltx2_venc_input_f32_1x192x256x3.raw"));
        // more compression points from a ComfyUI measurement folder (pre_<c>_images_f32_1x192x256x3.raw) when SPAWNGEN_PREPROCESS_DIR names one
        var dir = Environment.GetEnvironmentVariable("SPAWNGEN_PREPROCESS_DIR");
        var points = new SortedDictionary<int, float[]>();
        foreach (var c in new[] { 33, 35, 100 }) points[c] = Raw($"ltxv_preprocess_{c}_f32_1x192x256x3.raw");
        if (!string.IsNullOrEmpty(dir))
            foreach (var f in Directory.GetFiles(dir, "pre_*_images_f32_1x192x256x3.raw"))
                if (int.TryParse(Path.GetFileName(f).Split('_')[1], out int c) && !points.ContainsKey(c))
                {
                    var b = File.ReadAllBytes(f); var d = new float[b.Length / 4]; Buffer.BlockCopy(b, 0, d, 0, b.Length); points[c] = d;
                }
        foreach (var (compression, theirs) in points)
        {
            double comfyChange = RelRms(theirs, picture.Data);
            var errors = new List<(int Qp, double Err, double Change)>();
            for (int qp = 10; qp <= 51; qp++)
            {
                var ours = LTXVPreprocess.H264RoundTrip(picture, qp).Data;
                errors.Add((qp, RelRms(ours, theirs), RelRms(ours, picture.Data)));
            }
            var best = errors.MinBy(e => e.Err);
            TestContext.Out.WriteLine($"compression {compression}: ComfyUI changes the picture by {comfyChange:E2}; best QP {best.Qp}: {best.Err:E2} from ComfyUI "
                + $"(ours changes it by {best.Change:E2}); " + string.Join(", ", errors.Where(e => e.Qp % 3 == 0).Select(e => $"QP {e.Qp} {e.Err:E2}")));
            Assert.That(best.Err, Is.LessThan(comfyChange), $"compression {compression}: no QP comes closer to ComfyUI's result than the untouched picture");
        }
    }

    /// <summary>
    /// LTXVAddGuide and LTXVCropGuides against ComfyUI (2026-11-18 probes: _DevComms 2026-11-18-0400 and _scratch/probe/ltx2 guide2 .. guide5): the
    /// appended latent frames, the noise mask, the guide tokens' boxes (keyframe_idxs, exact) for a picture at frame 0 / -1, strength 0.5, a
    /// picture of another size than the latent, an attention mask, two chained guides (and their crop), and video guides at frame 16 (9 frames,
    /// the first frame repeated and the first latent dropped) and frame 8 (17 frames).
    /// </summary>
    [Test, Explicit(Needs)]
    public async Task LTXVAddGuide_matches_ComfyUI()
    {
        var (runtime, services) = Runtime();
        using var _r = runtime;
        NodeContext Ctx(string type) => Context(services, type);
        var (_, _, vae) = await new CheckpointLoaderSimple().ExecuteAsync(Checkpoint, Ctx("CheckpointLoaderSimple"), services);
        var picture = new ImageBatch(1, 192, 256, 3, Raw("ltx2_venc_input_f32_1x192x256x3.raw"));
        var empty = new Conditioning([new CondEntry(new float[4], [1, 1, 4]) { FrameRate = 25 }]);
        var report = new List<string>();
        void Latent(string name, LatentBatch ours, string fixture)
        {
            double e = RelRms(ours.Data, Raw(fixture));
            report.Add($"{name} {e:E2}");
            Assert.That(double.IsFinite(e) && e < 1e-4, $"{name} {e:E2}: {string.Join("; ", report)}");
        }
        void Exact(string name, float[] ours, string fixture) =>
            Assert.That(ours, Is.EqualTo(Raw(fixture)).Within(0), $"{name} must equal ComfyUI's");
        float[] Mask(LatentBatch l) => l.NoiseMask!.Data;

        var lat17 = new EmptyLTXVLatentVideo().Execute(256, 192, 17, 1);
        var (p0, n0, l0) = await new LTXVAddGuide().ExecuteAsync(empty, empty, vae, lat17, picture, 0, 1.0);
        Latent("picture at 0", l0, "addguide_guide_i0_s10_latent_samples_f32_1x128x4x6x8.raw");
        Exact("mask, strength 1", Mask(l0), "addguide_guide_i0_s10_latent_noise_mask_f32_1x1x4x1x1.raw");
        Exact("boxes, frame 0", p0.Entries[0].KeyframeBoxes!, "addguide_guide_i0_s10_pos_0_keyframe_idxs_f32_1x3x48x2.raw");
        Assert.That(n0.Entries[0].KeyframeBoxes, Is.EqualTo(p0.Entries[0].KeyframeBoxes), "the negative gets the same boxes");
        var (cp, _, cl) = new LTXVCropGuides().Execute(p0, n0, l0);
        Exact("crop (all zeros: an empty latent)", cl.Data, "addguide_guide_i0_s10_crop_latent_samples_f32_1x128x3x6x8.raw");
        Exact("crop mask", Mask(cl), "addguide_guide_i0_s10_crop_latent_noise_mask_f32_1x1x3x1x1.raw");
        Assert.That(cp.Entries[0].KeyframeBoxes, Is.Null);
        Assert.That(cp.Entries[0].GuideAttention, Has.Count.EqualTo(1), "guide_attention_entries stay, as in ComfyUI");

        var (pl, _, ll) = await new LTXVAddGuide().ExecuteAsync(empty, empty, vae, lat17, picture, -1, 1.0);
        Latent("picture at -1", ll, "addguide_guide_i-1_s10_latent_samples_f32_1x128x4x6x8.raw");
        Exact("boxes, frame -1", pl.Entries[0].KeyframeBoxes!, "addguide_guide_i-1_s10_pos_0_keyframe_idxs_f32_1x3x48x2.raw");
        var (_, _, lh) = await new LTXVAddGuide().ExecuteAsync(empty, empty, vae, lat17, picture, 0, 0.5);
        Exact("mask, strength 0.5", Mask(lh), "addguide_guide_i0_s05_latent_noise_mask_f32_1x1x4x1x1.raw");

        var (ps, _, ls) = await new LTXVAddGuide().ExecuteAsync(empty, empty, vae, new EmptyLTXVLatentVideo().Execute(320, 320, 9, 1), picture, 0, 1.0);
        Latent("picture into 320 x 320", ls, "addguide_size_320x320_latent_samples_f32_1x128x3x10x10.raw");
        Exact("boxes, 320 x 320", ps.Entries[0].KeyframeBoxes!, "addguide_size_320x320_pos_0_keyframe_idxs_f32_1x3x100x2.raw");

        var solid = MaskBatch.Filled(1, 192, 256, 0.5f);
        var (pm, _, lm) = await new LTXVAddGuide().ExecuteAsync(empty, empty, vae, lat17, picture, 0, 0.8, solid);
        Exact("mask, strength 0.8 with an attention mask", Mask(lm), "addguide_mask_latent_noise_mask_f32_1x1x4x1x1.raw");
        Assert.That((pm.Entries[0].GuideAttention![0].Strength, pm.Entries[0].GuideAttention![0].PixelMask), Is.EqualTo((0.8f, solid)));

        var (p1, n1, l1) = await new LTXVAddGuide().ExecuteAsync(empty, empty, vae, lat17, picture, 0, 1.0);
        var (p2, n2, l2) = await new LTXVAddGuide().ExecuteAsync(p1, n1, vae, l1, picture, -1, 0.7);
        Latent("two guides", l2, "addguide_two_latent_samples_f32_1x128x5x6x8.raw");
        Exact("two guides' mask", Mask(l2), "addguide_two_latent_noise_mask_f32_1x1x5x1x1.raw");
        Exact("two guides' boxes", p2.Entries[0].KeyframeBoxes!, "addguide_two_pos_0_keyframe_idxs_f32_1x3x96x2.raw");
        Exact("two guides cropped", new LTXVCropGuides().Execute(p2, n2, l2).Latent.Data, "addguide_two_crop_latent_samples_f32_1x128x3x6x8.raw");

        // video guides of distinct frames: 0 picture, 1 flipped rows, 2 flipped columns, 3 both, 4 .. 7 the same inverted, 8 picture (17: then 1 .. 8 again)
        ImageBatch Map(Func<int, int, int, int> source, bool invert)
        {
            var o = new ImageBatch(1, 192, 256, 3);
            for (int y = 0; y < 192; y++)
                for (int x = 0; x < 256; x++)
                    for (int c = 0; c < 3; c++)
                    {
                        float v = picture.Data[source(y, x, c)];
                        o.Data[(y * 256 + x) * 3 + c] = invert ? 1 - v : v;
                    }
            return o;
        }
        ImageBatch[] Eight(bool invert) =>
        [
            Map((y, x, c) => (y * 256 + x) * 3 + c, invert), Map((y, x, c) => ((191 - y) * 256 + x) * 3 + c, invert),
            Map((y, x, c) => (y * 256 + 255 - x) * 3 + c, invert), Map((y, x, c) => ((191 - y) * 256 + 255 - x) * 3 + c, invert),
        ];
        ImageBatch[] first = [.. Eight(false), .. Eight(true)];
        ImageBatch Video(IEnumerable<ImageBatch> frames)
        {
            var list = frames.ToList();
            var v = new ImageBatch(list.Count, 192, 256, 3);
            for (int i = 0; i < list.Count; i++) list[i].Data.CopyTo(v.Data, i * list[i].Data.Length);
            return v;
        }
        var nine = Video([.. first, picture]);
        var seventeen = Video([.. first, picture, .. first[1..], picture]);
        var lat33 = new EmptyLTXVLatentVideo().Execute(256, 192, 33, 1);
        var (_, _, lv9) = await new LTXVAddGuide().ExecuteAsync(empty, empty, vae, lat33, nine, 16, 1.0);
        Latent("9-frame video at 16", lv9, "addguide_d9_i16_latent_samples_f32_1x128x6x6x8.raw");
        var (pv17, _, lv17) = await new LTXVAddGuide().ExecuteAsync(empty, empty, vae, lat33, seventeen, 8, 1.0);
        Latent("17-frame video at 8", lv17, "addguide_d17_i8_latent_samples_f32_1x128x7x6x8.raw");
        Exact("boxes, 17-frame video at 8", pv17.Entries[0].KeyframeBoxes!, "addguide_d17_i8_pos_0_keyframe_idxs_f32_1x3x96x2.raw");
        Assert.ThrowsAsync<ArgumentException>(async () => await new LTXVAddGuide().ExecuteAsync(empty, empty, vae, lat33, seventeen, -1, 1.0),
            "a guide past the latent's end is ComfyUI's error");
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: " + string.Join("; ", report));
    }

    /// <summary>
    /// A keyframe-guided LTX-2 sample against ComfyUI (_DevComms 2026-11-18-0600, guidecall_s10 / s06): the 192 x 256 picture as a guide at the last
    /// frame of a 128 x 96, 9-frame video, strength 1 and 0.6, cfg 4, euler, LTXVScheduler 2 steps, seed 10. The sampler's input (the guide frame
    /// appended, its mask) and its output latent (video and audio) are compared: the per-frame denoise mask, the guide tokens' boxes and, at 0.6, the
    /// ln(strength) attention bias all reach the model.
    /// </summary>
    [TestCase(1.0, "s10"), TestCase(0.6, "s06"), Explicit(Needs)]
    public async Task A_keyframe_guided_sample_matches_ComfyUI(double strength, string tag)
    {
        var (runtime, services) = Runtime();
        using var _r = runtime;
        NodeContext Ctx(string type) => Context(services, type);
        var clip = await ClipAsync(services);
        var pos = await new CLIPTextEncode().ExecuteAsync(Prompt.GetProperty("positive").GetString()!, clip, Ctx("CLIPTextEncode"));
        var neg = await new CLIPTextEncode().ExecuteAsync(Prompt.GetProperty("negative").GetString()!, clip, Ctx("CLIPTextEncode"));
        var (positive, negative) = new LTXVConditioning().Execute(pos, neg, 25);
        var (model, _, vae) = await new CheckpointLoaderSimple().ExecuteAsync(Checkpoint, Ctx("CheckpointLoaderSimple"), services);
        var audioVae = await new LTXVAudioVAELoader().ExecuteAsync(Checkpoint, services);
        var picture = new ImageBatch(1, 192, 256, 3, Raw("ltx2_venc_input_f32_1x192x256x3.raw"));
        var (gp, gn, glat) = await new LTXVAddGuide().ExecuteAsync(positive, negative, vae, new EmptyLTXVLatentVideo().Execute(128, 96, 9, 1), picture, -1, strength);
        double inErr = RelRms(glat.Data, Raw($"addguide_call_{tag}_guide_av_in_samples_part0_f32_1x128x3x3x4.raw"));
        Assert.That(glat.NoiseMask!.Data, Is.EqualTo(Raw($"addguide_call_{tag}_guide_av_in_noise_mask_part0_f32_1x1x3x1x1.raw")).Within(1e-7));
        var av = new LTXVConcatAVLatent().Execute(glat, new LTXVEmptyLatentAudio().Execute(9, 25, 1, audioVae));
        var (output, _) = await new SamplerCustomAdvanced().ExecuteAsync(new RandomNoise().Execute(10), new CFGGuider().Execute(model, gp, gn, 4.0),
            new KSamplerSelect().Execute("euler"), new LTXVScheduler().Execute(2, 2.05, 0.95, true, 0.1, av), av, Ctx("SamplerCustomAdvanced"), services);
        var (video, audio) = new LTXVSeparateAVLatent().Execute(output);
        double v = RelRms(video.Data, Raw($"addguide_call_{tag}_guide_out_samples_part0_f32_1x128x3x3x4.raw")),
            a = RelRms(audio.Data, Raw($"addguide_call_{tag}_guide_out_samples_part1_f32_1x8x9x16.raw"));
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: strength {strength}: sampler input (guide appended) {inErr:E2}; final latent video {v:E2}, audio {a:E2} (relative RMS vs ComfyUI)");
        Assert.That(inErr, Is.LessThan(1e-4));
        Assert.That(video.NoiseMask!.Data, Is.EqualTo(glat.NoiseMask.Data), "the output keeps the guide frame's mask, as ComfyUI's");
        Assert.That(v, Is.LessThan(3e-4));
        Assert.That(a, Is.LessThan(3e-4));
    }

    /// <summary>
    /// The run settings for speed (ltx2.host_cache_gb, ltx2.read_ahead_gb: SpawnDev.ILGPU.ML's streamed-block cache and read-ahead) must not change the
    /// result: the same small sample three times on the one model the runtime keeps (defaults; cache 4 GB + read-ahead 2 GB + precision "fp32", which
    /// changes nothing outside CUDA; the same again, now from the cache) gives the same latents bit for bit. On CUDA "fp32" is a different arithmetic,
    /// so there the comparison is against ComfyUI's tolerance instead.
    /// </summary>
    [Test, Explicit(Needs)]
    public async Task The_streaming_settings_do_not_change_the_result()
    {
        var (runtime, services) = Runtime();
        using var _r = runtime;
        NodeContext Ctx(string type) => Context(services, type);
        var clip = await ClipAsync(services);
        var pos = await new CLIPTextEncode().ExecuteAsync(Prompt.GetProperty("positive").GetString()!, clip, Ctx("CLIPTextEncode"));
        var neg = await new CLIPTextEncode().ExecuteAsync(Prompt.GetProperty("negative").GetString()!, clip, Ctx("CLIPTextEncode"));
        var (positive, negative) = new LTXVConditioning().Execute(pos, neg, 25);
        var (model, _, _) = await new CheckpointLoaderSimple().ExecuteAsync(Checkpoint, Ctx("CheckpointLoaderSimple"), services);
        var audioVae = await new LTXVAudioVAELoader().ExecuteAsync(Checkpoint, services);
        async Task<(float[] Video, float[] Audio, double Seconds)> RunAsync(Dictionary<string, string>? settings)
        {
            var av = new LTXVConcatAVLatent().Execute(new EmptyLTXVLatentVideo().Execute(64, 64, 9, 1), new LTXVEmptyLatentAudio().Execute(9, 25, 1, audioVae));
            var sampler = Ctx("SamplerCustomAdvanced");
            typeof(NodeContext).GetProperty(nameof(NodeContext.Settings))!.SetValue(sampler, settings);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var (output, _) = await new SamplerCustomAdvanced().ExecuteAsync(new RandomNoise().Execute(10), new CFGGuider().Execute(model, positive, negative, 4.0),
                new KSamplerSelect().Execute("euler"), new LTXVScheduler().Execute(2, 2.05, 0.95, true, 0.1, av), av, sampler, services);
            var (video, audio) = new LTXVSeparateAVLatent().Execute(output);
            return (video.Data.ToArray(), audio.Data.ToArray(), sw.Elapsed.TotalSeconds);
        }
        var fast = new Dictionary<string, string> { [RunSettings.Ltx2HostCacheGb] = "4", [RunSettings.Ltx2ReadAheadGb] = "2", [RunSettings.Ltx2Precision] = "fp32" };
        var plain = await RunAsync(null);
        var first = await RunAsync(fast);
        var cached = await RunAsync(fast);
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: defaults {plain.Seconds:0.0} s; cache 4 GB + read-ahead 2 GB + fp32 {first.Seconds:0.0} s, again from the cache {cached.Seconds:0.0} s");
        if (runtime.Accelerator.AcceleratorType == ILGPU.Runtime.AcceleratorType.Cuda)
            Assert.That(RelRms(first.Video, plain.Video), Is.LessThan(1e-3), "CUDA: fp32 against the default exact tensor cores");
        else
        {
            Assert.That(first.Video, Is.EqualTo(plain.Video), "video, settings on");
            Assert.That(first.Audio, Is.EqualTo(plain.Audio), "audio, settings on");
        }
        Assert.That(cached.Video, Is.EqualTo(first.Video), "video, from the cache");
        Assert.That(cached.Audio, Is.EqualTo(first.Audio), "audio, from the cache");
    }

    /// <summary>
    /// A real-size run for scale (Tuvok, 2026-11-18): the t2v template's first pass at 384 x 256, 25 frames (4 latent frames of 8 x 12 = 384 video
    /// tokens), 2 steps, cfg 4, then both decodes; seconds per stage printed. No ComfyUI comparison (its CPU run at this size is the 0600
    /// guidecall folders' job); this measures that it runs and how long it takes. Any exception is printed in full and fails the test.
    /// </summary>
    [Test, Explicit(Needs)]
    public async Task The_template_graph_runs_at_384x256x25()
    {
        var (runtime, services) = Runtime();
        using var _r = runtime;
        NodeContext Ctx(string type) => Context(services, type);
        var stages = new List<string>();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        double Lap() { double t = sw.Elapsed.TotalSeconds; sw.Restart(); return t; }
        try
        {
            var clip = await ClipAsync(services);
            var pos = await new CLIPTextEncode().ExecuteAsync(Prompt.GetProperty("positive").GetString()!, clip, Ctx("CLIPTextEncode"));
            var neg = await new CLIPTextEncode().ExecuteAsync(Prompt.GetProperty("negative").GetString()!, clip, Ctx("CLIPTextEncode"));
            var (positive, negative) = new LTXVConditioning().Execute(pos, neg, 25);
            stages.Add($"encode {Lap():0} s");
            var (model, _, vae) = await new CheckpointLoaderSimple().ExecuteAsync(Checkpoint, Ctx("CheckpointLoaderSimple"), services);
            var audioVae = await new LTXVAudioVAELoader().ExecuteAsync(Checkpoint, services);
            var av = new LTXVConcatAVLatent().Execute(new EmptyLTXVLatentVideo().Execute(384, 256, 25, 1), new LTXVEmptyLatentAudio().Execute(25, 25, 1, audioVae));
            var (output, _) = await new SamplerCustomAdvanced().ExecuteAsync(new RandomNoise().Execute(10), new CFGGuider().Execute(model, positive, negative, 4.0),
                new KSamplerSelect().Execute("euler"), new LTXVScheduler().Execute(2, 2.05, 0.95, true, 0.1, av), av, Ctx("SamplerCustomAdvanced"), services);
            stages.Add($"sample (2 steps) {Lap():0} s");
            var (video, audio) = new LTXVSeparateAVLatent().Execute(output);
            var frames = await new VAEDecode().ExecuteAsync(video, vae, Ctx("VAEDecode"), services);
            stages.Add($"video decode {Lap():0} s ({frames.Batch} x {frames.Height} x {frames.Width})");
            var clipOut = await new LTXVAudioVAEDecode().ExecuteAsync(audio, audioVae, Ctx("LTXVAudioVAEDecode"));
            stages.Add($"audio decode {Lap():0} s ({clipOut.Length} samples)");
            Assert.That((frames.Batch, frames.Height, frames.Width), Is.EqualTo((25, 256, 384)));
        }
        catch (Exception e)
        {
            TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: FAILED after: {string.Join("; ", stages)}; at {sw.Elapsed.TotalSeconds:0} s into the next stage: {e}");
            throw;
        }
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: 384 x 256 x 25: " + string.Join("; ", stages));
    }
}
