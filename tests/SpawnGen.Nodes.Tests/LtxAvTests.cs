using System.Text.Json;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// LTX-2's host-side nodes against ComfyUI (bef11fc, measured 2026-11-17: Fixtures/ltx2, from _scratch/probe/ltx2/host_run*.py): empty latents,
/// the audio + video latent, LTXVScheduler, LTXVConditioning, LTXVCropGuides, and the noise the packed latent starts from.
/// </summary>
public class LtxAvTests
{
    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "ltx2");

    static JsonElement Latents => JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "ltx2_host_latents.json"))).RootElement.GetProperty("latents");

    static int[] ShapeOf(string name) => Latents.GetProperty(name).GetProperty("samples").EnumerateArray().Select(v => v.GetInt32()).ToArray();

    static float[] Sigmas(string name) =>
        // Python's json writes a NaN sigma as a bare NaN, which .NET's parser refuses: read it as null
        JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, name + ".json")).Replace("NaN", "null")).RootElement.GetProperty("sigmas").EnumerateArray()
            .Select(v => v.ValueKind == JsonValueKind.Number ? (float)v.GetDouble() : float.NaN).ToArray();

    /// <summary>LTX-2 19B's audio latent layout (its checkpoint config: 8 channels, 64 mel bins / 4, 16000 / 160 / 4 = 25 frames a second).</summary>
    static readonly LtxAudioLatentLayout Layout = LtxAudioLatentLayout.FromConfig(JsonDocument.Parse("""
        {"preprocessing": {"audio": {"sampling_rate": 16000}, "stft": {"hop_length": 160}},
         "model": {"params": {"ddconfig": {"z_channels": 8, "mel_bins": 64, "ch_mult": [1, 2, 4]}}}}
        """).RootElement);

    static VaeHandle AudioVae => new((IVaeModel)Activator.CreateInstance(typeof(LTXVAudioVAELoader).Assembly.GetType("SpawnGen.Nodes.ML.LtxAudioVaeModel")!,
        "ltx-2-19b-dev-fp8.safetensors", Layout)!);

    [Test]
    public void The_empty_video_latent_is_ComfyUIs()
    {
        foreach (var (w, h, n, b) in new[] { (64, 64, 9, 1), (768, 512, 97, 1), (96, 160, 17, 2), (64, 64, 1, 1), (100, 70, 10, 1), (65, 64, 9, 1), (64, 64, 8, 1) })
        {
            var latent = new EmptyLTXVLatentVideo().Execute(w, h, n, b);
            Assert.That(latent.Shape, Is.EqualTo(ShapeOf($"vid_{w}x{h}x{n}_b{b}")), $"{w} x {h} x {n}, batch {b}");
            Assert.That((latent.IsVideo, latent.DownscaleRatioSpacial, latent.Type), Is.EqualTo((true, (int?)32, (string?)null)));
            Assert.That(latent.Data, Is.All.EqualTo(0f));
        }
    }

    [Test]
    public void The_empty_audio_latent_is_ComfyUIs_with_its_rounding()
    {
        Assert.That(Layout, Is.EqualTo(new LtxAudioLatentLayout(8, 16, 25)));
        var cases = new (int N, double Fps, int B, string Name)[]
        {
            (9, 25, 1, "aud_9_25_b1"), (97, 25, 1, "aud_97_25_b1"), (1, 25, 1, "aud_1_25_b1"), (121, 24, 1, "aud_121_24_b1"), (10, 30, 2, "aud_10_30_b2"),
            (97, 23.976, 1, "aud_97_23.976_b1"), (1000, 1, 1, "aud_1000_1_b1"), (24, 24, 1, "aud_24_24_b1"), (25, 25, 1, "aud_25_25_b1"),
            (7, 24, 1, "aud2_7_24"), (13, 24, 1, "aud2_13_24"), (19, 24, 1, "aud2_19_24"), (11, 30, 1, "aud2_11_30"), (2, 3, 1, "aud2_2_3"),
            (1, 2, 1, "aud2_1_2"), (3, 2, 1, "aud2_3_2"), (5, 2, 1, "aud2_5_2"), (3, 50, 1, "aud2_3_50"), (2, 75, 1, "aud2_2_75"),
        };
        foreach (var (n, fps, b, name) in cases)
        {
            var latent = new LTXVEmptyLatentAudio().Execute(n, fps, b, AudioVae);
            Assert.That(latent.Shape, Is.EqualTo(ShapeOf(name)), $"{n} frames at {fps} fps");
            Assert.That((latent.IsAudio, latent.Type), Is.EqualTo((true, "audio")));
        }
        // 1 frame at 50 fps rounds to no audio frame: ComfyUI makes an empty latent, SpawnGen says so
        Assert.Throws<ArgumentException>(() => new LTXVEmptyLatentAudio().Execute(1, 50, 1, AudioVae));
    }

    [Test]
    public void Concat_and_separate_keep_ComfyUIs_layout_and_keys()
    {
        var video = new EmptyLTXVLatentVideo().Execute(96, 64, 17, 1);
        var audio = new LTXVEmptyLatentAudio().Execute(17, 25, 1, AudioVae);
        // distinct values, to see where each lands
        for (int i = 0; i < video.Data.Length; i++) video.Data[i] = i;
        for (int i = 0; i < audio.Data.Length; i++) audio.Data[i] = -1 - i;
        var av = new LTXVConcatAVLatent().Execute(video, audio);
        var parts = Latents.GetProperty("concat_96x64x17").GetProperty("samples").GetProperty("nested").EnumerateArray()
            .Select(p => p.EnumerateArray().Select(v => v.GetInt32()).ToArray()).ToArray();
        Assert.That(av.IsNested, Is.True);
        Assert.That(av.Parts, Is.EqualTo(parts));
        Assert.That(av.Shape, Is.EqualTo(new[] { 1, 1, video.ItemSize + audio.ItemSize }));
        Assert.That(av.Data, Is.EqualTo(video.Data.Concat(audio.Data).ToArray()), "video values, then audio values");
        Assert.That((av.Type, av.DownscaleRatioSpacial), Is.EqualTo(("audio", (int?)32)));
        var (v, a) = new LTXVSeparateAVLatent().Execute(av);
        Assert.That((v.Shape, a.Shape), Is.EqualTo((ShapeOf("sep_video"), ShapeOf("sep_audio"))));
        Assert.That((v.Data, a.Data), Is.EqualTo((video.Data, audio.Data)));
        // measured: both parts carry all of the audio + video latent's keys (the video part says type "audio")
        Assert.That((v.Type, v.DownscaleRatioSpacial, v.IsVideo), Is.EqualTo(("audio", (int?)32, true)));
        Assert.That((a.Type, a.DownscaleRatioSpacial, a.IsAudio), Is.EqualTo(("audio", (int?)32, true)));
        Assert.Throws<ArgumentException>(() => new LTXVSeparateAVLatent().Execute(video));
    }

    [Test]
    public void The_packed_latent_starts_from_ComfyUIs_noise()
    {
        // ComfyUI's model input at sigma 1 (empty latent, seed 10): the noise, the same for both CFG rows
        var b = File.ReadAllBytes(Path.Combine(Dir, "ltx2_tcall_0_input_f32_2x1x2176.raw"));
        var theirs = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, theirs, 0, b.Length);
        var av = LatentBatch.Nested(new EmptyLTXVLatentVideo().Execute(64, 64, 9, 1), new LTXVEmptyLatentAudio().Execute(9, 25, 1, AudioVae));
        Assert.That(av.ItemSize, Is.EqualTo(2176));
        var ours = (float[])typeof(LTXVScheduler).Assembly.GetType("SpawnGen.Nodes.ML.SamplingCore")!
            .GetMethod("InitialNoise", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!.Invoke(null, [10L, 0, av])!;
        // TorchRandom reproduces torch.randn to a float32 ulp or so (its Box-Muller), as elsewhere
        Assert.That(ours, Is.EqualTo(theirs[..2176]).Within(5e-6));
        Assert.That(theirs[2176..], Is.EqualTo(theirs[..2176]));
    }

    [Test]
    public void LTXVScheduler_matches_ComfyUI()
    {
        LatentBatch V(int w, int h, int n, int batch = 1) => new EmptyLTXVLatentVideo().Execute(w, h, n, batch);
        var cases = new (string Name, int Steps, double Max, double Base, bool Stretch, double Terminal, LatentBatch? Latent)[]
        {
            ("sched_nolatent_2", 2, 2.05, 0.95, true, 0.1, null), ("sched_nolatent_3", 3, 2.05, 0.95, true, 0.1, null),
            ("sched_nolatent_8", 8, 2.05, 0.95, true, 0.1, null), ("sched_nolatent_20", 20, 2.05, 0.95, true, 0.1, null),
            ("sched_nolatent_30", 30, 2.05, 0.95, true, 0.1, null), ("sched_nolatent_20_False_0.1", 20, 2.05, 0.95, false, 0.1, null),
            ("sched_nolatent_20_True_0.0", 20, 2.05, 0.95, true, 0.0, null), ("sched_nolatent_20_True_0.5", 20, 2.05, 0.95, true, 0.5, null),
            ("sched_nolatent_20_shift1_1", 20, 1, 1, true, 0.1, null), ("sched_nolatent_20_shift4_05", 20, 4, 0.5, true, 0.1, null),
            ("sched_video_64x64x9_20", 20, 2.05, 0.95, true, 0.1, V(64, 64, 9)), ("sched_video_768x512x97_20", 20, 2.05, 0.95, true, 0.1, V(768, 512, 97)),
            ("sched_video_1280x704x121_20", 20, 2.05, 0.95, true, 0.1, V(1280, 704, 121)), ("sched_video_256x256x33_20", 20, 2.05, 0.95, true, 0.1, V(256, 256, 33)),
            ("sched_video_64x64x9_b3_20", 20, 2.05, 0.95, true, 0.1, V(64, 64, 9, 3)),
            ("sched_runav_20", 20, 2.05, 0.95, true, 0.1, LatentBatch.Nested(V(64, 64, 9), new LTXVEmptyLatentAudio().Execute(9, 25, 1, AudioVae))),
            ("sched_audio97_20", 20, 2.05, 0.95, true, 0.1, new LTXVEmptyLatentAudio().Execute(97, 25, 1, AudioVae)),
            ("s1_nolatent_nostretch", 1, 2.05, 0.95, false, 0.1, null), ("s1_v8_nostretch", 1, 2.05, 0.95, false, 0.1, V(64, 64, 9)),
            ("sched_nolatent_1", 1, 2.05, 0.95, true, 0.1, null), ("s1_v4096", 1, 2.05, 0.95, true, 0.1, V(512, 512, 121)),
            ("s1_nolatent_term0", 1, 2.05, 0.95, true, 0.0, null),
        };
        foreach (var (name, steps, mx, bs, stretch, term, latent) in cases)
            Assert.That(new LTXVScheduler().Execute(steps, mx, bs, stretch, term, latent).Values, Is.EqualTo(Sigmas(name)).Within(1e-6), name);
        // One step with stretch where ComfyUI gives NaN (every shift but mu 2.05): SpawnGen gives the limit, [terminal, 0]
        foreach (var name in new[] { "s1_v8", "s1_v1024", "s1_v14080", "s1_nolatent_1_1", "s1_nolatent_4_05", "s1_nolatent_0_0" })
            Assert.That(float.IsNaN(Sigmas(name)[0]), Is.True, name + " is NaN in ComfyUI");
        Assert.That(new LTXVScheduler().Execute(1, 2.05, 0.95, true, 0.1, V(64, 64, 9)).Values, Is.EqualTo(new[] { 0.1f, 0f }).Within(1e-7));
    }

    [Test]
    public void LTXVConditioning_sets_the_frame_rate_and_CropGuides_passes_through()
    {
        var entry = new CondEntry([1f, 2f], [1, 2, 1]) { FrameRate = 25 };
        var (pos, neg) = new LTXVConditioning().Execute(new Conditioning([entry]), new Conditioning([entry]), 24);
        Assert.That((pos.Entries[0].FrameRate, neg.Entries[0].FrameRate), Is.EqualTo(((double?)24, (double?)24)));
        Assert.That(pos.Entries[0].Embeddings, Is.EqualTo(entry.Embeddings));

        var video = new LTXVSeparateAVLatent().Execute(LatentBatch.Nested(new EmptyLTXVLatentVideo().Execute(64, 64, 9, 1),
            new LTXVEmptyLatentAudio().Execute(9, 25, 1, AudioVae))).VideoLatent;
        var (p2, n2, latent) = new LTXVCropGuides().Execute(pos, neg, video);
        Assert.That((p2, n2), Is.EqualTo((pos, neg)));
        // measured, no guides: the same values, the keys gone, a noise mask of ones over the latent frames ([1, 1, 2, 1, 1] in ComfyUI)
        Assert.That(latent.Data, Is.EqualTo(video.Data));
        Assert.That((latent.Type, latent.DownscaleRatioSpacial), Is.EqualTo(((string?)null, (int?)null)));
        Assert.That(Latents.GetProperty("crop_v_latent").GetProperty("noise_mask").EnumerateArray().Select(v => v.GetInt32()), Is.EqualTo(new[] { 1, 1, 2, 1, 1 }));
        Assert.That(latent.NoiseMask!.Shape, Is.EqualTo(new[] { 1, 2, 1 }));
        Assert.That(latent.NoiseMask.Data, Is.All.EqualTo(1f));
    }

    [Test]
    public void Picture_nodes_refuse_the_audio_and_video_latents()
    {
        var registry = new SpawnGen.Engine.Nodes.NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly);
        var executor = new SpawnGen.Engine.Execution.PromptExecutor(registry);
        var g = new SpawnGen.Engine.Graph.PromptGraph();
        g.Add("1", "EmptyLTXVLatentVideo", ("width", 64L), ("height", 64L), ("length", 9L), ("batch_size", 1L));
        g.Add("2", "LatentUpscaleBy", ("samples", SpawnGen.Engine.Graph.PromptInput.Link("1")), ("upscale_method", "nearest-exact"), ("scale_by", 2.0));
        g.Add("3", "PreviewAny", ("source", SpawnGen.Engine.Graph.PromptInput.Link("2")));
        var e = Assert.ThrowsAsync<SpawnGen.Engine.Execution.NodeExecutionException>(() => executor.ExecuteAsync(g));
        Assert.That(e!.InnerException!.Message, Does.Contain("video latents"));
    }
}
