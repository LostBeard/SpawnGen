using ILGPU;
using ILGPU.Runtime.CPU;
using SpawnDev.Codecs.Video;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>Save Video / Save WEBM: the codec and quality choices, and files that load back (WebM through Codecs' loader) on the CPU accelerator.</summary>
public class SaveVideoTests
{
    static ImageBatch Frames(int n, int w, int h)
    {
        var b = new ImageBatch(n, h, w, 3);
        for (int f = 0; f < n; f++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int i = ((f * h + y) * w + x) * 3;
                    b.Data[i] = x / (float)(w - 1); b.Data[i + 1] = y / (float)(h - 1); b.Data[i + 2] = f / (float)Math.Max(1, n - 1);
                }
        return b;
    }

    [TestCase(0, 1)]
    [TestCase(24, 96)]
    [TestCase(32, 128)]
    [TestCase(61, 244)]
    [TestCase(62, 249)]
    [TestCase(63, 255)]
    public void Crf_maps_to_the_quantizer_index_as_libvpx_and_libaom_do(double crf, int qindex) =>
        Assert.That(VideoFilesAccess.QIndexFromCrf(crf), Is.EqualTo(qindex));

    [TestCase(24.0, 24, 1)]
    [TestCase(29.97, 30000, 1001)]
    [TestCase(23.976, 24000, 1001)]
    [TestCase(12.5, 25, 2)]
    public void The_frame_rate_becomes_a_fraction(double fps, int num, int den) => Assert.That(VideoFilesAccess.Rational(fps), Is.EqualTo((num, den)));

    [Test]
    public async Task A_VP9_WebM_with_sound_loads_back_with_its_frames_rate_and_sound()
    {
        using var context = Context.Create(b => b.CPU());
        using var accelerator = context.CreateCPUAccelerator(0);
        var frames = Frames(6, 64, 48);
        var tone = new AudioClip([Enumerable.Range(0, 22050).Select(i => MathF.Sin(i * 0.05f) * 0.4f).ToArray(), new float[22050]], 44100);
        var bytes = await VideoFilesAccess.EncodeAsync(frames, 12.0, tone, VideoContainer.WebM, VideoCodec.Vp9, VideoFilesAccess.QIndexFromCrf(10), accelerator);
        var video = await new WebMVideoLoaderGpu(accelerator).LoadAsync(bytes);
        Assert.That((video.Width, video.Height, video.Frames.Count, video.FrameRate), Is.EqualTo((64, 48, 6, 12.0)));
        double err = 0;
        for (int f = 0; f < 6; f++)
            for (int i = 0; i < video.Frames[f].Length; i++) err += Math.Abs(video.Frames[f][i] - frames.Data[f * 64 * 48 * 3 + i]);
        Assert.That(err / frames.Data.Length, Is.LessThan(0.03), "the frames come back");
        Assert.That((video.Audio!.SampleRate, video.Audio.Channels), Is.EqualTo((48000, 2)), "44.1 kHz sound is resampled to an Opus rate");
        Assert.That(video.Audio.Samples.Length / 2, Is.EqualTo(24000).Within(960), "half a second of sound");
    }

    [Test]
    public async Task An_AV1_MP4_is_an_MP4()
    {
        using var context = Context.Create(b => b.CPU());
        using var accelerator = context.CreateCPUAccelerator(0);
        var bytes = await VideoFilesAccess.EncodeAsync(Frames(3, 128, 64), 24.0, null, VideoContainer.Mp4, VideoCodec.Av1, null, accelerator);
        Assert.That(System.Text.Encoding.ASCII.GetString(bytes, 4, 4), Is.EqualTo("ftyp"));
    }

    [Test]
    public void Save_Video_offers_ComfyUIs_formats_codecs_and_encoding_modes()
    {
        var s = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(SaveVideo).Assembly).GetSchema("SaveVideo")!;
        Assert.That(s.Inputs.Select(i => i.Name), Is.EqualTo(new[] { "video", "filename_prefix", "format", "format.codec", "format.codec.encoding", "format.codec.encoding.crf" }));
        Assert.That(s.FindInput("format")!.Widget!.ResolveOptions(), Is.EqualTo(new[] { "auto", "mp4", "mkv", "webm" }));
        Assert.That(s.FindInput("format.codec")!.Widget!.ResolveOptions(), Is.EqualTo(new[] { "auto", "h264", "av1" }));
        Assert.That(s.FindInput("filename_prefix")!.Widget!.Default, Is.EqualTo("video/ComfyUI"));
        var w = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(SaveVideo).Assembly).GetSchema("SaveWEBM")!;
        Assert.That(w.Inputs.Select(i => i.Name), Is.EqualTo(new[] { "images", "filename_prefix", "codec", "fps", "crf" }));
    }

    [TestCase("auto", 96)]
    [TestCase("av1", 128)]
    public async Task Save_Video_writes_MKV_as_Matroska_named_and_described_as_ComfyUI_does(string codec, int width)
    {
        // Measured with ComfyUI (Save Video, format mkv, codec auto and av1, 3 frames 128x64 at 24 fps): video/probe_mkv_00001_.mkv,
        // metadata {kind video, width 128, height 64, duration 0.125, fps 24.0} with no frame_count (MP4 alone carries it).
        using var ilgpu = ILGPU.Context.Create(b => b.CPU());
        using var acc = ilgpu.CreateCPUAccelerator(0);
        using var runtime = new MlRuntime(new FolderModelSource(Path.GetTempPath()), acc);
        var store = new MemoryResultStore();
        byte[]? written = null;
        store.Saved += (_, data) => written = data;
        var dump = Environment.GetEnvironmentVariable("SPAWNGEN_DUMP_DIR");
        var services = new Services(store, runtime);
        var encoding = new DynamicChoice("re-encode", new Dictionary<string, object?> { ["crf"] = 20.0 });
        var format = new DynamicChoice("mkv", new Dictionary<string, object?> { ["codec"] = new DynamicChoice(codec, new Dictionary<string, object?> { ["encoding"] = encoding }) });

        var (_, ui) = await new SaveVideo().ExecuteAsync(new VideoClip(Frames(3, width, 64), 24.0), NodeCtx(services), services, "video/probe_mkv", format);

        var file = (ResultFile)((object[])ui["images"]!)[0];
        Assert.That((file.Filename, file.Subfolder, file.Type), Is.EqualTo(("probe_mkv_00001_.mkv", "video", "output")));
        var m = file.Metadata!;
        Assert.That((m.Kind, m.Width, m.Height, m.Duration, m.Fps, m.FrameCount), Is.EqualTo(("video", width, 64, (double?)0.125, (double?)24.0, (int?)null)));
        Assert.That(System.Text.Encoding.ASCII.GetString(written!).Contains("matroska"), Is.True, "EBML DocType matroska");
        Assert.That(System.Text.Encoding.ASCII.GetString(written!).Contains("webm"), Is.False, "not a WebM DocType");
        if (dump is not null) File.WriteAllBytes(Path.Combine(dump, $"savevideo_{codec}.mkv"), written!);
        if (codec == "av1") return; // Codecs' loader decodes AV1 key frames only so far; the AV1 .mkv was checked with FFmpeg (PyAV) instead
        var video = await new WebMVideoLoaderGpu(acc).LoadAsync(written!);
        Assert.That((video.Width, video.Height, video.Frames.Count, video.FrameRate), Is.EqualTo((96, 64, 3, 24.0)), "auto at 96x64 is VP9");
    }

    sealed class Services(IResultStore store, MlRuntime runtime) : IServiceProvider
    {
        public object? GetService(Type t) => t == typeof(IResultStore) ? store : t == typeof(MlRuntime) ? runtime : null;
    }

    static NodeContext NodeCtx(IServiceProvider services) =>
        (NodeContext)Activator.CreateInstance(typeof(NodeContext), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, null,
            ["1", "SaveVideo", services, CancellationToken.None, null, new Lazy<string>(() => "{}"), new Dictionary<string, string>(), null, null, 0], null)!;

    static class VideoFilesAccess
    {
        static readonly Type T = typeof(SaveVideo).Assembly.GetType("SpawnGen.Nodes.ML.VideoFiles")!;
        public static int QIndexFromCrf(double crf) => (int)T.GetMethod("QIndexFromCrf")!.Invoke(null, [crf])!;
        public static (int, int) Rational(double fps) => ((int, int))T.GetMethod("Rational")!.Invoke(null, [fps])!;
        public static Task<byte[]> EncodeAsync(ImageBatch frames, double fps, AudioClip? audio, VideoContainer container, VideoCodec codec, int? qindex, ILGPU.Runtime.Accelerator acc) =>
            (Task<byte[]>)T.GetMethod("EncodeAsync")!.Invoke(null, [frames, fps, audio, container, codec, qindex, acc])!;
    }
}
