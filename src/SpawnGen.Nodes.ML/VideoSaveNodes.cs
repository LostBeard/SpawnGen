using SpawnDev.Codecs.Video;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries (Save Video, Save WEBM: inputs, tooltips, defaults) and black-box
// runs (2026-11-12): file names (prefix_00001_.ext in the prefix's subfolder), the ui entry and its metadata (kind video, size, duration, fps;
// frame_count for MP4 only), auto picking MP4 (WebM for AV1). Encoding is SpawnDev.Codecs' VideoFileEncoderGpu. SpawnDev.Codecs is patent-free:
// no H.264, so SpawnGen writes AV1 (sides that are multiples of 64) or VP9 where ComfyUI writes H.264. crf maps to the encoders' quantizer index as
// libvpx / libaom map cq-level (crf x 4; 62 -> 249, 63 -> 255); every frame is a key frame for now, so files are larger than ComfyUI's.

/// <summary>What the video save nodes share: the codec choice, the frame rate as a fraction, the sound, the ui entry.</summary>
internal static class VideoFiles
{
    /// <summary>libvpx / libaom's quantizer_to_qindex: crf (cq-level) 0..63 to the 0..255 quantizer index (at least 1: lossless is not available).</summary>
    public static int QIndexFromCrf(double crf)
    {
        int q = (int)Math.Round(Math.Clamp(crf, 0, 63));
        return Math.Max(1, q switch { 62 => 249, 63 => 255, _ => q * 4 });
    }

    /// <summary>AV1 needs both sides to be multiples of 64 in Codecs' encoder.</summary>
    public static bool Av1Fits(int width, int height) => width % 64 == 0 && height % 64 == 0;

    /// <summary>The frame rate as a fraction: whole numbers over 1, the NTSC rates (29.97, 23.976, 59.94) over 1001, others over 1000 reduced.</summary>
    public static (int Num, int Den) Rational(double fps)
    {
        if (fps <= 0 || double.IsNaN(fps)) throw new ArgumentOutOfRangeException(nameof(fps), "The frame rate must be above 0.");
        if (Math.Abs(fps - Math.Round(fps)) < 1e-6) return ((int)Math.Round(fps), 1);
        double ntsc = fps * 1001 / 1000;
        if (Math.Abs(ntsc - Math.Round(ntsc)) < 1e-3) return ((int)Math.Round(ntsc) * 1000, 1001);
        int num = (int)Math.Round(fps * 1000), den = 1000, g = Gcd(num, den);
        return (num / g, den / g);
        static int Gcd(int a, int b) => b == 0 ? a : Gcd(b, a % b);
    }

    /// <summary>The sound as the Opus encoder takes it: interleaved, at an Opus rate (others resampled to 48 kHz).</summary>
    public static LoadedAudio? Audio(AudioClip? clip)
    {
        if (clip is null || clip.ChannelCount == 0) return null;
        if (clip.SampleRate is not (8000 or 12000 or 16000 or 24000 or 48000)) clip = AudioDsp.Resample(clip, 48000);
        int n = clip.Channels[0].Length, c = clip.ChannelCount;
        var interleaved = new float[n * c];
        for (int ch = 0; ch < c; ch++)
            for (int i = 0; i < n; i++) interleaved[i * c + ch] = clip.Channels[ch][i];
        return new LoadedAudio(clip.SampleRate, c, interleaved);
    }

    public static async Task<(ResultFile File, UiOutput Ui)> SaveAsync(ImageBatch frames, double fps, AudioClip? audio, string filenamePrefix, VideoContainer container,
        VideoCodec codec, int? qindex, NodeContext context, IServiceProvider services)
    {
        var accelerator = (await GpuHelpers.RuntimeAsync(services)).Accelerator;
        var bytes = await EncodeAsync(frames, fps, audio, container, codec, qindex, accelerator);
        int w = frames.Width, h = frames.Height;
        var prefix = SavePrefix.ComputeVars(filenamePrefix, w, h, DateTime.Now);
        var mp4 = container == VideoContainer.Mp4;
        var file = (await Results.For(services).SaveAsync("output", prefix, bytes, mp4 ? "mp4" : "webm")) with
        {
            Metadata = new ResultMetadata("video", w, h)
            {
                Duration = Math.Round(frames.Batch / fps, 6), Fps = fps, FrameCount = mp4 ? frames.Batch : null,
            },
        };
        var ui = new UiOutput("images", new object[] { file });
        ui["animated"] = new object[] { true };
        return (file, ui);
    }

    /// <summary>The frames (and sound) as a video file's bytes.</summary>
    public static async Task<byte[]> EncodeAsync(ImageBatch frames, double fps, AudioClip? audio, VideoContainer container, VideoCodec codec, int? qindex,
        ILGPU.Runtime.Accelerator accelerator)
    {
        if (frames.Batch == 0) throw new ArgumentException("There are no frames to save.");
        int w = frames.Width, h = frames.Height, per = w * h * frames.Channels;
        var list = new List<float[]>(frames.Batch);
        for (int f = 0; f < frames.Batch; f++)
        {
            var rgb = new float[w * h * 3];
            int o = f * per, ch = frames.Channels;
            for (int p = 0; p < w * h; p++)
                for (int k = 0; k < 3; k++) rgb[p * 3 + k] = frames.Data[o + p * ch + Math.Min(k, ch - 1)];
            list.Add(rgb);
        }
        var (num, den) = Rational(fps);
        using var encoder = new VideoFileEncoderGpu(accelerator);
        return await encoder.EncodeAsync(list, w, h, num, den, Audio(audio), new VideoFileEncodeOptions { Container = container, Codec = codec, QIndex = qindex });
    }
}

/// <summary>
/// Save Video: the VIDEO as a file in the output folder (prefix "video/ComfyUI" gives video/ComfyUI_00001_.mp4), with its sound (Opus). ComfyUI's
/// format / codec / encoding choices; codec auto writes AV1 when both sides are multiples of 64, else VP9 (ComfyUI writes H.264, which SpawnGen
/// does not: SpawnDev.Codecs is patent-free). Format auto is MP4, WebM for AV1. MKV waits for Codecs' Matroska writer. The prompt and workflow are
/// not embedded in the file yet.
/// </summary>
[Node("SaveVideo", Category = "video", DisplayName = "Save Video", IsOutput = true, Description = "Saves the input videos to your ComfyUI output directory.")]
public sealed class SaveVideo
{
    const string CodecTip = "The output video codec. Auto preserves a compatible source stream. H.264 and AV1 re-encoding support SDR, HDR (HLG), and HDR PQ.";

    static DynamicOption Codec(string key, double crf, double max) => new(key, DynamicInput.Nested("encoding",
        [
            new DynamicOption("auto"),
            new DynamicOption("re-encode", DynamicInput.Float("crf", crf, 0, max, 1, "Lower values produce higher quality and larger files.")),
        ], tooltip: $"Automatic preserves compatible {(key == "h264" ? "H.264" : "AV1")} streams. Re-encode applies custom encoding options."));

    static DynamicInput Codecs(bool h264) => DynamicInput.Nested("codec",
        h264 ? [new DynamicOption("auto"), Codec("h264", 18, 51), Codec("av1", 24, 63)] : [new DynamicOption("auto"), Codec("av1", 24, 63)], tooltip: CodecTip);

    public static IEnumerable<DynamicOption> Formats() =>
        [new("auto", Codecs(true)), new("mp4", Codecs(true)), new("mkv", Codecs(true)), new("webm", Codecs(false))];

    public async Task<(VideoClip Video, UiOutput Ui)> ExecuteAsync(
        [Input(Tooltip = "The video to save.")] VideoClip video, NodeContext context, IServiceProvider services,
        [Input(Tooltip = "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.")]
        string filenamePrefix = "video/ComfyUI",
        [DynamicCombo(nameof(Formats)), Input(Tooltip = "The output container. Auto uses MP4 for Auto/H.264 and WebM for AV1. MP4, MKV, and WebM select a specific container.")]
        DynamicChoice? format = null)
    {
        var codecChoice = format?.Get<DynamicChoice?>("codec", null);
        string formatKey = format?.Key ?? "auto", codecKey = codecChoice?.Key ?? "auto";
        var encoding = codecChoice?.Get<DynamicChoice?>("encoding", null);
        int? qindex = encoding?.Key == "re-encode" ? VideoFiles.QIndexFromCrf(encoding.Get("crf", codecKey == "h264" ? 18.0 : 24.0)) : null;

        if (codecKey == "h264")
            throw new NotSupportedException("H.264 is not available: SpawnGen's codecs are patent-free. Choose codec auto (AV1 or VP9) or av1.");
        if (formatKey == "mkv")
            throw new NotSupportedException("MKV is not available yet (SpawnDev.Codecs writes MP4 and WebM so far); choose mp4, webm or auto.");
        var frames = video.Frames;
        bool av1 = codecKey == "av1" || VideoFiles.Av1Fits(frames.Width, frames.Height);
        if (av1 && !VideoFiles.Av1Fits(frames.Width, frames.Height))
            throw new NotSupportedException($"AV1 needs sides that are multiples of 64 for now, got {frames.Width}x{frames.Height}; choose codec auto (VP9).");
        var container = formatKey switch { "webm" => VideoContainer.WebM, "mp4" => VideoContainer.Mp4, _ => codecKey == "av1" ? VideoContainer.WebM : VideoContainer.Mp4 };
        var (_, ui) = await VideoFiles.SaveAsync(frames, video.Fps, video.Audio, filenamePrefix, container, av1 ? VideoCodec.Av1 : VideoCodec.Vp9, qindex, context, services);
        return (video, ui);
    }
}

/// <summary>
/// Save WEBM: the pictures as a WebM video (VP9 or AV1, no sound) in the output folder, prefix_00001_.webm. crf as ComfyUI's (lower is better and
/// larger). An alpha channel is not kept yet (ComfyUI keeps it with VP9).
/// </summary>
[Node("SaveWEBM", Category = "video", DisplayName = "Save WEBM", IsOutput = true, OutputNames = ["images"])]
public sealed class SaveWEBM
{
    public async Task<(ImageBatch Images, UiOutput Ui)> ExecuteAsync(
        [Input(Tooltip = "RGBA images are saved with their alpha channel as transparency (vp9 codec only).")] ImageBatch images, NodeContext context, IServiceProvider services,
        string filenamePrefix = "ComfyUI",
        [Input(Options = ["vp9", "av1"])] string codec = "vp9",
        [Input(Min = 0.01, Max = 1000.0, Step = 0.01)] double fps = 24.0,
        [Input(Min = 0.0, Max = 63.0, Step = 1, Tooltip = "Higher crf means lower quality with a smaller file size, lower crf means higher quality higher filesize.")] double crf = 32.0)
    {
        if (codec == "av1" && !VideoFiles.Av1Fits(images.Width, images.Height))
            throw new NotSupportedException($"AV1 needs sides that are multiples of 64 for now, got {images.Width}x{images.Height}; choose vp9.");
        var (_, ui) = await VideoFiles.SaveAsync(images, fps, null, filenamePrefix, VideoContainer.WebM, codec == "av1" ? VideoCodec.Av1 : VideoCodec.Vp9,
            VideoFiles.QIndexFromCrf(crf), context, services);
        return (images, ui);
    }
}
