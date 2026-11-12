using SpawnDev.Codecs.Video;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries and black-box runs (2026-11-12): the files Load Video lists (by
// extension), what Get Video Components gives for a loaded file (bit depth "8", colour space "sRGB", no audio when the file has none), and Load
// Video's ui entry. The decoding is SpawnDev.Codecs' WebM loader, whose frames Geordi checked bit for bit against ComfyUI's LoadVideo
// (Fixtures/codecs-load-refs).

/// <summary>
/// Load Video: a video file from the input folder as a VIDEO (frames, frame rate, sound). Lists what ComfyUI lists (3gp, avi, m4v, mkv, mov, mp4,
/// mpeg, mpg, ogv, webm, wmv); SpawnGen decodes WebM with VP9 video (and Opus or Vorbis sound) so far. Frames are RGB in 0..1 (an alpha channel is
/// dropped, as ComfyUI does); the sound is channels x samples.
/// </summary>
[Node("LoadVideo", Category = "video", DisplayName = "Load Video")]
public sealed class LoadVideo
{
    /// <summary>The extensions ComfyUI's Load Video lists (measured).</summary>
    public static readonly string[] Extensions = [".3gp", ".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ogv", ".webm", ".wmv"];

    public static string[] VideoNames() =>
        [.. InputFiles.Store.List(n => Extensions.Contains(Path.GetExtension(n).ToLowerInvariant()))];

    public async Task<(VideoClip Video, UiOutput Ui)> ExecuteAsync([Input(OptionsProvider = nameof(VideoNames))] string file, IServiceProvider services)
    {
        var name = RecordAudio.StripAnnotation(file);
        if (Path.GetExtension(name).ToLowerInvariant() != ".webm")
            throw new NotSupportedException($"'{name}': SpawnGen loads WebM (VP9) videos so far; {Path.GetExtension(name)} files are not decoded yet.");
        var bytes = await InputFiles.Store.ReadAsync(name);
        var accelerator = (await GpuHelpers.RuntimeAsync(services)).Accelerator;
        var video = await new WebMVideoLoaderGpu(accelerator).LoadAsync(bytes);
        var clip = new VideoClip(Frames(video), video.FrameRate, Audio(video.Audio), "8", "sRGB");

        var ui = new UiOutput("images", new object[]
        {
            new Dictionary<string, object?>
            {
                ["filename"] = Path.GetFileName(name),
                ["subfolder"] = Path.GetDirectoryName(name)?.Replace('\\', '/') ?? "",
                ["type"] = "input",
                ["metadata"] = new Dictionary<string, object?>
                {
                    ["kind"] = "video", ["width"] = video.Width, ["height"] = video.Height,
                    ["duration"] = Math.Round(Math.Max(clip.Seconds, clip.Audio is { } a ? a.Seconds : 0), 3), ["fps"] = video.FrameRate,
                },
            },
        });
        ui["animated"] = new object[] { true };
        return (clip, ui);
    }

    /// <summary>The decoded frames (each H x W x 3 interleaved, ComfyUI's IMAGE layout) as one batch.</summary>
    internal static ImageBatch Frames(LoadedVideo video)
    {
        int per = video.Width * video.Height * 3;
        var batch = new ImageBatch(video.Frames.Count, video.Height, video.Width, 3);
        for (int f = 0; f < video.Frames.Count; f++)
        {
            if (video.Frames[f].Length != per) throw new InvalidDataException($"Frame {f} has {video.Frames[f].Length} values, expected {per}.");
            Array.Copy(video.Frames[f], 0, batch.Data, f * per, per);
        }
        return batch;
    }

    /// <summary>Interleaved samples to channels x samples (ComfyUI's waveform layout); null without sound.</summary>
    internal static AudioClip? Audio(LoadedAudio? audio)
    {
        if (audio is null || audio.Channels <= 0) return null;
        int n = audio.Samples.Length / audio.Channels;
        var channels = new float[audio.Channels][];
        for (int c = 0; c < audio.Channels; c++)
        {
            var s = channels[c] = new float[n];
            for (int i = 0; i < n; i++) s[i] = audio.Samples[i * audio.Channels + c];
        }
        return new AudioClip(channels, audio.SampleRate);
    }
}
