using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// ComfyUI's video nodes, from its /object_info entries (schemas and tooltips): a VIDEO is frames + frame rate (+ sound); saving encodes it.

/// <summary>Create Video: the images as the frames of a video at <c>fps</c>, with the sound if given. The frames stay as they are; a save node encodes them.</summary>
[Node("CreateVideo", Category = "video", DisplayName = "Create Video", Description = "Create a video from images.", OutputNames = ["VIDEO"])]
public sealed class CreateVideo
{
    public VideoClip Execute(
        [Input(Tooltip = "The images to create a video from.")] ImageBatch images,
        [Input(Min = 1.0, Max = 120.0, Step = 1.0)] double fps = 30.0,
        [Input(Optional = true, Tooltip = "The audio to add to the video.")] AudioClip? audio = null,
        [Input(Optional = true, Options = ["auto", "8", "10"], NumericOptions = true, Tooltip = "Auto uses 8-bit for sRGB and 10-bit for HDR. Explicit 8-bit and 10-bit choices are independent of colorspace.")] string bitDepth = "auto",
        [Input(Optional = true, Options = ["sRGB", "HDR", "HDR PQ"], Tooltip = "Colorspace of the input images. HDR selects BT.2020/HLG and HDR PQ selects BT.2020/PQ.")] string colorSpace = "sRGB",
        [Input(Optional = true, Options = ["none", "auto", "h264", "av1"], Tooltip = "Optionally encode the video immediately. None keeps the images in tensor form; Auto uses H.264.")] string codec = "none")
        => new(images, fps, audio, bitDepth, colorSpace);
}

/// <summary>Get Video Components: a video's frames, sound (none when it has none), frame rate, bit depth and colour space.</summary>
[Node("GetVideoComponents", Category = "video", DisplayName = "Get Video Components", Description = "Extracts video frames, audio, frame rate, bit depth, and color space.",
    OutputNames = ["images", "audio", "fps", "bit_depth", "color_space"], OutputTypes = [null, null, null, "COMBO", "COMBO"])]
public sealed class GetVideoComponents
{
    public (ImageBatch Images, AudioClip? Audio, double Fps, string BitDepth, string ColorSpace) Execute(
        [Input(Tooltip = "The video to extract components from.")] VideoClip video)
        => (video.Frames, video.Audio, video.Fps, video.BitDepth, video.ColorSpace);
}

/// <summary>
/// Trim Video (ComfyUI's "Video Slice"): the frames from <c>start_time</c> on for <c>duration</c> seconds (0: to the end), with the matching stretch of
/// sound. Measured with ComfyUI (2026-11-13; 30 cases on a 12-frame, 4 fps clip): frame i (at i / fps) is kept when start &lt;= i / fps &lt; start +
/// duration; a negative start counts back from the end (and stops at 0); start + duration (the start as given) above the clip's length fails with
/// ComfyUI's message; a start at or after the last frame gives a video with no frames. strict_duration changed nothing in any measured case.
/// ComfyUI re-encodes the trimmed clip (its frames and sound come back lossy and the sound shifted by the codec's priming); SpawnGen cuts the frames
/// and samples as they are.
/// </summary>
[Node("Video Slice", Category = "video", DisplayName = "Trim Video", OutputNames = ["VIDEO"])]
public sealed class VideoSlice
{
    public VideoClip Execute(VideoClip video,
        [Input(Min = -100000.0, Max = 100000.0, Step = 0.001, Tooltip = "Start time in seconds")] double startTime = 0.0,
        [Input(Min = 0.0, Step = 0.001, Tooltip = "Duration in seconds, or 0 for unlimited duration")] double duration = 0.0,
        [Input(Tooltip = "If True, when the specified duration is not possible, an error will be raised.")] bool strictDuration = false)
    {
        var frames = video.Frames;
        double length = frames.Batch / video.Fps;
        if (startTime + duration > length)
            throw new InvalidOperationException($"Failed to slice video:\nSource duration: {Py(length)}\nStart time: {Py(startTime)}\nTarget duration: {Py(duration)}\n");
        double start = startTime < 0 ? Math.Max(0, length + startTime) : startTime;
        double end = duration > 0 ? start + duration : double.PositiveInfinity;
        int first = 0;
        while (first < frames.Batch && first / video.Fps < start) first++;
        int last = first;
        while (last < frames.Batch && last / video.Fps < end) last++;
        int per = frames.Height * frames.Width * frames.Channels;
        var kept = new ImageBatch(last - first, frames.Height, frames.Width, frames.Channels, frames.Data.AsSpan(first * per, (last - first) * per).ToArray());
        AudioClip? audio = null;
        if (video.Audio is { } sound)
        {
            int from = (int)Math.Min(sound.Length, Math.Round(start * sound.SampleRate, MidpointRounding.ToEven));
            int to = double.IsPositiveInfinity(end) ? sound.Length : (int)Math.Min(sound.Length, Math.Round(end * sound.SampleRate, MidpointRounding.ToEven));
            audio = new AudioClip([.. sound.Channels.Select(c => c.AsSpan(from, Math.Max(0, to - from)).ToArray())], sound.SampleRate);
        }
        return new VideoClip(kept, video.Fps, audio, video.BitDepth, video.ColorSpace);
    }

    /// <summary>A number as Python prints a float (3.0, 2.5, 1.01).</summary>
    static string Py(double v) =>
        v == Math.Floor(v) && Math.Abs(v) < 1e16 ? v.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture) : v.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
}
