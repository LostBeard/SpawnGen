using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Types;

/// <summary>
/// A video moving between nodes (ComfyUI's VIDEO): its frames as an image batch, the frame rate, the sound if any, and the bit depth and colour
/// space ComfyUI's Create Video records (auto / 8 / 10; sRGB, HDR, HDR PQ). Kept as frames: a save node encodes it.
/// </summary>
[Socket("VIDEO")]
public sealed class VideoClip
{
    public VideoClip(ImageBatch frames, double fps, AudioClip? audio = null, string bitDepth = "auto", string colorSpace = "sRGB")
    {
        if (fps <= 0 || double.IsNaN(fps)) throw new ArgumentOutOfRangeException(nameof(fps), "A video needs a frame rate above 0.");
        Frames = frames;
        Fps = fps;
        Audio = audio;
        BitDepth = bitDepth;
        ColorSpace = colorSpace;
    }

    public ImageBatch Frames { get; }
    public double Fps { get; }
    public AudioClip? Audio { get; }
    public string BitDepth { get; }
    public string ColorSpace { get; }
    public double Seconds => Frames.Batch / Fps;
}
