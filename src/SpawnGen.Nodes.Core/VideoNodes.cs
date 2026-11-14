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
