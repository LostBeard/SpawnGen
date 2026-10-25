using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.ML;

[Node(Category = "model/conditioning", DisplayName = "CLIP Set Last Layer")]
public sealed class CLIPSetLastLayer
{
    /// <summary>
    /// Which CLIP layer prompts are read from: -1 the last, -2 the one before it ("CLIP skip 2"), and so on. The CLIP that comes in is not changed, as in ComfyUI.
    /// </summary>
    public ClipHandle Execute(ClipHandle clip, [Input(Min = -24, Max = -1, Step = 1)] int stopAtClipLayer = -1) =>
        new(clip.Checkpoint) { Layer = stopAtClipLayer, Loras = clip.Loras };
}

[Node(Category = "model/conditioning/stable diffusion", DisplayName = "CLIP Text Encode (SDXL)")]
public sealed class CLIPTextEncodeSDXL
{
    /// <summary>
    /// SDXL's prompt with its size conditioning: <c>text_g</c> goes to the OpenCLIP bigG tower, <c>text_l</c> to CLIP-L; width and height are the size the model
    /// is told the picture had, crop where it was cut from, target the size wanted (ComfyUI's CLIPTextEncodeSDXL).
    /// </summary>
    public async Task<Conditioning> ExecuteAsync(ClipHandle clip,
        [Input(Min = 0, Max = 16384, Default = 1024)] int width, [Input(Min = 0, Max = 16384, Default = 1024)] int height,
        [Input(Min = 0, Max = 16384, Default = 0)] int cropW, [Input(Min = 0, Max = 16384, Default = 0)] int cropH,
        [Input(Min = 0, Max = 16384, Default = 1024)] int targetWidth, [Input(Min = 0, Max = 16384, Default = 1024)] int targetHeight,
        [Input(Multiline = true, Default = "")] string textG, [Input(Multiline = true, Default = "")] string textL, NodeContext context)
    {
        if (clip.Checkpoint.Family != SpawnDev.ILGPU.ML.StableDiffusion.SdFamily.SDXL)
            throw new ArgumentException($"CLIP Text Encode (SDXL) needs an SDXL checkpoint's CLIP; '{clip.Checkpoint.Name}' is {clip.Checkpoint.Family}.", nameof(clip));
        var e = await CLIPTextEncode.EncodeAsync(clip, textL, textG, context.CancellationToken);
        return new Conditioning([new CondEntry(e.Context, e.Shape)
        {
            Pooled = e.Pooled,
            Size = new SdxlSize(width, height, cropW, cropH, targetWidth, targetHeight),
        }]);
    }
}
