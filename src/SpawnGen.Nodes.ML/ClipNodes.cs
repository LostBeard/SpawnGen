using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries and LoraNodeTests (CLIP Set Last Layer keeps the LoRAs). SDXL's
// size conditioning is the published SDXL micro-conditioning (original size, crop, target size).

[Node("CLIPSetLastLayer", Category = "model/conditioning", DisplayName = "CLIP Set Last Layer")]
public sealed class CLIPSetLastLayer
{
    /// <summary>A copy of the CLIP that stops at that layer (-1 the last, -2 the one before, ...); its weight steps are kept.</summary>
    public ClipHandle Execute(ClipHandle clip, [Input(Min = -24, Max = -1, Step = 1)] int stopAtClipLayer = -1) =>
        new(clip.Encoder) { Layer = stopAtClipLayer, Weights = clip.Weights };
}

[Node("CLIPTextEncodeSDXL", Category = "model/conditioning/stable diffusion", DisplayName = "CLIP Text Encode (SDXL)")]
public sealed class CLIPTextEncodeSDXL
{
    /// <summary>
    /// SDXL conditioning: <paramref name="textL"/> for CLIP-L and <paramref name="textG"/> for OpenCLIP-bigG, with the pooled output and the size
    /// numbers SDXL's UNet takes (the picture's size, the crop's corner, the target size).
    /// </summary>
    public async Task<Conditioning> ExecuteAsync(ClipHandle clip,
        [Input(Min = 0, Max = 16384)] int width = 1024, [Input(Min = 0, Max = 16384)] int height = 1024,
        [Input(Min = 0, Max = 16384)] int cropW = 0, [Input(Min = 0, Max = 16384)] int cropH = 0,
        [Input(Min = 0, Max = 16384)] int targetWidth = 1024, [Input(Min = 0, Max = 16384)] int targetHeight = 1024,
        [Input(Multiline = true)] string textG = "", [Input(Multiline = true)] string textL = "", NodeContext? context = null)
    {
        var encoding = await CLIPTextEncode.EncodeAsync(clip, textL, textG, context?.CancellationToken ?? CancellationToken.None);
        return new Conditioning([new CondEntry(encoding.Context, encoding.Shape)
        {
            Pooled = encoding.Pooled,
            Size = new SdxlSize(width, height, cropW, cropH, targetWidth, targetHeight),
        }]);
    }
}
