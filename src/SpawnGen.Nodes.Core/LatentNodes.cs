using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// A batch of zero latents with 4 channels at an eighth of the pixel size. Measured with ComfyUI: width 100 and height 36 give 12 x 4 cells
/// (the pixel size divided by 8, rounded down); the latent has no noise mask and no batch index.
/// </summary>
[Node(Category = "model/latent", DisplayName = "Empty Latent Image", Description = "Create a new batch of empty latent images to be denoised via sampling.")]
public sealed class EmptyLatentImage
{
    public LatentBatch Execute(
        [Input(Min = 16, Max = 16384, Step = 8, Tooltip = "The width of the latent images in pixels.")] int width = 1024,
        [Input(Min = 16, Max = 16384, Step = 8, Tooltip = "The height of the latent images in pixels.")] int height = 1024,
        [Input(Min = 1, Max = 4096, Tooltip = "The number of latent images in the batch.")] int batchSize = 1) =>
        new(batchSize, 4, height / 8, width / 8);
}

/// <summary>
/// Empty Hunyuan Video 1.0 Latent (also Wan 2.1's empty latent): zeros [batch, 16, (length - 1) / 4 + 1, height / 8, width / 8], as measured.
/// </summary>
[Node("EmptyHunyuanLatentVideo", Category = "model/latent/hunyuan video", DisplayName = "Empty Hunyuan Video 1.0 Latent")]
public sealed class EmptyHunyuanLatentVideo
{
    public LatentBatch Execute(
        [Input(Min = 16, Max = 16384, Step = 16)] int width = 848,
        [Input(Min = 16, Max = 16384, Step = 16)] int height = 480,
        [Input(Min = 1, Max = 16384, Step = 4)] int length = 25,
        [Input(Min = 1, Max = 4096)] int batchSize = 1) =>
        new(batchSize, 16, (length - 1) / 4 + 1, height / 8, width / 8);
}
