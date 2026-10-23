using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

[Node(Category = "model/latent", DisplayName = "Empty Latent Image", Description = "Create a new batch of empty latent images to be denoised via sampling.")]
public sealed class EmptyLatentImage
{
    /// <summary>Zero latent at 1/8 of the image size, 4 channels (SD 1.x / 2.x / Turbo layout).</summary>
    public LatentBatch Execute(
        [Input(Min = 16, Max = 16384, Step = 8, Tooltip = "The width of the latent images in pixels.")] int width = 1024,
        [Input(Min = 16, Max = 16384, Step = 8, Tooltip = "The height of the latent images in pixels.")] int height = 1024,
        [Input(Min = 1, Max = 4096, Tooltip = "The number of latent images in the batch.")] int batchSize = 1) =>
        new(batchSize, 4, height / 8, width / 8);
}
