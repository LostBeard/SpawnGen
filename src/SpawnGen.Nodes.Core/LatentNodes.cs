using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

[Node(Category = "latent")]
public sealed class EmptyLatentImage
{
    /// <summary>Zero latent at 1/8 of the image size, 4 channels (SD 1.x / 2.x / Turbo layout).</summary>
    public LatentBatch Execute(
        [Input(Min = 16, Max = 8192, Step = 8)] int width = 512,
        [Input(Min = 16, Max = 8192, Step = 8)] int height = 512,
        [Input(Min = 1, Max = 64)] int batchSize = 1) =>
        new(batchSize, 4, height / 8, width / 8);
}
