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
/// Empty SD3 latent (also the empty latent of FLUX.1, Z-Image, Qwen-Image and the other 16-channel picture models): zeros
/// [batch, 16, height / 8, width / 8]. Measured with ComfyUI (2026-11-13): 1024x1024 gives 128 x 128 cells, 100x36 gives 12 x 4 (rounded down as
/// Empty Latent Image), all values 0. Sampling turns an all-zero latent into the model's channel count (see SamplingCore.ToModelChannels), so an
/// SD 1.x model samples from this latent as from Empty Latent Image (measured: the same result to the last bit).
/// </summary>
[Node("EmptySD3LatentImage", Category = "model/latent/stable diffusion", DisplayName = "EmptySD3LatentImage")]
public sealed class EmptySD3LatentImage
{
    public LatentBatch Execute(
        [Input(Min = 16, Max = 16384, Step = 16)] int width = 1024,
        [Input(Min = 16, Max = 16384, Step = 16)] int height = 1024,
        [Input(Min = 1, Max = 4096)] int batchSize = 1) =>
        new(batchSize, 16, height / 8, width / 8);
}

/// <summary>
/// Empty Flux 2 Latent: zeros [batch, 128, height / 16, width / 16]. Measured with ComfyUI (2026-11-13): 1024x1024 gives 64 x 64 cells, 100x36
/// 6 x 2 and 17x33 1 x 2 (rounded down), all values 0.
/// </summary>
[Node("EmptyFlux2LatentImage", Category = "model/latent/flux", DisplayName = "Empty Flux 2 Latent")]
public sealed class EmptyFlux2LatentImage
{
    public LatentBatch Execute(
        [Input(Min = 16, Max = 16384, Step = 16)] int width = 1024,
        [Input(Min = 16, Max = 16384, Step = 16)] int height = 1024,
        [Input(Min = 1, Max = 4096)] int batchSize = 1) =>
        new(batchSize, 128, height / 16, width / 16);
}

/// <summary>
/// Trim Video Latent: drops the first <c>trim_amount</c> frames; trimming all of them or more leaves no frames. The noise mask and batch indexes are
/// passed on as they are. Measured with ComfyUI (2026-11-13): a picture latent loses its first rows instead (its third dimension is the height).
/// </summary>
[VideoLatents]
[Node("TrimVideoLatent", Category = "model/latent", DisplayName = "Trim Video Latent")]
public sealed class TrimVideoLatent
{
    public LatentBatch Execute(LatentBatch samples, [Input(Min = 0, Max = 99999)] int trimAmount = 0)
    {
        int b = samples.Batch, c = samples.Channels, w = samples.Width;
        int depth = samples.IsVideo ? samples.Frames : samples.Height, rows = samples.IsVideo ? samples.Height * w : w;
        int keep = Math.Max(0, depth - trimAmount), cut = depth - keep;
        var data = new float[b * c * keep * rows];
        for (int bc = 0; bc < b * c; bc++)
            Array.Copy(samples.Data, (bc * depth + cut) * rows, data, bc * keep * rows, keep * rows);
        var trimmed = samples.IsVideo ? new LatentBatch(b, c, keep, samples.Height, w, data) : new LatentBatch(b, c, keep, w, data);
        return trimmed.With(samples);
    }
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
