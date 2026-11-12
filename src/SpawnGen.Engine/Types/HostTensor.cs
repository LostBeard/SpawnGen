using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Types;

/// <summary>Dense float32 tensor on the host.</summary>
public class HostTensor
{
    public HostTensor(int[] shape, float[]? data = null)
    {
        Shape = shape;
        var length = 1L;
        foreach (var d in shape) length *= d;
        if (length > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(shape), "Tensor is too large.");
        Data = data ?? new float[length];
        if (Data.Length != length)
            throw new ArgumentException($"Data length {Data.Length} does not match shape [{string.Join(",", shape)}].", nameof(data));
    }

    public int[] Shape { get; }
    public float[] Data { get; }

    /// <summary>The kind and shape ("ImageBatch [1, 512, 512, 3]"): how Preview as Text shows a tensor, also inside a list.</summary>
    public override string ToString() => $"{GetType().Name} [{string.Join(", ", Shape)}]";
}

/// <summary>A batch of images, NHWC, RGB(A), values in 0..1 (same layout as ComfyUI's IMAGE).</summary>
[Socket("IMAGE")]
public sealed class ImageBatch : HostTensor
{
    public ImageBatch(int batch, int height, int width, int channels = 3, float[]? data = null)
        : base([batch, height, width, channels], data) { }

    public int Batch => Shape[0];
    public int Height => Shape[1];
    public int Width => Shape[2];
    public int Channels => Shape[3];
}

/// <summary>
/// A batch of latents, NCHW, or NCFHW for a video model's latents (frames between channels and height), the same layouts as ComfyUI's LATENT "samples".
/// Only nodes marked <see cref="VideoLatentsAttribute"/> are given a video latent; the others refuse it before they run.
/// </summary>
[Socket("LATENT")]
public sealed class LatentBatch : HostTensor
{
    public LatentBatch(int batch, int channels, int height, int width, float[]? data = null)
        : base([batch, channels, height, width], data) { }

    /// <summary>A video latent [batch, channels, frames, height, width].</summary>
    public LatentBatch(int batch, int channels, int frames, int height, int width, float[]? data = null)
        : base([batch, channels, frames, height, width], data) { }

    public int Batch => Shape[0];
    public int Channels => Shape[1];
    public int Height => Shape[^2];
    public int Width => Shape[^1];

    /// <summary>Whether this is a video latent (5-D, with a frames dimension).</summary>
    public bool IsVideo => Shape.Length == 5;

    /// <summary>The latent frames of a video latent; 1 for a picture latent.</summary>
    public int Frames => IsVideo ? Shape[2] : 1;

    /// <summary>One batch item's values: channels x frames x height x width.</summary>
    public int ItemSize => Channels * Frames * Height * Width;

    /// <summary>An empty latent shaped like <paramref name="template"/> (picture or video) with another batch size and these values.</summary>
    public static LatentBatch Shaped(LatentBatch template, int batch, float[]? data = null) => template.IsVideo
        ? new LatentBatch(batch, template.Channels, template.Frames, template.Height, template.Width, data)
        : new LatentBatch(batch, template.Channels, template.Height, template.Width, data);

    /// <summary>A copy of these values with <paramref name="source"/>'s noise mask and batch indexes.</summary>
    public LatentBatch With(LatentBatch source) =>
        IsVideo ? new LatentBatch(Batch, Channels, Frames, Height, Width, Data) { NoiseMask = source.NoiseMask, BatchIndex = source.BatchIndex }
                : new LatentBatch(Batch, Channels, Height, Width, Data) { NoiseMask = source.NoiseMask, BatchIndex = source.BatchIndex };

    /// <summary>
    /// Optional inpainting mask at latent size (batch, h, w): 1 = this area is regenerated, 0 = it is kept. Set by Set Latent Noise Mask. A video
    /// latent's has its frames stacked (batch, frames x h, w), as ComfyUI's [batch, 1, frames, h, w].
    /// </summary>
    public MaskBatch? NoiseMask { get; init; }

    /// <summary>
    /// ComfyUI's <c>batch_index</c>: which noise of the seed each picture starts from (null = picture i gets noise i). Get Latent From Batch keeps a picture's
    /// original noise this way, and Latent Batch Seed Behavior "fixed" gives the whole batch one noise.
    /// </summary>
    public int[]? BatchIndex { get; init; }
}

/// <summary>A batch of masks, NHW (batch, height, width), values 0..1 (same layout as ComfyUI's MASK). 1 = selected.</summary>
[Socket("MASK")]
public sealed class MaskBatch : HostTensor
{
    public MaskBatch(int batch, int height, int width, float[]? data = null) : base([batch, height, width], data) { }

    public int Batch => Shape[0];
    public int Height => Shape[1];
    public int Width => Shape[2];

    public static MaskBatch Filled(int batch, int height, int width, float value)
    {
        var m = new MaskBatch(batch, height, width);
        Array.Fill(m.Data, value);
        return m;
    }
}
