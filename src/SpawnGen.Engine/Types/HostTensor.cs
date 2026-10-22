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

/// <summary>A batch of latents, NCHW (same layout as ComfyUI's LATENT "samples").</summary>
[Socket("LATENT")]
public sealed class LatentBatch : HostTensor
{
    public LatentBatch(int batch, int channels, int height, int width, float[]? data = null)
        : base([batch, channels, height, width], data) { }

    public int Batch => Shape[0];
    public int Channels => Shape[1];
    public int Height => Shape[2];
    public int Width => Shape[3];

    /// <summary>Optional inpainting mask at latent size (batch, h, w): 1 = this area is regenerated, 0 = it is kept. Set by Set Latent Noise Mask.</summary>
    public MaskBatch? NoiseMask { get; init; }
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
