using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// IMAGE conversions on the device (AGENTS.md rule 4: pictures are never converted in a .NET pixel loop). RGBA8 is packed one int per pixel,
/// little-endian (R | G &lt;&lt; 8 | B &lt;&lt; 16 | A &lt;&lt; 24, the byte order the encoders read), so no kernel stores single bytes (WebGPU has no byte stores).
/// </summary>
public static class ImageKernels
{
    sealed class Loaded(Accelerator accelerator)
    {
        public readonly Action<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int> ToRgba8 =
            accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, int>(ToRgba8Kernel);
    }

    static readonly ConditionalWeakTable<Accelerator, Loaded> Kernels = new();

    /// <summary>
    /// ComfyUI's IMAGE -> Pillow conversion, <c>np.clip(255 * x, 0, 255).astype(uint8)</c> (truncated; NaN gives 0), opaque without alpha:
    /// <paramref name="image"/> NHWC with <paramref name="channels"/> 3 or 4, into <paramref name="rgba"/> (one packed int per pixel, all frames).
    /// </summary>
    public static void ToRgba8(Accelerator accelerator, ArrayView1D<float, Stride1D.Dense> image, int channels, ArrayView1D<int, Stride1D.Dense> rgba)
    {
        if (channels is not (3 or 4)) throw new ArgumentOutOfRangeException(nameof(channels), "RGBA8 needs 3 or 4 channels.");
        if (rgba.Length * channels != image.Length) throw new ArgumentException($"{image.Length} values at {channels} channels are not {rgba.Length} pixels.");
        if (rgba.Length > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(rgba), "Too many pixels for one launch.");
        Kernels.GetValue(accelerator, a => new Loaded(a)).ToRgba8((int)rgba.Length, image, rgba, channels);
    }

    static int Byte(float v)
    {
        float x = 255f * v;
        x = x > 0f ? x : 0f; // NaN and negatives -> 0
        x = x < 255f ? x : 255f;
        return (int)x;        // truncated, as astype(uint8) after the clip
    }

    static void ToRgba8Kernel(Index1D p, ArrayView1D<float, Stride1D.Dense> image, ArrayView1D<int, Stride1D.Dense> rgba, int channels)
    {
        long o = (long)p.X * channels;
        int r = Byte(image[o]), g = Byte(image[o + 1]), b = Byte(image[o + 2]);
        int a = channels == 4 ? Byte(image[o + 3]) : 255;
        rgba[p] = r | (g << 8) | (b << 16) | (a << 24);
    }
}
