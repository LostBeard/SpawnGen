using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Images;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// Load Image on the GPU (AGENTS.md rule 4): the stored file goes to the device as the store keeps it (from JS in the browser,
/// <see cref="StoredBytes.Upload"/>), SpawnDev.Codecs decodes it there into RGBA8 (RGBA16 for a 16-bit file), and one kernel turns that into
/// IMAGE and MASK, upright by the EXIF orientation. Checked against the spec-correct pictures of the decode corpus (LoadImageCorpusTests and
/// CodecsDecodeCorpusTests: PNG exact, JPEG equal to libjpeg). Animated files are left to the host path (null): Load Image gives every frame.
/// </summary>
public sealed class GpuPictureDecoder(IServiceProvider services) : IPictureDecoder
{
    sealed class State(Accelerator accelerator)
    {
        public readonly ImageDecoderGpu Decoder = new(accelerator);
        public readonly SemaphoreSlim Gate = new(1, 1); // the decoder's staging buffers are not shared between calls
        public readonly Action<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int, int> FromRgba =
            accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, ArrayView1D<float, Stride1D.Dense>, int, int, int, int, int>(FromRgbaKernel);
    }

    static readonly ConditionalWeakTable<Accelerator, State> States = new();

    public async Task<(ImageBatch Image, MaskBatch Mask)?> LoadAsync(StoredBytes file)
    {
        if (file.Length < 12 || file.Length > int.MaxValue) return null;
        var head = file.Read(0, 12);
        var format = ImageDecoderGpu.Detect(head);
        if (format is not (ImageFormat.Png or ImageFormat.Jpeg or ImageFormat.WebP or ImageFormat.Gif or ImageFormat.Bmp)) return null;
        // A PNG's colour type (a palette picture always gets a full-size mask, as ComfyUI's Load Image) and eXIf orientation, from its chunk headers.
        var layout = format == ImageFormat.Png ? await PngReader.ReadLayoutAsync(file.Length, (o, n) => Task.FromResult(file.Read(o, n))) : null;
        if (format == ImageFormat.Png && layout is null) return null;

        var accelerator = (await GpuHelpers.RuntimeAsync(services)).Accelerator;
        var state = States.GetValue(accelerator, a => new State(a));
        int length = (int)file.Length;
        await state.Gate.WaitAsync();
        try
        {
            using var onDevice = file.Upload(accelerator);
            ImageInfo info;
            try { info = await state.Decoder.ReadInfoAsync(onDevice.View, length); }
            catch (Exception ex) when (ex is NotSupportedException or InvalidDataException) { return null; } // the host path says what is wrong
            if (info.FrameCount > 1) return null;
            var pixelFormat = info.BitDepth == 16 ? PixelFormat.Rgba16 : PixelFormat.Rgba8;
            using var rgba = accelerator.Allocate1D<byte>(info.GetPixelBufferSize(pixelFormat));
            try { info = await state.Decoder.DecodeAsync(onDevice.View, length, rgba.View, pixelFormat); }
            catch (Exception ex) when (ex is NotSupportedException or InvalidDataException) { return null; }

            int orientation = layout?.Orientation ?? info.Orientation;
            bool transparent = info.HasAlpha || layout?.ColorType == 3;
            int w = info.Width, h = info.Height, n = w * h;
            bool swap = orientation is >= 5 and <= 8;
            int uw = swap ? h : w, uh = swap ? w : h;
            var image = accelerator.Allocate1D<float>((long)n * 3);
            var mask = transparent ? accelerator.Allocate1D<float>(n) : accelerator.Allocate1D<float>(64 * 64);
            if (!transparent) mask.MemSetToZero();
            state.FromRgba(n, rgba.View.BaseView.Cast<int>(), image.View, mask.View, w, h, orientation, pixelFormat == PixelFormat.Rgba16 ? 1 : 0, transparent ? 1 : 0);
            return (ImageBatch.FromDevice(1, uh, uw, 3, image), transparent ? MaskBatch.FromDevice(1, uh, uw, mask) : MaskBatch.FromDevice(1, 64, 64, mask));
        }
        finally { state.Gate.Release(); }
    }

    /// <summary>
    /// One source pixel <paramref name="i"/> of the decoded RGBA (packed little-endian: one int per pixel for 8-bit, two for 16-bit) to its place
    /// in the upright picture (EXIF orientation 1..8, as <see cref="ExifOrientation"/>): IMAGE R, G, B as s / 255 (or s / 65535) and, when the
    /// picture can be transparent, MASK = 1 - alpha.
    /// </summary>
    static void FromRgbaKernel(Index1D i, ArrayView1D<int, Stride1D.Dense> rgba, ArrayView1D<float, Stride1D.Dense> image, ArrayView1D<float, Stride1D.Dense> mask,
        int w, int h, int orientation, int wide, int writeMask)
    {
        int x = i % w, y = i / w;
        int tx = x, ty = y, uw = w;
        if (orientation == 2) tx = w - 1 - x;
        else if (orientation == 3) { tx = w - 1 - x; ty = h - 1 - y; }
        else if (orientation == 4) ty = h - 1 - y;
        else if (orientation == 5) { tx = y; ty = x; uw = h; }
        else if (orientation == 6) { tx = h - 1 - y; ty = x; uw = h; }
        else if (orientation == 7) { tx = h - 1 - y; ty = w - 1 - x; uw = h; }
        else if (orientation == 8) { tx = y; ty = w - 1 - x; uw = h; }
        int t = ty * uw + tx;
        float r, g, b, a;
        if (wide == 0)
        {
            int v = rgba[i];
            r = (v & 0xFF) / 255f; g = ((v >> 8) & 0xFF) / 255f; b = ((v >> 16) & 0xFF) / 255f; a = ((v >> 24) & 0xFF) / 255f;
        }
        else
        {
            int lo = rgba[2 * i], hi = rgba[2 * i + 1];
            r = (lo & 0xFFFF) / 65535f; g = ((lo >> 16) & 0xFFFF) / 65535f; b = (hi & 0xFFFF) / 65535f; a = ((hi >> 16) & 0xFFFF) / 65535f;
        }
        image[3 * t] = r; image[3 * t + 1] = g; image[3 * t + 2] = b;
        if (writeMask != 0) mask[t] = 1f - a;
    }
}
