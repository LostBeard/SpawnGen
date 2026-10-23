using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

[Node(Category = "image/transform", DisplayName = "Flip Image")]
public sealed class ImageFlip
{
    public static string[] Axes() => ["x-axis: vertically", "y-axis: horizontally"];

    public ImageBatch Execute(ImageBatch image, [Input(OptionsProvider = nameof(Axes))] string flipMethod = "x-axis: vertically")
    {
        bool vertical = flipMethod.StartsWith("x-axis", StringComparison.Ordinal);
        if (!vertical && !flipMethod.StartsWith("y-axis", StringComparison.Ordinal)) throw new ArgumentException($"Unknown flip method '{flipMethod}'.", nameof(flipMethod));
        int w = image.Width, h = image.Height, c = image.Channels;
        return Pixels.Map(image, f =>
        {
            var o = new float[f.Length];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    Array.Copy(f, (y * w + x) * c, o, (vertical ? (h - 1 - y) * w + x : y * w + (w - 1 - x)) * c, c);
            return o;
        });
    }
}

[Node(Category = "image/transform", DisplayName = "Rotate Image")]
public sealed class ImageRotate
{
    public static string[] Rotations() => ["none", "90 degrees", "180 degrees", "270 degrees"];

    /// <summary>Rotates clockwise by a multiple of 90 degrees (ComfyUI's names); 90 and 270 swap width and height.</summary>
    public ImageBatch Execute(ImageBatch image, [Input(OptionsProvider = nameof(Rotations))] string rotation = "none")
    {
        int quarter = rotation switch { "none" => 0, "90 degrees" => 1, "180 degrees" => 2, "270 degrees" => 3, _ => throw new ArgumentException($"Unknown rotation '{rotation}'.", nameof(rotation)) };
        if (quarter == 0) return image;
        int w = image.Width, h = image.Height, c = image.Channels;
        int nw = quarter == 2 ? w : h, nh = quarter == 2 ? h : w;
        return Pixels.Map(image, f =>
        {
            var o = new float[f.Length];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    // Clockwise: the top row becomes the right column.
                    var (nx, ny) = quarter switch { 1 => (h - 1 - y, x), 2 => (w - 1 - x, h - 1 - y), _ => (y, w - 1 - x) };
                    Array.Copy(f, (y * w + x) * c, o, (ny * nw + nx) * c, c);
                }
            return o;
        }, nh, nw);
    }
}

[Node(Category = "image/filters", DisplayName = "Quantize Image")]
public sealed class ImageQuantize
{
    public static string[] Dithers() => ["none", "floyd-steinberg", "bayer-2", "bayer-4", "bayer-8", "bayer-16"];

    /// <summary>
    /// ComfyUI's Quantize, byte for byte: each picture's RGB goes to 8 bits by <c>(rgb * 255).to(torch.uint8)</c> (truncation; outside 0..255 it wraps like
    /// torch's cast), Pillow builds a palette of at most <paramref name="colors"/> entries by median cut (<c>quantize(colors)</c>), and every pixel takes its
    /// nearest entry, with Pillow's Floyd-Steinberg or after adding ComfyUI's ordered (bayer) threshold map. The result is <c>/ 255</c>; an alpha channel
    /// is passed through, and other extra channels are dropped like ComfyUI does.
    /// </summary>
    public ImageBatch Execute(ImageBatch image, [Input(Min = 1, Max = 256)] int colors = 256, [Input(OptionsProvider = nameof(Dithers))] string dither = "none")
    {
        int order = dither switch
        {
            "none" or "floyd-steinberg" => 0,
            "bayer-2" => 2, "bayer-4" => 4, "bayer-8" => 8, "bayer-16" => 16,
            _ => throw new ArgumentException($"Unknown dither '{dither}'.", nameof(dither)),
        };
        if (colors is < 1 or > 256) throw new ArgumentOutOfRangeException(nameof(colors), "bad number of colors");
        int w = image.Width, h = image.Height, c = image.Channels, n = w * h;
        if (c < 3) throw new ArgumentException("Quantize needs a colour picture.");
        int oc = c == 4 ? 4 : 3;
        var result = new ImageBatch(image.Batch, h, w, oc);
        for (int b = 0; b < image.Batch; b++)
        {
            int src = b * n * c, dst = b * n * oc;
            var rgb = new byte[n * 3];
            for (int i = 0; i < n; i++)
                for (int ch = 0; ch < 3; ch++) rgb[i * 3 + ch] = TorchToUInt8(image.Data[src + i * c + ch] * 255f);

            var palette = PilQuantize.MedianCutPalette(rgb, colors);
            var index = order == 0
                ? PilQuantize.ToPalette(rgb, w, h, palette, dither == "floyd-steinberg")
                : PilQuantize.ToPalette(Bayer(rgb, w, h, order, palette.Length / 3), w, h, palette, dither: false);

            for (int i = 0; i < n; i++)
            {
                for (int ch = 0; ch < 3; ch++) result.Data[dst + i * oc + ch] = palette[index[i] * 3 + ch] / 255f;
                if (oc == 4) result.Data[dst + i * 4 + 3] = image.Data[src + i * 4 + 3];
            }
        }
        return result;
    }

    /// <summary>torch's float32 to uint8 cast on the CPU: truncation toward zero, then the low byte (so -1 is 255 and 256 is 0); NaN, infinities and anything past int32 give 0.</summary>
    public static byte TorchToUInt8(float x) => MathF.Abs(x) < 2147483648f ? unchecked((byte)(int)x) : (byte)0;

    /// <summary>
    /// Quantize.bayer before its palette step: <c>spread * normalized_bayer_matrix(log2 order) + 0.5</c> with spread = 512 / palette size (numpy float64),
    /// tiled over the picture from the top left and added to the 8-bit values (float32 += float64, so the sum is rounded to float32), clamped to 0..255
    /// and truncated back to 8 bits.
    /// </summary>
    static byte[] Bayer(byte[] rgb, int w, int h, int order, int paletteSize)
    {
        double spread = 2 * 256 / (double)paletteSize;
        var m = NormalizedBayer((int)Math.Log2(order));
        var o = new byte[rgb.Length];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double t = spread * m[y % order, x % order] + 0.5;
                for (int ch = 0; ch < 3; ch++)
                {
                    int p = (y * w + x) * 3 + ch;
                    float v = (float)(rgb[p] + t);
                    o[p] = (byte)Math.Clamp(v, 0f, 255f);
                }
            }
        return o;
    }

    /// <summary>ComfyUI's normalized_bayer_matrix: M(0) = 0; M(n) = [[m - 1.5, m + 0.5], [m + 1.5, m - 0.5]] / 4^n with m = 4^n M(n - 1) (all exact in binary).</summary>
    static double[,] NormalizedBayer(int n)
    {
        if (n == 0) return new double[1, 1];
        var prev = NormalizedBayer(n - 1);
        int half = 1 << (n - 1), size = half * 2;
        double q = Math.Pow(4, n);
        var r = new double[size, size];
        for (int y = 0; y < half; y++)
            for (int x = 0; x < half; x++)
            {
                double v = q * prev[y, x];
                r[y, x] = (v - 1.5) / q;
                r[y, x + half] = (v + 0.5) / q;
                r[y + half, x] = (v + 1.5) / q;
                r[y + half, x + half] = (v - 0.5) / q;
            }
        return r;
    }
}


[Node(Category = "image/filters", DisplayName = "Apply Morphology")]
public sealed class Morphology
{
    public static string[] Operations() => ["erode", "dilate", "open", "close", "gradient", "bottom_hat", "top_hat"];

    /// <summary>
    /// Grey-scale morphology with a flat square kernel, per channel (ComfyUI's Morphology node, kornia.morphology): erode takes the minimum, dilate the
    /// maximum, and the rest combine them. Like kornia, the window of pixel x runs from x - k/2 to x - k/2 + k - 1 (one pixel more before than after
    /// for an even size), and pixels outside the picture are ignored (kornia's geodesic border).
    /// </summary>
    public ImageBatch Execute(ImageBatch image, [Input(OptionsProvider = nameof(Operations))] string operation = "erode", [Input(Min = 3, Max = 999)] int kernelSize = 3)
    {
        if (!Operations().Contains(operation)) throw new ArgumentException($"Unknown operation '{operation}'.", nameof(operation));
        int before = kernelSize / 2, after = kernelSize - before - 1;
        int w = image.Width, h = image.Height, c = image.Channels;
        return Pixels.Map(image, f =>
        {
            float[] Erode(float[] s) => Extreme(s, w, h, c, before, after, min: true);
            float[] Dilate(float[] s) => Extreme(s, w, h, c, before, after, min: false);
            float[] Sub(float[] a, float[] b) { var r = new float[a.Length]; for (int i = 0; i < r.Length; i++) r[i] = a[i] - b[i]; return r; }
            return operation switch
            {
                "erode" => Erode(f),
                "dilate" => Dilate(f),
                "open" => Dilate(Erode(f)),
                "close" => Erode(Dilate(f)),
                "gradient" => Sub(Dilate(f), Erode(f)),
                "top_hat" => Sub(f, Dilate(Erode(f))),
                _ => Sub(Erode(Dilate(f)), f), // bottom_hat
            };
        });
    }

    /// <summary>Minimum or maximum over the square from -before to +after around each pixel, as two 1D passes; pixels outside the picture are ignored.</summary>
    static float[] Extreme(float[] src, int w, int h, int c, int before, int after, bool min)
    {
        float Pick(float a, float b) => min ? Math.Min(a, b) : Math.Max(a, b);
        var tmp = new float[src.Length];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                for (int ch = 0; ch < c; ch++)
                {
                    float v = src[(y * w + x) * c + ch];
                    for (int k = Math.Max(0, x - before); k <= Math.Min(w - 1, x + after); k++) v = Pick(v, src[(y * w + k) * c + ch]);
                    tmp[(y * w + x) * c + ch] = v;
                }
        var dst = new float[src.Length];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                for (int ch = 0; ch < c; ch++)
                {
                    float v = tmp[(y * w + x) * c + ch];
                    for (int k = Math.Max(0, y - before); k <= Math.Min(h - 1, y + after); k++) v = Pick(v, tmp[(k * w + x) * c + ch]);
                    dst[(y * w + x) * c + ch] = v;
                }
        return dst;
    }
}

[Node(Category = "image/mask", DisplayName = "Convert Image Color to Mask")]
public sealed class ImageColorToMask
{
    /// <summary>1 where a pixel is exactly the colour (0xRRGGBB as a number, like ComfyUI), else 0. Channels are clamped to 0..1, then rounded to 0..255 (half to even).</summary>
    public MaskBatch Execute(ImageBatch image, [Input(Min = 0, Max = 0xFFFFFF)] int color = 0)
    {
        var mask = new MaskBatch(image.Batch, image.Height, image.Width);
        int c = image.Channels;
        for (int i = 0; i < mask.Data.Length; i++)
        {
            int Q(int k) => (int)MathF.Round(Math.Clamp(image.Data[i * c + Math.Min(k, c - 1)], 0f, 1f) * 255f);
            mask.Data[i] = (Q(0) << 16) + (Q(1) << 8) + Q(2) == color ? 1f : 0f;
        }
        return mask;
    }
}
