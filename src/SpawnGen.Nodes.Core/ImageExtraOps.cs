using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Flip, rotate, quantize, morphology and colour-to-mask. Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries, SpawnGen's tests
// and fixtures made by running ComfyUI, and black-box measurements; the palette work uses PilQuantize (Pillow's documented quantize, see
// THIRD-PARTY-NOTICES.md).

/// <summary>Mirrors every picture: "x-axis: vertically" turns it upside down (rows reversed), "y-axis: horizontally" mirrors left and right.</summary>
[Node(Category = "image/transform", DisplayName = "Flip Image")]
public sealed class ImageFlip
{
    public ImageBatch Execute(ImageBatch image, [Input(Options = ["x-axis: vertically", "y-axis: horizontally"])] string flipMethod)
    {
        bool rows = flipMethod switch
        {
            "x-axis: vertically" => true,
            "y-axis: horizontally" => false,
            _ => throw new ArgumentException($"Unknown flip method '{flipMethod}'.", nameof(flipMethod)),
        };
        int h = image.Height, w = image.Width, c = image.Channels;
        var result = new ImageBatch(image.Batch, h, w, c);
        for (int b = 0; b < image.Batch; b++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int sy = rows ? h - 1 - y : y, sx = rows ? x : w - 1 - x;
                    Array.Copy(image.Data, (((long)b * h + sy) * w + sx) * c, result.Data, (((long)b * h + y) * w + x) * c, c);
                }
        return result;
    }
}

/// <summary>Turns every picture clockwise by a quarter, half or three quarters; quarter turns swap width and height. "none" passes the input on.</summary>
[Node(Category = "image/transform", DisplayName = "Rotate Image")]
public sealed class ImageRotate
{
    public ImageBatch Execute(ImageBatch image, [Input(Options = ["none", "90 degrees", "180 degrees", "270 degrees"])] string rotation)
    {
        int turns = rotation switch
        {
            "none" => 0,
            "90 degrees" => 1,
            "180 degrees" => 2,
            "270 degrees" => 3,
            _ => throw new ArgumentException($"Unknown rotation '{rotation}'.", nameof(rotation)),
        };
        if (turns == 0) return image;
        int h = image.Height, w = image.Width, c = image.Channels;
        int oh = turns == 2 ? h : w, ow = turns == 2 ? w : h;
        var result = new ImageBatch(image.Batch, oh, ow, c);
        for (int b = 0; b < image.Batch; b++)
            for (int y = 0; y < oh; y++)
                for (int x = 0; x < ow; x++)
                {
                    // Where output pixel (x, y) comes from, turning clockwise.
                    var (sx, sy) = turns switch
                    {
                        1 => (y, h - 1 - x),
                        2 => (w - 1 - x, h - 1 - y),
                        _ => (w - 1 - y, x),
                    };
                    Array.Copy(image.Data, (((long)b * h + sy) * w + sx) * c, result.Data, (((long)b * oh + y) * ow + x) * c, c);
                }
        return result;
    }
}

/// <summary>
/// Reduces every picture to a palette of at most <c>colors</c> colours. The palette is Pillow's median cut of the picture's 8-bit RGB
/// (<see cref="PilQuantize.MedianCutPalette"/>); pixels then take the nearest palette entry, plainly, with Floyd-Steinberg error diffusion, or after an
/// ordered (Bayer) threshold of 2x2 to 16x16. The result is the palette colours as bytes / 255; an alpha channel passes through unchanged.
/// </summary>
[Node(Category = "image/filters", DisplayName = "Quantize Image")]
public sealed class ImageQuantize
{
    static readonly string[] DitherNames = ["none", "floyd-steinberg", "bayer-2", "bayer-4", "bayer-8", "bayer-16"];

    /// <summary>The dither choices, in the order the node lists them.</summary>
    public static IReadOnlyList<string> Dithers() => DitherNames;

    /// <summary>
    /// A float (already scaled by 255) as the byte PyTorch's float-to-uint8 conversion gives on the CPU: the value truncated toward zero, then wrapped
    /// modulo 256 (measured with the fixtures' out-of-range pictures).
    /// </summary>
    public static byte TorchToUInt8(float value) => float.IsNaN(value) ? (byte)0 : unchecked((byte)(long)value);

    public ImageBatch Execute(ImageBatch image,
        [Input(Min = 1, Max = 256, Step = 1)] int colors = 256,
        [Input(Options = ["none", "floyd-steinberg", "bayer-2", "bayer-4", "bayer-8", "bayer-16"])] string dither = "none")
    {
        int bayer = dither switch
        {
            "none" or "floyd-steinberg" => 0,
            "bayer-2" => 2,
            "bayer-4" => 4,
            "bayer-8" => 8,
            "bayer-16" => 16,
            _ => throw new ArgumentException($"Unknown dither '{dither}'.", nameof(dither)),
        };
        int h = image.Height, w = image.Width, c = image.Channels, n = h * w;
        int outC = c == 4 ? 4 : 3;
        var result = new ImageBatch(image.Batch, h, w, outC);
        var rgb = new byte[n * 3];
        for (int b = 0; b < image.Batch; b++)
        {
            long frame = (long)b * n;
            for (int i = 0; i < n; i++)
                for (int ch = 0; ch < 3; ch++)
                    rgb[i * 3 + ch] = TorchToUInt8(image.Data[(frame + i) * c + Math.Min(ch, c - 1)] * 255f);
            var palette = PilQuantize.MedianCutPalette(rgb, colors);
            byte[] indices = bayer == 0
                ? PilQuantize.ToPalette(rgb, w, h, palette, dither == "floyd-steinberg")
                : PilQuantize.ToPalette(OrderedThreshold(rgb, w, h, bayer, palette.Length / 3), w, h, palette, false);
            for (int i = 0; i < n; i++)
            {
                long o = (frame + i) * outC;
                for (int ch = 0; ch < 3; ch++) result.Data[o + ch] = palette[indices[i] * 3 + ch] / 255f;
                if (outC == 4) result.Data[o + 3] = image.Data[(frame + i) * c + 3];
            }
        }
        return result;
    }

    /// <summary>
    /// Ordered dithering before the palette lookup: every byte moves by spread * t + 0.5 and is truncated into 0..255, where t is the Bayer threshold of
    /// the pixel's place in the tiled size x size matrix, normalized to (m + 0.5) / size² - 0.5, and spread = 512 / (palette entries). Measured with
    /// ComfyUI (2026-11-08) on grey pictures holding exactly P levels: spread ≈ 170.7, 128, 102.4, 64 and 32 for P = 3, 4, 5, 8 and 16; the fixture
    /// tells 512 from 510 (16 cases differ).
    /// </summary>
    static byte[] OrderedThreshold(byte[] rgb, int w, int h, int size, int entries)
    {
        var matrix = BayerMatrix(size);
        double spread = 512.0 / entries;
        var result = new byte[rgb.Length];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double t = (matrix[(y % size) * size + x % size] + 0.5) / (size * size) - 0.5;
                double shift = spread * t + 0.5;
                for (int ch = 0; ch < 3; ch++)
                {
                    int i = (y * w + x) * 3 + ch;
                    result[i] = (byte)Math.Clamp((int)Math.Truncate(rgb[i] + shift), 0, 255);
                }
            }
        return result;
    }

    /// <summary>The recursive Bayer index matrix (values 0 .. size² - 1, row by row): M(2n) = [[4M, 4M + 2], [4M + 3, 4M + 1]].</summary>
    static int[] BayerMatrix(int size)
    {
        int[] m = [0];
        for (int n = 1; n < size; n *= 2)
        {
            var next = new int[4 * n * n];
            for (int y = 0; y < n; y++)
                for (int x = 0; x < n; x++)
                {
                    int v = 4 * m[y * n + x];
                    next[y * 2 * n + x] = v;
                    next[y * 2 * n + x + n] = v + 2;
                    next[(y + n) * 2 * n + x] = v + 3;
                    next[(y + n) * 2 * n + x + n] = v + 1;
                }
            m = next;
        }
        return m;
    }
}

/// <summary>
/// Grey-scale morphology with a flat square structuring element of side <c>kernel_size</c>, every channel on its own. Output pixel x looks at
/// x - k/2 .. x - k/2 + k - 1 in each direction (an even kernel reaches one pixel further back than forward); pixels outside the picture are ignored.
/// open = dilate(erode), close = erode(dilate), gradient = dilate - erode, top_hat = image - open, bottom_hat = close - image.
/// </summary>
[Node(Category = "image/filters", DisplayName = "Apply Morphology")]
public sealed class Morphology
{
    public ImageBatch Execute(ImageBatch image,
        [Input(Options = ["erode", "dilate", "open", "close", "gradient", "bottom_hat", "top_hat"])] string operation,
        [Input(Min = 3, Max = 999, Step = 1)] int kernelSize = 3)
    {
        if (kernelSize < 1) throw new ArgumentOutOfRangeException(nameof(kernelSize), "kernel_size must be at least 1.");
        float[] Erode(float[] d) => Filter(d, image, kernelSize, max: false);
        float[] Dilate(float[] d) => Filter(d, image, kernelSize, max: true);
        var src = image.Data;
        float[] data = operation switch
        {
            "erode" => Erode(src),
            "dilate" => Dilate(src),
            "open" => Dilate(Erode(src)),
            "close" => Erode(Dilate(src)),
            "gradient" => Subtract(Dilate(src), Erode(src)),
            "top_hat" => Subtract(src, Dilate(Erode(src))),
            "bottom_hat" => Subtract(Erode(Dilate(src)), src),
            _ => throw new ArgumentException($"Invalid operation '{operation}'.", nameof(operation)),
        };
        return new ImageBatch(image.Batch, image.Height, image.Width, image.Channels, data);
    }

    static float[] Subtract(float[] a, float[] b)
    {
        var r = new float[a.Length];
        for (int i = 0; i < r.Length; i++) r[i] = a[i] - b[i];
        return r;
    }

    /// <summary>Running min or max over the window, rows then columns (a square window separates exactly).</summary>
    static float[] Filter(float[] data, ImageBatch shape, int k, bool max)
    {
        int h = shape.Height, w = shape.Width, c = shape.Channels;
        var across = new float[data.Length];
        var result = new float[data.Length];
        int longest = Math.Max(h, w);
        var line = new float[longest];
        var output = new float[longest];
        for (int b = 0; b < shape.Batch; b++)
        {
            long frame = (long)b * h * w * c;
            for (int ch = 0; ch < c; ch++)
            {
                for (int y = 0; y < h; y++)
                {
                    long row = frame + (long)y * w * c + ch;
                    for (int x = 0; x < w; x++) line[x] = data[row + (long)x * c];
                    Window(line, output, w, k, max);
                    for (int x = 0; x < w; x++) across[row + (long)x * c] = output[x];
                }
                for (int x = 0; x < w; x++)
                {
                    long col = frame + (long)x * c + ch;
                    for (int y = 0; y < h; y++) line[y] = across[col + (long)y * w * c];
                    Window(line, output, h, k, max);
                    for (int y = 0; y < h; y++) result[col + (long)y * w * c] = output[y];
                }
            }
        }
        return result;
    }

    /// <summary>
    /// output[i] = min or max of line[i - k/2 .. i - k/2 + k - 1] clipped to 0 .. n - 1, in O(n) whatever k is: block-wise prefix and suffix
    /// extremes over the line padded with the neutral value (van Herk / Gil-Werman).
    /// </summary>
    static void Window(float[] line, float[] output, int n, int k, bool max)
    {
        int before = k / 2;
        int padded = n + k - 1;
        float neutral = max ? float.NegativeInfinity : float.PositiveInfinity;
        var p = new float[padded];
        for (int i = 0; i < padded; i++)
        {
            int s = i - before;
            p[i] = s >= 0 && s < n ? line[s] : neutral;
        }
        var prefix = new float[padded];
        var suffix = new float[padded];
        for (int i = 0; i < padded; i++)
            prefix[i] = i % k == 0 ? p[i] : Pick(prefix[i - 1], p[i], max);
        for (int i = padded - 1; i >= 0; i--)
            suffix[i] = i % k == k - 1 || i == padded - 1 ? p[i] : Pick(suffix[i + 1], p[i], max);
        // The window of output i is p[i .. i + k - 1]: the suffix extreme from i to the end of its block and the prefix extreme of the next block up to
        // i + k - 1 (one of them covers it all when i starts a block).
        for (int i = 0; i < n; i++)
            output[i] = i % k == 0 ? suffix[i] : Pick(suffix[i], prefix[i + k - 1], max);
    }

    static float Pick(float a, float b, bool max) => max ? MathF.Max(a, b) : MathF.Min(a, b);
}

/// <summary>
/// A mask that is 1 where a pixel's colour equals <c>color</c> (0xRRGGBB) and 0 elsewhere. Each channel is clamped to 0..1, scaled by 255 and rounded half
/// to even before the comparison (measured: 1.2 counts as 255, 0.5 as 128).
/// </summary>
[Node(Category = "image/mask", DisplayName = "Convert Image Color to Mask")]
public sealed class ImageColorToMask
{
    public MaskBatch Execute(ImageBatch image, [Input(Min = 0, Max = 0xFFFFFF, Step = 1)] int color = 0)
    {
        int c = image.Channels, n = image.Batch * image.Height * image.Width;
        var mask = new MaskBatch(image.Batch, image.Height, image.Width);
        for (int i = 0; i < n; i++)
        {
            int packed = 0;
            for (int ch = 0; ch < 3; ch++)
            {
                float v = Math.Clamp(image.Data[(long)i * c + Math.Min(ch, c - 1)], 0f, 1f) * 255f;
                packed = packed << 8 | (int)MathF.Round(v, MidpointRounding.ToEven);
            }
            mask.Data[i] = packed == color ? 1f : 0f;
        }
        return mask;
    }
}
