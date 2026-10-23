using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

[Node(Category = "image/transform", DisplayName = "Image Flip")]
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

[Node(Category = "image/transform", DisplayName = "Image Rotate")]
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

[Node(Category = "image/postprocessing", DisplayName = "Image Quantize")]
public sealed class ImageQuantize
{
    public static string[] Dithers() => ["none", "floyd-steinberg"];

    /// <summary>Reduces each picture to at most <paramref name="colors"/> colours: a palette by median cut, every pixel mapped to the nearest entry, optionally with error diffusion.</summary>
    public ImageBatch Execute(ImageBatch image, [Input(Min = 1, Max = 256)] int colors = 256, [Input(OptionsProvider = nameof(Dithers))] string dither = "none")
    {
        if (dither is not ("none" or "floyd-steinberg")) throw new ArgumentException($"Unknown dither '{dither}'.", nameof(dither));
        int w = image.Width, h = image.Height, c = image.Channels;
        if (c < 3) throw new ArgumentException("Quantize needs a colour picture.");
        return Pixels.Map(image, f =>
        {
            var palette = MedianCut(f, w * h, c, Math.Clamp(colors, 1, 256));
            var o = (float[])f.Clone();
            var work = dither == "none" ? null : (float[])f.Clone();
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int p = (y * w + x) * c;
                    var src = work ?? f;
                    int best = 0;
                    float bestD = float.MaxValue;
                    for (int k = 0; k < palette.Length; k++)
                    {
                        float dr = src[p] - palette[k][0], dg = src[p + 1] - palette[k][1], db = src[p + 2] - palette[k][2];
                        float d = dr * dr + dg * dg + db * db;
                        if (d < bestD) { bestD = d; best = k; }
                    }
                    for (int ch = 0; ch < 3; ch++)
                    {
                        o[p + ch] = palette[best][ch];
                        if (work is null) continue;
                        float err = src[p + ch] - palette[best][ch];
                        void Push(int dx, int dy, float weight)
                        {
                            int nx = x + dx, ny = y + dy;
                            if (nx >= 0 && nx < w && ny < h) work[(ny * w + nx) * c + ch] += err * weight;
                        }
                        Push(1, 0, 7 / 16f); Push(-1, 1, 3 / 16f); Push(0, 1, 5 / 16f); Push(1, 1, 1 / 16f);
                    }
                }
            return o;
        });
    }

    static float[][] MedianCut(float[] f, int pixels, int c, int colors)
    {
        // Boxes of pixel indices; split the box with the widest colour range at the median of that channel until there are enough.
        var boxes = new List<int[]> { Enumerable.Range(0, pixels).ToArray() };
        while (boxes.Count < colors)
        {
            int pick = -1, channel = 0;
            float widest = 0;
            for (int b = 0; b < boxes.Count; b++)
            {
                if (boxes[b].Length < 2) continue;
                for (int ch = 0; ch < 3; ch++)
                {
                    float lo = float.MaxValue, hi = float.MinValue;
                    foreach (int i in boxes[b]) { float v = f[i * c + ch]; if (v < lo) lo = v; if (v > hi) hi = v; }
                    if (hi - lo > widest) { widest = hi - lo; pick = b; channel = ch; }
                }
            }
            if (pick < 0) break; // every box is one colour already
            var sorted = boxes[pick].OrderBy(i => f[i * c + channel]).ToArray();
            int half = sorted.Length / 2;
            boxes[pick] = sorted[..half];
            boxes.Add(sorted[half..]);
        }
        return boxes.Select(box =>
        {
            var mean = new float[3];
            foreach (int i in box) for (int ch = 0; ch < 3; ch++) mean[ch] += f[i * c + ch];
            for (int ch = 0; ch < 3; ch++) mean[ch] /= box.Length;
            return mean;
        }).ToArray();
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
