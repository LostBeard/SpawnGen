using System.Globalization;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// Detect Edges (Canny): Canny's published algorithm (J. Canny, "A Computational Approach to Edge Detection", 1986) with the defaults kornia documents for
/// <c>kornia.filters.canny</c>: grey (0.299 R + 0.587 G + 0.114 B), a 5x5 Gaussian of sigma 1, Sobel gradients, non-maximum suppression along the gradient
/// quantized to 45 degrees, the two thresholds and hysteresis over 8-connected neighbours. The float32 order of each step was measured against kornia's
/// public functions so ComfyUI's edge maps come out pixel for pixel. The edges are white on black in three channels.
/// </summary>
[Node("Canny", Category = "image/filters", DisplayName = "Detect Edges (Canny)")]
public sealed class Canny
{
    // kornia.filters.get_gaussian_kernel1d(5, 1.0), float32 as measured.
    static readonly float[] Gaussian = [0.054488688707351685f, 0.24420136213302612f, 0.40261995792388916f, 0.24420136213302612f, 0.054488688707351685f];

    public ImageBatch Execute(ImageBatch image,
        [Input(Min = 0.01, Max = 0.99, Step = 0.01)] float lowThreshold = 0.4f,
        [Input(Min = 0.01, Max = 0.99, Step = 0.01)] float highThreshold = 0.8f)
    {
        // Measured with ComfyUI: these checks and messages (kornia's), in this order; equal thresholds are allowed, a high threshold of 1 or more too.
        if (lowThreshold > highThreshold)
            throw new ArgumentException($"Invalid input thresholds. low_threshold should be smaller than or equal to the high_threshold. Got: {Repr(lowThreshold)}>{Repr(highThreshold)}");
        if (lowThreshold <= 0)
            throw new ArgumentException($"Invalid low threshold. Should be positive. Got: {Repr(lowThreshold)}");
        int c = image.Channels;
        if (c == 2) throw new ArgumentException($"Not a color or gray tensor. Got shape [{image.Batch}, 2, {image.Height}, {image.Width}].\ncanny expects 1 or 3 channels.");
        int h = image.Height, w = image.Width;
        if (h < 3 || w < 3)
            throw new ArgumentException($"Detect Edges needs a picture at least 3 pixels wide and high, not {w}x{h} (the 5x5 Gaussian reflects 2 pixels at each border).");

        var result = new ImageBatch(image.Batch, h, w, 3);
        var gray = new float[h * w];
        for (int f = 0; f < image.Batch; f++)
        {
            int frame = f * h * w;
            for (int p = 0; p < h * w; p++)
            {
                int i = (frame + p) * c;
                // A fourth channel (alpha) and beyond are left out (measured: RGBA gives the RGB edges).
                gray[p] = c == 1 ? image.Data[i] : (0.299f * image.Data[i] + 0.587f * image.Data[i + 1]) + 0.114f * image.Data[i + 2];
            }
            var edges = Edges(gray, h, w, lowThreshold, highThreshold);
            for (int p = 0; p < h * w; p++)
                if (edges[p])
                {
                    int o = (frame + p) * 3;
                    result.Data[o] = result.Data[o + 1] = result.Data[o + 2] = 1f;
                }
        }
        return result;
    }

    static string Repr(float v) => PyNumber.FloatRepr(double.Parse(v.ToString("R", CultureInfo.InvariantCulture), CultureInfo.InvariantCulture));

    static bool[] Edges(float[] gray, int h, int w, float low, float high)
    {
        // Gaussian, rows then columns, reflected at the borders (the edge pixel not repeated). Measured float32 order: the first tap's product, three
        // fused multiply-adds, then the last tap's product added.
        var blurred = Blur(Blur(gray, h, w, horizontal: true), h, w, horizontal: false);

        // Sobel with the border pixels repeated; the order of the additions is the measured one.
        var magnitude = new float[h * w];
        var direction = new int[h * w];
        for (int y = 0; y < h; y++)
        {
            int ym = Math.Max(y - 1, 0), yp = Math.Min(y + 1, h - 1);
            for (int x = 0; x < w; x++)
            {
                int xm = Math.Max(x - 1, 0), xp = Math.Min(x + 1, w - 1);
                float p00 = blurred[ym * w + xm], p01 = blurred[ym * w + x], p02 = blurred[ym * w + xp];
                float p10 = blurred[y * w + xm], p12 = blurred[y * w + xp];
                float p20 = blurred[yp * w + xm], p21 = blurred[yp * w + x], p22 = blurred[yp * w + xp];
                float gx = (((-p00 + p02) + -p20) + (-2f * p10 + 2f * p12)) + p22;
                float gy = (((-p00 + -p02) + p20) + (-2f * p01 + 2f * p21)) + p22;
                magnitude[y * w + x] = MathF.Sqrt((gx * gx + gy * gy) + 1e-6f);
                // The gradient direction in degrees, rounded half to even to a multiple of 45: index 0 = +x (right), 2 = +y (down), counter-wise.
                float degrees = 180f * MathF.Atan2(gy, gx) / MathF.PI;
                int q = (int)MathF.Round(degrees / 45f, MidpointRounding.ToEven);
                direction[y * w + x] = ((q % 8) + 8) % 8;
            }
        }

        // Non-maximum suppression: a pixel stays when it is larger than both neighbours along its direction (outside the picture counts as 0). Measured
        // tie rule: being equal to the neighbour that comes later in reading order (right, down-right, down, down-left) still counts as a maximum.
        var weak = new bool[h * w];
        var strong = new bool[h * w];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int p = y * w + x, d = direction[p];
                float m = magnitude[p];
                if (!Above(m, Neighbour(magnitude, h, w, y, x, d), d) || !Above(m, Neighbour(magnitude, h, w, y, x, (d + 4) % 8), (d + 4) % 8)) continue;
                weak[p] = m > low;
                strong[p] = m > high;
            }

        // Hysteresis: weak pixels joined to a strong one through 8-connected weak pixels become edges.
        var edges = (bool[])strong.Clone();
        var queue = new Queue<int>();
        for (int p = 0; p < h * w; p++) if (strong[p]) queue.Enqueue(p);
        while (queue.Count > 0)
        {
            int p = queue.Dequeue(), y = p / w, x = p % w;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                {
                    int v = y + dy, u = x + dx;
                    if (v < 0 || v >= h || u < 0 || u >= w) continue;
                    int n = v * w + u;
                    if (weak[n] && !edges[n]) { edges[n] = true; queue.Enqueue(n); }
                }
        }
        return edges;
    }

    static readonly (int Dy, int Dx)[] Offsets = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)];

    static float Neighbour(float[] magnitude, int h, int w, int y, int x, int d)
    {
        int v = y + Offsets[d].Dy, u = x + Offsets[d].Dx;
        return v < 0 || v >= h || u < 0 || u >= w ? 0f : magnitude[v * w + u];
    }

    static bool Above(float m, float neighbour, int d)
    {
        float diff = m - neighbour;
        return d < 4 ? diff >= 0f : diff > 0f;
    }

    static float[] Blur(float[] src, int h, int w, bool horizontal)
    {
        var dst = new float[h * w];
        int n = horizontal ? w : h;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int i = horizontal ? x : y;
                float Tap(int j)
                {
                    int k = Reflect(i + j - 2, n);
                    return horizontal ? src[y * w + k] : src[k * w + x];
                }
                float s = Gaussian[0] * Tap(0);
                s = MathF.FusedMultiplyAdd(Gaussian[1], Tap(1), s);
                s = MathF.FusedMultiplyAdd(Gaussian[2], Tap(2), s);
                s = MathF.FusedMultiplyAdd(Gaussian[3], Tap(3), s);
                dst[y * w + x] = s + Gaussian[4] * Tap(4);
            }
        return dst;
    }

    static int Reflect(int i, int n) => i < 0 ? -i : i >= n ? 2 * (n - 1) - i : i;
}

/// <summary>
/// Add Noise to Image, as measured with ComfyUI: <c>torch.randn</c> on a CPU generator seeded with the seed, drawn for the whole picture batch in its own
/// layout (batch, height, width, channels), times the strength, added and clamped to 0..1 in float32. A fourth channel (alpha) is kept as it was.
/// </summary>
[Node("ImageAddNoise", Category = "image/filters", DisplayName = "Add Noise to Image")]
public sealed class ImageAddNoise
{
    public ImageBatch Execute(ImageBatch image,
        [Input(Min = 0, Max = 18446744073709551615d, Tooltip = "The random seed used for creating the noise.")] long seed = 0,
        [Input(Min = 0.0, Max = 1.0, Step = 0.01)] float strength = 0.5f)
    {
        var noise = TorchRandom.Randn(seed, image.Data.Length);
        int c = image.Channels;
        var result = new ImageBatch(image.Batch, image.Height, image.Width, c);
        for (int i = 0; i < noise.Length; i++)
        {
            float v = image.Data[i];
            result.Data[i] = c == 4 && i % 4 == 3 ? v : Math.Clamp(v + strength * noise[i], 0f, 1f);
        }
        return result;
    }
}
