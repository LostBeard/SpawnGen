namespace SpawnGen.Nodes.Core;

public enum ResampleMethod { NearestExact, Bilinear, Area, Bicubic, Lanczos }

/// <summary>
/// Image resizing on host float data (interleaved channels: NHWC pixels, or NCHW planes with one channel per plane), separable like PIL and
/// PyTorch. Bilinear and bicubic follow PyTorch (no anti-aliasing, align_corners false); area is a box average; Lanczos widens its kernel
/// when shrinking so it does not alias.
/// </summary>
public static class Resample
{
    public static readonly string[] MethodNames = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"];

    public static ResampleMethod Parse(string name) => name switch
    {
        "nearest-exact" or "nearest" => ResampleMethod.NearestExact,
        "bilinear" => ResampleMethod.Bilinear,
        "area" => ResampleMethod.Area,
        "bicubic" => ResampleMethod.Bicubic,
        "lanczos" => ResampleMethod.Lanczos,
        _ => throw new ArgumentException($"Unknown upscale method '{name}'. Known: {string.Join(", ", MethodNames)}.", nameof(name)),
    };

    /// <summary>Resizes one picture of <paramref name="channels"/> interleaved channels from (w, h) to (newW, newH).</summary>
    public static float[] Resize(float[] src, int w, int h, int channels, int newW, int newH, ResampleMethod method)
    {
        if (w <= 0 || h <= 0 || newW <= 0 || newH <= 0) throw new ArgumentOutOfRangeException(nameof(newW), "Sizes must be positive.");
        if (w == newW && h == newH) return (float[])src.Clone();

        // Horizontal pass into a (newW x h) buffer, then vertical into (newW x newH).
        var tmp = new float[newW * h * channels];
        var xs = Weights(w, newW, method);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < newW; x++)
            {
                var (start, weights) = xs[x];
                for (int c = 0; c < channels; c++)
                {
                    float sum = 0;
                    for (int k = 0; k < weights.Length; k++) sum += weights[k] * src[(y * w + start + k) * channels + c];
                    tmp[(y * newW + x) * channels + c] = sum;
                }
            }

        var dst = new float[newW * newH * channels];
        var ys = Weights(h, newH, method);
        for (int y = 0; y < newH; y++)
        {
            var (start, weights) = ys[y];
            for (int x = 0; x < newW; x++)
                for (int c = 0; c < channels; c++)
                {
                    float sum = 0;
                    for (int k = 0; k < weights.Length; k++) sum += weights[k] * tmp[((start + k) * newW + x) * channels + c];
                    dst[(y * newW + x) * channels + c] = sum;
                }
        }
        return dst;
    }

    /// <summary>For each output sample: the first source index and the weights over the following source samples (edges are clamped by shrinking the window, weights renormalised).</summary>
    static (int Start, float[] Weights)[] Weights(int inSize, int outSize, ResampleMethod method)
    {
        double scale = (double)inSize / outSize;
        var result = new (int, float[])[outSize];
        for (int o = 0; o < outSize; o++)
        {
            if (method == ResampleMethod.NearestExact)
            {
                // PyTorch "nearest-exact": the source pixel whose centre is nearest to the output pixel's centre.
                int idx = Math.Min(inSize - 1, (int)Math.Floor((o + 0.5) * scale));
                result[o] = (idx, [1f]);
                continue;
            }

            double centre = (o + 0.5) * scale - 0.5; // source coordinate of this output pixel's centre
            double support, filterScale = 1;
            Func<double, double> kernel;
            switch (method)
            {
                case ResampleMethod.Bilinear: support = 1; kernel = Triangle; break;
                case ResampleMethod.Bicubic: support = 2; kernel = Cubic; break;
                case ResampleMethod.Lanczos: filterScale = Math.Max(1, scale); support = 3 * filterScale; kernel = Lanczos3; break;
                case ResampleMethod.Area:
                {
                    // Box over the source footprint [o*scale, (o+1)*scale): each source pixel weighted by its overlap.
                    double lo = o * scale, hi = (o + 1) * scale;
                    int first = (int)Math.Floor(lo + 1e-9), last = Math.Min(inSize - 1, (int)Math.Ceiling(hi - 1e-9) - 1);
                    var wts = new float[Math.Max(1, last - first + 1)];
                    for (int i = 0; i < wts.Length; i++) wts[i] = (float)(Math.Min(hi, first + i + 1) - Math.Max(lo, first + i));
                    Normalise(wts);
                    result[o] = (first, wts);
                    continue;
                }
                default: throw new NotSupportedException();
            }

            int start = (int)Math.Ceiling(centre - support), end = (int)Math.Floor(centre + support);
            start = Math.Max(start, 0);
            end = Math.Min(end, inSize - 1);
            if (end < start) { start = end = Math.Clamp((int)Math.Round(centre), 0, inSize - 1); }
            var w2 = new float[end - start + 1];
            for (int i = 0; i < w2.Length; i++) w2[i] = (float)kernel((start + i - centre) / filterScale);
            Normalise(w2);
            result[o] = (start, w2);
        }
        return result;
    }

    static void Normalise(float[] w)
    {
        float sum = 0;
        foreach (var v in w) sum += v;
        if (Math.Abs(sum) < 1e-12f) { Array.Fill(w, 1f / w.Length); return; }
        for (int i = 0; i < w.Length; i++) w[i] /= sum;
    }

    static double Triangle(double x) => Math.Abs(x) < 1 ? 1 - Math.Abs(x) : 0;

    /// <summary>The cubic convolution kernel with a = -0.75, as PyTorch uses for bicubic.</summary>
    static double Cubic(double x)
    {
        const double a = -0.75;
        x = Math.Abs(x);
        if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
        if (x < 2) return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
        return 0;
    }

    static double Lanczos3(double x)
    {
        x = Math.Abs(x);
        if (x < 1e-9) return 1;
        if (x >= 3) return 0;
        double px = Math.PI * x;
        return 3 * Math.Sin(px) * Math.Sin(px / 3) / (px * px);
    }
}
