namespace SpawnGen.Nodes.Core;

public enum ResampleMethod { NearestExact, Bilinear, Area, Bicubic, Lanczos }

/// <summary>
/// Image resizing on host float data (interleaved channels: NHWC pixels, or NCHW planes with one channel per plane). Nearest-exact, bilinear,
/// bicubic and area go through <see cref="TorchInterpolate"/>, the exact port of PyTorch's interpolate that ComfyUI's common_upscale calls;
/// Lanczos is separable here and widens its kernel when shrinking (ComfyUI does it in PIL on 8-bit pictures, not ported yet).
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

    /// <summary>
    /// ComfyUI's <c>comfy.utils.common_upscale</c>: with <paramref name="centerCrop"/> the source is first cut to the target's aspect ratio around its
    /// middle (offsets rounded half to even, as Python does), then resized. The crop happens before the resize, so the pixels match ComfyUI's.
    /// </summary>
    public static float[] CommonUpscale(float[] src, int w, int h, int channels, int newW, int newH, ResampleMethod method, bool centerCrop)
    {
        if (!centerCrop) return Resize(src, w, h, channels, newW, newH, method);
        double oldAspect = (double)w / h, newAspect = (double)newW / newH;
        int x0 = 0, y0 = 0;
        if (oldAspect > newAspect) x0 = (int)Math.Round((w - w * (newAspect / oldAspect)) / 2);
        else if (oldAspect < newAspect) y0 = (int)Math.Round((h - h * (oldAspect / newAspect)) / 2);
        int cw = w - 2 * x0, ch = h - 2 * y0;
        if (x0 == 0 && y0 == 0) return Resize(src, w, h, channels, newW, newH, method);
        var cut = new float[cw * ch * channels];
        for (int y = 0; y < ch; y++) Array.Copy(src, ((y0 + y) * w + x0) * channels, cut, y * cw * channels, cw * channels);
        return Resize(cut, cw, ch, channels, newW, newH, method);
    }

    /// <summary>
    /// The PyTorch modes (what ComfyUI's common_upscale uses for everything but lanczos), through the exact interpolate port: the interleaved picture
    /// becomes one plane per channel and back.
    /// </summary>
    static float[] Torch(float[] src, int w, int h, int channels, int newW, int newH, ResampleMethod method)
    {
        var planes = new float[channels * h * w];
        for (int i = 0; i < h * w; i++)
            for (int c = 0; c < channels; c++) planes[c * h * w + i] = src[i * channels + c];
        var mode = method switch { ResampleMethod.NearestExact => "nearest-exact", ResampleMethod.Bilinear => "bilinear", ResampleMethod.Bicubic => "bicubic", _ => "area" };
        var resized = TorchInterpolate.Resize(planes, channels, h, w, newH, newW, mode);
        var dst = new float[newW * newH * channels];
        for (int i = 0; i < newH * newW; i++)
            for (int c = 0; c < channels; c++) dst[i * channels + c] = resized[c * newH * newW + i];
        return dst;
    }

    /// <summary>Resizes one picture of <paramref name="channels"/> interleaved channels from (w, h) to (newW, newH).</summary>
    public static float[] Resize(float[] src, int w, int h, int channels, int newW, int newH, ResampleMethod method)
    {
        if (w <= 0 || h <= 0 || newW <= 0 || newH <= 0) throw new ArgumentOutOfRangeException(nameof(newW), "Sizes must be positive.");
        if (w == newW && h == newH) return (float[])src.Clone();
        if (method != ResampleMethod.Lanczos) return Torch(src, w, h, channels, newW, newH, method);

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

    /// <summary>Lanczos-3 weights for each output sample: the first source index and the weights over the following source samples (edges are clamped by shrinking the window, weights renormalised; the kernel widens when shrinking).</summary>
    static (int Start, float[] Weights)[] Weights(int inSize, int outSize, ResampleMethod method)
    {
        if (method != ResampleMethod.Lanczos) throw new NotSupportedException($"{method} goes through TorchInterpolate.");
        double scale = (double)inSize / outSize;
        double filterScale = Math.Max(1, scale), support = 3 * filterScale;
        var result = new (int, float[])[outSize];
        for (int o = 0; o < outSize; o++)
        {
            double centre = (o + 0.5) * scale - 0.5; // source coordinate of this output pixel's centre
            int start = Math.Max((int)Math.Ceiling(centre - support), 0), end = Math.Min((int)Math.Floor(centre + support), inSize - 1);
            if (end < start) { start = end = Math.Clamp((int)Math.Round(centre), 0, inSize - 1); }
            var w2 = new float[end - start + 1];
            for (int i = 0; i < w2.Length; i++) w2[i] = (float)Lanczos3((start + i - centre) / filterScale);
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

    static double Lanczos3(double x)
    {
        x = Math.Abs(x);
        if (x < 1e-9) return 1;
        if (x >= 3) return 0;
        double px = Math.PI * x;
        return 3 * Math.Sin(px) * Math.Sin(px / 3) / (px * px);
    }
}
