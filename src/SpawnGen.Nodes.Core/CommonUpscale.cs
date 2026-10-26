using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// ComfyUI's <c>comfy.utils.common_upscale</c>, value for value: "nearest-exact", "bilinear", "bicubic" and "area" are PyTorch's
/// <c>F.interpolate</c> (align_corners false, no anti-aliasing; area is <c>adaptive_avg_pool2d</c>), "lanczos" is Pillow's LANCZOS on the
/// picture quantised to 8 bits (and back), and crop "center" cuts the middle to the new aspect ratio first.
/// Frames are interleaved (height, width, channels) floats, like <see cref="Resample"/>. The float arithmetic follows PyTorch's CPU kernels,
/// so results agree with ComfyUI to float rounding (Pillow's lanczos is integer arithmetic and agrees exactly).
/// </summary>
public static class CommonUpscale
{
    /// <summary>Resizes one frame of <paramref name="c"/> interleaved channels from (w, h) to (newW, newH).</summary>
    public static float[] Resize(float[] src, int w, int h, int c, int newW, int newH, string method, string crop = "disabled")
    {
        if (newW <= 0 || newH <= 0) throw new ArgumentOutOfRangeException(nameof(newW), $"The new size must be positive, got {newW}x{newH}.");
        if (crop == "center") (src, w, h) = CenterCrop(src, w, h, c, newW, newH);
        else if (crop != "disabled") throw new ArgumentException($"Unknown crop '{crop}'.", nameof(crop));
        return method switch
        {
            "nearest-exact" => Nearest(src, w, h, c, newW, newH),
            "bilinear" => Separable(src, w, h, c, newW, newH, LinearTaps),
            "bicubic" => Separable(src, w, h, c, newW, newH, CubicTaps),
            "area" => Area(src, w, h, c, newW, newH),
            "lanczos" => PilLanczos.Resize(src, w, h, c, newW, newH),
            _ => throw new ArgumentException($"Unknown upscale method '{method}'.", nameof(method)),
        };
    }

    /// <summary>Every picture of the batch resized (ComfyUI's <c>common_upscale(image.movedim(-1, 1), ...)</c>).</summary>
    public static ImageBatch Resize(ImageBatch image, int newW, int newH, string method, string crop = "disabled") =>
        Pixels.Map(image, f => Resize(f, image.Width, image.Height, image.Channels, newW, newH, method, crop), newH, newW);

    /// <summary>Every mask of the batch resized (one channel).</summary>
    public static MaskBatch Resize(MaskBatch mask, int newW, int newH, string method, string crop = "disabled")
    {
        var r = new MaskBatch(mask.Batch, newH, newW);
        int n = mask.Height * mask.Width;
        for (int b = 0; b < mask.Batch; b++)
        {
            var one = new float[n];
            Array.Copy(mask.Data, b * n, one, 0, n);
            Array.Copy(Resize(one, mask.Width, mask.Height, 1, newW, newH, method, crop), 0, r.Data, b * newW * newH, newW * newH);
        }
        return r;
    }

    /// <summary>common_upscale's crop "center": the middle part with the target aspect ratio (Python <c>round</c>, half to even).</summary>
    static (float[], int, int) CenterCrop(float[] src, int w, int h, int c, int newW, int newH)
    {
        double oldAspect = (double)w / h, newAspect = (double)newW / newH;
        int x = 0, y = 0;
        if (oldAspect > newAspect) x = (int)Math.Round((w - w * (newAspect / oldAspect)) / 2);
        else if (oldAspect < newAspect) y = (int)Math.Round((h - h * (oldAspect / newAspect)) / 2);
        if (x == 0 && y == 0) return (src, w, h);
        int cw = w - 2 * x, ch = h - 2 * y;
        var cut = new float[cw * ch * c];
        for (int row = 0; row < ch; row++) Array.Copy(src, ((y + row) * w + x) * c, cut, row * cw * c, cw * c);
        return (cut, cw, ch);
    }

    // PyTorch computes the source/destination ratio in float: (float)input_size / output_size.
    static float Ratio(int inSize, int outSize) => (float)inSize / outSize;

    static float[] Nearest(float[] src, int w, int h, int c, int newW, int newH)
    {
        // nearest_neighbor_exact_compute_source_index: min(floorf((dst + 0.5) * scale), input_size - 1).
        static int[] Index(int inSize, int outSize)
        {
            float scale = Ratio(inSize, outSize);
            var idx = new int[outSize];
            for (int o = 0; o < outSize; o++) idx[o] = Math.Min((int)MathF.Floor((float)((o + 0.5) * scale)), inSize - 1);
            return idx;
        }
        var xs = Index(w, newW);
        var ys = Index(h, newH);
        var dst = new float[newW * newH * c];
        for (int y = 0; y < newH; y++)
            for (int x = 0; x < newW; x++)
                Array.Copy(src, (ys[y] * w + xs[x]) * c, dst, (y * newW + x) * c, c);
        return dst;
    }

    /// <summary>The taps of one output sample along one axis: source indices (already clamped) and their weights.</summary>
    readonly record struct Taps(int[] Index, float[] Weight);

    /// <summary>HelperInterpLinear: two taps; same size copies; a negative source coordinate is clamped to 0.</summary>
    static Taps[] LinearTaps(int inSize, int outSize)
    {
        var taps = new Taps[outSize];
        float scale = Ratio(inSize, outSize);
        for (int o = 0; o < outSize; o++)
        {
            if (inSize == outSize) { taps[o] = new([o, o], [1f, 0f]); continue; }
            float real = scale * (o + 0.5f) - 0.5f;
            if (real < 0f) real = 0f;
            int i0 = Math.Min((int)MathF.Floor(real), inSize - 1);
            float l1 = Math.Min(Math.Max(real - i0, 0f), 1f);
            int i1 = i0 + (i0 < inSize - 1 ? 1 : 0);
            taps[o] = new([i0, i1], [1f - l1, l1]);
        }
        return taps;
    }

    /// <summary>HelperInterpCubic: four taps around floor(source), indices clamped to the picture, cubic convolution weights with A = -0.75.</summary>
    static Taps[] CubicTaps(int inSize, int outSize)
    {
        const float A = -0.75f;
        static float Conv1(float x) => ((A + 2) * x - (A + 3)) * x * x + 1;
        static float Conv2(float x) => ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
        var taps = new Taps[outSize];
        float scale = Ratio(inSize, outSize);
        for (int o = 0; o < outSize; o++)
        {
            float real = scale * (o + 0.5f) - 0.5f; // not clamped for cubic
            int i0 = Math.Min((int)MathF.Floor(real), inSize - 1);
            float t = Math.Min(Math.Max(real - i0, 0f), 1f);
            float x2 = 1f - t;
            var idx = new int[4];
            for (int j = 0; j < 4; j++) idx[j] = Math.Clamp(i0 - 1 + j, 0, inSize - 1);
            taps[o] = new(idx, [Conv2(t + 1f), Conv1(t), Conv1(x2), Conv2(x2 + 1f)]);
        }
        return taps;
    }

    /// <summary>
    /// PyTorch's generic separable kernel: for each output pixel, sum over the row taps of (sum over the column taps of pixel * weight) * row weight,
    /// accumulated in that order in float.
    /// </summary>
    static float[] Separable(float[] src, int w, int h, int c, int newW, int newH, Func<int, int, Taps[]> taps)
    {
        var xs = taps(w, newW);
        var ys = taps(h, newH);
        var dst = new float[newW * newH * c];
        for (int y = 0; y < newH; y++)
        {
            var ty = ys[y];
            for (int x = 0; x < newW; x++)
            {
                var tx = xs[x];
                for (int ch = 0; ch < c; ch++)
                {
                    float output = 0;
                    for (int j = 0; j < ty.Index.Length; j++)
                    {
                        int row = ty.Index[j] * w;
                        float t = src[(row + tx.Index[0]) * c + ch] * tx.Weight[0];
                        for (int i = 1; i < tx.Index.Length; i++) t += src[(row + tx.Index[i]) * c + ch] * tx.Weight[i];
                        output = j == 0 ? t * ty.Weight[0] : output + t * ty.Weight[j];
                    }
                    dst[(y * newW + x) * c + ch] = output;
                }
            }
        }
        return dst;
    }

    /// <summary>adaptive_avg_pool2d: output (oy, ox) is the plain mean of rows floor(oy*H/OH) up to ceil((oy+1)*H/OH) and the same for columns.</summary>
    static float[] Area(float[] src, int w, int h, int c, int newW, int newH)
    {
        // AdaptivePooling.h start_index / end_index, in integers: floor(a * size / b) and ceil((a + 1) * size / b).
        static int Start(int a, int b, int size) => (int)((long)a / b * size + (long)a % b * size / b);
        static int End(int a, int b, int size) => (int)(1 + ((long)(a + 1) * size - 1) / b);
        var dst = new float[newW * newH * c];
        for (int oy = 0; oy < newH; oy++)
        {
            int y0 = Start(oy, newH, h), y1 = End(oy, newH, h);
            for (int ox = 0; ox < newW; ox++)
            {
                int x0 = Start(ox, newW, w), x1 = End(ox, newW, w);
                int count = (y1 - y0) * (x1 - x0);
                for (int ch = 0; ch < c; ch++)
                {
                    float sum = 0;
                    for (int y = y0; y < y1; y++)
                        for (int x = x0; x < x1; x++) sum += src[(y * w + x) * c + ch];
                    dst[(oy * newW + ox) * c + ch] = sum / count;
                }
            }
        }
        return dst;
    }
}

/// <summary>
/// ComfyUI's <c>comfy.utils.lanczos</c>: <c>np.clip(255 * x, 0, 255).astype(uint8)</c> (truncation), Pillow's <c>Image.resize(LANCZOS)</c>
/// (two 8-bit passes, 22-bit fixed-point weights, RGBA and LA resized premultiplied), then <c>/ 255</c>.
/// </summary>
public static class PilLanczos
{
    const int PrecisionBits = 32 - 8 - 2;

    public static float[] Resize(float[] src, int w, int h, int c, int newW, int newH)
    {
        if (c is not (1 or 2 or 3 or 4)) throw new ArgumentException($"Lanczos (Pillow) works on 1 to 4 channels, got {c}.", nameof(c));
        return FromBytes(Resize(ToBytes(src), w, h, c, newW, newH));
    }

    /// <summary>
    /// Pillow's <c>Image.resize(LANCZOS)</c> on 8-bit pixels of <paramref name="c"/> interleaved channels (1 L, 2 LA, 3 RGB, 4 RGBA; LA and RGBA are
    /// resized premultiplied). The same size gives a copy, as in Pillow.
    /// </summary>
    public static byte[] Resize(byte[] px, int w, int h, int c, int newW, int newH)
    {
        if (newW == w && newH == h) return (byte[])px.Clone(); // Pillow returns a copy for the same size
        bool alpha = c is 2 or 4; // LA and RGBA are resized as La / RGBa
        if (alpha) { px = (byte[])px.Clone(); Premultiply(px, c); }
        // Pillow (Resample.c) runs the vertical pass first when the height shrinks by more than twice what the width shrinks; each pass rounds to 8 bits.
        bool verticalFirst = h - newH > 0 && h - newH > (w - newW) * 2;
        if (verticalFirst && newH != h) { px = Pass(px, w, h, c, newH, horizontal: false); h = newH; }
        if (newW != w) { px = Pass(px, w, h, c, newW, horizontal: true); w = newW; }
        if (newH != h) { px = Pass(px, w, h, c, newH, horizontal: false); h = newH; }
        if (alpha) Unpremultiply(px, c);
        return px;
    }

    /// <summary>ComfyUI's tensor-to-Pillow step: <c>np.clip(255 * x, 0, 255).astype(uint8)</c>, truncated, not rounded.</summary>
    public static byte[] ToBytes(float[] src)
    {
        var px = new byte[src.Length];
        for (int i = 0; i < src.Length; i++) px[i] = (byte)Math.Clamp(255f * src[i], 0f, 255f);
        return px;
    }

    /// <summary>And back: <c>np.array(img).astype(np.float32) / 255.0</c>.</summary>
    public static float[] FromBytes(byte[] px)
    {
        var dst = new float[px.Length];
        for (int i = 0; i < px.Length; i++) dst[i] = px[i] / 255f;
        return dst;
    }

    // Pillow's MULDIV255 (RGBA -> RGBa) and its inverse (RGBa -> RGBA: integer division, clipped; alpha 0 or 255 left alone).
    static void Premultiply(byte[] px, int c)
    {
        for (int p = 0; p < px.Length; p += c)
        {
            int a = px[p + c - 1];
            for (int k = 0; k < c - 1; k++) { int tmp = px[p + k] * a + 128; px[p + k] = (byte)(((tmp >> 8) + tmp) >> 8); }
        }
    }

    static void Unpremultiply(byte[] px, int c)
    {
        for (int p = 0; p < px.Length; p += c)
        {
            int a = px[p + c - 1];
            if (a is 0 or 255) continue;
            for (int k = 0; k < c - 1; k++) px[p + k] = (byte)Math.Min(255, 255 * px[p + k] / a);
        }
    }

    static double Sinc(double x)
    {
        if (x == 0.0) return 1.0;
        x *= Math.PI;
        return Math.Sin(x) / x;
    }

    static double Lanczos(double x) => -3.0 <= x && x < 3.0 ? Sinc(x) * Sinc(x / 3) : 0.0;

    /// <summary>precompute_coeffs + normalize_coeffs_8bpc: per output sample the first source index and the fixed-point weights.</summary>
    static (int Min, int[] K)[] Coeffs(int inSize, int outSize)
    {
        double scale = (double)inSize / outSize, filterscale = Math.Max(scale, 1.0);
        double support = 3.0 * filterscale, ss = 1.0 / filterscale;
        var result = new (int, int[])[outSize];
        for (int xx = 0; xx < outSize; xx++)
        {
            double center = (xx + 0.5) * scale;
            int xmin = Math.Max(0, (int)(center - support + 0.5));
            int xmax = Math.Min(inSize, (int)(center + support + 0.5)) - xmin;
            var k = new double[Math.Max(0, xmax)];
            double ww = 0;
            for (int x = 0; x < k.Length; x++) { k[x] = Lanczos((x + xmin - center + 0.5) * ss); ww += k[x]; }
            var fixedK = new int[k.Length];
            for (int x = 0; x < k.Length; x++)
            {
                double v = ww != 0.0 ? k[x] / ww : k[x];
                fixedK[x] = v < 0 ? (int)(-0.5 + v * (1 << PrecisionBits)) : (int)(0.5 + v * (1 << PrecisionBits));
            }
            result[xx] = (xmin, fixedK);
        }
        return result;
    }

    static byte[] Pass(byte[] src, int w, int h, int c, int outSize, bool horizontal)
    {
        var coeffs = Coeffs(horizontal ? w : h, outSize);
        int ow = horizontal ? outSize : w, oh = horizontal ? h : outSize;
        var dst = new byte[ow * oh * c];
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++)
            {
                var (min, k) = coeffs[horizontal ? x : y];
                for (int ch = 0; ch < c; ch++)
                {
                    int ss = 1 << (PrecisionBits - 1);
                    for (int i = 0; i < k.Length; i++)
                    {
                        int sx = horizontal ? min + i : x, sy = horizontal ? y : min + i;
                        ss += src[(sy * w + sx) * c + ch] * k[i];
                    }
                    dst[(y * ow + x) * c + ch] = (byte)Math.Clamp(ss >> PrecisionBits, 0, 255);
                }
            }
        return dst;
    }
}
