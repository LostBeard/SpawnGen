namespace SpawnGen.Nodes.Core;

/// <summary>
/// PyTorch's <c>torch.nn.functional.interpolate</c> (align_corners false, no anti-aliasing) and ComfyUI's <c>comfy.utils.common_upscale</c>, <c>bislerp</c> and
/// <c>repeat_to_batch_size</c>, ported index for index on NCHW planes so latents resize to the same numbers as in ComfyUI. The arithmetic is float, in the
/// order the CPU kernels use (a row is interpolated across, then the rows down).
/// </summary>
public static class TorchInterpolate
{
    /// <summary>The methods ComfyUI's latent upscale nodes offer.</summary>
    public static readonly string[] LatentMethods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"];

    /// <summary>
    /// <c>interpolate(x, size=(newH, newW), mode)</c> for <paramref name="planes"/> planes of h x w (batch x channels of an NCHW tensor).
    /// Modes: nearest, nearest-exact, bilinear, bicubic, area (all but the legacy "nearest" go through <see cref="Core.CommonUpscale"/>).
    /// </summary>
    public static float[] Resize(float[] src, int planes, int h, int w, int newH, int newW, string mode)
    {
        if (h <= 0 || w <= 0 || newH <= 0 || newW <= 0)
            throw new ArgumentException($"Cannot resize {w}x{h} to {newW}x{newH}: every size must be at least 1 (PyTorch refuses it too).");
        if (mode is "nearest-exact" or "bilinear" or "bicubic" or "area")
        {
            // One implementation of PyTorch's kernels: the picture one (CommonUpscale), plane by plane.
            var result = new float[planes * newH * newW];
            var plane = new float[h * w];
            for (int p = 0; p < planes; p++)
            {
                Array.Copy(src, p * h * w, plane, 0, h * w);
                Array.Copy(Core.CommonUpscale.Resize(plane, w, h, 1, newW, newH, mode), 0, result, p * newH * newW, newH * newW);
            }
            return result;
        }
        if (mode != "nearest") throw new ArgumentException($"Unknown interpolation mode '{mode}'.", nameof(mode));
        // PyTorch's legacy "nearest": source index floor(o * in / out) (exact when doubling), each axis on its own.
        int[] xs = Nearest(w, newW), ys = Nearest(h, newH);
        var dst = new float[planes * newH * newW];
        for (int p = 0; p < planes; p++)
            for (int y = 0; y < newH; y++)
                for (int x = 0; x < newW; x++) dst[(p * newH + y) * newW + x] = src[(p * h + ys[y]) * w + xs[x]];
        return dst;
    }

    static int[] Nearest(int inSize, int outSize)
    {
        float scale = (float)inSize / outSize; // area_pixel_compute_scale with no explicit scale factor
        return Enumerable.Range(0, outSize).Select(o => inSize == outSize ? o : outSize == 2 * inSize ? o >> 1 : Math.Min((int)MathF.Floor(o * scale), inSize - 1)).ToArray();
    }

    /// <summary>
    /// ComfyUI's <c>bislerp</c>: bilinear resizing that spherically interpolates the channel vector at each position instead of mixing it linearly
    /// (so the length of a latent "pixel" is kept). Width first, then height.
    /// </summary>
    public static float[] Bislerp(float[] src, int n, int c, int h, int w, int newH, int newW)
    {
        if (h <= 0 || w <= 0 || newH <= 0 || newW <= 0)
            throw new ArgumentException($"Cannot resize {w}x{h} to {newW}x{newH}: every size must be at least 1.");
        // Across.
        var (rw, aw, bw) = BilinearData(w, newW);
        var mid = new float[n * c * h * newW];
        var v1 = new float[c];
        var v2 = new float[c];
        var res = new float[c];
        for (int b = 0; b < n; b++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < newW; x++)
                {
                    for (int ch = 0; ch < c; ch++)
                    {
                        v1[ch] = src[((b * c + ch) * h + y) * w + aw[x]];
                        v2[ch] = src[((b * c + ch) * h + y) * w + bw[x]];
                    }
                    Slerp(v1, v2, rw[x], res);
                    for (int ch = 0; ch < c; ch++) mid[((b * c + ch) * h + y) * newW + x] = res[ch];
                }
        // Down.
        var (rh, ah, bh) = BilinearData(h, newH);
        var dst = new float[n * c * newH * newW];
        for (int b = 0; b < n; b++)
            for (int y = 0; y < newH; y++)
                for (int x = 0; x < newW; x++)
                {
                    for (int ch = 0; ch < c; ch++)
                    {
                        v1[ch] = mid[((b * c + ch) * h + ah[y]) * newW + x];
                        v2[ch] = mid[((b * c + ch) * h + bh[y]) * newW + x];
                    }
                    Slerp(v1, v2, rh[y], res);
                    for (int ch = 0; ch < c; ch++) dst[((b * c + ch) * newH + y) * newW + x] = res[ch];
                }
        return dst;
    }

    /// <summary>bislerp's <c>generate_bilinear_data</c>: a ramp 0..old-1 (and one shifted by one, its last value held) bilinearly resized; the ratios are the fractions.</summary>
    static (float[] Ratio, int[] First, int[] Second) BilinearData(int oldLength, int newLength)
    {
        var ramp1 = new float[oldLength];
        var ramp2 = new float[oldLength];
        for (int i = 0; i < oldLength; i++) { ramp1[i] = i; ramp2[i] = i + 1; }
        ramp2[^1] -= 1;
        var c1 = Resize(ramp1, 1, 1, oldLength, 1, newLength, "bilinear");
        var c2 = Resize(ramp2, 1, 1, oldLength, 1, newLength, "bilinear");
        var ratio = new float[newLength];
        var first = new int[newLength];
        var second = new int[newLength];
        for (int i = 0; i < newLength; i++)
        {
            ratio[i] = c1[i] - MathF.Floor(c1[i]);
            first[i] = (int)c1[i];
            second[i] = (int)c2[i];
        }
        return (ratio, first, second);
    }

    /// <summary>bislerp's <c>slerp</c> for one pair of channel vectors.</summary>
    static void Slerp(float[] b1, float[] b2, float r, float[] res)
    {
        int c = b1.Length;
        double s1 = 0, s2 = 0;
        for (int i = 0; i < c; i++) { s1 += (double)b1[i] * b1[i]; s2 += (double)b2[i] * b2[i]; }
        float n1 = (float)Math.Sqrt(s1), n2 = (float)Math.Sqrt(s2);
        float dot = 0;
        for (int i = 0; i < c; i++)
        {
            float a = n1 == 0f ? 0f : b1[i] / n1, b = n2 == 0f ? 0f : b2[i] / n2;
            dot += a * b;
        }
        if (dot > 1f - 1e-5f) { Array.Copy(b1, res, c); return; }                        // the same direction: the first
        if (dot < 1e-5f - 1f) { for (int i = 0; i < c; i++) res[i] = b1[i] * (1f - r) + b2[i] * r; return; } // opposite: linear
        float omega = MathF.Acos(dot), so = MathF.Sin(omega);
        float wa = MathF.Sin((1f - r) * omega) / so, wb = MathF.Sin(r * omega) / so, length = n1 * (1f - r) + n2 * r;
        for (int i = 0; i < c; i++)
        {
            float a = n1 == 0f ? 0f : b1[i] / n1, b = n2 == 0f ? 0f : b2[i] / n2;
            res[i] = (wa * a + wb * b) * length;
        }
    }

    /// <summary>
    /// <c>comfy.utils.common_upscale</c> on an NCHW tensor: with crop "center" the middle part with the target's aspect ratio is cut out first
    /// (Python rounding), then it is resized with <paramref name="method"/> (an interpolate mode or "bislerp").
    /// </summary>
    public static (float[] Data, int Height, int Width) CommonUpscale(float[] src, int n, int c, int h, int w, int width, int height, string method, string crop)
    {
        if (crop == "center")
        {
            double oldAspect = (double)w / h, newAspect = (double)width / height;
            int x = 0, y = 0;
            if (oldAspect > newAspect) x = (int)Math.Round((w - w * (newAspect / oldAspect)) / 2, MidpointRounding.ToEven);
            else if (oldAspect < newAspect) y = (int)Math.Round((h - h * (oldAspect / newAspect)) / 2, MidpointRounding.ToEven);
            if (x != 0 || y != 0)
            {
                int ch = h - y * 2, cw = w - x * 2;
                var cut = new float[n * c * ch * cw];
                for (int p = 0; p < n * c; p++)
                    for (int row = 0; row < ch; row++)
                        Array.Copy(src, (p * h + y + row) * w + x, cut, (p * ch + row) * cw, cw);
                src = cut; h = ch; w = cw;
            }
        }
        var data = method == "bislerp" ? Bislerp(src, n, c, h, w, height, width) : Resize(src, n * c, h, w, height, width, method);
        return (data, height, width);
    }

    /// <summary><c>comfy.utils.repeat_to_batch_size</c> for a batch dimension: picture i of the result is picture i mod n of the source.</summary>
    public static int RepeatIndex(int i, int sourceBatch) => i % sourceBatch;
}
