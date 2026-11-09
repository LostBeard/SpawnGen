namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md). The plane resizes follow PyTorch's documented torch.nn.functional.interpolate (BSD-3-Clause, see
// CommonUpscale); bislerp and the latent crop were measured with ComfyUI (2026-11-08, Upscale Latent on random 4-channel latents up and down: the model
// below reproduces them to 1.4e-6; edge cases probed with zero, parallel and opposite channel vectors).

/// <summary>Resizes of planar (batch, channels, height, width) tensors such as latents.</summary>
public static class TorchInterpolate
{
    /// <summary>The latent upscale methods, in the order the nodes list them.</summary>
    public static readonly string[] LatentMethods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"];

    /// <summary>
    /// Resizes each of <paramref name="planes"/> planes of <paramref name="h"/> x <paramref name="w"/> to <paramref name="newH"/> x <paramref name="newW"/>
    /// with a PyTorch interpolate mode: nearest-exact, bilinear, bicubic, area, or nearest (PyTorch's legacy nearest: input index floor(o * in / out)).
    /// </summary>
    public static float[] Resize(float[] src, int planes, int h, int w, int newH, int newW, string mode)
    {
        int n = h * w, m = newH * newW;
        var result = new float[(long)planes * m];
        var plane = new float[n];
        for (int p = 0; p < planes; p++)
        {
            Array.Copy(src, (long)p * n, plane, 0, n);
            var resized = mode == "nearest" ? Nearest(plane, h, w, newH, newW) : global::SpawnGen.Nodes.Core.CommonUpscale.Resize(plane, w, h, 1, newW, newH, mode);
            Array.Copy(resized, 0, result, (long)p * m, m);
        }
        return result;
    }

    static float[] Nearest(float[] src, int h, int w, int newH, int newW)
    {
        if (h <= 0 || w <= 0 || newH <= 0 || newW <= 0)
            throw new ArgumentException($"Input and output sizes should be greater than 0, but got input (H: {h}, W: {w}) output (H: {newH}, W: {newW})");
        float sy = (float)h / newH, sx = (float)w / newW;
        var result = new float[newH * newW];
        for (int y = 0; y < newH; y++)
        {
            int iy = Math.Min((int)MathF.Floor(y * sy), h - 1);
            for (int x = 0; x < newW; x++) result[y * newW + x] = src[iy * w + Math.Min((int)MathF.Floor(x * sx), w - 1)];
        }
        return result;
    }

    /// <summary>
    /// The latent resize of the upscale nodes: crop "center" first cuts to the target aspect (as <see cref="Core.CommonUpscale"/>), then every channel plane is
    /// resized with <paramref name="method"/>, or all channels together with bislerp.
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
                int cw = Math.Max(w - 2 * x, 0), ch = Math.Max(h - 2 * y, 0);
                var cut = new float[(long)n * c * cw * ch];
                for (int p = 0; p < n * c; p++)
                    for (int row = 0; row < ch; row++)
                        Array.Copy(src, ((long)p * h + row + y) * w + x, cut, ((long)p * ch + row) * cw, cw);
                (src, w, h) = (cut, cw, ch);
            }
        }
        else if (crop != "disabled") throw new ArgumentException($"Unknown crop '{crop}'.", nameof(crop));
        var data = method == "bislerp" ? Bislerp(src, n, c, h, w, height, width) : Resize(src, n * c, h, w, height, width, method);
        return (data, height, width);
    }

    /// <summary>
    /// bislerp, as measured with ComfyUI: two passes, width first, then height. Along each, output position o samples input position
    /// (o + 0.5) * in / out - 0.5 clamped to the picture (as bilinear), between pixels i0 = floor and i1 = i0 + 1 (the last pixel repeats), with t the
    /// fraction. The channel vectors a and b of those pixels are combined as the spherical interpolation of their directions (a zero vector has
    /// direction 0) times the linear interpolation of their lengths; when the directions' dot product exceeds 0.99999 the result is a itself, and below
    /// -0.99999 it is the plain linear interpolation (1 - t) a + t b.
    /// </summary>
    public static float[] Bislerp(float[] src, int n, int c, int h, int w, int newH, int newW)
    {
        if (h <= 0 || w <= 0 || newH <= 0 || newW <= 0)
            throw new ArgumentException($"Input and output sizes should be greater than 0, but got input (H: {h}, W: {w}) output (H: {newH}, W: {newW})");
        var across = Pass(src, n, c, h, w, newW, alongWidth: true);
        return Pass(across, n, c, h, newW, newH, alongWidth: false);
    }

    /// <summary>One bislerp pass along the width (to <paramref name="size"/> columns) or the height (to <paramref name="size"/> rows).</summary>
    static float[] Pass(float[] src, int n, int c, int h, int w, int size, bool alongWidth)
    {
        int inSize = alongWidth ? w : h;
        int oh = alongWidth ? h : size, ow = alongWidth ? size : w;
        var result = new float[(long)n * c * oh * ow];
        var i0 = new int[size];
        var i1 = new int[size];
        var t = new float[size];
        for (int o = 0; o < size; o++)
        {
            float s = Math.Clamp((float)((o + 0.5) * inSize / size - 0.5), 0f, inSize - 1);
            i0[o] = Math.Min((int)MathF.Floor(s), inSize - 1);
            i1[o] = Math.Min(i0[o] + 1, inSize - 1);
            t[o] = s - i0[o];
        }
        var a = new float[c];
        var b = new float[c];
        var r = new float[c];
        long plane = (long)h * w, outPlane = (long)oh * ow;
        for (int bi = 0; bi < n; bi++)
            for (int y = 0; y < oh; y++)
                for (int x = 0; x < ow; x++)
                {
                    int o = alongWidth ? x : y;
                    long at0 = alongWidth ? (long)y * w + i0[o] : (long)i0[o] * w + x;
                    long at1 = alongWidth ? (long)y * w + i1[o] : (long)i1[o] * w + x;
                    for (int k = 0; k < c; k++)
                    {
                        a[k] = src[((long)bi * c + k) * plane + at0];
                        b[k] = src[((long)bi * c + k) * plane + at1];
                    }
                    Slerp(a, b, t[o], r);
                    long dst = (long)y * ow + x;
                    for (int k = 0; k < c; k++) result[((long)bi * c + k) * outPlane + dst] = r[k];
                }
        return result;
    }

    static void Slerp(float[] a, float[] b, float t, float[] r)
    {
        int c = a.Length;
        float na = 0, nb = 0;
        for (int k = 0; k < c; k++) { na += a[k] * a[k]; nb += b[k] * b[k]; }
        na = MathF.Sqrt(na);
        nb = MathF.Sqrt(nb);
        float dot = 0;
        for (int k = 0; k < c; k++) dot += (na > 0 ? a[k] / na : 0f) * (nb > 0 ? b[k] / nb : 0f);
        if (dot > 0.99999f)
        {
            Array.Copy(a, r, c);
            return;
        }
        if (dot < -0.99999f)
        {
            for (int k = 0; k < c; k++) r[k] = a[k] * (1 - t) + b[k] * t;
            return;
        }
        float omega = MathF.Acos(Math.Clamp(dot, -1f, 1f));
        float so = MathF.Sin(omega);
        float wa = MathF.Sin((1 - t) * omega) / so, wb = MathF.Sin(t * omega) / so;
        float length = na * (1 - t) + nb * t;
        for (int k = 0; k < c; k++)
            r[k] = (wa * (na > 0 ? a[k] / na : 0f) + wb * (nb > 0 ? b[k] / nb : 0f)) * length;
    }
}
