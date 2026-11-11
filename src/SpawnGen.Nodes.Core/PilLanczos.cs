namespace SpawnGen.Nodes.Core;

/// <summary>
/// Pillow's <c>Image.resize(size, Image.LANCZOS)</c> on 8-bit pictures, following Pillow 13.0.0 (src/libImaging/Resample.c and Convert.c, MIT-CMU;
/// see THIRD-PARTY-NOTICES.md): separable lanczos-3 coefficients in 22-bit fixed point, one pass per direction with the result rounded to 8 bits in
/// between, the vertical pass first only when the height shrinks by more than twice the width. LA and RGBA pictures are premultiplied by their alpha
/// for the resize (Pillow's La / RGBa modes). Pixels are interleaved bytes, rows top to bottom, 1 to 4 channels (L, LA, RGB, RGBA).
/// </summary>
public static class PilLanczos
{
    const int PrecisionBits = 32 - 8 - 2;
    const double Support = 3.0;

    /// <summary>
    /// Floats to bytes the way a float32 picture becomes an 8-bit one for Pillow (measured with ComfyUI's lanczos path): 255 * v in float32, clamped to
    /// 0..255, truncated.
    /// </summary>
    public static byte[] ToBytes(float[] values)
    {
        var bytes = new byte[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            float v = 255f * values[i];
            bytes[i] = (byte)(v > 255f ? 255f : v > 0f ? v : 0f);
        }
        return bytes;
    }

    /// <summary>Bytes back to float32 values: b / 255.</summary>
    public static float[] FromBytes(byte[] bytes)
    {
        var values = new float[bytes.Length];
        for (int i = 0; i < bytes.Length; i++) values[i] = bytes[i] / 255f;
        return values;
    }

    /// <summary><see cref="ToBytes"/>, <see cref="Resize(byte[], int, int, int, int, int)"/>, <see cref="FromBytes"/>.</summary>
    public static float[] Resize(float[] frame, int w, int h, int channels, int newW, int newH) =>
        FromBytes(Resize(ToBytes(frame), w, h, channels, newW, newH));

    /// <summary>Pillow's resize with LANCZOS of an 8-bit picture of 1 to 4 channels; the same size gives a copy.</summary>
    public static byte[] Resize(byte[] px, int w, int h, int channels, int newW, int newH)
    {
        if (newW <= 0 || newH <= 0) throw new ArgumentException("height and width must be > 0");
        if (channels is < 1 or > 4) throw new ArgumentException($"Pillow has no {channels}-channel 8-bit mode.", nameof(channels));
        if (px.Length != w * h * channels) throw new ArgumentException("Pixel count does not match the size.", nameof(px));
        if (newW == w && newH == h) return (byte[])px.Clone();
        bool alpha = channels is 2 or 4;
        var src = px;
        if (alpha)
        {
            src = (byte[])px.Clone();
            Premultiply(src, channels);
        }
        bool needHorizontal = newW != w, needVertical = newH != h;
        bool horizontalFirst = !(h - newH > 0 && h - newH > (w - newW) * 2);
        byte[] cur = src;
        int cw = w, ch = h;
        if (horizontalFirst)
        {
            if (needHorizontal) { cur = Horizontal(cur, cw, ch, channels, newW); cw = newW; }
            if (needVertical) { cur = Vertical(cur, cw, ch, channels, newH); ch = newH; }
        }
        else
        {
            if (needVertical) { cur = Vertical(cur, cw, ch, channels, newH); ch = newH; }
            if (needHorizontal) { cur = Horizontal(cur, cw, ch, channels, newW); cw = newW; }
        }
        if (alpha) Unpremultiply(cur, channels);
        return cur;
    }

    /// <summary>Pillow's RGBA to RGBa (and LA to La): every colour byte becomes MULDIV255(v, alpha) = ((t &gt;&gt; 8) + t) &gt;&gt; 8 with t = v * alpha + 128.</summary>
    public static void Premultiply(byte[] px, int channels)
    {
        int a = channels - 1;
        for (int p = 0; p < px.Length; p += channels)
        {
            uint alpha = px[p + a];
            for (int k = 0; k < a; k++)
            {
                uint t = px[p + k] * alpha + 128;
                px[p + k] = (byte)(((t >> 8) + t) >> 8);
            }
        }
    }

    /// <summary>Pillow's RGBa to RGBA (and La to LA): alpha 0 or 255 keeps the bytes, otherwise CLIP8(255 * v / alpha) with integer division.</summary>
    public static void Unpremultiply(byte[] px, int channels)
    {
        int a = channels - 1;
        for (int p = 0; p < px.Length; p += channels)
        {
            int alpha = px[p + a];
            if (alpha is 0 or 255) continue;
            for (int k = 0; k < a; k++) px[p + k] = (byte)Math.Min(255, 255 * px[p + k] / alpha);
        }
    }

    static double Sinc(double x)
    {
        if (x == 0.0) return 1.0;
        x *= Math.PI;
        return Math.Sin(x) / x;
    }

    static double Lanczos(double x) => -3.0 <= x && x < 3.0 ? Sinc(x) * Sinc(x / 3) : 0.0;

    /// <summary>Pillow's precompute_coeffs and normalize_coeffs_8bpc for one direction: per output index, the first input index, the count, the weights.</summary>
    static (int[] Bounds, int[] Weights, int KSize) Coefficients(int inSize, int outSize)
    {
        double scale = (double)(float)inSize / outSize;
        double filterScale = Math.Max(scale, 1.0);
        double support = Support * filterScale;
        int ksize = (int)Math.Ceiling(support) * 2 + 1;
        var bounds = new int[outSize * 2];
        var weights = new int[outSize * ksize];
        var k = new double[ksize];
        double invFilterScale = 1.0 / filterScale;
        for (int xx = 0; xx < outSize; xx++)
        {
            double center = (xx + 0.5) * scale;
            double ww = 0.0;
            int xmin = (int)(center - support + 0.5);
            if (xmin < 0) xmin = 0;
            int xmax = (int)(center + support + 0.5);
            if (xmax > inSize) xmax = inSize;
            xmax -= xmin;
            for (int x = 0; x < xmax; x++)
            {
                double w = Lanczos((x + xmin - center + 0.5) * invFilterScale);
                k[x] = w;
                ww += w;
            }
            for (int x = 0; x < xmax; x++)
            {
                double v = ww != 0.0 ? k[x] / ww : k[x];
                weights[xx * ksize + x] = v < 0 ? (int)(-0.5 + v * (1 << PrecisionBits)) : (int)(0.5 + v * (1 << PrecisionBits));
            }
            bounds[xx * 2] = xmin;
            bounds[xx * 2 + 1] = Math.Max(xmax, 0);
        }
        return (bounds, weights, ksize);
    }

    /// <summary>Pillow's clip8: the accumulator shifted down by the precision, clamped to a byte.</summary>
    static byte Clip8(int v)
    {
        int s = v >> PrecisionBits;
        return (byte)(s < 0 ? 0 : s > 255 ? 255 : s);
    }

    static byte[] Horizontal(byte[] src, int w, int h, int c, int newW)
    {
        var (bounds, weights, ksize) = Coefficients(w, newW);
        var dst = new byte[newW * h * c];
        for (int y = 0; y < h; y++)
            for (int xx = 0; xx < newW; xx++)
            {
                int xmin = bounds[xx * 2], count = bounds[xx * 2 + 1], kOff = xx * ksize;
                for (int b = 0; b < c; b++)
                {
                    int ss = 1 << (PrecisionBits - 1);
                    for (int x = 0; x < count; x++) ss += src[(y * w + x + xmin) * c + b] * weights[kOff + x];
                    dst[(y * newW + xx) * c + b] = Clip8(ss);
                }
            }
        return dst;
    }

    static byte[] Vertical(byte[] src, int w, int h, int c, int newH)
    {
        var (bounds, weights, ksize) = Coefficients(h, newH);
        var dst = new byte[w * newH * c];
        for (int yy = 0; yy < newH; yy++)
        {
            int ymin = bounds[yy * 2], count = bounds[yy * 2 + 1], kOff = yy * ksize;
            for (int x = 0; x < w; x++)
                for (int b = 0; b < c; b++)
                {
                    int ss = 1 << (PrecisionBits - 1);
                    for (int y = 0; y < count; y++) ss += src[((y + ymin) * w + x) * c + b] * weights[kOff + y];
                    dst[(yy * w + x) * c + b] = Clip8(ss);
                }
        }
        return dst;
    }
}
