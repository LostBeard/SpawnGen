namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md) from black-box measurements of ComfyUI's compositor (2026-11-11, fixture Fixtures/bbox/compositor.json.gz):
// the separable modes follow the usual definitions (the W3C Compositing and Blending spec, the Photoshop / GIMP light modes) in the float32 order measured;
// hue / color are HLS and saturation HSV as Python's colorsys documents them; luminosity scales by a measured luma vector.

/// <summary>
/// The compositor's 26 blend modes and its compositing step. Canvas and layer colours are linear light (RGBA floats). "Linear" modes blend those
/// values directly; the others ("perceptual") blend the sRGB-encoded colours and decode the result. A layer mixes into the canvas by its alpha times its
/// opacity: normal is Porter-Duff "over" on straight colours; every other mode keeps the canvas alpha and changes nothing where the canvas is transparent.
/// </summary>
public static class CompositorBlend
{
    public static readonly string[] ModeNames =
    [
        "normal", "multiply", "screen", "overlay", "darken", "lighten", "color-dodge", "color-burn", "hard-light", "soft-light", "difference", "exclusion",
        "linear-dodge", "linear-burn", "vivid-light", "pin-light", "linear-light", "hard-mix", "subtract", "divide", "grain-extract", "grain-merge", "hue",
        "saturation", "color", "luminosity",
    ];

    static readonly HashSet<string> LinearModes = ["normal", "multiply", "darken", "lighten", "subtract", "divide", "linear-dodge", "luminosity"];

    /// <summary>Whether the mode blends linear values (true) or sRGB-encoded ones (false), and its index in <see cref="ModeNames"/>.</summary>
    public static (bool Linear, int Index) Resolve(string mode)
    {
        int index = Array.IndexOf(ModeNames, mode);
        if (index < 0) throw new ArgumentException($"Unknown blend mode '{mode}'.", nameof(mode));
        return (LinearModes.Contains(mode), index);
    }

    public static float SrgbToLinear(float x) => x <= 0.04045f ? x / 12.92f : MathF.Pow((x + 0.055f) / 1.055f, 2.4f);

    public static float LinearToSrgb(float x) => x <= 0.0031308f ? x * 12.92f : 1.055f * MathF.Pow(Math.Max(x, 0f), 1f / 2.4f) - 0.055f;

    static float Screen(float a, float b) => 1f - (1f - a) * (1f - b);

    static float Dodge(float a, float b) => Math.Clamp(a / Math.Max(1f - b, 1e-7f), 0f, 1f);

    static float Burn(float a, float b) => Math.Clamp(1f - (1f - a) / Math.Max(b, 1e-7f), 0f, 1f);

    /// <summary>The blend function B(base, layer) of one pixel (3 channels).</summary>
    public static void BlendPixel(string mode, ReadOnlySpan<float> i, ReadOnlySpan<float> l, Span<float> o)
    {
        switch (mode)
        {
            case "hue": NonSeparable(0, i, l, o); return;
            case "saturation": NonSeparable(1, i, l, o); return;
            case "color": NonSeparable(2, i, l, o); return;
            case "luminosity":
            {
                float yi = (i[0] * 0.22248840f + i[1] * 0.71690369f) + i[2] * 0.06060791f;
                float yl = (l[0] * 0.22248840f + l[1] * 0.71690369f) + l[2] * 0.06060791f;
                if (yi > 0f) { float k = yl / yi; for (int c = 0; c < 3; c++) o[c] = i[c] * k; }
                else for (int c = 0; c < 3; c++) o[c] = yl;
                return;
            }
        }
        for (int c = 0; c < 3; c++) o[c] = Separable(mode, i[c], l[c]);
    }

    static float Separable(string mode, float i, float l) => mode switch
    {
        "normal" => l,
        "multiply" => i * l,
        "screen" => Screen(i, l),
        "overlay" => i < 0.5f ? l * (2f * i) : Screen(l, 2f * i - 1f),
        "darken" => Math.Min(i, l),
        "lighten" => Math.Max(i, l),
        "color-dodge" => Dodge(i, l),
        "color-burn" => Burn(i, l),
        "hard-light" => Math.Min(l <= 0.5f ? i * (2f * l) : Screen(i, 2f * l - 1f), 1f),
        "soft-light" => (1f - i) * (i * l) + i * Screen(i, l),
        "difference" => Math.Abs(i - l),
        "exclusion" => 0.5f - 2f * (i - 0.5f) * (l - 0.5f),
        "linear-dodge" => i + l,
        "linear-burn" => i + l - 1f,
        "vivid-light" => l <= 0.5f ? Burn(i, 2f * l) : Dodge(i, 2f * (l - 0.5f)),
        "pin-light" => l <= 0.5f ? Math.Min(i, 2f * l) : Math.Max(i, 2f * l - 1f),
        "linear-light" => i + 2f * l - 1f,
        "hard-mix" => i + l >= 1f ? 1f : 0f,
        "subtract" => Math.Max(i - l, 0f),
        "divide" => Math.Clamp(i / Math.Max(l, 1e-6f), 0f, 1f),
        "grain-extract" => i - l + 0.5f,
        "grain-merge" => i + l - 0.5f,
        _ => throw new ArgumentException($"Unknown blend mode '{mode}'.", nameof(mode)),
    };

    /// <summary>hue (0), saturation (1) and color (2), in double precision.</summary>
    static void NonSeparable(int which, ReadOnlySpan<float> i, ReadOnlySpan<float> l, Span<float> o)
    {
        double[] bi = [i[0], i[1], i[2]], bl = [l[0], l[1], l[2]];
        double[] r;
        switch (which)
        {
            case 0:
            {
                var (hl, _, sl) = ToHls(bl);
                var (_, li, si) = ToHls(bi);
                r = sl == 0 ? bi : FromHls(hl, li, si);
                break;
            }
            case 1:
            {
                var (hi, si, vi) = ToHsv(bi);
                var (_, sl, _) = ToHsv(bl);
                r = si <= 1e-6 ? [vi, vi, vi] : FromHsv(hi, sl, vi);
                break;
            }
            default:
            {
                var (hl, _, sl) = ToHls(bl);
                var (_, li, _) = ToHls(bi);
                r = FromHls(hl, li, sl);
                break;
            }
        }
        for (int c = 0; c < 3; c++) o[c] = (float)r[c];
    }

    static double PyMod1(double x) => x - Math.Floor(x);

    static (double H, double L, double S) ToHls(double[] c)
    {
        double r = c[0], g = c[1], b = c[2], max = Math.Max(Math.Max(r, g), b), min = Math.Min(Math.Min(r, g), b), l = (min + max) / 2;
        if (min == max) return (0, l, 0);
        double d = max - min, s = l <= 0.5 ? d / (max + min) : d / (2 - max - min);
        double rc = (max - r) / d, gc = (max - g) / d, bc = (max - b) / d;
        double h = r == max ? bc - gc : g == max ? 2 + rc - bc : 4 + gc - rc;
        return (PyMod1(h / 6), l, s);
    }

    static double[] FromHls(double h, double l, double s)
    {
        if (s == 0) return [l, l, l];
        double m2 = l <= 0.5 ? l * (1 + s) : l + s - l * s, m1 = 2 * l - m2;
        double V(double hue)
        {
            hue = PyMod1(hue);
            if (hue < 1.0 / 6) return m1 + (m2 - m1) * hue * 6;
            if (hue < 0.5) return m2;
            if (hue < 2.0 / 3) return m1 + (m2 - m1) * (2.0 / 3 - hue) * 6;
            return m1;
        }
        return [V(h + 1.0 / 3), V(h), V(h - 1.0 / 3)];
    }

    static (double H, double S, double V) ToHsv(double[] c)
    {
        double r = c[0], g = c[1], b = c[2], max = Math.Max(Math.Max(r, g), b), min = Math.Min(Math.Min(r, g), b);
        if (min == max) return (0, 0, max);
        double d = max - min, s = d / max;
        double rc = (max - r) / d, gc = (max - g) / d, bc = (max - b) / d;
        double h = r == max ? bc - gc : g == max ? 2 + rc - bc : 4 + gc - rc;
        return (PyMod1(h / 6), s, max);
    }

    static double[] FromHsv(double h, double s, double v)
    {
        if (s == 0) return [v, v, v];
        double sector = Math.Floor(h * 6), f = h * 6 - sector;
        double p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
        return (((int)sector % 6 + 6) % 6) switch
        {
            0 => new[] { v, t, p },
            1 => new[] { q, v, p },
            2 => new[] { p, v, t },
            3 => new[] { p, q, v },
            4 => new[] { t, p, v },
            _ => new[] { v, p, q },
        };
    }

    /// <summary>
    /// Composites a <paramref name="width"/> x <paramref name="height"/> block of <paramref name="layer"/> (RGBA, linear) onto <paramref name="canvas"/>
    /// (RGBA, linear) in place. Offsets are in pixels; a stride of 0 means <paramref name="width"/>.
    /// </summary>
    public static void Composite(string mode, float[] canvas, int canvasOffset, int canvasStride, float[] layer, int layerOffset, int layerStride, int width,
        int height, double opacity)
    {
        var (linear, _) = Resolve(mode);
        if (canvasStride == 0) canvasStride = width;
        if (layerStride == 0) layerStride = width;
        float op = (float)opacity;
        Span<float> cb = stackalloc float[3], cs = stackalloc float[3], bl = stackalloc float[3];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int p = (canvasOffset + y * canvasStride + x) * 4, q = (layerOffset + y * layerStride + x) * 4;
                float ab = canvas[p + 3], aS = layer[q + 3] * op;
                if (mode == "normal")
                {
                    float ao = aS + ab * (1f - aS);
                    if (ao > 0f)
                    {
                        float t = aS / ao;
                        for (int c = 0; c < 3; c++) canvas[p + c] = ab > 0f ? canvas[p + c] + (layer[q + c] - canvas[p + c]) * t : layer[q + c];
                    }
                    canvas[p + 3] = ao;
                    continue;
                }
                float w = ab > 0f ? aS : 0f;
                for (int c = 0; c < 3; c++) { cb[c] = canvas[p + c]; cs[c] = layer[q + c]; }
                if (linear) BlendPixel(mode, cb, cs, bl);
                else
                {
                    Span<float> eb = stackalloc float[3], es = stackalloc float[3];
                    for (int c = 0; c < 3; c++) { eb[c] = LinearToSrgb(cb[c]); es[c] = LinearToSrgb(cs[c]); }
                    BlendPixel(mode, eb, es, bl);
                    for (int c = 0; c < 3; c++) bl[c] = SrgbToLinear(bl[c]);
                }
                for (int c = 0; c < 3; c++) canvas[p + c] = cb[c] * (1f - w) + bl[c] * w;
            }
    }
}
