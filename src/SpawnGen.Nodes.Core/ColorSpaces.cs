namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md) from the published standards cited below, ComfyUI's node description of Convert Image Color Space, and
// fixtures made by running ComfyUI.

/// <summary>
/// The colour spaces of Convert Image Color Space and the conversions between them. Every pixel is computed in double precision and rounded to float32
/// once at the end. ComfyUI's results agree with the exact values to a few float32 ulps; its own rounding order is not reproduced.
/// </summary>
public static class ColorSpaces
{
    public enum Space { Srgb, Hlg, Pq, Linear, LogC3, AcesCct }

    public static Space Parse(string name, string parameter) => name switch
    {
        "sRGB" => Space.Srgb,
        "HDR" => Space.Hlg,
        "HDR PQ" => Space.Pq,
        "linear" => Space.Linear,
        "HDR LogC3" => Space.LogC3,
        "HDR ACEScct" => Space.AcesCct,
        _ => throw new ArgumentException($"Unknown color space '{name}'.", parameter),
    };

    /// <summary>
    /// Converts every pixel of the whole batch; the working space is linear Rec.709 with 1.0 = 203 nits (sRGB white). SDR output (sRGB) is first fitted
    /// into 0..1 by <see cref="FitToSdr"/>.
    /// </summary>
    public static ImageBatchData Convert(float[] data, int pixels, int channels, Space from, Space to)
    {
        if (channels < 3) throw new ArgumentException("Convert Image Color Space needs RGB or RGBA pictures.");
        if (from == Space.Pq && to == Space.Hlg) return PqToHlg(data, pixels, channels);
        var linear = new double[(long)pixels * 3];
        for (int i = 0; i < pixels; i++)
        {
            long p = (long)i * channels;
            var px = linear.AsSpan(i * 3, 3);
            px[0] = data[p]; px[1] = data[p + 1]; px[2] = data[p + 2];
            ToLinear709(px, from);
        }
        if (to == Space.Srgb) FitToSdr(linear);
        var result = (float[])data.Clone();
        for (int i = 0; i < pixels; i++)
        {
            long p = (long)i * channels;
            var px = linear.AsSpan(i * 3, 3);
            FromLinear709(px, to);
            result[p] = (float)px[0]; result[p + 1] = (float)px[1]; result[p + 2] = (float)px[2];
        }
        return new ImageBatchData(result);
    }

    /// <summary>
    /// Fits linear Rec.709 pixels into the SDR range 0..1, as measured with ComfyUI (2026-11-08, linear to sRGB on chosen pixels):
    /// <list type="number">
    /// <item>W is the highest Rec.709 luminance Y of the whole batch. When W &gt; 1, every pixel is scaled by Y' / Y with the extended Reinhard curve
    /// Y' = Y (1 + Y / W²) / (1 + Y) (Reinhard et al. 2002, eq. 4), which takes W to exactly 1 (measured: batch peak 4 gives 0.5 -> 0.34375, 1 -> 0.53125,
    /// 2 -> 0.75; a batch peaking at 0.8 is unchanged).</item>
    /// <item>A pixel with a channel below 0 or above 1 is then moved toward the grey of its own luminance, just far enough that all channels fit; its
    /// luminance is kept (measured: (-0.2, 0.5, 0.5) -> (0, 0.44599, 0.44599)).</item>
    /// </list>
    /// </summary>
    static void FitToSdr(double[] linear)
    {
        int n = linear.Length / 3;
        double peak = double.NegativeInfinity;
        for (int i = 0; i < n; i++) peak = Math.Max(peak, Luminance709(linear.AsSpan(i * 3, 3)));
        for (int i = 0; i < n; i++)
        {
            var px = linear.AsSpan(i * 3, 3);
            double y = Luminance709(px);
            if (peak > 1 && y > 0)
            {
                double mapped = y * (1 + y / (peak * peak)) / (1 + y);
                for (int k = 0; k < 3; k++) px[k] *= mapped / y;
                y = mapped;
            }
            double lo = Math.Min(px[0], Math.Min(px[1], px[2])), hi = Math.Max(px[0], Math.Max(px[1], px[2]));
            if (lo >= 0 && hi <= 1) continue;
            double t = 1;
            if (lo < 0) t = Math.Min(t, y / (y - lo));
            if (hi > 1) t = Math.Min(t, (1 - y) / (hi - y));
            t = Math.Clamp(t, 0, 1);
            for (int k = 0; k < 3; k++) px[k] = Math.Clamp(y + t * (px[k] - y), 0, 1);
        }
    }

    /// <summary>
    /// PQ to HLG, both Rec.2020, as measured with ComfyUI (2026-11-08, PQ to HDR on chosen pixels): PQ display light is taken in units of the HLG
    /// display peak (1000 nits); when the batch's highest Rec.2020 luminance W exceeds 1 every pixel is scaled by the extended Reinhard curve of
    /// <see cref="FitToSdr"/> (measured: peak 4000 nits gives 500 -> 343.75, 1000 -> 531.25, 2000 -> 750, 4000 -> 1000 nits). The BT.2100 inverse OOTF then
    /// gives scene light, and a pixel with a channel above 1 is moved toward the grey of its scene luminance until it fits (measured: always, also when
    /// no tone mapping was needed: 1000-nit red gives scene (1, 0.0889, 0.0889)). Last comes the HLG OETF.
    /// </summary>
    static ImageBatchData PqToHlg(float[] data, int pixels, int channels)
    {
        var light = new double[(long)pixels * 3];
        double peak = double.NegativeInfinity;
        for (int i = 0; i < pixels; i++)
        {
            for (int k = 0; k < 3; k++) light[i * 3 + k] = PqDecode(data[(long)i * channels + k]) * (10000 / HlgDisplayNits);
            peak = Math.Max(peak, Luminance2020(light.AsSpan(i * 3, 3)));
        }
        var result = (float[])data.Clone();
        for (int i = 0; i < pixels; i++)
        {
            var px = light.AsSpan(i * 3, 3);
            double yd = Luminance2020(px);
            if (peak > 1 && yd > 0)
            {
                double mapped = yd * (1 + yd / (peak * peak)) / (1 + yd);
                for (int k = 0; k < 3; k++) px[k] *= mapped / yd;
                yd = mapped;
            }
            double ys = yd > 0 ? Math.Pow(yd, 1 / HlgGamma) : 0;
            double gain = ys > 0 ? Math.Pow(ys, 1 - HlgGamma) : 0;
            for (int k = 0; k < 3; k++) px[k] *= gain;
            double hi = Math.Max(px[0], Math.Max(px[1], px[2]));
            if (hi > 1)
            {
                double t = Math.Clamp((1 - ys) / (hi - ys), 0, 1);
                for (int k = 0; k < 3; k++) px[k] = Math.Min(ys + t * (px[k] - ys), 1);
            }
            long p = (long)i * channels;
            for (int k = 0; k < 3; k++) result[p + k] = (float)HlgOetf(px[k]);
        }
        return new ImageBatchData(result);
    }

    static double Luminance2020(ReadOnlySpan<double> px) => Y2020R * px[0] + Y2020G * px[1] + Y2020B * px[2];

    /// <summary>Rec.709 relative luminance: the Y row of the RGB to XYZ matrix.</summary>
    static double Luminance709(ReadOnlySpan<double> px) => Rec709ToXyz[3] * px[0] + Rec709ToXyz[4] * px[1] + Rec709ToXyz[5] * px[2];

    /// <summary>Converted pixel values, same layout as the input.</summary>
    public readonly record struct ImageBatchData(float[] Data);

    // ---- Transfer functions ----

    /// <summary>IEC 61966-2-1 sRGB decoding (signal to linear), odd-symmetric for negative values.</summary>
    public static double SrgbToLinear(double v)
    {
        double a = Math.Abs(v);
        double l = a <= 0.04045 ? a / 12.92 : Math.Pow((a + 0.055) / 1.055, 2.4);
        return Math.CopySign(l, v);
    }

    /// <summary>IEC 61966-2-1 sRGB encoding (linear to signal), odd-symmetric for negative values.</summary>
    public static double LinearToSrgb(double v)
    {
        double a = Math.Abs(v);
        double s = a <= 0.0031308 ? a * 12.92 : 1.055 * Math.Pow(a, 1 / 2.4) - 0.055;
        return Math.CopySign(s, v);
    }

    // ITU-R BT.2100 HLG constants: a, b = 1 - 4a, c = 0.5 - a ln(4a).
    const double HlgA = 0.17883277, HlgB = 1 - 4 * HlgA;
    static readonly double HlgC = 0.5 - HlgA * Math.Log(4 * HlgA);

    /// <summary>BT.2100 HLG OETF: scene light 0..1 to signal.</summary>
    public static double HlgOetf(double e) => e <= 1.0 / 12 ? Math.Sqrt(3 * Math.Max(e, 0)) : HlgA * Math.Log(12 * e - HlgB) + HlgC;

    /// <summary>BT.2100 HLG inverse OETF: signal to scene light 0..1.</summary>
    public static double HlgInverseOetf(double s) => s <= 0.5 ? s * s / 3 : (Math.Exp((s - HlgC) / HlgA) + HlgB) / 12;

    // SMPTE ST 2084 (PQ) constants.
    const double PqM1 = 2610.0 / 16384, PqM2 = 2523.0 / 4096 * 128, PqC1 = 3424.0 / 4096, PqC2 = 2413.0 / 4096 * 32, PqC3 = 2392.0 / 4096 * 32;

    /// <summary>ST 2084 inverse EOTF: luminance / 10000 nits to signal.</summary>
    public static double PqEncode(double y)
    {
        double ym = Math.Pow(Math.Max(y, 0), PqM1);
        return Math.Pow((PqC1 + PqC2 * ym) / (1 + PqC3 * ym), PqM2);
    }

    /// <summary>ST 2084 EOTF: signal to luminance / 10000 nits.</summary>
    public static double PqDecode(double e)
    {
        double ep = Math.Pow(Math.Max(e, 0), 1 / PqM2);
        return Math.Pow(Math.Max(ep - PqC1, 0) / (PqC2 - PqC3 * ep), 1 / PqM1);
    }

    // ARRI LogC3, EI 800 (ARRI "ALEXA Log C Curve" white paper).
    const double LogCCut = 0.010591, LogCA = 5.555556, LogCB = 0.052272, LogCC = 0.247190, LogCD = 0.385537, LogCE = 5.367655, LogCF = 0.092809;

    /// <summary>LogC3 encoding. Negative light encodes as 0 (measured: such pixels come out as the curve black, 0.092809).</summary>
    public static double LogC3Encode(double x)
    {
        x = Math.Max(x, 0);
        return x > LogCCut ? LogCC * Math.Log10(LogCA * x + LogCB) + LogCD : LogCE * x + LogCF;
    }

    /// <summary>LogC3 decoding. Codes below the curve black give 0, not negative light (measured).</summary>
    public static double LogC3Decode(double t) =>
        Math.Max(t > LogCE * LogCCut + LogCF ? (Math.Pow(10, (t - LogCD) / LogCC) - LogCB) / LogCA : (t - LogCF) / LogCE, 0);

    // ACEScct (Academy S-2016-001).
    const double CctX = 0.0078125, CctA = 10.5402377416545, CctB = 0.0729055341958355, CctY = 0.155251141552511;

    public static double AcesCctEncode(double x) => x <= CctX ? CctA * x + CctB : (Math.Log2(x) + 9.72) / 17.52;

    public static double AcesCctDecode(double t) => t <= CctY ? (t - CctB) / CctA : Math.Pow(2, t * 17.52 - 9.72);

    // ---- Primaries ----

    const double ReferenceWhiteNits = 203, HlgDisplayNits = 1000, HlgGamma = 1.2;

    static readonly double[] Rec709ToXyz = RgbToXyz(0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290);
    static readonly double[] Rec2020ToXyz = RgbToXyz(0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290);
    static readonly double[] Ap1ToXyzD60 = RgbToXyz(0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767);

    static readonly double[] M709To2020 = Multiply(Invert(Rec2020ToXyz), Rec709ToXyz);
    static readonly double[] M2020To709 = Multiply(Invert(Rec709ToXyz), Rec2020ToXyz);
    static readonly double[] M709ToAp1 = Multiply(Invert(Ap1ToXyzD60), Multiply(Bradford(0.3127, 0.3290, 0.32168, 0.33767), Rec709ToXyz));
    static readonly double[] MAp1To709 = Multiply(Invert(Rec709ToXyz), Multiply(Bradford(0.32168, 0.33767, 0.3127, 0.3290), Ap1ToXyzD60));

    /// <summary>BT.2100 luminance weights of Rec.2020 RGB.</summary>
    const double Y2020R = 0.2627, Y2020G = 0.6780, Y2020B = 0.0593;

    static void ToLinear709(Span<double> px, Space from)
    {
        switch (from)
        {
            case Space.Linear:
                break;
            case Space.Srgb:
                for (int k = 0; k < 3; k++) px[k] = SrgbToLinear(px[k]);
                break;
            case Space.LogC3:
                for (int k = 0; k < 3; k++) px[k] = LogC3Decode(px[k]);
                break;
            case Space.AcesCct:
                for (int k = 0; k < 3; k++) px[k] = AcesCctDecode(px[k]);
                Apply(MAp1To709, px);
                break;
            case Space.Pq:
                for (int k = 0; k < 3; k++) px[k] = PqDecode(px[k]) * (10000 / ReferenceWhiteNits);
                Apply(M2020To709, px);
                break;
            case Space.Hlg:
            {
                for (int k = 0; k < 3; k++) px[k] = HlgInverseOetf(px[k]);
                // BT.2100 OOTF: display light = Lw * Ys^(gamma - 1) * Es.
                double ys = Y2020R * px[0] + Y2020G * px[1] + Y2020B * px[2];
                double gain = ys > 0 ? Math.Pow(ys, HlgGamma - 1) : 0;
                for (int k = 0; k < 3; k++) px[k] *= gain * (HlgDisplayNits / ReferenceWhiteNits);
                Apply(M2020To709, px);
                break;
            }
        }
    }

    static void FromLinear709(Span<double> px, Space to)
    {
        switch (to)
        {
            case Space.Linear:
                break;
            case Space.Srgb:
                for (int k = 0; k < 3; k++) px[k] = LinearToSrgb(px[k]);
                break;
            case Space.LogC3:
                for (int k = 0; k < 3; k++) px[k] = Math.Clamp(LogC3Encode(px[k]), 0, 1);
                break;
            case Space.AcesCct:
                Apply(M709ToAp1, px);
                for (int k = 0; k < 3; k++) px[k] = AcesCctEncode(px[k]);
                break;
            case Space.Pq:
                Apply(M709To2020, px);
                for (int k = 0; k < 3; k++) px[k] = PqEncode(Math.Max(px[k], 0) * (ReferenceWhiteNits / 10000));
                break;
            case Space.Hlg:
            {
                Apply(M709To2020, px);
                for (int k = 0; k < 3; k++) px[k] = Math.Max(px[k], 0) * (ReferenceWhiteNits / HlgDisplayNits);
                // BT.2100 inverse OOTF: Ys = (Yd / Lw)^(1 / gamma), Es = Fd / Lw / Ys^(gamma - 1).
                double yd = Y2020R * px[0] + Y2020G * px[1] + Y2020B * px[2];
                double gain = yd > 0 ? Math.Pow(yd, (1 - HlgGamma) / HlgGamma) : 0;
                for (int k = 0; k < 3; k++) px[k] = HlgOetf(px[k] * gain);
                break;
            }
        }
    }

    static void Apply(double[] m, Span<double> px)
    {
        double r = px[0], g = px[1], b = px[2];
        px[0] = m[0] * r + m[1] * g + m[2] * b;
        px[1] = m[3] * r + m[4] * g + m[5] * b;
        px[2] = m[6] * r + m[7] * g + m[8] * b;
    }

    // ---- 3x3 matrix helpers (row-major) ----

    /// <summary>RGB to XYZ for primaries and a white point given as xy chromaticities (SMPTE RP 177).</summary>
    static double[] RgbToXyz(double rx, double ry, double gx, double gy, double bx, double by, double wx, double wy)
    {
        double[] p = [rx / ry, gx / gy, bx / by, 1, 1, 1, (1 - rx - ry) / ry, (1 - gx - gy) / gy, (1 - bx - by) / by];
        var s = MultiplyVector(Invert(p), [wx / wy, 1, (1 - wx - wy) / wy]);
        return [p[0] * s[0], p[1] * s[1], p[2] * s[2], p[3] * s[0], p[4] * s[1], p[5] * s[2], p[6] * s[0], p[7] * s[1], p[8] * s[2]];
    }

    /// <summary>Bradford chromatic adaptation from one white point to another (xy chromaticities).</summary>
    static double[] Bradford(double fromX, double fromY, double toX, double toY)
    {
        double[] mb = [0.8951, 0.2664, -0.1614, -0.7502, 1.7135, 0.0367, 0.0389, -0.0685, 1.0296];
        double[] src = MultiplyVector(mb, [fromX / fromY, 1, (1 - fromX - fromY) / fromY]);
        double[] dst = MultiplyVector(mb, [toX / toY, 1, (1 - toX - toY) / toY]);
        double[] scale = [dst[0] / src[0], 0, 0, 0, dst[1] / src[1], 0, 0, 0, dst[2] / src[2]];
        return Multiply(Invert(mb), Multiply(scale, mb));
    }

    static double[] MultiplyVector(double[] m, double[] v) =>
        [m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2], m[6] * v[0] + m[7] * v[1] + m[8] * v[2]];

    static double[] Multiply(double[] a, double[] b)
    {
        var r = new double[9];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
        return r;
    }

    static double[] Invert(double[] m)
    {
        double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], i = m[8];
        double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        return
        [
            (e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det,
            (f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det,
            (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det,
        ];
    }
}
