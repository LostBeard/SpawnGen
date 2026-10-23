using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

[Node(Category = "image/color", DisplayName = "Image RGB to YUV", OutputNames = ["Y", "U", "V"])]
public sealed class ImageRGBToYUV
{
    /// <summary>kornia's <c>rgb_to_ycbcr</c>: Y = 0.299 R + 0.587 G + 0.114 B, U = (B - Y) 0.564 + 0.5, V = (R - Y) 0.713 + 0.5; each as a grey picture.</summary>
    public (ImageBatch Y, ImageBatch U, ImageBatch V) Execute(ImageBatch image)
    {
        int c = image.Channels;
        if (c < 3) throw new ArgumentException("RGB to YUV needs a colour picture (3 channels or more).");
        var ys = new ImageBatch(image.Batch, image.Height, image.Width, 3);
        var us = new ImageBatch(image.Batch, image.Height, image.Width, 3);
        var vs = new ImageBatch(image.Batch, image.Height, image.Width, 3);
        for (int p = 0; p < image.Batch * image.Height * image.Width; p++)
        {
            float r = image.Data[p * c], g = image.Data[p * c + 1], b = image.Data[p * c + 2];
            float y = 0.299f * r + 0.587f * g + 0.114f * b;
            float u = (b - y) * 0.564f + 0.5f, v = (r - y) * 0.713f + 0.5f;
            for (int k = 0; k < 3; k++) { ys.Data[p * 3 + k] = y; us.Data[p * 3 + k] = u; vs.Data[p * 3 + k] = v; }
        }
        return (ys, us, vs);
    }
}

[Node(Category = "image/color", DisplayName = "Image YUV to RGB")]
public sealed class ImageYUVToRGB
{
    /// <summary>
    /// kornia's <c>ycbcr_to_rgb</c> of the mean of the first three channels of each input: R = Y + 1.403 (V - 0.5), G = Y - 0.714 (V - 0.5) - 0.344 (U - 0.5),
    /// B = Y + 1.773 (U - 0.5), clamped to 0..1 (kornia 0.8 clamps).
    /// </summary>
    public ImageBatch Execute([Input(Name = "Y")] ImageBatch y, [Input(Name = "U")] ImageBatch u, [Input(Name = "V")] ImageBatch v)
    {
        if (u.Batch != y.Batch || v.Batch != y.Batch || u.Height != y.Height || v.Height != y.Height || u.Width != y.Width || v.Width != y.Width)
            throw new ArgumentException("Y, U and V must have the same batch size and picture size.");
        var result = new ImageBatch(y.Batch, y.Height, y.Width, 3);
        static float Mean(ImageBatch img, int p)
        {
            int c = img.Channels, n = Math.Min(c, 3);
            float sum = 0;
            for (int k = 0; k < n; k++) sum += img.Data[p * c + k];
            return sum / n;
        }
        for (int p = 0; p < y.Batch * y.Height * y.Width; p++)
        {
            float yy = Mean(y, p), cb = Mean(u, p) - 0.5f, cr = Mean(v, p) - 0.5f;
            result.Data[p * 3] = Math.Clamp(yy + 1.403f * cr, 0f, 1f);
            result.Data[p * 3 + 1] = Math.Clamp(yy - 0.714f * cr - 0.344f * cb, 0f, 1f);
            result.Data[p * 3 + 2] = Math.Clamp(yy + 1.773f * cb, 0f, 1f);
        }
        return result;
    }
}

[Node(Category = "image/color", DisplayName = "Convert Image Color Space",
    Description = "Convert sRGB, linear Rec.709, HDR (Rec.2020 HLG), HDR PQ (Rec.2020 PQ), HDR LogC3, and HDR ACEScct. LogC3 uses the EI 800 curve with Rec.709 primaries and codes clamped to [0, 1]. ACEScct uses AP1 primaries and D60 white, with Bradford adaptation to D65. Convert LogC3 or ACEScct to linear for EXR saving. Linear 1.0 uses the same 203-nit reference white as sRGB; HLG uses a 1000-nit reference display. Linear and ACEScct outputs preserve extended values. Linear-to-HDR conversions preserve highlights without tone mapping; HLG and PQ outputs clip negative channels. SDR output and PQ-to-HLG conversion tone-map excess luminance across the batch and compress out-of-gamut colors. Conversions compute in float32 and return the intermediate device and dtype. Straight alpha is not color-transformed.")]
public sealed class ImageColorSpace
{
    public static string[] Spaces() => ["sRGB", "HDR", "HDR PQ", "linear", "HDR LogC3", "HDR ACEScct"];

    /// <summary>
    /// ComfyUI's colour space conversion through display-linear Rec.2020 in cd/m² (BT.2100): sRGB, linear Rec.709, HDR (Rec.2020 HLG, 1000-nit display),
    /// HDR PQ, LogC3 (EI 800, Rec.709 primaries) and ACEScct. Linear 1.0 is the 203-nit reference white. SDR output, and PQ to HLG, tone-map
    /// excess luminance with one white point for the whole batch and compress out-of-gamut colours. A fourth (alpha) channel is passed through.
    /// </summary>
    public ImageBatch Execute(ImageBatch image,
        [Input(OptionsProvider = nameof(Spaces), Tooltip = "Color space of the input pixels.")] string source = "sRGB",
        [Input(OptionsProvider = nameof(Spaces), Tooltip = "Color space of the output pixels. Set the save node to this same color space.")] string destination = "sRGB")
    {
        if (!Spaces().Contains(source)) throw new ArgumentException($"Unsupported source color space: {source}", nameof(source));
        if (!Spaces().Contains(destination)) throw new ArgumentException($"Unsupported destination color space: {destination}", nameof(destination));
        if (source == destination) return image;
        int c = image.Channels, n = image.Batch * image.Height * image.Width;
        if (c < 3) throw new ArgumentException("Color space conversion needs a colour picture (3 channels or more).");
        var rgb = new float[n * 3];
        for (int p = 0; p < n; p++) for (int k = 0; k < 3; k++) rgb[p * 3 + k] = image.Data[p * c + k];

        ToNits(rgb, source);
        FromNits(rgb, source, destination);

        int oc = c == 4 ? 4 : 3;
        var result = new ImageBatch(image.Batch, image.Height, image.Width, oc);
        for (int p = 0; p < n; p++)
        {
            for (int k = 0; k < 3; k++) result.Data[p * oc + k] = rgb[p * 3 + k];
            if (oc == 4) result.Data[p * 4 + 3] = image.Data[p * 4 + 3];
        }
        return result;
    }

    // ---- constants, as in nodes_images.py (used in float32 like the torch code) ----

    static readonly float[,] Rec709ToRec2020 = M(
        0.6274038959346991, 0.3292830383778837, 0.0433130656874172,
        0.0690972893582320, 0.9195403950754587, 0.0113623155663092,
        0.0163914388751503, 0.0880133078772259, 0.8955952532476238);
    static readonly float[,] Rec2020ToRec709 = M(
        1.6604910021084338, -0.5876411387885494, -0.0728498633198846,
        -0.1245504745215905, 1.1328998971259600, -0.0083494226043695,
        -0.0181507633549053, -0.1005788980080076, 1.1187296613629125);
    static readonly float[,] AcesCgToRec2020 = M(
        1.0258247476660107, -0.0200531908382148, -0.0057715568277955,
        -0.0022343695199762, 1.0045865018884792, -0.0023521323685036,
        -0.0050133514680893, -0.0252900718107852, 1.0303034232788744);
    static readonly float[,] Rec2020ToAcesCg = M(
        0.9748949779244186, 0.0195991086370050, 0.0055059134385761,
        0.0021795627977041, 0.9955354688932213, 0.0022849683090752,
        0.0047972396837727, 0.0245320166345895, 0.9706707436816380);
    static readonly float[] Rec709Luma = [0.2126390058715103f, 0.7151686787677559f, 0.0721923153607337f];
    static readonly float[] Rec2020Luma = [0.2627f, 0.6780f, 0.0593f];

    const double PqM1 = 2610.0 / 16384, PqM2 = 2523.0 / 32, PqC1 = 3424.0 / 4096, PqC2 = 2413.0 / 128, PqC3 = 2392.0 / 128;
    const float SdrWhite = 203f, HlgPeak = 1000f;
    const double HlgGamma = 1.2;
    const double HlgA = 0.17883277, HlgB = 0.28466892, HlgC = 0.55991072928;
    const double LogC3A = 5.555556, LogC3B = 0.052272, LogC3C = 0.247190, LogC3D = 0.385537, LogC3E = 5.367655, LogC3F = 0.092809, LogC3Cut = 0.010591;
    const double AcesSlope = 10.5402377416545, AcesOffset = 0.0729055341958355, AcesLogScale = 17.52, AcesLogOffset = 9.72, AcesCut = 0.0078125, AcesLogCut = 0.155251141552511;

    static float[,] M(params double[] v)
    {
        var m = new float[3, 3];
        for (int i = 0; i < 9; i++) m[i / 3, i % 3] = (float)v[i];
        return m;
    }

    static void Primaries(float[] rgb, float[,] m, float divide = 1f, float multiply = 1f)
    {
        for (int p = 0; p < rgb.Length; p += 3)
        {
            float r = rgb[p], g = rgb[p + 1], b = rgb[p + 2];
            if (divide != 1f) { r /= divide; g /= divide; b /= divide; }
            for (int k = 0; k < 3; k++) rgb[p + k] = (r * m[k, 0] + g * m[k, 1] + b * m[k, 2]) * multiply;
        }
    }

    static float Luma(float[] rgb, int p, float[] wts) => rgb[p] * wts[0] + rgb[p + 1] * wts[1] + rgb[p + 2] * wts[2];

    static void Each(float[] rgb, Func<float, float> f) { for (int i = 0; i < rgb.Length; i++) rgb[i] = f(rgb[i]); }

    // Accurate log1p / expm1 (the PQ code relies on them to avoid cancellation around 1).
    static double Log1p(double x) { double u = 1 + x; return u == 1 ? x : Math.Log(u) * x / (u - 1); }
    static double Expm1(double x)
    {
        double u = Math.Exp(x);
        if (u == 1) return x;
        double um1 = u - 1;
        return um1 == -1 ? -1 : um1 * x / Math.Log(u);
    }

    static float SrgbToLinear(float t) => t <= 0.04045f ? t / 12.92f : MathF.Pow((Math.Max(t, 0f) + 0.055f) / 1.055f, 2.4f);

    static float HlgToLinear(float t) => t <= 0.5f ? t * t / 3f : (MathF.Exp((Math.Max(t, 0.5f) - (float)HlgC) / (float)HlgA) + (float)HlgB) / 12f;

    /// <summary>Extended Reinhard on luminance, one white point (the brightest pixel of the batch); off when that peak is at SDR white.</summary>
    static void ToneMap(float[] rgb, float[] wts)
    {
        float peak = 0f;
        for (int p = 0; p < rgb.Length; p += 3) peak = Math.Max(peak, Math.Max(Luma(rgb, p, wts), 0f));
        peak = Math.Max(peak, 1f);
        if (!(peak > 1.0001f)) return;
        for (int p = 0; p < rgb.Length; p += 3)
        {
            float l = Math.Max(Luma(rgb, p, wts), 0f);
            float scale = (1f + l / (peak * peak)) / (1f + l);
            for (int k = 0; k < 3; k++) rgb[p + k] *= scale;
        }
    }

    /// <summary>Out-of-gamut colours are desaturated towards their luminance just enough to fit; everything is clamped to 0..1.</summary>
    static void CompressGamut(float[] rgb, float[] wts)
    {
        const float tiny = 1.17549435e-38f; // torch.finfo(float32).tiny
        for (int p = 0; p < rgb.Length; p += 3)
        {
            float l = Math.Clamp(Luma(rgb, p, wts), 0f, 1f);
            float mn = Math.Min(Math.Min(rgb[p], rgb[p + 1]), rgb[p + 2]), mx = Math.Max(Math.Max(rgb[p], rgb[p + 1]), rgb[p + 2]);
            float upper = (1f - l) / Math.Max(mx - l, tiny), lower = l / Math.Max(l - mn, tiny);
            float saturation = Math.Clamp(Math.Min(upper, lower), 0f, 1f);
            bool inGamut = mn >= -1e-5f && mx <= 1.00001f;
            for (int k = 0; k < 3; k++) rgb[p + k] = Math.Clamp(inGamut ? rgb[p + k] : l + (rgb[p + k] - l) * saturation, 0f, 1f);
        }
    }

    /// <summary>To display-linear Rec.2020 in cd/m² (BT.2100 EOTFs).</summary>
    static void ToNits(float[] rgb, string source)
    {
        switch (source)
        {
            case "sRGB":
                Each(rgb, SrgbToLinear);
                Primaries(rgb, Rec709ToRec2020, multiply: SdrWhite);
                break;
            case "linear":
                Primaries(rgb, Rec709ToRec2020, multiply: SdrWhite);
                break;
            case "HDR LogC3":
                float logCut = (float)(LogC3E * LogC3Cut + LogC3F);
                Each(rgb, t =>
                {
                    t = Math.Clamp(t, 0f, 1f);
                    float v = t >= logCut ? (MathF.Pow(10f, (t - (float)LogC3D) / (float)LogC3C) - (float)LogC3B) / (float)LogC3A : (t - (float)LogC3F) / (float)LogC3E;
                    return Math.Max(v, 0f);
                });
                Primaries(rgb, Rec709ToRec2020, multiply: SdrWhite);
                break;
            case "HDR ACEScct":
                Each(rgb, t => t > (float)AcesLogCut ? MathF.Pow(2f, t * (float)AcesLogScale - (float)AcesLogOffset) : (t - (float)AcesOffset) / (float)AcesSlope);
                Primaries(rgb, AcesCgToRec2020, multiply: SdrWhite);
                break;
            case "HDR":
                Each(rgb, HlgToLinear);
                for (int p = 0; p < rgb.Length; p += 3)
                {
                    float l = Math.Max(Luma(rgb, p, Rec2020Luma), 0f);
                    float gain = MathF.Pow(l, (float)(HlgGamma - 1.0)) * HlgPeak;
                    for (int k = 0; k < 3; k++) rgb[p + k] *= gain;
                }
                break;
            case "HDR PQ":
                // Evaluated around 1 to avoid cancellation, like ComfyUI.
                Each(rgb, t =>
                {
                    float pp = (float)Expm1(MathF.Log(Math.Max(t, 0f)) / (float)PqM2);
                    return MathF.Pow(Math.Max(pp + (float)(1.0 - PqC1), 0f) / ((float)(PqC2 - PqC3) - (float)PqC3 * pp), (float)(1.0 / PqM1)) * 10000f;
                });
                break;
        }
    }

    static void FromNits(float[] rgb, string source, string destination)
    {
        switch (destination)
        {
            case "linear":
                Primaries(rgb, Rec2020ToRec709, divide: SdrWhite);
                break;
            case "HDR LogC3":
                Primaries(rgb, Rec2020ToRec709, divide: SdrWhite);
                Each(rgb, t =>
                {
                    t = Math.Max(t, 0f);
                    float v = t >= (float)LogC3Cut ? (float)LogC3C * MathF.Log10((float)LogC3A * t + (float)LogC3B) + (float)LogC3D : (float)LogC3E * t + (float)LogC3F;
                    return Math.Clamp(v, 0f, 1f);
                });
                break;
            case "HDR ACEScct":
                Primaries(rgb, Rec2020ToAcesCg, divide: SdrWhite);
                Each(rgb, t => t > (float)AcesCut ? (MathF.Log2(Math.Max(t, (float)AcesCut)) + (float)AcesLogOffset) / (float)AcesLogScale : (float)AcesSlope * t + (float)AcesOffset);
                break;
            case "sRGB":
                Primaries(rgb, Rec2020ToRec709, divide: SdrWhite);
                ToneMap(rgb, Rec709Luma);
                CompressGamut(rgb, Rec709Luma);
                Each(rgb, t => t <= 0.0031308f ? t * 12.92f : 1.055f * MathF.Pow(t, (float)(1.0 / 2.4)) - 0.055f);
                break;
            case "HDR":
                // Negative channels are clipped before luminance so out-of-gamut shadows are not amplified.
                Each(rgb, t => Math.Max(t / HlgPeak, 0f));
                if (source == "HDR PQ") ToneMap(rgb, Rec2020Luma);
                for (int p = 0; p < rgb.Length; p += 3)
                {
                    float l = Math.Max(Luma(rgb, p, Rec2020Luma), 1.17549435e-38f);
                    float gain = MathF.Pow(l, (float)(1.0 / HlgGamma - 1.0));
                    for (int k = 0; k < 3; k++) rgb[p + k] *= gain;
                }
                if (source == "HDR PQ") CompressGamut(rgb, Rec2020Luma);
                float twelfth = (float)(1.0 / 12.0);
                Each(rgb, t => t <= twelfth ? MathF.Sqrt(3f * Math.Max(t, 0f)) : (float)HlgA * MathF.Log(12f * Math.Max(t, twelfth) - (float)HlgB) + (float)HlgC);
                break;
            case "HDR PQ":
                Each(rgb, t =>
                {
                    float pp = MathF.Pow(Math.Max(t, 0f) / 10000f, (float)PqM1);
                    pp = ((float)(PqC1 - 1.0) + (float)(PqC2 - PqC3) * pp) / (1f + (float)PqC3 * pp);
                    return MathF.Exp((float)Log1p(pp) * (float)PqM2);
                });
                break;
        }
    }
}
