using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// ComfyUI's Convert SDR to HDR (nodes_images.ImageSDRToHDR): sRGB brightness expanded into Rec.2020 HLG with ComfyUI's "lottes" curve, a Lottes-style
/// rational expansion blended half and half with a calibrated log-gain curve, optionally rolled off towards a peak-white target. Computed in float32
/// step by step like the torch code. A fourth (alpha) channel is passed through; a grey picture becomes RGB.
/// </summary>
[Node("ImageSDRToHDR", Category = "image/color", DisplayName = "Convert SDR to HDR",
    Description = "Expand sRGB brightness into Rec.2020 HLG HDR. Save the result with the HDR color space selected. Alpha is preserved.")]
public sealed class ImageSDRToHDR
{
    public static DynamicOption[] Algorithms() =>
    [
        new("lottes", DynamicInput.Float("target_luminance", 650.0, 0.0, 10000.0, 1.0,
            "Peak-white target in nits. 0 keeps the calibrated rendering. Dark images are not normalized to this peak.")),
    ];

    // The calibrated parameters of ComfyUI's _LOTTES_HDR_PARAMETERS: four for the rational curve, 17 log-gain knot steps, the blend.
    static readonly double[] LottesParameters =
    [
        -0.1517456472, -1.465687394, -0.2663205862, -1.11367178,
        -2.938027859, -2.948823929, -2.97376132, -2.978893757,
        -2.991618395, -2.956256151, -2.810376167, -2.666443586,
        -2.584222317, -2.532898188, -2.458465099, -2.309132099,
        -2.216364861, -2.167824268, -2.096168995, -2.062468767,
        -2.130614519, 0.5,
    ];

    static readonly double[] Rec709ToRec2020 =
    [
        0.6274038959346991, 0.3292830383778837, 0.0433130656874172,
        0.0690972893582320, 0.9195403950754587, 0.0113623155663092,
        0.0163914388751503, 0.0880133078772259, 0.8955952532476238,
    ];

    const double HlgA = 0.17883277, HlgB = 0.28466892, HlgC = 0.55991072928, SdrWhiteNits = 203.0, HlgPeakNits = 1000.0;

    public ImageBatch Execute([Input(Tooltip = "SDR image(s) in sRGB.")] ImageBatch image,
        [DynamicCombo(nameof(Algorithms)), Input(Tooltip = "Lottes-style rational expansion blended with a calibrated monotonic log-gain curve.")] DynamicChoice algorithm)
    {
        if (algorithm.Key != "lottes") throw new ArgumentException($"Unsupported HDR algorithm: {algorithm.Key}");
        double targetLuminance = algorithm.Get("target_luminance", 650.0); // 0 means none (Python's "or None")
        int c = image.Channels;
        if (c is not (1 or 3 or 4)) throw new ArgumentException($"SDR to HDR needs a grey, RGB or RGBA picture, not {c} channels.");
        int oc = c == 1 ? 3 : c, n = image.Batch * image.Height * image.Width;
        var lottes = new Lottes(targetLuminance);
        var result = new ImageBatch(image.Batch, image.Height, image.Width, oc);
        Span<float> rgb = stackalloc float[3];
        for (int p = 0; p < n; p++)
        {
            for (int k = 0; k < 3; k++) rgb[k] = image.Data[p * c + (c == 1 ? 0 : k)];
            lottes.Apply(rgb);
            for (int k = 0; k < 3; k++) result.Data[p * oc + k] = rgb[k];
            if (oc == 4) result.Data[p * 4 + 3] = image.Data[p * 4 + 3];
        }
        return result;
    }

    /// <summary>ComfyUI's _sdr_to_hdr_lottes for one pixel, every constant and step in float32 as torch evaluates it.</summary>
    sealed class Lottes
    {
        readonly float a, ad, invPeakMinusC, cc, blend, oneMinusBlend, rollOff, displayScale, gammaExponent;
        readonly bool hasTarget;
        readonly float[] knots = new float[17], m = new float[9];

        public Lottes(double targetLuminance)
        {
            var p = LottesParameters.Select(v => (float)v).ToArray();
            static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
            a = 0.7f + 0.6f * Sigmoid(p[0]);
            float d = 0.5f + 2f * Sigmoid(p[1]);
            float peak = 1f + 7f * Sigmoid(p[2]);
            cc = 0.5f + 2f * Sigmoid(p[3]);
            ad = a * d;
            invPeakMinusC = 1f / peak - cc;
            // softplus (beta 1, threshold 20), then a cumulative sum that torch accumulates in double and stores in float.
            double sum = 0;
            for (int i = 0; i < 17; i++)
            {
                float x = p[4 + i];
                sum += x > 20f ? x : (float)Math.Log(1.0 + MathF.Exp(x));
                knots[i] = (float)sum;
            }
            blend = Math.Clamp(p[21], 0f, 1f);
            oneMinusBlend = 1f - blend;
            for (int i = 0; i < 9; i++) m[i] = (float)Rec709ToRec2020[i];

            hasTarget = targetLuminance != 0;
            if (hasTarget)
            {
                float whiteGain = blend * Math.Clamp(MathF.Log2(peak), -1f, 3.5f) + oneMinusBlend * knots[16];
                float sourceWhite = Exp2(whiteGain);
                double targetWhite = targetLuminance / SdrWhiteNits;
                rollOff = (float)(1 / targetWhite) - 1f / sourceWhite;
            }
            // Inverse HLG OOTF for a display at least 1000 nits bright, with zero black.
            double peakNits = Math.Max(HlgPeakNits, hasTarget ? targetLuminance : 0.0);
            double gamma = 1.2 + 0.42 * Math.Log10(peakNits / 1000.0);
            displayScale = (float)(SdrWhiteNits / peakNits);
            gammaExponent = (float)(1 / gamma - 1);
        }

        static float Exp2(float x) => MathF.Pow(2f, x);

        static float SrgbToLinear(float t) => t <= 0.04045f ? t / 12.92f : MathF.Pow((Math.Max(t, 0f) + 0.055f) / 1.055f, 2.4f);

        public void Apply(Span<float> rgb)
        {
            float r = SrgbToLinear(Math.Clamp(rgb[0], 0f, 1f)), g = SrgbToLinear(Math.Clamp(rgb[1], 0f, 1f)), b = SrgbToLinear(Math.Clamp(rgb[2], 0f, 1f));
            float y = r * 0.2126f + g * 0.7152f + b * 0.0722f;
            float safeY = Math.Max(y, 1e-6f);
            float expanded = MathF.Pow(safeY, a) / Math.Max(invPeakMinusC * MathF.Pow(safeY, ad) + cc, 1e-5f);
            float rationalGain = Math.Clamp(MathF.Log2(Math.Max(expanded / safeY, 1e-3f)), -1f, 3.5f);
            float x = MathF.Sqrt(Math.Clamp(y, 0f, 1f)) * 16f;
            int index = Math.Min((int)x, 15);
            float curveGain = knots[index] + (x - index) * (knots[index + 1] - knots[index]);
            float gain = Exp2(blend * rationalGain + oneMinusBlend * curveGain);
            float h0 = (r * m[0] + g * m[1] + b * m[2]) * gain;
            float h1 = (r * m[3] + g * m[4] + b * m[5]) * gain;
            float h2 = (r * m[6] + g * m[7] + b * m[8]) * gain;
            if (hasTarget)
            {
                float yHdr = h0 * 0.2627f + h1 * 0.678f + h2 * 0.0593f;
                float den = Math.Max(1f + rollOff * yHdr, 1e-6f);
                h0 /= den; h1 /= den; h2 /= den;
            }
            float d0 = Math.Max(h0, 0f) * displayScale, d1 = Math.Max(h1, 0f) * displayScale, d2 = Math.Max(h2, 0f) * displayScale;
            float lum = d0 * 0.2627f + d1 * 0.678f + d2 * 0.0593f;
            float ootf = MathF.Pow(Math.Max(lum, 1e-12f), gammaExponent);
            rgb[0] = Oetf(Math.Clamp(d0 * ootf, 0f, 1f));
            rgb[1] = Oetf(Math.Clamp(d1 * ootf, 0f, 1f));
            rgb[2] = Oetf(Math.Clamp(d2 * ootf, 0f, 1f));
        }

        static float Oetf(float scene) => scene <= (float)(1.0 / 12) ? MathF.Sqrt(3f * scene) : (float)HlgA * MathF.Log(Math.Max(12f * scene - (float)HlgB, 1e-12f)) + (float)HlgC;
    }
}

/// <summary>
/// ComfyUI's Transfer Color (nodes_post_processing.ColorTransfer): matches the colours of the target pictures to the reference pictures. reinhard_lab
/// matches each CIE Lab channel's mean and spread; mkl_lab maps the Lab covariance with the Monge-Kantorovich linear transform; histogram matches each
/// channel's histogram (256 bins). per_frame matches every frame to its own reference frame (the last one when there are fewer; the pooled reference when
/// there is one), uniform pools the statistics of all target frames, target_frame uses one target frame's statistics for all. Lab is kornia's (D65).
/// </summary>
[Node("ColorTransfer", Category = "image/filters", DisplayName = "Transfer Color",
    Description = "Match the colors of one image to another using various algorithms.", OutputNames = ["image"])]
public sealed class ColorTransfer
{
    static readonly string[] Methods = ["reinhard_lab", "mkl_lab", "histogram"];
    const float Eps = 1e-6f;

    public static string[] MethodOptions() => Methods;

    public static DynamicOption[] SourceStats() =>
    [
        new("per_frame"),
        new("uniform"),
        new("target_frame", new DynamicInput("target_index", SocketType.Int, new WidgetSpec(WidgetKind.Int) { Default = 0L, Min = 0, Max = 10000 },
            Tooltip: "Frame index used as the source baseline for computing the transform to image_ref")),
    ];

    public ImageBatch Execute([Input(Tooltip = "Image(s) to apply the color transform to.")] ImageBatch imageTarget,
        [Input(Tooltip = "Reference image(s) to match colors to.")] ImageBatch imageRef,
        [Input(OptionsProvider = nameof(MethodOptions))] string method,
        [DynamicCombo(nameof(SourceStats)), Input(Tooltip = "per_frame: each frame matched to image_ref individually. uniform: pool stats across all source frames as baseline, match to image_ref. target_frame: use one chosen frame as the baseline for the transform to image_ref, applied uniformly to all frames (preserves relative differences)")] DynamicChoice sourceStats,
        [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double strength = 1.0)
    {
        string mode = sourceStats.Key;
        int targetIndex = sourceStats.Get("target_index", 0);
        if (strength == 0) return imageTarget;
        if (!Methods.Contains(method)) throw new ArgumentException($"Unknown method '{method}'.", nameof(method));
        if (mode is not ("per_frame" or "uniform" or "target_frame")) throw new ArgumentException($"Unknown source_stats '{mode}'.");
        if (imageRef.Channels != imageTarget.Channels)
            throw new ArgumentException($"image_target has {imageTarget.Channels} channels and image_ref {imageRef.Channels}: they must match.");
        return method == "histogram" ? Histogram(imageTarget, imageRef, mode, targetIndex, (float)strength)
            : LabTransfer(imageTarget, imageRef, mode, targetIndex, method == "reinhard_lab", (float)strength);
    }

    // ---- histogram ----

    /// <summary>The 256-bin index torch takes: (value * 255) truncated towards zero, clamped to 0..255.</summary>
    static int Bin(float v)
    {
        float s = v * 255f;
        long i = float.IsNaN(s) ? long.MinValue : s >= 9.2e18f ? long.MaxValue : s <= -9.2e18f ? long.MinValue : (long)s;
        return (int)Math.Clamp(i, 0, 255);
    }

    /// <summary>Per-channel histograms of the frames (in [channel, bin]), as their cumulative distributions (cumsum in double stored as float, then / total).</summary>
    static float[] Cdf(ImageBatch images, int first, int count)
    {
        int c = images.Channels, frame = images.Height * images.Width * c;
        var hist = new float[c * 256];
        for (int f = first; f < first + count; f++)
            for (int i = 0; i < frame; i += c)
                for (int k = 0; k < c; k++) hist[k * 256 + Bin(images.Data[f * frame + i + k])] += 1f;
        var cdf = new float[c * 256];
        for (int k = 0; k < c; k++)
        {
            double sum = 0;
            for (int j = 0; j < 256; j++) { sum += hist[k * 256 + j]; cdf[k * 256 + j] = (float)sum; }
            float total = cdf[k * 256 + 255];
            for (int j = 0; j < 256; j++) cdf[k * 256 + j] /= total;
        }
        return cdf;
    }

    /// <summary>torch.searchsorted (left) of the source CDF in the reference CDF, per channel: the matched bin of every source bin, / 255.</summary>
    static float[] Lut(float[] sourceCdf, float[] refCdf, int c)
    {
        var lut = new float[c * 256];
        for (int k = 0; k < c; k++)
            for (int j = 0; j < 256; j++)
            {
                float v = sourceCdf[k * 256 + j];
                int lo = 0, hi = 256;
                while (lo < hi)
                {
                    int mid = (lo + hi) >> 1;
                    if (refCdf[k * 256 + mid] < v) lo = mid + 1; else hi = mid;
                }
                lut[k * 256 + j] = Math.Min(lo, 255) / 255f;
            }
        return lut;
    }

    static ImageBatch Histogram(ImageBatch target, ImageBatch reference, string mode, int targetIndex, float strength)
    {
        int b = target.Batch, c = target.Channels, frame = target.Height * target.Width * c;
        float[]? uniformLut = null;
        if (mode != "per_frame")
        {
            var refCdf = Cdf(reference, 0, reference.Batch);
            var srcCdf = mode == "target_frame" ? Cdf(target, Math.Min(targetIndex, b - 1), 1) : Cdf(target, 0, b);
            uniformLut = Lut(srcCdf, refCdf, c);
        }
        var result = new ImageBatch(b, target.Height, target.Width, c);
        for (int f = 0; f < b; f++)
        {
            var lut = uniformLut ?? Lut(Cdf(target, f, 1), Cdf(reference, Math.Min(f, reference.Batch - 1), 1), c);
            for (int i = 0; i < frame; i++)
            {
                float src = target.Data[f * frame + i];
                float matched = lut[i % c * 256 + Bin(src)];
                float v = strength == 1f ? matched : TorchMath.Lerp(src, matched, strength);
                result.Data[f * frame + i] = Math.Clamp(v, 0f, 1f);
            }
        }
        return result;
    }

    // ---- Lab (reinhard_lab, mkl_lab) ----

    static ImageBatch LabTransfer(ImageBatch target, ImageBatch reference, string mode, int targetIndex, bool reinhard, float strength)
    {
        if (target.Channels != 3) throw new ArgumentException($"The Lab methods need RGB pictures (3 channels), not {target.Channels}.");
        int b = target.Batch, hw = target.Height * target.Width;
        bool singleRef = reference.Batch == 1;
        Stats? refStats = singleRef || mode != "per_frame" ? PoolStats(reference, reinhard) : null;

        // uniform / target_frame: one affine transform for every frame.
        Func<float[], int, float[]> transform;
        if (mode != "per_frame")
        {
            var s = mode == "target_frame" ? FrameStats(Lab.FromRgb(target, Math.Min(targetIndex, b - 1)), hw, reinhard) : PoolStats(target, reinhard);
            var r = refStats!;
            if (reinhard)
            {
                var scale = new float[3];
                var offset = new float[3];
                for (int k = 0; k < 3; k++) { scale[k] = r.Spread[k] / s.Spread[k]; offset[k] = r.Mean[k] - scale[k] * s.Mean[k]; }
                transform = (lab, _) => MapChannels(lab, (k, v) => v * scale[k] + offset[k]);
            }
            else
            {
                var t = MklMatrix(s.Spread, r.Spread);
                var offset = new float[3];
                for (int k = 0; k < 3; k++) offset[k] = r.Mean[k] - (t[k * 3] * s.Mean[0] + t[k * 3 + 1] * s.Mean[1] + t[k * 3 + 2] * s.Mean[2]);
                transform = (lab, _) => Affine(lab, t, offset);
            }
        }
        else
        {
            transform = (lab, f) =>
            {
                var r = singleRef ? refStats! : FrameStats(Lab.FromRgb(reference, Math.Min(f, reference.Batch - 1)), reference.Height * reference.Width, reinhard);
                if (reinhard)
                {
                    var s = FrameStats(lab, hw, reinhard: true);
                    var ratio = new float[3];
                    for (int k = 0; k < 3; k++) ratio[k] = r.Spread[k] / s.Spread[k];
                    return MapChannels(lab, (k, v) => (v - s.Mean[k]) * ratio[k] + r.Mean[k]);
                }
                var mean = Means(lab, hw);
                var centered = MapChannels(lab, (k, v) => v - mean[k]);
                return Affine(centered, MklMatrix(Covariance(centered, hw), r.Spread), r.Mean);
            };
        }

        var result = new ImageBatch(b, target.Height, target.Width, 3);
        for (int f = 0; f < b; f++)
        {
            var lab = Lab.FromRgb(target, f);
            var corrected = transform(lab, f);
            if (strength != 1f)
                for (int i = 0; i < corrected.Length; i++) corrected[i] = TorchMath.LerpFma(lab[i], corrected[i], strength);
            Lab.ToRgb(corrected, result.Data, f * hw * 3);
        }
        return result;
    }

    /// <summary>Mean per channel and, for reinhard, the standard deviation (at least eps) or, for mkl, the 3x3 covariance (row-major).</summary>
    sealed record Stats(float[] Mean, float[] Spread);

    static float[] MapChannels(float[] lab, Func<int, float, float> f)
    {
        int hw = lab.Length / 3;
        var o = new float[lab.Length];
        for (int k = 0; k < 3; k++) for (int i = 0; i < hw; i++) o[k * hw + i] = f(k, lab[k * hw + i]);
        return o;
    }

    /// <summary>t @ x + offset for channel-planar x (float products summed in float, like the 3x3 matmul).</summary>
    static float[] Affine(float[] x, float[] t, float[] offset)
    {
        int hw = x.Length / 3;
        var o = new float[x.Length];
        for (int i = 0; i < hw; i++)
        {
            float a = x[i], b = x[hw + i], c = x[2 * hw + i];
            for (int k = 0; k < 3; k++) o[k * hw + i] = (t[k * 3] * a + t[k * 3 + 1] * b + t[k * 3 + 2] * c) + offset[k];
        }
        return o;
    }

    static float[] Means(float[] lab, int hw)
    {
        var mean = new float[3];
        for (int k = 0; k < 3; k++)
        {
            double sum = 0;
            for (int i = 0; i < hw; i++) sum += lab[k * hw + i];
            mean[k] = (float)sum / hw;
        }
        return mean;
    }

    /// <summary>centered @ centered.T / hw.</summary>
    static float[] Covariance(float[] centered, int hw)
    {
        var cov = new float[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
            {
                double sum = 0;
                for (int i = 0; i < hw; i++) sum += (double)centered[r * hw + i] * centered[c * hw + i];
                cov[r * 3 + c] = (float)sum / hw;
            }
        return cov;
    }

    /// <summary>ComfyUI's _frame_stats: mean and the population standard deviation (clamped to eps) or covariance of one frame.</summary>
    static Stats FrameStats(float[] lab, int hw, bool reinhard)
    {
        var mean = Means(lab, hw);
        if (reinhard)
        {
            var std = new float[3];
            for (int k = 0; k < 3; k++)
            {
                double m = 0, ss = 0;
                for (int i = 0; i < hw; i++) m += lab[k * hw + i];
                m /= hw;
                for (int i = 0; i < hw; i++) { double d = lab[k * hw + i] - m; ss += d * d; }
                std[k] = Math.Max((float)Math.Sqrt(ss / hw), Eps);
            }
            return new(mean, std);
        }
        return new(mean, Covariance(MapChannels(lab, (k, v) => v - mean[k]), hw));
    }

    /// <summary>ComfyUI's _pool_stats: the mean of the frame means, then the mean of the frames' squared deviations (reinhard: its root) or covariances.</summary>
    static Stats PoolStats(ImageBatch images, bool reinhard)
    {
        int n = images.Batch, hw = images.Height * images.Width;
        var mean = new float[3];
        for (int f = 0; f < n; f++)
        {
            var m = Means(Lab.FromRgb(images, f), hw);
            for (int k = 0; k < 3; k++) mean[k] += m[k];
        }
        for (int k = 0; k < 3; k++) mean[k] /= n;
        var acc = new float[reinhard ? 3 : 9];
        for (int f = 0; f < n; f++)
        {
            var centered = MapChannels(Lab.FromRgb(images, f), (k, v) => v - mean[k]);
            if (reinhard)
            {
                var sq = Means(MapChannels(centered, (_, v) => v * v), hw);
                for (int k = 0; k < 3; k++) acc[k] += sq[k];
            }
            else
            {
                var cov = Covariance(centered, hw);
                for (int k = 0; k < 9; k++) acc[k] += cov[k];
            }
        }
        if (reinhard) return new(mean, acc.Select(v => Math.Max(MathF.Sqrt(v / n), Eps)).ToArray());
        return new(mean, acc.Select(v => v / n).ToArray());
    }

    /// <summary>
    /// ComfyUI's _mkl_matrix: T = Cs^-1/2 (Cs^1/2 Cr Cs^1/2)^1/2 Cs^-1/2 through two symmetric eigen decompositions, Cs's root eigenvalues clamped to at
    /// least eps (the result does not depend on the eigenvectors' signs or order). Evaluated in double from the float covariances.
    /// </summary>
    static float[] MklMatrix(float[] covS, float[] covR)
    {
        var (valS, vecS) = SymmetricEigen.Decompose(covS.Select(v => (double)v).ToArray());
        var sqrtS = valS.Select(v => Math.Max(Math.Sqrt(Math.Max(v, 0)), (double)Eps)).ToArray();
        var scaled = new double[9];
        var invScaled = new double[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) { scaled[r * 3 + c] = vecS[r * 3 + c] * sqrtS[c]; invScaled[r * 3 + c] = vecS[r * 3 + c] / sqrtS[c]; }
        var cr = covR.Select(v => (double)v).ToArray();
        var mid = Mul(Mul(Transpose(scaled), cr), scaled);
        var (valM, vecM) = SymmetricEigen.Decompose(mid);
        var half = new double[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
            {
                double s = 0;
                for (int k = 0; k < 3; k++) s += vecM[r * 3 + k] * Math.Sqrt(Math.Max(valM[k], 0)) * vecM[c * 3 + k];
                half[r * 3 + c] = s;
            }
        return Mul(Mul(invScaled, half), Transpose(invScaled)).Select(v => (float)v).ToArray();
    }

    static double[] Mul(double[] a, double[] b)
    {
        var o = new double[9];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++) o[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
        return o;
    }

    static double[] Transpose(double[] a) => [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
}

/// <summary>kornia's rgb_to_lab / lab_to_rgb (D65, 2° observer; sRGB companding), in float32 like the torch code. Lab is channel-planar: [L.., a.., b..].</summary>
internal static class Lab
{
    static readonly float[] RgbToXyz = [0.412453f, 0.357580f, 0.180423f, 0.212671f, 0.715160f, 0.072169f, 0.019334f, 0.119193f, 0.950227f];
    static readonly float[] XyzToRgb =
    [
        (float)3.2404813432005266, (float)-1.5371515162713185, (float)-0.4985363261688878,
        (float)-0.9692549499965682, (float)1.8759900014898907, (float)0.0415559265582928,
        (float)0.0556466391351772, (float)-0.2040413383665112, (float)1.0573110696453443,
    ];
    static readonly float[] White = [0.95047f, 1.0f, 1.08883f];
    const float FourTwentyNinths = (float)(4.0 / 29.0), Third = (float)(1 / 3.0), InvGamma = (float)(1 / 2.4);

    public static float[] FromRgb(ImageBatch images, int frame)
    {
        int hw = images.Height * images.Width, c = images.Channels;
        var lab = new float[3 * hw];
        Span<float> lin = stackalloc float[3];
        Span<float> f = stackalloc float[3];
        for (int i = 0; i < hw; i++)
        {
            int p = (frame * hw + i) * c;
            for (int k = 0; k < 3; k++)
            {
                float v = images.Data[p + k];
                lin[k] = v > 0.04045f ? MathF.Pow((v + 0.055f) / 1.055f, 2.4f) : v / 12.92f;
            }
            for (int k = 0; k < 3; k++)
            {
                float xyz = RgbToXyz[k * 3] * lin[0] + RgbToXyz[k * 3 + 1] * lin[1] + RgbToXyz[k * 3 + 2] * lin[2];
                float n = xyz / White[k];
                f[k] = n > 0.008856f ? MathF.Pow(Math.Max(n, 0.008856f), Third) : 7.787f * n + FourTwentyNinths;
            }
            lab[i] = 116f * f[1] - 16f;
            lab[hw + i] = 500f * (f[0] - f[1]);
            lab[2 * hw + i] = 200f * (f[1] - f[2]);
        }
        return lab;
    }

    /// <summary>Lab back to sRGB, clipped to 0..1, written channels-last at <paramref name="offset"/>.</summary>
    public static void ToRgb(float[] lab, float[] dst, int offset)
    {
        int hw = lab.Length / 3;
        Span<float> f = stackalloc float[3];
        Span<float> xyz = stackalloc float[3];
        for (int i = 0; i < hw; i++)
        {
            float fy = (lab[i] + 16f) / 116f;
            f[0] = lab[hw + i] / 500f + fy;
            f[1] = fy;
            f[2] = Math.Max(fy - lab[2 * hw + i] / 200f, 0f);
            for (int k = 0; k < 3; k++)
            {
                float v = f[k];
                xyz[k] = (v > 0.2068966f ? v * v * v : (v - FourTwentyNinths) / 7.787f) * White[k];
            }
            for (int k = 0; k < 3; k++)
            {
                float lin = XyzToRgb[k * 3] * xyz[0] + XyzToRgb[k * 3 + 1] * xyz[1] + XyzToRgb[k * 3 + 2] * xyz[2];
                float s = lin > 0.0031308f ? 1.055f * MathF.Pow(Math.Max(lin, 0.0031308f), InvGamma) - 0.055f : 12.92f * lin;
                dst[offset + i * 3 + k] = Math.Clamp(s, 0f, 1f);
            }
        }
    }
}

/// <summary>Eigenvalues and eigenvectors (as columns, row-major) of a symmetric 3x3 matrix by cyclic Jacobi rotations, in double.</summary>
internal static class SymmetricEigen
{
    public static (double[] Values, double[] Vectors) Decompose(double[] m)
    {
        var a = (double[])m.Clone();
        // Symmetrise exactly like eigh, which reads only the lower triangle.
        a[1] = a[3]; a[2] = a[6]; a[5] = a[7];
        var v = new double[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        for (int sweep = 0; sweep < 64; sweep++)
        {
            double off = a[1] * a[1] + a[2] * a[2] + a[5] * a[5];
            double scale = a[0] * a[0] + a[4] * a[4] + a[8] * a[8] + 2 * off;
            if (off == 0 || off <= 1e-34 * scale) break;
            for (int p = 0; p < 2; p++)
                for (int q = p + 1; q < 3; q++)
                {
                    double apq = a[p * 3 + q];
                    if (apq == 0) continue;
                    double app = a[p * 3 + p], aqq = a[q * 3 + q];
                    double theta = (aqq - app) / (2 * apq);
                    double t = Math.Sign(theta == 0 ? 1 : theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1));
                    double c = 1 / Math.Sqrt(t * t + 1), s = t * c;
                    for (int k = 0; k < 3; k++)
                    {
                        double akp = a[k * 3 + p], akq = a[k * 3 + q];
                        a[k * 3 + p] = c * akp - s * akq;
                        a[k * 3 + q] = s * akp + c * akq;
                    }
                    for (int k = 0; k < 3; k++)
                    {
                        double apk = a[p * 3 + k], aqk = a[q * 3 + k];
                        a[p * 3 + k] = c * apk - s * aqk;
                        a[q * 3 + k] = s * apk + c * aqk;
                    }
                    for (int k = 0; k < 3; k++)
                    {
                        double vkp = v[k * 3 + p], vkq = v[k * 3 + q];
                        v[k * 3 + p] = c * vkp - s * vkq;
                        v[k * 3 + q] = s * vkp + c * vkq;
                    }
                }
        }
        // Ascending eigenvalues, like torch.linalg.eigh.
        var order = new[] { 0, 1, 2 }.OrderBy(i => a[i * 4]).ToArray();
        var values = order.Select(i => a[i * 4]).ToArray();
        var vectors = new double[9];
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) vectors[r * 3 + c] = v[r * 3 + order[c]];
        return (values, vectors);
    }
}
