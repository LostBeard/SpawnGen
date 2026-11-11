using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entry, SpawnGen's tests and fixture (comfy_pad_color_nodes.json.gz) and black-box
// probes of ComfyUI (2026-11-11). Sources read: kornia 0.9.0's color/lab.py, rgb.py and xyz.py (Apache-2.0) for the Lab conversion; the colour
// transfer methods are the published ones: Reinhard et al. 2001 ("Color Transfer between Images") and Pitie & Kokaram 2007 ("The linear
// Monge-Kantorovitch linear colour mapping for example-based colour transfer").

/// <summary>
/// Transfer Color, as measured in ComfyUI:
/// <list type="bullet">
/// <item>reinhard_lab: in kornia's Lab, each channel (x - mean_s) / max(std_s, 1e-6) * std_r + mean_r with population standard deviations.</item>
/// <item>mkl_lab: in Lab, (x - mean_s) T + mean_r with Pitie's map T = Cs^-1/2 (Cs^1/2 Cr Cs^1/2)^1/2 Cs^-1/2 on population covariances.</item>
/// <item>histogram: per channel of the picture as it is, a 256-entry table: bins trunc(255 v) clamped to 0..255, each source bin goes to the first
/// reference bin whose normalised cumulative count reaches the source's (searchsorted left), value bin / 255.</item>
/// </list>
/// source_stats: per_frame takes the statistics of each target frame and of reference frame min(b, last); uniform pools every target frame and every
/// reference frame; target_frame uses target frame min(target_index, last) and every reference frame, for all frames. Strength blends with torch.lerp:
/// in Lab before converting back (clipped to 0..1) for the Lab methods, in RGB then clamped to 0..1 for histogram; strength 0 returns the target as is.
/// The Lab methods need 3-channel pictures; histogram works on any channel count, splitting the reference (channel-major) into that many parts.
/// </summary>
[Node("ColorTransfer", Category = "image/filters", DisplayName = "Transfer Color",
    Description = "Match the colors of one image to another using various algorithms.", OutputNames = ["image"])]
public sealed class ColorTransfer
{
    public static string[] MethodOptions() => ["reinhard_lab", "mkl_lab", "histogram"];

    public static DynamicOption[] SourceStats() =>
    [
        new("per_frame"),
        new("uniform"),
        new("target_frame",
            DynamicInput.Int("target_index", 0, 0, 10000, tooltip: "Frame index used as the source baseline for computing the transform to image_ref")),
    ];

    public ImageBatch Execute(
        [Input(Tooltip = "Image(s) to apply the color transform to.")] ImageBatch imageTarget,
        [Input(Tooltip = "Reference image(s) to match colors to.")] ImageBatch imageRef,
        [Input(OptionsProvider = nameof(MethodOptions))] string method,
        [DynamicCombo(nameof(SourceStats)), Input(Tooltip = "per_frame: each frame matched to image_ref individually. uniform: pool stats across all source frames as baseline, match to image_ref. target_frame: use one chosen frame as the baseline for the transform to image_ref, applied uniformly to all frames (preserves relative differences)")]
        DynamicChoice sourceStats,
        [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double strength = 1.0)
    {
        if (!MethodOptions().Contains(method)) throw new ArgumentException($"Unknown method '{method}'.", nameof(method));
        if (strength == 0) return imageTarget;
        int batch = imageTarget.Batch;
        int pick = sourceStats.Key == "target_frame" ? (int)Math.Clamp(sourceStats.Get("target_index", 0L), 0, batch - 1) : 0;
        // The frames whose statistics stand for frame b, on each side.
        int[] Sources(int b) => sourceStats.Key switch
        {
            "per_frame" => [b],
            "uniform" => [.. Enumerable.Range(0, batch)],
            "target_frame" => [pick],
            _ => throw new ArgumentException($"Unknown source_stats '{sourceStats.Key}'.", nameof(sourceStats)),
        };
        int[] References(int b) => sourceStats.Key == "per_frame" ? [Math.Min(b, imageRef.Batch - 1)] : [.. Enumerable.Range(0, imageRef.Batch)];
        float s = (float)strength;
        return method == "histogram" ? Histogram(imageTarget, imageRef, Sources, References, s) : LabTransfer(imageTarget, imageRef, method, Sources, References, s);
    }

    // ---- Lab methods ----

    static ImageBatch LabTransfer(ImageBatch target, ImageBatch reference, string method, Func<int, int[]> sources, Func<int, int[]> references, float strength)
    {
        foreach (var image in new[] { target, reference })
            if (image.Channels != 3)
                throw new ArgumentException($"Input size must have a shape of (*, 3, H, W). Got {image.Channels} channels.");
        var labT = Lab.FromRgb(target.Data);
        var labR = Lab.FromRgb(reference.Data);
        int nT = target.Height * target.Width, nR = reference.Height * reference.Width;
        var result = new ImageBatch(target.Batch, target.Height, target.Width, 3);
        var outLab = new float[nT * 3];
        for (int b = 0; b < target.Batch; b++)
        {
            var src = Frames(labT, nT, sources(b));
            var refs = Frames(labR, nR, references(b));
            var frame = new ReadOnlySpan<float>(labT, b * nT * 3, nT * 3);
            if (method == "reinhard_lab") Reinhard(frame, src, refs, outLab);
            else Mkl(frame, src, refs, outLab);
            for (int i = 0; i < outLab.Length; i++) outLab[i] = TorchMath.Lerp(frame[i], outLab[i], strength);
            Lab.ToRgb(outLab, result.Data.AsSpan(b * nT * 3, nT * 3));
        }
        return result;
    }

    /// <summary>The pixels of the given frames, one after another (interleaved channels).</summary>
    static float[] Frames(float[] data, int pixels, int[] frames)
    {
        var all = new float[frames.Length * pixels * 3];
        for (int k = 0; k < frames.Length; k++) Array.Copy(data, (long)frames[k] * pixels * 3, all, (long)k * pixels * 3, pixels * 3);
        return all;
    }

    static (float[] Mean, float[] Std) MeanStd(float[] px)
    {
        int n = px.Length / 3;
        var mean = new double[3];
        for (int i = 0; i < n; i++) for (int c = 0; c < 3; c++) mean[c] += px[i * 3 + c];
        for (int c = 0; c < 3; c++) mean[c] /= n;
        var m = new float[3];
        for (int c = 0; c < 3; c++) m[c] = (float)mean[c];
        var variance = new double[3];
        for (int i = 0; i < n; i++) for (int c = 0; c < 3; c++) { double d = px[i * 3 + c] - mean[c]; variance[c] += d * d; }
        var std = new float[3];
        for (int c = 0; c < 3; c++) std[c] = (float)Math.Sqrt(variance[c] / n);
        return (m, std);
    }

    static void Reinhard(ReadOnlySpan<float> frame, float[] src, float[] refs, float[] result)
    {
        var (ms, ss) = MeanStd(src);
        var (mr, sr) = MeanStd(refs);
        for (int c = 0; c < 3; c++) ss[c] = Math.Max(ss[c], 1e-6f);
        for (int i = 0; i < frame.Length; i++)
        {
            int c = i % 3;
            result[i] = (frame[i] - ms[c]) / ss[c] * sr[c] + mr[c];
        }
    }

    static (float[] Mean, double[] Cov) MeanCov(float[] px)
    {
        int n = px.Length / 3;
        var mean = new double[3];
        for (int i = 0; i < n; i++) for (int c = 0; c < 3; c++) mean[c] += px[i * 3 + c];
        for (int c = 0; c < 3; c++) mean[c] /= n;
        var cov = new double[9];
        for (int i = 0; i < n; i++)
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    cov[r * 3 + c] += (px[i * 3 + r] - mean[r]) * (px[i * 3 + c] - mean[c]);
        for (int k = 0; k < 9; k++) cov[k] /= n;
        return ([(float)mean[0], (float)mean[1], (float)mean[2]], cov);
    }

    static void Mkl(ReadOnlySpan<float> frame, float[] src, float[] refs, float[] result)
    {
        var (ms, cs) = MeanCov(src);
        var (mr, cr) = MeanCov(refs);
        var t = MklMap(cs, cr);
        for (int i = 0; i < frame.Length; i += 3)
        {
            float x0 = frame[i] - ms[0], x1 = frame[i + 1] - ms[1], x2 = frame[i + 2] - ms[2];
            for (int r = 0; r < 3; r++) result[i + r] = x0 * t[r * 3] + x1 * t[r * 3 + 1] + x2 * t[r * 3 + 2] + mr[r];
        }
    }

    /// <summary>The float32 form the tests check: <see cref="MklMap"/> of float covariances.</summary>
    static float[] MklMatrix(float[] cs, float[] cr) => MklMap([.. cs.Select(v => (double)v)], [.. cr.Select(v => (double)v)]);

    /// <summary>
    /// Pitie's linear Monge-Kantorovich map from covariance <paramref name="cs"/> to <paramref name="cr"/> (row-major 3 x 3): with Cs = Va Da^2 Va^T and
    /// C = Da Va^T Cr Va Da = Vc Dc^2 Vc^T, T = Va Da^-1 Vc Dc Vc^T Da^-1 Va^T, so T Cs T^T = Cr and T is symmetric. Eigenvalues below 1e-6 are raised to
    /// it (a degenerate covariance has no unique map; ComfyUI's own result there depends on its LAPACK build).
    /// </summary>
    static float[] MklMap(double[] cs, double[] cr)
    {
        const double eps = 1e-6;
        var (wa, va) = SymmetricEigen.Decompose(cs);
        var da = wa.Select(w => Math.Sqrt(Math.Max(w, eps))).ToArray();
        // C = Da Va^T Cr Va Da
        var vtcv = Mul(Transpose(va), Mul(cr, va));
        var c = new double[9];
        for (int r = 0; r < 3; r++) for (int k = 0; k < 3; k++) c[r * 3 + k] = da[r] * vtcv[r * 3 + k] * da[k];
        var (wc, vc) = SymmetricEigen.Decompose(c);
        var dc = wc.Select(w => Math.Sqrt(Math.Max(w, eps))).ToArray();
        // middle = Vc Dc Vc^T, then T = Va Da^-1 middle Da^-1 Va^T
        var middle = new double[9];
        for (int r = 0; r < 3; r++) for (int k = 0; k < 3; k++) { double sum = 0; for (int j = 0; j < 3; j++) sum += vc[r * 3 + j] * dc[j] * vc[k * 3 + j]; middle[r * 3 + k] = sum; }
        var scaled = new double[9];
        for (int r = 0; r < 3; r++) for (int k = 0; k < 3; k++) scaled[r * 3 + k] = middle[r * 3 + k] / (da[r] * da[k]);
        var t = Mul(va, Mul(scaled, Transpose(va)));
        return [.. t.Select(v => (float)v)];
    }

    static double[] Mul(double[] a, double[] b)
    {
        var r = new double[9];
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { double s = 0; for (int k = 0; k < 3; k++) s += a[i * 3 + k] * b[k * 3 + j]; r[i * 3 + j] = s; }
        return r;
    }

    static double[] Transpose(double[] a) => [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];

    // ---- histogram ----

    static ImageBatch Histogram(ImageBatch target, ImageBatch reference, Func<int, int[]> sources, Func<int, int[]> references, float strength)
    {
        int c = target.Channels, nT = target.Height * target.Width, nR = reference.Height * reference.Width, cr = reference.Channels;
        var result = new ImageBatch(target.Batch, target.Height, target.Width, c);
        for (int b = 0; b < target.Batch; b++)
        {
            var src = sources(b);
            var refs = references(b);
            // The reference frames channel-major (all of channel 0, then channel 1 ...), cut into as many equal parts as the target has channels.
            long refTotal = (long)refs.Length * nR * cr;
            if (refTotal % c != 0) throw new ArgumentException($"shape '[{c}, -1]' is invalid for input of size {refTotal}");
            var refValues = new float[refTotal];
            long k = 0;
            for (int ch = 0; ch < cr; ch++)
                foreach (int f in refs)
                    for (int p = 0; p < nR; p++) refValues[k++] = reference.Data[((long)f * nR + p) * cr + ch];
            int part = (int)(refTotal / c);
            for (int ch = 0; ch < c; ch++)
            {
                var srcCounts = new long[256];
                foreach (int f in src)
                    for (int p = 0; p < nT; p++) srcCounts[Bin(target.Data[((long)f * nT + p) * c + ch])]++;
                var refCounts = new long[256];
                for (int i = 0; i < part; i++) refCounts[Bin(refValues[(long)ch * part + i])]++;
                var lut = Table(srcCounts, refCounts);
                for (int p = 0; p < nT; p++)
                {
                    long at = ((long)b * nT + p) * c + ch;
                    float v = target.Data[at];
                    result.Data[at] = Math.Clamp(TorchMath.Lerp(v, lut[Bin(v)], strength), 0f, 1f);
                }
            }
        }
        return result;
    }

    /// <summary>trunc(255 v) clamped to 0..255 (NaN to 0).</summary>
    static int Bin(float v)
    {
        float scaled = v * 255f;
        if (!(scaled > 0f)) return 0;
        return scaled >= 255f ? 255 : (int)scaled;
    }

    /// <summary>Source bin to output value: the first reference bin whose normalised cumulative count (float32) is at least the source bin's.</summary>
    static float[] Table(long[] srcCounts, long[] refCounts)
    {
        static float[] Cdf(long[] counts)
        {
            var cdf = new float[256];
            float sum = 0;
            for (int i = 0; i < 256; i++) { sum += counts[i]; cdf[i] = sum; }
            float total = cdf[255];
            for (int i = 0; i < 256; i++) cdf[i] /= total;
            return cdf;
        }
        var cs = Cdf(srcCounts);
        var cr = Cdf(refCounts);
        var lut = new float[256];
        for (int i = 0; i < 256; i++)
        {
            int lo = 0, hi = 256;
            while (lo < hi) { int mid = (lo + hi) >> 1; if (cr[mid] < cs[i]) lo = mid + 1; else hi = mid; }
            lut[i] = Math.Min(lo, 255) / 255f;
        }
        return lut;
    }
}

/// <summary>
/// kornia's sRGB to CIE Lab and back (kornia 0.9.0 color/lab.py, rgb.py, xyz.py; Apache-2.0), float32 step by step: D65 white (0.95047, 1, 1.08883),
/// kornia's sRGB to XYZ matrix and its inverse, the 0.008856 / 7.787 linear segment, and a clip to 0..1 on the way back.
/// </summary>
internal static class Lab
{
    public static float[] FromRgb(float[] rgb)
    {
        var lab = new float[rgb.Length];
        for (int i = 0; i < rgb.Length; i += 3)
        {
            float r = Linear(rgb[i]), g = Linear(rgb[i + 1]), b = Linear(rgb[i + 2]);
            float x = 0.412453f * r + 0.357580f * g + 0.180423f * b;
            float y = 0.212671f * r + 0.715160f * g + 0.072169f * b;
            float z = 0.019334f * r + 0.119193f * g + 0.950227f * b;
            float fx = F(x / 0.95047f), fy = F(y / 1f), fz = F(z / 1.08883f);
            lab[i] = 116f * fy - 16f;
            lab[i + 1] = 500f * (fx - fy);
            lab[i + 2] = 200f * (fy - fz);
        }
        return lab;
    }

    public static void ToRgb(ReadOnlySpan<float> lab, Span<float> rgb)
    {
        for (int i = 0; i < lab.Length; i += 3)
        {
            float fy = (lab[i] + 16f) / 116f;
            float fx = lab[i + 1] / 500f + fy;
            float fz = Math.Max(fy - lab[i + 2] / 200f, 0f);
            float x = FInverse(fx) * 0.95047f, y = FInverse(fy) * 1f, z = FInverse(fz) * 1.08883f;
            float r = 3.2404813432005266f * x + -1.5371515162713185f * y + -0.4985363261688878f * z;
            float g = -0.9692549499965682f * x + 1.8759900014898907f * y + 0.0415559265582928f * z;
            float b = 0.0556466391351772f * x + -0.2040413383665112f * y + 1.0573110696453443f * z;
            rgb[i] = Math.Clamp(Encoded(r), 0f, 1f);
            rgb[i + 1] = Math.Clamp(Encoded(g), 0f, 1f);
            rgb[i + 2] = Math.Clamp(Encoded(b), 0f, 1f);
        }
    }

    static float Linear(float v) => v > 0.04045f ? MathF.Pow(Math.Max(v + 0.055f, 0f) / 1.055f, 2.4f) : v / 12.92f;

    static float Encoded(float v) => v > 0.0031308f ? 1.055f * MathF.Pow(Math.Max(v, 0.0031308f), 1f / 2.4f) - 0.055f : 12.92f * v;

    static float F(float t) => t > 0.008856f ? MathF.Pow(Math.Max(t, 0.008856f), 1f / 3f) : 7.787f * t + 4f / 29f;

    static float FInverse(float f) => f > 0.2068966f ? MathF.Pow(f, 3f) : (f - 4f / 29f) / 7.787f;
}

/// <summary>Eigenvalues (ascending) and unit eigenvectors (columns, row-major 3 x 3) of a symmetric 3 x 3 matrix, by cyclic Jacobi rotations.</summary>
internal static class SymmetricEigen
{
    public static (double[] Values, double[] Vectors) Decompose(double[] m)
    {
        var a = (double[])m.Clone();
        double[] v = [1, 0, 0, 0, 1, 0, 0, 0, 1];
        for (int sweep = 0; sweep < 64; sweep++)
        {
            double off = a[1] * a[1] + a[2] * a[2] + a[5] * a[5];
            if (off < 1e-300) break;
            for (int p = 0; p < 2; p++)
                for (int q = p + 1; q < 3; q++)
                {
                    double apq = a[p * 3 + q];
                    if (apq == 0) continue;
                    double app = a[p * 3 + p], aqq = a[q * 3 + q];
                    double theta = (aqq - app) / (2 * apq);
                    double t = Math.Sign(theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1));
                    if (theta == 0) t = 1;
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
        int[] order = [.. Enumerable.Range(0, 3).OrderBy(i => a[i * 3 + i])];
        var values = order.Select(i => a[i * 3 + i]).ToArray();
        var vectors = new double[9];
        for (int r = 0; r < 3; r++) for (int k = 0; k < 3; k++) vectors[r * 3 + k] = v[r * 3 + order[k]];
        return (values, vectors);
    }
}
