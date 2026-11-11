using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entry, docs.comfy.org's description of the node (stages only) and black-box
// probes of ComfyUI (2026-11-11). ComfyUI's calibrated luminance curve is not described anywhere we could find, so it is a table measured from ComfyUI
// (tools/parity/make_sdr_to_hdr_table.py) and interpolated: a documented difference of up to about 2e-4 in output values (PARITY.md).

/// <summary>
/// Convert SDR to HDR, as measured in ComfyUI: each pixel (clamped to 0..1) is decoded from sRGB, converted to Rec.2020 with BT.2087's matrix, scaled
/// by one gain that depends on its Rec.709 luminance Y and the target luminance, and encoded with the BT.2100 HLG OETF. White reaches
/// (target / 1000)^(1/1.2) (HLG's 1000-nit reference display, system gamma 1.2), 1 from 1000 nits up; 0 keeps the calibrated rendering. The gain curve
/// comes from <c>Resources/sdr_to_hdr_gain.bin</c>: 101 target knots by the 256 sRGB levels of Y, cubic in log(target) and in level. A one-channel
/// picture becomes three; channels after the third (alpha) are copied; two channels are an error.
/// </summary>
[Node("ImageSDRToHDR", Category = "image/color", DisplayName = "Convert SDR to HDR",
    Description = "Expand sRGB brightness into Rec.2020 HLG HDR. Save the result with the HDR color space selected. Alpha is preserved.", OutputNames = ["IMAGE"])]
public sealed class ImageSDRToHDR
{
    public static DynamicOption[] Algorithms() =>
    [
        new("lottes",
            DynamicInput.Float("target_luminance", 650.0, 0.0, 10000.0, 1.0,
                "Peak-white target in nits. 0 keeps the calibrated rendering. Dark images are not normalized to this peak.")),
    ];

    public ImageBatch Execute(
        [Input(Tooltip = "SDR image(s) in sRGB.")] ImageBatch image,
        [DynamicCombo(nameof(Algorithms)), Input(Tooltip = "Lottes-style rational expansion blended with a calibrated monotonic log-gain curve.")]
        DynamicChoice algorithm)
    {
        if (algorithm.Key != "lottes") throw new ArgumentException($"Unknown algorithm '{algorithm.Key}'.", nameof(algorithm));
        int c = image.Channels;
        if (c == 2) throw new ArgumentException("Convert SDR to HDR needs a picture of 1, 3 or more channels, not 2.", nameof(image));
        var gain = GainTable.Row(algorithm.Get("target_luminance", 650.0));
        int outC = Math.Max(c, 3), n = image.Batch * image.Height * image.Width;
        var result = new ImageBatch(image.Batch, image.Height, image.Width, outC);
        var src = image.Data;
        var dst = result.Data;
        for (int i = 0; i < n; i++)
        {
            long s = (long)i * c, d = (long)i * outC;
            double r = Decode(src[s]), g = Decode(src[s + (c == 1 ? 0 : 1)]), b = Decode(src[s + (c == 1 ? 0 : 2)]);
            double y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            double k = GainTable.At(gain, Encode(y) * 255);
            dst[d] = (float)Hlg(k * (0.627403895934699 * r + 0.329283038377884 * g + 0.043313065687417 * b));
            dst[d + 1] = (float)Hlg(k * (0.069097289358232 * r + 0.919540395075459 * g + 0.011362315566309 * b));
            dst[d + 2] = (float)Hlg(k * (0.016391438875150 * r + 0.088013307877226 * g + 0.895595253247624 * b));
            for (int extra = 3; extra < c; extra++) dst[d + extra] = src[s + extra];
        }
        return result;
    }

    /// <summary>The sRGB decoding of a value clamped to 0..1.</summary>
    static double Decode(float v)
    {
        double x = Math.Clamp((double)v, 0, 1);
        return x <= 0.04045 ? x / 12.92 : Math.Pow((x + 0.055) / 1.055, 2.4);
    }

    static double Encode(double y) => y <= 0.0031308 ? 12.92 * y : 1.055 * Math.Pow(y, 1 / 2.4) - 0.055;

    /// <summary>BT.2100's HLG OETF of linear scene light (negative values as 0).</summary>
    static double Hlg(double e)
    {
        const double a = 0.17883277, b = 0.28466892, c = 0.55991073;
        e = Math.Max(e, 0);
        return e <= 1.0 / 12 ? Math.Sqrt(3 * e) : a * Math.Log(12 * e - b) + c;
    }

    /// <summary>The measured gain table and its interpolation.</summary>
    static class GainTable
    {
        static readonly Lazy<(float[] Knots, float[][] Rows)> Data = new(Load);

        static (float[] Knots, float[][] Rows) Load()
        {
            using var stream = typeof(ImageSDRToHDR).Assembly.GetManifestResourceStream("SpawnGen.Nodes.Core.sdr_to_hdr_gain.bin")
                ?? throw new InvalidOperationException("The SDR to HDR gain table is missing from the assembly.");
            using var reader = new BinaryReader(stream);
            int rows = reader.ReadInt32(), columns = reader.ReadInt32();
            var knots = new float[rows];
            for (int i = 0; i < rows; i++) knots[i] = reader.ReadSingle();
            var table = new float[rows][];
            for (int i = 0; i < rows; i++)
            {
                table[i] = new float[columns];
                for (int j = 0; j < columns; j++) table[i][j] = reader.ReadSingle();
            }
            return (knots, table);
        }

        /// <summary>
        /// The 256 gains for a target luminance: knot 0 for 0; otherwise a cubic Hermite in log(target) through the knots on its side of 1000 nits
        /// (Catmull-Rom tangents on the uneven spacing, one-sided at the ends), the target clamped to 0.001..10000.
        /// </summary>
        public static double[] Row(double target)
        {
            var (knots, rows) = Data.Value;
            int columns = rows[0].Length;
            if (target <= 0) return [.. rows[0].Select(v => (double)v)];
            double t = Math.Clamp(target, knots[1], knots[^1]);
            var side = Enumerable.Range(1, knots.Length - 1).Where(i => t <= 1000 ? knots[i] <= 1000 : knots[i] >= 1000).ToArray();
            var x = side.Select(i => Math.Log(knots[i])).ToArray();
            double xt = Math.Log(t);
            int seg = 0;
            while (seg < x.Length - 2 && x[seg + 1] < xt) seg++;
            double x0 = x[seg], x1 = x[seg + 1], h = x1 - x0, u = (xt - x0) / h;
            float[] P(int j) => rows[side[Math.Clamp(j, 0, side.Length - 1)]];
            double X(int j) => x[Math.Clamp(j, 0, x.Length - 1)];
            var p0 = P(seg);
            var p1 = P(seg + 1);
            var pm = P(seg - 1);
            var pp = P(seg + 2);
            double h00 = 2 * u * u * u - 3 * u * u + 1, h10 = u * u * u - 2 * u * u + u, h01 = -2 * u * u * u + 3 * u * u, h11 = u * u * u - u * u;
            var result = new double[columns];
            for (int j = 0; j < columns; j++)
            {
                double m0 = seg > 0 ? (p1[j] - pm[j]) / (x1 - X(seg - 1)) * h : p1[j] - p0[j];
                double m1 = seg + 2 < x.Length ? (pp[j] - p0[j]) / (X(seg + 2) - x0) * h : p1[j] - p0[j];
                result[j] = h00 * p0[j] + h10 * m0 + h01 * p1[j] + h11 * m1;
            }
            return result;
        }

        /// <summary>A gain at a fractional sRGB level (0..255): Catmull-Rom between levels, exact on whole levels.</summary>
        public static double At(double[] row, double level)
        {
            level = Math.Clamp(level, 0, row.Length - 1);
            int i = Math.Min((int)level, row.Length - 2);
            double u = level - i;
            double p0 = row[i], p1 = row[i + 1], pm = row[Math.Max(i - 1, 0)], pp = row[Math.Min(i + 2, row.Length - 1)];
            double m0 = (p1 - pm) / (i > 0 ? 2 : 1), m1 = (pp - p0) / (i + 2 < row.Length ? 2 : 1);
            return (2 * u * u * u - 3 * u * u + 1) * p0 + (u * u * u - 2 * u * u + u) * m0 + (-2 * u * u * u + 3 * u * u) * p1 + (u * u * u - u * u) * m1;
        }
    }
}
