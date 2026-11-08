using System.Globalization;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// A tone curve (the CURVE socket): control points in the unit square and how to join them. "linear" joins neighbours with straight lines; anything else is a
/// monotone cubic (the Fritsch-Carlson method: Hermite segments whose tangents are limited so the curve never overshoots between two points). Left of the first
/// point and right of the last the curve stays at their heights; with one point it is flat, with none it is 0. Nothing is clamped.
/// </summary>
/// <remarks>
/// Measured with ComfyUI (Fixtures/curves): points are sorted by x keeping the given order of equal x; a missing or unknown interpolation is the monotone
/// cubic; a lookup table of n entries samples x = i / (n - 1) (x = 0 for a table of one).
/// </remarks>
[Socket("CURVE")]
public sealed class Curve
{
    readonly double[] _x, _y, _tangents;

    public Curve(IEnumerable<(double X, double Y)> points, string? interpolation = "monotone_cubic")
    {
        // OrderBy is a stable sort: points at the same x keep the order they were given in.
        var sorted = points.OrderBy(p => p.X).ToArray();
        Points = sorted;
        Interpolation = interpolation == "linear" ? "linear" : "monotone_cubic";
        _x = sorted.Select(p => p.X).ToArray();
        _y = sorted.Select(p => p.Y).ToArray();
        _tangents = Interpolation == "linear" ? [] : Tangents(_x, _y);
    }

    /// <summary>The control points, sorted by x.</summary>
    public IReadOnlyList<(double X, double Y)> Points { get; }

    /// <summary>"linear" or "monotone_cubic".</summary>
    public string Interpolation { get; }

    /// <summary>
    /// A curve from the CURVE widget's value: <c>{"points": [[x, y], ...], "interpolation": ...}</c>, or just the list of points (then monotone cubic).
    /// A curve passes through. Numbers may also be given as text, as Python's <c>float()</c> reads them.
    /// </summary>
    public static Curve FromRaw(object? raw)
    {
        switch (raw)
        {
            case Curve curve:
                return curve;
            case IEnumerable<KeyValuePair<string, object?>> dict:
            {
                var map = dict.ToDictionary(kv => kv.Key, kv => kv.Value, StringComparer.Ordinal);
                if (!map.TryGetValue("points", out var pts)) throw new KeyNotFoundException("'points'");
                return new Curve(ReadPoints(pts), map.GetValueOrDefault("interpolation") as string);
            }
            case System.Collections.IEnumerable list and not string:
                return new Curve(ReadPoints(list));
            default:
                throw new ArgumentException($"A curve is a dictionary with points or a list of points, not {raw?.GetType().Name ?? "None"}.");
        }
    }

    static List<(double, double)> ReadPoints(object? raw)
    {
        var result = new List<(double, double)>();
        if (raw is not System.Collections.IEnumerable items || raw is string) throw new ArgumentException("The curve's points must be a list of [x, y] pairs.");
        foreach (var item in items)
        {
            var xy = (item as System.Collections.IEnumerable)?.Cast<object?>().ToList();
            if (xy is null || item is string || xy.Count < 2) throw new ArgumentException("Each curve point is an [x, y] pair.");
            result.Add((Number(xy[0]), Number(xy[1])));
        }
        return result;
    }

    static double Number(object? v) => v switch
    {
        string s => double.Parse(s.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture),
        null => throw new ArgumentException("A curve point coordinate is None."),
        _ => Convert.ToDouble(v, CultureInfo.InvariantCulture),
    };

    /// <summary>
    /// Fritsch-Carlson tangents: the secant slopes of the segments; inside, their mean where the neighbours rise or fall together and 0 at a local extremum; at
    /// the ends the end segment's slope. Then per segment: a flat segment flattens both tangents, and tangents whose ratios to the slope (a, b) leave the circle
    /// a^2 + b^2 &lt;= 9 are scaled back onto it. A segment of zero width counts as flat.
    /// </summary>
    static double[] Tangents(double[] x, double[] y)
    {
        int n = x.Length;
        if (n < 2) return new double[n];
        var slope = new double[n - 1];
        for (int k = 0; k < n - 1; k++)
        {
            double h = x[k + 1] - x[k];
            slope[k] = h != 0 ? (y[k + 1] - y[k]) / h : 0.0;
        }
        var m = new double[n];
        m[0] = slope[0];
        m[n - 1] = slope[n - 2];
        for (int k = 1; k < n - 1; k++)
            m[k] = slope[k - 1] * slope[k] <= 0 ? 0.0 : (slope[k - 1] + slope[k]) / 2;
        for (int k = 0; k < n - 1; k++)
        {
            if (slope[k] == 0)
            {
                m[k] = m[k + 1] = 0.0;
                continue;
            }
            double a = m[k] / slope[k], b = m[k + 1] / slope[k];
            double r = a * a + b * b;
            if (r > 9)
            {
                double tau = 3 / Math.Sqrt(r);
                m[k] = tau * a * slope[k];
                m[k + 1] = tau * b * slope[k];
            }
        }
        return m;
    }

    /// <summary>The curve's height at <paramref name="x"/>.</summary>
    public double Interp(double x)
    {
        int n = _x.Length;
        if (n == 0) return 0.0;
        if (n == 1 || x <= _x[0]) return _y[0];
        if (x >= _x[n - 1]) return _y[n - 1];
        // The segment [x_k, x_k+1) holding x (the last segment starting at or before it).
        int k = 0;
        while (k < n - 2 && x >= _x[k + 1]) k++;
        double h = _x[k + 1] - _x[k];
        if (h == 0) return _y[k];
        double t = (x - _x[k]) / h;
        if (Interpolation == "linear") return _y[k] + t * (_y[k + 1] - _y[k]);
        // Cubic Hermite basis.
        double t2 = t * t, t3 = t2 * t;
        return (2 * t3 - 3 * t2 + 1) * _y[k] + (t3 - 2 * t2 + t) * h * _tangents[k] + (-2 * t3 + 3 * t2) * _y[k + 1] + (t3 - t2) * h * _tangents[k + 1];
    }

    /// <summary>The curve sampled at <paramref name="size"/> evenly spaced x from 0 to 1 (GLSL Shader's u_curve textures).</summary>
    public double[] ToLut(int size = 256)
    {
        var lut = new double[Math.Max(0, size)];
        for (int i = 0; i < lut.Length; i++) lut[i] = Interp(size > 1 ? i / (double)(size - 1) : 0.0);
        return lut;
    }

    /// <summary>How Preview as Text shows a curve, measured with ComfyUI: <c>MonotoneCubicCurve(points=[(0.0, 0.0), (1.0, 1.0)])</c>.</summary>
    public override string ToString() =>
        (Interpolation == "linear" ? "LinearCurve" : "MonotoneCubicCurve") + "(points=["
        + string.Join(", ", Points.Select(p => "(" + PyNumber.FloatRepr(p.X) + ", " + PyNumber.FloatRepr(p.Y) + ")")) + "])";
}

/// <summary>
/// Curve Editor: the curve drawn in its widget, as a CURVE. With a histogram connected (Image Histogram) the node shows it behind the curve (ui "histogram",
/// one count per level), and keeps showing it after runs that do not reach the node.
/// </summary>
[Node("CurveEditor", Category = "utilities", DisplayName = "Curve Editor", HasIntermediateOutput = true)]
public sealed class CurveEditor
{
    public (Curve Curve, UiOutput? Ui) Execute(
        [Input(Type = "CURVE", Json = """{"points": [[0.0, 0.0], [1.0, 1.0]], "interpolation": "monotone_cubic"}""")] object curve,
        [Input(Optional = true)] Histogram? histogram = null)
    {
        // Measured with ComfyUI: no ui without a histogram; with one, ui {"histogram": [counts]}.
        var ui = histogram is null ? null : new UiOutput("histogram", histogram.Bins.ToArray());
        return (Curve.FromRaw(curve), ui);
    }
}
