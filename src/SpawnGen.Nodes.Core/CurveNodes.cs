using System.Globalization;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// ComfyUI's CURVE (comfy_api CurveInput): control points in [0, 1] with monotone cubic (default) or linear interpolation, as ComfyUI evaluates them
/// (MonotoneCubicCurve / LinearCurve, which mirror the frontend's curve editor).
/// </summary>
[Socket("CURVE")]
public sealed class Curve
{
    public IReadOnlyList<(double X, double Y)> Points { get; }
    public string Interpolation { get; }
    readonly double[] _xs, _ys, _slopes;

    public Curve(IEnumerable<(double X, double Y)> points, string interpolation = "monotone_cubic")
    {
        // Python's sorted is stable: points with equal x keep their order.
        Points = points.Select((p, i) => (p, i)).OrderBy(t => t.p.X).ThenBy(t => t.i).Select(t => t.p).ToList();
        Interpolation = interpolation == "linear" ? "linear" : "monotone_cubic";
        _xs = [.. Points.Select(p => p.X)];
        _ys = [.. Points.Select(p => p.Y)];
        _slopes = Interpolation == "linear" ? [] : Slopes(_xs, _ys);
    }

    /// <summary>ComfyUI's CurveInput.from_raw: a {"points": [[x, y], ...], "interpolation": ...} value or a bare point list.</summary>
    public static Curve FromRaw(object? value)
    {
        if (value is Curve c) return c;
        object? raw = value;
        string interpolation = "monotone_cubic";
        if (value is DictValue d)
        {
            raw = d.GetValueOrDefault("points");
            if (d.GetValueOrDefault("interpolation") is string s) interpolation = s;
        }
        if (raw is not ArrayValue list) throw new ArgumentException("A curve is a list of [x, y] points or {\"points\": [...], \"interpolation\": ...}.");
        var points = list.Select(p => p is ArrayValue { Count: >= 2 } xy
            ? (Number(xy[0]), Number(xy[1]))
            : throw new ArgumentException("Each curve point is [x, y].")).ToList();
        return new Curve(points, interpolation);
    }

    static double Number(object? v) => Convert.ToDouble(v, CultureInfo.InvariantCulture);

    static double[] Slopes(double[] xs, double[] ys)
    {
        int n = xs.Length;
        var slopes = new double[n];
        if (n < 2) return slopes;
        var deltas = new double[n - 1];
        for (int i = 0; i < n - 1; i++)
        {
            double dx = xs[i + 1] - xs[i];
            deltas[i] = dx == 0 ? 0.0 : (ys[i + 1] - ys[i]) / dx;
        }
        slopes[0] = deltas[0];
        slopes[n - 1] = deltas[n - 2];
        for (int i = 1; i < n - 1; i++)
            slopes[i] = deltas[i - 1] * deltas[i] <= 0 ? 0.0 : (deltas[i - 1] + deltas[i]) / 2;
        for (int i = 0; i < n - 1; i++)
        {
            if (deltas[i] == 0) { slopes[i] = 0.0; slopes[i + 1] = 0.0; continue; }
            double alpha = slopes[i] / deltas[i], beta = slopes[i + 1] / deltas[i];
            double s = alpha * alpha + beta * beta;
            if (s > 9)
            {
                double t = 3 / Math.Sqrt(s);
                slopes[i] = t * alpha * deltas[i];
                slopes[i + 1] = t * beta * deltas[i];
            }
        }
        return slopes;
    }

    /// <summary>The curve at <paramref name="x"/> (ComfyUI's interp_array, element by element).</summary>
    public double Interp(double x)
    {
        int n = _xs.Length;
        if (n == 0) return 0.0;
        if (n == 1) return _ys[0];
        if (Interpolation == "linear") return LinearInterp(x);
        if (x <= _xs[0]) return _ys[0];
        if (x >= _xs[n - 1]) return _ys[n - 1];
        // numpy's searchsorted(side='right'), clipped to [1, n - 1].
        int hi = 0;
        while (hi < n && _xs[hi] <= x) hi++;
        hi = Math.Clamp(hi, 1, n - 1);
        int lo = hi - 1;
        double dxs = _xs[hi] - _xs[lo];
        double t = dxs == 0 ? 0.0 : (x - _xs[lo]) / dxs;
        double t2 = t * t, t3 = t2 * t;
        double h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + t, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2;
        return h00 * _ys[lo] + h10 * dxs * _slopes[lo] + h01 * _ys[hi] + h11 * dxs * _slopes[hi];
    }

    /// <summary>numpy's interp: clamped at the ends; between equal x values the last point's y wins.</summary>
    double LinearInterp(double x)
    {
        int n = _xs.Length;
        if (x <= _xs[0]) return x < _xs[0] || n == 1 ? _ys[0] : _ys[UpperIndex(x) - 1];
        if (x >= _xs[n - 1]) return _ys[n - 1];
        int hi = UpperIndex(x);
        int lo = hi - 1;
        if (_xs[lo] == x) return _ys[lo];
        double dx = _xs[hi] - _xs[lo];
        return dx == 0 ? _ys[hi] : _ys[lo] + (x - _xs[lo]) * (_ys[hi] - _ys[lo]) / dx;
    }

    int UpperIndex(double x)
    {
        int hi = 0;
        while (hi < _xs.Length && _xs[hi] <= x) hi++;
        return hi;
    }

    /// <summary>ComfyUI's to_lut: <paramref name="size"/> samples at numpy's linspace(0, 1, size).</summary>
    public double[] ToLut(int size = 256)
    {
        var lut = new double[size];
        for (int i = 0; i < size; i++)
        {
            double x = size == 1 ? 0.0 : i == size - 1 ? 1.0 : i * (1.0 / (size - 1));
            lut[i] = Interp(x);
        }
        return lut;
    }

    public override string ToString() => $"{(Interpolation == "linear" ? "LinearCurve" : "MonotoneCubicCurve")}(points={string.Join(", ", Points.Select(p => $"({p.X}, {p.Y})"))})";
}

[Node(Category = "utilities", DisplayName = "Curve Editor", OutputNames = ["curve"])]
public sealed class CurveEditor
{
    /// <summary>
    /// ComfyUI's Curve Editor: the curve its widget holds (control points, monotone cubic or linear), for GLSL Shader's u_curve inputs and other curve users.
    /// A histogram, when connected, is shown behind the curve (ui "histogram"). The editor edits the points as JSON until it has ComfyUI's curve widget.
    /// </summary>
    public (Curve Curve, UiOutput? Ui) Execute([Input(Type = "CURVE", Json = """{"points": [[0.0, 0.0], [1.0, 1.0]], "interpolation": "monotone_cubic"}""")] object curve,
        Histogram? histogram = null) =>
        (Curve.FromRaw(curve), histogram is null ? null : new UiOutput("histogram", histogram.Bins));
}
