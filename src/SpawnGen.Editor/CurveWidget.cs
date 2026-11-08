using System.Globalization;
using SpawnGen.Engine.Types;

namespace SpawnGen.Editor;

/// <summary>
/// The CURVE widget (Curve Editor, GLSL Shader's curves): its value is ComfyUI's <c>{"points": [[x, y], ...], "interpolation": "monotone_cubic" | "linear"}</c>;
/// the editor draws it in a square plot and edits the points by dragging (a click on empty space adds one, a double-click on one removes it, two stay).
/// </summary>
public static class CurveWidget
{
    public const string TypeName = "CURVE";
    /// <summary>The plot's inset inside the widget row and how close (in world units) the pointer must be to pick a point.</summary>
    public const double Inset = 8, PickRadius = 9, LabelHeight = 22;

    public static List<(double X, double Y)> Points(object? value)
    {
        var raw = value is DictValue d ? d.GetValueOrDefault("points") : value;
        var list = new List<(double, double)>();
        if (raw is ArrayValue a)
            foreach (var p in a)
                if (p is ArrayValue { Count: >= 2 } xy) list.Add((Convert.ToDouble(xy[0], CultureInfo.InvariantCulture), Convert.ToDouble(xy[1], CultureInfo.InvariantCulture)));
        return list.Count > 0 ? list : [(0.0, 0.0), (1.0, 1.0)];
    }

    public static string Interpolation(object? value) =>
        value is DictValue d && d.GetValueOrDefault("interpolation") is "linear" ? "linear" : "monotone_cubic";

    public static DictValue Make(IEnumerable<(double X, double Y)> points, string interpolation)
    {
        var list = new ArrayValue();
        foreach (var (x, y) in points) list.Add(new ArrayValue { Math.Round(x, 4), Math.Round(y, 4) });
        return new DictValue { ["points"] = list, ["interpolation"] = interpolation };
    }

    /// <summary>The square-ish plot inside a widget row (below the interpolation label).</summary>
    public static Rect Plot(Rect row) => new(row.X + Inset, row.Y + LabelHeight, row.Width - 2 * Inset, row.Height - LabelHeight - Inset);

    /// <summary>The interpolation label at the row's top right: a click toggles monotone cubic / linear.</summary>
    public static Rect Label(Rect row) => new(row.Right - 100, row.Y, 100, LabelHeight - 6);

    public static Point ToWorld(Rect plot, (double X, double Y) p) => new(plot.X + p.X * plot.Width, plot.Bottom - p.Y * plot.Height);

    public static (double X, double Y) FromWorld(Rect plot, Point w) =>
        (Math.Clamp((w.X - plot.X) / plot.Width, 0, 1), Math.Clamp((plot.Bottom - w.Y) / plot.Height, 0, 1));

    /// <summary>
    /// The histogram outline behind the curve, as ComfyUI's curve widget draws it (histogramToPath): each bin's square root, scaled so the 99.5th
    /// percentile bin reaches the top, clamped; x spreads the bins over 0..1 and y is the height in 0..1. Empty when there is nothing to draw.
    /// </summary>
    public static List<(double X, double Y)> HistogramOutline(IReadOnlyList<double> bins)
    {
        int n = bins.Count;
        var outline = new List<(double, double)>(n);
        if (n == 0) return outline;
        var roots = bins.Select(b => (double)MathF.Sqrt((float)b)).ToArray(); // Float32Array in the frontend
        double top = roots.Order().ElementAt((int)Math.Floor((n - 1) * 0.995));
        if (top == 0) return outline;
        for (int i = 0; i < n; i++) outline.Add((n == 1 ? 0.5 : i / (double)(n - 1), Math.Min(1, roots[i] * (1 / top))));
        return outline;
    }

    /// <summary>The point under <paramref name="world"/> (within <see cref="PickRadius"/>), or -1.</summary>
    public static int PointAt(Rect plot, IReadOnlyList<(double X, double Y)> points, Point world)
    {
        int best = -1;
        double bestD = PickRadius;
        for (int i = 0; i < points.Count; i++)
        {
            double d = ToWorld(plot, points[i]).DistanceTo(world);
            if (d <= bestD) { bestD = d; best = i; }
        }
        return best;
    }
}
