namespace SpawnGen.Editor;

/// <summary>The curve a link is drawn with: a cubic Bezier leaving the output to the right and entering the input from the left, like ComfyUI.</summary>
public static class LinkGeometry
{
    public static (Point P0, Point C1, Point C2, Point P3) Curve(Point from, Point to)
    {
        double reach = Math.Max(30, from.DistanceTo(to) * 0.25);
        return (from, new Point(from.X + reach, from.Y), new Point(to.X - reach, to.Y), to);
    }

    public static Point At(Point from, Point to, double t)
    {
        var (p0, c1, c2, p3) = Curve(from, to);
        double u = 1 - t;
        double a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
        return new Point(a * p0.X + b * c1.X + c * c2.X + d * p3.X, a * p0.Y + b * c1.Y + c * c2.Y + d * p3.Y);
    }

    /// <summary>The curve as a polyline, for drawing and for distance tests.</summary>
    public static IReadOnlyList<Point> Sample(Point from, Point to, int segments = 32)
    {
        var points = new Point[segments + 1];
        for (int i = 0; i <= segments; i++) points[i] = At(from, to, (double)i / segments);
        return points;
    }

    /// <summary>Distance from <paramref name="p"/> to the curve (to the nearest point on its polyline).</summary>
    public static double Distance(Point p, Point from, Point to, int segments = 32)
    {
        var pts = Sample(from, to, segments);
        double best = double.MaxValue;
        for (int i = 0; i < pts.Count - 1; i++) best = Math.Min(best, DistanceToSegment(p, pts[i], pts[i + 1]));
        return best;
    }

    public static Point Midpoint(Point from, Point to) => At(from, to, 0.5);

    static double DistanceToSegment(Point p, Point a, Point b)
    {
        double dx = b.X - a.X, dy = b.Y - a.Y, len2 = dx * dx + dy * dy;
        if (len2 == 0) return p.DistanceTo(a);
        double t = Math.Clamp(((p.X - a.X) * dx + (p.Y - a.Y) * dy) / len2, 0, 1);
        return p.DistanceTo(new Point(a.X + t * dx, a.Y + t * dy));
    }
}
