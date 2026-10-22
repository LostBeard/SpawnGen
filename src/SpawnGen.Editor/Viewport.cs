namespace SpawnGen.Editor;

public readonly record struct Point(double X, double Y)
{
    public static Point operator +(Point a, Point b) => new(a.X + b.X, a.Y + b.Y);
    public static Point operator -(Point a, Point b) => new(a.X - b.X, a.Y - b.Y);
    public double DistanceTo(Point o) => Math.Sqrt((X - o.X) * (X - o.X) + (Y - o.Y) * (Y - o.Y));
}

public readonly record struct Rect(double X, double Y, double Width, double Height)
{
    public double Right => X + Width;
    public double Bottom => Y + Height;
    public Point Center => new(X + Width / 2, Y + Height / 2);
    public bool Contains(Point p) => p.X >= X && p.X <= Right && p.Y >= Y && p.Y <= Bottom;
    public bool Intersects(Rect o) => X <= o.Right && Right >= o.X && Y <= o.Bottom && Bottom >= o.Y;

    /// <summary>The rectangle spanned by two corner points in any order (a drag-select box).</summary>
    public static Rect FromCorners(Point a, Point b) => new(Math.Min(a.X, b.X), Math.Min(a.Y, b.Y), Math.Abs(a.X - b.X), Math.Abs(a.Y - b.Y));

    public static Rect Union(Rect a, Rect b)
    {
        var x = Math.Min(a.X, b.X);
        var y = Math.Min(a.Y, b.Y);
        return new(x, y, Math.Max(a.Right, b.Right) - x, Math.Max(a.Bottom, b.Bottom) - y);
    }
}

/// <summary>Maps between world coordinates (where nodes live) and screen coordinates (canvas pixels): <c>screen = world * Scale + Offset</c>.</summary>
public sealed class Viewport
{
    public const double MinScale = 0.1;
    public const double MaxScale = 4.0;

    public double Scale { get; private set; } = 1;
    public double OffsetX { get; private set; }
    public double OffsetY { get; private set; }

    public Point ScreenToWorld(Point s) => new((s.X - OffsetX) / Scale, (s.Y - OffsetY) / Scale);
    public Point WorldToScreen(Point w) => new(w.X * Scale + OffsetX, w.Y * Scale + OffsetY);

    /// <summary>The world-space rectangle visible in a canvas of the given pixel size.</summary>
    public Rect VisibleWorld(double screenWidth, double screenHeight) =>
        new(-OffsetX / Scale, -OffsetY / Scale, screenWidth / Scale, screenHeight / Scale);

    public void Pan(double screenDx, double screenDy)
    {
        OffsetX += screenDx;
        OffsetY += screenDy;
    }

    /// <summary>Zooms by <paramref name="factor"/> keeping the world point under <paramref name="screenAnchor"/> fixed (wheel zoom at the cursor).</summary>
    public void ZoomAt(Point screenAnchor, double factor)
    {
        var world = ScreenToWorld(screenAnchor);
        Scale = Math.Clamp(Scale * factor, MinScale, MaxScale);
        OffsetX = screenAnchor.X - world.X * Scale;
        OffsetY = screenAnchor.Y - world.Y * Scale;
    }

    /// <summary>Fits <paramref name="world"/> into a canvas with some padding (zoom to all / to selection). Never zooms in past 1:1.</summary>
    public void Fit(Rect world, double screenWidth, double screenHeight, double padding = 40)
    {
        if (world.Width <= 0 || world.Height <= 0 || screenWidth <= 2 * padding || screenHeight <= 2 * padding) return;
        Scale = Math.Clamp(Math.Min((screenWidth - 2 * padding) / world.Width, (screenHeight - 2 * padding) / world.Height), MinScale, 1.0);
        OffsetX = (screenWidth - world.Width * Scale) / 2 - world.X * Scale;
        OffsetY = (screenHeight - world.Height * Scale) / 2 - world.Y * Scale;
    }

    public void Reset()
    {
        Scale = 1;
        OffsetX = OffsetY = 0;
    }
}
