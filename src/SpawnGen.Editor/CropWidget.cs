using System.Globalization;
using SpawnGen.Engine.Types;

namespace SpawnGen.Editor;

/// <summary>
/// Crop Image's crop region (ImageCropV2's BOUNDING_BOX widget): the input picture with the region's rectangle on it, a Ratio choice with an aspect
/// lock, and the X / Y / Width / Height fields. Its value is ComfyUI's <c>{"x", "y", "width", "height"}</c> in picture pixels.
/// </summary>
/// <remarks>
/// Measured on ComfyUI's own editor (its Node 2.0 view, black box: tools/studio-browser-test/probe-comfy-crop.js, 2026-11-14, a 768 x 768 picture):
/// a drag inside the rectangle moves it by the drag rounded to whole pixels, kept inside the picture; a drag on a corner moves that corner (the
/// opposite one stays), kept inside the picture, with a side never under 16; with a ratio chosen, or the lock on (the current aspect), the drag is
/// projected on the corner's outward direction along the aspect, the width rounded and the height taken from it; choosing a ratio keeps the width
/// and sets the height; a drag outside the rectangle does nothing; typed values are stored as they are. ComfyUI's look is its own.
/// </remarks>
public static class CropWidget
{
    public const string TypeName = "BOUNDING_BOX";
    public const double MinSide = 16;
    public const double HandleReach = 7, RowHeight = 24, RatioRowHeight = 26;
    public static readonly string[] Ratios = ["1:1", "3:4", "4:3", "16:9", "9:16", "Custom"];
    public static readonly string[] Fields = ["x", "y", "width", "height"];

    public readonly record struct Region(double X, double Y, double W, double H);

    public static Region Read(object? value)
    {
        static double Get(DictValue d, string key) =>
            d.GetValueOrDefault(key) is { } v and not string ? Convert.ToDouble(v, CultureInfo.InvariantCulture)
            : d.GetValueOrDefault(key) is string s && double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var p) ? p : 0;
        return value is DictValue d ? new Region(Get(d, "x"), Get(d, "y"), Get(d, "width"), Get(d, "height")) : new Region(0, 0, 512, 512);
    }

    public static DictValue Make(Region r) => new() { ["x"] = Store(r.X), ["y"] = Store(r.Y), ["width"] = Store(r.W), ["height"] = Store(r.H) };

    static object Store(double v) => v == Math.Floor(v) && Math.Abs(v) < 1e15 ? (object)(long)v : v;

    /// <summary>JavaScript's Math.round: halves go up.</summary>
    public static double Round(double v) => Math.Floor(v + 0.5);

    /// <summary>A ratio's width : height, or null for Custom.</summary>
    public static (double W, double H)? Aspect(string ratio)
    {
        var parts = ratio.Split(':');
        return parts.Length == 2 && double.TryParse(parts[0], NumberStyles.Float, CultureInfo.InvariantCulture, out var w)
            && double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var h) && w > 0 && h > 0 ? (w, h) : null;
    }

    /// <summary>Choosing a ratio: the width stays, the height follows it (16:9 on 512 wide: 288).</summary>
    public static Region ApplyRatio(Region r, string ratio) => Aspect(ratio) is { } a ? r with { H = Round(r.W * a.H / a.W) } : r;

    /// <summary>The rectangle moved by (dx, dy) picture pixels, the drag rounded, kept inside a picture of <paramref name="pw"/> x <paramref name="ph"/>.</summary>
    public static Region Move(Region r, double dx, double dy, double pw, double ph) =>
        r with { X = Math.Clamp(r.X + Round(dx), 0, Math.Max(0, pw - r.W)), Y = Math.Clamp(r.Y + Round(dy), 0, Math.Max(0, ph - r.H)) };

    [Flags]
    public enum Corner { None = 0, Left = 1, Right = 2, Top = 4, Bottom = 8 }

    /// <summary>
    /// The rectangle with <paramref name="corner"/> dragged by (dx, dy), the opposite corner fixed, inside the picture, sides at least 16; with
    /// <paramref name="aspect"/> (a ratio or the lock) the drag is projected on the corner's outward direction along it and the height follows the
    /// rounded width.
    /// </summary>
    public static Region Resize(Region r, Corner corner, double dx, double dy, (double W, double H)? aspect, double pw, double ph)
    {
        double sx = corner.HasFlag(Corner.Left) ? -1 : 1, sy = corner.HasFlag(Corner.Top) ? -1 : 1;
        // The fixed corner and how much room the picture leaves from it in the dragged directions.
        double fx = sx < 0 ? r.X + r.W : r.X, fy = sy < 0 ? r.Y + r.H : r.Y;
        double roomW = sx < 0 ? fx : pw - fx, roomH = sy < 0 ? fy : ph - fy;
        double w, h;
        if (aspect is { } a)
        {
            double along = (sx * dx * a.W + sy * dy * a.H) / (a.W * a.W + a.H * a.H);
            w = Round(r.W + along * a.W);
            // At least 16 on the shorter side, at most what the picture leaves: then the height from the width.
            double minW = a.W >= a.H ? Round(MinSide * a.W / a.H) : MinSide;
            double maxW = Math.Min(roomW, Math.Floor(roomH * a.W / a.H));
            w = Math.Clamp(w, Math.Min(minW, maxW), Math.Max(minW, maxW));
            h = Round(w * a.H / a.W);
        }
        else
        {
            w = Math.Clamp(r.W + sx * Round(dx), MinSide, Math.Max(MinSide, roomW));
            h = Math.Clamp(r.H + sy * Round(dy), MinSide, Math.Max(MinSide, roomH));
        }
        return new Region(sx < 0 ? fx - w : fx, sy < 0 ? fy - h : fy, w, h);
    }

    // ---- layout (world units) ----

    /// <summary>Where the parts are: the picture box (square), the ratio row (its choice and the lock), the four fields.</summary>
    public sealed record Layout(Rect PictureBox, Rect RatioLabel, Rect RatioButton, Rect LockButton, IReadOnlyList<(string Field, Rect Label, Rect Value)> FieldRows);

    public static double Height(double rowWidth) => rowWidth + RatioRowHeight + Fields.Length * RowHeight + 4;

    public static Layout Lay(Rect row)
    {
        var picture = new Rect(row.X, row.Y, row.Width, row.Width);
        double y = picture.Bottom + 4;
        var label = new Rect(row.X + 4, y, 42, RatioRowHeight - 6);
        var ratio = new Rect(label.Right + 4, y, 86, RatioRowHeight - 6);
        var lockRect = new Rect(ratio.Right + 6, y, RatioRowHeight - 6, RatioRowHeight - 6);
        y += RatioRowHeight;
        var rows = new List<(string, Rect, Rect)>();
        foreach (var f in Fields)
        {
            rows.Add((f, new Rect(row.X + 4, y, 50, RowHeight - 4), new Rect(row.X + 58, y, row.Width - 62, RowHeight - 4)));
            y += RowHeight;
        }
        return new Layout(picture, label, ratio, lockRect, rows);
    }

    /// <summary>The picture inside its box, its aspect kept.</summary>
    public static Rect Fit(Rect box, double pw, double ph)
    {
        if (pw <= 0 || ph <= 0) return box;
        double s = Math.Min(box.Width / pw, box.Height / ph);
        double w = pw * s, h = ph * s;
        return new Rect(box.X + (box.Width - w) / 2, box.Y + (box.Height - h) / 2, w, h);
    }

    public static Rect RegionRect(Rect picture, double pw, double ph, Region r) =>
        new(picture.X + r.X / pw * picture.Width, picture.Y + r.Y / ph * picture.Height, r.W / pw * picture.Width, r.H / ph * picture.Height);

    public static Corner CornerAt(Rect region, Point world)
    {
        foreach (var (c, x, y) in new[] { (Corner.Left | Corner.Top, region.X, region.Y), (Corner.Right | Corner.Top, region.Right, region.Y),
                     (Corner.Left | Corner.Bottom, region.X, region.Bottom), (Corner.Right | Corner.Bottom, region.Right, region.Bottom) })
            if (Math.Abs(world.X - x) <= HandleReach && Math.Abs(world.Y - y) <= HandleReach) return c;
        return Corner.None;
    }
}
