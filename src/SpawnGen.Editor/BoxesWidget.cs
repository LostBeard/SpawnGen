using System.Globalization;
using SpawnGen.Engine.Types;

namespace SpawnGen.Editor;

/// <summary>
/// The BOUNDING_BOXES widget (Create Bounding Boxes' editor_state): a canvas the boxes are drawn on, with a toolbar (Grid, Clear all) above and the
/// selected box's panel below (obj / text, the text to render, a description, a colour palette). Its value is ComfyUI's list of
/// <c>{"x", "y", "width", "height", "metadata": {"type", "text", "desc", "palette"}}</c> in canvas pixels (the node's width x height).
/// </summary>
/// <remarks>
/// Measured on ComfyUI's own editor (its Node 2.0 view, black box: tools/studio-browser-test/probe-comfy-bboxes*.js, 2026-11-14): a drag draws a box in
/// any direction and stores it normalised; with Grid on (the default) both corners round to the nearest multiple of 32 canvas pixels at any canvas size,
/// with it off the corner and the size round to whole pixels; a drag inside a box moves it, a drag on a corner resizes it, by the drag rounded the same
/// way; a new box is <c>{"type": "obj", "text": "", "desc": "", "palette": []}</c>; "+" adds #ffffff to the palette, a right-click on a swatch removes it;
/// Clear all empties the list. How ComfyUI draws all this is its own; SpawnGen's look is its own.
/// </remarks>
public static class BoxesWidget
{
    public const string TypeName = "BOUNDING_BOXES";
    /// <summary>The snap step with Grid on, in canvas pixels, whatever the canvas size.</summary>
    public const double GridStep = 32;
    /// <summary>The toolbar's height, the panel's height, a corner handle's reach and the gap between parts (world units).</summary>
    public const double ToolbarHeight = 26, PanelHeight = 160, HandleReach = 7, Gap = 4;

    /// <summary>One box: canvas pixels, and its metadata (the dictionary as read, so keys SpawnGen does not edit are kept).</summary>
    public sealed record Box(double X, double Y, double W, double H, DictValue Metadata)
    {
        public string Type => Metadata.GetValueOrDefault("type") as string ?? "obj";
        public string Text => Metadata.GetValueOrDefault("text") as string ?? "";
        public string Desc => Metadata.GetValueOrDefault("desc") as string ?? "";
        public List<string> Palette => Metadata.GetValueOrDefault("palette") is ArrayValue a ? a.Select(c => c as string ?? "#ffffff").ToList() : [];

        public Box With(string key, object? value)
        {
            var m = new DictValue();
            foreach (var kv in Metadata) m[kv.Key] = kv.Value;
            m[key] = value;
            return this with { Metadata = m };
        }

        public bool Contains(double x, double y) => x >= X && x <= X + W && y >= Y && y <= Y + H;
    }

    public static DictValue NewMetadata() => new() { ["type"] = "obj", ["text"] = "", ["desc"] = "", ["palette"] = new ArrayValue() };

    public static List<Box> Boxes(object? value)
    {
        var list = new List<Box>();
        if (value is not ArrayValue a) return list;
        foreach (var item in a)
        {
            if (item is not DictValue d) continue;
            var meta = d.GetValueOrDefault("metadata") as DictValue ?? NewMetadata();
            list.Add(new Box(Num(d.GetValueOrDefault("x")), Num(d.GetValueOrDefault("y")), Num(d.GetValueOrDefault("width")), Num(d.GetValueOrDefault("height")), meta));
        }
        return list;
    }

    public static ArrayValue Make(IEnumerable<Box> boxes)
    {
        var a = new ArrayValue();
        foreach (var b in boxes)
            a.Add(new DictValue { ["x"] = Store(b.X), ["y"] = Store(b.Y), ["width"] = Store(b.W), ["height"] = Store(b.H), ["metadata"] = b.Metadata });
        return a;
    }

    static double Num(object? v) => v switch { null => 0, string s => double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var d) ? d : 0, _ => Convert.ToDouble(v, CultureInfo.InvariantCulture) };

    /// <summary>Whole numbers as integers, as ComfyUI's editor stores them.</summary>
    static object Store(double v) => v == Math.Floor(v) && Math.Abs(v) < 1e15 ? (object)(long)v : v;

    /// <summary>The canvas size: the node's width and height widgets (ComfyUI's defaults 1024).</summary>
    public static (double W, double H) CanvasSize(IReadOnlyDictionary<string, object?> widgets)
    {
        static double Get(IReadOnlyDictionary<string, object?> w, string key) =>
            w.TryGetValue(key, out var v) && v is not null && Num(v) is var d && d >= 1 ? d : 1024;
        return (Get(widgets, "width"), Get(widgets, "height"));
    }

    // ---- rounding (JavaScript's Math.round: halves go up) ----

    public static double Round(double v) => Math.Floor(v + 0.5);

    public static double Snap(double v, bool grid) => grid ? Round(v / GridStep) * GridStep : Round(v);

    /// <summary>A drawn box from two canvas points: normalised; with the grid each corner snaps, without it the corner and the size round. Null when empty.</summary>
    public static (double X, double Y, double W, double H)? Draw((double X, double Y) a, (double X, double Y) b, bool grid)
    {
        double x0 = Math.Min(a.X, b.X), x1 = Math.Max(a.X, b.X), y0 = Math.Min(a.Y, b.Y), y1 = Math.Max(a.Y, b.Y);
        (double X, double Y, double W, double H) r = grid
            ? (Snap(x0, true), Snap(y0, true), Snap(x1, true) - Snap(x0, true), Snap(y1, true) - Snap(y0, true))
            : (Round(x0), Round(y0), Round(x1 - x0), Round(y1 - y0));
        return r.W > 0 && r.H > 0 ? r : null;
    }

    /// <summary>A box moved by a drag of (dx, dy) canvas pixels, the drag rounded as the grid says.</summary>
    public static Box Move(Box b, double dx, double dy, bool grid) => b with { X = b.X + Snap(dx, grid), Y = b.Y + Snap(dy, grid) };

    [Flags]
    public enum Corner { None = 0, Left = 1, Right = 2, Top = 4, Bottom = 8 }

    /// <summary>A box resized by dragging <paramref name="corner"/> by (dx, dy), the drag rounded as the grid says; kept normalised.</summary>
    public static Box Resize(Box b, Corner corner, double dx, double dy, bool grid)
    {
        double sx = Snap(dx, grid), sy = Snap(dy, grid);
        double x0 = b.X, x1 = b.X + b.W, y0 = b.Y, y1 = b.Y + b.H;
        if (corner.HasFlag(Corner.Left)) x0 += sx;
        if (corner.HasFlag(Corner.Right)) x1 += sx;
        if (corner.HasFlag(Corner.Top)) y0 += sy;
        if (corner.HasFlag(Corner.Bottom)) y1 += sy;
        return b with { X = Math.Min(x0, x1), Y = Math.Min(y0, y1), W = Math.Abs(x1 - x0), H = Math.Abs(y1 - y0) };
    }

    // ---- layout (world units) ----

    /// <summary>Where each part of the widget is drawn, for a row of the node.</summary>
    public sealed record Layout(Rect Toolbar, Rect GridButton, Rect ClearButton, Rect Canvas, Rect Panel, Rect ObjButton, Rect TextButton, Rect TextField,
        Rect DescField, Rect PaletteRow);

    /// <summary>The canvas's height for a row this wide: the boxes' aspect, at most twice the row's width (a tall canvas then narrows).</summary>
    public static double CanvasHeight(double rowWidth, double w, double h) => Math.Clamp(rowWidth * h / w, 40, rowWidth * 2);

    public static double RowHeight(double rowWidth, double w, double h) => ToolbarHeight + CanvasHeight(rowWidth, w, h) + Gap + PanelHeight;

    public static Layout Lay(Rect row, double w, double h)
    {
        var toolbar = new Rect(row.X, row.Y, row.Width, ToolbarHeight - Gap);
        var grid = new Rect(toolbar.X + 4, toolbar.Y + 2, 58, toolbar.Height - 4);
        var clear = new Rect(toolbar.Right - 84, toolbar.Y + 2, 80, toolbar.Height - 4);
        double ch = CanvasHeight(row.Width, w, h), cw = Math.Min(row.Width, ch * w / h);
        var canvas = new Rect(row.X + (row.Width - cw) / 2, row.Y + ToolbarHeight, cw, ch);
        var panel = new Rect(row.X, canvas.Bottom + Gap, row.Width, PanelHeight - Gap);
        double half = (panel.Width - 12) / 2;
        var obj = new Rect(panel.X + 4, panel.Y + 4, half, 20);
        var text = new Rect(obj.Right + 4, panel.Y + 4, half, 20);
        var textField = new Rect(panel.X + 4, obj.Bottom + 6, panel.Width - 8, 36);
        var desc = new Rect(panel.X + 4, textField.Bottom + 6, panel.Width - 8, 54);
        var palette = new Rect(panel.X + 4, desc.Bottom + 8, panel.Width - 8, 18);
        return new Layout(toolbar, grid, clear, canvas, panel, obj, text, textField, desc, palette);
    }

    public static Point ToWorld(Rect canvas, double w, double h, double x, double y) => new(canvas.X + x / w * canvas.Width, canvas.Y + y / h * canvas.Height);

    public static (double X, double Y) ToCanvas(Rect canvas, double w, double h, Point world) =>
        ((world.X - canvas.X) / canvas.Width * w, (world.Y - canvas.Y) / canvas.Height * h);

    public static Rect BoxRect(Rect canvas, double w, double h, Box b)
    {
        var p = ToWorld(canvas, w, h, b.X, b.Y);
        var q = ToWorld(canvas, w, h, b.X + b.W, b.Y + b.H);
        return new Rect(p.X, p.Y, q.X - p.X, q.Y - p.Y);
    }

    /// <summary>The topmost box under a canvas point (the last drawn wins), or -1.</summary>
    public static int BoxAt(IReadOnlyList<Box> boxes, (double X, double Y) p)
    {
        for (int i = boxes.Count - 1; i >= 0; i--)
            if (boxes[i].Contains(p.X, p.Y)) return i;
        return -1;
    }

    /// <summary>The corner of a box's rectangle (world units) within <see cref="HandleReach"/> of <paramref name="world"/>, or None.</summary>
    public static Corner CornerAt(Rect box, Point world)
    {
        foreach (var (c, x, y) in new[] { (Corner.Left | Corner.Top, box.X, box.Y), (Corner.Right | Corner.Top, box.Right, box.Y),
                     (Corner.Left | Corner.Bottom, box.X, box.Bottom), (Corner.Right | Corner.Bottom, box.Right, box.Bottom) })
            if (Math.Abs(world.X - x) <= HandleReach && Math.Abs(world.Y - y) <= HandleReach) return c;
        return Corner.None;
    }

    /// <summary>Palette swatch <paramref name="i"/> in the palette row (after its label), and the "+" after the last.</summary>
    public static Rect Swatch(Rect paletteRow, int i) => new(paletteRow.X + 86 + i * 22, paletteRow.Y + 1, 16, paletteRow.Height - 2);

    public static Rect Plus(Rect paletteRow, int count) => new(paletteRow.X + 86 + count * 22, paletteRow.Y + 1, 18, paletteRow.Height - 2);

    /// <summary>The swatch under <paramref name="world"/>, or -1.</summary>
    public static int SwatchAt(Rect paletteRow, int count, Point world)
    {
        for (int i = 0; i < count; i++)
            if (Swatch(paletteRow, i).Contains(world)) return i;
        return -1;
    }

    /// <summary>A box's number tag as ComfyUI shows it: "01", "02", ...</summary>
    public static string Tag(int index) => (index + 1).ToString("00", CultureInfo.InvariantCulture);
}
