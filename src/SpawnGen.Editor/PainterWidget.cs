using System.Globalization;

namespace SpawnGen.Editor;

/// <summary>
/// Painter's canvas (the PAINTER widget on its <c>mask</c> input): a picture painted with a round brush and an eraser, uploaded when the prompt is
/// built and named in <c>mask</c>. This class is the widget's logic: the brush settings, the stamp, where a stroke puts its stamps, the canvas
/// size, which rows show and where they are. The Studio keeps the painting itself in the page (JS canvases, AGENTS.md rule 4) and draws it.
/// </summary>
/// <remarks>
/// Measured on ComfyUI's own widget (black box, its Node 2.0 view, 2026-11-25 .. 27: tools/studio-browser-test/probe-comfy-painter*.js,
/// out/painter-brush, docs/PLANS.md "Now: Painter"):
/// <list type="bullet">
/// <item>Cursor Size is the stamp's diameter in canvas pixels, centred on the pointer's canvas position (a click at (256, 256) is the pixel
/// corner: size 50 paints x 231..280).</item>
/// <item>Hardness h: alpha falls linearly from r x h to r x (1 + (1 - h) / 2) (r = size / 2), measured at pixel centres and rounded: every
/// ramp pixel of nine stamps exact. At 100 it is a hard disc with an antialiased rim (the browser's rasteriser; not matched pixel for pixel).</item>
/// <item>The opacity field applies once per stroke: inside a stroke nothing builds up past the stroke's own coverage, separate strokes
/// composite source-over (two 50 % dots overlap at 192).</item>
/// <item>A soft stroke builds up along its path and does not depend on how the path is cut into pointer events (1, 2, 30 and 300 events:
/// byte-identical). Stamps every r / 5, source-over inside the stroke, come to a mean 0.25 / 255 of ComfyUI's (max 31 at rim pixels): the
/// exact rule inside a stroke is still open (PLANS).</item>
/// <item>The Eraser has only Cursor Size, Width and Height: hard and full, whatever the brush's opacity and hardness.</item>
/// <item>The canvas is the linked picture's size, else Width x Height; a new size keeps the painting's pixels from the top-left (not
/// scaled) and what falls outside is gone. With a picture linked the Width, Height and Background rows are hidden and a "W x H" label
/// shows under the canvas.</item>
/// <item>The upload is "painter-&lt;node id&gt;-&lt;milliseconds&gt;.png" into the input folder; <c>mask</c> becomes "&lt;that&gt; [input]".</item>
/// </list>
/// </remarks>
public static class PainterWidget
{
    public const string TypeName = "PAINTER";
    public const string WidthInput = "width", HeightInput = "height", BackgroundInput = "bg_color";

    public const int MinSize = 1, MaxSize = 200, DefaultSize = 20;
    public const int DefaultHardness = 100, DefaultOpacity = 100;
    public const string DefaultColor = "#ffffff";

    public enum Tool { Brush, Eraser }

    /// <summary>The brush as the widget's controls set it (opacity and hardness in percent, as shown).</summary>
    public sealed record Brush(Tool Tool = Tool.Brush, int Size = DefaultSize, string Color = DefaultColor, int Opacity = DefaultOpacity, int Hardness = DefaultHardness)
    {
        public Brush Clamped() => this with
        {
            Size = Math.Clamp(Size, MinSize, MaxSize),
            Opacity = Math.Clamp(Opacity, 0, 100),
            Hardness = Math.Clamp(Hardness, 0, 100),
            Color = NormalizeColor(Color) ?? DefaultColor,
        };

        /// <summary>What a stroke of this brush actually uses: the Eraser is always hard and full.</summary>
        public Brush Effective() => Tool == Tool.Eraser ? this with { Opacity = 100, Hardness = 100 } : this;
    }

    // ---- the stamp ----

    public static double Radius(int size) => size / 2.0;

    /// <summary>Where the stamp starts to fade: r x hardness.</summary>
    public static double InnerRadius(int size, int hardness) => Radius(size) * hardness / 100.0;

    /// <summary>Where the stamp has faded out: r x (1 + (1 - hardness) / 2); r itself for a hard stamp.</summary>
    public static double OuterRadius(int size, int hardness) => Radius(size) * (1 + (1 - hardness / 100.0) / 2);

    /// <summary>
    /// The stamp's coverage (0..1) at distance <paramref name="d"/> from its centre, before antialiasing: 1 inside the inner radius, falling
    /// linearly to 0 at the outer radius. A hard stamp (inner = outer) is 1 up to its radius.
    /// </summary>
    public static double Coverage(double d, int size, int hardness)
    {
        double rin = InnerRadius(size, hardness), rout = OuterRadius(size, hardness);
        if (rout - rin < 1e-9) return d <= rout ? 1 : 0;
        return Math.Clamp((rout - d) / (rout - rin), 0, 1);
    }

    /// <summary>The coverage as the painting stores it (0..255, rounded half up), at the centre of pixel (x, y) for a stamp at (cx, cy).</summary>
    public static int StampAlpha(int x, int y, double cx, double cy, int size, int hardness) =>
        (int)Math.Floor(255 * Coverage(Math.Sqrt((x + 0.5 - cx) * (x + 0.5 - cx) + (y + 0.5 - cy) * (y + 0.5 - cy)), size, hardness) + 0.5);

    /// <summary>
    /// The radial gradient the canvas draws a soft stamp with, as stops from the centre (offset 0..1 of the outer radius): opaque to the inner
    /// radius, transparent at the outer one. A hard stamp is a filled circle instead (null).
    /// </summary>
    public static (double Offset, double Alpha)[]? GradientStops(int size, int hardness)
    {
        double rin = InnerRadius(size, hardness), rout = OuterRadius(size, hardness);
        if (rout - rin < 1e-9) return null;
        return [(rin / rout, 1), (1, 0)];
    }

    /// <summary>The distance between stamps along a stroke: r / 5 (measured fit), at least a quarter pixel.</summary>
    public static double Spacing(int size) => Math.Max(0.25, Radius(size) / 5);

    /// <summary>
    /// Places a stroke's stamps along the pointer's path at a fixed spacing carried from one pointer event to the next, so the stroke does
    /// not depend on how the path was cut into events (as measured on ComfyUI). The first stamp is where the press was.
    /// </summary>
    public sealed class Stroke
    {
        readonly double _spacing;
        double _x, _y, _carry;

        public Stroke(double x, double y, int size)
        {
            _spacing = Spacing(size);
            (_x, _y) = (x, y);
        }

        /// <summary>The stamp of the press itself.</summary>
        public (double X, double Y) First => (_x, _y);

        /// <summary>The stamps the pointer's move to (x, y) adds, in order.</summary>
        public List<(double X, double Y)> MoveTo(double x, double y)
        {
            var stamps = new List<(double, double)>();
            double dx = x - _x, dy = y - _y, len = Math.Sqrt(dx * dx + dy * dy);
            if (len <= 0) return stamps;
            // the next stamp is one spacing after the last one: _carry is how far along the path the last stamp already is past (_x, _y)
            double t = _spacing - _carry;
            while (t <= len + 1e-9)
            {
                stamps.Add((_x + dx * t / len, _y + dy * t / len));
                t += _spacing;
            }
            _carry = len - (t - _spacing);
            (_x, _y) = (x, y);
            return stamps;
        }
    }

    // ---- the canvas ----

    /// <summary>The canvas size: the linked picture's when there is one, else the Width x Height widgets (64..4096, as the node takes them).</summary>
    public static (int W, int H) CanvasSize((int W, int H)? linkedPicture, object? width, object? height) =>
        linkedPicture is { W: > 0, H: > 0 } p ? p : (Side(width), Side(height));

    static int Side(object? v)
    {
        double d = v switch
        {
            null => 512,
            string s when double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var p) => p,
            string => 512,
            IConvertible c => c.ToDouble(CultureInfo.InvariantCulture),
            _ => 512,
        };
        return (int)Math.Clamp(Math.Round(d), 64, 4096);
    }

    /// <summary>
    /// What survives a resize from <paramref name="from"/> to <paramref name="to"/>: the region copied from the old painting's top-left to the new
    /// one's (not scaled); everything outside it is gone.
    /// </summary>
    public static (int W, int H) KeptOnResize((int W, int H) from, (int W, int H) to) => (Math.Min(from.W, to.W), Math.Min(from.H, to.H));

    // ---- the rows ----

    public enum Row { Tool, CursorSize, Color, Hardness, Width, Height, Background, Clear }

    /// <summary>The rows under the canvas, in ComfyUI's order: the Eraser hides the colour and Hardness, a linked picture Width, Height and Background.</summary>
    public static IReadOnlyList<Row> Rows(Tool tool, bool pictureLinked)
    {
        var rows = new List<Row> { Row.Tool, Row.CursorSize };
        if (tool == Tool.Brush) { rows.Add(Row.Color); rows.Add(Row.Hardness); }
        if (!pictureLinked) { rows.Add(Row.Width); rows.Add(Row.Height); rows.Add(Row.Background); }
        rows.Add(Row.Clear);
        return rows;
    }

    public static string RowLabel(Row row) => row switch
    {
        Row.Tool => "Tool",
        Row.CursorSize => "Cursor Size",
        Row.Color => "Color Picker",
        Row.Hardness => "Hardness",
        Row.Width => "Width",
        Row.Height => "Height",
        Row.Background => "Background",
        _ => "Clear",
    };

    /// <summary>The label under the canvas while a picture is linked ("512 × 512", as ComfyUI's widget shows it).</summary>
    public static string SizeLabel(int w, int h) => $"{w} × {h}";

    public const double RowHeight = 28, SizeLabelHeight = 18;

    /// <summary>Where the parts are, in world units: the square canvas box, the size label (with a picture linked) and each row.</summary>
    public sealed record Layout(Rect CanvasBox, Rect? SizeLabel, IReadOnlyList<(Row Row, Rect Bounds)> Rows);

    public static double Height(double rowWidth, Tool tool, bool pictureLinked) =>
        rowWidth + (pictureLinked ? SizeLabelHeight : 0) + 4 + Rows(tool, pictureLinked).Count * RowHeight;

    public static Layout Lay(Rect row, Tool tool, bool pictureLinked)
    {
        var box = new Rect(row.X, row.Y, row.Width, row.Width);
        double y = box.Bottom + 4;
        Rect? label = null;
        if (pictureLinked) { label = new Rect(row.X, y - 2, row.Width, SizeLabelHeight); y += SizeLabelHeight; }
        var rows = new List<(Row, Rect)>();
        foreach (var r in Rows(tool, pictureLinked)) { rows.Add((r, new Rect(row.X, y, row.Width, RowHeight - 4))); y += RowHeight; }
        return new Layout(box, label, rows);
    }

    /// <summary>The canvas inside its box, its aspect kept and centred (a 256 x 640 canvas is narrower than the box).</summary>
    public static Rect Fit(Rect box, int w, int h)
    {
        if (w <= 0 || h <= 0) return box;
        double s = Math.Min(box.Width / w, box.Height / h);
        return new Rect(box.X + (box.Width - w * s) / 2, box.Y + (box.Height - h * s) / 2, w * s, h * s);
    }

    /// <summary>What a press on the widget lands on.</summary>
    public enum Part { None, Canvas, Brush, Eraser, SizeSlider, Color, Opacity, HardnessSlider, WidthSlider, HeightSlider, Background, Clear }

    /// <summary>A row's control area: right of its label (the Clear row is all control).</summary>
    public static Rect Control(Rect row) => new(row.X + row.Width * 0.3, row.Y, row.Width * 0.7, row.Height);

    /// <summary>A slider row's track: the control less the value shown at its right.</summary>
    public static Rect Track(Rect row) { var c = Control(row); return new Rect(c.X + 6, c.Y, Math.Max(10, c.Width - 58), c.Height); }

    /// <summary>The colour row's opacity field (the percent at its right); the rest of the control is the colour.</summary>
    public static Rect OpacityField(Rect row) { var c = Control(row); return new Rect(c.Right - 52, c.Y, 52, c.Height); }

    /// <summary>A slider row's range and step (Cursor Size 1..200, Hardness 0..100, Width and Height 64..4096 by 64), or null.</summary>
    public static (int Min, int Max, int Step)? Range(Row row) => row switch
    {
        Row.CursorSize => (MinSize, MaxSize, 1),
        Row.Hardness => (0, 100, 1),
        Row.Width or Row.Height => (64, 4096, 64),
        _ => null,
    };

    /// <summary>The value a slider takes at world x: the track's fraction of the range, to the nearest step, clamped.</summary>
    public static int SliderValue(Rect track, double x, int min, int max, int step)
    {
        double f = Math.Clamp((x - track.X) / track.Width, 0, 1);
        return Math.Clamp(min + (int)Math.Round(f * (max - min) / step) * step, min, max);
    }

    /// <summary>Where a slider's thumb sits for a value.</summary>
    public static double ThumbX(Rect track, int value, int min, int max) => track.X + track.Width * Math.Clamp((value - min) / (double)(max - min), 0, 1);

    /// <summary>The part of the widget at a world point (<paramref name="shown"/> is the canvas as drawn, see <see cref="Fit"/>).</summary>
    public static (Part Part, Row? Row, Rect Bounds) PartAt(Layout lay, Rect shown, Point world)
    {
        if (shown.Contains(world)) return (Part.Canvas, null, shown);
        foreach (var (row, b) in lay.Rows)
        {
            if (!b.Contains(world)) continue;
            var c = Control(b);
            if (row == Row.Clear) return (Part.Clear, row, b);
            if (!c.Contains(world)) return (Part.None, row, b);
            return row switch
            {
                Row.Tool => (world.X < c.X + c.Width / 2 ? Part.Brush : Part.Eraser, row, b),
                Row.CursorSize => (Part.SizeSlider, row, b),
                Row.Hardness => (Part.HardnessSlider, row, b),
                Row.Width => (Part.WidthSlider, row, b),
                Row.Height => (Part.HeightSlider, row, b),
                Row.Color => (OpacityField(b).Contains(world) ? Part.Opacity : Part.Color, row, b),
                Row.Background => (Part.Background, row, b),
                _ => (Part.None, row, b),
            };
        }
        return (Part.None, null, default);
    }

    /// <summary>A world point as a canvas position (fractional pixels: a stamp centre).</summary>
    public static (double X, double Y) ToCanvas(Rect shown, int w, int h, Point world) =>
        ((world.X - shown.X) / shown.Width * w, (world.Y - shown.Y) / shown.Height * h);

    /// <summary>The widget names <see cref="EditorView.WidgetActivated"/> reports for the brush's colour and opacity (they are not inputs).</summary>
    public const string ColorEdit = "painter:color", OpacityEdit = "painter:opacity";

    // ---- the upload ----

    /// <summary>The file name the painting is uploaded under: painter-&lt;node id&gt;-&lt;milliseconds since 1970&gt;.png.</summary>
    public static string UploadName(int nodeId, long unixMilliseconds) => $"painter-{nodeId}-{unixMilliseconds}.png";

    /// <summary>The <c>mask</c> value for an uploaded painting: "&lt;name&gt; [input]".</summary>
    public static string MaskValue(string storedName) => $"{storedName} [input]";

    /// <summary>The uploaded file a <c>mask</c> value names (without " [input]"), or null for an empty value.</summary>
    public static string? MaskFile(object? value) =>
        value is string s && s.Trim() is { Length: > 0 } t ? (t.EndsWith(" [input]", StringComparison.Ordinal) ? t[..^8] : t) : null;

    /// <summary>"#rrggbb" in lower case, or null when it is not a colour (the widget's colour input gives "#rrggbb").</summary>
    public static string? NormalizeColor(string? c)
    {
        if (c is null) return null;
        c = c.Trim();
        if (c.Length == 4 && c[0] == '#' && c.Skip(1).All(Uri.IsHexDigit)) c = $"#{c[1]}{c[1]}{c[2]}{c[2]}{c[3]}{c[3]}";
        return c.Length == 7 && c[0] == '#' && c.Skip(1).All(Uri.IsHexDigit) ? c.ToLowerInvariant() : null;
    }
}
