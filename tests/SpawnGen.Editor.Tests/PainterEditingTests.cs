using SpawnGen.Editor;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor.Tests;

/// <summary>
/// Painter's widget driven with the pointer through the editor: what reaches the page's painting (<see cref="IPainterCanvas"/>) and what changes
/// in the node, against the behaviour measured on ComfyUI's widget (PainterWidget's remarks).
/// </summary>
public class PainterEditingTests
{
    static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(SpawnGen.Nodes.Core.SaveImage).Assembly);

    /// <summary>The page side as the editor sees it: every call, in order (the real canvas is checked by the browser drive).</summary>
    sealed class Recorder : IPainterCanvas
    {
        public readonly List<string> Calls = [];
        public readonly List<(double X, double Y)> StampsSeen = [];
        public void BeginStroke(int nodeId, (int W, int H) size, PainterWidget.Brush brush, double x, double y)
        {
            Calls.Add($"begin {nodeId} {size.W}x{size.H} {brush.Tool} size {brush.Size} opacity {brush.Opacity} hardness {brush.Hardness} at {x:F1},{y:F1}");
            StampsSeen.Add((x, y));
        }
        public void Stamps(int nodeId, IReadOnlyList<(double X, double Y)> stamps) { Calls.Add($"stamps {stamps.Count}"); StampsSeen.AddRange(stamps); }
        public void EndStroke(int nodeId) => Calls.Add($"end {nodeId}");
        public void Clear(int nodeId) => Calls.Add($"clear {nodeId}");
    }

    sealed class Rig
    {
        public required EditorView View { get; init; }
        public required EditorNode Node { get; init; }
        public required Recorder Canvas { get; init; }
        public WidgetRow Row => View.Hits.Geometry(Node).Widgets.Single(w => w.Name == "mask");
        public (PainterWidget.Layout Layout, Rect Shown, int W, int H) G => View.PainterGeometry(Node, Row);
        Point S(Point world) => View.Viewport.WorldToScreen(world);
        /// <summary>A canvas position as a world point.</summary>
        public Point At(double x, double y) { var g = G; return new(g.Shown.X + x / g.W * g.Shown.Width, g.Shown.Y + y / g.H * g.Shown.Height); }
        public Rect RowOf(PainterWidget.Row row) => G.Layout.Rows.Single(r => r.Row == row).Bounds;
        public void Down(Point w) => View.PointerDown(S(w), PointerButton.Left);
        public void Move(Point w) => View.PointerMove(S(w));
        public void Up(Point w) => View.PointerUp(S(w), PointerButton.Left);
        public void Click(Point w) { Down(w); Up(w); }
    }

    static Rig Setup()
    {
        var doc = new EditorDocument(Registry());
        var node = doc.AddNode("Painter", 0, 0);
        var canvas = new Recorder();
        return new Rig { View = new EditorView(doc) { PainterCanvas = canvas }, Node = node, Canvas = canvas };
    }

    [Test]
    public void The_painter_is_a_canvas_with_its_rows_and_no_separate_background_row()
    {
        var r = Setup();
        var rows = r.View.Hits.Geometry(r.Node).Widgets;
        Assert.That(r.Row.IsPainter, Is.True);
        Assert.That(rows.Select(w => w.Name), Does.Not.Contain("bg_color"), "Background is a row of the painter widget");
        Assert.That(rows.Select(w => w.Name), Does.Not.Contain("width").And.Not.Contain("height"), "hidden inputs");
        Assert.That(r.View.Hits.Geometry(r.Node).Bounds.Width, Is.EqualTo(450), "ComfyUI opens it 450 wide");
        Assert.That(r.G.Layout.Rows.Select(x => x.Row), Is.EqualTo(PainterWidget.Rows(PainterWidget.Tool.Brush, false)));
        Assert.That((r.G.W, r.G.H), Is.EqualTo((512, 512)));
        Assert.That(r.G.Shown.Width, Is.EqualTo(r.Row.Bounds.Width).Within(1e-9), "the canvas fills the box across");
    }

    [Test]
    public void A_stroke_reaches_the_page_as_stamps_in_canvas_pixels()
    {
        var r = Setup();
        r.Down(r.At(100, 256));
        r.Move(r.At(250, 256));
        r.Move(r.At(400, 256));
        r.Up(r.At(400, 256));
        Assert.That(r.Canvas.Calls[0], Is.EqualTo("begin 1 512x512 Brush size 20 opacity 100 hardness 100 at 100.0,256.0"));
        Assert.That(r.Canvas.Calls[^1], Is.EqualTo("end 1"));
        // size 20: a stamp every 2 canvas pixels from 100 to 400, however the moves came
        Assert.That(r.Canvas.StampsSeen, Has.Count.EqualTo(151));
        Assert.That(r.Canvas.StampsSeen[^1].X, Is.EqualTo(400).Within(1e-6));
        Assert.That(r.Canvas.StampsSeen.All(p => Math.Abs(p.Y - 256) < 1e-6), Is.True);
        Assert.That(r.Node.Widgets.GetValueOrDefault("mask") ?? "", Is.EqualTo(""), "a stroke changes no widget value (ComfyUI sends nothing until the prompt)");
    }

    [Test]
    public void The_eraser_paints_hard_and_full_and_hides_colour_and_hardness()
    {
        var r = Setup();
        r.View.SetPainterBrush(r.Node.Id, r.Node.PainterBrush with { Opacity = 50, Hardness = 0, Size = 50 });
        var tool = PainterWidget.Control(r.RowOf(PainterWidget.Row.Tool));
        r.Click(new Point(tool.X + tool.Width * 0.75, tool.Center.Y));
        Assert.That(r.Node.PainterBrush.Tool, Is.EqualTo(PainterWidget.Tool.Eraser));
        Assert.That(r.G.Layout.Rows.Select(x => x.Row), Is.EqualTo(PainterWidget.Rows(PainterWidget.Tool.Eraser, false)));
        r.Down(r.At(256, 256)); r.Up(r.At(256, 256));
        Assert.That(r.Canvas.Calls[0], Is.EqualTo("begin 1 512x512 Eraser size 50 opacity 100 hardness 100 at 256.0,256.0"));
        // back to the Brush: its own settings are still there
        tool = PainterWidget.Control(r.RowOf(PainterWidget.Row.Tool));
        r.Click(new Point(tool.X + tool.Width * 0.25, tool.Center.Y));
        Assert.That(r.Node.PainterBrush, Is.EqualTo(new PainterWidget.Brush(PainterWidget.Tool.Brush, 50, "#ffffff", 50, 0)));
    }

    [Test]
    public void The_sliders_set_the_brush_and_the_canvas_size()
    {
        var r = Setup();
        var track = PainterWidget.Track(r.RowOf(PainterWidget.Row.CursorSize));
        r.Down(new Point(track.X + track.Width / 2, track.Center.Y));
        r.Move(new Point(track.Right + 50, track.Center.Y));
        r.Up(new Point(track.Right + 50, track.Center.Y));
        Assert.That(r.Node.PainterBrush.Size, Is.EqualTo(200), "dragged past the end: the maximum");
        track = PainterWidget.Track(r.RowOf(PainterWidget.Row.Hardness));
        r.Click(new Point(track.X + track.Width / 2, track.Center.Y));
        Assert.That(r.Node.PainterBrush.Hardness, Is.EqualTo(50));
        // Width is the node's input (saved and sent): 64 steps, and the canvas follows
        track = PainterWidget.Track(r.RowOf(PainterWidget.Row.Width));
        r.Click(new Point(track.X + 0.0469 * track.Width, track.Center.Y));
        Assert.That(r.Node.Widgets["width"], Is.EqualTo(256L));
        Assert.That((r.G.W, r.G.H), Is.EqualTo((256, 512)));
        Assert.That(r.G.Shown.Height, Is.EqualTo(r.G.Layout.CanvasBox.Height).Within(1e-9), "a narrow canvas fits the box's height");
    }

    [Test]
    public void A_linked_picture_sets_the_canvas_size_and_hides_width_height_background()
    {
        var r = Setup();
        var doc = r.View.Document;
        var load = doc.AddNode("LoadImage", -400, 0);
        Assert.That(doc.Connect(load.Id, 0, r.Node.Id, "image", out var error), Is.Not.Null, error);
        r.Node.PainterPicture = (300, 200);
        Assert.That((r.G.W, r.G.H), Is.EqualTo((300, 200)));
        Assert.That(r.G.Layout.SizeLabel, Is.Not.Null);
        Assert.That(r.G.Layout.Rows.Select(x => x.Row), Is.EqualTo(PainterWidget.Rows(PainterWidget.Tool.Brush, true)));
        // until the page has the picture, the canvas is Width x Height
        r.Node.PainterPicture = null;
        Assert.That((r.G.W, r.G.H), Is.EqualTo((512, 512)));
    }

    [Test]
    public void Clear_colour_opacity_and_background()
    {
        var r = Setup();
        var asked = new List<string>();
        r.View.WidgetActivated += (_, name) => asked.Add(name);
        var colour = PainterWidget.Control(r.RowOf(PainterWidget.Row.Color));
        r.Click(new Point(colour.X + 10, colour.Center.Y));
        var opacity = PainterWidget.OpacityField(r.RowOf(PainterWidget.Row.Color));
        r.Click(opacity.Center);
        r.Click(PainterWidget.Control(r.RowOf(PainterWidget.Row.Background)).Center);
        r.Click(r.RowOf(PainterWidget.Row.Clear).Center);
        Assert.That(asked, Is.EqualTo(new[] { PainterWidget.ColorEdit, PainterWidget.OpacityEdit, "bg_color" }));
        Assert.That(r.Canvas.Calls, Is.EqualTo(new[] { "clear 1" }));
    }
}
