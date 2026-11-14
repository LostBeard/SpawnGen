using SpawnGen.Editor;
using SpawnGen.Engine.Types;

namespace SpawnGen.Editor.Tests;

/// <summary>
/// The bounding-box canvas's rounding and editing against what ComfyUI's own editor stored for the same pointer actions (measured black-box,
/// tools/studio-browser-test/probe-comfy-bboxes-steps*.js, 2026-11-14; canvas positions are the pointer's fraction of the canvas times its size).
/// </summary>
public class BoxesWidgetTests
{
    static (double, double) At(double fx, double fy, double w = 1024, double h = 1024) => (fx * w, fy * h);

    [TestCase(0.2, 0.2, 0.7, 0.6, 192, 192, 512, 416)]   // draw A
    [TestCase(0.7, 0.6, 0.2, 0.2, 192, 192, 512, 416)]   // the same box drawn backwards
    [TestCase(0.9, 0.1, 0.6, 0.3, 608, 96, 320, 224)]    // drawn from the top right to the bottom left
    [TestCase(0.13, 0.71, 0.37, 0.93, 128, 736, 256, 224)] // draw B
    public void With_the_grid_both_corners_snap_to_32(double fx0, double fy0, double fx1, double fy1, double x, double y, double w, double h)
    {
        Assert.That(BoxesWidget.Draw(At(fx0, fy0), At(fx1, fy1), grid: true), Is.EqualTo((x, y, w, h)));
    }

    [Test]
    public void The_grid_is_32_canvas_pixels_at_any_canvas_size()
    {
        // 2048 x 1024: ComfyUI stored x 416 (409.6 to the nearest 32) and width 1024.
        Assert.That(BoxesWidget.Draw(At(0.2, 0.2, 2048), At(0.7, 0.6, 2048), grid: true), Is.EqualTo((416.0, 192.0, 1024.0, 416.0)));
    }

    [Test]
    public void Without_the_grid_the_corner_and_the_size_round_to_whole_pixels()
    {
        // ComfyUI: {"x": 829, "y": 113, "width": 123, "height": 164} (the corners rounded alone would give a height of 163).
        Assert.That(BoxesWidget.Draw(At(0.81, 0.11), At(0.93, 0.27), grid: false), Is.EqualTo((829.0, 113.0, 123.0, 164.0)));
    }

    [Test]
    public void A_drag_that_draws_nothing_adds_no_box()
    {
        Assert.That(BoxesWidget.Draw((100, 100), (110, 300), grid: true), Is.Null, "under half a grid step wide");
        Assert.That(BoxesWidget.Draw((100, 100), (100, 300), grid: false), Is.Null);
    }

    static BoxesWidget.Box Box(double x, double y, double w, double h) => new(x, y, w, h, BoxesWidget.NewMetadata());

    [Test]
    public void Moving_rounds_the_drag()
    {
        // B (128, 736) dragged by 0.05 of the canvas: ComfyUI (192, 800). A dragged by (0.07, 0.06): +64, +64. C without the grid by (0.03, 0.02): +31, +20.
        var b = BoxesWidget.Move(Box(128, 736, 256, 224), 0.05 * 1024, 0.05 * 1024, grid: true);
        Assert.That((b.X, b.Y, b.W, b.H), Is.EqualTo((192.0, 800.0, 256.0, 224.0)));
        var a = BoxesWidget.Move(Box(192, 192, 512, 416), 0.07 * 1024, 0.06 * 1024, grid: true);
        Assert.That((a.X, a.Y), Is.EqualTo((256.0, 256.0)));
        var c = BoxesWidget.Move(Box(829, 113, 123, 164), 0.03 * 1024, 0.02 * 1024, grid: false);
        Assert.That((c.X, c.Y), Is.EqualTo((860.0, 133.0)));
        var wide = BoxesWidget.Move(Box(352, 800, 512, 224), 0.05 * 2048, 0, grid: true);
        Assert.That(wide.X, Is.EqualTo(448.0), "ComfyUI at 2048 wide: +96");
    }

    [Test]
    public void Resizing_a_corner_rounds_the_drag_and_stays_normalised()
    {
        // A's bottom-right corner dragged by 0.1 of the canvas: ComfyUI 608 x 512.
        var a = BoxesWidget.Resize(Box(192, 192, 512, 416), BoxesWidget.Corner.Right | BoxesWidget.Corner.Bottom, 102.4, 102.4, grid: true);
        Assert.That((a.X, a.Y, a.W, a.H), Is.EqualTo((192.0, 192.0, 608.0, 512.0)));
        var flipped = BoxesWidget.Resize(Box(100, 100, 64, 64), BoxesWidget.Corner.Left | BoxesWidget.Corner.Top, 128, 0, grid: true);
        Assert.That((flipped.X, flipped.W), Is.EqualTo((164.0, 64.0)), "the left edge dragged past the right one");
    }

    [Test]
    public void The_value_keeps_ComfyUIs_layout_whole_numbers_and_unknown_metadata()
    {
        var meta = BoxesWidget.NewMetadata();
        meta["future"] = 7L;
        var value = BoxesWidget.Make([new BoxesWidget.Box(192, 192, 512, 416, meta)]);
        Assert.That(JsonLiteral.ToJson(value), Is.EqualTo("""[{"x":192,"y":192,"width":512,"height":416,"metadata":{"type":"obj","text":"","desc":"","palette":[],"future":7}}]"""));
        var read = BoxesWidget.Boxes(value).Single();
        Assert.That((read.X, read.W, read.Type, read.Metadata["future"]), Is.EqualTo((192.0, 512.0, "obj", (object)7L)));
        Assert.That(BoxesWidget.Boxes(JsonLiteral.Parse("""[{"x":1.5,"y":2,"width":3,"height":4}]""")).Single().Type, Is.EqualTo("obj"), "no metadata: a new box's");
    }

    [Test]
    public void Metadata_edits_keep_the_rest()
    {
        var b = Box(0, 0, 64, 64).With("type", "text").With("desc", "a red apple").With("palette", new ArrayValue { "#ffffff" });
        Assert.That((b.Type, b.Text, b.Desc, b.Palette.Single()), Is.EqualTo(("text", "", "a red apple", "#ffffff")));
    }

    [Test]
    public void Layout_keeps_the_boxes_aspect_and_hits_the_topmost_box_and_corners()
    {
        var lay = BoxesWidget.Lay(new Rect(0, 0, 300, BoxesWidget.RowHeight(300, 2048, 1024)), 2048, 1024);
        Assert.That((lay.Canvas.Width, lay.Canvas.Height), Is.EqualTo((300.0, 150.0)));
        var tall = BoxesWidget.Lay(new Rect(0, 0, 300, BoxesWidget.RowHeight(300, 512, 4096)), 512, 4096);
        Assert.That((tall.Canvas.Height, tall.Canvas.Width), Is.EqualTo((600.0, 75.0)), "at most twice the row's width tall, narrowed to keep the aspect");
        var boxes = new List<BoxesWidget.Box> { Box(0, 0, 512, 512), Box(256, 256, 512, 512) };
        Assert.That(BoxesWidget.BoxAt(boxes, (300, 300)), Is.EqualTo(1), "the later box is on top");
        Assert.That(BoxesWidget.BoxAt(boxes, (900, 900)), Is.EqualTo(-1));
        var r = BoxesWidget.BoxRect(lay.Canvas, 2048, 1024, boxes[1]);
        Assert.That(BoxesWidget.CornerAt(r, new Point(r.Right + 3, r.Bottom - 3)), Is.EqualTo(BoxesWidget.Corner.Right | BoxesWidget.Corner.Bottom));
        Assert.That(BoxesWidget.CornerAt(r, r.Center), Is.EqualTo(BoxesWidget.Corner.None));
        Assert.That(BoxesWidget.Tag(0), Is.EqualTo("01"));
    }
}
