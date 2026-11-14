using SpawnGen.Editor;
using SpawnGen.Engine.Types;

namespace SpawnGen.Editor.Tests;

/// <summary>
/// The crop region's editing against what ComfyUI's own editor stored for the same pointer actions on a 768 x 768 picture (measured black-box,
/// tools/studio-browser-test/probe-comfy-crop.js, 2026-11-14).
/// </summary>
public class CropWidgetTests
{
    static CropWidget.Region R(double x, double y, double w, double h) => new(x, y, w, h);
    const double P = 768;

    [Test]
    public void A_drag_inside_moves_by_the_rounded_drag_and_stays_in_the_picture()
    {
        // 0.15 / 0.10 of the picture: ComfyUI +115, +77.
        Assert.That(CropWidget.Move(R(0, 0, 512, 512), 0.15 * P, 0.10 * P, P, P), Is.EqualTo(R(115, 77, 512, 512)));
        // Past the edge: ComfyUI stops at 768 - size.
        Assert.That(CropWidget.Move(R(100, 100, 400, 200), 600, 600, P, P), Is.EqualTo(R(368, 568, 400, 200)));
    }

    [Test]
    public void Choosing_a_ratio_keeps_the_width()
    {
        Assert.That(CropWidget.ApplyRatio(R(115, 77, 512, 512), "16:9"), Is.EqualTo(R(115, 77, 512, 288)));
        Assert.That(CropWidget.ApplyRatio(R(115, 77, 512, 288), "Custom"), Is.EqualTo(R(115, 77, 512, 288)));
        Assert.That(CropWidget.Ratios, Is.EqualTo(new[] { "1:1", "3:4", "4:3", "16:9", "9:16", "Custom" }));
    }

    [Test]
    public void A_corner_with_a_ratio_follows_the_drag_projected_on_the_aspect()
    {
        // ComfyUI: the bottom-right corner of 512 x 288 at 16:9 dragged (+50, +100): 593 x 334.
        var bottomRight = CropWidget.Corner.Right | CropWidget.Corner.Bottom;
        Assert.That(CropWidget.Resize(R(115, 77, 512, 288), bottomRight, 50, 100, (16, 9), P, P), Is.EqualTo(R(115, 77, 593, 334)));
        // The top-left corner dragged past the other: the bottom-right stays at (708, 411), 16 on the shorter side: 28 x 16.
        Assert.That(CropWidget.Resize(R(115, 77, 593, 334), CropWidget.Corner.Left | CropWidget.Corner.Top, 2000, 2000, (16, 9), P, P),
            Is.EqualTo(R(680, 395, 28, 16)));
        // Then the bottom-left corner dragged (-60, -30): ComfyUI 61 x 34 with the top-right at (708, 395).
        Assert.That(CropWidget.Resize(R(680, 395, 28, 16), CropWidget.Corner.Left | CropWidget.Corner.Bottom, -60, -30, (16, 9), P, P),
            Is.EqualTo(R(647, 395, 61, 34)));
    }

    [Test]
    public void The_lock_keeps_the_current_aspect()
    {
        // ComfyUI, locked on 400 x 200: the bottom-right corner dragged (+100, +10) gives 484 x 242.
        var r = R(100, 100, 400, 200);
        Assert.That(CropWidget.Resize(r, CropWidget.Corner.Right | CropWidget.Corner.Bottom, 100, 10, (r.W, r.H), P, P), Is.EqualTo(R(100, 100, 484, 242)));
    }

    [Test]
    public void Without_a_ratio_a_corner_moves_alone_with_16_as_the_least_side_and_the_picture_as_the_most()
    {
        var bottomRight = CropWidget.Corner.Right | CropWidget.Corner.Bottom;
        // ComfyUI (Custom): 61 x 34 dragged (-100, +20) -> 16 x 54; 400 x 200 at (100, 100) dragged (+600, +50) -> 668 x 250.
        Assert.That(CropWidget.Resize(R(647, 395, 61, 34), bottomRight, -100, 20, null, P, P), Is.EqualTo(R(647, 395, 16, 54)));
        Assert.That(CropWidget.Resize(R(100, 100, 400, 200), bottomRight, 600, 50, null, P, P), Is.EqualTo(R(100, 100, 668, 250)));
    }

    [Test]
    public void The_value_is_ComfyUIs_and_the_layout_fits_the_picture()
    {
        Assert.That(JsonLiteral.ToJson(CropWidget.Make(R(115, 77, 512, 288))), Is.EqualTo("""{"x":115,"y":77,"width":512,"height":288}"""));
        Assert.That(CropWidget.Read(JsonLiteral.Parse("""{"x":0,"y":0,"width":512,"height":512}""")), Is.EqualTo(R(0, 0, 512, 512)));
        var lay = CropWidget.Lay(new Rect(0, 0, 300, CropWidget.Height(300)));
        Assert.That(lay.PictureBox, Is.EqualTo(new Rect(0, 0, 300, 300)));
        Assert.That(lay.FieldRows.Select(f => f.Field), Is.EqualTo(new[] { "x", "y", "width", "height" }));
        var wide = CropWidget.Fit(lay.PictureBox, 1024, 512);
        Assert.That((wide.Width, wide.Height, wide.Y), Is.EqualTo((300.0, 150.0, 75.0)));
        var region = CropWidget.RegionRect(wide, 1024, 512, R(512, 0, 512, 512));
        Assert.That(CropWidget.CornerAt(region, new Point(region.Right - 2, region.Bottom + 3)), Is.EqualTo(CropWidget.Corner.Right | CropWidget.Corner.Bottom));
    }
}
