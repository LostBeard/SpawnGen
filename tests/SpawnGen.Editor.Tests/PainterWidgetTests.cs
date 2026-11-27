using SpawnGen.Editor;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Editor.Tests;

/// <summary>
/// Painter's widget logic against what ComfyUI's own widget painted (black box, tools/studio-browser-test/probe-comfy-painter-brush.js,
/// 2026-11-26 / 27; Fixtures/painter-brush are the paintings it uploaded, 512 x 512, the stamp clicked at the pixel corner (256, 256)).
/// </summary>
public class PainterWidgetTests
{
    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "painter-brush");

    /// <summary>The painting's alpha as stored (0..255).</summary>
    static int[,] Alpha(string name)
    {
        var image = ImageDecoder.Decode(File.ReadAllBytes(Path.Combine(Dir, name + ".png")), out var alpha);
        Assert.That(alpha, Is.Not.Null, name + " has no alpha");
        var a = new int[image.Height, image.Width];
        for (int y = 0; y < image.Height; y++)
            for (int x = 0; x < image.Width; x++) a[y, x] = (int)MathF.Round(alpha![y * image.Width + x] * 255f);
        return a;
    }

    [TestCase(50, 0), TestCase(50, 10), TestCase(50, 25), TestCase(50, 50), TestCase(50, 75), TestCase(50, 90), TestCase(100, 0), TestCase(100, 50)]
    public void A_soft_stamp_matches_ComfyUI_on_every_pixel(int size, int hardness)
    {
        // ComfyUI's stamp is the browser's radial gradient, which Chrome dithers: 2 to 7 % of the painted pixels come out one step under the
        // exact value, in a symmetric pattern (counts in multiples of 4), never over it. The last pixel before the outer radius is also cut
        // by the circle's antialiased edge (up to 4 under at hardness 90). Everything else is the formula, exactly.
        var want = Alpha($"dot-size{size}-hard{hardness:000}-b");
        double rout = PainterWidget.OuterRadius(size, hardness);
        int painted = 0, off = 0, worstRamp = 0, worstRim = 0;
        for (int y = 0; y < 512; y++)
            for (int x = 0; x < 512; x++)
            {
                int ours = PainterWidget.StampAlpha(x, y, 256, 256, size, hardness);
                if (ours == 0 && want[y, x] == 0) continue;
                painted++;
                int diff = ours - want[y, x];
                Assert.That(diff, Is.GreaterThanOrEqualTo(0), $"ComfyUI over the formula at ({x}, {y})");
                if (diff > 0) off++;
                double d = Math.Sqrt((x + 0.5 - 256) * (x + 0.5 - 256) + (y + 0.5 - 256) * (y + 0.5 - 256));
                if (d < rout - 1) worstRamp = Math.Max(worstRamp, diff); else worstRim = Math.Max(worstRim, diff);
            }
        TestContext.Out.WriteLine($"size {size} hardness {hardness}: {painted} painted pixels, {off} one or more under, ramp worst {worstRamp}, rim worst {worstRim}");
        Assert.That(worstRamp, Is.LessThanOrEqualTo(1), "ramp (dither)");
        Assert.That(worstRim, Is.LessThanOrEqualTo(4), "the antialiased rim");
        Assert.That(off, Is.LessThanOrEqualTo(painted / 12), "share of dithered pixels (measured: at most 6.8 %)");
    }

    [Test]
    public void The_measured_radii_hold()
    {
        // r 25: hardness 0 fades out at 37.5, hardness 50 from 12.5 to 31.25, hardness 90 from 22.5 to 26.25 (ComfyUI: 51 119 187 at its rim)
        Assert.That((PainterWidget.InnerRadius(50, 0), PainterWidget.OuterRadius(50, 0)), Is.EqualTo((0.0, 37.5)));
        Assert.That((PainterWidget.InnerRadius(50, 50), PainterWidget.OuterRadius(50, 50)), Is.EqualTo((12.5, 31.25)));
        Assert.That(PainterWidget.OuterRadius(50, 100), Is.EqualTo(25));
        Assert.That(Enumerable.Range(230, 3).Select(x => PainterWidget.StampAlpha(x, 255, 256, 256, 50, 90)), Is.EqualTo(new[] { 51, 119, 187 }));
        Assert.That(PainterWidget.GradientStops(50, 50), Is.EqualTo(new[] { (0.4, 1.0), (1.0, 0.0) }));
        Assert.That(PainterWidget.GradientStops(50, 100), Is.Null);
    }

    [Test]
    public void A_hard_stamp_covers_the_measured_box()
    {
        // ComfyUI: size 50 at (256, 256) paints x and y 231..280 with an antialiased rim; the coverage inside is 1, outside 0.
        var want = Alpha("dot-size50-hard100-b");
        for (int y = 0; y < 512; y++)
            for (int x = 0; x < 512; x++)
            {
                double d = Math.Sqrt((x + 0.5 - 256) * (x + 0.5 - 256) + (y + 0.5 - 256) * (y + 0.5 - 256));
                if (d < 24) Assert.That(want[y, x], Is.EqualTo(255), $"inside ({x}, {y})");
                if (d > 26) Assert.That(want[y, x], Is.EqualTo(0), $"outside ({x}, {y})");
                if (d < 24.5 || d > 25.5) Assert.That(PainterWidget.StampAlpha(x, y, 256, 256, 50, 100), Is.EqualTo(want[y, x]), $"ours at ({x}, {y})");
            }
    }

    static List<(double X, double Y)> Stamps(int events, int size)
    {
        var stroke = new PainterWidget.Stroke(100, 256, size);
        var all = new List<(double, double)> { stroke.First };
        for (int i = 1; i <= events; i++) all.AddRange(stroke.MoveTo(100 + 300.0 * i / events, 256));
        return all;
    }

    [Test]
    public void A_stroke_does_not_depend_on_how_the_path_is_cut_into_events()
    {
        // ComfyUI: the same path in 1, 2, 30 or 300 pointer events paints byte-identical strokes.
        var one = Stamps(1, 50);
        foreach (int n in new[] { 2, 30, 300, 7 })
        {
            var other = Stamps(n, 50);
            Assert.That(other, Has.Count.EqualTo(one.Count), $"{n} events");
            for (int i = 0; i < one.Count; i++)
                Assert.That(Math.Abs(other[i].X - one[i].X) + Math.Abs(other[i].Y - one[i].Y), Is.LessThan(1e-6), $"{n} events, stamp {i}");
        }
        Assert.That(one, Has.Count.EqualTo(61)); // every 5 (r / 5) from 100 to 400
        // and ComfyUI's own: the path in 1 event and in 300 painted the same bytes
        var (a, b) = (Alpha("synth-soft-seg001"), Alpha("synth-soft-seg300"));
        Assert.That(a.Cast<int>().SequenceEqual(b.Cast<int>()), Is.True, "ComfyUI's 1-event and 300-event strokes");
    }

    /// <summary>A stroke painted with the model: the stamps composited source-over at full alpha, then the stroke at its opacity.</summary>
    static double[,] Paint(IEnumerable<(double X, double Y)> stamps, int size, int hardness, double opacity)
    {
        var a = new double[512, 512];
        double reach = PainterWidget.OuterRadius(size, hardness) + 1;
        foreach (var (cx, cy) in stamps)
            for (int y = (int)Math.Max(0, cy - reach); y < Math.Min(512, cy + reach); y++)
                for (int x = (int)Math.Max(0, cx - reach); x < Math.Min(512, cx + reach); x++)
                {
                    double c = PainterWidget.Coverage(Math.Sqrt((x + 0.5 - cx) * (x + 0.5 - cx) + (y + 0.5 - cy) * (y + 0.5 - cy)), size, hardness);
                    a[y, x] += c * (1 - a[y, x]);
                }
        for (int y = 0; y < 512; y++) for (int x = 0; x < 512; x++) a[y, x] *= opacity;
        return a;
    }

    [TestCase("synth-soft-seg001", 100)]
    [TestCase("line-soft-hard000-opacity50", 50)]
    public void A_soft_line_is_within_the_measured_tolerance(string fixture, int opacity)
    {
        // The rule inside a stroke is not settled exactly (PLANS): the model must stay within what was measured for it on the synthetic line
        // (over this window: mean 1.96 / 255, max 29). The opacity 50 line came through Playwright's real pointer, whose build-up depends on
        // the event timing (up to 43 off the model at its rim), so it checks the opacity rule only: the interior at half strength.
        var want = Alpha(fixture);
        var ours = Paint(Stamps(1, 50), 50, 0, opacity / 100.0);
        double sum = 0; int worst = 0, n = 0;
        for (int y = 200; y < 312; y++)
            for (int x = 60; x < 452; x++)
            {
                int d = Math.Abs((int)Math.Floor(ours[y, x] * 255 + 0.5) - want[y, x]);
                sum += d; n++; worst = Math.Max(worst, d);
            }
        TestContext.Out.WriteLine($"{fixture}: mean {sum / n:F3}, worst {worst}");
        if (opacity == 100) { Assert.That(sum / n, Is.LessThan(2.0)); Assert.That(worst, Is.LessThanOrEqualTo(29)); }
        else
        {
            // interior at half strength in ComfyUI (127..128) and in ours
            for (int x = 150; x < 350; x++)
            {
                Assert.That(want[256, x], Is.InRange(127, 128), $"ComfyUI at x {x}");
                Assert.That((int)Math.Floor(ours[256, x] * 255 + 0.5), Is.InRange(127, 128), $"ours at x {x}");
            }
        }
    }

    [Test]
    public void The_eraser_is_hard_and_full_whatever_the_brush()
    {
        var b = new PainterWidget.Brush(PainterWidget.Tool.Eraser, 50, "#ff0000", 50, 0).Effective();
        Assert.That((b.Opacity, b.Hardness, b.Size), Is.EqualTo((100, 100, 50)));
        var brush = new PainterWidget.Brush(PainterWidget.Tool.Brush, 50, "#ff0000", 50, 0).Effective();
        Assert.That((brush.Opacity, brush.Hardness), Is.EqualTo((50, 0)));
        Assert.That(new PainterWidget.Brush(Size: 500, Opacity: -3, Hardness: 140, Color: "#ABC").Clamped(),
            Is.EqualTo(new PainterWidget.Brush(PainterWidget.Tool.Brush, 200, "#aabbcc", 0, 100)));
    }

    [Test]
    public void The_rows_follow_the_tool_and_a_linked_picture()
    {
        var R = PainterWidget.Row.Tool;
        Assert.That(PainterWidget.Rows(PainterWidget.Tool.Brush, false), Is.EqualTo(new[] { R, PainterWidget.Row.CursorSize, PainterWidget.Row.Color,
            PainterWidget.Row.Hardness, PainterWidget.Row.Width, PainterWidget.Row.Height, PainterWidget.Row.Background, PainterWidget.Row.Clear }));
        // ComfyUI's Eraser: Cursor Size, Width, Height (no colour, no Hardness)
        Assert.That(PainterWidget.Rows(PainterWidget.Tool.Eraser, false), Is.EqualTo(new[] { R, PainterWidget.Row.CursorSize,
            PainterWidget.Row.Width, PainterWidget.Row.Height, PainterWidget.Row.Background, PainterWidget.Row.Clear }));
        // a picture linked: no Width, Height, Background; the size label under the canvas
        Assert.That(PainterWidget.Rows(PainterWidget.Tool.Brush, true), Is.EqualTo(new[] { R, PainterWidget.Row.CursorSize, PainterWidget.Row.Color,
            PainterWidget.Row.Hardness, PainterWidget.Row.Clear }));
        Assert.That(PainterWidget.SizeLabel(512, 512), Is.EqualTo("512 × 512"));
        var lay = PainterWidget.Lay(new Rect(0, 0, 400, 1000), PainterWidget.Tool.Brush, true);
        Assert.That(lay.CanvasBox, Is.EqualTo(new Rect(0, 0, 400, 400)));
        Assert.That(lay.SizeLabel, Is.Not.Null);
        Assert.That(lay.Rows[0].Bounds.Y, Is.GreaterThan(lay.SizeLabel!.Value.Bottom - 1));
        Assert.That(PainterWidget.Height(400, PainterWidget.Tool.Brush, true), Is.EqualTo(lay.Rows[^1].Bounds.Bottom + 4).Within(1e-9));
    }

    [Test]
    public void The_canvas_size_and_a_resize()
    {
        Assert.That(PainterWidget.CanvasSize(null, 512L, 512L), Is.EqualTo((512, 512)));
        Assert.That(PainterWidget.CanvasSize(null, 256L, 640.0), Is.EqualTo((256, 640)));
        Assert.That(PainterWidget.CanvasSize((64, 512), 512L, 512L), Is.EqualTo((64, 512)));   // the linked picture's size wins
        Assert.That(PainterWidget.CanvasSize(null, 10L, 99999L), Is.EqualTo((64, 4096)));
        // 512 -> 256 x 640 keeps the top-left 256 x 512; growing back keeps only what survived (ComfyUI: the dot at 384 is gone)
        Assert.That(PainterWidget.KeptOnResize((512, 512), (256, 640)), Is.EqualTo((256, 512)));
        // a 256 x 640 canvas in the 426 box: 426 tall, 170.4 wide, centred
        var fit = PainterWidget.Fit(new Rect(0, 0, 426, 426), 256, 640);
        Assert.That((fit.X, fit.Y, fit.Width, fit.Height), Is.EqualTo((127.8, 0.0, 170.4, 426.0)).Using<(double, double, double, double)>((a, b) =>
            Math.Abs(a.Item1 - b.Item1) + Math.Abs(a.Item2 - b.Item2) + Math.Abs(a.Item3 - b.Item3) + Math.Abs(a.Item4 - b.Item4) < 1e-9 ? 0 : 1));
    }

    [Test]
    public void The_upload_name_and_the_mask_value()
    {
        Assert.That(PainterWidget.UploadName(1, 1795812003375), Is.EqualTo("painter-1-1795812003375.png"));
        Assert.That(PainterWidget.MaskValue("painter-1-1795812003375.png"), Is.EqualTo("painter-1-1795812003375.png [input]"));
        Assert.That(PainterWidget.MaskFile("painter-1-1795812003375.png [input]"), Is.EqualTo("painter-1-1795812003375.png"));
        Assert.That(PainterWidget.MaskFile(""), Is.Null);
        Assert.That(PainterWidget.MaskFile(null), Is.Null);
    }
}
