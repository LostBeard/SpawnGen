using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// The compositor's composition as layers for a layered file (Export PSD): the records ComfyUI's export was measured to write (a Background
/// record, then the layers in drawing order, each already rasterised as drawn) and the flattened picture of a run.
/// </summary>
public class CompositorExportTests
{
    static (byte[] Rgba, int Width, int Height) Solid(int w, int h, byte r, byte g, byte b, byte a)
    {
        var px = new byte[w * h * 4];
        for (int i = 0; i < w * h; i++) { px[i * 4] = r; px[i * 4 + 1] = g; px[i * 4 + 2] = b; px[i * 4 + 3] = a; }
        return (px, w, h);
    }

    static DictValue Layer(string name, double x, double y, double w, double h, double rotation = 0, double opacity = 1, string blend = "normal",
        bool visible = true, bool flipH = false, bool flipV = false) => new()
    {
        ["name"] = name, ["visible"] = visible, ["opacity"] = opacity, ["blend"] = blend,
        ["transform"] = new DictValue { ["x"] = x, ["y"] = y, ["w"] = w, ["h"] = h, ["rotation"] = rotation },
        ["flipH"] = flipH, ["flipV"] = flipV,
    };

    static DictValue Composition(int w, int h, bool background, string colour, params DictValue[] layers) => new()
    {
        ["version"] = 1L, ["canvas"] = new DictValue { ["w"] = (long)w, ["h"] = (long)h }, ["inputs"] = new ArrayValue(),
        ["background"] = new DictValue { ["color"] = colour, ["opacity"] = 1L, ["visible"] = background },
        ["layers"] = new ArrayValue(layers.Cast<object?>()),
    };

    static byte[] Pixel(byte[] rgba, int width, int x, int y) => rgba[((y * width + x) * 4)..((y * width + x) * 4 + 4)];

    [Test]
    public void A_Background_record_comes_first_then_the_layers_as_they_are_drawn()
    {
        var red = Solid(4, 3, 255, 0, 0, 255);
        var green = Solid(2, 2, 0, 255, 0, 128);
        var doc = ImageCompositor.Export([red, green], Composition(8, 6, background: false, "#336699",
            Layer("red", 0, 0, 4, 3), Layer("grün ✓", 5, 2, 2, 2, opacity: 0.5, blend: "multiply", visible: false)));
        Assert.Multiple(() =>
        {
            Assert.That((doc.Width, doc.Height), Is.EqualTo((8, 6)));
            Assert.That(doc.Layers.Select(l => l.Name), Is.EqualTo(new[] { "Background", "red", "grün ✓" }));
            var bg = doc.Layers[0];
            Assert.That((bg.Left, bg.Top, bg.Width, bg.Height, bg.Visible, bg.Blend, bg.Opacity), Is.EqualTo((0, 0, 8, 6, false, "normal", 1d)));
            Assert.That(Pixel(bg.Rgba, 8, 7, 5), Is.EqualTo(new byte[] { 0x33, 0x66, 0x99, 255 }));
            var r = doc.Layers[1];
            Assert.That((r.Left, r.Top, r.Width, r.Height, r.Visible, r.Blend, r.Opacity), Is.EqualTo((0, 0, 4, 3, true, "normal", 1d)));
            Assert.That(r.Rgba, Is.EqualTo(red.Rgba));
            var g = doc.Layers[2];
            Assert.That((g.Left, g.Top, g.Width, g.Height, g.Visible, g.Blend, g.Opacity), Is.EqualTo((5, 2, 2, 2, false, "multiply", 0.5)));
            Assert.That(g.Rgba, Is.EqualTo(green.Rgba));
            // the flattened picture is the run's: the red layer on a transparent canvas (the green one is hidden)
            Assert.That(doc.HasAlpha, Is.True);
            Assert.That(Pixel(doc.Composite, 8, 1, 1), Is.EqualTo(new byte[] { 255, 0, 0, 255 }));
            Assert.That(Pixel(doc.Composite, 8, 6, 3)[3], Is.EqualTo(0));
        });
    }

    [Test]
    public void A_shown_background_makes_the_flattened_picture_opaque()
    {
        var doc = ImageCompositor.Export([Solid(4, 3, 255, 0, 0, 255)], Composition(8, 6, background: true, "#336699", Layer("red", 2, 1, 4, 3)));
        Assert.Multiple(() =>
        {
            Assert.That(doc.HasAlpha, Is.False);
            Assert.That(doc.Layers[0].Visible, Is.True);
            Assert.That(Pixel(doc.Composite, 8, 0, 0), Is.EqualTo(new byte[] { 0x33, 0x66, 0x99, 255 }));
            Assert.That(Pixel(doc.Composite, 8, 3, 2), Is.EqualTo(new byte[] { 255, 0, 0, 255 }));
        });
    }

    [Test]
    public void A_turned_layer_is_stored_in_its_bounding_box_and_may_run_past_the_canvas()
    {
        // 6 x 2 turned a quarter about its centre (4, 2): a 2 x 6 box from (3, -1), one row above the canvas
        var doc = ImageCompositor.Export([Solid(6, 2, 10, 20, 30, 255)], Composition(8, 6, background: false, "#ffffff", Layer("bar", 1, 1, 6, 2, rotation: Math.PI / 2)));
        var l = doc.Layers[1];
        Assert.Multiple(() =>
        {
            Assert.That((l.Left, l.Top, l.Width, l.Height), Is.EqualTo((3, -1, 2, 6)));
            Assert.That(l.Rgba.Length, Is.EqualTo(2 * 6 * 4));
            Assert.That(Pixel(l.Rgba, 2, 1, 3), Is.EqualTo(new byte[] { 10, 20, 30, 255 }));
            // the flattened picture shows the part inside the canvas
            Assert.That(Pixel(doc.Composite, 8, 3, 0), Is.EqualTo(new byte[] { 10, 20, 30, 255 }));
            Assert.That(Pixel(doc.Composite, 8, 1, 1)[3], Is.EqualTo(0));
        });
    }

    [Test]
    public void Resizes_and_flips_are_applied_to_the_stored_picture()
    {
        // left half red, right half blue, 16 x 2; flipped horizontally the red is on the right; shown at 32 x 4 (Lanczos: the far ends are
        // clear of the edge between the colours)
        var px = new byte[16 * 2 * 4];
        for (int y = 0; y < 2; y++) for (int x = 0; x < 16; x++) { int i = (y * 16 + x) * 4; px[i] = (byte)(x < 8 ? 255 : 0); px[i + 2] = (byte)(x < 8 ? 0 : 255); px[i + 3] = 255; }
        var doc = ImageCompositor.Export([(px, 16, 2)], Composition(32, 4, background: false, "#ffffff", Layer("halves", 0, 0, 32, 4, flipH: true)));
        var l = doc.Layers[1];
        Assert.Multiple(() =>
        {
            Assert.That((l.Left, l.Top, l.Width, l.Height), Is.EqualTo((0, 0, 32, 4)));
            Assert.That(Pixel(l.Rgba, 32, 0, 1), Is.EqualTo(new byte[] { 0, 0, 255, 255 }));
            Assert.That(Pixel(l.Rgba, 32, 31, 2), Is.EqualTo(new byte[] { 255, 0, 0, 255 }));
        });
    }

    [Test]
    public void The_editors_order_decides_which_layer_is_on_top()
    {
        var red = Solid(4, 4, 255, 0, 0, 255);
        var blue = Solid(4, 4, 0, 0, 255, 255);
        var composition = Composition(4, 4, background: false, "#ffffff", Layer("red", 0, 0, 4, 4), Layer("blue", 0, 0, 4, 4));
        var plain = ImageCompositor.Export([red, blue], composition);
        composition["order"] = new ArrayValue { 1L, 0L };
        var swapped = ImageCompositor.Export([red, blue], composition);
        Assert.Multiple(() =>
        {
            Assert.That(plain.Layers.Select(l => l.Name), Is.EqualTo(new[] { "Background", "red", "blue" }));
            Assert.That(Pixel(plain.Composite, 4, 2, 2), Is.EqualTo(new byte[] { 0, 0, 255, 255 }));
            Assert.That(swapped.Layers.Select(l => l.Name), Is.EqualTo(new[] { "Background", "blue", "red" }));
            Assert.That(Pixel(swapped.Composite, 4, 2, 2), Is.EqualTo(new byte[] { 255, 0, 0, 255 }));
        });
    }

    [Test]
    public void A_composition_that_does_not_fit_the_pictures_is_refused()
    {
        var red = Solid(4, 4, 255, 0, 0, 255);
        Assert.Multiple(() =>
        {
            Assert.That(() => ImageCompositor.Export([red, red], Composition(4, 4, false, "#ffffff", Layer("one", 0, 0, 4, 4))), Throws.Exception);
            Assert.That(() => ImageCompositor.Export([red], new DictValue()), Throws.Exception);
            Assert.That(() => ImageCompositor.Export([(new byte[5], 4, 4)], Composition(4, 4, false, "#ffffff", Layer("one", 0, 0, 4, 4))), Throws.Exception);
        });
    }
}
