using SpawnGen.Editor;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Editor.Tests;

/// <summary>
/// Crop Image's crop region edited with the pointer: the actions the probe made on ComfyUI's own editor (probe-comfy-crop.js, 2026-11-14, a 768 x 768
/// picture), expecting the crop_region ComfyUI stored after each.
/// </summary>
public class CropEditingTests
{
    static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(SpawnGen.Nodes.Core.SaveImage).Assembly);

    sealed class Rig
    {
        public required EditorView View { get; init; }
        public required EditorNode Node { get; init; }
        public WidgetRow Row => View.Hits.Geometry(Node).Widgets.Single(w => w.Name == "crop_region");
        public CropWidget.Layout Lay => CropWidget.Lay(Row.Bounds);
        public Rect Picture => CropWidget.Fit(Lay.PictureBox, 768, 768);
        public string State => JsonLiteral.ToJson(Node.Widgets["crop_region"]);
        Point S(Point world) => View.Viewport.WorldToScreen(world);
        /// <summary>A point of the picture, in picture pixels.</summary>
        public Point At(double x, double y) => new(Picture.X + x / 768 * Picture.Width, Picture.Y + y / 768 * Picture.Height);

        public void Drag(Point a, Point b)
        {
            View.PointerDown(S(a), PointerButton.Left);
            View.PointerMove(S(new Point((a.X + b.X) / 2, (a.Y + b.Y) / 2)));
            View.PointerMove(S(b));
            View.PointerUp(S(b), PointerButton.Left);
        }

        public void Click(Point world) { View.PointerDown(S(world), PointerButton.Left); View.PointerUp(S(world), PointerButton.Left); }
    }

    static Rig Setup()
    {
        var doc = new EditorDocument(Registry());
        var node = doc.AddNode("ImageCropV2", 0, 0);
        var view = new EditorView(doc) { CropPictureSize = _ => (768, 768) };
        return new Rig { View = view, Node = node };
    }

    [Test]
    public void The_crop_region_is_a_picture_with_a_ratio_and_fields_not_a_JSON_row()
    {
        var r = Setup();
        Assert.That(r.Row.IsCrop, Is.True);
        Assert.That(r.Row.Bounds.Height, Is.GreaterThan(r.Row.Bounds.Width), "the picture box, the ratio row and four fields");
        Assert.That(r.State, Is.EqualTo("""{"x":0,"y":0,"width":512,"height":512}"""));
        Assert.That(r.View.CropRatio(r.Node.Id, "crop_region"), Is.EqualTo("Custom"));
    }

    [Test]
    public void The_probes_actions_give_ComfyUIs_regions()
    {
        var r = Setup();
        // Inside the rectangle, 0.15 / 0.10 of the picture: (115, 77).
        r.Drag(r.At(0.2 * 768, 0.2 * 768), r.At(0.35 * 768, 0.3 * 768));
        Assert.That(r.State, Is.EqualTo("""{"x":115,"y":77,"width":512,"height":512}"""));
        // Outside it: nothing.
        r.Drag(r.At(0.05 * 768, 0.05 * 768), r.At(0.6 * 768, 0.7 * 768));
        Assert.That(r.State, Is.EqualTo("""{"x":115,"y":77,"width":512,"height":512}"""));
        // 16:9 keeps the width.
        r.View.SetCropField(r.Node.Id, "crop_region", "ratio", "16:9");
        Assert.That(r.State, Is.EqualTo("""{"x":115,"y":77,"width":512,"height":288}"""));
        // The bottom-right corner (+50, +100) keeps 16:9: 593 x 334.
        r.Drag(r.At(115 + 512, 77 + 288), r.At(115 + 512 + 50, 77 + 288 + 100));
        Assert.That(r.State, Is.EqualTo("""{"x":115,"y":77,"width":593,"height":334}"""));
    }

    [Test]
    public void The_lock_keeps_the_current_aspect_and_fields_and_the_ratio_open_in_place()
    {
        var r = Setup();
        foreach (var (f, v) in new[] { ("x", "100"), ("y", "100"), ("width", "400"), ("height", "200") }) r.View.SetCropField(r.Node.Id, "crop_region", f, v);
        Assert.That(r.State, Is.EqualTo("""{"x":100,"y":100,"width":400,"height":200}"""));
        r.Click(r.Lay.LockButton.Center);
        Assert.That(r.View.CropLocked(r.Node.Id, "crop_region"), Is.True);
        r.Drag(r.At(500, 300), r.At(600, 310));
        Assert.That(r.State, Is.EqualTo("""{"x":100,"y":100,"width":484,"height":242}"""), "ComfyUI, locked: 484 x 242");
        r.View.SetCropField(r.Node.Id, "crop_region", "width", "900");
        Assert.That(r.State, Is.EqualTo("""{"x":100,"y":100,"width":900,"height":242}"""), "typed values are stored as typed");

        var asked = new List<string>();
        r.View.CropFieldActivated += (_, _, field) => asked.Add(field);
        r.Click(r.Lay.RatioButton.Center);
        r.Click(r.Lay.FieldRows[2].Value.Center);
        Assert.That(asked, Is.EqualTo(new[] { "ratio", "width" }));
        r.View.Document.Undo();
        Assert.That(r.State, Is.EqualTo("""{"x":100,"y":100,"width":484,"height":242}"""), "undo");
    }
}
