using SpawnGen.Editor;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Editor.Tests;

/// <summary>
/// Create Bounding Boxes' canvas edited with the pointer, the same actions the probe made on ComfyUI's own editor (probe-comfy-bboxes-steps2.js,
/// 2026-11-14), expecting the editor_state ComfyUI stored after each.
/// </summary>
public class BoxesEditingTests
{
    static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(SpawnGen.Nodes.Core.SaveImage).Assembly);

    sealed class Rig
    {
        public required EditorView View { get; init; }
        public required EditorNode Node { get; init; }
        public WidgetRow Row => View.Hits.Geometry(Node).Widgets.Single(w => w.Name == "editor_state");
        public BoxesWidget.Layout Lay => BoxesWidget.Lay(Row.Bounds, 1024, 1024);
        public string State => JsonLiteral.ToJson(Node.Widgets["editor_state"]);
        Point S(Point world) => View.Viewport.WorldToScreen(world);
        public Point At(double fx, double fy) => new(Lay.Canvas.X + fx * Lay.Canvas.Width, Lay.Canvas.Y + fy * Lay.Canvas.Height);

        public void Drag(double fx0, double fy0, double fx1, double fy1)
        {
            View.PointerDown(S(At(fx0, fy0)), PointerButton.Left);
            View.PointerMove(S(At((fx0 + fx1) / 2, (fy0 + fy1) / 2)));
            View.PointerMove(S(At(fx1, fy1)));
            View.PointerUp(S(At(fx1, fy1)), PointerButton.Left);
        }

        public void Click(Point world, PointerButton button = PointerButton.Left)
        {
            View.PointerDown(S(world), button);
            View.PointerUp(S(world), button);
        }
    }

    static Rig Setup()
    {
        var doc = new EditorDocument(Registry());
        var node = doc.AddNode("CreateBoundingBoxes", 0, 0);
        return new Rig { View = new EditorView(doc), Node = node };
    }

    const string Meta = "\"metadata\":{\"type\":\"obj\",\"text\":\"\",\"desc\":\"\",\"palette\":[]}";

    [Test]
    public void The_canvas_replaces_the_JSON_row_and_last_incoming_is_hidden()
    {
        var r = Setup();
        Assert.That(r.Row.IsBoxes, Is.True);
        Assert.That(r.Lay.Canvas.Width, Is.EqualTo(r.Lay.Canvas.Height).Within(1e-9), "1024 x 1024 is square");
        Assert.That(r.View.Hits.Geometry(r.Node).Widgets.Select(w => w.Name), Does.Not.Contain("last_incoming"));
        Assert.That(r.State, Is.EqualTo("[]"));
    }

    [Test]
    public void The_probes_actions_give_ComfyUIs_states()
    {
        var r = Setup();
        r.Drag(0.2, 0.2, 0.7, 0.6);
        Assert.That(r.State, Is.EqualTo("[{\"x\":192,\"y\":192,\"width\":512,\"height\":416," + Meta + "}]"));
        Assert.That(r.View.SelectedBox?.Index, Is.EqualTo(0), "a new box is selected");
        r.Drag(0.13, 0.71, 0.37, 0.93);
        Assert.That(r.State, Does.EndWith("{\"x\":128,\"y\":736,\"width\":256,\"height\":224," + Meta + "}]"));
        r.Drag(0.25, 0.8, 0.3, 0.85);
        Assert.That(r.State, Does.EndWith("{\"x\":192,\"y\":800,\"width\":256,\"height\":224," + Meta + "}]"), "moved by its drag rounded to 32");
        r.Drag(0.45, 0.3, 0.52, 0.36);
        Assert.That(r.State, Does.StartWith("[{\"x\":256,\"y\":256,\"width\":512,\"height\":416,"));
        r.Click(r.Lay.GridButton.Center);
        Assert.That(r.View.BoxGrid(r.Node.Id, "editor_state"), Is.False);
        r.Drag(0.81, 0.11, 0.93, 0.27);
        Assert.That(r.State, Does.EndWith("{\"x\":829,\"y\":113,\"width\":123,\"height\":164," + Meta + "}]"));
        r.Drag(0.85, 0.15, 0.88, 0.17);
        Assert.That(r.State, Does.EndWith("{\"x\":860,\"y\":133,\"width\":123,\"height\":164," + Meta + "}]"));
        r.Click(BoxesWidget.Plus(r.Lay.PaletteRow, 0).Center);
        Assert.That(r.State, Does.EndWith("\"palette\":[\"#ffffff\"]}}]"));
        r.Click(BoxesWidget.Swatch(r.Lay.PaletteRow, 0).Center, PointerButton.Right);
        Assert.That(r.State, Does.EndWith("\"palette\":[]}}]"), "a right-click on the swatch removes it");
        r.Click(r.Lay.ClearButton.Center);
        Assert.That(r.State, Is.EqualTo("[]"));
    }

    [Test]
    public void The_panel_edits_the_selected_box_and_Delete_removes_the_box_not_the_node()
    {
        var r = Setup();
        r.Drag(0.2, 0.2, 0.7, 0.6);
        r.Click(r.Lay.TextButton.Center);
        Assert.That(BoxesWidget.Boxes(r.Node.Widgets["editor_state"])[0].Type, Is.EqualTo("text"));
        (int, string, int, string)? asked = null;
        r.View.BoxFieldActivated += (n, name, i, field) => asked = (n, name, i, field);
        r.Click(r.Lay.TextField.Center);
        Assert.That(asked, Is.EqualTo((r.Node.Id, "editor_state", 0, "text")));
        r.View.SetBoxField(r.Node.Id, "editor_state", 0, "text", "SALE");
        r.Click(r.Lay.DescField.Center);
        Assert.That(asked?.Item4, Is.EqualTo("desc"));
        r.View.SetBoxField(r.Node.Id, "editor_state", 0, "desc", "a red apple");
        r.Click(BoxesWidget.Plus(r.Lay.PaletteRow, 0).Center);
        r.Click(BoxesWidget.Swatch(r.Lay.PaletteRow, 0).Center);
        Assert.That(asked?.Item4, Is.EqualTo("palette:0"));
        r.View.SetBoxField(r.Node.Id, "editor_state", 0, "palette:0", "#ff0000");
        Assert.That(r.State, Is.EqualTo("[{\"x\":192,\"y\":192,\"width\":512,\"height\":416,\"metadata\":{\"type\":\"text\",\"text\":\"SALE\",\"desc\":\"a red apple\",\"palette\":[\"#ff0000\"]}}]"));

        r.Click(r.At(0.5, 0.5));
        Assert.That(r.View.Key("delete"), Is.True);
        Assert.That(r.State, Is.EqualTo("[]"));
        Assert.That(r.View.Document.Graph.FindNode(r.Node.Id), Is.Not.Null, "the node stays");
        r.View.Document.Undo();
        Assert.That(BoxesWidget.Boxes(r.Node.Widgets["editor_state"]), Has.Count.EqualTo(1), "undo brings the box back");
    }

    [Test]
    public void A_selected_boxs_corner_resizes_it()
    {
        var r = Setup();
        r.Drag(0.2, 0.2, 0.7, 0.6);
        // ComfyUI: the bottom-right corner dragged by 0.1 of the canvas makes it 608 x 512.
        var corner = r.At(704 / 1024.0, 608 / 1024.0);
        r.View.PointerDown(r.View.Viewport.WorldToScreen(corner), PointerButton.Left);
        var to = new Point(corner.X + 0.1 * r.Lay.Canvas.Width, corner.Y + 0.1 * r.Lay.Canvas.Height);
        r.View.PointerMove(r.View.Viewport.WorldToScreen(to));
        r.View.PointerUp(r.View.Viewport.WorldToScreen(to), PointerButton.Left);
        Assert.That(r.State, Does.StartWith("[{\"x\":192,\"y\":192,\"width\":608,\"height\":512,"));
    }
}
