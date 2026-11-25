using SpawnGen.Editor;
using SpawnGen.Engine.Types;

namespace SpawnGen.Editor.Tests;

/// <summary>
/// The compositor editor's document and actions, against what ComfyUI's own editor stored and showed after the same actions (black box:
/// tools/studio-browser-test/probe-comfy-compositor-steps.js and -steps2.js, 2026-11-25): two 512 x 512 layers, "coat" at (0, 0) and "apple"
/// at (120, 60) with opacity 0.8, on a 632 x 572 canvas.
/// </summary>
public class CompositorEditorTests
{
    const string Start = "{\"version\":1,\"canvas\":{\"w\":632,\"h\":572},\"inputs\":[\"294ff15e38d1a1b3\",\"1dc6628ccc13a574\"],\"background\":{\"color\":\"#ffffff\",\"opacity\":1,\"visible\":false},";
    const string Coat = "{\"name\":\"coat\",\"visible\":true,\"opacity\":1,\"blend\":\"normal\",\"transform\":{\"x\":0,\"y\":0,\"w\":512,\"h\":512,\"rotation\":0},\"flipH\":false,\"flipV\":false}";

    static DictValue Box(string name, double x, double y, double opacity) => new()
    {
        ["x"] = (long)x, ["y"] = (long)y, ["width"] = 512L, ["height"] = 512L, ["rotation"] = 0.0, ["name"] = name,
        ["visible"] = true, ["opacity"] = opacity, ["blend"] = "normal", ["flipH"] = false, ["flipV"] = false,
    };

    static CompositorEditor Open(object? saved = null) =>
        new(["294ff15e38d1a1b3", "1dc6628ccc13a574"], [Box("coat", 0, 0, 1), Box("apple", 120, 60, 0.8)], (632, 572), saved);

    static string Json(CompositorEditor e) => JsonLiteral.ToJson(e.Value());

    static void Drag(CompositorEditor e, Action<CompositorEditor.Layer> moves)
    {
        var start = e.Layers[e.Selected!.Value].Clone();
        e.BeginGesture();
        moves(start);
        e.EndGesture();
    }

    [Test]
    public void Opens_with_the_top_layer_selected_and_stores_nothing()
    {
        var e = Open();
        Assert.Multiple(() =>
        {
            Assert.That(Json(e), Is.EqualTo("{}"));
            Assert.That(e.Selected, Is.EqualTo(1));
            Assert.That(e.Fields(), Is.EqualTo((120d, 60d, 512d, 512d, 0d, 80d)));
            Assert.That(e.Rows().Select(r => $"{r.Name} {r.Line}"), Is.EqualTo(new[] { "apple N 80%", "coat N 100%" }));
            Assert.That(e.BackgroundLine, Is.EqualTo("632×572"));
            Assert.That(e.CanUndo || e.CanRedo, Is.False);
        });
    }

    [Test]
    public void A_body_drag_moves_by_the_drag_and_writes_the_whole_composition()
    {
        var e = Open();
        // +45, +27 screen pixels at the editor's "90%" zoom
        Drag(e, s => { e.DragBody(s, 25, 15); e.DragBody(s, 50.00027470263439, 30.00018211066259); });
        Assert.Multiple(() =>
        {
            Assert.That(Json(e), Is.EqualTo(Start + "\"layers\":[" + Coat + ",{\"name\":\"apple\",\"visible\":true,\"opacity\":0.8,\"blend\":\"normal\",\"transform\":{\"x\":170.00027470263439,\"y\":90.00018211066259,\"w\":512,\"h\":512,\"rotation\":0},\"flipH\":false,\"flipV\":false}]}"));
            Assert.That(e.Fields(), Is.EqualTo((170d, 90d, 512d, 512d, 0d, 80d)));
        });
    }

    [Test]
    public void A_drag_is_one_undo_step_and_undo_redo_go_one_action_at_a_time()
    {
        var e = Open();
        Drag(e, s => { e.DragBody(s, 20, 10); e.DragBody(s, 50, 30); });
        Drag(e, s => e.DragHandle(s, CompositorEditor.Handle.BottomRight, 51, 50, false));
        Drag(e, s => e.DragHandle(s, CompositorEditor.Handle.BottomRight, 60, 10, false));
        Assert.That(e.Fields(), Is.EqualTo((170d, 90d, 623d, 572d, 0d, 80d)));   // free: the aspect is not kept
        Assert.That(e.Undo(), Is.True);
        Assert.That(e.Fields(), Is.EqualTo((170d, 90d, 563d, 562d, 0d, 80d)));
        Assert.That(e.Undo(), Is.True);
        Assert.That(e.Fields(), Is.EqualTo((170d, 90d, 512d, 512d, 0d, 80d)));
        Assert.That(e.Redo(), Is.True);
        Assert.That(e.Fields(), Is.EqualTo((170d, 90d, 563d, 562d, 0d, 80d)));
        Assert.That(e.Undo(), Is.True);
        Assert.That(e.Undo(), Is.True);
        Assert.That(e.Fields(), Is.EqualTo((120d, 60d, 512d, 512d, 0d, 80d)));
        Assert.That(e.Undo(), Is.False);
        // an action after an undo drops the redo stack
        e.FlipHorizontal();
        Assert.That(e.CanRedo, Is.False);
    }

    [Test]
    public void A_drag_that_changes_nothing_is_not_an_action()
    {
        var e = Open();
        e.BeginGesture();
        e.EndGesture();
        Assert.Multiple(() => { Assert.That(e.CanUndo, Is.False); Assert.That(Json(e), Is.EqualTo("{}")); });
    }

    [Test]
    public void Handles_resize_with_the_opposite_side_staying_put()
    {
        var e = Open();
        // the corner goes to the pointer: 412.4543 x 411.8987 stored, 412 x 412 shown, the top-left unmoved
        Drag(e, s => e.DragHandle(s, CompositorEditor.Handle.BottomRight, 412.4543006414307 - 512, 411.8986843453122 - 512, false));
        var l = e.Layers[1];
        Assert.That((l.X, l.Y, l.W, l.H), Is.EqualTo((120d, 60d, 412.4543006414307, 411.8986843453122)));
        Assert.That(e.Fields(), Is.EqualTo((120d, 60d, 412d, 412d, 0d, 80d)));
        e.Undo();
        // an edge changes one side only
        Drag(e, s => e.DragHandle(s, CompositorEditor.Handle.Right, 51, 7, false));
        Assert.That(e.Fields(), Is.EqualTo((120d, 60d, 563d, 512d, 0d, 80d)));
        e.Undo();
        // the top-left handle moves the top-left and keeps the bottom-right
        Drag(e, s => e.DragHandle(s, CompositorEditor.Handle.TopLeft, 12, 20, false));
        l = e.Layers[1];
        Assert.That((l.X, l.Y, l.W, l.H), Is.EqualTo((132d, 80d, 500d, 492d)));
        e.Undo();
        // a side never goes under one pixel
        Drag(e, s => e.DragHandle(s, CompositorEditor.Handle.Left, 900, 0, false));
        l = e.Layers[1];
        Assert.That((l.X, l.W), Is.EqualTo((631d, 1d)));
    }

    [Test]
    public void Shift_keeps_the_aspect_by_projecting_the_drag_on_the_diagonal()
    {
        var e = Open();
        Drag(e, s => e.DragHandle(s, CompositorEditor.Handle.BottomRight, 50, 10, true));
        var l = e.Layers[1];
        Assert.Multiple(() =>
        {
            Assert.That(e.Fields(), Is.EqualTo((120d, 60d, 542d, 542d, 0d, 80d)));
            Assert.That(l.W, Is.EqualTo(l.H).Within(1e-9));
            Assert.That((l.X, l.Y), Is.EqualTo((120d, 60d)));
        });
    }

    [Test]
    public void The_rotation_handle_turns_the_layer_to_the_pointer_about_its_centre()
    {
        var e = Open();
        // the layer's centre is (376, 316); a pointer level with it on the right is a quarter turn
        Drag(e, s => e.DragRotate(s, 376 + 280, 316));
        Assert.That(e.Layers[1].Rotation, Is.EqualTo(Math.PI / 2).Within(1e-12));
        Assert.That(e.Fields(), Is.EqualTo((120d, 60d, 512d, 512d, 90d, 80d)));
        // the measured drag ended half a screen pixel above the centre line: 1.5688117998196385 rad, still shown as 90
        Drag(e, s => e.DragRotate(s, 376 + 279.56, 316 - 0.5549));
        Assert.That(e.Layers[1].Rotation, Is.EqualTo(1.5688117998196385).Within(1e-5));
        Drag(e, s => e.DragRotate(s, 376, 316 + 100));
        Assert.That(e.Fields()!.Value.Degrees, Is.EqualTo(180d));
        Drag(e, s => e.DragRotate(s, 376 - 100, 316));
        Assert.That(e.Fields()!.Value.Degrees, Is.EqualTo(-90d));
    }

    [Test]
    public void A_rotated_layer_resizes_in_its_own_axes_with_the_opposite_corner_fixed()
    {
        var e = Open();
        e.SetRotationDegrees(90);
        // turned a quarter, the layer's own right is the canvas's down: its right edge handle follows a downward drag
        var before = e.Layers[1].Clone();
        Drag(e, s => e.DragHandle(s, CompositorEditor.Handle.Right, 0, 40, false));
        var l = e.Layers[1];
        Assert.Multiple(() =>
        {
            Assert.That(l.W, Is.EqualTo(552).Within(1e-9));
            Assert.That(l.H, Is.EqualTo(512).Within(1e-9));
            // the left edge's middle (on the canvas: straight above the centre) has not moved
            Assert.That(l.CenterX, Is.EqualTo(before.CenterX).Within(1e-9));
            Assert.That(l.CenterY - l.W / 2, Is.EqualTo(before.CenterY - before.W / 2).Within(1e-9));
        });
    }

    [Test]
    public void Alignment_is_to_the_canvas()
    {
        var e = Open();
        var seen = new List<(double, double)>();
        foreach (var a in Enum.GetValues<CompositorEditor.Alignment>())
        {
            e.Align(a);
            seen.Add((e.Fields()!.Value.X, e.Fields()!.Value.Y));
        }
        // left, horizontal centre, right (y stays 60), then top, vertical centre, bottom (x stays 120)
        Assert.That(seen, Is.EqualTo(new[] { (0d, 60d), (60d, 60d), (120d, 60d), (120d, 0d), (120d, 30d), (120d, 60d) }));
    }

    [Test]
    public void Flips_and_typed_fields_are_stored_as_measured()
    {
        var e = Open();
        e.FlipHorizontal();
        e.FlipVertical();
        Assert.That(e.Fields(), Is.EqualTo((120d, 60d, 512d, 512d, 0d, 80d)));
        e.SetRotationDegrees(30);
        e.SetOpacityPercent(55);
        e.SetPosition(10, null);
        e.SetSize(300, null);
        e.SetSize(null, 100);
        Assert.Multiple(() =>
        {
            Assert.That(Json(e), Is.EqualTo(Start + "\"layers\":[" + Coat + ",{\"name\":\"apple\",\"visible\":true,\"opacity\":0.55,\"blend\":\"normal\",\"transform\":{\"x\":10,\"y\":60,\"w\":300,\"h\":100,\"rotation\":0.5235987755982988},\"flipH\":true,\"flipV\":true}]}"));
            Assert.That(e.Fields(), Is.EqualTo((10d, 60d, 300d, 100d, 30d, 55d)));
            Assert.That(e.Rows().First().Line, Is.EqualTo("N 55%"));
        });
        e.SetSize(0, -4);
        Assert.That(e.Fields()!.Value.W, Is.EqualTo(300d));
    }

    [Test]
    public void The_blend_list_is_the_editors_and_every_id_is_one_the_node_knows()
    {
        Assert.That(CompositorEditor.BlendModes.Select(m => m.Name), Is.EqualTo(new[]
        {
            "Normal", "Multiply", "Screen", "Overlay", "Darken", "Lighten", "Color Dodge", "Color Burn", "Hard Light", "Soft Light", "Difference", "Exclusion",
            "Linear Dodge", "Linear Burn", "Vivid Light", "Pin Light", "Linear Light", "Hard Mix", "Subtract", "Divide", "Grain Extract", "Grain Merge", "Hue",
            "Saturation", "Color", "Luminosity",
        }));
        Assert.That(CompositorEditor.BlendModes.Select(m => m.Id), Is.EqualTo(SpawnGen.Nodes.Core.CompositorBlend.ModeNames));
        var e = Open();
        e.SetBlend("multiply");
        Assert.Multiple(() =>
        {
            Assert.That(e.Rows().First().Line, Is.EqualTo("M 80%"));
            Assert.That(Json(e), Does.Contain("\"name\":\"apple\",\"visible\":true,\"opacity\":0.8,\"blend\":\"multiply\""));
        });
        e.SetBlend("no-such-mode");
        Assert.That(e.Layers[1].Blend, Is.EqualTo("multiply"));
    }

    [Test]
    public void Reordering_writes_an_order_key_only_while_the_order_differs()
    {
        var e = Open();
        e.SetOpacityPercent(55);
        e.Select(0);
        Assert.That(e.Fields(), Is.EqualTo((0d, 0d, 512d, 512d, 0d, 100d)));
        Assert.That(e.MoveSelected(up: true), Is.True);
        Assert.Multiple(() =>
        {
            Assert.That(e.Rows().Select(r => r.Name), Is.EqualTo(new[] { "coat", "apple" }));
            // layers stay in input order; "order" lists input indices bottom first
            Assert.That(Json(e), Does.StartWith(Start + "\"order\":[1,0],\"layers\":[" + Coat + ",{\"name\":\"apple\""));
        });
        Assert.That(e.MoveSelected(up: true), Is.False);   // already on top
        Assert.That(e.MoveSelected(up: false), Is.True);
        Assert.Multiple(() =>
        {
            Assert.That(e.Rows().Select(r => r.Name), Is.EqualTo(new[] { "apple", "coat" }));
            Assert.That(Json(e), Does.Not.Contain("\"order\""));
        });
    }

    [Test]
    public void Visibility_toggles_and_the_background()
    {
        var e = Open();
        e.ToggleVisible(1);
        Assert.That(Json(e), Does.Contain("{\"name\":\"apple\",\"visible\":false,"));
        e.ToggleVisible(1);
        e.ToggleBackgroundVisible();
        Assert.That(Json(e), Does.Contain("\"background\":{\"color\":\"#ffffff\",\"opacity\":1,\"visible\":true}"));
        e.Select(null);   // a click on empty space: the Background
        Assert.That(e.Fields(), Is.Null);
        e.SetCanvasSize(700, null);
        e.SetBackgroundColor("#102030");
        Assert.Multiple(() =>
        {
            Assert.That(e.BackgroundLine, Is.EqualTo("700×572"));
            Assert.That(Json(e), Does.StartWith("{\"version\":1,\"canvas\":{\"w\":700,\"h\":572},"));
            Assert.That(Json(e), Does.Contain("\"color\":\"#102030\""));
        });
        // layer actions do nothing with the Background selected
        e.FlipHorizontal();
        e.Align(CompositorEditor.Alignment.Left);
        Assert.That(e.MoveSelected(true), Is.False);
        Assert.That(e.Layers.Any(l => l.FlipH), Is.False);
    }

    [Test]
    public void Restore_returns_to_the_composition_the_editor_opened_with()
    {
        var e = Open();
        e.SetOpacityPercent(55);
        e.SetPosition(10, 20);
        e.Restore();
        Assert.Multiple(() =>
        {
            Assert.That(e.Fields(), Is.EqualTo((120d, 60d, 512d, 512d, 0d, 80d)));
            Assert.That(e.Rows().First().Line, Is.EqualTo("N 80%"));
        });
        // reopened on the saved value, Restore keeps what was saved (measured: Close, open again, Restore changed nothing)
        e.SetBlend("multiply");
        var saved = JsonLiteral.Parse(Json(e));
        var again = Open(saved);
        Assert.That(again.Layers[1].Blend, Is.EqualTo("multiply"));
        var before = Json(again);
        again.Restore();
        Assert.That(Json(again), Is.EqualTo(before));
    }

    [Test]
    public void A_saved_value_is_taken_only_for_the_same_inputs()
    {
        var e = Open();
        e.SetPosition(10, 20);
        e.Select(0);
        e.MoveSelected(true);
        var saved = (DictValue)JsonLiteral.Parse(Json(e))!;
        var same = Open(saved);
        Assert.Multiple(() =>
        {
            Assert.That(Json(same), Is.EqualTo(Json(e)));
            Assert.That(same.Order, Is.EqualTo(new[] { 1, 0 }));
            Assert.That(same.Selected, Is.EqualTo(0));   // the top layer
        });
        saved["inputs"] = new ArrayValue { "294ff15e38d1a1b3", "ffffffffffffffff" };
        var other = Open(saved);
        Assert.Multiple(() =>
        {
            Assert.That(Json(other), Is.EqualTo("{}"));
            Assert.That(other.Fields(), Is.EqualTo((120d, 60d, 512d, 512d, 0d, 80d)));
        });
    }

    [Test]
    public void Hit_testing_finds_handles_of_the_selection_then_the_top_layer_under_the_point()
    {
        var e = Open();
        Assert.Multiple(() =>
        {
            Assert.That(e.HitTest(632, 572, 6, 24), Is.EqualTo(((int?)1, CompositorEditor.Handle.BottomRight)));
            Assert.That(e.HitTest(376, 60, 6, 24), Is.EqualTo(((int?)1, CompositorEditor.Handle.Top)));
            Assert.That(e.HitTest(376, 36, 6, 24), Is.EqualTo(((int?)1, CompositorEditor.Handle.Rotate)));
            Assert.That(e.HitTest(300, 300, 6, 24), Is.EqualTo(((int?)1, CompositorEditor.Handle.Body)));
            Assert.That(e.HitTest(50, 300, 6, 24), Is.EqualTo(((int?)0, CompositorEditor.Handle.Body)));     // only the coat is there
            Assert.That(e.HitTest(600, 20, 6, 24), Is.EqualTo(((int?)null, CompositorEditor.Handle.None)));  // neither: the Background
        });
        e.ToggleVisible(1);
        Assert.That(e.HitTest(300, 300, 6, 24).Layer, Is.EqualTo(0));   // a hidden layer is not picked through its body
    }
}
