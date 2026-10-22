using System.Text.Json.Nodes;
using SpawnGen.Editor;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor.Tests;

public class PinTests
{
    static (EditorView View, EditorDocument Doc, Dictionary<string, int> Ids) Setup()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        return (new EditorView(doc), doc, ids);
    }

    static void Drag(EditorView v, Point fromWorld, Point toWorld, int steps = 6)
    {
        var a = v.Viewport.WorldToScreen(fromWorld);
        var b = v.Viewport.WorldToScreen(toWorld);
        v.PointerDown(a, PointerButton.Left);
        for (int i = 1; i <= steps; i++) v.PointerMove(new Point(a.X + (b.X - a.X) * i / steps, a.Y + (b.Y - a.Y) * i / steps));
        v.PointerUp(b, PointerButton.Left);
    }

    [Test]
    public void P_pins_the_selected_nodes_and_pins_again_unpins_in_one_undo_step()
    {
        var (view, doc, ids) = Setup();
        view.SelectOnly(ids["pos"]);
        view.ToggleSelect(ids["neg"]);
        Assert.That(view.Key("p"), Is.True);
        Assert.That(doc.Graph.GetNode(ids["pos"]).Pinned && doc.Graph.GetNode(ids["neg"]).Pinned, Is.True);
        Assert.That(doc.Graph.GetNode(ids["sampler"]).Pinned, Is.False);
        view.Key("p");
        Assert.That(doc.Graph.Nodes.Any(n => n.Pinned), Is.False, "every selected node was pinned, so P unpins them");
        doc.Undo();
        Assert.That(doc.Graph.GetNode(ids["pos"]).Pinned && doc.Graph.GetNode(ids["neg"]).Pinned, Is.True, "the unpin was one step");
        Assert.That(new EditorView(Fixtures.Txt2Img().Doc).Key("p"), Is.False, "nothing selected");
    }

    [Test]
    public void A_pinned_node_cannot_be_dragged_but_the_others_in_the_selection_can()
    {
        var (view, doc, ids) = Setup();
        var pinned = doc.Graph.GetNode(ids["pos"]);
        var free = doc.Graph.GetNode(ids["neg"]);
        view.SelectOnly(ids["pos"]);
        view.Key("p");
        double px = pinned.X, py = pinned.Y, fx = free.X, fy = free.Y;

        view.ToggleSelect(ids["neg"]);
        var b = view.Hits.Geometry(pinned).TitleBar;
        Drag(view, new Point(b.X + 60, b.Y + 8), new Point(b.X + 160, b.Y + 58));

        Assert.That((pinned.X, pinned.Y), Is.EqualTo((px, py)), "the pinned node stays");
        Assert.That(free.X, Is.EqualTo(fx + 100).Within(1e-6), "the unpinned one moved with the drag");
        Assert.That(free.Y, Is.EqualTo(fy + 50).Within(1e-6));

        doc.SetPinned(ids["pos"], false);
        doc.MoveNodes([ids["pos"]], 10, 10);
        Assert.That(pinned.X, Is.EqualTo(px + 10), "unpinned, it moves again");
    }

    [Test]
    public void A_pinned_node_has_no_resize_handle_and_cannot_be_collapsed_from_the_canvas()
    {
        var (view, doc, ids) = Setup();
        var node = doc.Graph.GetNode(ids["sampler"]);
        var geo = view.Hits.Geometry(node);
        var grip = new Point(geo.ResizeHandle.X + geo.ResizeHandle.Width / 2, geo.ResizeHandle.Y + geo.ResizeHandle.Height / 2);
        var arrow = new Point(geo.TitleBar.X + 12, geo.TitleBar.Y + geo.TitleBar.Height / 2);
        Assert.That(view.Hits.Test(grip).Kind, Is.EqualTo(HitKind.ResizeHandle));
        Assert.That(view.Hits.Test(arrow).Kind, Is.EqualTo(HitKind.CollapseButton));

        doc.SetPinned(node.Id, true);
        Assert.That(view.Hits.Test(grip).Kind, Is.Not.EqualTo(HitKind.ResizeHandle));
        Assert.That(view.Hits.Test(arrow).Kind, Is.EqualTo(HitKind.TitleBar));
    }

    [Test]
    public void Delete_keeps_pinned_nodes_and_says_so()
    {
        var (view, doc, ids) = Setup();
        string? message = null;
        view.Message += m => message = m;
        int before = doc.Graph.Nodes.Count;
        view.SelectOnly(ids["pos"]);
        view.Key("p");
        view.Key("delete");
        Assert.That(doc.Graph.Nodes.Count, Is.EqualTo(before));
        Assert.That(message, Does.Contain("Pinned"));

        view.ToggleSelect(ids["neg"]);
        message = null;
        view.Key("delete");
        Assert.That(doc.Graph.FindNode(ids["neg"]), Is.Null, "the unpinned one is deleted");
        Assert.That(doc.Graph.FindNode(ids["pos"]), Is.Not.Null, "the pinned one stays");
        Assert.That(message, Does.Contain("kept"));
    }

    [Test]
    public void Pins_are_saved_as_ComfyUIs_flags_and_survive_open_and_copy()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        doc.SetPinned(ids["sampler"], true);
        doc.SetCollapsed(ids["sampler"], true);
        var json = ComfyWorkflow.Save(doc);
        var node = JsonNode.Parse(json)!["nodes"]!.AsArray().First(n => (int)n!["id"]! == ids["sampler"])!;
        Assert.That((bool?)node["flags"]!["pinned"], Is.True);
        Assert.That((bool?)node["flags"]!["collapsed"], Is.True);
        var other = JsonNode.Parse(json)!["nodes"]!.AsArray().First(n => (int)n!["id"]! == ids["save"])!;
        Assert.That(other["flags"]!["pinned"], Is.Null);

        var registry = new NodeRegistry().AddAssembly(typeof(SpawnGen.Nodes.Core.SaveImage).Assembly).AddAssembly(typeof(SpawnGen.Nodes.ML.KSampler).Assembly);
        var reopened = new EditorDocument(registry);
        ComfyWorkflow.Load(json, reopened);
        Assert.That(reopened.Graph.GetNode(ids["sampler"]).Pinned, Is.True);
        Assert.That(reopened.Graph.GetNode(ids["save"]).Pinned, Is.False);
        Assert.That(doc.Graph.GetNode(ids["sampler"]).Clone(999).Pinned, Is.True);
    }

    [Test]
    public void A_pinned_group_cannot_be_moved_resized_or_deleted_and_keeps_its_other_flags()
    {
        var (view, doc, ids) = Setup();
        var inside = doc.Graph.GetNode(ids["pos"]);
        var group = doc.AddGroupAround([inside], n => view.Hits.Geometry(n).Bounds)!;
        group.Extra = new JsonObject { ["flags"] = new JsonObject { ["other"] = 1 } };
        doc.ClearHistory();
        doc.SetGroupPinned(group.Id, true);
        Assert.That(group.Pinned, Is.True);
        Assert.That((int?)group.Extra!["flags"]!["other"], Is.EqualTo(1), "flags the editor does not know stay");

        double gx = group.X, gy = group.Y, gw = group.Width, nx = inside.X;
        Drag(view, new Point(group.X + 30, group.Y + 10), new Point(group.X + 130, group.Y + 70));
        Assert.That((group.X, group.Y, inside.X), Is.EqualTo((gx, gy, nx)));
        doc.SetGroupBounds(group.Id, group.X, group.Y, group.Width + 50, group.Height);
        Assert.That(group.Width, Is.EqualTo(gw));
        Assert.That(view.Hits.Test(new Point(group.X + group.Width - 4, group.Y + group.Height - 4)).Kind, Is.Not.EqualTo(HitKind.GroupResize));

        view.Key("delete");
        Assert.That(doc.Graph.Groups, Has.Count.EqualTo(1), "a pinned group is not deleted");

        var json = ComfyWorkflow.Save(doc);
        var written = JsonNode.Parse(json)!["groups"]![0]!;
        Assert.That((bool?)written["flags"]!["pinned"], Is.True);

        doc.SetGroupPinned(group.Id, false);
        Assert.That(group.Pinned, Is.False);
        Assert.That(group.Extra!["flags"]!["pinned"], Is.Null);
        doc.Undo();
        Assert.That(group.Pinned, Is.True);
    }
}
