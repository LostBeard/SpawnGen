using System.Text.Json.Nodes;
using SpawnGen.Editor;

namespace SpawnGen.Editor.Tests;

public class GroupTests
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

    Rect BoundsOf(EditorView v, EditorNode n) => v.Hits.Geometry(n).Bounds;

    [Test]
    public void Ctrl_G_makes_a_group_that_fits_the_selected_nodes_with_room_for_its_title_and_undo_removes_it()
    {
        var (view, doc, ids) = Setup();
        view.SelectOnly(ids["pos"]);
        view.ToggleSelect(ids["neg"]);
        Assert.That(view.Key("g", Modifiers.Ctrl), Is.True);

        var group = doc.Graph.Groups.Single();
        foreach (var id in new[] { ids["pos"], ids["neg"] })
        {
            var b = BoundsOf(view, doc.Graph.GetNode(id));
            Assert.That(group.Bounds.Contains(new Point(b.X, b.Y)) && group.Bounds.Contains(new Point(b.Right, b.Bottom)), Is.True, $"node {id} inside");
        }
        Assert.That(doc.Graph.GetNode(ids["pos"]).Y - group.Y, Is.GreaterThanOrEqualTo(EditorGroup.TitleHeight), "the title bar is above the nodes");
        Assert.That(view.SelectedGroup, Is.EqualTo(group.Id));

        doc.Undo();
        Assert.That(doc.Graph.Groups, Is.Empty);
    }

    [Test]
    public void Ctrl_G_with_nothing_selected_does_nothing()
    {
        var (view, doc, _) = Setup();
        Assert.That(view.Key("g", Modifiers.Ctrl), Is.False);
        Assert.That(doc.Graph.Groups, Is.Empty);
    }

    [Test]
    public void Dragging_a_group_title_moves_the_group_and_the_nodes_inside_it_as_one_undo_step()
    {
        var (view, doc, ids) = Setup();
        var inside = doc.Graph.GetNode(ids["pos"]);
        var outside = doc.Graph.GetNode(ids["decode"]);
        var group = doc.AddGroupAround([inside], n => BoundsOf(view, n));
        doc.ClearHistory();
        double gx = group!.X, gy = group.Y, nx = inside.X, ny = inside.Y, ox = outside.X, oy = outside.Y;

        Drag(view, new Point(group.X + 30, group.Y + 10), new Point(group.X + 30 + 100, group.Y + 10 + 60));

        Assert.That((group.X, group.Y), Is.EqualTo((gx + 100, gy + 60)));
        Assert.That((inside.X, inside.Y), Is.EqualTo((nx + 100, ny + 60)));
        Assert.That((outside.X, outside.Y), Is.EqualTo((ox, oy)), "a node outside the group stays");
        doc.Undo();
        Assert.That((group.X, group.Y, inside.X, inside.Y), Is.EqualTo((gx, gy, nx, ny)));
        Assert.That(doc.CanUndo, Is.False, "the whole drag was one step");
    }

    [Test]
    public void Dragging_the_group_body_pans_the_canvas_like_empty_space()
    {
        var (view, doc, _) = Setup();
        var group = doc.AddGroup(-600, 1200, 400, 300);
        var before = (group.X, group.Y);
        Drag(view, new Point(-400, 1400), new Point(-300, 1450));
        Assert.That((group.X, group.Y), Is.EqualTo(before));
        Assert.That(view.Viewport.OffsetX, Is.Not.EqualTo(0));
    }

    [Test]
    public void The_corner_resizes_a_group_down_to_a_minimum()
    {
        var (view, doc, _) = Setup();
        var group = doc.AddGroup(-600, 1200, 400, 300);
        doc.ClearHistory();
        var corner = new Point(group.X + group.Width - 5, group.Y + group.Height - 5);
        Drag(view, corner, new Point(corner.X + 80, corner.Y + 40));
        Assert.That((group.Width, group.Height), Is.EqualTo((480.0, 340.0)));

        Drag(view, new Point(group.X + group.Width - 5, group.Y + group.Height - 5), new Point(group.X - 500, group.Y - 500));
        Assert.That((group.Width, group.Height), Is.EqualTo((EditorGroup.MinWidth, EditorGroup.MinHeight)));
        Assert.That((group.X, group.Y), Is.EqualTo((-600.0, 1200.0)), "the top-left corner does not move");
    }

    [Test]
    public void Double_clicking_a_group_title_asks_to_edit_it_and_the_title_and_colour_can_change_with_undo()
    {
        var (view, doc, _) = Setup();
        var group = doc.AddGroup(-600, 1200, 400, 300);
        int? asked = null;
        view.GroupTitleEditRequested += id => asked = id;
        var p = view.Viewport.WorldToScreen(new Point(group.X + 50, group.Y + 10));
        view.PointerDown(p, PointerButton.Left, Modifiers.None, 2);
        view.PointerUp(p, PointerButton.Left);
        Assert.That(asked, Is.EqualTo(group.Id));

        doc.SetGroupTitle(group.Id, "Prompts");
        doc.SetGroupColor(group.Id, "#8A8");
        Assert.That((group.Title, group.Color), Is.EqualTo(("Prompts", "#8A8")));
        doc.Undo(); doc.Undo();
        Assert.That((group.Title, group.Color), Is.EqualTo(("Group", EditorGroup.DefaultColor)));
    }

    [Test]
    public void Delete_removes_the_selected_group_and_undo_puts_it_back_in_the_same_place()
    {
        var (view, doc, _) = Setup();
        var a = doc.AddGroup(-600, 1200, 400, 300, "A");
        var b = doc.AddGroup(-100, 1200, 400, 300, "B");
        var c = doc.AddGroup(400, 1200, 400, 300, "C");
        var p = view.Viewport.WorldToScreen(new Point(b.X + 50, b.Y + 10));
        view.PointerDown(p, PointerButton.Left);
        view.PointerUp(p, PointerButton.Left);
        Assert.That(view.SelectedGroup, Is.EqualTo(b.Id));
        view.Key("delete");
        Assert.That(doc.Graph.Groups.Select(g => g.Title), Is.EqualTo(new[] { "A", "C" }));
        doc.Undo();
        Assert.That(doc.Graph.Groups.Select(g => g.Title), Is.EqualTo(new[] { "A", "B", "C" }));
    }

    [Test]
    public void Only_the_title_bar_and_corner_of_a_group_are_hit_and_nodes_in_front_win()
    {
        var (view, doc, ids) = Setup();
        var group = doc.AddGroupAround([doc.Graph.GetNode(ids["pos"])], n => BoundsOf(view, n));
        Assert.That(view.Hits.Test(new Point(group!.X + 20, group.Y + 10)).Kind, Is.EqualTo(HitKind.GroupTitle));
        Assert.That(view.Hits.Test(new Point(group.Right() - 4, group.Bottom() - 4)).Kind, Is.EqualTo(HitKind.GroupResize));
        Assert.That(view.Hits.Test(new Point(group.X + 4, group.Bottom() - 40)).Kind, Is.EqualTo(HitKind.None), "the body is the canvas");
        var title = doc.Graph.GetNode(ids["pos"]);
        Assert.That(view.Hits.Test(new Point(title.X + 60, title.Y + 10)).Kind, Is.EqualTo(HitKind.TitleBar), "the node's own title bar");
    }

    [Test]
    public void Groups_round_trip_through_ComfyUI_workflow_json_with_unknown_fields_kept()
    {
        const string json = """
        {"last_node_id": 0, "last_link_id": 0, "version": 0.4, "nodes": [], "links": [],
         "groups": [{"id": 7, "title": "Sampling", "bounding": [100, 50, 600, 400], "color": "#8A8", "font_size": 30, "flags": {"pinned": true}}],
         "config": {}, "extra": {}}
        """;
        var doc = new EditorDocument(Fixtures.Registry());
        ComfyWorkflow.Load(json, doc);
        var g = doc.Graph.Groups.Single();
        Assert.That((g.Id, g.Title, g.X, g.Y, g.Width, g.Height, g.Color, g.FontSize), Is.EqualTo((7, "Sampling", 100.0, 50.0, 600.0, 400.0, "#8A8", 30.0)));

        doc.SetGroupTitle(7, "Renamed");
        var saved = JsonNode.Parse(ComfyWorkflow.Save(doc))!["groups"]!.AsArray().Single()!.AsObject();
        Assert.That((string)saved["title"]!, Is.EqualTo("Renamed"));
        Assert.That(saved["bounding"]!.AsArray().Select(v => (double)v!), Is.EqualTo(new[] { 100.0, 50.0, 600.0, 400.0 }));
        Assert.That((bool)saved["flags"]!["pinned"]!, Is.True, "fields the editor does not know survive");
        Assert.That(doc.Graph.NextGroupId, Is.EqualTo(8));
    }
}

static class RectEdges
{
    public static double Right(this EditorGroup g) => g.X + g.Width;
    public static double Bottom(this EditorGroup g) => g.Y + g.Height;
}
