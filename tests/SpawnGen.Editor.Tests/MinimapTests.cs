using SpawnGen.Editor;

namespace SpawnGen.Editor.Tests;

public class MinimapTests
{
    static (EditorView View, EditorDocument Doc, Dictionary<string, int> Ids) Setup()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        var view = new EditorView(doc) { ScreenSize = new Point(1200, 800) };
        view.FitToView();
        return (view, doc, ids);
    }

    [Test]
    public void The_minimap_sits_in_the_bottom_right_corner_and_shows_every_node_and_the_view()
    {
        var (view, doc, _) = Setup();
        var r = view.MinimapScreenRect;
        Assert.That(r.Right, Is.LessThanOrEqualTo(1200 - EditorView.MinimapMargin + 0.001));
        Assert.That(r.Bottom, Is.LessThanOrEqualTo(800 - EditorView.MinimapMargin + 0.001));

        var map = new Rect(r.X, r.Y, r.Width, r.Height);
        foreach (var n in doc.Graph.Nodes)
        {
            var b = view.Hits.Geometry(n).Bounds;
            Assert.That(map.Contains(view.WorldToMinimap(new Point(b.X, b.Y))), Is.True, $"top left of node {n.Id}");
            Assert.That(map.Contains(view.WorldToMinimap(new Point(b.Right, b.Bottom))), Is.True, $"bottom right of node {n.Id}");
        }
        var vis = view.Viewport.VisibleWorld(1200, 800);
        Assert.That(map.Contains(view.WorldToMinimap(new Point(vis.X, vis.Y))), Is.True, "the view's corner");
        Assert.That(map.Contains(view.WorldToMinimap(new Point(vis.Right, vis.Bottom))), Is.True, "the view's other corner");
    }

    [Test]
    public void World_and_minimap_points_convert_both_ways()
    {
        var (view, _, _) = Setup();
        var world = new Point(123, 45);
        var back = view.MinimapToWorld(view.WorldToMinimap(world));
        Assert.That(back.X, Is.EqualTo(world.X).Within(1e-6));
        Assert.That(back.Y, Is.EqualTo(world.Y).Within(1e-6));
    }

    [Test]
    public void Clicking_the_minimap_centres_the_view_there_and_dragging_follows_without_touching_the_graph()
    {
        var (view, doc, ids) = Setup();
        var before = doc.Graph.Nodes.Select(n => (n.X, n.Y)).ToList();
        var r = view.MinimapScreenRect;
        var target = new Point(r.X + 40, r.Y + 30);
        var targetWorld = view.MinimapToWorld(target);

        view.PointerDown(target, PointerButton.Left);
        var centre = view.Viewport.ScreenToWorld(new Point(600, 400));
        Assert.That(centre.X, Is.EqualTo(targetWorld.X).Within(1e-6));
        Assert.That(centre.Y, Is.EqualTo(targetWorld.Y).Within(1e-6));

        var other = new Point(r.X + 150, r.Y + 100);
        var otherWorld = view.MinimapToWorld(other); // the map holds still while dragging, so this is what the drag aims at
        view.PointerMove(other);
        var moved = view.Viewport.ScreenToWorld(new Point(600, 400));
        Assert.That(moved.X, Is.EqualTo(otherWorld.X).Within(1e-6));
        view.PointerUp(other, PointerButton.Left);

        Assert.That(doc.Graph.Nodes.Select(n => (n.X, n.Y)).ToList(), Is.EqualTo(before), "no node moved or got selected");
        Assert.That(view.SelectedNodes, Is.Empty);
        Assert.That(doc.CanUndo, Is.False, "moving the view is not an edit");
    }

    [Test]
    public void With_the_minimap_hidden_a_press_there_goes_to_the_canvas_as_usual()
    {
        var (view, _, _) = Setup();
        view.MinimapVisible = false;
        var scale = view.Viewport.Scale;
        var r = view.MinimapScreenRect;
        var p = new Point(r.X + 10, r.Y + 10);
        view.PointerDown(p, PointerButton.Left);
        view.PointerMove(new Point(p.X - 50, p.Y - 20));
        view.PointerUp(new Point(p.X - 50, p.Y - 20), PointerButton.Left);
        Assert.That(view.Viewport.Scale, Is.EqualTo(scale));
    }
}
