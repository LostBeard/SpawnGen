using SpawnGen.Editor;

namespace SpawnGen.Editor.Tests;

public class SettingsTests
{
    [Test]
    public void Settings_survive_a_round_trip_and_defaults_fill_the_gaps()
    {
        var s = new StudioSettings { LinkStyle = LinkStyle.Straight, SnapToGrid = true, GridSize = 25, ShowMinimap = false, ShowNodeIds = true };
        var back = StudioSettings.FromJson(s.ToJson());
        Assert.That((back.LinkStyle, back.SnapToGrid, back.GridSize, back.ShowMinimap), Is.EqualTo((LinkStyle.Straight, true, 25, false)));
        Assert.That(back.ShowNodeIds, Is.True);

        var empty = StudioSettings.FromJson(null);
        Assert.That((empty.LinkStyle, empty.SnapToGrid, empty.GridSize, empty.ShowMinimap), Is.EqualTo((LinkStyle.Spline, false, 10, true)));
        Assert.That(empty.ShowNodeIds, Is.False);
    }

    [TestCase("not json")]
    [TestCase("[1,2]")]
    [TestCase("{\"LinkStyle\": 99, \"GridSize\": 100000, \"SnapToGrid\": \"yes\"}")]
    public void Damaged_or_out_of_range_values_fall_back_to_the_defaults(string json)
    {
        var s = StudioSettings.FromJson(json);
        Assert.That((s.LinkStyle, s.SnapToGrid, s.GridSize), Is.EqualTo((LinkStyle.Spline, false, 10)));
    }

    [Test, NonParallelizable]
    public void Link_styles_change_the_polyline_and_hidden_links_cannot_be_hit()
    {
        var from = new Point(0, 0);
        var to = new Point(200, 100);
        try
        {
            LinkGeometry.Style = LinkStyle.Spline;
            Assert.That(LinkGeometry.Sample(from, to), Has.Count.EqualTo(33));

            LinkGeometry.Style = LinkStyle.Straight;
            var straight = LinkGeometry.Sample(from, to);
            Assert.That(straight, Is.EqualTo(new[] { from, new Point(100, 0), new Point(100, 100), to }), "right angles, the vertical run in the middle");
            Assert.That(LinkGeometry.Distance(new Point(100, 50), from, to), Is.EqualTo(0).Within(1e-9));
            Assert.That(LinkGeometry.Midpoint(from, to), Is.EqualTo(new Point(100, 50)));

            LinkGeometry.Style = LinkStyle.Linear;
            var linear = LinkGeometry.Sample(from, to);
            Assert.That(linear[1], Is.EqualTo(new Point(14, 0)));
            Assert.That(linear[2], Is.EqualTo(new Point(186, 100)));

            LinkGeometry.Style = LinkStyle.Hidden;
            Assert.That(LinkGeometry.Distance(new Point(100, 50), from, to), Is.EqualTo(double.MaxValue));
        }
        finally { LinkGeometry.Style = LinkStyle.Spline; }
    }

    [Test]
    public void With_snapping_a_dragged_node_lands_on_the_grid_and_small_moves_still_add_up()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        var view = new EditorView(doc) { SnapToGrid = true, GridSize = 20 };
        var node = doc.Graph.GetNode(ids["sampler"]);
        var title = view.Hits.Geometry(node).TitleBar;
        var start = view.Viewport.WorldToScreen(new Point(title.X + 60, title.Y + 15));

        view.PointerDown(start, PointerButton.Left);
        // Eight 3-pixel steps to the right: 24 pixels in all. Each step alone is far below half a grid cell.
        for (int i = 1; i <= 8; i++) view.PointerMove(new Point(start.X + 3 * i, start.Y));
        view.PointerUp(new Point(start.X + 24, start.Y), PointerButton.Left);

        Assert.That(node.X % 20, Is.EqualTo(0).Within(1e-9), "on a grid line");
        Assert.That(node.Y % 20, Is.EqualTo(0).Within(1e-9));
    }

    [Test]
    public void Without_snapping_the_node_follows_the_pointer_exactly()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        var view = new EditorView(doc);
        var node = doc.Graph.GetNode(ids["sampler"]);
        double x0 = node.X;
        var title = view.Hits.Geometry(node).TitleBar;
        var start = view.Viewport.WorldToScreen(new Point(title.X + 60, title.Y + 15));
        view.PointerDown(start, PointerButton.Left);
        for (int i = 1; i <= 5; i++) view.PointerMove(new Point(start.X + 3 * i, start.Y));
        view.PointerUp(new Point(start.X + 15, start.Y), PointerButton.Left);
        Assert.That(node.X, Is.EqualTo(x0 + 15).Within(1e-9));
    }
}
