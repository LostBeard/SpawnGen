using SpawnGen.Editor;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor.Tests;

/// <summary>The CURVE widget (Curve Editor): a plot edited with the pointer: add a point, drag it, remove it with a double-click, toggle the interpolation.</summary>
public class CurveWidgetTests
{
    static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(SpawnGen.Nodes.Core.SaveImage).Assembly);

    static (EditorView View, EditorDocument Doc, EditorNode Node, WidgetRow Row, Rect Plot) Setup()
    {
        var doc = new EditorDocument(Registry());
        var node = doc.AddNode("CurveEditor", 0, 0);
        var view = new EditorView(doc);
        var row = view.Hits.Geometry(node).Widgets.Single(w => w.Name == "curve");
        return (view, doc, node, row, CurveWidget.Plot(row.Bounds));
    }

    static Point S(EditorView v, Point world) => v.Viewport.WorldToScreen(world);

    [Test]
    public void The_curve_is_a_plot_and_the_pointer_adds_drags_and_removes_points()
    {
        var (v, doc, node, row, plot) = Setup();
        Assert.That(row.IsCurve, Is.True);
        Assert.That(row.Bounds.Height, Is.GreaterThan(100), "a plot, not a one-line row");
        Assert.That(CurveWidget.Points(node.Widgets["curve"]), Is.EqualTo(new[] { (0.0, 0.0), (1.0, 1.0) }));

        // A press on empty space adds a point there; dragging moves it.
        var at = CurveWidget.ToWorld(plot, (0.5, 0.2));
        v.PointerDown(S(v, at), PointerButton.Left);
        v.PointerMove(S(v, CurveWidget.ToWorld(plot, (0.55, 0.7))));
        v.PointerUp(S(v, CurveWidget.ToWorld(plot, (0.55, 0.7))), PointerButton.Left);
        var points = CurveWidget.Points(node.Widgets["curve"]);
        Assert.That(points, Has.Count.EqualTo(3));
        Assert.That(points[1].X, Is.EqualTo(0.55).Within(0.01));
        Assert.That(points[1].Y, Is.EqualTo(0.7).Within(0.01));

        // A dragged point stays between its neighbours' x.
        var mid = CurveWidget.ToWorld(plot, points[1]);
        v.PointerDown(S(v, mid), PointerButton.Left);
        v.PointerMove(S(v, CurveWidget.ToWorld(plot, (1.0, 0.5))));
        v.PointerUp(S(v, CurveWidget.ToWorld(plot, (1.0, 0.5))), PointerButton.Left);
        Assert.That(CurveWidget.Points(node.Widgets["curve"])[1].X, Is.LessThanOrEqualTo(1.0));
        Assert.That(CurveWidget.Points(node.Widgets["curve"]).Select(p => p.X), Is.Ordered);

        // The label toggles the interpolation.
        var label = CurveWidget.Label(row.Bounds);
        v.PointerDown(S(v, label.Center), PointerButton.Left);
        v.PointerUp(S(v, label.Center), PointerButton.Left);
        Assert.That(CurveWidget.Interpolation(node.Widgets["curve"]), Is.EqualTo("linear"));

        // A double-click on a point removes it; the last two stay.
        var p1 = CurveWidget.ToWorld(plot, CurveWidget.Points(node.Widgets["curve"])[1]);
        v.PointerDown(S(v, p1), PointerButton.Left, clickCount: 2);
        v.PointerUp(S(v, p1), PointerButton.Left);
        Assert.That(CurveWidget.Points(node.Widgets["curve"]), Has.Count.EqualTo(2));
        var p0 = CurveWidget.ToWorld(plot, (0.0, 0.0));
        v.PointerDown(S(v, p0), PointerButton.Left, clickCount: 2);
        v.PointerUp(S(v, p0), PointerButton.Left);
        Assert.That(CurveWidget.Points(node.Widgets["curve"]), Has.Count.EqualTo(2), "a curve keeps two points");

        doc.Undo();
        Assert.That(CurveWidget.Points(node.Widgets["curve"]), Has.Count.EqualTo(3), "undo brings the removed point back");
    }
}
