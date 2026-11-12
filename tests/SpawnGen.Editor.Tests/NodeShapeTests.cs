using System.Text.Json.Nodes;
using SpawnGen.Editor;

namespace SpawnGen.Editor.Tests;

/// <summary>Node shapes (the node menu's Shapes) in ComfyUI's format: <c>shape</c> 1 box, 2 round, 4 card, written only when set (measured from its frontend).</summary>
public class NodeShapeTests
{
    static JsonObject SavedNode(EditorDocument doc, int id) =>
        JsonNode.Parse(ComfyWorkflow.Save(doc))!["nodes"]!.AsArray().OfType<JsonObject>().Single(n => (int)n["id"]! == id);

    [Test]
    public void A_shape_is_saved_as_ComfyUIs_number_and_read_back()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        Assert.That(SavedNode(doc, ids["sampler"]).ContainsKey("shape"), Is.False, "the default is not written");
        doc.SetShape(ids["sampler"], NodeShape.Card);
        doc.SetShape(ids["save"], NodeShape.Box);
        doc.SetShape(ids["decode"], NodeShape.Round);
        Assert.That((int)SavedNode(doc, ids["sampler"])["shape"]!, Is.EqualTo(4));
        Assert.That((int)SavedNode(doc, ids["save"])["shape"]!, Is.EqualTo(1));
        Assert.That((int)SavedNode(doc, ids["decode"])["shape"]!, Is.EqualTo(2), "round chosen is written, as ComfyUI writes it");

        var back = new EditorDocument(doc.Registry);
        ComfyWorkflow.Load(ComfyWorkflow.Save(doc), back);
        Assert.That(back.Graph.GetNode(ids["sampler"]).Shape, Is.EqualTo(NodeShape.Card));
        Assert.That(back.Graph.GetNode(ids["save"]).Shape, Is.EqualTo(NodeShape.Box));
        Assert.That(back.Graph.GetNode(ids["pos"]).Shape, Is.Null);
    }

    [Test]
    public void Names_and_unknown_values_are_read_safely()
    {
        Assert.That(ComfyWorkflow.ReadShape(JsonValue.Create("card")), Is.EqualTo(NodeShape.Card));
        Assert.That(ComfyWorkflow.ReadShape(JsonValue.Create("BOX")), Is.EqualTo(NodeShape.Box));
        Assert.That(ComfyWorkflow.ReadShape(JsonValue.Create(2.0)), Is.EqualTo(NodeShape.Round));
        Assert.That(ComfyWorkflow.ReadShape(JsonValue.Create(3)), Is.Null, "3 (circle) is not a node shape ComfyUI offers");
        Assert.That(ComfyWorkflow.ReadShape(JsonValue.Create("hexagon")), Is.Null);
        Assert.That(ComfyWorkflow.ReadShape(null), Is.Null);
    }

    [Test]
    public void Setting_a_shape_is_one_undo_step_and_copies_keep_shape_and_colour()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        doc.SetShape(ids["sampler"], NodeShape.Box);
        doc.SetNodeColor(ids["sampler"], "#432", "#653");
        doc.EndGesture();
        doc.SetShape(ids["sampler"], null);
        doc.EndGesture();
        Assert.That(doc.Graph.GetNode(ids["sampler"]).Shape, Is.Null);
        doc.Undo();
        Assert.That(doc.Graph.GetNode(ids["sampler"]).Shape, Is.EqualTo(NodeShape.Box));

        var view = new EditorView(doc);
        view.Duplicate([ids["sampler"]]);
        var copy = doc.Graph.GetNode(view.SelectedNodes.Single());
        Assert.That((copy.Shape, copy.Color, copy.BgColor), Is.EqualTo(((NodeShape?)NodeShape.Box, "#432", "#653")));
        Assert.That(doc.Graph.GetNode(ids["sampler"]).Clone(999).Shape, Is.EqualTo(NodeShape.Box));
    }
}

/// <summary>ComfyUI's Mode menu: LiteGraph's On Event (1) and On Trigger (3) are kept in the workflow and run like Always (measured: its frontend sends them).</summary>
public class NodeModeTests
{
    [Test]
    public void Event_and_trigger_modes_are_kept_and_run_like_always()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        doc.SetMode(ids["decode"], NodeMode.OnEvent);
        doc.SetMode(ids["sampler"], NodeMode.OnTrigger);
        var json = ComfyWorkflow.Save(doc);
        var back = new EditorDocument(doc.Registry);
        ComfyWorkflow.Load(json, back);
        Assert.That((back.Graph.GetNode(ids["decode"]).Mode, back.Graph.GetNode(ids["sampler"]).Mode), Is.EqualTo((NodeMode.OnEvent, NodeMode.OnTrigger)));
        var compiled = GraphCompiler.Compile(back.Graph, back.Registry).Graph;
        Assert.That(compiled.Nodes.ContainsKey(ids["decode"].ToString()) && compiled.Nodes.ContainsKey(ids["sampler"].ToString()), Is.True);
        Assert.That(compiled.Nodes[ids["save"].ToString()].Inputs["images"].NodeId, Is.EqualTo(ids["decode"].ToString()));
        Assert.That((NodeMode.OnEvent.Runs(), NodeMode.OnTrigger.Runs(), NodeMode.Never.Runs(), NodeMode.Bypass.Runs()), Is.EqualTo((true, true, false, false)));
    }

    [Test]
    public void Resize_drops_a_hand_set_width()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        doc.SetWidth(ids["sampler"], 900);
        doc.EndGesture();
        doc.SetWidth(ids["sampler"], null);
        Assert.That(doc.Graph.GetNode(ids["sampler"]).Width, Is.Null);
        doc.Undo();
        Assert.That(doc.Graph.GetNode(ids["sampler"]).Width, Is.EqualTo(900));
    }
}
