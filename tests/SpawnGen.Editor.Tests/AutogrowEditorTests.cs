using System.Text.Json.Nodes;
using SpawnGen.Editor;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor.Tests;

/// <summary>An Autogrow group (Batch Images' images.image0, images.image1, ...) shows its connected members and one free one, and saves like ComfyUI.</summary>
public class AutogrowEditorTests
{
    static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(SpawnGen.Nodes.Core.SaveImage).Assembly).AddAssembly(typeof(SpawnGen.Nodes.ML.KSampler).Assembly);

    static List<string> Shown(EditorDocument doc, EditorNode node) =>
        NodeLayout.VisibleInputs(doc.Registry.GetSchema(node.ClassType)!, node, doc.Graph).Select(i => i.Name).ToList();

    [Test]
    public void The_node_shows_one_free_socket_after_the_last_connected_one()
    {
        var doc = new EditorDocument(Registry());
        var a = doc.AddNode("LoadImage", 0, 0);
        var batch = doc.AddNode("BatchImagesNode", 300, 0);
        Assert.That(Shown(doc, batch), Is.EqualTo(new[] { "images.image0" }), "the minimum, which is also the free one");

        doc.Connect(a.Id, 0, batch.Id, "images.image0", out _);
        Assert.That(Shown(doc, batch), Is.EqualTo(new[] { "images.image0", "images.image1" }));
        doc.Connect(a.Id, 0, batch.Id, "images.image1", out _);
        Assert.That(Shown(doc, batch), Is.EqualTo(new[] { "images.image0", "images.image1", "images.image2" }));

        doc.Undo();
        Assert.That(Shown(doc, batch), Is.EqualTo(new[] { "images.image0", "images.image1" }), "disconnecting shrinks the row again");
        var geo = NodeLayout.Compute(batch, doc.Registry.GetSchema(batch.ClassType)!, doc.Graph);
        Assert.That(geo.Inputs.Select(i => NodeLayout.Label(i.Name)), Is.EqualTo(new[] { "image0", "image1" }));
    }

    [Test]
    public void A_workflow_saves_the_shown_members_with_ComfyUIs_names_and_opens_again()
    {
        var doc = new EditorDocument(Registry());
        var a = doc.AddNode("LoadImage", 0, 0);
        var batch = doc.AddNode("BatchImagesNode", 300, 0);
        doc.Connect(a.Id, 0, batch.Id, "images.image0", out _);
        doc.Connect(a.Id, 0, batch.Id, "images.image1", out _);
        var json = JsonNode.Parse(ComfyWorkflow.Save(doc))!;
        var saved = json["nodes"]!.AsArray().First(n => (string?)n!["type"] == "BatchImagesNode")!;
        Assert.That(saved["inputs"]!.AsArray().Select(i => (string?)i!["name"]), Is.EqualTo(new[] { "images.image0", "images.image1", "images.image2" }));
        Assert.That(saved["widgets_values"]!.AsArray(), Is.Empty);
        Assert.That(json["links"]!.AsArray().Select(l => (int)l![4]!), Is.EquivalentTo(new[] { 0, 1 }), "links point at slots 0 and 1");

        var reopened = new EditorDocument(Registry());
        ComfyWorkflow.Load(json.ToJsonString(), reopened);
        Assert.That(reopened.Graph.LinkInto(batch.Id, "images.image1")?.FromNode, Is.EqualTo(a.Id));

        var prompt = GraphCompiler.Compile(reopened.Graph, reopened.Registry).Graph.Nodes[batch.Id.ToString()];
        Assert.That(prompt.Inputs.Keys, Is.EquivalentTo(new[] { "images.image0", "images.image1" }), "the prompt names the members as ComfyUI's does");
    }

    [Test]
    public void A_ComfyUI_blueprint_style_node_with_members_opens_connected()
    {
        // As ComfyUI's frontend saves a Batch Images node with two pictures connected.
        var doc = new EditorDocument(Registry());
        var a = doc.AddNode("LoadImage", 0, 0);
        var b = doc.AddNode("LoadImage", 0, 300);
        var batch = doc.AddNode("BatchImagesNode", 300, 0);
        doc.Connect(a.Id, 0, batch.Id, "images.image0", out _);
        doc.Connect(b.Id, 0, batch.Id, "images.image1", out _);
        var reopened = new EditorDocument(Registry());
        ComfyWorkflow.Load(ComfyWorkflow.Save(doc), reopened);
        Assert.That(Shown(reopened, reopened.Graph.GetNode(batch.Id)), Has.Count.EqualTo(3));
    }
}
