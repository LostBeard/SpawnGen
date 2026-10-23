using System.Text.Json.Nodes;
using SpawnGen.Editor;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor.Tests;

/// <summary>Opening a subgraph to edit its inside, as ComfyUI does: boundary nodes for its ports, and the edit written back to the shared definition.</summary>
public class SubgraphEditingTests
{
    static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(SpawnGen.Nodes.Core.SaveImage).Assembly).AddAssembly(typeof(SpawnGen.Nodes.ML.KSampler).Assembly);

    /// <summary>Load Image -> [subgraph: Invert Image] -> Preview Image, made by packing the invert node.</summary>
    static (EditorDocument Doc, EditorNode Instance, SubgraphDefinition Def) Packed()
    {
        var doc = new EditorDocument(Registry());
        var load = doc.AddNode("LoadImage", 0, 0);
        var invert = doc.AddNode("ImageInvert", 300, 0);
        var preview = doc.AddNode("PreviewImage", 600, 0);
        doc.Connect(load.Id, 0, invert.Id, "image", out _);
        doc.Connect(invert.Id, 0, preview.Id, "images", out _);
        var instance = doc.ConvertToSubgraph([invert.Id], "Inverter", out var error)!;
        Assert.That(error, Is.Null);
        return (doc, instance, SubgraphLibrary.Get(instance.ClassType)!);
    }

    [Test]
    public void Opening_shows_the_inside_with_boundary_nodes_for_the_ports_and_committing_without_changes_keeps_it()
    {
        var (_, _, def) = Packed();
        var inside = new EditorDocument(Registry());
        SubgraphEditing.Open(inside, def);
        Assert.That(SubgraphEditing.EditedId(inside), Is.EqualTo(def.Id));
        var inner = def.Nodes.Single();
        Assert.That(inside.Graph.LinkInto(inner.Id, "image")?.FromNode, Is.EqualTo(SubgraphEditing.InputsNodeId), "the input port feeds the inner node");
        Assert.That(inside.Graph.LinksFrom(inner.Id, 0).Single().ToNode, Is.EqualTo(SubgraphEditing.OutputsNodeId));
        Assert.That(inside.Registry.GetSchema(inside.Graph.GetNode(SubgraphEditing.InputsNodeId).ClassType)!.Outputs.Single().Type.Name, Is.EqualTo("IMAGE"));

        int revision = def.Revision;
        SubgraphEditing.Commit(inside, def);
        Assert.That(def.Revision, Is.EqualTo(revision + 1));
        Assert.That((def.Nodes.Single().ClassType, def.Inputs[0].Targets.Single(), def.Outputs[0].Source), Is.EqualTo(("ImageInvert", (inner.Id, "image"), (inner.Id, 0))));
    }

    [Test]
    public void An_edit_inside_changes_what_every_instance_runs()
    {
        var (doc, _, def) = Packed();
        var inside = new EditorDocument(Registry());
        SubgraphEditing.Open(inside, def);
        // Put a blur between the input port and the invert node.
        var invert = def.Nodes.Single();
        var blur = inside.AddNode("ImageBlur", 100, 0);
        inside.Connect(SubgraphEditing.InputsNodeId, 0, blur.Id, "image", out var e1);
        inside.Connect(blur.Id, 0, invert.Id, "image", out var e2);
        Assert.That((e1, e2), Is.EqualTo(((string?)null, (string?)null)));
        SubgraphEditing.Commit(inside, def);

        Assert.That(def.Inputs[0].Targets.Single(), Is.EqualTo((blur.Id, "image")));
        var prompt = GraphCompiler.Compile(doc.Graph, doc.Registry).Graph;
        Assert.That(prompt.Nodes.Values.Select(n => n.ClassType), Is.SupersetOf(new[] { "ImageBlur", "ImageInvert" }));
        var blurred = prompt.Nodes.Single(kv => kv.Value.ClassType == "ImageBlur");
        Assert.That(prompt.Nodes.Values.Single(n => n.ClassType == "ImageInvert").Inputs["image"].NodeId, Is.EqualTo(blurred.Key));
    }

    [Test]
    public void Reopening_the_workflow_it_came_from_keeps_the_edited_definition()
    {
        var (doc, _, def) = Packed();
        var saved = ComfyWorkflow.Save(doc); // holds the definition before the edit
        var inside = new EditorDocument(Registry());
        SubgraphEditing.Open(inside, def);
        inside.AddNode("ImageBlur", 100, 200);
        SubgraphEditing.Commit(inside, def);
        int nodes = def.Nodes.Count;

        var reopened = new EditorDocument(Registry());
        ComfyWorkflow.Load(saved, reopened);
        Assert.That(SubgraphLibrary.Get(def.Id)!.Nodes, Has.Count.EqualTo(nodes), "the older copy in the file does not replace the edit");
        var resaved = JsonNode.Parse(ComfyWorkflow.Save(reopened))!["definitions"]!["subgraphs"]![0]!;
        Assert.That(((int)resaved["revision"]!, resaved["nodes"]!.AsArray().Count), Is.EqualTo((def.Revision, nodes)));
    }
}
