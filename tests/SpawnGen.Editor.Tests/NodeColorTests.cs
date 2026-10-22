using System.Text.Json.Nodes;
using SpawnGen.Editor;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor.Tests;

public class NodeColorTests
{
    [Test]
    public void Colouring_a_node_is_one_undo_step_and_clearing_goes_back_to_the_theme()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        var id = ids["sampler"];
        doc.SetNodeColor(id, "#322", "#533");
        var node = doc.Graph.GetNode(id);
        Assert.That((node.Color, node.BgColor), Is.EqualTo(("#322", "#533")));
        doc.SetNodeColor(id, null, null);
        Assert.That((node.Color, node.BgColor), Is.EqualTo(((string?)null, (string?)null)));
        doc.Undo();
        Assert.That((node.Color, node.BgColor), Is.EqualTo(("#322", "#533")));
        doc.Undo();
        Assert.That(node.Color, Is.Null);
    }

    [Test]
    public void Colours_are_saved_with_ComfyUIs_keys_and_come_back_on_open()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        doc.SetNodeColor(ids["sampler"], "#232", "#353");
        var json = ComfyWorkflow.Save(doc);
        var node = JsonNode.Parse(json)!["nodes"]!.AsArray().First(n => (int)n!["id"]! == ids["sampler"])!;
        Assert.That((string?)node["color"], Is.EqualTo("#232"));
        Assert.That((string?)node["bgcolor"], Is.EqualTo("#353"));
        var other = JsonNode.Parse(json)!["nodes"]!.AsArray().First(n => (int)n!["id"]! == ids["save"])!;
        Assert.That(other["color"], Is.Null, "an uncoloured node has no colour keys");

        var registry = new NodeRegistry().AddAssembly(typeof(SpawnGen.Nodes.Core.SaveImage).Assembly).AddAssembly(typeof(SpawnGen.Nodes.ML.KSampler).Assembly);
        var reopened = new EditorDocument(registry);
        ComfyWorkflow.Load(json, reopened);
        var sampler = reopened.Graph.GetNode(ids["sampler"]);
        Assert.That((sampler.Color, sampler.BgColor), Is.EqualTo(("#232", "#353")));
    }

    [Test]
    public void A_copied_node_keeps_its_colours()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        doc.SetNodeColor(ids["sampler"], "#432", "#653");
        var copy = doc.Graph.GetNode(ids["sampler"]).Clone(999);
        Assert.That((copy.Color, copy.BgColor), Is.EqualTo(("#432", "#653")));
    }

    [Test]
    public void The_palette_has_ComfyUIs_nine_presets()
    {
        Assert.That(EditorGroup.NodePalette.Select(p => p.Name), Is.EqualTo(new[] { "red", "brown", "green", "blue", "pale blue", "cyan", "purple", "yellow", "black" }));
    }
}
