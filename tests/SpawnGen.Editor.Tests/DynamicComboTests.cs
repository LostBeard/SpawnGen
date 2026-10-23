using System.Text.Json.Nodes;
using SpawnGen.Editor;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor.Tests;

/// <summary>A dynamic combo (Resize Image/Mask's resize_type) shows, saves and sends only the chosen option's inputs, in ComfyUI's order.</summary>
public class DynamicComboTests
{
    static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(SpawnGen.Nodes.Core.SaveImage).Assembly).AddAssembly(typeof(SpawnGen.Nodes.ML.KSampler).Assembly);

    static JsonObject Saved(EditorDocument doc, int id) =>
        JsonNode.Parse(ComfyWorkflow.Save(doc))!["nodes"]!.AsArray().First(n => (int)n!["id"]! == id)!.AsObject();

    static List<object?> Values(JsonObject node) => node["widgets_values"]!.AsArray().Select(v => (object?)v?.ToJsonString()).ToList();

    [Test]
    public void The_schema_keeps_each_options_inputs_in_its_own_order()
    {
        var s = Registry().GetSchema("ResizeImageMaskNode")!;
        Assert.That(s.Inputs[0].Name, Is.EqualTo("input"));
        Assert.That(s.Inputs[1].Name, Is.EqualTo("resize_type"));
        Assert.That(s.Inputs.Last().Name, Is.EqualTo("scale_method"));
        Assert.That(s.DynamicCombos["resize_type"]["match size"], Is.EqualTo(new[] { "resize_type.match", "resize_type.crop" }));
        Assert.That(s.FindInput("resize_type.width")!.DynamicOptions, Is.EquivalentTo(new[] { "scale dimensions", "scale width" }));
        Assert.That(s.FindInput("resize_type")!.Widget!.ResolveOptions()[0], Is.EqualTo("scale dimensions"));
    }

    [Test]
    public void Widget_values_are_read_and_written_with_the_chosen_options_inputs_after_the_combo()
    {
        var doc = new EditorDocument(Registry());
        var node = doc.AddNode("ResizeImageMaskNode", 0, 0);
        Assert.That(Values(Saved(doc, node.Id)), Is.EqualTo(new object?[] { "\"scale dimensions\"", "512", "512", "\"center\"", "\"area\"" }), "ComfyUI's default");

        doc.SetWidget(node.Id, "resize_type", "scale width");
        doc.SetWidget(node.Id, "resize_type.width", 300L);
        Assert.That(Values(Saved(doc, node.Id)), Is.EqualTo(new object?[] { "\"scale width\"", "300", "\"area\"" }));
        Assert.That(NodeLayout.VisibleInputs(doc.Registry.GetSchema(node.ClassType)!, node, doc.Graph).Select(i => i.Name),
            Is.EqualTo(new[] { "input", "resize_type", "resize_type.width", "scale_method" }));

        doc.SetWidget(node.Id, "resize_type", "match size");
        Assert.That(Values(Saved(doc, node.Id)), Is.EqualTo(new object?[] { "\"match size\"", "\"center\"", "\"area\"" }), "match is a socket: only crop is a value");
    }

    [Test]
    public void A_ComfyUI_saved_node_opens_with_its_option_and_values_and_sends_only_them()
    {
        var doc = new EditorDocument(Registry());
        var node = doc.AddNode("ResizeImageMaskNode", 0, 0);
        var json = JsonNode.Parse(ComfyWorkflow.Save(doc))!;
        var saved = json["nodes"]!.AsArray().First(n => (int)n!["id"]! == node.Id)!;
        saved["widgets_values"] = new JsonArray("scale longer dimension", 768, "lanczos");

        var reopened = new EditorDocument(Registry());
        ComfyWorkflow.Load(json.ToJsonString(), reopened);
        var n = reopened.Graph.GetNode(node.Id);
        Assert.That((n.Widgets["resize_type"], n.Widgets["resize_type.longer_size"], n.Widgets["scale_method"]), Is.EqualTo(("scale longer dimension", 768L, "lanczos")));

        var prompt = GraphCompiler.Compile(reopened.Graph, reopened.Registry).Graph.Nodes[node.Id.ToString()];
        Assert.That(prompt.Inputs.Keys, Is.EquivalentTo(new[] { "resize_type", "resize_type.longer_size", "scale_method" }));
    }
}
