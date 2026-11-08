using System.Text.Json.Nodes;
using SpawnGen.Editor;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor.Tests;

/// <summary>
/// Custom Combo's frontend-defined widgets (ComfyUI's onCustomComboCreated): option text widgets that grow with one empty one last, a choice offering the
/// filled ones, a hidden index; saved as <c>[choice, index, option1, ...]</c> and sent in the prompt.
/// </summary>
public class CustomComboEditorTests
{
    static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(SpawnGen.Nodes.Core.SaveImage).Assembly);

    static Dictionary<string, object?> W(EditorNode n) => n.Widgets.Where(kv => kv.Value is not null).ToDictionary(kv => kv.Key, kv => kv.Value);

    [Test]
    public void Typing_options_grows_the_list_and_the_choice_and_index_follow()
    {
        var doc = new EditorDocument(Registry());
        var n = doc.AddNode("CustomCombo", 0, 0);
        Assert.That(W(n), Is.EquivalentTo(new Dictionary<string, object?> { ["choice"] = "", ["index"] = 0L, ["option1"] = "" }));

        doc.SetWidget(n.Id, "option1", "Gaussian"); doc.EndGesture();
        doc.SetWidget(n.Id, "option2", "Box"); doc.EndGesture();
        doc.SetWidget(n.Id, "option3", "Radial"); doc.EndGesture();
        Assert.That(CustomComboWidgets.Options(n.Widgets), Is.EqualTo(new[] { "Gaussian", "Box", "Radial", "" }), "one empty option last");
        Assert.That((n.Widgets["choice"], n.Widgets["index"]), Is.EqualTo(("Gaussian", 0L)), "the first option is chosen once there is one");

        var schema = doc.Registry.GetSchema("CustomCombo")!;
        var choice = NodeLayout.OrderedInputs(schema, n.Widgets).First();
        Assert.That(choice.Widget!.ResolveOptions(), Is.EqualTo(new[] { "Gaussian", "Box", "Radial" }));
        var geo = NodeLayout.Compute(n, schema, doc.Graph);
        Assert.That(geo.Widgets.Select(w => w.Name), Is.EqualTo(new[] { "choice", "option1", "option2", "option3", "option4" }), "the index is not drawn");

        doc.SetWidget(n.Id, "choice", "Radial"); doc.EndGesture();
        Assert.That(n.Widgets["index"], Is.EqualTo(2L));
        doc.SetWidget(n.Id, "option3", ""); doc.EndGesture();
        Assert.That(CustomComboWidgets.Options(n.Widgets), Is.EqualTo(new[] { "Gaussian", "Box", "" }), "clearing the last filled option drops the spare one");
        Assert.That((n.Widgets["choice"], n.Widgets["index"]), Is.EqualTo(("Gaussian", 0L)), "its option is gone, so the choice moves to the first");

        doc.Undo();
        Assert.That(CustomComboWidgets.Options(n.Widgets), Is.EqualTo(new[] { "Gaussian", "Box", "Radial", "" }));
        Assert.That((n.Widgets["choice"], n.Widgets["index"]), Is.EqualTo(("Radial", 2L)), "one undo step brings it all back");
    }

    [Test]
    public void A_ComfyUI_blueprint_node_opens_saves_and_compiles_like_ComfyUI()
    {
        // As ComfyUI's Image Blur blueprint stores it.
        var json = """
        {"last_node_id": 1, "last_link_id": 0, "nodes": [{"id": 1, "type": "CustomCombo", "pos": [0, 0], "size": [270, 200], "flags": {}, "order": 0, "mode": 0,
          "inputs": [{"label": "blur_type", "localized_name": "choice", "name": "choice", "type": "COMBO", "widget": {"name": "choice"}, "link": null}],
          "outputs": [{"name": "STRING", "type": "STRING", "links": null}, {"name": "INDEX", "type": "INT", "links": []}],
          "widgets_values": ["Box", 1, "Gaussian", "Box", "Radial", ""], "properties": {}}], "links": [], "groups": [], "config": {}, "extra": {}, "version": 0.4}
        """;
        var doc = new EditorDocument(Registry());
        ComfyWorkflow.Load(json, doc);
        var n = doc.Graph.GetNode(1);
        Assert.That((n.Widgets["choice"], n.Widgets["index"]), Is.EqualTo(("Box", 1L)));
        Assert.That(CustomComboWidgets.Options(n.Widgets), Is.EqualTo(new[] { "Gaussian", "Box", "Radial", "" }));

        var saved = JsonNode.Parse(ComfyWorkflow.Save(doc))!["nodes"]![0]!["widgets_values"]!.AsArray();
        Assert.That(saved.ToJsonString(), Is.EqualTo("""["Box",1,"Gaussian","Box","Radial",""]"""));

        var prompt = GraphCompiler.Compile(doc.Graph, doc.Registry).Graph.Nodes["1"];
        Assert.That(prompt.Inputs.Keys, Is.EqualTo(new[] { "choice", "index", "option1", "option2", "option3", "option4" }), "ComfyUI's frontend sends every widget");
    }

    [Test]
    public async Task The_server_side_node_returns_the_choice_and_index_without_checking_the_options()
    {
        var registry = Registry();
        var graph = new PromptGraph();
        graph.Add("1", "CustomCombo", ("choice", "Radial"), ("index", 2L), ("option1", "Gaussian"), ("option2", "Box"), ("option3", "Radial"), ("option4", ""));
        graph.Add("2", "PreviewAny", ("source", PromptInput.Link("1", 1)));
        var executor = new PromptExecutor(registry);
        Assert.That(executor.Validate(graph).IsValid, Is.True, "the choice is not checked against the (empty) option list");
        var result = await executor.ExecuteAsync(graph);
        Assert.That(result.Outputs["2"].Values[0], Is.EqualTo("2"), "INDEX");
    }
}
