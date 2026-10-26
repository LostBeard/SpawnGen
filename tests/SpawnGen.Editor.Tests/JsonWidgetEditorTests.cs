using System.Text.Json.Nodes;
using SpawnGen.Editor;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Editor.Tests;

/// <summary>
/// Object-valued widgets (ComfyUI's BOUNDING_BOX on Crop Image, BOUNDING_BOXES on Create Bounding Boxes) through the editor: a ComfyUI workflow opens with
/// them, saves them back as ComfyUI writes them, compiles them into a prompt (a list wrapped as {"__value__": ...}) and the prompt runs.
/// </summary>
public class JsonWidgetEditorTests
{
    // As ComfyUI saves these nodes: the box object and the box list sit in widgets_values like any other value.
    const string Workflow = """
    {
      "last_node_id": 3, "last_link_id": 2,
      "nodes": [
        {"id": 1, "type": "EmptyImage", "pos": [0, 0], "size": [315, 130], "flags": {}, "order": 0, "mode": 0,
         "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [1]}], "properties": {}, "widgets_values": [40, 30, 1, 3368601]},
        {"id": 2, "type": "ImageCropV2", "pos": [400, 0], "size": [315, 80], "flags": {}, "order": 1, "mode": 0,
         "inputs": [{"name": "image", "type": "IMAGE", "link": 1}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [2]}],
         "properties": {}, "widgets_values": [{"x": 8, "y": 4, "width": 20, "height": 10}]},
        {"id": 3, "type": "PreviewImage", "pos": [800, 0], "size": [210, 250], "flags": {}, "order": 2, "mode": 0,
         "inputs": [{"name": "images", "type": "IMAGE", "link": 2}], "properties": {}},
        {"id": 4, "type": "CreateBoundingBoxes", "pos": [0, 300], "size": [400, 400], "flags": {}, "order": 3, "mode": 0,
         "outputs": [{"name": "preview", "type": "IMAGE", "links": []}, {"name": "bboxes", "type": "BOUNDING_BOX", "links": []},
                     {"name": "elements", "type": "ARRAY", "links": []}],
         "properties": {}, "widgets_values": [256, 128, [{"x": 10, "y": 20, "width": 30, "height": 40, "metadata": {"type": "obj", "desc": "box", "palette": ["#FF0000"]}}], []]}
      ],
      "links": [[1, 1, 0, 2, 0, "IMAGE"], [2, 2, 0, 3, 0, "IMAGE"]],
      "groups": [], "config": {}, "extra": {}, "version": 0.4
    }
    """;

    [Test]
    public void A_workflow_opens_and_saves_its_object_valued_widgets()
    {
        var doc = new EditorDocument(Fixtures.Registry());
        var warnings = ComfyWorkflow.Load(Workflow, doc);
        Assert.That(warnings, Is.Empty);
        var crop = doc.Graph.GetNode(2);
        Assert.That(crop.Widgets["crop_region"], Is.InstanceOf<DictValue>());
        Assert.That(JsonLiteral.ToJson(crop.Widgets["crop_region"]), Is.EqualTo("""{"x":8,"y":4,"width":20,"height":10}"""));
        var boxes = doc.Graph.GetNode(4);
        Assert.That(boxes.Widgets["editor_state"], Is.InstanceOf<ArrayValue>().And.Count.EqualTo(1));
        Assert.That(boxes.Widgets["last_incoming"], Is.InstanceOf<ArrayValue>().And.Empty);

        var saved = JsonNode.Parse(ComfyWorkflow.Save(doc))!["nodes"]!.AsArray();
        JsonNode Values(int id) => saved.First(n => (int)n!["id"]! == id)!["widgets_values"]!;
        Assert.That(Values(2).ToJsonString(), Is.EqualTo("""[{"x":8,"y":4,"width":20,"height":10}]"""));
        Assert.That(Values(4).ToJsonString(), Is.EqualTo(
            """[256,128,[{"x":10,"y":20,"width":30,"height":40,"metadata":{"type":"obj","desc":"box","palette":["#FF0000"]}}],[]]"""));
    }

    [Test]
    public void A_new_node_starts_from_the_widget_default()
    {
        var doc = new EditorDocument(Fixtures.Registry());
        var node = doc.AddNode("ImageCropV2", 0, 0);
        Assert.That(JsonLiteral.ToJson(node.Widgets["crop_region"]), Is.EqualTo("""{"x":0,"y":0,"width":512,"height":512}"""));
    }

    [Test]
    public async Task The_compiled_prompt_carries_the_values_and_runs()
    {
        var doc = new EditorDocument(Fixtures.Registry());
        ComfyWorkflow.Load(Workflow, doc);
        var prompt = GraphCompiler.Compile(doc.Graph, doc.Registry).Graph;
        var text = PromptJson.ToJson(prompt);
        Assert.That(text, Does.Contain("\"crop_region\":{\"x\":8,\"y\":4,\"width\":20,\"height\":10}"));
        Assert.That(text, Does.Contain("\"editor_state\":{\"__value__\":[{\"x\":10"), "a list value is wrapped, as ComfyUI's frontend sends it");
        Assert.That(text, Does.Contain("\"last_incoming\":{\"__value__\":[]}"));

        // The prompt as the server receives it: parsed back from its text, then run.
        var store = new MemoryResultStore();
        var executor = new PromptExecutor(Fixtures.Registry(), new Services(store));
        var result = await executor.ExecuteAsync(PromptJson.Parse(text));
        var cropped = (ImageBatch)result.Outputs["3"].Values[0]!;
        Assert.That((cropped.Width, cropped.Height), Is.EqualTo((20, 10)));
        Assert.That(cropped.Data[0], Is.EqualTo(0x33 / 255f));
        var preview = (ImageBatch)result.Outputs["4"].Values[0]!;
        Assert.That((preview.Width, preview.Height), Is.EqualTo((256, 128)), "Create Bounding Boxes ran with the canvas size from its widgets");
        var bboxes = result.Outputs["4"].Values[1];
        Assert.That(JsonLiteral.ToJson(bboxes), Is.EqualTo("""[[{"x":10,"y":20,"width":30,"height":40}]]"""));
    }

    [Test]
    public void An_api_prompt_with_an_object_value_opens_in_the_editor()
    {
        var doc = new EditorDocument(Fixtures.Registry());
        ApiPromptImport.Load("""
        { "1": { "class_type": "EmptyImage", "inputs": { "width": 40, "height": 30, "batch_size": 1, "color": 0 } },
          "2": { "class_type": "ImageCropV2", "inputs": { "image": ["1", 0], "crop_region": {"x": 1, "y": 2, "width": 3, "height": 4} } } }
        """, doc);
        var crop = doc.Graph.Nodes.Single(n => n.ClassType == "ImageCropV2");
        Assert.That(JsonLiteral.ToJson(crop.Widgets["crop_region"]), Is.EqualTo("""{"x":1,"y":2,"width":3,"height":4}"""));
    }

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }
}
