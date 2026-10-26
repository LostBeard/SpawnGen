using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Engine.Tests;

/// <summary>A node with object-valued widgets, as ComfyUI's BOUNDING_BOX (an object) and BOUNDING_BOXES (a list) widgets.</summary>
[Node(Category = "test")]
public sealed class BoxWidgets
{
    public string Execute(
        [Input(Type = "BOUNDING_BOX", Json = """{"x": 0, "y": 0, "width": 512, "height": 512}""")] object region,
        [Input(Type = "BOUNDING_BOXES", Json = "[]", JsonSocket = true)] object boxes,
        [Input(Type = "BOUNDING_BOX", ForceInput = true, Json = """{"x": 0, "y": 0, "width": 512, "height": 512}""", Optional = true)] object? linked = null)
    {
        Calls.Hit(nameof(BoxWidgets));
        return $"{JsonLiteral.ToJson(region)}|{JsonLiteral.ToJson(boxes)}|{region.GetType().Name}|{boxes.GetType().Name}";
    }
}

/// <summary>A MultiType socket, as ComfyUI's bboxes input of Create Bounding Boxes: no widget, but a prompt may give it a value.</summary>
[Node(Category = "test")]
public sealed class MultiTypeSocket
{
    public string Execute([Input(Type = "BOUNDING_BOX,ARRAY,STRING")] object bboxes) => bboxes is string s ? "text:" + s : JsonLiteral.ToJson(bboxes);
}

public class JsonWidgetTests
{
    NodeRegistry _registry = null!;

    [SetUp]
    public void SetUp()
    {
        Calls.Reset();
        _registry = new NodeRegistry().AddAssembly(typeof(JsonWidgetTests).Assembly);
    }

    const string Prompt = """
    {
      "1": { "class_type": "BoxWidgets", "inputs": { "region": {"x": 10, "y": 20, "width": 30.5, "height": 40},
             "boxes": {"__value__": [{"x": 1, "y": 2, "width": 3, "height": 4, "metadata": {"type": "text"}}]} } },
      "2": { "class_type": "StringSink", "inputs": { "text": ["1", 0] } }
    }
    """;

    [Test]
    public void Schema_has_object_valued_widgets_with_their_defaults()
    {
        var schema = _registry.GetSchema("BoxWidgets")!;
        var region = schema.FindInput("region")!;
        Assert.That(region.Type.Name, Is.EqualTo("BOUNDING_BOX"));
        Assert.That(region.Widget!.Kind, Is.EqualTo(WidgetKind.Json));
        Assert.That(JsonLiteral.ToJson(region.Widget.Default), Is.EqualTo("""{"x":0,"y":0,"width":512,"height":512}"""));
        Assert.That(((DictValue)region.Widget.Default!)["width"], Is.EqualTo(512L), "whole numbers are long, as Python ints");
        Assert.That(region.Widget.Socketless, Is.True);
        var boxes = schema.FindInput("boxes")!;
        Assert.That(boxes.Widget!.Default, Is.InstanceOf<ArrayValue>().And.Empty);
        Assert.That(boxes.Widget.Socketless, Is.False, "JsonSocket: ComfyUI's socketless=False");
        var linked = schema.FindInput("linked")!;
        Assert.That(linked.Widget, Is.Null, "a forceInput box is a socket");
        Assert.That(JsonLiteral.ToJson(linked.StatedDefault), Is.EqualTo("""{"x":0,"y":0,"width":512,"height":512}"""));
    }

    [Test]
    public void Prompts_carry_objects_and_wrapped_lists()
    {
        var g = PromptJson.Parse(Prompt);
        var region = (DictValue)g.Nodes["1"].Inputs["region"].Literal!;
        Assert.That(region["x"], Is.EqualTo(10L));
        Assert.That(region["width"], Is.EqualTo(30.5));
        var boxes = (ArrayValue)g.Nodes["1"].Inputs["boxes"].Literal!;
        Assert.That(((DictValue)((DictValue)boxes[0]!)["metadata"]!)["type"], Is.EqualTo("text"));

        var text = PromptJson.ToJson(g);
        Assert.That(text, Does.Contain("""{"__value__":[{"x":1,"y":2,"width":3,"height":4,"metadata":{"type":"text"}}]}"""), "a list is written wrapped, a bare list is a link");
        Assert.That(PromptJson.ToJson(PromptJson.Parse(text)), Is.EqualTo(text));
        Assert.Throws<FormatException>(() => PromptJson.Parse("""{ "1": { "class_type": "A", "inputs": { "x": [1, 2, 3] } } }"""), "a bare list that is not a link");
    }

    [Test]
    public async Task The_node_gets_the_values_and_equal_values_are_served_from_the_cache()
    {
        var executor = new PromptExecutor(_registry);
        var r = await executor.ExecuteAsync(PromptJson.Parse(Prompt));
        Assert.That(r.Outputs["2"].Ui!["text"], Is.EqualTo("""{"x":10,"y":20,"width":30.5,"height":40}|[{"x":1,"y":2,"width":3,"height":4,"metadata":{"type":"text"}}]|DictValue|ArrayValue"""));

        // The same prompt parsed again (new objects with the same content) is the same work.
        r = await executor.ExecuteAsync(PromptJson.Parse(Prompt));
        Assert.That(r.Cached, Is.EqualTo(new[] { "1", "2" }));
        Assert.That(Calls.Of(nameof(BoxWidgets)), Is.EqualTo(1));

        r = await executor.ExecuteAsync(PromptJson.Parse(Prompt.Replace("\"x\": 10", "\"x\": 11")));
        Assert.That(r.Executed, Does.Contain("1"), "a changed box runs the node again");
        Assert.That(Calls.Of(nameof(BoxWidgets)), Is.EqualTo(2));
    }

    [Test]
    public async Task Defaults_reach_the_node_when_the_prompt_gives_nothing()
    {
        var g = new PromptGraph();
        g.Add("1", "BoxWidgets", ("region", _registry.GetSchema("BoxWidgets")!.FindInput("region")!.Widget!.Default), ("boxes", new ArrayValue()));
        g.Add("2", "StringSink", ("text", PromptInput.Link("1", 0)));
        Assert.That(PromptValidator.Validate(_registry, g).Errors, Is.Empty);
        var r = await new PromptExecutor(_registry).ExecuteAsync(g);
        Assert.That(r.Outputs["2"].Ui!["text"], Is.EqualTo("""{"x":0,"y":0,"width":512,"height":512}|[]|DictValue|ArrayValue"""));
    }

    [Test]
    public void A_servers_object_info_brings_its_object_valued_widgets()
    {
        // As ComfyUI (and SpawnGen's server) list Crop Image and Draw BBoxes.
        var info = System.Text.Json.Nodes.JsonNode.Parse("""
        { "ImageCropV2": { "input": { "required": { "image": ["IMAGE", {}],
              "crop_region": ["BOUNDING_BOX", {"default": {"x": 0, "y": 0, "width": 512, "height": 512}, "socketless": true, "component": "ImageCrop"}] } },
            "input_order": { "required": ["image", "crop_region"] }, "output": ["IMAGE"], "output_name": ["IMAGE"], "name": "ImageCropV2" },
          "DrawBBoxes": { "input": { "required": { "bboxes": ["BOUNDING_BOX", {"default": {"x": 0, "y": 0, "width": 512, "height": 512}, "socketless": true, "forceInput": true}] } },
            "output": ["IMAGE"], "name": "DrawBBoxes" } }
        """)!.AsObject();
        var schemas = NodeSchemaJson.FromObjectInfo(info).ToDictionary(s => s.Id);
        var crop = schemas["ImageCropV2"].FindInput("crop_region")!;
        Assert.That(crop.Widget!.Kind, Is.EqualTo(WidgetKind.Json));
        Assert.That(JsonLiteral.ToJson(crop.Widget.Default), Is.EqualTo("""{"x":0,"y":0,"width":512,"height":512}"""));
        Assert.That(crop.Widget.Socketless, Is.True);
        var boxes = schemas["DrawBBoxes"].FindInput("bboxes")!;
        Assert.That(boxes.Widget, Is.Null);
        Assert.That(boxes.ForceInput, Is.True);
        Assert.That(boxes.StatedDefault, Is.InstanceOf<DictValue>());
    }

    [Test]
    public async Task A_multitype_socket_takes_a_value_in_a_prompt_as_ComfyUI_does()
    {
        static PromptGraph With(string value) => PromptJson.Parse($$"""
            { "1": { "class_type": "MultiTypeSocket", "inputs": { "bboxes": {{value}} } }, "2": { "class_type": "StringSink", "inputs": { "text": ["1", 0] } } }
            """);
        var executor = new PromptExecutor(_registry);
        var text = await executor.ExecuteAsync(With("\"[{\\\"x\\\": 1}]\""));
        Assert.That(text.Outputs["2"].Ui!["text"], Is.EqualTo("text:[{\"x\": 1}]"), "a STRING part takes text");
        var dict = await executor.ExecuteAsync(With("""{"x": 1, "y": 2, "width": 3, "height": 4}"""));
        Assert.That(dict.Outputs["2"].Ui!["text"], Is.EqualTo("""{"x":1,"y":2,"width":3,"height":4}"""), "a BOUNDING_BOX part takes an object");
        var list = await executor.ExecuteAsync(With("""{"__value__": [[{"x": 1}]]}"""));
        Assert.That(list.Outputs["2"].Ui!["text"], Is.EqualTo("""[[{"x":1}]]"""));
        var errors = PromptValidator.Validate(_registry, With("5")).Errors;
        Assert.That(errors.Select(e => e.Code), Is.EqualTo(new[] { "bad_value" }), "a number fits no part");
    }

    [Test]
    public void Json_text_round_trips_python_values()
    {
        var v = JsonLiteral.Parse("""{"a": 1, "b": 1.0, "c": 123456789012345678901234567890, "d": [true, null, "s"], "a": 2}""");
        Assert.That(JsonLiteral.ToJson(v), Is.EqualTo("""{"a":2,"b":1.0,"c":123456789012345678901234567890,"d":[true,null,"s"]}"""));
        Assert.That(((DictValue)v!)["c"], Is.InstanceOf<System.Numerics.BigInteger>());
        Assert.That(JsonLiteral.TryParse("{nope", out _), Is.False);
    }
}
