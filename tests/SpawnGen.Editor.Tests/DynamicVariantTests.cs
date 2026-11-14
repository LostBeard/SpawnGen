using System.Text.Json.Nodes;
using SpawnGen.Editor;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor.Tests;

/// <summary>
/// Save Image (Advanced)'s formats all bring "bit_depth" and "input_color_space", each with its own choices (ComfyUI's object_info, 2026-11-13):
/// png 8-bit / 16-bit, exr 32-bit float / 16-bit float (default 16-bit float), avif auto / 8-bit YUV420 / 10-bit YUV420. The editor shows, keeps
/// and sends the chosen format's.
/// </summary>
public class DynamicVariantTests
{
    static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(SpawnGen.Nodes.Core.SaveImage).Assembly);

    static JsonArray Widgets(EditorDocument doc, int id) =>
        JsonNode.Parse(ComfyWorkflow.Save(doc))!["nodes"]!.AsArray().First(n => (int)n!["id"]! == id)!["widgets_values"]!.AsArray();

    [Test]
    public void Switching_the_format_gives_its_own_choices_and_default_in_one_undo_step()
    {
        var doc = new EditorDocument(Registry());
        var node = doc.AddNode("SaveImageAdvanced", 0, 0);
        var schema = doc.Registry.GetSchema("SaveImageAdvanced")!;
        Assert.That(Widgets(doc, node.Id).ToJsonString(), Is.EqualTo("""["ComfyUI","png","8-bit","sRGB"]"""));

        doc.SetWidget(node.Id, "format", "exr");
        Assert.That(Widgets(doc, node.Id).ToJsonString(), Is.EqualTo("""["ComfyUI","exr","16-bit float","sRGB"]"""), "8-bit is not an exr choice: exr's default");
        var shown = NodeLayout.OrderedInputs(schema, node.Widgets).Single(i => i.Name == "format.bit_depth");
        Assert.That(shown.Widget!.ResolveOptions(), Is.EqualTo(new[] { "32-bit float", "16-bit float" }));
        Assert.That(NodeLayout.OrderedInputs(schema, node.Widgets).Single(i => i.Name == "format.input_color_space").Widget!.ResolveOptions(),
            Is.EqualTo(new[] { "sRGB", "HDR", "linear" }));

        doc.SetWidget(node.Id, "format.bit_depth", "32-bit float");
        doc.SetWidget(node.Id, "format.input_color_space", "linear");
        var prompt = GraphCompiler.Compile(doc.Graph, doc.Registry).Graph.Nodes[node.Id.ToString()].Inputs;
        Assert.That((prompt["format"].Literal, prompt["format.bit_depth"].Literal, prompt["format.input_color_space"].Literal),
            Is.EqualTo(((object?)"exr", (object?)"32-bit float", (object?)"linear")));

        doc.SetWidget(node.Id, "format", "png");
        Assert.That(Widgets(doc, node.Id).ToJsonString(), Is.EqualTo("""["ComfyUI","png","8-bit","sRGB"]"""), "linear is not a png choice either");
        doc.Undo();
        Assert.That(Widgets(doc, node.Id).ToJsonString(), Is.EqualTo("""["ComfyUI","exr","32-bit float","linear"]"""), "one step back to exr with its values");
    }

    [Test]
    public void Option_inputs_are_labelled_with_their_own_part_as_in_ComfyUI()
    {
        // ComfyUI's frontend (2026-11-13): widget "format.bit_depth" is labelled "bit_depth"; Save Video's "format.codec" is "codec".
        var doc = new EditorDocument(Registry());
        var node = doc.AddNode("SaveImageAdvanced", 0, 0);
        var rows = NodeLayout.Compute(node, doc.Registry.GetSchema("SaveImageAdvanced")!, doc.Graph).Widgets;
        Assert.That(rows.Select(r => (r.Name, r.Label)), Is.EqualTo(new[]
        {
            ("filename_prefix", "filename_prefix"), ("format", "format"), ("format.bit_depth", "bit_depth"), ("format.input_color_space", "input_color_space"),
        }));
    }

    [Test]
    public void A_saved_exr_node_opens_with_exrs_values()
    {
        var doc = new EditorDocument(Registry());
        var node = doc.AddNode("SaveImageAdvanced", 0, 0);
        var json = JsonNode.Parse(ComfyWorkflow.Save(doc))!;
        json["nodes"]![0]!["widgets_values"] = new JsonArray("shot", "avif", "10-bit YUV420", "HDR PQ", 30, "animated", 12.0, 2);
        var reopened = new EditorDocument(Registry());
        Assert.That(ComfyWorkflow.Load(json.ToJsonString(), reopened), Is.Empty);
        var n = reopened.Graph.Nodes.Single();
        Assert.That((n.Widgets["format.bit_depth"], n.Widgets["format.input_color_space"], n.Widgets["format.crf"], n.Widgets["format.save_mode.fps"]),
            Is.EqualTo(((object?)"10-bit YUV420", (object?)"HDR PQ", (object?)30L, (object?)12.0)));
        Assert.That(Widgets(reopened, n.Id).ToJsonString(), Is.EqualTo("""["shot","avif","10-bit YUV420","HDR PQ",30,"animated",12,2]"""));
    }
}
