using System.Text.Json.Nodes;
using SpawnGen.Editor;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor.Tests;

/// <summary>
/// ComfyUI lists Create Video's bit_depth options as "auto", 8 and 10 (numbers, read from its object_info, 2026-11-13). Its frontend saves and
/// sends 8 as a number, and its server rejects the text "8". The editor keeps the option as text and writes the number.
/// </summary>
public class NumericComboTests
{
    static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(SpawnGen.Nodes.Core.SaveImage).Assembly).AddAssembly(typeof(SpawnGen.Nodes.ML.KSampler).Assembly);

    [Test]
    public void A_numeric_option_is_saved_and_sent_as_a_number_and_read_back_as_its_text()
    {
        var doc = new EditorDocument(Registry());
        var node = doc.AddNode("CreateVideo", 0, 0);
        doc.SetWidget(node.Id, "bit_depth", "8");
        var saved = JsonNode.Parse(ComfyWorkflow.Save(doc))!;
        var values = saved["nodes"]![0]!["widgets_values"]!.AsArray();
        Assert.That(values.Any(v => v is JsonValue n && n.GetValueKind() == System.Text.Json.JsonValueKind.Number && (long)n == 8), Is.True, values.ToJsonString());
        Assert.That(values.Any(v => v is JsonValue t && t.GetValueKind() == System.Text.Json.JsonValueKind.String && (string?)t == "auto"), Is.False, "auto is not the value");

        var reopened = new EditorDocument(Registry());
        var warnings = ComfyWorkflow.Load(saved.ToJsonString(), reopened);
        Assert.That(warnings, Is.Empty, "a ComfyUI number is a valid option (it used to be reset to the default with a warning)");
        var back = reopened.Graph.Nodes.Single();
        Assert.That(back.Widgets["bit_depth"], Is.EqualTo("8"));

        var prompt = GraphCompiler.Compile(reopened.Graph, reopened.Registry).Graph;
        Assert.That(prompt.Nodes[back.Id.ToString()].Inputs["bit_depth"].Literal, Is.EqualTo(8L));
        doc.SetWidget(node.Id, "bit_depth", "auto");
        Assert.That(GraphCompiler.Compile(doc.Graph, doc.Registry).Graph.Nodes[node.Id.ToString()].Inputs["bit_depth"].Literal, Is.EqualTo("auto"));
    }

    [Test]
    public void An_API_prompt_with_the_number_opens_with_the_option()
    {
        var doc = new EditorDocument(Registry());
        ApiPromptImport.Load("""{ "1": { "class_type": "CreateVideo", "inputs": { "fps": 24.0, "bit_depth": 10 } } }""", doc);
        Assert.That(doc.Graph.Nodes.Single().Widgets["bit_depth"], Is.EqualTo("10"));
    }

    [Test]
    public void Other_combos_stay_text()
    {
        var spec = Registry().GetSchema("KSampler")!.FindInput("sampler_name")!.Widget!;
        Assert.That(spec.NumericOptions, Is.False);
        Assert.That(spec.OptionValue("euler"), Is.EqualTo("euler"));
        var bitDepth = Registry().GetSchema("CreateVideo")!.FindInput("bit_depth")!.Widget!;
        Assert.That((bitDepth.OptionValue("8"), bitDepth.OptionValue("auto")), Is.EqualTo(((object)8L, (object)"auto")));
    }
}
