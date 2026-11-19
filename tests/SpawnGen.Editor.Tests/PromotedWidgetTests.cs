using System.Text.Json.Nodes;
using SpawnGen.Editor;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor.Tests;

/// <summary>
/// ComfyUI's promoted widgets: a subgraph input that feeds an inner node's widget shows on the subgraph node as that widget (its templates save
/// them as properties.proxyWidgets, with the values in the inner nodes). Fixture: Comfy-Org's video_wan2_2_14B_t2v template (MIT, the
/// workflow_templates commit of template-needs.json), the one TJ opened on 2026-11-19 to a "Text to Video (Wan2.2)" node whose inputs were
/// connected to nothing.
/// </summary>
public class PromotedWidgetTests
{
    static NodeRegistry Registry()
    {
        var registry = new NodeRegistry().AddAssembly(typeof(SpawnGen.Nodes.Core.SaveImage).Assembly).AddAssembly(typeof(SpawnGen.Nodes.ML.KSampler).Assembly);
        SubgraphLibrary.Install(registry);
        return registry;
    }

    static string Template() => File.ReadAllText(Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "template_video_wan2_2_14B_t2v.json"));

    const string Prompt = "Beautiful young European woman with honey blonde hair";

    static (EditorDocument Doc, EditorNode Instance) Open(string json)
    {
        var doc = new EditorDocument(Registry());
        ComfyWorkflow.Load(json, doc);
        return (doc, doc.Graph.Nodes.Single(n => n.ClassType.StartsWith(SubgraphDefinition.Prefix, StringComparison.Ordinal)));
    }

    [Test]
    public void The_template_s_subgraph_node_shows_its_promoted_widgets_with_the_inner_values()
    {
        var (doc, instance) = Open(Template());
        var schema = doc.Registry.GetSchema(instance.ClassType)!;
        var widgets = schema.Inputs.Where(i => i.Widget is not null).Select(i => i.Name).ToList();
        Assert.That(widgets, Is.EquivalentTo(new[] { "text", "width", "height", "value", "unet_name", "lora_name", "unet_name_1", "lora_name_1", "clip_name", "vae_name", "value_1" }),
            "every proxyWidgets entry of the template shows as a widget");
        Assert.That((string)instance.Widgets["text"]!, Does.StartWith(Prompt), "the prompt is the inner CLIP Text Encode's");
        Assert.That(Convert.ToInt64(instance.Widgets["width"]), Is.EqualTo(640));
        Assert.That(instance.Widgets["unet_name"], Is.EqualTo("wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"));
        Assert.That(instance.Widgets["unet_name_1"], Is.EqualTo("wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"));
        // The template's labels, as ComfyUI shows them.
        var labels = schema.Inputs.ToDictionary(i => i.Name, i => i.Label);
        Assert.That((labels["value"], labels["unet_name"], labels["value_1"], labels["text"]), Is.EqualTo(("duration", "high_noise_model", "enable_turbo_mode", (string?)null)));
    }

    [Test]
    public void An_edited_promoted_widget_runs_and_saves_as_ComfyUI_saves_it()
    {
        var (doc, instance) = Open(Template());
        doc.SetWidget(instance.Id, "text", "a red fox in the snow");
        doc.SetWidget(instance.Id, "width", 832L);

        // Run: the inner nodes get the instance's values.
        var prompt = GraphCompiler.Compile(doc.Graph, doc.Registry).Graph;
        var texts = prompt.Nodes.Values.Where(n => n.ClassType == "CLIPTextEncode").Select(n => n.Inputs["text"].Literal as string).ToList();
        Assert.That(texts, Does.Contain("a red fox in the snow"), "the positive prompt (the negative one is not promoted)");
        Assert.That(texts.Any(t => t is not null && t.StartsWith(Prompt)), Is.False, "the template's prompt is replaced");
        var latent = prompt.Nodes.Values.Single(n => n.ClassType == "EmptyHunyuanLatentVideo");
        Assert.That(Convert.ToInt64(latent.Inputs["width"].Literal), Is.EqualTo(832));

        // Save: the instance lists every input and keeps no widgets_values; the values are in the definition's inner nodes.
        var saved = JsonNode.Parse(ComfyWorkflow.Save(doc))!;
        var node = saved["nodes"]!.AsArray().Single(n => (int)n!["id"]! == instance.Id)!;
        Assert.That(node["widgets_values"]!.AsArray(), Is.Empty);
        Assert.That(node["inputs"]!.AsArray().Count, Is.EqualTo(11));
        Assert.That(node["properties"]!["proxyWidgets"], Is.Not.Null, "proxyWidgets is kept");
        var inner = saved["definitions"]!["subgraphs"]![0]!["nodes"]!.AsArray().Single(n => (int)n!["id"]! == 89)!;
        Assert.That((string?)inner["widgets_values"]![0], Is.EqualTo("a red fox in the snow"));
        Assert.That((string?)saved["definitions"]!["subgraphs"]![0]!["inputs"]!.AsArray().Single(i => (string?)i!["name"] == "value")!["label"], Is.EqualTo("duration"));

        // And it opens again with the edited values.
        var (_, reopened) = Open(saved.ToJsonString());
        Assert.That(reopened.Widgets["text"], Is.EqualTo("a red fox in the snow"));
        Assert.That(Convert.ToInt64(reopened.Widgets["width"]), Is.EqualTo(832));
    }
}
