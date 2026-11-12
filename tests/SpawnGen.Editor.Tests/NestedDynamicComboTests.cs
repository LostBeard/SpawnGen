using System.Text.Json.Nodes;
using SpawnGen.Editor;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor.Tests;

/// <summary>A node shaped like ComfyUI's Save Video: format brings codec, codec brings an encoding mode, re-encode brings crf (nested dynamic combos).</summary>
[Node("NestedComboProbe", Category = "_test")]
public sealed class NestedComboProbe
{
    static DynamicOption Codec(string key, double crf) => new(key, DynamicInput.Nested("encoding",
        [new DynamicOption("auto"), new DynamicOption("re-encode", DynamicInput.Float("crf", crf, 0, 63, 1))]));

    public static IEnumerable<DynamicOption> Formats() =>
    [
        new("auto", DynamicInput.Nested("codec", [new DynamicOption("auto"), Codec("h264", 18), Codec("av1", 24)])),
        new("mp4", DynamicInput.Nested("codec", [new DynamicOption("auto"), Codec("h264", 18), Codec("av1", 24)])),
        new("webm", DynamicInput.Nested("codec", [new DynamicOption("auto"), Codec("av1", 24)])),
    ];

    public string Execute(string filenamePrefix = "video/ComfyUI", [DynamicCombo(nameof(Formats))] DynamicChoice? format = null)
    {
        var codec = format!.Get<DynamicChoice>("codec");
        var encoding = codec?.Get<DynamicChoice>("encoding");
        return $"{format.Key}/{codec?.Key}/{encoding?.Key}/{encoding?.Get<double>("crf", -1)}";
    }
}

public class NestedDynamicComboTests
{
    static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(SpawnGen.Nodes.Core.SaveImage).Assembly).AddAssembly(typeof(NestedComboProbe).Assembly);

    static List<string> Values(EditorDocument doc, int id) =>
        JsonNode.Parse(ComfyWorkflow.Save(doc))!["nodes"]!.AsArray().First(n => (int)n!["id"]! == id)!["widgets_values"]!.AsArray().Select(v => v!.ToJsonString()).ToList();

    [Test]
    public void Nested_combos_are_inputs_of_their_parents_option_with_the_union_of_their_options()
    {
        var s = Registry().GetSchema("NestedComboProbe")!;
        Assert.That(s.Inputs.Select(i => i.Name), Is.EqualTo(new[]
            { "filename_prefix", "format", "format.codec", "format.codec.encoding", "format.codec.encoding.crf" }));
        Assert.That(s.FindInput("format.codec")!.Widget!.ResolveOptions(), Is.EqualTo(new[] { "auto", "h264", "av1" }), "webm's codec has no h264; the input offers every format's");
        Assert.That(s.FindInput("format.codec.encoding")!.DynamicParent, Is.EqualTo("format.codec"));
        Assert.That(s.FindInput("format.codec.encoding")!.DynamicOptions, Is.EqualTo(new[] { "h264", "av1" }));
        Assert.That(s.FindInput("format.codec.encoding.crf")!.DynamicOptions, Is.EqualTo(new[] { "re-encode" }));
    }

    [Test]
    public void The_widgets_show_and_save_depth_first_as_ComfyUIs_template_stores_them()
    {
        var doc = new EditorDocument(Registry());
        var node = doc.AddNode("NestedComboProbe", 0, 0);
        Assert.That(Values(doc, node.Id), Is.EqualTo(new[] { "\"video/ComfyUI\"", "\"auto\"", "\"auto\"" }), "ComfyUI's Wan template saves Save Video as these three");
        doc.SetWidget(node.Id, "format", "webm");
        doc.SetWidget(node.Id, "format.codec", "av1");
        doc.SetWidget(node.Id, "format.codec.encoding", "re-encode");
        doc.SetWidget(node.Id, "format.codec.encoding.crf", 30.0);
        Assert.That(Values(doc, node.Id), Is.EqualTo(new[] { "\"video/ComfyUI\"", "\"webm\"", "\"av1\"", "\"re-encode\"", "30" }));
        Assert.That(NodeLayout.VisibleInputs(doc.Registry.GetSchema("NestedComboProbe")!, node, doc.Graph).Select(i => i.Name),
            Is.EqualTo(new[] { "filename_prefix", "format", "format.codec", "format.codec.encoding", "format.codec.encoding.crf" }));
        doc.SetWidget(node.Id, "format.codec", "auto");
        Assert.That(Values(doc, node.Id), Is.EqualTo(new[] { "\"video/ComfyUI\"", "\"webm\"", "\"auto\"" }), "the encoding mode goes with av1");
    }

    [Test]
    public async Task A_run_receives_the_nested_choices()
    {
        var registry = Registry();
        var g = new PromptGraph();
        g.Add("1", "NestedComboProbe", ("filename_prefix", "x"), ("format", "mp4"), ("format.codec", "av1"), ("format.codec.encoding", "re-encode"),
            ("format.codec.encoding.crf", 30.0));
        g.Add("2", "PreviewAny", ("source", PromptInput.Link("1")));
        var result = await new PromptExecutor(registry, new Services()).ExecuteAsync(g);
        Assert.That(((object[])result.Outputs["2"].Ui!["text"]!)[0], Is.EqualTo("mp4/av1/re-encode/30"));

        var plain = new PromptGraph();
        plain.Add("1", "NestedComboProbe", ("filename_prefix", "x"), ("format", "auto"), ("format.codec", "auto"));
        plain.Add("2", "PreviewAny", ("source", PromptInput.Link("1")));
        var defaults = await new PromptExecutor(registry, new Services()).ExecuteAsync(plain);
        Assert.That(((object[])defaults.Outputs["2"].Ui!["text"]!)[0], Is.EqualTo("auto/auto//"), "codec auto brings no encoding mode");
    }

    sealed class Services : IServiceProvider
    {
        public object? GetService(Type serviceType) => null;
    }
}
