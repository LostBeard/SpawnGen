using System.Text.Json.Nodes;
using SpawnGen.Editor;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor.Tests;

public class NodePropertiesTests
{
    static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(SpawnGen.Nodes.Core.SaveImage).Assembly).AddAssembly(typeof(SpawnGen.Nodes.ML.KSampler).Assembly);

    static JsonObject SavedNode(EditorDocument doc, int id) =>
        JsonNode.Parse(ComfyWorkflow.Save(doc))!["nodes"]!.AsArray().First(n => (int)n!["id"]! == id)!.AsObject();

    [Test]
    public void A_new_node_is_saved_with_its_type_as_the_search_and_replace_name()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        Assert.That((string?)SavedNode(doc, ids["sampler"])["properties"]![ComfyWorkflow.NameForSearchReplace], Is.EqualTo("KSampler"));
    }

    [Test]
    public void Properties_ComfyUI_wrote_survive_open_and_save_and_can_be_edited_with_undo()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        var json = JsonNode.Parse(ComfyWorkflow.Save(doc))!;
        var saved = json["nodes"]!.AsArray().First(n => (int)n!["id"]! == ids["sampler"])!;
        saved["properties"] = new JsonObject { ["Node name for S&R"] = "Sampler A", ["cnr_id"] = "comfy-core", ["ver"] = "0.3.40", ["widget_ue_connectable"] = new JsonObject() };

        var reopened = new EditorDocument(Registry());
        ComfyWorkflow.Load(json.ToJsonString(), reopened);
        var node = reopened.Graph.GetNode(ids["sampler"]);
        Assert.That((string?)node.Properties["cnr_id"], Is.EqualTo("comfy-core"));

        reopened.SetNodeProperty(node.Id, ComfyWorkflow.NameForSearchReplace, "Sampler B");
        var again = SavedNode(reopened, node.Id)["properties"]!.AsObject();
        Assert.That((string?)again[ComfyWorkflow.NameForSearchReplace], Is.EqualTo("Sampler B"));
        Assert.That((string?)again["ver"], Is.EqualTo("0.3.40"));
        Assert.That(again["widget_ue_connectable"], Is.InstanceOf<JsonObject>());

        reopened.Undo();
        Assert.That((string?)node.Properties[ComfyWorkflow.NameForSearchReplace], Is.EqualTo("Sampler A"));
        reopened.Redo();
        Assert.That(GraphCompiler.SearchReplaceName(node), Is.EqualTo("Sampler B"));
    }

    [Test]
    public void Copies_do_not_share_their_properties()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        var node = doc.Graph.GetNode(ids["sampler"]);
        node.Properties["x"] = "1";
        var copy = node.Clone(99);
        copy.Properties["x"] = "2";
        Assert.That((string?)node.Properties["x"], Is.EqualTo("1"));
    }

    [Test]
    public void File_name_prefixes_take_dates_and_widget_values_like_ComfyUI()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        doc.SetWidget(ids["sampler"], "seed", 1234L);
        doc.SetWidget(ids["sampler"], "cfg", 1.5);
        doc.SetWidget(ids["pos"], "text", "a/b\\c");
        doc.SetTitle(ids["pos"], "Positive");
        var g = doc.Graph;
        var at = new DateTime(2026, 3, 7, 9, 5, 2);

        Assert.That(GraphCompiler.ApplyTextReplacements(g, "%date:yyyy-MM-dd%/img_%date:hhmmss%_%date:yy%", at), Is.EqualTo("2026-03-07/img_090502_26"));
        Assert.That(GraphCompiler.ApplyTextReplacements(g, "s%KSampler.seed%_c%KSampler.cfg%", at), Is.EqualTo("s1234_c1.5"));
        Assert.That(GraphCompiler.ApplyTextReplacements(g, "%Positive.text%", at), Is.EqualTo("a_b_c"), "found by title; slashes become underscores");
        Assert.That(GraphCompiler.ApplyTextReplacements(g, "%width%x%height%_%Nobody.seed%_%KSampler.nope%", at), Is.EqualTo("%width%x%height%_%Nobody.seed%_%KSampler.nope%"));

        doc.SetNodeProperty(ids["sampler"], ComfyWorkflow.NameForSearchReplace, "Main");
        Assert.That(GraphCompiler.ApplyTextReplacements(g, "%Main.seed%|%KSampler.seed%", at), Is.EqualTo("1234|%KSampler.seed%"), "a renamed node answers to its new name only");
    }

    [Test]
    public void The_compiled_prompt_has_the_filled_in_prefix_and_the_editor_keeps_the_pattern()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        doc.SetWidget(ids["sampler"], "seed", 77L);
        doc.SetWidget(ids["save"], "filename_prefix", "%date:yyyyMMdd%/seed_%KSampler.seed%");
        var compiled = GraphCompiler.Compile(doc.Graph, doc.Registry, new DateTime(2026, 10, 23, 1, 2, 3));
        Assert.That(compiled.Graph.Nodes[ids["save"].ToString()].Inputs["filename_prefix"].Literal, Is.EqualTo("20261023/seed_77"));
        Assert.That(doc.Graph.GetNode(ids["save"]).Widgets["filename_prefix"], Is.EqualTo("%date:yyyyMMdd%/seed_%KSampler.seed%"));
    }
}
