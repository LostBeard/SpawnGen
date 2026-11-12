using System.Text.Json.Nodes;
using SpawnGen.Editor;

namespace SpawnGen.Editor.Tests;

/// <summary>
/// Node templates in ComfyUI's comfy.templates.json format. Fixtures/comfy.templates.json is the file ComfyUI's frontend wrote for "Save Selected
/// as Template" on its default graph's VAE Decode and Save Image (a black-box run of it).
/// </summary>
public class NodeTemplatesTests
{
    static string Fixture => File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "comfy.templates.json"));

    [Test]
    public void A_template_ComfyUI_saved_opens_and_inserts_its_nodes_with_the_links_among_them()
    {
        var templates = NodeTemplates.Parse(Fixture);
        Assert.That(templates.Select(t => t.Name), Is.EqualTo(new[] { "My pair" }));
        var part = NodeTemplates.Open(templates[0].Data, Fixtures.Registry(), out var warnings);
        Assert.That(part.Graph.Nodes.Select(n => n.ClassType).Order(), Is.EqualTo(new[] { "SaveImage", "VAEDecode" }));
        Assert.That(part.Graph.Links, Has.Count.EqualTo(1), "the links from nodes outside the template are left out");

        var (doc, _) = Fixtures.Txt2Img();
        var view = new EditorView(doc);
        int nodes = doc.Graph.Nodes.Count, links = doc.Graph.Links.Count;
        view.PasteNodes([.. part.Graph.Nodes], part.Graph.Links, new Point(2000, 800));
        Assert.That((doc.Graph.Nodes.Count, doc.Graph.Links.Count), Is.EqualTo((nodes + 2, links + 1)), "as ComfyUI inserts it (10 -> 12 nodes, 10 -> 11 links)");
        var save = doc.Graph.GetNode(view.SelectedNodes.Single(id => doc.Graph.GetNode(id).ClassType == "SaveImage"));
        Assert.That(save.Widgets["filename_prefix"], Is.EqualTo("ComfyUI"));
        Assert.That(doc.Graph.LinkInto(save.Id, "images")!.FromNode, Is.EqualTo(view.SelectedNodes.Single(id => id != save.Id)));
        Assert.That(view.SelectedNodes.Min(id => doc.Graph.GetNode(id).X), Is.EqualTo(2000));
        doc.Undo();
        Assert.That(doc.Graph.Nodes.Count, Is.EqualTo(nodes), "one undo step");
    }

    [Test]
    public void Saving_writes_ComfyUIs_shape_and_reads_back()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        doc.SetWidget(ids["pos"], "text", "a red apple");
        var data = NodeTemplates.Build(doc, [ids["pos"], ids["sampler"], 9999]);
        var t = JsonNode.Parse(data)!.AsObject();
        Assert.That(t.Select(kv => kv.Key), Is.EqualTo(new[] { "nodes", "groups", "reroutes", "links", "subgraphs" }));
        Assert.That(t["nodes"]!.AsArray().Select(n => (int)n!["id"]!).Order(), Is.EqualTo(new[] { ids["pos"], ids["sampler"] }.Order()), "ids kept, missing ones ignored");
        var link = t["links"]!.AsArray().Single()!.AsObject();
        Assert.That(((int)link["origin_id"]!, (int)link["target_id"]!, (int)link["target_slot"]!, (string)link["type"]!), Is.EqualTo((ids["pos"], ids["sampler"], 1, "CONDITIONING")));

        var file = NodeTemplates.Serialize([new NodeTemplate("Prompt and sampler", data), new NodeTemplate("Other", data)]);
        var back = NodeTemplates.Parse(file);
        Assert.That(back.Select(b => b.Name), Is.EqualTo(new[] { "Prompt and sampler", "Other" }));
        var part = NodeTemplates.Open(back[0].Data, doc.Registry, out _);
        Assert.That(part.Graph.Nodes.Single(n => n.ClassType == "CLIPTextEncode").Widgets["text"], Is.EqualTo("a red apple"));
        Assert.That(part.Graph.Links, Has.Count.EqualTo(1));
    }

    [Test]
    public void A_template_with_a_subgraph_carries_its_definition()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        var instance = doc.ConvertToSubgraph([ids["pos"]], "Prompt", out var error)!;
        Assert.That(error, Is.Null);
        var data = NodeTemplates.Build(doc, [instance.Id]);
        Assert.That(JsonNode.Parse(data)!["subgraphs"]!.AsArray(), Has.Count.EqualTo(1));
        var part = NodeTemplates.Open(data, doc.Registry, out _);
        Assert.That(SubgraphLibrary.Get(part.Graph.Nodes.Single().ClassType)?.Name, Is.EqualTo("Prompt"));
    }

    [Test]
    public void Damaged_files_and_data_are_handled()
    {
        Assert.That(NodeTemplates.Parse(null), Is.Empty);
        Assert.That(NodeTemplates.Parse("{not json"), Is.Empty);
        Assert.That(NodeTemplates.Parse("""[{"name": "ok", "data": "{}"}, {"name": 5}, "x", {"data": "{}"}]""").Select(t => t.Name), Is.EqualTo(new[] { "ok" }));
        Assert.Throws<InvalidDataException>(() => NodeTemplates.Open("{}", Fixtures.Registry(), out _));
        Assert.Catch(() => NodeTemplates.Open("not json", Fixtures.Registry(), out _));
    }
}
