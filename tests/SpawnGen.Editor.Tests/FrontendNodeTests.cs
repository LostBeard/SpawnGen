using SpawnGen.Editor;
using SpawnGen.Engine.Execution;

namespace SpawnGen.Editor.Tests;

/// <summary>Reroute and Note exist only in the editor; PrimitiveNode from ComfyUI files is folded into the widget it fed.</summary>
public class FrontendNodeTests
{
    [Test]
    public void A_reroute_chain_is_compiled_away_and_the_consumer_links_to_the_real_source()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        var r1 = doc.AddNode("Reroute", 600, 700);
        var r2 = doc.AddNode("Reroute", 700, 700);
        // The sampler's latent_image now comes from the empty latent through two reroutes.
        Assert.That(doc.Connect(ids["latent"], 0, r1.Id, "value", out var e1), Is.Not.Null, e1);
        Assert.That(doc.Connect(r1.Id, 0, r2.Id, "value", out var e2), Is.Not.Null, e2);
        Assert.That(doc.Connect(r2.Id, 0, ids["sampler"], "latent_image", out var e3), Is.Not.Null, e3);

        var compiled = GraphCompiler.Compile(doc.Graph, doc.Registry);
        Assert.That(compiled.Graph.Nodes.Keys, Does.Not.Contain(r1.Id.ToString()).And.Not.Contain(r2.Id.ToString()));
        var link = compiled.Graph.Nodes[ids["sampler"].ToString()].Inputs["latent_image"];
        Assert.That(link.IsLink && link.NodeId == ids["latent"].ToString(), Is.True, "the sampler reads the latent directly");
        Assert.That(compiled.Notes, Is.Empty);
        Assert.That(new PromptExecutor(doc.Registry).Validate(compiled.Graph).Errors, Is.Empty);
    }

    [Test]
    public void A_reroute_with_nothing_connected_drops_what_reads_it_and_says_so()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        var r = doc.AddNode("Reroute", 600, 700);
        Assert.That(doc.Connect(r.Id, 0, ids["sampler"], "latent_image", out var err), Is.Not.Null, err);
        var compiled = GraphCompiler.Compile(doc.Graph, doc.Registry);
        Assert.That(compiled.Graph.Nodes.Keys, Does.Not.Contain(ids["sampler"].ToString()));
        Assert.That(compiled.Notes, Has.Some.Contains("latent_image"));
    }

    [Test]
    public void A_note_is_never_sent_and_does_not_make_noise()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        doc.AddNode("Note", 0, 600);
        var compiled = GraphCompiler.Compile(doc.Graph, doc.Registry);
        Assert.That(compiled.Graph.Nodes, Has.Count.EqualTo(7));
        Assert.That(compiled.Notes, Is.Empty);
    }

    [Test]
    public void A_reroute_is_a_small_body_with_two_dots()
    {
        var (doc, _) = Fixtures.Txt2Img();
        var r = doc.AddNode("Reroute", 100, 200);
        var geo = NodeLayout.Compute(r, doc.Registry.GetSchema("Reroute")!, doc.Graph);
        Assert.That((geo.Bounds.Width, geo.Bounds.Height), Is.EqualTo((NodeLayout.RerouteWidth, NodeLayout.RerouteHeight)));
        Assert.That(geo.TitleBar.Height, Is.EqualTo(0));
        Assert.That(geo.Inputs.Single().Position.X, Is.EqualTo(100));
        Assert.That(geo.Outputs.Single().Position.X, Is.EqualTo(100 + NodeLayout.RerouteWidth));
        Assert.That(geo.Widgets, Is.Empty);
    }

    [Test]
    public void ComfyUIs_reroute_json_with_an_unnamed_input_loads_with_its_link()
    {
        const string json = """
        {"last_node_id": 3, "last_link_id": 2, "version": 0.4,
         "nodes": [
           {"id": 1, "type": "EmptyLatentImage", "pos": [0, 0], "size": [315, 106], "mode": 0, "order": 0,
            "outputs": [{"name": "LATENT", "type": "LATENT", "links": [1]}], "widgets_values": [512, 512, 1]},
           {"id": 2, "type": "Reroute", "pos": [400, 0], "size": [75, 26], "mode": 0, "order": 1,
            "inputs": [{"name": "", "type": "*", "link": 1}], "outputs": [{"name": "", "type": "LATENT", "links": [2]}]},
           {"id": 3, "type": "LatentFlip", "pos": [600, 0], "size": [315, 60], "mode": 0, "order": 2,
            "inputs": [{"name": "samples", "type": "LATENT", "link": 2}], "outputs": [{"name": "LATENT", "type": "LATENT", "links": []}], "widgets_values": ["x-axis: vertically"]}
         ],
         "links": [[1, 1, 0, 2, 0, "LATENT"], [2, 2, 0, 3, 0, "LATENT"]], "groups": [], "config": {}, "extra": {}}
        """;
        var doc = new EditorDocument(Fixtures.Registry());
        var warnings = ComfyWorkflow.Load(json, doc);
        Assert.That(warnings, Is.Empty, string.Join("; ", warnings));
        Assert.That(doc.Graph.Links, Has.Count.EqualTo(2));
        var compiled = GraphCompiler.Compile(doc.Graph, doc.Registry);
        Assert.That(compiled.Graph.Nodes["3"].Inputs["samples"].NodeId, Is.EqualTo("1"));
    }

    [Test]
    public void A_legacy_PrimitiveNode_is_folded_into_the_widget_it_fed()
    {
        const string json = """
        {"last_node_id": 2, "last_link_id": 1, "version": 0.4,
         "nodes": [
           {"id": 1, "type": "PrimitiveNode", "pos": [0, 0], "size": [210, 82], "mode": 0, "order": 0,
            "outputs": [{"name": "INT", "type": "INT", "links": [1], "widget": {"name": "width"}}], "widgets_values": [768, "fixed"]},
           {"id": 2, "type": "EmptyLatentImage", "pos": [300, 0], "size": [315, 106], "mode": 0, "order": 1,
            "inputs": [{"name": "width", "type": "INT", "link": 1, "widget": {"name": "width"}}],
            "outputs": [{"name": "LATENT", "type": "LATENT", "links": []}], "widgets_values": [512, 512, 1]}
         ],
         "links": [[1, 1, 0, 2, 0, "INT"]], "groups": [], "config": {}, "extra": {}}
        """;
        var doc = new EditorDocument(Fixtures.Registry());
        var warnings = ComfyWorkflow.Load(json, doc);
        Assert.That(doc.Graph.Nodes.Select(n => n.ClassType), Is.EqualTo(new[] { "EmptyLatentImage" }));
        Assert.That(doc.Graph.Links, Is.Empty);
        Assert.That(doc.Graph.GetNode(2).Widgets["width"], Is.EqualTo(768L));
        Assert.That(warnings, Has.Some.Contains("Primitive node #1"));
    }
}
