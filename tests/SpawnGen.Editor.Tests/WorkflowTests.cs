using System.Text.Json.Nodes;
using SpawnGen.Editor;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;

namespace SpawnGen.Editor.Tests;

/// <summary>ComfyUI's stock text-to-image workflow, as ComfyUI saves it (old-style object sizes included).</summary>
static class ComfySamples
{
    public const string DefaultWorkflow = """
    {
      "last_node_id": 9, "last_link_id": 9,
      "nodes": [
        {"id": 7, "type": "CLIPTextEncode", "pos": [413, 389], "size": {"0": 425.27801513671875, "1": 180.6060791015625}, "flags": {}, "order": 3, "mode": 0,
         "inputs": [{"name": "clip", "type": "CLIP", "link": 5}],
         "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [6], "slot_index": 0}],
         "properties": {}, "widgets_values": ["text, watermark"]},
        {"id": 6, "type": "CLIPTextEncode", "pos": [415, 186], "size": {"0": 422.84503173828125, "1": 164.31304931640625}, "flags": {}, "order": 2, "mode": 0,
         "inputs": [{"name": "clip", "type": "CLIP", "link": 3}],
         "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [4], "slot_index": 0}],
         "properties": {}, "widgets_values": ["beautiful scenery nature glass bottle landscape, , purple galaxy bottle,"]},
        {"id": 5, "type": "EmptyLatentImage", "pos": [473, 609], "size": {"0": 315, "1": 106}, "flags": {}, "order": 1, "mode": 0,
         "outputs": [{"name": "LATENT", "type": "LATENT", "links": [2], "slot_index": 0}],
         "properties": {}, "widgets_values": [512, 512, 1]},
        {"id": 3, "type": "KSampler", "pos": [863, 186], "size": {"0": 315, "1": 262}, "flags": {}, "order": 4, "mode": 0,
         "inputs": [{"name": "model", "type": "MODEL", "link": 1}, {"name": "positive", "type": "CONDITIONING", "link": 4},
                    {"name": "negative", "type": "CONDITIONING", "link": 6}, {"name": "latent_image", "type": "LATENT", "link": 2}],
         "outputs": [{"name": "LATENT", "type": "LATENT", "links": [7], "slot_index": 0}],
         "properties": {}, "widgets_values": [156680208700286, "randomize", 20, 8, "euler", "normal", 1]},
        {"id": 8, "type": "VAEDecode", "pos": [1209, 188], "size": {"0": 210, "1": 46}, "flags": {}, "order": 5, "mode": 0,
         "inputs": [{"name": "samples", "type": "LATENT", "link": 7}, {"name": "vae", "type": "VAE", "link": 8}],
         "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [9], "slot_index": 0}], "properties": {}},
        {"id": 9, "type": "SaveImage", "pos": [1451, 189], "size": {"0": 210, "1": 58}, "flags": {}, "order": 6, "mode": 0,
         "inputs": [{"name": "images", "type": "IMAGE", "link": 9}], "properties": {}, "widgets_values": ["ComfyUI"]},
        {"id": 4, "type": "CheckpointLoaderSimple", "pos": [26, 474], "size": {"0": 315, "1": 98}, "flags": {}, "order": 0, "mode": 0,
         "outputs": [{"name": "MODEL", "type": "MODEL", "links": [1], "slot_index": 0}, {"name": "CLIP", "type": "CLIP", "links": [3, 5], "slot_index": 1},
                     {"name": "VAE", "type": "VAE", "links": [8], "slot_index": 2}],
         "properties": {}, "widgets_values": ["v1-5-pruned-emaonly.ckpt"]}
      ],
      "links": [[1, 4, 0, 3, 0, "MODEL"], [2, 5, 0, 3, 3, "LATENT"], [3, 4, 1, 6, 0, "CLIP"], [4, 6, 0, 3, 1, "CONDITIONING"], [5, 4, 1, 7, 0, "CLIP"],
                [6, 7, 0, 3, 2, "CONDITIONING"], [7, 3, 0, 8, 0, "LATENT"], [8, 4, 2, 8, 1, "VAE"], [9, 8, 0, 9, 0, "IMAGE"]],
      "groups": [{"title": "Group", "bounding": [10, 10, 300, 200], "color": "#3f789e"}],
      "config": {}, "extra": {"ds": {"scale": 1, "offset": [0, 0]}}, "version": 0.4
    }
    """;
}

public class WorkflowTests
{
    static EditorDocument Open(string json, out IReadOnlyList<string> warnings)
    {
        var doc = new EditorDocument(Fixtures.Registry());
        warnings = ComfyWorkflow.Load(json, doc);
        return doc;
    }

    [Test]
    public void Opens_ComfyUIs_default_workflow_with_ids_positions_links_and_widget_values()
    {
        var doc = Open(ComfySamples.DefaultWorkflow, out var warnings);
        Assert.That(warnings, Is.Empty);
        Assert.That(doc.Graph.Nodes.Count, Is.EqualTo(7));
        Assert.That(doc.Graph.Links.Count, Is.EqualTo(9));

        var sampler = doc.Graph.GetNode(3);
        Assert.That((sampler.ClassType, sampler.X, sampler.Y), Is.EqualTo(("KSampler", 863.0, 186.0)));
        Assert.That(sampler.Width, Is.EqualTo(315));
        Assert.That(sampler.Widgets["seed"], Is.EqualTo(156680208700286L));
        Assert.That(sampler.Widgets["steps"], Is.EqualTo(20L), "The 'randomize' control value after the seed must not shift the widgets.");
        Assert.That(sampler.Widgets["cfg"], Is.EqualTo(8.0));
        Assert.That(sampler.Widgets["sampler_name"], Is.EqualTo("euler"));
        Assert.That(sampler.Widgets["denoise"], Is.EqualTo(1.0));

        Assert.That(doc.Graph.GetNode(5).Widgets["width"], Is.EqualTo(512L));
        Assert.That(doc.Graph.GetNode(6).Widgets["text"], Does.StartWith("beautiful scenery"));
        Assert.That(doc.Graph.GetNode(9).Widgets["filename_prefix"], Is.EqualTo("ComfyUI"));

        var link = doc.Graph.LinkInto(3, "negative")!;
        Assert.That((link.Id, link.FromNode, link.FromOutput), Is.EqualTo((6, 7, 0)));
        Assert.That(doc.Graph.LinkInto(8, "vae")!.FromOutput, Is.EqualTo(2));
        Assert.That(doc.CanUndo, Is.False, "Opening a workflow is not an undo step.");
    }

    [Test]
    public void An_opened_workflow_compiles_to_a_prompt_the_engine_accepts()
    {
        var doc = Open(ComfySamples.DefaultWorkflow, out _);
        var compiled = GraphCompiler.Compile(doc.Graph, doc.Registry);
        Assert.That(compiled.Notes, Is.Empty);
        Assert.That(compiled.Graph.Nodes.Count, Is.EqualTo(7));

        var validation = new PromptExecutor(doc.Registry).Validate(compiled.Graph);
        // The checkpoint name is ComfyUI's, not one of ours: that is the only thing the engine may object to.
        Assert.That(validation.Errors.Select(e => (e.NodeId, e.Input)), Is.EqualTo(new[] { ("4", "ckpt_name") }));
    }

    [Test]
    public void Saving_and_reopening_keeps_everything()
    {
        var first = Open(ComfySamples.DefaultWorkflow, out _);
        var saved = ComfyWorkflow.Save(first);
        var second = Open(saved, out var warnings);

        Assert.That(warnings, Is.Empty);
        Assert.That(second.Graph.Nodes.Select(n => n.Id).Order(), Is.EqualTo(first.Graph.Nodes.Select(n => n.Id).Order()));
        foreach (var n in first.Graph.Nodes)
        {
            var m = second.Graph.GetNode(n.Id);
            Assert.That((m.ClassType, m.X, m.Y, m.Mode, m.Collapsed), Is.EqualTo((n.ClassType, n.X, n.Y, n.Mode, n.Collapsed)), $"node {n.Id}");
            Assert.That(m.Widgets, Is.EquivalentTo(n.Widgets), $"widgets of node {n.Id}");
        }
        Assert.That(second.Graph.Links.Select(l => (l.Id, l.FromNode, l.FromOutput, l.ToNode, l.ToInput)).Order(),
            Is.EqualTo(first.Graph.Links.Select(l => (l.Id, l.FromNode, l.FromOutput, l.ToNode, l.ToInput)).Order()));
    }

    [Test]
    public void Saved_json_follows_the_ComfyUI_format_so_ComfyUI_can_open_it()
    {
        var doc = Open(ComfySamples.DefaultWorkflow, out _);
        var root = JsonNode.Parse(ComfyWorkflow.Save(doc))!.AsObject();

        Assert.That((double)root["version"]!, Is.EqualTo(0.4));
        Assert.That((int)root["last_node_id"]!, Is.EqualTo(9));
        var sampler = root["nodes"]!.AsArray().Single(n => (int)n!["id"]! == 3)!.AsObject();
        // seed, control_after_generate, steps, cfg, sampler_name, scheduler, denoise
        Assert.That(sampler["widgets_values"]!.AsArray().Select(v => v!.ToJsonString()),
            Is.EqualTo(new[] { "156680208700286", "\"fixed\"", "20", "8", "\"euler\"", "\"normal\"", "1" }));
        // Socket inputs in order; widgets are not listed unless linked.
        Assert.That(sampler["inputs"]!.AsArray().Select(i => (string)i!["name"]!), Is.EqualTo(new[] { "model", "positive", "negative", "latent_image" }));
        // A link's last-but-one number is the slot in the target's "inputs" list: sampler "negative" is slot 2.
        var link = root["links"]!.AsArray().Single(l => (int)l![0]! == 6)!.AsArray();
        Assert.That(link.Select(x => x!.ToJsonString()), Is.EqualTo(new[] { "6", "7", "0", "3", "2", "\"CONDITIONING\"" }));
        // Execution order: the loader before the sampler before the decode.
        int Order(int id) => (int)root["nodes"]!.AsArray().Single(n => (int)n!["id"]! == id)!["order"]!;
        Assert.That(Order(4), Is.LessThan(Order(3)));
        Assert.That(Order(3), Is.LessThan(Order(8)));
    }

    [Test]
    public void Groups_and_extra_pass_through_untouched()
    {
        var doc = Open(ComfySamples.DefaultWorkflow, out _);
        var root = JsonNode.Parse(ComfyWorkflow.Save(doc))!.AsObject();
        Assert.That((string)root["groups"]![0]!["title"]!, Is.EqualTo("Group"));
        Assert.That((double)root["extra"]!["ds"]!["scale"]!, Is.EqualTo(1));
    }

    [Test]
    public void A_node_type_that_is_not_installed_survives_a_round_trip_as_a_placeholder()
    {
        var json = ComfySamples.DefaultWorkflow.Replace("\"type\": \"VAEDecode\"", "\"type\": \"FancyDecodeFromAPack\"");
        var doc = Open(json, out var warnings);
        Assert.That(warnings.Single(), Does.Contain("FancyDecodeFromAPack").And.Contain("not installed"));
        Assert.That(doc.Graph.LinkInto(8, "samples"), Is.Not.Null, "Links into a missing node are kept.");

        var again = Open(ComfyWorkflow.Save(doc), out _);
        Assert.That(again.Graph.GetNode(8).ClassType, Is.EqualTo("FancyDecodeFromAPack"));
        Assert.That(again.Graph.Links.Count, Is.EqualTo(9));
    }

    [Test]
    public void An_API_format_prompt_is_refused_with_a_message_that_says_so()
    {
        var doc = new EditorDocument(Fixtures.Registry());
        var ex = Assert.Throws<InvalidDataException>(() => ComfyWorkflow.Load("""{"3": {"class_type": "KSampler", "inputs": {}}}""", doc));
        Assert.That(ex!.Message, Does.Contain("API-format"));
    }

    [Test]
    public void A_widget_value_of_the_wrong_kind_becomes_the_default_with_a_warning()
    {
        var json = ComfySamples.DefaultWorkflow.Replace("[512, 512, 1]", "[\"wide\", 512, 1]");
        var doc = Open(json, out var warnings);
        Assert.That(warnings.Single(), Does.Contain("width"));
        Assert.That(doc.Graph.GetNode(5).Widgets["width"], Is.EqualTo(512L));
    }

    [Test]
    public void Mute_bypass_and_collapse_round_trip()
    {
        var doc = Open(ComfySamples.DefaultWorkflow, out _);
        doc.SetMode(3, NodeMode.Bypass);
        doc.SetMode(7, NodeMode.Never);
        doc.SetCollapsed(6, true);
        var again = Open(ComfyWorkflow.Save(doc), out _);
        Assert.That(again.Graph.GetNode(3).Mode, Is.EqualTo(NodeMode.Bypass));
        Assert.That(again.Graph.GetNode(7).Mode, Is.EqualTo(NodeMode.Never));
        Assert.That(again.Graph.GetNode(6).Collapsed, Is.True);
    }

    [Test]
    public void A_workflow_built_in_the_editor_round_trips_too()
    {
        var (doc, _) = Fixtures.Txt2Img();
        doc.Graph.GetNode(1).Widgets["ckpt_name"] = "sd_turbo";
        var again = Open(ComfyWorkflow.Save(doc), out var warnings);
        Assert.That(warnings, Is.Empty);
        Assert.That((again.Graph.Nodes.Count, again.Graph.Links.Count), Is.EqualTo((doc.Graph.Nodes.Count, doc.Graph.Links.Count)));
        Assert.That(new PromptExecutor(again.Registry).Validate(GraphCompiler.Compile(again.Graph, again.Registry).Graph).Errors, Is.Empty);
    }
}

public class CompilerTests
{
    [Test]
    public void Widgets_become_literals_and_links_become_links()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        doc.SetWidget(ids["sampler"], "steps", 4L);
        var p = GraphCompiler.Compile(doc.Graph, doc.Registry).Graph;

        var sampler = p.Nodes[ids["sampler"].ToString()];
        Assert.That(sampler.Inputs["steps"].Literal, Is.EqualTo(4L));
        Assert.That(sampler.Inputs["positive"], Is.EqualTo(PromptInput.Link(ids["pos"].ToString(), 0)));
        Assert.That(sampler.Inputs["model"], Is.EqualTo(PromptInput.Link(ids["ckpt"].ToString(), 0)));
        Assert.That(new PromptExecutor(doc.Registry).Validate(p).Errors, Is.Empty);
    }

    [Test]
    public void A_linked_widget_input_takes_the_link_and_not_the_stored_value()
    {
        var registry = Fixtures.Registry().Add<IntConstant>();
        var doc = new EditorDocument(registry);
        var constant = doc.AddNode("IntConstant", 0, 0).Id;
        var sampler = doc.AddNode("KSampler", 300, 0).Id;
        doc.SetWidget(sampler, "steps", 7L);
        Assert.That(doc.Connect(constant, 0, sampler, "steps", out var error), Is.Not.Null, error);

        var p = GraphCompiler.Compile(doc.Graph, registry).Graph;
        Assert.That(p.Nodes[sampler.ToString()].Inputs["steps"], Is.EqualTo(PromptInput.Link(constant.ToString(), 0)), "A fed widget takes the link, not its own value.");
        Assert.That(p.Nodes[sampler.ToString()].Inputs["cfg"].IsLink, Is.False);
    }

    [Test]
    public void A_widget_that_is_fed_by_a_link_turns_into_a_slot_row_and_its_widget_row_disappears()
    {
        var registry = Fixtures.Registry().Add<IntConstant>();
        var doc = new EditorDocument(registry);
        var constant = doc.AddNode("IntConstant", 0, 0).Id;
        var sampler = doc.AddNode("KSampler", 300, 0).Id;
        var hits = new HitTester(doc);
        var before = hits.Geometry(doc.Graph.GetNode(sampler));
        Assert.That(before.Widgets.Select(w => w.Name), Does.Contain("steps"));

        doc.Connect(constant, 0, sampler, "steps", out _);
        var after = hits.Geometry(doc.Graph.GetNode(sampler));
        Assert.That(after.Widgets.Select(w => w.Name), Does.Not.Contain("steps"));
        Assert.That(after.Input("steps")!.IsConnected, Is.True);
        Assert.That(after.Bounds.Height, Is.LessThan(before.Bounds.Height + NodeLayout.SlotRowHeight), "One widget row became one slot row.");
    }

    [Test]
    public void A_muted_node_and_everything_that_needs_it_is_left_out_with_a_note()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        doc.SetMode(ids["latent"], NodeMode.Never);
        var r = GraphCompiler.Compile(doc.Graph, doc.Registry);

        Assert.That(r.Graph.Nodes.Keys, Does.Not.Contain(ids["latent"].ToString()));
        Assert.That(r.Graph.Nodes.Keys, Does.Not.Contain(ids["sampler"].ToString()), "The sampler needs the muted latent.");
        Assert.That(r.Graph.Nodes.Keys, Does.Not.Contain(ids["save"].ToString()));
        Assert.That(r.Graph.Nodes.Keys, Does.Contain(ids["pos"].ToString()), "Unrelated branches still run.");
        Assert.That(r.Notes, Has.Some.Contain("muted"));
        Assert.That(r.Notes, Has.Some.Contain("depends on a muted or missing node"));
    }

    [Test]
    public void A_bypassed_node_hands_its_input_of_the_same_type_to_its_consumers()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        doc.SetMode(ids["sampler"], NodeMode.Bypass); // LATENT in -> LATENT out
        var p = GraphCompiler.Compile(doc.Graph, doc.Registry).Graph;

        Assert.That(p.Nodes.Keys, Does.Not.Contain(ids["sampler"].ToString()));
        Assert.That(p.Nodes[ids["decode"].ToString()].Inputs["samples"], Is.EqualTo(PromptInput.Link(ids["latent"].ToString(), 0)));
        Assert.That(new PromptExecutor(doc.Registry).Validate(p).Errors, Is.Empty);
    }

    [Test]
    public void Bypassing_a_node_with_no_matching_input_cuts_its_consumers_off()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        doc.SetMode(ids["decode"], NodeMode.Bypass); // IMAGE out, but nothing IMAGE comes in
        var r = GraphCompiler.Compile(doc.Graph, doc.Registry);
        Assert.That(r.Graph.Nodes.Keys, Does.Not.Contain(ids["save"].ToString()));
        Assert.That(r.Notes, Has.Some.Contain("Save Image"));
    }

    [Test]
    public void Chained_bypasses_are_looked_through()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        // latent -> sampler(bypassed) -> sampler2(bypassed) -> decode
        var second = doc.AddNode("KSampler", 1000, 300);
        doc.Connect(ids["sampler"], 0, second.Id, "latent_image", out _);
        doc.Connect(second.Id, 0, ids["decode"], "samples", out _);
        doc.SetMode(ids["sampler"], NodeMode.Bypass);
        doc.SetMode(second.Id, NodeMode.Bypass);

        var p = GraphCompiler.Compile(doc.Graph, doc.Registry).Graph;
        Assert.That(p.Nodes[ids["decode"].ToString()].Inputs["samples"], Is.EqualTo(PromptInput.Link(ids["latent"].ToString(), 0)));
    }

    [Test]
    public void An_unknown_node_type_is_skipped_with_a_note_instead_of_failing_the_compile()
    {
        var doc = new EditorDocument(Fixtures.Registry());
        ComfyWorkflow.Load("""{"nodes":[{"id":1,"type":"NopeNode","pos":[0,0]}],"links":[]}""", doc);
        var r = GraphCompiler.Compile(doc.Graph, doc.Registry);
        Assert.That(r.Graph.Nodes, Is.Empty);
        Assert.That(r.Notes.Single(), Does.Contain("not installed"));
    }
}

/// <summary>A node that only produces an integer, to feed widget inputs in tests.</summary>
[SpawnGen.Engine.Nodes.Node(Category = "test")]
public sealed class IntConstant
{
    public long Execute(long value = 0) => value;
}
