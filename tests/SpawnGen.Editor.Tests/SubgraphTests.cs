using SpawnGen.Editor;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;

namespace SpawnGen.Editor.Tests;

public class SubgraphTests
{
    /// <summary>The compiled prompt as a comparable shape that ignores node ids: each node by class and literal inputs, each link by the same descriptions of its two ends.</summary>
    static (List<string> Nodes, List<string> Links) Shape(EditorDocument doc)
    {
        var prompt = GraphCompiler.Compile(doc.Graph, doc.Registry).Graph;
        string Describe(string id)
        {
            var n = prompt.Nodes[id];
            var literals = n.Inputs.Where(i => !i.Value.IsLink).OrderBy(i => i.Key).Select(i => $"{i.Key}={i.Value.Literal}");
            return n.ClassType + "(" + string.Join(",", literals) + ")";
        }
        var nodes = prompt.Nodes.Keys.Select(Describe).OrderBy(s => s, StringComparer.Ordinal).ToList();
        var links = prompt.Nodes.SelectMany(n => n.Value.Inputs.Where(i => i.Value.IsLink).Select(i => $"{Describe(i.Value.NodeId!)}[{i.Value.OutputIndex}] -> {Describe(n.Key)}.{i.Key}")).OrderBy(s => s, StringComparer.Ordinal).ToList();
        return (nodes, links);
    }

    static (EditorDocument Doc, Dictionary<string, int> Ids) Setup() => Fixtures.Txt2Img();

    [Test]
    public void Packing_nodes_makes_one_node_whose_ports_are_the_links_that_crossed_the_boundary()
    {
        var (doc, ids) = Setup();
        int nodesBefore = doc.Graph.Nodes.Count;
        var instance = doc.ConvertToSubgraph([ids["sampler"], ids["decode"]], "Sample and decode", out var error);
        Assert.That(error, Is.Null);
        Assert.That(instance, Is.Not.Null);
        Assert.That(doc.Graph.Nodes.Count, Is.EqualTo(nodesBefore - 2 + 1));
        var schema = doc.Registry.GetSchema(instance!.ClassType)!;
        Assert.That(schema.DisplayName, Is.EqualTo("Sample and decode"));
        // Five sources fed the sampler and the decode from outside: model, positive, negative, latent, and the VAE (same checkpoint, different output).
        Assert.That(schema.Inputs.Select(i => i.Type.Name), Is.EquivalentTo(new[] { "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE" }));
        Assert.That(schema.Outputs.Select(o => o.Type.Name), Is.EqualTo(new[] { "IMAGE" }));
        Assert.That(doc.Graph.Links.Count(l => l.ToNode == instance.Id), Is.EqualTo(5));
        Assert.That(doc.Graph.Links.Count(l => l.FromNode == instance.Id), Is.EqualTo(1));
        Assert.That(doc.Graph.Links.Count(l => l.ToNode == ids["save"]), Is.EqualTo(1), "the picture still reaches Save Image");
    }

    [Test]
    public void A_graph_with_a_subgraph_compiles_to_the_same_prompt_as_the_original()
    {
        var (doc, ids) = Setup();
        var before = Shape(doc);
        doc.ConvertToSubgraph([ids["sampler"], ids["decode"]], "S", out _);
        var after = Shape(doc);
        Assert.That(after.Nodes, Is.EqualTo(before.Nodes));
        Assert.That(after.Links, Is.EqualTo(before.Links));
        var compiled = GraphCompiler.Compile(doc.Graph, doc.Registry);
        Assert.That(new PromptExecutor(doc.Registry).Validate(compiled.Graph).Errors, Is.Empty);
        Assert.That(compiled.Graph.Nodes.Keys.Any(k => doc.Graph.FindNode(int.Parse(k)) is null), Is.True, "the inner nodes got fresh ids");
        Assert.That(compiled.Graph.Nodes.ContainsKey(ids["save"].ToString()), Is.True, "ordinary nodes keep their ids");
    }

    [Test]
    public void Packing_is_one_undo_step_that_brings_everything_back_and_redo_packs_again()
    {
        var (doc, ids) = Setup();
        var nodes = doc.Graph.Nodes.Select(n => n.Id).OrderBy(i => i).ToList();
        var links = doc.Graph.Links.Select(l => (l.FromNode, l.FromOutput, l.ToNode, l.ToInput)).OrderBy(l => l).ToList();
        doc.ConvertToSubgraph([ids["sampler"], ids["decode"]], "S", out _);
        doc.EndGesture();
        doc.Undo();
        Assert.That(doc.Graph.Nodes.Select(n => n.Id).OrderBy(i => i), Is.EqualTo(nodes));
        Assert.That(doc.Graph.Links.Select(l => (l.FromNode, l.FromOutput, l.ToNode, l.ToInput)).OrderBy(l => l), Is.EqualTo(links));
        doc.Redo();
        Assert.That(doc.Graph.Nodes.Count, Is.EqualTo(nodes.Count - 1));
    }

    [Test]
    public void Unpacking_restores_the_nodes_with_their_values_and_the_same_compiled_prompt()
    {
        var (doc, ids) = Setup();
        doc.SetWidget(ids["sampler"], "steps", 7L);
        doc.SetWidget(ids["pos"], "text", "an apple");
        var before = Shape(doc);
        var instance = doc.ConvertToSubgraph([ids["sampler"], ids["decode"]], "S", out _)!;
        Assert.That(doc.UnpackSubgraph(instance.Id, out var error), Is.True, error);
        Assert.That(doc.Graph.FindNode(instance.Id), Is.Null);
        Assert.That(doc.Graph.Nodes.Count, Is.EqualTo(7));
        var sampler = doc.Graph.Nodes.Single(n => n.ClassType == "KSampler");
        Assert.That(sampler.Widgets["steps"], Is.EqualTo(7L));
        var after = Shape(doc);
        Assert.That(after.Nodes, Is.EqualTo(before.Nodes));
        Assert.That(after.Links, Is.EqualTo(before.Links));
        doc.Undo();
        Assert.That(doc.Graph.FindNode(instance.Id), Is.Not.Null, "unpacking is one undo step");
    }

    [Test]
    public void A_source_that_feeds_several_inner_inputs_is_one_port()
    {
        var (doc, ids) = Setup();
        // Both text encodes read the checkpoint's clip: packing the two of them gives one input for it, feeding both.
        var instance = doc.ConvertToSubgraph([ids["pos"], ids["neg"]], "Prompts", out _)!;
        var schema = doc.Registry.GetSchema(instance.ClassType)!;
        Assert.That(schema.Inputs, Has.Count.EqualTo(1));
        Assert.That(schema.Outputs, Has.Count.EqualTo(2));
        Assert.That(schema.Outputs.Select(o => o.Name).Distinct().Count(), Is.EqualTo(2), "output names are made unique");
        var def = SubgraphLibrary.Get(instance.ClassType)!;
        Assert.That(def.Inputs[0].Targets, Has.Count.EqualTo(2));
    }

    [Test]
    public void Subgraphs_nest()
    {
        var (doc, ids) = Setup();
        var before = Shape(doc);
        var inner = doc.ConvertToSubgraph([ids["sampler"], ids["decode"]], "Inner", out _)!;
        var outer = doc.ConvertToSubgraph([inner.Id, ids["pos"]], "Outer", out var error);
        Assert.That(error, Is.Null);
        Assert.That(outer, Is.Not.Null);
        var after = Shape(doc);
        Assert.That(after.Nodes, Is.EqualTo(before.Nodes));
        Assert.That(after.Links, Is.EqualTo(before.Links));
    }

    [Test]
    public void A_muted_subgraph_runs_nothing_and_so_does_what_reads_from_it()
    {
        var (doc, ids) = Setup();
        var instance = doc.ConvertToSubgraph([ids["sampler"], ids["decode"]], "S", out _)!;
        doc.SetMode(instance.Id, NodeMode.Never);
        var compiled = GraphCompiler.Compile(doc.Graph, doc.Registry);
        Assert.That(compiled.Graph.Nodes.Values.Any(n => n.ClassType is "KSampler" or "VAEDecode" or "SaveImage"), Is.False);
        Assert.That(compiled.Notes.Any(n => n.Contains("muted")), Is.True);
    }

    [Test]
    public void The_workflow_file_carries_the_definitions_and_opens_in_a_fresh_editor()
    {
        var (doc, ids) = Setup();
        var before = Shape(doc);
        doc.SetWidget(ids["sampler"], "steps", 9L);
        var instance = doc.ConvertToSubgraph([ids["sampler"], ids["decode"]], "Sample and decode", out _)!;
        var json = ComfyWorkflow.Save(doc);
        Assert.That(json, Does.Contain("\"definitions\"").And.Contain("Sample and decode"));

        var fresh = new EditorDocument(Fixtures.Registry());
        var warnings = ComfyWorkflow.Load(json, fresh);
        Assert.That(warnings, Is.Empty, string.Join("; ", warnings));
        Assert.That(fresh.Graph.Nodes.Count, Is.EqualTo(doc.Graph.Nodes.Count));
        var node = fresh.Graph.Nodes.Single(n => n.ClassType == instance.ClassType);
        Assert.That(fresh.Registry.GetSchema(node.ClassType)!.Inputs, Has.Count.EqualTo(5));
        Assert.That(fresh.Graph.Links.Count, Is.EqualTo(doc.Graph.Links.Count));
        var shape = Shape(fresh);
        Assert.That(shape.Nodes.Single(n => n.StartsWith("KSampler")), Does.Contain("steps=9"));
        Assert.That(shape.Links.Count, Is.EqualTo(before.Links.Count));
    }

    [Test]
    public void Nothing_to_pack_and_not_a_subgraph_say_so()
    {
        var (doc, ids) = Setup();
        Assert.That(doc.ConvertToSubgraph([], "x", out var error), Is.Null);
        Assert.That(error, Is.Not.Null);
        Assert.That(doc.UnpackSubgraph(ids["save"], out var error2), Is.False);
        Assert.That(error2, Is.Not.Null);
    }

    [Test]
    public void Ctrl_shift_E_packs_the_selection_and_unpacks_a_selected_subgraph()
    {
        var (doc, ids) = Setup();
        var view = new EditorView(doc);
        string? message = null;
        view.Message += m => message = m;
        Assert.That(view.Key("e", Modifiers.Ctrl | Modifiers.Shift), Is.False, "nothing selected");
        Assert.That(message, Is.Not.Null);
        view.SelectOnly(ids["sampler"]);
        view.ToggleSelect(ids["decode"]);
        Assert.That(view.Key("e", Modifiers.Ctrl | Modifiers.Shift), Is.True);
        Assert.That(doc.Graph.Nodes.Count(n => n.ClassType.StartsWith("subgraph:")), Is.EqualTo(1));
        Assert.That(view.SelectedNodes.Count, Is.EqualTo(1));
        Assert.That(view.Key("e", Modifiers.Ctrl | Modifiers.Shift), Is.True);
        Assert.That(doc.Graph.Nodes.Any(n => n.ClassType.StartsWith("subgraph:")), Is.False);
        Assert.That(doc.Graph.Nodes.Count, Is.EqualTo(7));
    }
}
