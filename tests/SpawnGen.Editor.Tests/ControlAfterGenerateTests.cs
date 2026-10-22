using SpawnGen.Editor;

namespace SpawnGen.Editor.Tests;

public class ControlAfterGenerateTests
{
    [Test]
    public void Next_follows_the_mode_and_stays_inside_the_seed_range()
    {
        var random = new Random(1);
        Assert.That(ControlAfterGenerate.Next(5, "fixed", random), Is.EqualTo(5));
        Assert.That(ControlAfterGenerate.Next(5, "increment", random), Is.EqualTo(6));
        Assert.That(ControlAfterGenerate.Next(5, "decrement", random), Is.EqualTo(4));
        Assert.That(ControlAfterGenerate.Next((1L << 53) - 1, "increment", random), Is.EqualTo(0), "wraps at the top");
        Assert.That(ControlAfterGenerate.Next(0, "decrement", random), Is.EqualTo((1L << 53) - 1), "wraps at the bottom");
        for (int i = 0; i < 100; i++) Assert.That(ControlAfterGenerate.Next(5, "randomize", random), Is.InRange(0L, (1L << 53) - 1));
        Assert.That(ControlAfterGenerate.Next(5, "randomize", new Random(1)), Is.Not.EqualTo(5));
    }

    [Test]
    public void The_seed_row_is_followed_by_a_control_row_that_is_not_an_input()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        var node = doc.Graph.GetNode(ids["sampler"]);
        var geo = NodeLayout.Compute(node, doc.Registry.GetSchema("KSampler")!, doc.Graph);
        var names = geo.Widgets.Select(w => w.Name).ToList();
        Assert.That(names.Take(3), Is.EqualTo(new[] { "seed", ControlAfterGenerate.Name, "steps" }));
        Assert.That(geo.Widgets[1].Bounds.Y, Is.GreaterThan(geo.Widgets[0].Bounds.Bottom - 1));
        Assert.That(geo.Widgets[1].Bounds.Bottom, Is.LessThanOrEqualTo(geo.Widgets[2].Bounds.Y + 0.01));
        Assert.That(geo.Inputs.Any(i => i.Name == ControlAfterGenerate.Name), Is.False, "it has no socket");
        Assert.That(geo.Bounds.Bottom, Is.GreaterThan(geo.Widgets[^1].Bounds.Bottom));
    }

    [Test]
    public void A_new_node_randomizes_like_ComfyUI_and_the_mode_can_be_set()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        var sampler = doc.Graph.GetNode(ids["sampler"]);
        Assert.That(ControlAfterGenerate.ModeOf(sampler), Is.EqualTo("randomize"));
        doc.SetWidget(sampler.Id, ControlAfterGenerate.Name, "increment");
        Assert.That(ControlAfterGenerate.ModeOf(sampler), Is.EqualTo("increment"));
    }

    [Test]
    public void After_a_queue_every_seed_moves_on_by_its_own_mode_in_one_undo_step()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        var sampler = doc.Graph.GetNode(ids["sampler"]);
        var second = doc.AddNode("KSamplerAdvanced", 900, 600);
        doc.SetWidget(sampler.Id, "seed", 10L);
        doc.SetWidget(sampler.Id, ControlAfterGenerate.Name, "increment");
        doc.SetWidget(second.Id, "noise_seed", 20L);
        doc.SetWidget(second.Id, ControlAfterGenerate.Name, "fixed");
        doc.ClearHistory();

        ControlAfterGenerate.ApplyAfterQueue(doc, doc.Registry.GetSchema, new Random(3));
        Assert.That(sampler.Widgets["seed"], Is.EqualTo(11L));
        Assert.That(second.Widgets["noise_seed"], Is.EqualTo(20L), "fixed stays");

        doc.Undo();
        Assert.That(sampler.Widgets["seed"], Is.EqualTo(10L));
        Assert.That(doc.CanUndo, Is.False, "the whole change was one step");
    }

    [Test]
    public void A_seed_fed_by_a_link_and_a_muted_node_keep_their_values()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        var sampler = doc.Graph.GetNode(ids["sampler"]);
        doc.SetWidget(sampler.Id, "seed", 10L);
        doc.SetWidget(sampler.Id, ControlAfterGenerate.Name, "increment");
        var primitive = doc.AddNode("PrimitiveInt", 400, 800);
        Assert.That(doc.Connect(primitive.Id, 0, sampler.Id, "seed", out var err), Is.Not.Null, err);
        ControlAfterGenerate.ApplyAfterQueue(doc, doc.Registry.GetSchema, new Random(3));
        Assert.That(sampler.Widgets["seed"], Is.EqualTo(10L), "the linked value belongs to its source");

        var (doc2, ids2) = Fixtures.Txt2Img();
        var muted = doc2.Graph.GetNode(ids2["sampler"]);
        doc2.SetWidget(muted.Id, "seed", 10L);
        doc2.SetWidget(muted.Id, ControlAfterGenerate.Name, "increment");
        doc2.SetMode(muted.Id, NodeMode.Never);
        ControlAfterGenerate.ApplyAfterQueue(doc2, doc2.Registry.GetSchema, new Random(3));
        Assert.That(muted.Widgets["seed"], Is.EqualTo(10L));
    }

    [Test]
    public void The_mode_is_read_from_and_written_to_ComfyUI_workflow_files()
    {
        var doc = new EditorDocument(Fixtures.Registry());
        ComfyWorkflow.Load(ComfySamples.DefaultWorkflow, doc);
        var sampler = doc.Graph.GetNode(3);
        Assert.That(ControlAfterGenerate.ModeOf(sampler), Is.EqualTo("randomize"));

        doc.SetWidget(3, ControlAfterGenerate.Name, "decrement");
        var json = ComfyWorkflow.Save(doc);
        var again = new EditorDocument(Fixtures.Registry());
        ComfyWorkflow.Load(json, again);
        Assert.That(ControlAfterGenerate.ModeOf(again.Graph.GetNode(3)), Is.EqualTo("decrement"));
        Assert.That(again.Graph.GetNode(3).Widgets["steps"], Is.EqualTo(20L), "the control value does not shift the other widgets");
    }

    [Test]
    public void The_control_mode_does_not_reach_the_prompt()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        doc.SetWidget(ids["sampler"], ControlAfterGenerate.Name, "increment");
        var compiled = GraphCompiler.Compile(doc.Graph, doc.Registry);
        Assert.That(compiled.Graph.Nodes[ids["sampler"].ToString()].Inputs.ContainsKey(ControlAfterGenerate.Name), Is.False);
    }
}
