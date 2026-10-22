using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Tests;

public class ValidationTests
{
    static readonly NodeRegistry Registry = new NodeRegistry().AddAssembly(typeof(ValidationTests).Assembly);

    static ValidationResult Validate(PromptGraph g) => PromptValidator.Validate(Registry, g);
    static PromptInput L(string id, int i = 0) => PromptInput.Link(id, i);

    static void AssertError(ValidationResult r, string code, string? nodeId = null, string? input = null)
    {
        Assert.That(r.IsValid, Is.False);
        Assert.That(r.Errors.Any(e => e.Code == code && (nodeId is null || e.NodeId == nodeId) && (input is null || e.Input == input)),
            Is.True, "expected " + code + " but got: " + string.Join(" | ", r.Errors.Select(e => $"{e.Code}@{e.NodeId}.{e.Input}")));
    }

    [Test]
    public void A_valid_graph_has_a_dependency_ordered_plan()
    {
        var g = new PromptGraph();
        g.Add("3", "Sink", ("value", L("2")));
        g.Add("2", "AddNumbers", ("a", L("1")));
        g.Add("1", "IntSource");
        var r = Validate(g);
        Assert.That(r.IsValid, Is.True);
        Assert.That(r.Plan, Is.EqualTo(new[] { "1", "2", "3" }));
        Assert.That(r.OutputNodeIds, Is.EqualTo(new[] { "3" }));
    }

    [Test]
    public void No_output_node_is_an_error()
    {
        var g = new PromptGraph();
        g.Add("1", "IntSource");
        AssertError(Validate(g), "no_outputs");
    }

    [Test]
    public void Unknown_node_type()
    {
        var g = new PromptGraph();
        g.Add("1", "NopeNode");
        g.Add("2", "Sink", ("value", L("1")));
        AssertError(Validate(g), "unknown_node_type", "1");
    }

    [Test]
    public void Missing_required_input()
    {
        var g = new PromptGraph();
        g.Add("1", "AddNumbers"); // 'a' is required
        g.Add("2", "Sink", ("value", L("1")));
        AssertError(Validate(g), "required_input_missing", "1", "a");
    }

    [Test]
    public void Link_to_missing_node_and_bad_output_index()
    {
        var g = new PromptGraph();
        g.Add("1", "Sink", ("value", L("42")));
        AssertError(Validate(g), "missing_node", "1", "value");

        var h = new PromptGraph();
        h.Add("1", "IntSource");
        h.Add("2", "Sink", ("value", L("1", 3)));
        AssertError(Validate(h), "bad_link", "2", "value");
    }

    [Test]
    public void Socket_type_mismatch()
    {
        var g = new PromptGraph();
        g.Add("1", "MakeImage");
        g.Add("2", "LatentSink", ("latent", L("1"))); // IMAGE into LATENT
        AssertError(Validate(g), "type_mismatch", "2", "latent");
    }

    [Test]
    public void Out_of_range_and_wrong_kind_literals()
    {
        var tooBig = new PromptGraph();
        tooBig.Add("1", "Widgets", ("steps", 500));
        tooBig.Add("2", "StringSink", ("text", L("1")));
        AssertError(Validate(tooBig), "bad_value", "1", "steps");

        var fractional = new PromptGraph();
        fractional.Add("1", "Widgets", ("steps", 2.5));
        fractional.Add("2", "StringSink", ("text", L("1")));
        AssertError(Validate(fractional), "bad_value", "1", "steps");

        var wrongKind = new PromptGraph();
        wrongKind.Add("1", "Widgets", ("flag", "yes"));
        wrongKind.Add("2", "StringSink", ("text", L("1")));
        AssertError(Validate(wrongKind), "bad_value", "1", "flag");
    }

    [Test]
    public void Combo_values_must_be_listed()
    {
        var g = new PromptGraph();
        g.Add("1", "Widgets", ("pick", "zzz"), ("mode", "Slow"));
        g.Add("2", "StringSink", ("text", L("1")));
        var r = Validate(g);
        AssertError(r, "bad_value", "1", "pick");
        Assert.That(r.Errors.Any(e => e.Input == "mode"), Is.False, "'Slow' is a valid Mode");
    }

    [Test]
    public void A_value_where_a_link_is_required()
    {
        var g = new PromptGraph();
        g.Add("1", "LatentSink", ("latent", 5));
        AssertError(Validate(g), "bad_value", "1", "latent");
    }

    [Test]
    public void Cycle_is_detected()
    {
        var g = new PromptGraph();
        g.Add("1", "AddNumbers", ("a", L("2")));
        g.Add("2", "AddNumbers", ("a", L("1")));
        g.Add("3", "Sink", ("value", L("2")));
        AssertError(Validate(g), "cycle");
    }

    [Test]
    public void Problems_outside_the_output_subgraph_are_ignored()
    {
        var g = new PromptGraph();
        g.Add("1", "IntSource");
        g.Add("2", "Sink", ("value", L("1")));
        g.Add("9", "NopeNode"); // unused, broken
        g.Add("8", "AddNumbers"); // unused, missing required
        Assert.That(Validate(g).IsValid, Is.True);
    }

    [Test]
    public void Explicit_output_selection_limits_the_run()
    {
        var g = new PromptGraph();
        g.Add("1", "IntSource");
        g.Add("2", "Sink", ("value", L("1")));
        g.Add("3", "Sink", ("value", L("4"))); // broken sibling output
        var r = PromptValidator.Validate(Registry, g, ["2"]);
        Assert.That(r.IsValid, Is.True);
        Assert.That(r.Plan, Is.EqualTo(new[] { "1", "2" }));
    }

    [Test]
    public void Executing_an_invalid_prompt_throws_with_details()
    {
        var g = new PromptGraph();
        g.Add("1", "AddNumbers");
        g.Add("2", "Sink", ("value", L("1")));
        var ex = Assert.ThrowsAsync<PromptValidationException>(() => new PromptExecutor(Registry).ExecuteAsync(g));
        Assert.That(ex!.Result.Errors, Is.Not.Empty);
    }
}
