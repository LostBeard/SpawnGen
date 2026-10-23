using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Tests;

[Node(Category = "test")]
public sealed class SumAll
{
    public long Execute([Autogrow(Prefix = "value", Min = 1, Max = 5)] IReadOnlyList<long> values) => values.Sum();
}

[Node(Category = "test")]
public sealed class NameAll
{
    public string Execute([Autogrow(Names = ["a", "b", "c"], Min = 0)] IReadOnlyDictionary<string, long> named) =>
        string.Join(",", named.Select(kv => $"{kv.Key}={kv.Value}"));
}

public class AutogrowTests
{
    NodeRegistry _registry = null!;
    PromptExecutor _executor = null!;

    [SetUp]
    public void SetUp()
    {
        Calls.Reset();
        _registry = new NodeRegistry().AddAssembly(typeof(AutogrowTests).Assembly);
        _executor = new PromptExecutor(_registry);
    }

    [Test]
    public void A_group_becomes_member_inputs_named_like_ComfyUIs()
    {
        var s = _registry.GetSchema("SumAll")!;
        Assert.That(s.Inputs.Select(i => i.Name), Is.EqualTo(new[] { "values.value0", "values.value1", "values.value2", "values.value3", "values.value4" }));
        Assert.That(s.Inputs.Select(i => i.Required), Is.EqualTo(new[] { true, false, false, false, false }), "only the minimum is required");
        Assert.That(s.Inputs.All(i => i.Group == "values" && i.GroupMin == 1 && i.Type == SocketType.Int));
        Assert.That(_registry.GetSchema("NameAll")!.Inputs.Select(i => i.Name), Is.EqualTo(new[] { "named.a", "named.b", "named.c" }));
    }

    [Test]
    public async Task The_node_gets_the_given_members_in_order_with_gaps_skipped()
    {
        var g = new PromptGraph();
        g.Add("1", "IntSource", ("value", 5L));
        g.Add("2", "SumAll", ("values.value0", PromptInput.Link("1")), ("values.value2", 7L), ("values.value4", PromptInput.Link("1")));
        g.Add("3", "Sink", ("value", PromptInput.Link("2")));
        var r = await _executor.ExecuteAsync(g);
        Assert.That(((long[])r.Outputs["3"].Ui!["values"]!)[0], Is.EqualTo(17));
    }

    [Test]
    public async Task The_dictionary_form_knows_which_socket_each_value_came_from()
    {
        var g = new PromptGraph();
        g.Add("1", "NameAll", ("named.c", 3L), ("named.a", 1L));
        g.Add("2", "StringSink", ("text", PromptInput.Link("1")));
        var r = await _executor.ExecuteAsync(g);
        Assert.That(r.Outputs["2"].Ui!["text"], Is.EqualTo("a=1,c=3"));
    }

    [Test]
    public void A_missing_required_member_is_a_validation_error()
    {
        var g = new PromptGraph();
        g.Add("2", "SumAll", ("values.value1", 7L));
        g.Add("3", "Sink", ("value", PromptInput.Link("2")));
        var v = _executor.Validate(g);
        Assert.That(v.Errors.Select(e => e.Input), Does.Contain("values.value0"));

        g.Nodes["2"].Inputs["values.value0"] = PromptInput.Value("not a number");
        Assert.That(_executor.Validate(g).Errors.Select(e => e.Code), Is.EqualTo(new[] { "bad_value" }), "a member given a value must be of its type");
    }
}
