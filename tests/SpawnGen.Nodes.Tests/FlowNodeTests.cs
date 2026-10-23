using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

public class FlowNodeTests
{
    static PromptExecutor Executor() => new(new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly));

    static string Shown(ExecutionResult r, string id) => ((IEnumerable<string>)r.Outputs[id].Ui!["text"]!).Single();

    [TestCase(true, "yes", "1")]
    [TestCase(false, "no", "2")]
    public async Task The_switch_passes_the_chosen_branch_and_only_that_branch_runs(bool choice, string expected, string ranBranch)
    {
        var g = new PromptGraph();
        g.Add("1", "PrimitiveString", ("value", "yes"));
        g.Add("2", "PrimitiveString", ("value", "no"));
        g.Add("3", "ComfySwitchNode", ("switch", choice), ("on_true", PromptInput.Link("1")), ("on_false", PromptInput.Link("2")));
        g.Add("4", "PreviewAny", ("source", PromptInput.Link("3")));
        var r = await Executor().ExecuteAsync(g);
        Assert.That(Shown(r, "4"), Is.EqualTo(expected));
        Assert.That(r.Executed, Does.Contain(ranBranch));
        Assert.That(r.Executed, Does.Not.Contain(ranBranch == "1" ? "2" : "1"), "the other branch never ran");
    }

    [Test]
    public void The_switch_matches_ComfyUIs_schema()
    {
        var s = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).GetSchema("ComfySwitchNode")!;
        Assert.That(s.Inputs.Select(i => (i.Name, i.Lazy, i.ListedAsRequired)), Is.EqualTo(new[] { ("switch", false, true), ("on_false", true, false), ("on_true", true, false) }));
        Assert.That(s.Outputs.Select(o => o.Name), Is.EqualTo(new[] { "output" }));
        Assert.That(s.DisplayName, Is.EqualTo("If/Else Switch"));
    }

    [TestCase(0L, "a")]
    [TestCase(2L, "c")]
    [TestCase(-1L, "c")]
    public async Task Get_item_from_list_picks_one_item_of_a_list(long index, string expected)
    {
        // The list comes from a node with a list output, as it would from Create List.
        var g = new PromptGraph();
        g.Add("1", "PrimitiveString", ("value", "a"));
        g.Add("2", "PrimitiveString", ("value", "b"));
        g.Add("3", "PrimitiveString", ("value", "c"));
        g.Add("4", "TestListOf3", ("first", PromptInput.Link("1")), ("second", PromptInput.Link("2")), ("third", PromptInput.Link("3")));
        g.Add("5", "GetItemFromList", ("list", PromptInput.Link("4")), ("index", index));
        g.Add("6", "PreviewAny", ("source", PromptInput.Link("5")));
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).Add<TestListOf3>();
        var r = await new PromptExecutor(registry).ExecuteAsync(g);
        Assert.That(Shown(r, "6"), Is.EqualTo(expected));
    }

    [Test]
    public void An_index_outside_the_list_is_an_error()
    {
        var g = new PromptGraph();
        g.Add("1", "PrimitiveString", ("value", "a"));
        g.Add("4", "TestListOf3", ("first", PromptInput.Link("1")), ("second", PromptInput.Link("1")), ("third", PromptInput.Link("1")));
        g.Add("5", "GetItemFromList", ("list", PromptInput.Link("4")), ("index", 3L));
        g.Add("6", "PreviewAny", ("source", PromptInput.Link("5")));
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).Add<TestListOf3>();
        Assert.ThrowsAsync<NodeExecutionException>(() => new PromptExecutor(registry).ExecuteAsync(g));
    }
}

/// <summary>A list of three strings (a list output), standing in for Create List until autogrow inputs exist.</summary>
[Node(Category = "test", OutputIsList = [true])]
public sealed class TestListOf3
{
    public IReadOnlyList<string> Execute(string first, string second, string third) => [first, second, third];
}
