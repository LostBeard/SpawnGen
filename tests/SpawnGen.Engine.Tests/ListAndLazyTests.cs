using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Tests;

[Node(Category = "test", OutputIsList = [true])]
public sealed class RangeList
{
    public IReadOnlyList<long> Execute(long count = 3) { Calls.Hit(nameof(RangeList)); return Enumerable.Range(0, (int)count).Select(i => (long)i).ToList(); }
}

[Node(Category = "test", IsOutput = true, InputIsList = true)]
public sealed class Collect
{
    public UiOutput Execute(IReadOnlyList<long> values) { Calls.Hit(nameof(Collect)); return new UiOutput("values", values.ToArray()); }
}

/// <summary>ComfyUI's GetItemFromList: takes whole lists, a widget among them.</summary>
[Node(Category = "test", InputIsList = true)]
public sealed class PickItem
{
    public long Execute(IReadOnlyList<long> list, [Input(Default = 0L)] IReadOnlyList<long> index) => list[(int)index[0]];
}

[Node(Category = "test")]
public sealed class TrueSource
{
    public long Execute(long value = 1) { Calls.Hit(nameof(TrueSource)); return value; }
}

[Node(Category = "test")]
public sealed class FalseSource
{
    public long Execute(long value = 0) { Calls.Hit(nameof(FalseSource)); return value; }
}

/// <summary>ComfyUI's If/Else Switch: only the chosen branch runs.</summary>
[Node(Category = "test")]
public sealed class LazySwitch
{
    public static IEnumerable<string> CheckLazyStatus(bool @switch, long? onFalse, long? onTrue) =>
        @switch && onTrue is null ? ["on_true"] : !@switch && onFalse is null ? ["on_false"] : [];

    public long Execute(bool @switch, [Input(Lazy = true)] long? onFalse = null, [Input(Lazy = true)] long? onTrue = null) => (@switch ? onTrue : onFalse) ?? -1;
}

/// <summary>ComfyUI's Soft Switch: when only one branch is connected it passes that one whatever the switch says (IsConnected tells "not connected" from "not evaluated yet").</summary>
[Node(Category = "test")]
public sealed class SoftSwitch
{
    public static IEnumerable<string> CheckLazyStatus(bool @switch, long? onFalse, long? onTrue, NodeContext context)
    {
        if (!context.IsConnected("on_false")) return ["on_true"];
        if (!context.IsConnected("on_true")) return ["on_false"];
        return @switch && onTrue is null ? ["on_true"] : !@switch && onFalse is null ? ["on_false"] : [];
    }

    public long Execute(bool @switch, NodeContext context, [Input(Lazy = true)] long? onFalse = null, [Input(Lazy = true)] long? onTrue = null) =>
        !context.IsConnected("on_true") ? onFalse ?? -1 : !context.IsConnected("on_false") ? onTrue ?? -1 : (@switch ? onTrue : onFalse) ?? -1;
}

[Node(Category = "test")]
public sealed class Gate
{
    public object Execute(long value, bool block = false, string message = "") => block ? new ExecutionBlocker(message.Length == 0 ? null : message) : value;
}

public class ListAndLazyTests
{
    NodeRegistry _registry = null!;
    PromptExecutor _executor = null!;

    [SetUp]
    public void SetUp()
    {
        Calls.Reset();
        _registry = new NodeRegistry().AddAssembly(typeof(ListAndLazyTests).Assembly);
        _executor = new PromptExecutor(_registry);
    }

    static PromptInput L(string id, int i = 0) => PromptInput.Link(id, i);

    static long[] SinkValues(ExecutionResult r, string id) => ((System.Collections.IEnumerable)r.Outputs[id].Ui!["values"]!).Cast<object>().Select(Convert.ToInt64).ToArray();

    [Test]
    public async Task A_list_output_runs_what_follows_once_per_item_and_the_previews_are_joined()
    {
        var g = new PromptGraph();
        g.Add("1", "RangeList", ("count", 3L));
        g.Add("2", "AddNumbers", ("a", L("1")), ("b", 10L));
        g.Add("3", "Sink", ("value", L("2")));
        var r = await _executor.ExecuteAsync(g);
        Assert.That(Calls.Of("AddNumbers"), Is.EqualTo(3));
        Assert.That(Calls.Of("Sink"), Is.EqualTo(3));
        Assert.That(SinkValues(r, "3"), Is.EqualTo(new long[] { 10, 11, 12 }));
        Assert.That(r.Outputs["3"].Values, Has.Length.EqualTo(0));
    }

    [Test]
    public async Task Shorter_lists_repeat_their_last_item()
    {
        var g = new PromptGraph();
        g.Add("1", "RangeList", ("count", 3L));
        g.Add("2", "RangeList", ("count", 2L));
        g.Add("3", "AddNumbers", ("a", L("1")), ("b", L("2")));
        g.Add("4", "Collect", ("values", L("3")));
        var r = await _executor.ExecuteAsync(g);
        Assert.That(SinkValues(r, "4"), Is.EqualTo(new long[] { 0 + 0, 1 + 1, 2 + 1 }));
        Assert.That(Calls.Of("Collect"), Is.EqualTo(1), "a list node runs once with the whole list");
    }

    [Test]
    public async Task A_list_node_takes_widget_values_as_lists_of_one()
    {
        var g = new PromptGraph();
        g.Add("1", "RangeList", ("count", 5L));
        g.Add("2", "AddNumbers", ("a", L("1")), ("b", 100L));
        g.Add("3", "PickItem", ("list", L("2")), ("index", 3L));
        g.Add("4", "Sink", ("value", L("3")));
        var r = await _executor.ExecuteAsync(g);
        Assert.That(SinkValues(r, "4"), Is.EqualTo(new long[] { 103 }));
        Assert.That(_registry.GetSchema("PickItem")!.FindInput("index")!.Widget!.Default, Is.EqualTo(0L));
        Assert.That(_registry.GetSchema("RangeList")!.Outputs[0].IsList, Is.True);
    }

    [Test]
    public async Task A_lazy_switch_runs_only_the_branch_it_chooses()
    {
        PromptGraph Graph(bool choice)
        {
            var g = new PromptGraph();
            g.Add("1", "TrueSource", ("value", 7L));
            g.Add("2", "FalseSource", ("value", 3L));
            g.Add("3", "LazySwitch", ("switch", choice), ("on_true", L("1")), ("on_false", L("2")));
            g.Add("4", "Sink", ("value", L("3")));
            return g;
        }
        var r = await _executor.ExecuteAsync(Graph(true));
        Assert.That(SinkValues(r, "4"), Is.EqualTo(new long[] { 7 }));
        Assert.That((Calls.Of("TrueSource"), Calls.Of("FalseSource")), Is.EqualTo((1, 0)), "the false branch never ran");

        r = await _executor.ExecuteAsync(Graph(false));
        Assert.That(SinkValues(r, "4"), Is.EqualTo(new long[] { 3 }));
        Assert.That((Calls.Of("TrueSource"), Calls.Of("FalseSource")), Is.EqualTo((1, 1)));
    }

    [Test]
    public async Task A_soft_switch_with_one_branch_passes_it_whatever_the_switch_says()
    {
        var g = new PromptGraph();
        g.Add("1", "FalseSource", ("value", 3L));
        g.Add("2", "SoftSwitch", ("switch", true), ("on_false", L("1")));
        g.Add("3", "Sink", ("value", L("2")));
        var r = await _executor.ExecuteAsync(g);
        Assert.That(SinkValues(r, "3"), Is.EqualTo(new long[] { 3 }));
    }

    [Test]
    public async Task A_blocker_stops_what_follows_and_its_message_is_reported_once()
    {
        var g = new PromptGraph();
        g.Add("1", "IntSource", ("value", 5L));
        g.Add("2", "Gate", ("value", L("1")), ("block", true), ("message", "not today"));
        g.Add("3", "AddNumbers", ("a", L("2")));
        g.Add("4", "Sink", ("value", L("3")));
        var obs = new Errors();
        var r = await _executor.ExecuteAsync(g, observer: obs);
        Assert.That(Calls.Of("AddNumbers"), Is.EqualTo(0));
        Assert.That(Calls.Of("Sink"), Is.EqualTo(0));
        Assert.That(r.Outputs["4"].Ui, Is.Null);
        Assert.That(obs.Messages, Is.EqualTo(new[] { "3: Execution Blocked: not today" }));

        Calls.Reset();
        g.Nodes["2"].Inputs["block"] = PromptInput.Value(false);
        r = await _executor.ExecuteAsync(g);
        Assert.That(SinkValues(r, "4"), Is.EqualTo(new long[] { 15 }));
    }

    [Test]
    public async Task An_empty_list_is_an_error_that_names_the_input()
    {
        var g = new PromptGraph();
        g.Add("1", "RangeList", ("count", 0L));
        g.Add("2", "AddNumbers", ("a", L("1")), ("b", 1L));
        g.Add("3", "Sink", ("value", L("2")));
        var ex = Assert.ThrowsAsync<NodeExecutionException>(() => _executor.ExecuteAsync(g));
        Assert.That(ex!.NodeId, Is.EqualTo("2"));
        Assert.That(ex.Message, Does.Contain("'a' is an empty list"));
    }

    sealed class Errors : IExecutionObserver
    {
        public List<string> Messages { get; } = [];
        public void OnError(string? nodeId, Exception error) => Messages.Add($"{nodeId}: {error.Message}");
    }
}
