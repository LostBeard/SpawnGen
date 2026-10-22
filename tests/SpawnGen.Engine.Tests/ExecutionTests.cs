using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Tests;

public class ExecutionTests
{
    NodeRegistry _registry = null!;
    PromptExecutor _executor = null!;

    [SetUp]
    public void SetUp()
    {
        Calls.Reset();
        _registry = new NodeRegistry().AddAssembly(typeof(ExecutionTests).Assembly);
        _executor = new PromptExecutor(_registry);
    }

    static PromptInput L(string id, int i = 0) => PromptInput.Link(id, i);

    /// <summary>1 -> 2 (+10) -> sink 3</summary>
    static PromptGraph Chain(long source = 5, long addend = 10)
    {
        var g = new PromptGraph();
        g.Add("1", "IntSource", ("value", source));
        g.Add("2", "AddNumbers", ("a", L("1")), ("b", addend));
        g.Add("3", "Sink", ("value", L("2")));
        return g;
    }

    [Test]
    public async Task Runs_the_chain_and_returns_output_node_results()
    {
        var r = await _executor.ExecuteAsync(Chain());
        Assert.That(((long[])r.Outputs["3"].Ui!["values"]!)[0], Is.EqualTo(15));
        Assert.That(r.Executed, Is.EqualTo(new[] { "1", "2", "3" }));
        Assert.That(r.Cached, Is.Empty);
    }

    [Test]
    public async Task An_identical_second_run_is_fully_cached_but_still_returns_outputs()
    {
        await _executor.ExecuteAsync(Chain());
        var r = await _executor.ExecuteAsync(Chain());
        Assert.That(r.Executed, Is.Empty);
        Assert.That(r.Cached, Is.EqualTo(new[] { "1", "2", "3" }));
        Assert.That(((long[])r.Outputs["3"].Ui!["values"]!)[0], Is.EqualTo(15));
        Assert.That(Calls.Of("IntSource"), Is.EqualTo(1));
    }

    [Test]
    public async Task Changing_a_widget_reruns_only_that_node_and_what_is_downstream()
    {
        await _executor.ExecuteAsync(Chain(source: 5, addend: 10));
        var r = await _executor.ExecuteAsync(Chain(source: 5, addend: 20));
        Assert.That(r.Cached, Is.EqualTo(new[] { "1" }));
        Assert.That(r.Executed, Is.EqualTo(new[] { "2", "3" }));
        Assert.That(((long[])r.Outputs["3"].Ui!["values"]!)[0], Is.EqualTo(25));
        Assert.That(Calls.Of("IntSource"), Is.EqualTo(1));
    }

    [Test]
    public async Task Changing_the_source_reruns_everything_downstream()
    {
        await _executor.ExecuteAsync(Chain(source: 5));
        var r = await _executor.ExecuteAsync(Chain(source: 6));
        Assert.That(r.Cached, Is.Empty);
        Assert.That(((long[])r.Outputs["3"].Ui!["values"]!)[0], Is.EqualTo(16));
    }

    [Test]
    public async Task Whole_number_literals_hit_the_cache_regardless_of_json_number_form()
    {
        var a = Chain(); // 5 as long
        var b = Chain();
        b.Nodes["1"].Inputs["value"] = PromptInput.Value(5.0); // 5.0 as double
        await _executor.ExecuteAsync(a);
        var r = await _executor.ExecuteAsync(b);
        Assert.That(r.Executed, Is.Empty);
    }

    [Test]
    public async Task Only_the_subgraph_feeding_outputs_is_run()
    {
        var g = Chain();
        g.Add("9", "IntSource", ("value", 99)); // unconnected
        await _executor.ExecuteAsync(g);
        Assert.That(Calls.Of("IntSource"), Is.EqualTo(1));
    }

    [Test]
    public async Task Multi_output_nodes_feed_the_right_downstream_value()
    {
        var g = new PromptGraph();
        g.Add("1", "Split", ("value", 10));
        g.Add("2", "Sink", ("value", L("1", 1)));      // high = 11
        g.Add("3", "StringSink", ("text", L("1", 2))); // label
        var r = await _executor.ExecuteAsync(g);
        Assert.That(((long[])r.Outputs["2"].Ui!["values"]!)[0], Is.EqualTo(11));
        Assert.That(r.Outputs["3"].Ui!["text"], Is.EqualTo("x"));
        Assert.That(Calls.Of("Split"), Is.EqualTo(1));
    }

    [Test]
    public async Task Async_nodes_work_and_whole_floats_bind_to_ints()
    {
        var g = new PromptGraph();
        g.Add("1", "AsyncDouble", ("value", 21.0));
        g.Add("2", "Sink", ("value", L("1")));
        var r = await _executor.ExecuteAsync(g);
        Assert.That(((long[])r.Outputs["2"].Ui!["values"]!)[0], Is.EqualTo(42));
    }

    [Test]
    public async Task Double_literals_bind_to_float_parameters()
    {
        Assert.That(ValueCoercion.TryCoerce(3.5, typeof(float), out var boxed), Is.True);
        Assert.That(boxed, Is.TypeOf<float>());

        var g = new PromptGraph();
        g.Add("1", "Widgets", ("cfg_scale", 3.5));
        g.Add("2", "StringSink", ("text", L("1")));
        var r = await _executor.ExecuteAsync(g);
        Assert.That((string)r.Outputs["2"].Ui!["text"]!, Does.Contain("/3.5/"));
    }

    [Test]
    public async Task AlwaysExecute_nodes_rerun_and_invalidate_their_dependents()
    {
        var g = new PromptGraph();
        g.Add("1", "Counter");
        g.Add("2", "Sink", ("value", L("1")));
        var first = await _executor.ExecuteAsync(g);
        var second = await _executor.ExecuteAsync(g);
        Assert.That(second.Executed, Is.EqualTo(new[] { "1", "2" }));
        Assert.That(((long[])second.Outputs["2"].Ui!["values"]!)[0],
            Is.Not.EqualTo(((long[])first.Outputs["2"].Ui!["values"]!)[0]));
    }

    [Test]
    public async Task Typed_values_flow_between_nodes()
    {
        var g = new PromptGraph();
        g.Add("1", "MakeImage", ("size", 16));
        g.Add("2", "ImageToLatent", ("image", L("1")));
        g.Add("3", "LatentSink", ("latent", L("2")));
        var r = await _executor.ExecuteAsync(g);
        Assert.That((int[])r.Outputs["3"].Ui!["shape"]!, Is.EqualTo(new[] { 1, 4, 2, 2 }));
    }

    [Test]
    public async Task Node_failure_is_wrapped_with_the_node_id_and_reported()
    {
        var g = new PromptGraph();
        g.Add("1", "IntSource");
        g.Add("2", "Boom", ("value", L("1")));
        g.Add("3", "Sink", ("value", L("2")));
        var obs = new Recorder();
        var ex = Assert.ThrowsAsync<NodeExecutionException>(() => _executor.ExecuteAsync(g, observer: obs));
        Assert.That(ex!.NodeId, Is.EqualTo("2"));
        Assert.That(ex.InnerException, Is.TypeOf<InvalidOperationException>());
        Assert.That(obs.Errors, Is.EqualTo(new[] { "2" }));
    }

    [Test]
    public async Task A_failed_node_does_not_poison_the_cache_for_the_next_run()
    {
        var g = new PromptGraph();
        g.Add("1", "IntSource");
        g.Add("2", "Boom", ("value", L("1")));
        g.Add("3", "Sink", ("value", L("2")));
        Assert.ThrowsAsync<NodeExecutionException>(() => _executor.ExecuteAsync(g));
        var ok = await _executor.ExecuteAsync(Chain());
        Assert.That(ok.Outputs, Contains.Key("3"));
    }

    [Test]
    public async Task Cancellation_stops_a_running_node()
    {
        var g = new PromptGraph();
        g.Add("1", "Slow", ("value", 1));
        g.Add("2", "Sink", ("value", L("1")));
        using var cts = new CancellationTokenSource();
        var obs = new Recorder { OnProgress = () => cts.Cancel() };
        var run = _executor.ExecuteAsync(g, observer: obs, cancellationToken: cts.Token);
        Assert.ThrowsAsync<TaskCanceledException>(async () => await run);
        Assert.That(obs.Progress, Does.Contain("starting"));
    }

    [Test]
    public async Task Observer_sees_the_run_in_order()
    {
        var obs = new Recorder();
        await _executor.ExecuteAsync(Chain(), observer: obs);
        await _executor.ExecuteAsync(Chain(), observer: obs);
        Assert.That(obs.Events, Is.EqualTo(new[]
        {
            "start", "exec:1", "done:1", "exec:2", "done:2", "exec:3", "done:3", "finish",
            "start", "cached:1", "cached:2", "cached:3", "finish",
        }));
    }

    [Test]
    public async Task Concurrent_runs_are_serialized()
    {
        var g = new PromptGraph();
        g.Add("1", "AsyncDouble", ("value", 1));
        g.Add("2", "Sink", ("value", L("1")));
        var runs = await Task.WhenAll(Enumerable.Range(0, 4).Select(_ => _executor.ExecuteAsync(g)));
        Assert.That(runs.Count(r => r.Executed.Count > 0), Is.EqualTo(1), "only the first run executes; the rest are cache hits");
        Assert.That(Calls.Of("AsyncDouble"), Is.EqualTo(1));
    }

    [Test]
    public async Task Cache_is_trimmed_to_capacity()
    {
        var exec = new PromptExecutor(_registry, cache: new ResultCache(capacity: 3));
        for (var i = 0; i < 5; i++) await exec.ExecuteAsync(Chain(source: i));
        Assert.That(exec.Cache.Count, Is.LessThanOrEqualTo(3));
    }

    sealed class Recorder : IExecutionObserver
    {
        public List<string> Events { get; } = [];
        public List<string> Errors { get; } = [];
        public List<string?> Progress { get; } = [];
        public Action? OnProgress { get; set; }
        public void OnStart(string promptId, IReadOnlyList<string> plan) => Events.Add("start");
        public void OnNodeCached(string nodeId) => Events.Add("cached:" + nodeId);
        public void OnNodeExecuting(string nodeId) => Events.Add("exec:" + nodeId);
        public void OnNodeExecuted(string nodeId, NodeOutputs outputs) => Events.Add("done:" + nodeId);
        public void OnNodeProgress(string nodeId, double value, double max, string? message) { Progress.Add(message); OnProgress?.Invoke(); }
        public void OnError(string? nodeId, Exception error) { if (nodeId is not null) Errors.Add(nodeId); }
        public void OnFinished(ExecutionResult result) => Events.Add("finish");
    }
}
