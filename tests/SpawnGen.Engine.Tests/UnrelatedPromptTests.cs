using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Tests;

/// <summary>A stand-in for a MODEL handle: not plain data, so a cached output holding one counts as holding a model.</summary>
public sealed class FakeModel(string name)
{
    public string Name => name;
}

[Node(Category = "test")]
public sealed class FakeModelLoader
{
    public FakeModel Execute(string name = "a") { Calls.Hit(nameof(FakeModelLoader)); return new FakeModel(name); }
}

[Node(Category = "test")]
public sealed class UsesFakeModel
{
    public long Execute(FakeModel model, long value = 1) { Calls.Hit(nameof(UsesFakeModel)); return value + model.Name.Length; }
}

/// <summary>
/// PromptExecutor.UnrelatedPromptStarting: a prompt that reuses no cached result holding a model drops those cache entries and tells the host
/// (which frees the earlier workflow's models) before any node runs; a prompt that still builds on one keeps everything. Found 2026-11-24:
/// a browser page that had run the Stable Diffusion examples kept both checkpoints on the GPU while FLUX.2 Klein loaded, and lost the device.
/// </summary>
public class UnrelatedPromptTests
{
    NodeRegistry _registry = null!;
    PromptExecutor _executor = null!;
    int _unrelated;

    [SetUp]
    public void SetUp()
    {
        Calls.Reset();
        _unrelated = 0;
        _registry = new NodeRegistry().AddAssembly(typeof(UnrelatedPromptTests).Assembly);
        _executor = new PromptExecutor(_registry) { UnrelatedPromptStarting = () => { _unrelated++; return Task.CompletedTask; } };
    }

    static PromptInput L(string id, int i = 0) => PromptInput.Link(id, i);

    /// <summary>loader 1 -> model user 2 -> sink 3</summary>
    static PromptGraph WithModel(string model = "a", long value = 1)
    {
        var g = new PromptGraph();
        g.Add("1", "FakeModelLoader", ("name", model));
        g.Add("2", "UsesFakeModel", ("model", L("1")), ("value", value));
        g.Add("3", "Sink", ("value", L("2")));
        return g;
    }

    /// <summary>number 1 -> (+b) 2 -> sink 3: plain data only</summary>
    static PromptGraph DataOnly(long source = 5, long addend = 10)
    {
        var g = new PromptGraph();
        g.Add("1", "IntSource", ("value", source));
        g.Add("2", "AddNumbers", ("a", L("1")), ("b", addend));
        g.Add("3", "Sink", ("value", L("2")));
        return g;
    }

    [Test]
    public async Task The_same_workflow_queued_again_keeps_its_models()
    {
        await _executor.ExecuteAsync(WithModel());
        int after1 = _unrelated;
        // a widget downstream of the loader changes: the loader's cached MODEL is reused, so nothing is freed
        await _executor.ExecuteAsync(WithModel(value: 2));
        Assert.That(_unrelated, Is.EqualTo(after1), "a prompt that reuses a cached model is not an unrelated one");
        Assert.That(Calls.Of(nameof(FakeModelLoader)), Is.EqualTo(1), "the loader did not run again");
        Assert.That(Calls.Of(nameof(UsesFakeModel)), Is.EqualTo(2));
    }

    [Test]
    public async Task A_workflow_that_reuses_no_cached_model_frees_the_earlier_ones_before_it_runs()
    {
        await _executor.ExecuteAsync(WithModel());
        int before = _unrelated;
        int ranBeforeCallback = -1;
        _executor.UnrelatedPromptStarting = () => { _unrelated++; ranBeforeCallback = Calls.Of(nameof(IntSource)); return Task.CompletedTask; };
        await _executor.ExecuteAsync(DataOnly());
        Assert.That(_unrelated, Is.EqualTo(before + 1), "the host is told once");
        Assert.That(ranBeforeCallback, Is.EqualTo(0), "before any node of the new prompt ran");

        // Back to the first workflow unchanged: its results are cached plain data, so no node needs the model and nothing loads.
        await _executor.ExecuteAsync(WithModel());
        Assert.That(Calls.Of(nameof(FakeModelLoader)), Is.EqualTo(1), "cached results need no model");
        // With a changed widget the model user must run: its MODEL entry was dropped (the host freed the model), so the loader runs again.
        await _executor.ExecuteAsync(WithModel(value: 7));
        Assert.That(Calls.Of(nameof(FakeModelLoader)), Is.EqualTo(2), "a freed model is loaded again, never taken from the cache");
    }

    [Test]
    public async Task Sharing_only_plain_data_with_an_earlier_workflow_still_frees_its_models_and_keeps_the_data()
    {
        // one prompt holds both a model chain and a plain-data chain
        var both = WithModel();
        both.Add("11", "IntSource", ("value", 5L));
        both.Add("12", "AddNumbers", ("a", L("11")), ("b", 10L));
        both.Add("13", "Sink", ("value", L("12")));
        await _executor.ExecuteAsync(both);
        int before = _unrelated;
        Assert.That(Calls.Of(nameof(IntSource)), Is.EqualTo(1));

        // the next workflow shares the number source (same signature) but no model
        var g = new PromptGraph();
        g.Add("11", "IntSource", ("value", 5L));
        g.Add("12", "AddNumbers", ("a", L("11")), ("b", 99L));
        g.Add("13", "Sink", ("value", L("12")));
        await _executor.ExecuteAsync(g);
        Assert.That(_unrelated, Is.EqualTo(before + 1), "a shared number is not a shared model");
        Assert.That(Calls.Of(nameof(IntSource)), Is.EqualTo(1), "the plain-data entry stayed in the cache");
    }

    [Test]
    public async Task Without_a_host_callback_nothing_is_dropped()
    {
        _executor.UnrelatedPromptStarting = null;
        await _executor.ExecuteAsync(WithModel());
        await _executor.ExecuteAsync(DataOnly());
        await _executor.ExecuteAsync(WithModel());
        Assert.That(Calls.Of(nameof(FakeModelLoader)), Is.EqualTo(1), "the cached model is still used (desktop / server default)");
    }
}
