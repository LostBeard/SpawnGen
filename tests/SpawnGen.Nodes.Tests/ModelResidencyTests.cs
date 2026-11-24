using ILGPU;
using ILGPU.Runtime.CPU;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Which loaded models stay on the GPU between prompts (MlRuntime.GetOrLoadPipelineAsync): before anything new loads, the pipelines the current
/// prompt has not used are freed; models handed out as MODEL / CLIP / VAE (GetOrLoadAsync) are never freed this way. Found 2026-11-19: running
/// every example in a row on the site lost the WebGPU device at FLUX 3 Klein (FLUX.2 Klein and the others still held), which runs alone.
/// </summary>
public class ModelResidencyTests
{
    /// <summary>A stand-in model that counts its loads and disposals.</summary>
    public sealed class Counted(string name) : IDisposable
    {
        public static readonly Dictionary<string, (int Loads, int Disposed)> Log = new();
        public string Name => name;
        public void Dispose() { lock (Log) Log[name] = (Log.GetValueOrDefault(name).Loads, Log.GetValueOrDefault(name).Disposed + 1); }
        public static Counted Load(string name) { lock (Log) Log[name] = (Log.GetValueOrDefault(name).Loads + 1, Log.GetValueOrDefault(name).Disposed); return new Counted(name); }
    }

    [Node(Category = "test", IsOutput = true)]
    public sealed class UsesPipeline
    {
        public async Task<string> ExecuteAsync(IServiceProvider services, string name = "a")
        {
            var runtime = (MlRuntime)services.GetService(typeof(MlRuntime))!;
            return (await runtime.GetOrLoadPipelineAsync("pipe:" + name, () => Task.FromResult(Counted.Load("pipe:" + name)))).Name;
        }
    }

    [Node(Category = "test", IsOutput = true)]
    public sealed class UsesHandleModel
    {
        public async Task<string> ExecuteAsync(IServiceProvider services, string name = "h")
        {
            var runtime = (MlRuntime)services.GetService(typeof(MlRuntime))!;
            return (await runtime.GetOrLoadAsync("handle:" + name, () => Task.FromResult(Counted.Load("handle:" + name)))).Name;
        }
    }

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }

    static (int Loads, int Disposed) Of(string key) { lock (Counted.Log) return Counted.Log.GetValueOrDefault(key); }

    [Test]
    public async Task A_new_prompt_frees_the_pipelines_it_does_not_use_before_loading_its_own()
    {
        lock (Counted.Log) Counted.Log.Clear();
        using var context = ILGPU.Context.Create(b => b.CPU());
        using var cpu = context.CreateCPUAccelerator(0);
        using var runtime = new MlRuntime(new DesktopModelSource(Path.GetTempPath(), MlRuntime.RepoFolders), cpu);
        var registry = new NodeRegistry().Add<UsesPipeline>().Add<UsesHandleModel>();
        var executor = new PromptExecutor(registry, new Services(runtime));

        async Task Run(params (string Id, string Type, string Name)[] nodes)
        {
            var g = new PromptGraph();
            foreach (var (id, type, name) in nodes) g.Add(id, type, ("name", name));
            await executor.ExecuteAsync(g);
        }

        await Run(("1", "UsesPipeline", "a"), ("2", "UsesHandleModel", "h"));
        Assert.That(Of("pipe:a"), Is.EqualTo((1, 0)));

        // The next prompt uses another pipeline: "a" is freed before "b" loads; the MODEL-style model stays.
        await Run(("1", "UsesPipeline", "b"));
        Assert.That(Of("pipe:a"), Is.EqualTo((1, 1)), "a pipeline the new prompt does not use is freed");
        Assert.That(Of("pipe:b"), Is.EqualTo((1, 0)));
        Assert.That(Of("handle:h"), Is.EqualTo((1, 0)), "a model handed out as MODEL / CLIP / VAE is never freed this way");

        // The same graph again loads nothing and frees nothing.
        await Run(("1", "UsesPipeline", "b"));
        Assert.That(Of("pipe:b"), Is.EqualTo((1, 0)), "a rerun reuses its pipeline");

        // One prompt using two pipelines keeps both: the second load does not free the first, which this prompt used.
        await Run(("1", "UsesPipeline", "d"), ("2", "UsesPipeline", "e"), ("3", "UsesPipeline", "d"));
        Assert.That(Of("pipe:d"), Is.EqualTo((1, 0)));
        Assert.That(Of("pipe:e"), Is.EqualTo((1, 0)));
        Assert.That(Of("pipe:b"), Is.EqualTo((1, 1)), "the earlier prompt's pipeline is freed");
    }

    /// <summary>
    /// The library is told "models are warm" once after a prompt that loaded a model ran to the end, and not again while nothing new loads
    /// (SpawnDev.ILGPU.ML's CompilerCachePolicy: a release per run would make later first-time kernels re-import shared code).
    /// </summary>
    [Test]
    public async Task The_library_is_told_models_are_warm_once_per_prompt_that_loaded_one()
    {
        lock (Counted.Log) Counted.Log.Clear();
        using var context = ILGPU.Context.Create(b => b.CPU());
        using var cpu = context.CreateCPUAccelerator(0);
        using var runtime = new MlRuntime(new DesktopModelSource(Path.GetTempPath(), MlRuntime.RepoFolders), cpu);
        var registry = new NodeRegistry().Add<UsesPipeline>().Add<UsesHandleModel>();
        var executor = new PromptExecutor(registry, new Services(runtime));
        async Task Run(string type, string name) { var g = new PromptGraph(); g.Add("1", type, ("name", name)); await executor.ExecuteAsync(g); }

        Assert.That(runtime.PromptSucceeded(), Is.False, "nothing loaded yet");
        await Run("UsesHandleModel", "w1");
        Assert.That(runtime.PromptSucceeded(), Is.True, "the prompt loaded a model");
        Assert.That(runtime.PromptSucceeded(), Is.False, "once per load, not per call");
        await Run("UsesHandleModel", "w1");
        Assert.That(runtime.PromptSucceeded(), Is.False, "a rerun loads nothing");
        await Run("UsesPipeline", "w2");
        Assert.That(runtime.PromptSucceeded(), Is.True, "a new pipeline is a new model");
        Assert.That(runtime.WarmUps, Is.EqualTo(2));
    }
}
