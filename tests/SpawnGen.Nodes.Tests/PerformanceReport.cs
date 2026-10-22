using System.Diagnostics;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Per-node wall-clock times for SD-Turbo txt2img on the local GPU, to see where the time goes in the SpawnDev libraries. Explicit: needs
/// SPAWNGEN_MODELS (with vae_encoder/) and a GPU. Run in Release.
/// </summary>
public class PerformanceReport
{
    sealed class Timer : IExecutionObserver
    {
        readonly Dictionary<string, Stopwatch> _running = new();
        readonly Dictionary<string, string> _types;
        public List<(string Node, string Type, double Ms)> Rows { get; } = new();
        public Timer(PromptGraph g, IEnumerable<string> ids) => _types = ids.ToDictionary(i => i, i => g.Nodes[i].ClassType);
        public void OnNodeExecuting(string nodeId) => _running[nodeId] = Stopwatch.StartNew();
        public void OnNodeExecuted(string nodeId, NodeOutputs outputs)
        {
            if (_running.Remove(nodeId, out var sw)) Rows.Add((nodeId, _types[nodeId], sw.Elapsed.TotalMilliseconds));
        }
    }

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type t) => instances.FirstOrDefault(t.IsInstanceOfType);
    }

    static async Task<Timer> TimedAsync(PromptExecutor ex, PromptGraph g, string label)
    {
        var timer = new Timer(g, g.Nodes.Keys);
        var total = Stopwatch.StartNew();
        var r = await ex.ExecuteAsync(g, observer: timer);
        TestContext.Out.WriteLine($"### {label}: total {total.Elapsed.TotalSeconds:F2} s, ran {r.Executed.Count} nodes, {r.Cached.Count} cached");
        foreach (var row in timer.Rows) TestContext.Out.WriteLine($"  {row.Type,-18} {row.Ms,9:F0} ms");
        return timer;
    }

    [Test, Explicit("Performance report; needs SPAWNGEN_MODELS and a GPU.")]
    public async Task Sd_turbo_node_timings()
    {
        var models = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        if (string.IsNullOrEmpty(models)) Assert.Ignore("SPAWNGEN_MODELS is not set.");
        var work = Path.Combine(Path.GetTempPath(), "spawngen-perf-" + Guid.NewGuid().ToString("N"));
        using var runtime = new MlRuntime(new FolderModelSource(models!));
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly);
        var ex = new PromptExecutor(registry, new Services(runtime, new OutputOptions { Directory = work }));
        var acc = runtime.Accelerator;
        TestContext.Out.WriteLine($"device: {acc.AcceleratorType} {acc.Name}, memory {acc.MemorySize / (1024 * 1024)} MiB");

        await TimedAsync(ex, Txt2ImgTests.Graph("a photo of a red apple on a wooden table", seed: 7, size: 512), "cold: 1 step, 512x512");
        await TimedAsync(ex, Txt2ImgTests.Graph("a photo of a red apple on a wooden table", seed: 8, size: 512), "warm: 1 step, new seed");

        var four = Txt2ImgTests.Graph("a photo of a red apple on a wooden table", seed: 9, size: 512);
        four.Nodes["5"].Inputs["steps"] = PromptInput.Value(4);
        await TimedAsync(ex, four, "warm: 4 steps (UNet time per step = (this KSampler - 1 step KSampler) / 3)");
    }
}
