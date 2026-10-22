using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Per-op wall-clock times of the SD-Turbo VAE decoder on the local GPU (GraphExecutor.PerOpSync + CapturedNodeTimingsMs), repeated so that
/// run-to-run swings show which op moves. Explicit: needs SPAWNGEN_MODELS and a GPU. PerOpSync makes every op wait for the GPU, so the total
/// is slower than a normal decode; the point is the split between ops, not the total.
/// </summary>
public class VaePerOpTimings
{
    [Test, Explicit("Diagnostic; needs SPAWNGEN_MODELS and a GPU.")]
    public async Task Vae_decode_per_op()
    {
        var models = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        if (string.IsNullOrEmpty(models)) Assert.Ignore("SPAWNGEN_MODELS is not set.");
        int runs = int.TryParse(Environment.GetEnvironmentVariable("SPAWNGEN_TIMING_RUNS"), out var n) ? n : 3;

        using var runtime = new MlRuntime(new FolderModelSource(models!));
        var acc = runtime.Accelerator;
        TestContext.Out.WriteLine($"device: {acc.AcceleratorType} {acc.Name}");
        await using var stream = await runtime.ModelSource.OpenAsync("", "vae_decoder/model.onnx");
        using var session = await InferenceSession.CreateFromOnnxStreamAsync(acc, stream,
            inputShapes: new Dictionary<string, int[]> { ["latent_sample"] = [1, 4, 64, 64] });
        TestContext.Out.WriteLine($"graph nodes: {session.NodeCount}");

        var latent = DiffusionScheduler.GenerateNoise(4, 64, 64, 1);
        async Task DecodeOnce()
        {
            using var buf = acc.Allocate1D(latent);
            var outs = await session.RunAsync(new Dictionary<string, Tensor> { [session.InputNames[0]] = new Tensor(buf.View, [1, 4, 64, 64]) });
            await acc.SynchronizeAsync();
            _ = outs[session.OutputNames[0]].Data.SubView(0, 16).GetAsArray1D(); // wait for the result like a real decode would
        }

        await DecodeOnce(); // warm-up: kernel compile, first-run shape work (not reported)
        var saved = (GraphExecutor.PerOpSync, GraphExecutor.CapturedNodeTimingsMs);
        var perRun = new List<(double WallMs, Dictionary<string, double> Nodes)>();
        try
        {
            GraphExecutor.PerOpSync = true;
            for (int r = 0; r < runs; r++)
            {
                var nodes = new Dictionary<string, double>();
                GraphExecutor.CapturedNodeTimingsMs = nodes;
                var sw = System.Diagnostics.Stopwatch.StartNew();
                await DecodeOnce();
                perRun.Add((sw.Elapsed.TotalMilliseconds, nodes));
            }
        }
        finally { (GraphExecutor.PerOpSync, GraphExecutor.CapturedNodeTimingsMs) = saved; }

        TestContext.Out.WriteLine($"## VAE decode 512x512, PerOpSync on, {runs} runs after a warm-up");
        for (int r = 0; r < perRun.Count; r++)
            TestContext.Out.WriteLine($"run {r + 1}: wall {perRun[r].WallMs:F0} ms, sum of node times {perRun[r].Nodes.Values.Sum():F0} ms, {perRun[r].Nodes.Count} nodes timed");

        // By op type: sum of node times per run.
        string Op(string key) => key.Split('_', 3)[1];
        var opTypes = perRun.SelectMany(p => p.Nodes.Keys).Select(Op).Distinct().ToList();
        TestContext.Out.WriteLine("\n## ms by op type (sum over nodes), per run");
        TestContext.Out.WriteLine("| op | " + string.Join(" | ", Enumerable.Range(1, perRun.Count).Select(i => $"run {i}")) + " | count |");
        TestContext.Out.WriteLine("|---|" + string.Concat(Enumerable.Repeat("---|", perRun.Count + 1)));
        foreach (var op in opTypes.OrderByDescending(o => perRun.Average(p => p.Nodes.Where(k => Op(k.Key) == o).Sum(k => k.Value))).Take(10))
            TestContext.Out.WriteLine($"| {op} | " + string.Join(" | ", perRun.Select(p => p.Nodes.Where(k => Op(k.Key) == op).Sum(k => k.Value).ToString("F0"))) +
                                      $" | {perRun[0].Nodes.Keys.Count(k => Op(k) == op)} |");

        // Individual nodes: slowest on average, and the ones whose time moves most between runs.
        var keys = perRun[0].Nodes.Keys.Where(k => perRun.All(p => p.Nodes.ContainsKey(k))).ToList();
        TestContext.Out.WriteLine("\n## slowest nodes (mean ms, then each run)");
        foreach (var k in keys.OrderByDescending(k => perRun.Average(p => p.Nodes[k])).Take(10))
            TestContext.Out.WriteLine($"{k}: mean {perRun.Average(p => p.Nodes[k]):F0} | " + string.Join(", ", perRun.Select(p => p.Nodes[k].ToString("F0"))));
        TestContext.Out.WriteLine("\n## nodes that move most between runs (max - min ms)");
        foreach (var k in keys.OrderByDescending(k => perRun.Max(p => p.Nodes[k]) - perRun.Min(p => p.Nodes[k])).Take(8))
            TestContext.Out.WriteLine($"{k}: spread {perRun.Max(p => p.Nodes[k]) - perRun.Min(p => p.Nodes[k]):F0} | " + string.Join(", ", perRun.Select(p => p.Nodes[k].ToString("F0"))));
    }
}
