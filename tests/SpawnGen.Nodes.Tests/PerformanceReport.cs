using ILGPU;
using ILGPU.Runtime;
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

    /// <summary>
    /// The same latent decoded many times in a VAE-only session, each time printed with a clock stamp. If the time still drifts, it is the GPU (or
    /// driver) changing speed over time, not the data and not the pipeline. SPAWNGEN_DECODE_COUNT sets the number of decodes (default 20).
    /// </summary>
    [Test, Explicit("Performance experiment; needs SPAWNGEN_MODELS and a GPU.")]
    public async Task Vae_decode_repeated_on_one_latent()
    {
        var models = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        if (string.IsNullOrEmpty(models)) Assert.Ignore("SPAWNGEN_MODELS is not set.");
        int count = int.TryParse(Environment.GetEnvironmentVariable("SPAWNGEN_DECODE_COUNT"), out var n) ? n : 20;
        using var runtime = new MlRuntime(new FolderModelSource(models!));
        var acc = runtime.Accelerator;
        await using var stream = await runtime.ModelSource.OpenAsync("", "vae_decoder/model.onnx");
        using var vae = await SpawnDev.ILGPU.ML.InferenceSession.CreateFromOnnxStreamAsync(acc, stream,
            inputShapes: new Dictionary<string, int[]> { ["latent_sample"] = [1, 4, 64, 64] });
        var latent = SpawnDev.ILGPU.ML.Preprocessing.DiffusionScheduler.GenerateNoise(4, 64, 64, 1);
        var start = DateTime.Now;
        for (int i = 1; i <= count; i++)
        {
            using var buf = acc.Allocate1D(latent);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var outs = await vae.RunAsync(new Dictionary<string, SpawnDev.ILGPU.ML.Tensors.Tensor>
                { [vae.InputNames[0]] = new SpawnDev.ILGPU.ML.Tensors.Tensor(buf.View, [1, 4, 64, 64]) });
            await acc.SynchronizeAsync();
            _ = outs[vae.OutputNames[0]].Data.SubView(0, 16).GetAsArray1D();
            TestContext.Out.WriteLine($"RESULT decode {i,2} at +{(DateTime.Now - start).TotalSeconds,5:F0} s: {sw.Elapsed.TotalMilliseconds:F0} ms");
        }
    }

    /// <summary>
    /// Is the VAE decode time a property of the latent being decoded? Samples a latent per seed (1 step), then decodes each one several times in a
    /// VAE-only session. If the time follows the latent and not the position in the run, the cause is data-dependent, not memory.
    /// </summary>
    [Test, Explicit("Performance experiment; needs SPAWNGEN_MODELS and a GPU.")]
    public async Task Vae_time_depends_on_the_latent()
    {
        var models = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        if (string.IsNullOrEmpty(models)) Assert.Ignore("SPAWNGEN_MODELS is not set.");
        using var runtime = new MlRuntime(new FolderModelSource(models!));
        var acc = runtime.Accelerator;
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly);
        var ex = new PromptExecutor(registry, new Services(runtime, new OutputOptions { Directory = Path.GetTempPath() }));

        var seeds = new[] { 100, 101, 102, 103, 104, 105 };
        var latents = new Dictionary<int, float[]>();
        foreach (var seed in seeds)
        {
            var g = Txt2ImgTests.Graph("a photo of a red apple on a wooden table", seed: seed, size: 512);
            var r = await ex.ExecuteAsync(g, new ExecutionOptions { OutputNodeIds = ["5"] });
            latents[seed] = ((SpawnGen.Engine.Types.LatentBatch)r.Outputs["5"].Values[0]!).Data;
            var l = latents[seed];
            TestContext.Out.WriteLine($"seed {seed}: latent mean {l.Average():F3} min {l.Min():F2} max {l.Max():F2} subnormals {l.Count(v => v != 0 && MathF.Abs(v) < 1.17549435e-38f)}");
        }

        await using var stream = await runtime.ModelSource.OpenAsync("", "vae_decoder/model.onnx");
        using var vae = await SpawnDev.ILGPU.ML.InferenceSession.CreateFromOnnxStreamAsync(acc, stream,
            inputShapes: new Dictionary<string, int[]> { ["latent_sample"] = [1, 4, 64, 64] });
        async Task<double> Decode(float[] latent)
        {
            var scaled = latent.Select(v => v / Checkpoint.VaeScale).ToArray();
            using var buf = acc.Allocate1D(scaled);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var outs = await vae.RunAsync(new Dictionary<string, SpawnDev.ILGPU.ML.Tensors.Tensor>
                { [vae.InputNames[0]] = new SpawnDev.ILGPU.ML.Tensors.Tensor(buf.View, [1, 4, 64, 64]) });
            await acc.SynchronizeAsync();
            _ = outs[vae.OutputNames[0]].Data.SubView(0, 16).GetAsArray1D();
            return sw.Elapsed.TotalMilliseconds;
        }
        await Decode(latents[seeds[0]]); // warm-up, not reported
        for (int round = 1; round <= 2; round++)
            foreach (var seed in seeds)
                TestContext.Out.WriteLine($"RESULT round {round} seed {seed}: decode {await Decode(latents[seed]):F0} ms");
    }

    /// <summary>Does the VAE decode time depend on how many UNet steps ran just before it? (Seen: ~13 s after 1 step, ~20 s after 4, RX 590 OpenCL.)</summary>
    [Test, Explicit("Performance experiment; needs SPAWNGEN_MODELS and a GPU.")]
    public async Task Vae_time_versus_preceding_unet_steps()
    {
        var models = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        if (string.IsNullOrEmpty(models)) Assert.Ignore("SPAWNGEN_MODELS is not set.");
        var work = Path.Combine(Path.GetTempPath(), "spawngen-perf-" + Guid.NewGuid().ToString("N"));
        using var runtime = new MlRuntime(new FolderModelSource(models!));
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly);
        var ex = new PromptExecutor(registry, new Services(runtime, new OutputOptions { Directory = work }));

        // Diagnostic knob (suggested by the ILGPU.ML team): how many bytes of released buffers the executor lets pile up before it drains them.
        TestContext.Out.WriteLine($"GraphExecutor.MaxPendingReleaseBytes default: {SpawnDev.ILGPU.ML.Graph.GraphExecutor.MaxPendingReleaseBytes / (1024 * 1024)} MiB");
        if (long.TryParse(Environment.GetEnvironmentVariable("SPAWNGEN_MAX_PENDING_RELEASE_MB"), out var mb))
        {
            SpawnDev.ILGPU.ML.Graph.GraphExecutor.MaxPendingReleaseBytes = mb * 1024 * 1024;
            TestContext.Out.WriteLine($"GraphExecutor.MaxPendingReleaseBytes set to {mb} MiB for this run");
        }
        var sequence = (Environment.GetEnvironmentVariable("SPAWNGEN_STEP_SEQUENCE") ?? "1,1,4,1,4,2,1").Split(',').Select(int.Parse).ToArray();
        int seed = 100;
        foreach (var steps in sequence)
        {
            var g = Txt2ImgTests.Graph("a photo of a red apple on a wooden table", seed: seed++, size: 512);
            g.Nodes["5"].Inputs["steps"] = PromptInput.Value(steps);
            var t = await TimedAsync(ex, g, $"{steps} step(s)");
            double vae = t.Rows.First(r => r.Type == "VAEDecode").Ms, unet = t.Rows.First(r => r.Type == "KSampler").Ms;
            TestContext.Out.WriteLine($"RESULT steps={steps} sampler={unet:F0} ms vae={vae:F0} ms");
        }
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
