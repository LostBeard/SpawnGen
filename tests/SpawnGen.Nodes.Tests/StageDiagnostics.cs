using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Diagnostics that compare one SD-Turbo stage on the GPU against the same stage on the CPU accelerator (the oracle), to find which
/// stage goes wrong on a backend. Explicit: needs SPAWNGEN_MODELS and a GPU; the CPU run is slow.
/// </summary>
public class StageDiagnostics
{
    sealed class Services(MlRuntime runtime) : IServiceProvider
    {
        public object? GetService(Type t) => t.IsInstanceOfType(runtime) ? runtime : null;
    }

    static (double MaxAbs, double MeanAbs, int NaNs, double Mean, double Std) Compare(float[] a, float[] b)
    {
        double max = 0, sum = 0, mean = 0;
        int nans = 0;
        for (int i = 0; i < a.Length; i++)
        {
            if (float.IsNaN(a[i]) || float.IsInfinity(a[i])) { nans++; continue; }
            double d = Math.Abs((double)a[i] - b[i]);
            max = Math.Max(max, d);
            sum += d;
            mean += a[i];
        }
        mean /= a.Length;
        double var = 0;
        foreach (var x in a) var += (x - mean) * (x - mean);
        return (max, sum / a.Length, nans, mean, Math.Sqrt(var / a.Length));
    }

    /// <summary>Runs the CLIP text encoder once on <paramref name="acc"/> and returns every node's first output values (key = "idx_Op_name") and node info.</summary>
    static async Task<(Dictionary<string, float[]> Values, Dictionary<string, string> Info)> RunClipCapturedAsync(
        Accelerator acc, IModelSource source, int[] ids)
    {
        await using var stream = await source.OpenAsync("", "text_encoder/model.onnx");
        using var session = await InferenceSession.CreateFromOnnxStreamAsync(acc, stream,
            inputShapes: new Dictionary<string, int[]> { ["input_ids"] = [1, Checkpoint.TokenCount] });
        using var buf = acc.Allocate1D(ids.Select(t => (float)t).ToArray());
        var values = new Dictionary<string, float[]>();
        var info = new Dictionary<string, string>();
        var (savedV, savedI, savedMax) = (GraphExecutor.CapturedOutputs, GraphExecutor.CapturedNodeInfo, GraphExecutor.CaptureMaxElements);
        try
        {
            GraphExecutor.CapturedOutputs = values;
            GraphExecutor.CapturedNodeInfo = info;
            GraphExecutor.CaptureMaxElements = 8192;
            await session.RunAsync(new Dictionary<string, Tensor>
                { [session.InputNames[0]] = new Tensor(buf.View, [1, Checkpoint.TokenCount]) });
        }
        finally
        {
            GraphExecutor.CapturedOutputs = savedV;
            GraphExecutor.CapturedNodeInfo = savedI;
            GraphExecutor.CaptureMaxElements = savedMax;
        }
        var table = session.TryGetWeight("text_model.embeddings.token_embedding.weight");
        TokenTable = table?.Data.SubView(0, table.ElementCount).GetAsArray1D();
        return (values, info);
    }

    /// <summary>Token-embedding table of the most recent captured run (fp32 view of the session weight), for row matching.</summary>
    static float[]? TokenTable;

    [Test, Explicit("Diagnostic; needs SPAWNGEN_MODELS (with vae_encoder/) and a GPU.")]
    public async Task Vae_encoder_io()
    {
        var models = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        if (string.IsNullOrEmpty(models)) Assert.Ignore("SPAWNGEN_MODELS is not set.");
        using var runtime = new MlRuntime(new FolderModelSource(models!));
        await using var stream = await runtime.ModelSource.OpenAsync("", "vae_encoder/model.onnx");
        using var s = await InferenceSession.CreateFromOnnxStreamAsync(runtime.Accelerator, stream);
        TestContext.Out.WriteLine("inputs: " + string.Join(", ", s.InputNames.Select(n => $"{n} [{string.Join(",", s.InputShapes[n])}]")));
        TestContext.Out.WriteLine("outputs: " + string.Join(", ", s.OutputNames.Select(n => $"{n} [{string.Join(",", s.OutputShapes[n])}]")));
    }

    [Test, Explicit("Diagnostic; needs SPAWNGEN_MODELS, a GPU, and minutes on the CPU accelerator.")]
    public async Task Clip_first_diverging_node_gpu_vs_cpu()
    {
        var models = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        if (string.IsNullOrEmpty(models)) Assert.Ignore("SPAWNGEN_MODELS is not set.");
        var source = new FolderModelSource(models!);
        var vocab = await source.FetchBytesAsync("", "tokenizer/vocab.json");
        var merges = await source.FetchBytesAsync("", "tokenizer/merges.txt");
        var tok = SpawnDev.ILGPU.ML.Preprocessing.BPETokenizer.Load(
            System.Text.Encoding.UTF8.GetString(vocab), System.Text.Encoding.UTF8.GetString(merges));
        var ids = tok.EncodeCLIP("a photo of a red apple on a wooden table", Checkpoint.TokenCount);

        using var gpuCtx = MLContext.Create().ToContext();
        using var gpuAcc = gpuCtx.GetPreferredDevice(false).CreateAccelerator(gpuCtx);
        TestContext.Out.WriteLine($"GPU: {gpuAcc.AcceleratorType} {gpuAcc.Name}");
        var gpu = await RunClipCapturedAsync(gpuAcc, source, ids);

        using var cpuCtx = MLContext.Create().ToContext();
        using var cpuAcc = MLContext.CreateCpuAccelerator(cpuCtx);
        var cpu = await RunClipCapturedAsync(cpuAcc, source, ids);

        TestContext.Out.WriteLine($"captured nodes: gpu={gpu.Values.Count} cpu={cpu.Values.Count}");
        int shown = 0;
        foreach (var key in gpu.Values.Keys.OrderBy(k => k, StringComparer.Ordinal))
        {
            if (!cpu.Values.TryGetValue(key, out var c)) continue;
            var g = gpu.Values[key];
            int n = Math.Min(g.Length, c.Length);
            double num = 0, den = 0;
            bool bad = false;
            for (int i = 0; i < n; i++)
            {
                if (!float.IsFinite(g[i])) bad = true;
                double d = (double)g[i] - c[i];
                num += d * d;
                den += (double)c[i] * c[i];
            }
            double rel = Math.Sqrt(num / Math.Max(den, 1e-12));
            if (bad || rel > 1e-2)
            {
                TestContext.Out.WriteLine($"DIVERGES {key} relRMS={rel:E2} nonFinite={bad} n={n}");
                if (gpu.Info.TryGetValue(key, out var info)) TestContext.Out.WriteLine($"   info: {info}");
                if (++shown >= 6) break;
            }
        }
        if (shown == 0) TestContext.Out.WriteLine("no node diverged above relRMS 1e-2 in the captured prefix");

        // Per-token view of the token-embedding Gather: is every row wrong, or only some (large ids, padding)?
        var gKey = gpu.Values.Keys.First(k => k.Contains("token_embedding/Gather_output_0"));
        var gg = gpu.Values[gKey];
        var cg = cpu.Values[gKey];
        for (int t = 0; t < gg.Length / 1024; t++)
        {
            double num = 0, den = 0, maxAbs = 0;
            for (int i = 0; i < 1024; i++)
            {
                double d = (double)gg[t * 1024 + i] - cg[t * 1024 + i];
                num += d * d; den += (double)cg[t * 1024 + i] * cg[t * 1024 + i]; maxAbs = Math.Max(maxAbs, Math.Abs(d));
            }
            TestContext.Out.WriteLine($"  token pos {t} id={ids[t]} relRMS={Math.Sqrt(num / Math.Max(den, 1e-12)):E2} maxAbs={maxAbs:E2} gpu[0..2]={gg[t * 1024]:F4},{gg[t * 1024 + 1]:F4} cpu[0..2]={cg[t * 1024]:F4},{cg[t * 1024 + 1]:F4}");
        }
        // Which table row did the GPU actually return for each wrong token? (TokenTable = the CPU session's table.)
        if (TokenTable != null)
        {
            TestContext.Out.WriteLine($"token table elements: {TokenTable.Length} (expect {49408 * 1024})");
            for (int t = 0; t < 8; t++)
            {
                int bestRow = -1; double best = double.MaxValue;
                for (int r = 0; r < TokenTable.Length / 1024; r++)
                {
                    double s = 0;
                    for (int i = 0; i < 16; i++) { double d = TokenTable[r * 1024 + i] - gg[t * 1024 + i]; s += d * d; }
                    if (s < best) { best = s; bestRow = r; }
                }
                TestContext.Out.WriteLine($"  token pos {t} id={ids[t]} -> GPU returned table row {bestRow} (dist {Math.Sqrt(best):E2})");
            }
        }
        var sw = new StringWriter();
        gpuAcc.Device.PrintInformation(sw);
        TestContext.Out.WriteLine(sw.ToString());
    }

    [Test, Explicit("Diagnostic; needs SPAWNGEN_MODELS, a GPU, and minutes on the CPU accelerator.")]
    public async Task Clip_embeddings_gpu_vs_cpu()
    {
        var models = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        if (string.IsNullOrEmpty(models)) Assert.Ignore("SPAWNGEN_MODELS is not set.");
        const string prompt = "a photo of a red apple on a wooden table";

        using var runtime = new MlRuntime(new FolderModelSource(models!));
        var ckpt = await runtime.GetCheckpointAsync("sd_turbo");
        var gpuEnc = await CLIPTextEncode.EncodeAsync(new ClipHandle(ckpt), prompt, prompt, CancellationToken.None);
        var gpu = new Conditioning(gpuEnc.Context, gpuEnc.Shape);
        TestContext.Out.WriteLine($"GPU ({runtime.Accelerator.AcceleratorType}) embeddings shape [{string.Join(",", gpu.Shape)}]");

        using var cpuCtx = MLContext.Create().ToContext();
        using var cpuAcc = MLContext.CreateCpuAccelerator(cpuCtx);
        await using var stream = await runtime.ModelSource.OpenAsync("", "text_encoder/model.onnx");
        using var cpuSession = await InferenceSession.CreateFromOnnxStreamAsync(cpuAcc, stream,
            inputShapes: new Dictionary<string, int[]> { ["input_ids"] = [1, Checkpoint.TokenCount] });
        var ids = ckpt.Tokenizer.EncodeCLIP(prompt, Checkpoint.TokenCount);
        using var buf = cpuAcc.Allocate1D(ids.Select(t => (float)t).ToArray());
        var outs = await cpuSession.RunAsync(new Dictionary<string, Tensor>
            { [cpuSession.InputNames[0]] = new Tensor(buf.View, [1, Checkpoint.TokenCount]) });
        var cpu = outs[cpuSession.OutputNames[0]].Data.SubView(0, gpu.Embeddings.Length).GetAsArray1D();

        var c = Compare(gpu.Embeddings, cpu);
        TestContext.Out.WriteLine($"GPU vs CPU: maxAbs={c.MaxAbs:E3} meanAbs={c.MeanAbs:E3} nonFinite={c.NaNs} gpuMean={c.Mean:F4} gpuStd={c.Std:F4}");
        var cc = Compare(cpu, cpu);
        TestContext.Out.WriteLine($"CPU stats: mean={cc.Mean:F4} std={cc.Std:F4} nonFinite={cc.NaNs}");
    }
}
