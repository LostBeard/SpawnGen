using System.Diagnostics;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Stage timings of single-file checkpoints on this machine's GPU: load, one prompt encode, one UNet call (median of several after warm-up) and one VAE
/// decode per picture size. A step of a CFG sampler is two UNet calls.
/// </summary>
public class SafetensorsCheckpointBenchmarks
{
    [TestCase("v1-5-pruned-emaonly.safetensors", new[] { 512, 768 })]
    [TestCase("sd_xl_base_1.0.safetensors", new[] { 768, 1024 })]
    [Explicit("Benchmark; needs SPAWNGEN_MODELS_ROOT (a models folder with checkpoints/<file>) and a GPU.")]
    public async Task Stage_timings(string file, int[] sizes)
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        CheckpointCatalog.ScanCheckpointFiles(root!);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var acc = runtime.Accelerator;
        TestContext.Out.WriteLine($"{file} on {acc.Name} ({acc.AcceleratorType})");

        var sw = Stopwatch.StartNew();
        var ckpt = await runtime.GetCheckpointAsync(file);
        TestContext.Out.WriteLine($"load {sw.Elapsed.TotalSeconds:F1} s ({ckpt.Family}, {ckpt.Sampling.Prediction})");

        sw.Restart();
        var text = await ckpt.EncodeAsync(ckpt.Tokenizer.EncodeCLIP("a photo of a red apple on a wooden table", Checkpoint.TokenCount));
        double first = sw.Elapsed.TotalMilliseconds;
        sw.Restart();
        text = await ckpt.EncodeAsync(ckpt.Tokenizer.EncodeCLIP("a photo of a red apple on a wooden table", Checkpoint.TokenCount));
        TestContext.Out.WriteLine($"text encode {first:F0} ms first, {sw.Elapsed.TotalMilliseconds:F0} ms again");
        var entry = new CondEntry(text.Context, text.Shape) { Pooled = text.Pooled };

        foreach (int size in sizes)
        {
            int h = size / 8, w = size / 8;
            var latent = new float[Checkpoint.LatentChannels * h * w];
            var rng = new Random(1);
            for (int i = 0; i < latent.Length; i++) latent[i] = (float)(rng.NextDouble() * 2 - 1);
            using (var run = ckpt.BeginRun(acc))
            {
                var times = new List<double>();
                for (int i = 0; i < 10; i++)
                {
                    sw.Restart();
                    var eps = await run.PredictAsync(latent, h, w, 500f, entry, null, UNetOptions.None, null, CancellationToken.None);
                    times.Add(sw.Elapsed.TotalMilliseconds);
                    Assert.That(eps.All(float.IsFinite), $"UNet output at {size} is finite");
                }
                var steady = times.Skip(2).OrderBy(t => t).ToList();
                TestContext.Out.WriteLine($"{size}x{size} UNet call: first {times[0]:F0} ms, median {steady[steady.Count / 2]:F0} ms, min {steady[0]:F0} ms");
            }
            var decodeTimes = new List<double>();
            for (int i = 0; i < 3; i++)
            {
                sw.Restart();
                var rgb = await ckpt.DecodeAsync(latent, h, w);
                decodeTimes.Add(sw.Elapsed.TotalMilliseconds);
                Assert.That(rgb.Length, Is.EqualTo(3 * size * size));
            }
            TestContext.Out.WriteLine($"{size}x{size} VAE decode: first {decodeTimes[0]:F0} ms, then {decodeTimes[1]:F0} / {decodeTimes[2]:F0} ms");
        }
    }
}
