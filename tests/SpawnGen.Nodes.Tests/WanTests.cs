using SpawnDev.ILGPU.ML.Wan;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>Wan 2.2 TI2V 5B through the library directly, for comparing devices (the numbers Tuvok's reference run computes too).</summary>
public class WanTests
{
    [Test, Explicit("Needs SPAWNGEN_MODELS (the models folder), a GPU and the Wan files (~9.7 GB). Writes wan_stats.txt next to the test output.")]
    public async Task Wan_latent_and_frame_statistics_for_a_device_comparison()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS is not set.");
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var pipeline = new Wan22Pipeline(runtime.Accelerator, runtime.ModelSource, Wan22Pipeline.DefaultRepo, Wan22Paths.InRepo(textEncoder: Wan22Paths.TextEncoderFile("q8")));
        var request = new WanRequest
        {
            Prompt = "A red apple slowly rotates on a wooden table, soft morning light", NegativePrompt = null!,
            Width = 640, Height = 352, Frames = 9, Steps = 50, GuidanceScale = 5, Shift = 5, Seed = 42, Sampler = WanSampler.UniPC, RgbaFrames = true,
        };
        var started = DateTime.Now;
        var video = await pipeline.GenerateAsync(request);
        var report = new System.Text.StringBuilder();
        report.AppendLine($"device: {runtime.Accelerator.Name} ({runtime.Accelerator.AcceleratorType}); {(DateTime.Now - started).TotalSeconds:F1} s");
        report.AppendLine("request: 640x352, 9 frames, 50 steps, cfg 5, shift 5, seed 42, UniPC, default negative (null), DiT q8, umT5 q8, VAE f16, ML 5.8.14");

        var latent = video.Latent;
        int channels = 48, per = latent.Length / channels;
        report.AppendLine($"latent: count {latent.Length} (48 x {per}), mean {Mean(latent, 0, latent.Length):G6}, std {Std(latent, 0, latent.Length):G6}, min {latent.Min():G6}, max {latent.Max():G6}");
        for (int c = 0; c < channels; c++)
            report.AppendLine($"  channel {c,2}: mean {Mean(latent, c * per, per),12:G6}  std {Std(latent, c * per, per),12:G6}");
        for (int f = 0; f < video.Frames; f++)
        {
            var rgba = video.GetFrameRgba(f);
            report.Append($"frame {f}:");
            for (int ch = 0; ch < 3; ch++)
            {
                var v = Enumerable.Range(0, rgba.Length / 4).Select(i => (double)rgba[i * 4 + ch]).ToArray();
                double mean = v.Average(), std = Math.Sqrt(v.Select(x => (x - mean) * (x - mean)).Average());
                double zero = 100.0 * v.Count(x => x == 0) / v.Length, full = 100.0 * v.Count(x => x == 255) / v.Length;
                report.Append($"  {"RGB"[ch]} mean {mean:F2} std {std:F2} 0s {zero:F2}% 255s {full:F2}%");
            }
            report.AppendLine();
        }
        var path = Path.Combine(TestContext.CurrentContext.WorkDirectory, "wan_stats.txt");
        File.WriteAllText(path, report.ToString());
        TestContext.Out.WriteLine(report.ToString());
        TestContext.Out.WriteLine(path);
    }

    static double Mean(float[] a, int start, int count) { double s = 0; for (int i = start; i < start + count; i++) s += a[i]; return s / count; }

    static double Std(float[] a, int start, int count)
    {
        double m = Mean(a, start, count), s = 0;
        for (int i = start; i < start + count; i++) s += (a[i] - m) * (a[i] - m);
        return Math.Sqrt(s / count);
    }
}
