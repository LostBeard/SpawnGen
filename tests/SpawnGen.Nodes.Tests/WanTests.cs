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
        var pipeline = new Wan22Pipeline(runtime.Accelerator, runtime.ModelSource, Wan22Pipeline.DefaultRepo, Wan22Paths.InRepo());
        var request = new WanRequest
        {
            Prompt = "A red apple slowly rotates on a wooden table, soft morning light", NegativePrompt = null!,
            Width = 640, Height = 352, Frames = 9, Steps = 50, GuidanceScale = 5, Shift = 5, Seed = 42, Sampler = WanSampler.UniPC, RgbaFrames = true,
        };
        var started = DateTime.Now;
        var video = await pipeline.GenerateAsync(request);
        var report = new System.Text.StringBuilder();
        report.AppendLine($"device: {runtime.Accelerator.Name} ({runtime.Accelerator.AcceleratorType}); {(DateTime.Now - started).TotalSeconds:F1} s");
        report.AppendLine("request: 640x352, 9 frames, 50 steps, cfg 5, shift 5, seed 42, UniPC, default negative (null), DiT q8, umT5 q8, VAE f16, ML 5.8.16-local.2");

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

    [Test, Explicit("Needs SPAWNGEN_MODELS, a GPU and the Wan Turbo files. Times every pipeline stage, twice in one process (the second run has no download or first-use compile). Writes wan_timing.txt.")]
    public async Task Wan_turbo_stage_timing_at_720p()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS is not set.");
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var pipeline = new Wan22Pipeline(runtime.Accelerator, runtime.ModelSource, Wan22Pipeline.DefaultRepo,
            Wan22Paths.InRepo(Wan22Paths.DitFile("turbo_q8")));
        var report = new System.Text.StringBuilder();
        report.AppendLine($"device: {runtime.Accelerator.Name} ({runtime.Accelerator.AcceleratorType})");
        var console = Console.Out;
        for (int run = 1; run <= 2; run++)
        {
            var log = new StringWriter();
            Console.SetOut(log); // WanRequest.Trace prints the decoder stages to the console
            var clock = System.Diagnostics.Stopwatch.StartNew();
            double last = 0;
            report.AppendLine($"run {run}: 1280x704, 9 frames, Turbo q8, 4 steps, cfg 1, FewStep, shift 5, seed 42, Trace on");
            var progress = new SyncProgress(p =>
            {
                double now = clock.Elapsed.TotalSeconds;
                report.AppendLine($"  {now,8:F1} s (+{now - last,7:F1})  {p.Stage} {p.Fraction:F3} {p.Message}");
                last = now;
            });
            var video = await pipeline.GenerateAsync(new WanRequest
            {
                Prompt = "A red apple slowly rotates on a wooden table, soft morning light", NegativePrompt = null!,
                Width = 1280, Height = 704, Frames = 9, Steps = 4, GuidanceScale = 1, Shift = 5, Seed = 42, Sampler = WanSampler.FewStep,
                RgbaFrames = true, Trace = true,
            }, progress);
            Console.SetOut(console);
            report.AppendLine($"  total {clock.Elapsed.TotalSeconds:F1} s, {video.Frames} frames");
            foreach (var line in log.ToString().Split('\n').Where(l => l.Contains("[Wan]"))) report.AppendLine("    " + line.Trim());
        }
        var path = Path.Combine(TestContext.CurrentContext.WorkDirectory, "wan_timing.txt");
        File.WriteAllText(path, report.ToString());
        TestContext.Out.WriteLine(report.ToString());
    }

    [Test, Explicit("Needs SPAWNGEN_MODELS, a GPU and the Wan Turbo files. One process: a warm-up run, then untiled and tiled decode, both warm and traced. Writes wan_decode_ab.txt (env WAN_ATTN_GEMM as set).")]
    public async Task Wan_turbo_warm_untiled_vs_tiled_decode_at_720p()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS is not set.");
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var pipeline = new Wan22Pipeline(runtime.Accelerator, runtime.ModelSource, Wan22Pipeline.DefaultRepo, Wan22Paths.InRepo(Wan22Paths.DitFile("turbo_q8")));
        var report = new System.Text.StringBuilder();
        report.AppendLine($"device: {runtime.Accelerator.Name} ({runtime.Accelerator.AcceleratorType}); WAN_ATTN_GEMM={Environment.GetEnvironmentVariable("WAN_ATTN_GEMM") ?? "(unset)"}");
        var console = Console.Out;
        foreach (var (label, tiled, trace) in new[] { ("warm-up", false, false), ("untiled", false, true), ("tiled", true, true) })
        {
            var log = new StringWriter();
            Console.SetOut(log);
            var clock = System.Diagnostics.Stopwatch.StartNew();
            double decodeStart = 0;
            var progress = new SyncProgress(p => { if (decodeStart == 0 && p.Stage.ToString().Contains("ecod")) decodeStart = clock.Elapsed.TotalSeconds; });
            await pipeline.GenerateAsync(new WanRequest
            {
                Prompt = "A red apple slowly rotates on a wooden table, soft morning light", NegativePrompt = null!,
                Width = 1280, Height = 704, Frames = 9, Steps = 4, GuidanceScale = 1, Shift = 5, Seed = 42, Sampler = WanSampler.FewStep,
                RgbaFrames = true, Trace = trace, TiledDecode = tiled,
            }, progress);
            Console.SetOut(console);
            double total = clock.Elapsed.TotalSeconds;
            report.AppendLine($"{label}: total {total:F1} s, decode (from its first progress report) {total - decodeStart:F1} s");
            foreach (var line in log.ToString().Split('\n').Where(l => l.Contains("[Wan]"))) report.AppendLine("    " + line.Trim());
        }
        var path = Path.Combine(TestContext.CurrentContext.WorkDirectory, $"wan_decode_ab_gemm{Environment.GetEnvironmentVariable("WAN_ATTN_GEMM") ?? "default"}.txt");
        File.WriteAllText(path, report.ToString());
        TestContext.Out.WriteLine(report.ToString());
    }

    /// <summary>An IProgress that runs its callback on the reporting thread (Progress&lt;T&gt; posts to the thread pool, which would blur the times).</summary>
    sealed class SyncProgress(Action<WanProgress> report) : IProgress<WanProgress>
    {
        public void Report(WanProgress value) => report(value);
    }

    static double Mean(float[] a, int start, int count) { double s = 0; for (int i = start; i < start + count; i++) s += a[i]; return s / count; }

    static double Std(float[] a, int start, int count)
    {
        double m = Mean(a, start, count), s = 0;
        for (int i = start; i < start + count; i++) s += (a[i] - m) * (a[i] - m);
        return Math.Sqrt(s / count);
    }
}
