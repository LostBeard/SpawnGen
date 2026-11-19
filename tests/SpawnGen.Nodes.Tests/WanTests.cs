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

        var latent = video.Latent ?? throw new AssertionException("the video carries no latent");
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

    /// <summary>The Wan authors' negative prompt (the library's DefaultNegativePrompt) in English, item for item.</summary>
    internal const string EnglishNegative = "oversaturated colors, overexposed, static, blurry details, subtitles, style, artwork, painting, frame, still, " +
        "overall gray, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, " +
        "deformed, disfigured, malformed limbs, fused fingers, motionless frame, cluttered background, three legs, many people in the background, walking backwards";

    /// <summary>
    /// Whether the Wan 2.2 Video node's default negative prompt can be English (TJ, 2026-11-19: the Chinese default is out of place in the UI;
    /// "if changing the language does not hurt it do it"). Same seed, prompt and settings with the reference's Chinese list, its English
    /// translation, and no negative (how much the negative matters at all). Writes wan_negative.txt (mean |pixel difference| between the runs)
    /// and the frames (wan_negative/<run>_<frame>.png) to look at.
    /// </summary>
    [Test, Explicit("Needs SPAWNGEN_MODELS, a GPU and the Wan 2.2 5B files. Three 20-step runs at 832x480 x 9 frames.")]
    public async Task An_English_negative_prompt_works_like_the_reference_Chinese_one()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS is not set.");
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var pipeline = new Wan22Pipeline(runtime.Accelerator, runtime.ModelSource, Wan22Pipeline.DefaultRepo, Wan22Paths.InRepo());
        var outDir = Path.Combine(TestContext.CurrentContext.WorkDirectory, "wan_negative");
        Directory.CreateDirectory(outDir);
        var runs = new List<(string Name, byte[][] Frames, double Seconds)>();
        foreach (var (name, negative) in new[] { ("chinese", Wan22Pipeline.DefaultNegativePrompt), ("english", EnglishNegative), ("none", "") })
        {
            var started = DateTime.Now;
            var video = await pipeline.GenerateAsync(new WanRequest
            {
                Prompt = "A young woman in a yellow raincoat walks along a beach at sunset and waves at the camera with her right hand, waves rolling in behind her",
                NegativePrompt = negative, Width = 832, Height = 480, Frames = 9, Steps = 20, GuidanceScale = 5, Shift = 5, Seed = 42,
                Sampler = WanSampler.UniPC, RgbaFrames = true,
            });
            var frames = Enumerable.Range(0, video.Frames).Select(video.GetFrameRgba).ToArray();
            for (int f = 0; f < frames.Length; f++)
                File.WriteAllBytes(Path.Combine(outDir, $"{name}_{f}.png"), SpawnGen.Nodes.Core.PngWriter.Encode(Rgb(frames[f], video.Width, video.Height)));
            runs.Add((name, frames, (DateTime.Now - started).TotalSeconds));
        }
        double Diff(byte[][] a, byte[][] b) => a.Zip(b).Average(p => Enumerable.Range(0, p.First.Length / 4).Average(i =>
            (Math.Abs(p.First[i * 4] - p.Second[i * 4]) + Math.Abs(p.First[i * 4 + 1] - p.Second[i * 4 + 1]) + Math.Abs(p.First[i * 4 + 2] - p.Second[i * 4 + 2])) / 3.0));
        var report = new System.Text.StringBuilder();
        report.AppendLine($"device: {runtime.Accelerator.Name}; 832x480 x 9, 20 steps, cfg 5, shift 5, seed 42, UniPC; seconds " + string.Join(", ", runs.Select(r => $"{r.Name} {r.Seconds:F0}")));
        report.AppendLine($"mean |pixel difference| (0-255): chinese vs english {Diff(runs[0].Frames, runs[1].Frames):F2}; chinese vs none {Diff(runs[0].Frames, runs[2].Frames):F2}; english vs none {Diff(runs[1].Frames, runs[2].Frames):F2}");
        File.WriteAllText(Path.Combine(TestContext.CurrentContext.WorkDirectory, "wan_negative.txt"), report.ToString());
        TestContext.Out.WriteLine(report.ToString() + outDir);
    }

    /// <summary>
    /// Follow-up to the test above (2026-11-19, RX 590: the English list broke the seed 42 video into colour noise while the Chinese list and no negative stayed clean):
    /// whether English negatives break Wan 2.2 5B in general. The full English list at two more seeds, and a short English negative at seed 42. Writes wan_negative2/*.png.
    /// </summary>
    [Test, Explicit("Needs SPAWNGEN_MODELS, a GPU and the Wan 2.2 5B files. Three 20-step runs at 832x480 x 9 frames.")]
    public async Task English_negative_prompts_across_seeds()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS is not set.");
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var pipeline = new Wan22Pipeline(runtime.Accelerator, runtime.ModelSource, Wan22Pipeline.DefaultRepo, Wan22Paths.InRepo());
        var outDir = Path.Combine(TestContext.CurrentContext.WorkDirectory, "wan_negative2");
        Directory.CreateDirectory(outDir);
        foreach (var (name, negative, seed) in new[] { ("english_s7", EnglishNegative, 7L), ("english_s123", EnglishNegative, 123L), ("short_s42", "blurry, low quality, static, deformed", 42L) })
        {
            var video = await pipeline.GenerateAsync(new WanRequest
            {
                Prompt = "A young woman in a yellow raincoat walks along a beach at sunset and waves at the camera with her right hand, waves rolling in behind her",
                NegativePrompt = negative, Width = 832, Height = 480, Frames = 9, Steps = 20, GuidanceScale = 5, Shift = 5, Seed = seed,
                Sampler = WanSampler.UniPC, RgbaFrames = true,
            });
            for (int f = 0; f < video.Frames; f++)
                File.WriteAllBytes(Path.Combine(outDir, $"{name}_{f}.png"), SpawnGen.Nodes.Core.PngWriter.Encode(Rgb(video.GetFrameRgba(f), video.Width, video.Height)));
        }
        TestContext.Out.WriteLine(outDir);
    }

    /// <summary>
    /// Bisects which part of the English negative breaks the video (seed 42, 20 steps; WAN_W / WAN_H / WAN_FRAMES / WAN_SHIFT, default 512x288 x 5, shift 5: too small for
    /// Wan 5B, every negative breaks there, so bisect at 832x480 or the node's 1280x704 / shift 8). Negatives from WAN_NEGATIVES ("name=text" separated by '|').
    /// Writes wan_bisect/<name>_<frame>.png and wan_bisect.txt: each run's mean |Laplacian| (high-frequency energy; the broken runs are colour noise, so theirs is much higher).
    /// </summary>
    [Test, Explicit("Needs SPAWNGEN_MODELS, WAN_NEGATIVES, a GPU and the Wan 2.2 5B files.")]
    public async Task Bisect_the_English_negative()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        var spec = Environment.GetEnvironmentVariable("WAN_NEGATIVES");
        if (string.IsNullOrEmpty(root) || string.IsNullOrEmpty(spec)) Assert.Ignore("SPAWNGEN_MODELS or WAN_NEGATIVES is not set.");
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var pipeline = new Wan22Pipeline(runtime.Accelerator, runtime.ModelSource, Wan22Pipeline.DefaultRepo, Wan22Paths.InRepo());
        var outDir = Path.Combine(TestContext.CurrentContext.WorkDirectory, "wan_bisect");
        Directory.CreateDirectory(outDir);
        var report = new System.Text.StringBuilder();
        foreach (var item in spec!.Split('|'))
        {
            var name = item[..item.IndexOf('=')]; var negative = item[(item.IndexOf('=') + 1)..];
            var video = await pipeline.GenerateAsync(new WanRequest
            {
                Prompt = "A young woman in a yellow raincoat walks along a beach at sunset and waves at the camera with her right hand, waves rolling in behind her",
                NegativePrompt = negative, Width = Env("WAN_W", 512), Height = Env("WAN_H", 288), Frames = Env("WAN_FRAMES", 5), Steps = 20, GuidanceScale = 5, Shift = Env("WAN_SHIFT", 5), Seed = 42,
                Sampler = WanSampler.UniPC, RgbaFrames = true,
            });
            double lap = 0;
            for (int f = 0; f < video.Frames; f++)
            {
                var rgba = video.GetFrameRgba(f); int w = video.Width, h = video.Height;
                File.WriteAllBytes(Path.Combine(outDir, $"{name}_{f}.png"), SpawnGen.Nodes.Core.PngWriter.Encode(Rgb(rgba, w, h)));
                double sum = 0;
                for (int y = 1; y < h - 1; y++)
                    for (int x = 1; x < w - 1; x++)
                        for (int c = 0; c < 3; c++)
                        {
                            int i = (y * w + x) * 4 + c;
                            sum += Math.Abs(4 * rgba[i] - rgba[i - 4] - rgba[i + 4] - rgba[i - w * 4] - rgba[i + w * 4]);
                        }
                lap += sum / ((w - 2) * (h - 2) * 3.0) / video.Frames;
            }
            report.AppendLine($"{name}: mean |Laplacian| {lap:F2}  [{negative}]");
            File.WriteAllText(Path.Combine(TestContext.CurrentContext.WorkDirectory, "wan_bisect.txt"), report.ToString());
        }
        TestContext.Out.WriteLine(report.ToString());
    }

    static int Env(string name, int fallback) => int.TryParse(Environment.GetEnvironmentVariable(name), out var v) ? v : fallback;

    static SpawnGen.Engine.Types.ImageBatch Rgb(byte[] rgba, int width, int height)
    {
        var image = new SpawnGen.Engine.Types.ImageBatch(1, height, width, 3);
        for (int i = 0; i < width * height; i++)
            for (int c = 0; c < 3; c++) image.Data[i * 3 + c] = rgba[i * 4 + c] / 255f;
        return image;
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
