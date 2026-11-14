using ILGPU;
using ILGPU.Runtime.CPU;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>ComfyUI's VIDEO: Create Video / Get Video Components, a combo whose list holds numbers, and Save Animated WEBP's encoding.</summary>
public class VideoNodeTests
{
    static ImageBatch Frames(int n, int w, int h)
    {
        var b = new ImageBatch(n, h, w, 3);
        for (int i = 0; i < b.Data.Length; i++) b.Data[i] = (i * 37 % 256) / 255f;
        return b;
    }

    [Test]
    public void Create_Video_and_Get_Video_Components_round_trip()
    {
        var frames = Frames(5, 8, 6);
        var tone = AudioClip.Mono(new float[4800], 48000);
        var video = new CreateVideo().Execute(frames, 24, tone, "10", "HDR");
        Assert.That((video.Fps, video.Seconds), Is.EqualTo((24.0, 5 / 24.0)));
        var (images, audio, fps, bitDepth, colorSpace) = new GetVideoComponents().Execute(video);
        Assert.That(images, Is.SameAs(frames));
        Assert.That(audio, Is.SameAs(tone));
        Assert.That((fps, bitDepth, colorSpace), Is.EqualTo((24.0, "10", "HDR")));
        Assert.That(new GetVideoComponents().Execute(new CreateVideo().Execute(frames)).Audio, Is.Null, "no sound: none");
        Assert.Throws<ArgumentOutOfRangeException>(() => new VideoClip(frames, 0));
    }

    [Test]
    public async Task A_combo_whose_list_holds_numbers_takes_a_number_as_ComfyUI_does()
    {
        // ComfyUI's Create Video lists bit_depth as ["auto", 8, 10]: a workflow sends the number 8.
        var registry = new NodeRegistry().AddAssembly(typeof(CreateVideo).Assembly);
        var graph = PromptJson.Parse("""
            { "1": { "class_type": "EmptyImage", "inputs": { "width": 8, "height": 8, "batch_size": 2, "color": 0 } },
              "2": { "class_type": "CreateVideo", "inputs": { "images": ["1", 0], "fps": 12.0, "bit_depth": 8 } },
              "3": { "class_type": "GetVideoComponents", "inputs": { "video": ["2", 0] } },
              "4": { "class_type": "PreviewAny", "inputs": { "source": ["3", 3] } } }
            """);
        Assert.That(PromptValidator.Validate(registry, graph).Errors, Is.Empty);
        var result = await new PromptExecutor(registry).ExecuteAsync(graph);
        Assert.That(result.Outputs["4"].Ui!["text"], Is.EqualTo(new object[] { "8" }));
        var wrong = PromptJson.Parse("""
            { "1": { "class_type": "EmptyImage", "inputs": { "width": 8, "height": 8, "batch_size": 2, "color": 0 } },
              "2": { "class_type": "CreateVideo", "inputs": { "images": ["1", 0], "fps": 12.0, "bit_depth": 12 } },
              "3": { "class_type": "GetVideoComponents", "inputs": { "video": ["2", 0] } },
              "4": { "class_type": "PreviewAny", "inputs": { "source": ["3", 3] } } }
            """);
        Assert.That(PromptValidator.Validate(registry, wrong).Errors.Single().Message, Does.Contain("'12' is not one of"));
    }

    [Test]
    public async Task Save_Animated_WEBP_writes_every_frame_losslessly()
    {
        using var context = Context.Create(b => b.CPU());
        using var accelerator = context.CreateCPUAccelerator(0);
        var frames = Frames(3, 16, 12);
        var webp = await SaveAnimatedWEBP.EncodeAsync(frames, 24, 80, accelerator);
        Assert.That(System.Text.Encoding.ASCII.GetString(webp, 8, 4), Is.EqualTo("WEBP"));
        var decoded = (await CodecsImages.DecodeAsync(accelerator, webp))!;
        Assert.That((decoded.Width, decoded.Height, decoded.Rgba.Count), Is.EqualTo((16, 12, 3)));
        for (int f = 0; f < 3; f++)
            for (int i = 0; i < 16 * 12; i++)
                for (int c = 0; c < 3; c++)
                    Assert.That(decoded.Rgba[f][i * 4 + c], Is.EqualTo((byte)Math.Round(frames.Data[(f * 16 * 12 + i) * 3 + c] * 255)), $"frame {f} pixel {i}");
    }

    /// <summary>
    /// How long Save Animated WEBP's GPU encode takes for a video model's clip (25 frames, 512x352) on this device, twice (the second without
    /// first compiles). SPAWNGEN_WEBP_FRAMES may name a folder of PNG frames (a real clip); otherwise the frames are a moving gradient with grain.
    /// </summary>
    [Test, Explicit("Needs SPAWNGEN_MODELS_ROOT and a GPU.")]
    public async Task Save_Animated_WEBP_encode_timing_for_a_clip()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        ImageBatch frames;
        var folder = Environment.GetEnvironmentVariable("SPAWNGEN_WEBP_FRAMES");
        string source;
        if (!string.IsNullOrEmpty(folder))
        {
            var files = Directory.GetFiles(folder, "*.png").Order().Take(25).ToArray();
            var first = PngReader.Decode(File.ReadAllBytes(files[0]));
            frames = new ImageBatch(files.Length, first.Height, first.Width, 3);
            for (int f = 0; f < files.Length; f++)
                Array.Copy(PngReader.Decode(File.ReadAllBytes(files[f])).Data, 0, frames.Data, (long)f * first.Height * first.Width * 3, first.Height * first.Width * 3);
            source = $"{files.Length} PNG frames";
        }
        else
        {
            frames = new ImageBatch(25, 352, 512, 3);
            var random = new Random(1);
            for (int f = 0; f < 25; f++)
                for (int y = 0; y < 352; y++)
                    for (int x = 0; x < 512; x++)
                        for (int c = 0; c < 3; c++)
                            frames.Data[(((long)f * 352 + y) * 512 + x) * 3 + c] = Math.Clamp(((x + 4 * f) * (c + 1) % 512) / 511f + (float)(random.NextDouble() - 0.5) * 0.04f, 0, 1);
            source = "synthetic frames";
        }
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var report = new System.Text.StringBuilder();
        report.AppendLine($"device: {runtime.Accelerator.Name} ({runtime.Accelerator.AcceleratorType}); {source}, {frames.Batch} x {frames.Width}x{frames.Height}");
        for (int run = 1; run <= 2; run++)
        {
            var clock = System.Diagnostics.Stopwatch.StartNew();
            var webp = await SaveAnimatedWEBP.EncodeAsync(frames, 24, 80, runtime.Accelerator);
            report.AppendLine($"run {run}: {clock.Elapsed.TotalSeconds:F1} s, {webp.Length} bytes");
        }
        File.WriteAllText(Path.Combine(TestContext.CurrentContext.WorkDirectory, "webp_timing.txt"), report.ToString());
        TestContext.Out.WriteLine(report.ToString());
    }
}
