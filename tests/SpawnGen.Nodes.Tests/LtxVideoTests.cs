using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>The LTX-Video 2B node: its schema, its input checks, and (explicit) a real clip on the GPU.</summary>
public class LtxVideoTests
{
    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }

    static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(LtxVideo).Assembly);

    [Test]
    public void The_node_takes_a_prompt_and_optional_start_and_end_pictures()
    {
        var schema = Registry().GetSchema("LtxVideo")!;
        Assert.That(schema.DisplayName, Is.EqualTo("LTX-Video 2B (0.9.8 distilled)"));
        Assert.That(schema.Outputs.Select(o => (o.Name, o.Type.Name)), Is.EqualTo(new[] { ("images", "IMAGE"), ("fps", "FLOAT") }));
        Assert.That(schema.FindInput("start_image")!.Required, Is.False);
        Assert.That(schema.FindInput("end_image")!.Required, Is.False);
        Assert.That(schema.FindInput("length")!.Widget!.Default, Is.EqualTo(121L));
        Assert.That(schema.FindInput("precision")!.Widget!.ResolveOptions(), Is.EqualTo(new[] { "q8", "bf16" }));
    }

    [TestCase(700, 480, 121, "multiples of 32")]
    [TestCase(704, 480, 120, "8n+1")]
    public void Sizes_and_lengths_the_model_cannot_make_are_refused(int width, int height, int length, string reason)
    {
        var error = Assert.ThrowsAsync<ArgumentException>(() => new LtxVideo().ExecuteAsync(null!, new Services(), "a cat", width, height, length));
        Assert.That(error!.Message, Does.Contain(reason));
    }

    /// <summary>A short clip through a real prompt (LTX-Video -> Save Animated WEBP); look at the saved file. Writes ltx_run.txt (timings).</summary>
    [Test, Explicit("Needs SPAWNGEN_MODELS_ROOT (the models folder), a GPU and the LTX-Video files (~8 GB, downloaded on first use).")]
    public async Task A_clip_is_made_on_the_GPU()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        var outDir = Path.Combine(TestContext.CurrentContext.WorkDirectory, "ltx_out");
        Directory.CreateDirectory(outDir);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var executor = new PromptExecutor(Registry(), new Services(runtime, new OutputOptions { Directory = outDir, TempDirectory = outDir }));
        var g = new PromptGraph();
        g.Add("1", "LtxVideo", ("prompt", "A red apple slowly rotates on a wooden table, soft morning light, the camera stays still"),
            ("width", 512L), ("height", 352L), ("length", 25L), ("frame_rate", 24.0), ("seed", 42L), ("upscale", false), ("precision", "q8"));
        g.Add("2", "SaveAnimatedWEBP", ("images", PromptInput.Link("1")), ("fps", 24.0), ("filename_prefix", "ltx"), ("lossless", true),
            ("quality", 80L), ("method", "default"));
        g.Add("3", "SaveImage", ("images", PromptInput.Link("1")), ("filename_prefix", "ltx_frame"));
        var clock = System.Diagnostics.Stopwatch.StartNew();
        await executor.ExecuteAsync(g);
        var cold = clock.Elapsed.TotalSeconds;
        g.Add("1", "LtxVideo", ("prompt", "A red apple slowly rotates on a wooden table, soft morning light, the camera stays still"),
            ("width", 512L), ("height", 352L), ("length", 25L), ("frame_rate", 24.0), ("seed", 43L), ("upscale", false), ("precision", "q8"));
        clock.Restart();
        await executor.ExecuteAsync(g);
        var warm = clock.Elapsed.TotalSeconds;
        var report = $"device {runtime.Accelerator.Name} ({runtime.Accelerator.AcceleratorType}); 512x352, 25 frames, q8: cold {cold:F1} s, warm (another seed) {warm:F1} s";
        File.WriteAllText(Path.Combine(TestContext.CurrentContext.WorkDirectory, "ltx_run.txt"), report);
        TestContext.Out.WriteLine(report + "\n" + outDir);
        Assert.That(Directory.GetFiles(outDir, "ltx_*.webp"), Is.Not.Empty);
        Assert.That(Directory.GetFiles(outDir, "ltx_frame_*.png"), Has.Length.EqualTo(50), "25 frames, two runs");
    }
}
