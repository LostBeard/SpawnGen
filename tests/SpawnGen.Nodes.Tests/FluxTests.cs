using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>FLUX.2 Klein as a node: the graph validates, and (explicit, GPU + a 4.6 GB download on first use) a real picture is drawn and looked at.</summary>
public class FluxTests
{
    static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly);

    [Test]
    public void A_flux_graph_validates()
    {
        var g = new PromptGraph();
        g.Add("1", "Flux2Klein", ("prompt", "a red apple"), ("width", 512), ("height", 512), ("seed", 1));
        g.Add("2", "SaveImage", ("images", PromptInput.Link("1")), ("filename_prefix", "flux"));
        var result = new PromptExecutor(Registry()).Validate(g);
        Assert.That(result.Errors, Is.Empty, string.Join("\n", result.Errors.Select(e => e.Message)));
        Assert.That(Flux2Klein.Weights(), Is.EqualTo(new[] { "4-bit", "8-bit" }));
    }

    [Test, Explicit("Needs SPAWNGEN_MODELS, a GPU with 4+ GB free, and a 4.6 GB download on first use. Look at the saved picture.")]
    public async Task Flux2_klein_draws_a_picture_from_a_prompt()
    {
        var files = await VisionTests.RunAsync("flux2", () =>
        {
            var g = new PromptGraph();
            g.Add("1", "Flux2Klein", ("prompt", "a photo of a single red apple on a wooden table, soft daylight"), ("width", 512), ("height", 512), ("seed", 7));
            g.Add("2", "SaveImage", ("images", PromptInput.Link("1")), ("filename_prefix", "flux"));
            return g;
        });
        var png = PngReader.Decode(File.ReadAllBytes(files.Single()));
        Assert.That((png.Width, png.Height), Is.EqualTo((512, 512)));
        var values = png.Data.Take(png.Width * png.Height * png.Channels).ToArray();
        TestContext.Out.WriteLine($"mean {values.Average():F2}");
        Assert.That(values.Max() - values.Min(), Is.GreaterThan(0.3f));
    }

    [Test]
    public void A_flux3_graph_validates()
    {
        var g = new PromptGraph();
        g.Add("1", "Flux3Klein", ("caption", "A red apple on a wooden table."), ("width", 512), ("height", 512), ("steps", 16), ("seed", 1));
        g.Add("2", "SaveImage", ("images", PromptInput.Link("1")), ("filename_prefix", "flux3"));
        var result = new PromptExecutor(Registry()).Validate(g);
        Assert.That(result.Errors, Is.Empty, string.Join("; ", result.Errors.Select(e => e.Message)));
    }

    [Test, Explicit("Needs SPAWNGEN_MODELS, a GPU with 5+ GB free and a 6 GB download on first use. Look at the saved picture.")]
    public async Task Flux3_klein_draws_a_picture_from_a_caption()
    {
        var files = await VisionTests.RunAsync("flux3", () =>
        {
            var g = new PromptGraph();
            g.Add("1", "Flux3Klein", ("caption", "A photograph of a single glossy red apple resting on a weathered wooden table, soft window light from the left, shallow depth of field, a blurred cream-coloured wall behind."), ("width", 512), ("height", 512), ("steps", 16), ("seed", 7));
            g.Add("2", "SaveImage", ("images", PromptInput.Link("1")), ("filename_prefix", "flux3"));
            return g;
        });
        var png = PngReader.Decode(File.ReadAllBytes(files.Single()));
        Assert.That((png.Width, png.Height), Is.EqualTo((512, 512)));
        Assert.That(png.Data.Max() - png.Data.Min(), Is.GreaterThan(0.3f));
    }

    [Test, Explicit("Needs SPAWNGEN_MODELS, a GPU and the FLUX files in the hub cache. Prints where the time goes (the library's own stage timings), twice: the second picture reuses the resident transformer.")]
    public async Task Flux_stage_timings_for_the_library_team()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS is not set.");
        using var runtime = new MlRuntime(new DesktopModelSource(Path.GetDirectoryName(Path.TrimEndingDirectorySeparator(root!))!, MlRuntime.RepoFolders));
        var acc = runtime.Accelerator;
        TestContext.Out.WriteLine($"{acc.AcceleratorType} {acc.Name}");
        var flux3 = await SpawnDev.ILGPU.ML.Flux3.Flux3Pipeline.CreateAsync(acc, runtime.ModelSource, "LostBeard/flux-3-image-klein-q4-ILGPU");
        flux3.KeepTransformerResident = true;
        for (int run = 1; run <= 2; run++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            await flux3.RunAsync(new SpawnDev.ILGPU.ML.Flux3.Flux3Request { Caption = "A red apple on a wooden table.", Seed = run, Width = 512, Height = 512, Steps = 16 });
            TestContext.Out.WriteLine($"FLUX 3 Klein 512x512 16 steps, picture {run}: {sw.Elapsed.TotalSeconds:F1} s: " + string.Join(", ", flux3.LastTimings.Select(t => $"{t.Stage} {t.Seconds:F1}")));
        }
        flux3.Dispose();
        var flux2 = await SpawnDev.ILGPU.ML.Flux2.Flux2Pipeline.CreateAsync(acc, runtime.ModelSource, "LostBeard/FLUX.2-klein-4B-q4-ILGPU");
        flux2.KeepTransformerResident = true;
        for (int run = 1; run <= 2; run++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            await flux2.RunAsync(new SpawnDev.ILGPU.ML.Flux2.Flux2Request { Prompt = "a red apple on a wooden table", Seed = run, Size = 512, Steps = 4 });
            TestContext.Out.WriteLine($"FLUX.2 klein 512x512 4 steps, picture {run}: {sw.Elapsed.TotalSeconds:F1} s");
        }
        flux2.Dispose();
    }
}
