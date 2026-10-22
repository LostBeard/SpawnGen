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
}
