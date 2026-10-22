using SpawnGen.Engine.Graph;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>Other checkpoints than SD-Turbo through the same nodes (explicit: GPU and a multi-GB download on first use).</summary>
public class CheckpointTests
{
    /// <summary>The picture side; SD15_SIZE overrides it (256 is what the browser check uses).</summary>
    static int Size => int.TryParse(Environment.GetEnvironmentVariable("SD15_SIZE"), out var v) ? v : 512;

    [Test]
    public void Known_checkpoints_have_details_for_the_loader()
    {
        foreach (var name in MlRuntime.KnownCheckpoints.Keys)
            Assert.That(MlRuntime.CheckpointDetails.ContainsKey(name), name);
        Assert.That(MlRuntime.CheckpointDetails["sd_v1-5"].TextWidth, Is.EqualTo(768));
    }

    [Test, Explicit("Needs SPAWNGEN_MODELS, a GPU with 5+ GB free and a 4.4 GB download on first use. Look at the saved picture.")]
    public async Task Stable_diffusion_1_5_draws_an_apple_with_real_guidance()
    {
        var files = await VisionTests.RunAsync("sd15", () =>
        {
            var g = new PromptGraph();
            g.Add("1", "CheckpointLoaderSimple", ("ckpt_name", "sd_v1-5"));
            g.Add("2", "CLIPTextEncode", ("text", "a photo of a red apple on a wooden table"), ("clip", PromptInput.Link("1", 1)));
            g.Add("3", "CLIPTextEncode", ("text", "blurry, low quality"), ("clip", PromptInput.Link("1", 1)));
            g.Add("4", "EmptyLatentImage", ("width", Size), ("height", Size), ("batch_size", 1));
            g.Add("5", "KSampler", ("model", PromptInput.Link("1")), ("positive", PromptInput.Link("2")), ("negative", PromptInput.Link("3")), ("latent_image", PromptInput.Link("4")),
                ("seed", 7), ("steps", 8), ("cfg", 7.0), ("sampler_name", "dpmpp_2m"), ("scheduler", "karras"), ("denoise", 1.0));
            g.Add("6", "VAEDecode", ("samples", PromptInput.Link("5")), ("vae", PromptInput.Link("1", 2)));
            g.Add("7", "SaveImage", ("images", PromptInput.Link("6")), ("filename_prefix", "sd15"));
            return g;
        });
        var png = PngReader.Decode(File.ReadAllBytes(files.Single()));
        var values = png.Data.ToArray();
        Assert.That(values.Max() - values.Min(), Is.GreaterThan(0.3f));
    }

    [Test, Explicit("Needs SPAWNGEN_MODELS and a GPU. Look at the saved pictures: the weighted word must change the picture the way its weight says.")]
    public async Task Prompt_weights_change_the_picture()
    {
        async Task<(string File, double GreenOverRed)> Draw(string prompt, string name)
        {
            var files = await VisionTests.RunAsync(name, () =>
            {
                var g = new PromptGraph();
                g.Add("1", "CheckpointLoaderSimple", ("ckpt_name", "sd_turbo"));
                g.Add("2", "CLIPTextEncode", ("text", prompt), ("clip", PromptInput.Link("1", 1)));
                g.Add("3", "CLIPTextEncode", ("text", ""), ("clip", PromptInput.Link("1", 1)));
                g.Add("4", "EmptyLatentImage", ("width", 512), ("height", 512), ("batch_size", 1));
                g.Add("5", "KSampler", ("model", PromptInput.Link("1")), ("positive", PromptInput.Link("2")), ("negative", PromptInput.Link("3")), ("latent_image", PromptInput.Link("4")),
                    ("seed", 7), ("steps", 1), ("cfg", 1.0), ("sampler_name", "euler"), ("scheduler", "normal"), ("denoise", 1.0));
                g.Add("6", "VAEDecode", ("samples", PromptInput.Link("5")), ("vae", PromptInput.Link("1", 2)));
                g.Add("7", "SaveImage", ("images", PromptInput.Link("6")), ("filename_prefix", name));
                return g;
            });
            var png = PngReader.Decode(File.ReadAllBytes(files.Single()));
            double r = 0, gr = 0;
            for (int i = 0; i < png.Width * png.Height; i++) { r += png.Data[i * 3]; gr += png.Data[i * 3 + 1]; }
            return (files.Single(), gr / r);
        }

        var weak = await Draw("a photo of a (green:0.2) apple on a wooden table", "weak");
        var strong = await Draw("a photo of a (green:1.8) apple on a wooden table", "strong");
        TestContext.Out.WriteLine($"green/red: weight 0.2 -> {weak.GreenOverRed:F3} ({weak.File}), weight 1.8 -> {strong.GreenOverRed:F3} ({strong.File})");
        Assert.That(strong.GreenOverRed, Is.GreaterThan(weak.GreenOverRed), "a higher weight on green must make the picture greener");
    }
}
