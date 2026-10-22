using SpawnGen.Engine.Graph;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>Whole recipes that ComfyUI users build from core nodes, run end to end (explicit: GPU).</summary>
public class WorkflowRecipeTests
{
    [Test, Explicit("Needs SPAWNGEN_MODELS and a GPU. Look at the picture: the 256 px sample upscaled in latent space and refined at 512 px.")]
    public async Task Hires_fix_draws_small_upscales_the_latent_and_refines()
    {
        var files = await VisionTests.RunAsync("hires", () =>
        {
            var g = new PromptGraph();
            g.Add("1", "CheckpointLoaderSimple", ("ckpt_name", "sd_turbo"));
            g.Add("2", "CLIPTextEncode", ("text", "a photo of a red apple on a wooden table"), ("clip", PromptInput.Link("1", 1)));
            g.Add("3", "CLIPTextEncode", ("text", ""), ("clip", PromptInput.Link("1", 1)));
            g.Add("4", "EmptyLatentImage", ("width", 256), ("height", 256), ("batch_size", 1));
            g.Add("5", "KSampler", ("model", PromptInput.Link("1")), ("positive", PromptInput.Link("2")), ("negative", PromptInput.Link("3")), ("latent_image", PromptInput.Link("4")),
                ("seed", 7), ("steps", 1), ("cfg", 1.0), ("sampler_name", "euler"), ("scheduler", "normal"), ("denoise", 1.0));
            g.Add("6", "LatentUpscaleBy", ("samples", PromptInput.Link("5")), ("upscale_method", "bilinear"), ("scale_by", 2.0));
            g.Add("7", "KSampler", ("model", PromptInput.Link("1")), ("positive", PromptInput.Link("2")), ("negative", PromptInput.Link("3")), ("latent_image", PromptInput.Link("6")),
                ("seed", 8), ("steps", 2), ("cfg", 1.0), ("sampler_name", "euler"), ("scheduler", "normal"), ("denoise", 0.5));
            g.Add("8", "VAEDecode", ("samples", PromptInput.Link("7")), ("vae", PromptInput.Link("1", 2)));
            g.Add("9", "SaveImage", ("images", PromptInput.Link("8")), ("filename_prefix", "hires"));
            return g;
        });
        var png = PngReader.Decode(File.ReadAllBytes(files.Single()));
        Assert.That((png.Width, png.Height), Is.EqualTo((512, 512)));
        Assert.That(png.Data.Max() - png.Data.Min(), Is.GreaterThan(0.3f));
        TestContext.Out.WriteLine(files.Single());
    }
}
