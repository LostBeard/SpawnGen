using SpawnGen.Engine.Graph;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

public class LlmTests
{
    [Test]
    public void The_model_list_has_repos_and_files_for_each_name()
    {
        Assert.That(TextGeneration.ModelNames(), Does.Contain("qwen2.5-0.5b-instruct"));
        foreach (var (name, m) in TextGeneration.Models)
        {
            Assert.That(m.Repo, Does.Contain("/"), name);
            Assert.That(m.File, Does.EndWith(".gguf"), name);
        }
    }

    [Test, Explicit("Needs SPAWNGEN_MODELS, a GPU and downloads on first use (Qwen2.5 0.5B 530 MB). Look at the picture and the text.")]
    public async Task A_language_model_writes_the_prompt_that_draws_the_picture()
    {
        var files = await VisionTests.RunAsync("llm", () =>
        {
            var g = new PromptGraph();
            g.Add("1", "TextGeneration", ("prompt", "Write one vivid sentence describing a single red apple on a wooden table, for an image generator. Answer with the sentence only."),
                ("system_prompt", "You write short prompts for image generators."), ("model", "qwen2.5-0.5b-instruct"), ("max_new_tokens", 60), ("temperature", 0.0));
            g.Add("2", "CheckpointLoaderSimple", ("ckpt_name", "sd_turbo"));
            g.Add("3", "CLIPTextEncode", ("text", PromptInput.Link("1")), ("clip", PromptInput.Link("2", 1)));
            g.Add("4", "CLIPTextEncode", ("text", ""), ("clip", PromptInput.Link("2", 1)));
            g.Add("5", "EmptyLatentImage", ("width", 512), ("height", 512), ("batch_size", 1));
            g.Add("6", "KSampler", ("model", PromptInput.Link("2")), ("positive", PromptInput.Link("3")), ("negative", PromptInput.Link("4")), ("latent_image", PromptInput.Link("5")),
                ("seed", 7), ("steps", 1), ("cfg", 1.0), ("sampler_name", "euler"), ("scheduler", "normal"), ("denoise", 1.0));
            g.Add("7", "VAEDecode", ("samples", PromptInput.Link("6")), ("vae", PromptInput.Link("2", 2)));
            g.Add("8", "SaveImage", ("images", PromptInput.Link("7")), ("filename_prefix", "llm"));
            g.Add("9", "CaptureText", ("text", PromptInput.Link("1")));
            return g;
        });
        TestContext.Out.WriteLine("written: " + VisionTests.CaptureText.Last);
        TestContext.Out.WriteLine(files.Single());
        Assert.That(VisionTests.CaptureText.Last.Length, Is.InRange(20, 600));
        Assert.That(VisionTests.CaptureText.Last.ToLowerInvariant(), Does.Contain("apple"));
    }
}
