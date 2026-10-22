using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

public class PngRoundTripTests
{
    [Test]
    public void Reader_returns_what_the_writer_wrote()
    {
        var rng = new Random(3);
        var src = new ImageBatch(1, 13, 21, 3);
        for (int i = 0; i < src.Data.Length; i++) src.Data[i] = rng.Next(256) / 255f;

        var back = PngReader.Decode(PngWriter.Encode(src));

        Assert.That((back.Width, back.Height, back.Channels), Is.EqualTo((21, 13, 3)));
        for (int i = 0; i < src.Data.Length; i++) Assert.That(back.Data[i], Is.EqualTo(src.Data[i]).Within(1e-6f));
    }

    [Test]
    public void Reader_drops_alpha_from_rgba()
    {
        var rgba = new ImageBatch(1, 2, 2, 4, [1, 0, 0, 0.5f, 0, 1, 0, 0.5f, 0, 0, 1, 0.5f, 1, 1, 1, 0]);
        var back = PngReader.Decode(PngWriter.Encode(rgba));
        Assert.That(back.Channels, Is.EqualTo(3));
        Assert.That(back.Data, Is.EqualTo(new float[] { 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1 }).Within(1e-6f));
    }

    [Test]
    public void Reader_rejects_non_png_data_with_a_clear_message()
    {
        var ex = Assert.Throws<InvalidDataException>(() => PngReader.Decode([0xFF, 0xD8, 0xFF, 0xE0, 0, 0, 0, 0, 0]));
        Assert.That(ex!.Message, Does.Contain("Not a PNG"));
    }
}

/// <summary>img2img: load a picture, VAE-encode it, sample from a partly noised latent. The GPU test needs SPAWNGEN_MODELS (with vae_encoder/).</summary>
public class Img2ImgTests
{
    static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly);

    static PromptGraph Graph(string image, string prompt, int seed, int steps, double denoise)
    {
        var g = new PromptGraph();
        g.Add("1", "CheckpointLoaderSimple", ("ckpt_name", "sd_turbo"));
        g.Add("2", "CLIPTextEncode", ("text", prompt), ("clip", PromptInput.Link("1", 1)));
        g.Add("3", "CLIPTextEncode", ("text", ""), ("clip", PromptInput.Link("1", 1)));
        g.Add("4", "LoadImage", ("image", image));
        g.Add("5", "VAEEncode", ("pixels", PromptInput.Link("4")), ("vae", PromptInput.Link("1", 2)));
        g.Add("6", "KSampler", ("model", PromptInput.Link("1", 0)), ("seed", seed), ("steps", steps), ("cfg", 1.0),
            ("sampler_name", "euler"), ("scheduler", "normal"), ("positive", PromptInput.Link("2")),
            ("negative", PromptInput.Link("3")), ("latent_image", PromptInput.Link("5")), ("denoise", denoise));
        g.Add("7", "VAEDecode", ("samples", PromptInput.Link("6")), ("vae", PromptInput.Link("1", 2)));
        g.Add("8", "SaveImage", ("images", PromptInput.Link("7")), ("filename_prefix", "img2img_test"));
        return g;
    }

    [Test]
    public void The_img2img_graph_validates_against_the_registered_schemas()
    {
        var dir = Path.Combine(Path.GetTempPath(), "spawngen-in-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        File.WriteAllBytes(Path.Combine(dir, "in.png"), PngWriter.Encode(new ImageBatch(1, 8, 8, 3)));
        InputFiles.Directory = dir;

        var result = new PromptExecutor(Registry()).Validate(Graph("in.png", "a cat", 1, 2, 0.5));
        Assert.That(result.Errors, Is.Empty, string.Join("\n", result.Errors.Select(e => e.Message)));
        Assert.That(result.Plan, Has.Count.EqualTo(8));
    }

    [Test]
    public async Task Img2img_keeps_the_composition_of_the_input_and_the_image_is_checked_by_eye()
    {
        var models = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        if (string.IsNullOrEmpty(models)) Assert.Ignore("SPAWNGEN_MODELS is not set.");

        var work = Path.Combine(Path.GetTempPath(), "spawngen-img2img-" + Guid.NewGuid().ToString("N"));
        InputFiles.Directory = Path.Combine(work, "input");
        Directory.CreateDirectory(InputFiles.Directory);
        using var runtime = new MlRuntime(new FolderModelSource(models!));
        var executor = new PromptExecutor(Registry(), new Services(runtime, new OutputOptions { Directory = Path.Combine(work, "output") }));

        // 1. txt2img: the red apple, saved as the input picture.
        var first = Txt2ImgTests.Graph("a photo of a red apple on a wooden table", seed: 7, size: 512);
        await executor.ExecuteAsync(first);
        var apple = Directory.GetFiles(Path.Combine(work, "output"), "txt2img_test_*.png").Single();
        File.Copy(apple, Path.Combine(InputFiles.Directory, "apple.png"));

        // 2. img2img: turn it into a green apple. Half denoise, so the table and framing should survive.
        var r = await executor.ExecuteAsync(Graph("apple.png", "a photo of a green apple on a wooden table", seed: 11, steps: 2, denoise: 0.5));
        TestContext.Out.WriteLine($"img2img elapsed {r.Elapsed}");
        var outFile = Directory.GetFiles(Path.Combine(work, "output"), "img2img_test_*.png").Single();
        TestContext.Out.WriteLine($"input:  {Path.Combine(InputFiles.Directory, "apple.png")}");
        TestContext.Out.WriteLine($"output: {outFile}");

        // Cheap numeric guard on top of the visual check: the result must resemble the input (shared layout), not be an unrelated image.
        var a = PngReader.Decode(File.ReadAllBytes(Path.Combine(InputFiles.Directory, "apple.png")));
        var b = PngReader.Decode(File.ReadAllBytes(outFile));
        double sum = 0;
        for (int i = 0; i < a.Data.Length; i++) sum += Math.Abs(a.Data[i] - b.Data[i]);
        double meanAbs = sum / a.Data.Length;
        TestContext.Out.WriteLine($"mean abs pixel difference input vs output: {meanAbs:F3}");
        Assert.That(meanAbs, Is.LessThan(0.2), "At denoise 0.5 the output should stay close to the input.");
    }

    [Test]
    public async Task Inpainting_repaints_only_the_masked_square_and_the_image_is_checked_by_eye()
    {
        var models = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        if (string.IsNullOrEmpty(models)) Assert.Ignore("SPAWNGEN_MODELS is not set.");

        var work = Path.Combine(Path.GetTempPath(), "spawngen-inpaint-" + Guid.NewGuid().ToString("N"));
        InputFiles.Directory = Path.Combine(work, "input");
        Directory.CreateDirectory(InputFiles.Directory);
        using var runtime = new MlRuntime(new FolderModelSource(models!));
        var executor = new PromptExecutor(Registry(), new Services(runtime, new OutputOptions { Directory = Path.Combine(work, "output") }));

        await executor.ExecuteAsync(Txt2ImgTests.Graph("a photo of a red apple on a wooden table", seed: 7, size: 512));
        File.Copy(Directory.GetFiles(Path.Combine(work, "output"), "txt2img_test_*.png").Single(), Path.Combine(InputFiles.Directory, "apple.png"));

        // A white square over the middle of the apple on a black picture is the mask.
        var square = new ImageBatch(1, 512, 512, 3);
        for (int y = 160; y < 352; y++) for (int x = 160; x < 352; x++) for (int c = 0; c < 3; c++) square.Data[(y * 512 + x) * 3 + c] = 1f;
        File.WriteAllBytes(Path.Combine(InputFiles.Directory, "mask.png"), PngWriter.Encode(square));

        var g = new PromptGraph();
        g.Add("1", "CheckpointLoaderSimple", ("ckpt_name", "sd_turbo"));
        g.Add("2", "CLIPTextEncode", ("text", "a photo of a blue ceramic vase on a wooden table"), ("clip", PromptInput.Link("1", 1)));
        g.Add("3", "CLIPTextEncode", ("text", ""), ("clip", PromptInput.Link("1", 1)));
        g.Add("4", "LoadImage", ("image", "apple.png"));
        g.Add("5", "LoadImage", ("image", "mask.png"));
        g.Add("6", "ImageToMask", ("image", PromptInput.Link("5")), ("channel", "red"));
        g.Add("7", "VAEEncodeForInpaint", ("pixels", PromptInput.Link("4")), ("vae", PromptInput.Link("1", 2)), ("mask", PromptInput.Link("6")), ("grow_mask_by", 6));
        g.Add("8", "KSampler", ("model", PromptInput.Link("1", 0)), ("seed", 5), ("steps", 4), ("cfg", 1.0),
            ("sampler_name", "euler_ancestral"), ("scheduler", "normal"), ("positive", PromptInput.Link("2")),
            ("negative", PromptInput.Link("3")), ("latent_image", PromptInput.Link("7")), ("denoise", 1.0));
        g.Add("9", "VAEDecode", ("samples", PromptInput.Link("8")), ("vae", PromptInput.Link("1", 2)));
        g.Add("10", "SaveImage", ("images", PromptInput.Link("9")), ("filename_prefix", "inpaint_test"));
        await executor.ExecuteAsync(g);

        var outFile = Directory.GetFiles(Path.Combine(work, "output"), "inpaint_test_*.png").Single();
        TestContext.Out.WriteLine($"input:  {Path.Combine(InputFiles.Directory, "apple.png")}");
        TestContext.Out.WriteLine($"output: {outFile}");
        var a = PngReader.Decode(File.ReadAllBytes(Path.Combine(InputFiles.Directory, "apple.png")));
        var b = PngReader.Decode(File.ReadAllBytes(outFile));
        double Diff(int x0, int x1, int y0, int y1)
        {
            double sum = 0; int n = 0;
            for (int y = y0; y < y1; y++) for (int x = x0; x < x1; x++) for (int c = 0; c < 3; c++) { sum += Math.Abs(a.Data[(y * 512 + x) * 3 + c] - b.Data[(y * 512 + x) * 3 + c]); n++; }
            return sum / n;
        }
        double inside = Diff(176, 336, 176, 336), outside = Diff(0, 100, 0, 512);
        TestContext.Out.WriteLine($"mean abs difference: inside the mask {inside:F3}, far outside {outside:F3}");
        Assert.That(outside, Is.LessThan(0.08), "away from the mask the picture survives the encode/decode round trip");
        Assert.That(inside, Is.GreaterThan(outside * 2), "the masked area was repainted");
    }

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }
}
