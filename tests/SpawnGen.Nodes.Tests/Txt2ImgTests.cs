using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

public class PngWriterTests
{
    [Test]
    public void Encodes_a_valid_png_signature_and_header()
    {
        var img = new ImageBatch(1, 2, 3, 3, [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0.5f, 0.5f, 0.5f]);
        var png = PngWriter.Encode(img);
        Assert.That(png.Take(8), Is.EqualTo(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A }));
        Assert.That(System.Text.Encoding.ASCII.GetString(png, 12, 4), Is.EqualTo("IHDR"));
        Assert.That(System.Buffers.Binary.BinaryPrimitives.ReadUInt32BigEndian(png.AsSpan(16)), Is.EqualTo(3u)); // width
        Assert.That(System.Buffers.Binary.BinaryPrimitives.ReadUInt32BigEndian(png.AsSpan(20)), Is.EqualTo(2u)); // height
    }
}

/// <summary>
/// End to end txt2img with the decomposed SD-Turbo nodes. Needs the model files on disk: set SPAWNGEN_MODELS to a
/// folder holding tokenizer/{vocab.json,merges.txt}, text_encoder/model.onnx, unet/model.onnx, vae_decoder/model.onnx
/// (the schmuell/sd-turbo-ort-web layout). Skipped when it is not set.
/// </summary>
public class Txt2ImgTests
{
    static PromptGraph Graph(string prompt, int seed, int size)
    {
        var g = new PromptGraph();
        g.Add("1", "CheckpointLoader", ("ckpt_name", "sd_turbo"));
        g.Add("2", "CLIPTextEncode", ("text", prompt), ("clip", PromptInput.Link("1", 1)));
        g.Add("3", "CLIPTextEncode", ("text", ""), ("clip", PromptInput.Link("1", 1)));
        g.Add("4", "EmptyLatentImage", ("width", size), ("height", size), ("batch_size", 1));
        g.Add("5", "KSampler", ("model", PromptInput.Link("1", 0)), ("seed", seed), ("steps", 1), ("cfg", 1.0),
            ("sampler_name", "euler"), ("scheduler", "normal"), ("positive", PromptInput.Link("2")),
            ("negative", PromptInput.Link("3")), ("latent_image", PromptInput.Link("4")), ("denoise", 1.0));
        g.Add("6", "VAEDecode", ("samples", PromptInput.Link("5")), ("vae", PromptInput.Link("1", 2)));
        g.Add("7", "SaveImage", ("images", PromptInput.Link("6")), ("filename_prefix", "txt2img_test"));
        return g;
    }

    [Test]
    public void The_txt2img_graph_validates_against_the_registered_schemas()
    {
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly);
        var result = new PromptExecutor(registry).Validate(Graph("a cat", seed: 1, size: 512));
        Assert.That(result.Errors, Is.Empty, string.Join("\n", result.Errors.Select(e => e.Message)));
        Assert.That(result.Plan, Has.Count.EqualTo(7));
    }

    [Test]
    public async Task Generates_a_non_flat_image_and_saves_it()
    {
        var models = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        if (string.IsNullOrEmpty(models)) Assert.Ignore("SPAWNGEN_MODELS is not set.");

        var outDir = Path.Combine(Path.GetTempPath(), "spawngen-tests-" + Guid.NewGuid().ToString("N"));
        using var runtime = new MlRuntime(new FolderModelSource(models!));
        var services = new SimpleServices(runtime, new OutputOptions { Directory = outDir });
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly);
        var executor = new PromptExecutor(registry, services);

        TestContext.Out.WriteLine($"accelerator: {runtime.Accelerator.AcceleratorType} {runtime.Accelerator.Name}");
        var r = await executor.ExecuteAsync(Graph("a photo of a red apple on a wooden table", seed: 7, size: 512));
        TestContext.Out.WriteLine($"elapsed {r.Elapsed}");

        var files = Directory.GetFiles(outDir, "txt2img_test_*.png");
        Assert.That(files, Has.Length.EqualTo(1));
        TestContext.Out.WriteLine(files[0]);
        // Weak check on purpose: it only rules out a flat image. Noise passes too, so LOOK at the saved PNG (printed above);
        // on an OpenCL RX 590 with NuGet 5.4.0 the output is noise because of the token-embedding Gather bug (see docs/library-bug-reports.md).
        Assert.That(new FileInfo(files[0]).Length, Is.GreaterThan(50_000), "A flat image would compress to almost nothing.");

        // Same graph again: every node is cached, nothing re-runs.
        var again = await executor.ExecuteAsync(Graph("a photo of a red apple on a wooden table", seed: 7, size: 512));
        Assert.That(again.Executed, Is.Empty);
    }

    sealed class SimpleServices(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }
}
