using SpawnDev.ILGPU.ML.Wan;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// ComfyUI's split Wan 2.2 5B graph (its video_wan2_2_5B_ti2v template) on the GPU through SpawnGen's executor: Load Diffusion Model, Load CLIP type
/// "wan", Load VAE, Wan22ImageToVideoLatent, ModelSamplingSD3, KSampler, VAE Decode. The model names are the library's hosted exports (ComfyUI's
/// repackaged files are not loadable yet); the rest are the template's values.
/// </summary>
public class WanSplitGraphTests
{
    static readonly string Fixtures = Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "wan22");

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }

    static string Models()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS is not set.");
        return root!;
    }

    [Test, Explicit("Needs SPAWNGEN_MODELS, a GPU and the Wan 2.2 VAE (downloads).")]
    public async Task The_Wan_VAE_encodes_a_picture_to_ComfyUIs_raw_latent()
    {
        using var runtime = new MlRuntime(new DesktopModelSource(Models(), MlRuntime.RepoFolders));
        using var vae = new WanVaeModel(runtime, WanFiles.Vae);
        var picture = ImageDecoder.Decode(File.ReadAllBytes(Path.Combine(Fixtures, "spawngen_wan_probe_256x160.png")));
        var latent = await vae.EncodeVideoAsync(picture);
        var bytes = File.ReadAllBytes(Path.Combine(Fixtures, "wan22_vaeencode_256x160_samples_f32_1x48x1x10x16.raw"));
        var comfy = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, comfy, 0, bytes.Length);
        Assert.That(latent.Shape, Is.EqualTo(new[] { 1, 48, 1, 10, 16 }));
        double maxDiff = 0, sumSq = 0, refSq = 0;
        for (int i = 0; i < comfy.Length; i++)
        {
            double d = latent.Data[i] - comfy[i];
            maxDiff = Math.Max(maxDiff, Math.Abs(d));
            sumSq += d * d;
            refSq += comfy[i] * (double)comfy[i];
        }
        double relRms = Math.Sqrt(sumSq / refSq);
        TestContext.Out.WriteLine($"max |diff| {maxDiff:G4}, relative RMS {relRms:G4} (fp16 VAE on the GPU vs ComfyUI's fp32 on the CPU)");
        Assert.That(relRms, Is.LessThan(0.01), "the same latent, in ComfyUI's raw units");

        var frames = await vae.DecodeVideoAsync(latent);
        Assert.That(frames.Shape, Is.EqualTo(new[] { 1, 160, 256, 3 }));
        double err = 0;
        for (int i = 0; i < frames.Data.Length; i++) err += Math.Abs(frames.Data[i] - picture.Data[i]);
        TestContext.Out.WriteLine($"decode back: mean |diff| {err / frames.Data.Length:G4}");
        Assert.That(err / frames.Data.Length, Is.LessThan(0.05), "encode then decode gives the picture back");
    }

    [Test, Explicit("Needs SPAWNGEN_MODELS, a GPU and the Wan 2.2 files (downloads ~7 GB). The template's settings at 9 frames: about 15 minutes on an RX 590.")]
    public async Task ComfyUIs_Wan_2_2_5B_template_graph_runs()
    {
        var root = Models();
        var outDir = Path.Combine(TestContext.CurrentContext.WorkDirectory, "wan_split_out");
        Directory.CreateDirectory(outDir);
        using var runtime = new MlRuntime(new DesktopModelSource(root, MlRuntime.RepoFolders));
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly);
        var executor = new PromptExecutor(registry, new Services(runtime, new OutputOptions { Directory = outDir, TempDirectory = outDir }));
        int length = int.TryParse(Environment.GetEnvironmentVariable("SPAWNGEN_WAN_LENGTH"), out var n) ? n : 9;

        var g = new PromptGraph();
        g.Add("37", "UNETLoader", ("unet_name", WanFiles.DiffusionModels[0]), ("weight_dtype", "default"));
        g.Add("38", "CLIPLoader", ("clip_name", WanFiles.TextEncoder), ("type", "wan"), ("device", "default"));
        g.Add("39", "VAELoader", ("vae_name", WanFiles.Vae));
        g.Add("6", "CLIPTextEncode", ("clip", PromptInput.Link("38")), ("text",
            "Low contrast. In a retro 1970s-style subway station, a street musician plays in dim colors and rough textures. He wears an old jacket, playing guitar with focus. Commuters hurry by, and a small crowd gathers to listen. The camera slowly moves right, capturing the blend of music and city noise, with old subway signs and mottled walls in the background."));
        g.Add("7", "CLIPTextEncode", ("clip", PromptInput.Link("38")), ("text", Wan22Pipeline.DefaultNegativePrompt));
        g.Add("55", "Wan22ImageToVideoLatent", ("vae", PromptInput.Link("39")), ("width", 1280L), ("height", 704L), ("length", (long)length), ("batch_size", 1L));
        g.Add("48", "ModelSamplingSD3", ("model", PromptInput.Link("37")), ("shift", 8.0));
        g.Add("3", "KSampler", ("model", PromptInput.Link("48")), ("positive", PromptInput.Link("6")), ("negative", PromptInput.Link("7")),
            ("latent_image", PromptInput.Link("55")), ("seed", 898471028164125L), ("steps", 20L), ("cfg", 5.0), ("sampler_name", "uni_pc"),
            ("scheduler", "simple"), ("denoise", 1.0));
        g.Add("8", "VAEDecode", ("samples", PromptInput.Link("3")), ("vae", PromptInput.Link("39")));
        g.Add("57", "CreateVideo", ("images", PromptInput.Link("8")), ("fps", 24.0));
        g.Add("58", "GetVideoComponents", ("video", PromptInput.Link("57")));
        g.Add("60", "SaveAnimatedWEBP", ("images", PromptInput.Link("58")), ("filename_prefix", "wan_split"), ("fps", PromptInput.Link("58", 2)),
            ("lossless", true), ("quality", 80L), ("method", "default"));
        g.Add("61", "ImageFromBatch", ("image", PromptInput.Link("8")), ("batch_index", (long)(length / 2)), ("length", 1L));
        g.Add("62", "SaveImage", ("images", PromptInput.Link("61")), ("filename_prefix", "wan_split_middle"));

        var clock = System.Diagnostics.Stopwatch.StartNew();
        await executor.ExecuteAsync(g);
        TestContext.Out.WriteLine($"ran in {clock.Elapsed.TotalSeconds:F0} s; output in {outDir}");
        Assert.That(Directory.GetFiles(outDir, "wan_split_*.webp"), Is.Not.Empty, "the clip was saved");
        var middle = ImageDecoder.Decode(File.ReadAllBytes(Directory.GetFiles(outDir, "wan_split_middle_*.png").OrderBy(f => f).Last()));
        Assert.That((middle.Width, middle.Height), Is.EqualTo((1280, 704)));
        double mean = middle.Data.Average(), spread = Math.Sqrt(middle.Data.Average(v => (v - mean) * (v - mean)));
        TestContext.Out.WriteLine($"middle frame mean {mean:F3}, spread {spread:F3}");
        Assert.That(spread, Is.GreaterThan(0.05), "a picture, not a flat field");
        foreach (var f in Directory.GetFiles(outDir)) TestContext.Out.WriteLine(f);
    }
}
