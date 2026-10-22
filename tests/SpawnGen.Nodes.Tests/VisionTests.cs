using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>Library vision models as nodes: packing, and (explicit, GPU + network on first use) real runs whose pictures are checked by eye.</summary>
public class VisionTests
{
    static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly).Add(typeof(CaptureText));

    /// <summary>Test-only output node that keeps the text it is given, so a run's string outputs can be read.</summary>
    [Node(Category = "test", IsOutput = true)]
    public sealed class CaptureText
    {
        public static string Last = "";
        public void Execute(string text) => Last = text;
    }

    [Test]
    public void Packing_pictures_for_the_library_round_trips_within_one_level()
    {
        var image = new ImageBatch(1, 2, 3, 3);
        for (int i = 0; i < image.Data.Length; i++) image.Data[i] = i / (float)image.Data.Length;
        var packed = Rgba.Pack(image, 0);
        Assert.That(packed, Has.Length.EqualTo(6));
        Assert.That(packed[0] >>> 24, Is.EqualTo(255), "opaque");
        var back = Rgba.Unpack([packed], 3, 2);
        for (int i = 0; i < image.Data.Length; i++) Assert.That(back.Data[i], Is.EqualTo(image.Data[i]).Within(1 / 255f));
    }

    [Test]
    public void Packing_a_four_channel_picture_keeps_its_alpha_and_a_one_channel_one_is_grey()
    {
        var rgba = new ImageBatch(1, 1, 1, 4, [1f, 0f, 0.5f, 0.25f]);
        int v = Rgba.Pack(rgba, 0)[0];
        Assert.That((v & 255, (v >> 8) & 255, (v >> 16) & 255, v >>> 24), Is.EqualTo((255, 0, 128, 64)));
        var alpha = new float[1];
        Rgba.Unpack([[v]], 1, 1, alpha);
        Assert.That(alpha[0], Is.EqualTo(64 / 255f).Within(1e-6));

        var grey = new ImageBatch(1, 1, 1, 1, [0.5f]);
        var g = Rgba.Pack(grey, 0)[0];
        Assert.That((g & 255, (g >> 8) & 255, (g >> 16) & 255), Is.EqualTo((128, 128, 128)));
    }

    [Test]
    public void The_upscale_model_loader_offers_the_known_models_and_rejects_others()
    {
        Assert.That(UpscaleModelLoader.Models(), Does.Contain("espcn_x3.onnx"));
        var model = new UpscaleModelLoader().Execute("espcn_x3.onnx");
        Assert.That(model.Scale, Is.EqualTo(3));
        Assert.That(() => new UpscaleModelLoader().Execute("nope.onnx"), Throws.ArgumentException);
    }

    [Test]
    public void An_upscale_graph_validates()
    {
        var g = new PromptGraph();
        g.Add("1", "UpscaleModelLoader", ("model_name", "espcn_x3.onnx"));
        g.Add("2", "LoadImage", ("image", "in.png"));
        g.Add("3", "ImageUpscaleWithModel", ("upscale_model", PromptInput.Link("1")), ("image", PromptInput.Link("2")));
        g.Add("4", "SaveImage", ("images", PromptInput.Link("3")), ("filename_prefix", "up"));
        var dir = Path.Combine(Path.GetTempPath(), "spawngen-in-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        File.WriteAllBytes(Path.Combine(dir, "in.png"), PngWriter.Encode(new ImageBatch(1, 8, 8, 3)));
        InputFiles.Directory = dir;
        var result = new PromptExecutor(Registry()).Validate(g);
        Assert.That(result.Errors, Is.Empty, string.Join("\n", result.Errors.Select(e => e.Message)));
    }

    static string? ModelsRoot()
    {
        var folder = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        return string.IsNullOrEmpty(folder) ? null : Path.GetDirectoryName(Path.TrimEndingDirectorySeparator(folder));
    }

    /// <summary>Runs <paramref name="build"/>'s graph on the real GPU with the desktop model source (downloads the vision model into the models folder's hub cache on first use).</summary>
    internal static async Task<string[]> RunAsync(string name, Func<PromptGraph> build, string? inputPicture = null)
    {
        var root = ModelsRoot();
        if (root is null) Assert.Ignore("SPAWNGEN_MODELS is not set.");
        var work = Path.Combine(Path.GetTempPath(), "spawngen-vision-" + Guid.NewGuid().ToString("N"));
        InputFiles.Directory = Path.Combine(work, "input");
        Directory.CreateDirectory(InputFiles.Directory);
        if (inputPicture is not null) File.Copy(inputPicture, Path.Combine(InputFiles.Directory, "in.png"));
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var executor = new PromptExecutor(Registry(), new Services(runtime, new OutputOptions { Directory = Path.Combine(work, "output") }));
        var r = await executor.ExecuteAsync(build());
        TestContext.Out.WriteLine($"{name}: {r.Elapsed.TotalSeconds:F1} s on {runtime.Accelerator.AcceleratorType} {runtime.Accelerator.Name}");
        var files = Directory.GetFiles(Path.Combine(work, "output"), "*.png");
        foreach (var f in files) TestContext.Out.WriteLine("  " + f);
        return files;
    }

    /// <summary>A picture with structure to upscale: the apple picture of the txt2img tests if SPAWNGEN_TEST_PICTURE points at one, else a synthetic gradient with edges.</summary>
    internal static string TestPicture(int size)
    {
        var path = Path.Combine(Path.GetTempPath(), $"spawngen-vision-in-{size}.png");
        var env = Environment.GetEnvironmentVariable("SPAWNGEN_TEST_PICTURE");
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;
        var img = new ImageBatch(1, size, size, 3);
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
            {
                bool disc = (x - size / 2) * (x - size / 2) + (y - size / 2) * (y - size / 2) < size * size / 9;
                float v = disc ? 0.9f : 0.15f + 0.5f * x / size;
                int o = (y * size + x) * 3;
                img.Data[o] = v; img.Data[o + 1] = disc ? 0.2f : v; img.Data[o + 2] = disc ? 0.2f : 1 - v;
            }
        File.WriteAllBytes(path, PngWriter.Encode(img));
        return path;
    }

    [Test, Explicit("Needs SPAWNGEN_MODELS, a GPU, and the network on first use (ESPCN, 240 KB).")]
    public async Task Upscale_with_ESPCN_makes_a_picture_three_times_as_large()
    {
        var files = await RunAsync("upscale", () =>
        {
            var g = new PromptGraph();
            g.Add("1", "UpscaleModelLoader", ("model_name", "espcn_x3.onnx"));
            g.Add("2", "LoadImage", ("image", "in.png"));
            g.Add("3", "ImageUpscaleWithModel", ("upscale_model", PromptInput.Link("1")), ("image", PromptInput.Link("2")));
            g.Add("4", "SaveImage", ("images", PromptInput.Link("3")), ("filename_prefix", "upscaled"));
            return g;
        }, TestPicture(256));
        Assert.That(files, Has.Length.EqualTo(1));
        var decoded = PngReader.Decode(File.ReadAllBytes(files[0]));
        Assert.That((decoded.Width, decoded.Height), Is.EqualTo((768, 768)));
    }

    [Test, Explicit("Needs SPAWNGEN_MODELS, a GPU, and the network on first use (RMBG 1.4, about 170 MB). Look at the saved pictures.")]
    public async Task Remove_background_finds_the_foreground_and_the_pictures_are_checked_by_eye()
    {
        var files = await RunAsync("rmbg", () =>
        {
            var g = new PromptGraph();
            g.Add("1", "LoadImage", ("image", "in.png"));
            g.Add("2", "RemoveBackground", ("image", PromptInput.Link("1")), ("threshold", 0.5));
            g.Add("3", "MaskToImage", ("mask", PromptInput.Link("2", 1)));
            g.Add("4", "SaveImage", ("images", PromptInput.Link("3")), ("filename_prefix", "rmbg_mask"));
            return g;
        }, TestPicture(512));
        Assert.That(files, Has.Length.EqualTo(1));
        var mask = PngReader.Decode(File.ReadAllBytes(files[0]));
        double centre = 0, corner = 0;
        for (int y = 0; y < 32; y++) for (int x = 0; x < 32; x++)
        {
            centre += mask.Data[((mask.Height / 2 + y) * mask.Width + mask.Width / 2 + x) * mask.Channels];
            corner += mask.Data[(y * mask.Width + x) * mask.Channels];
        }
        TestContext.Out.WriteLine($"mask mean: centre {centre / 1024:F2}, corner {corner / 1024:F2}");
        Assert.That(centre / 1024, Is.GreaterThan(corner / 1024 + 0.3), "the object in the middle is foreground, the corner is background");
    }

    [Test, Explicit("Needs SPAWNGEN_MODELS, a GPU, and the network on first use (Depth Anything V2 small). Look at the saved picture.")]
    public async Task Depth_anything_makes_a_depth_map_that_is_not_flat()
    {
        var files = await RunAsync("depth", () =>
        {
            var g = new PromptGraph();
            g.Add("1", "LoadImage", ("image", "in.png"));
            g.Add("2", "DepthAnythingV2", ("image", PromptInput.Link("1")));
            g.Add("3", "SaveImage", ("images", PromptInput.Link("2")), ("filename_prefix", "depth"));
            return g;
        }, TestPicture(512));
        var depth = PngReader.Decode(File.ReadAllBytes(files.Single()));
        var values = Enumerable.Range(0, depth.Width * depth.Height).Select(i => depth.Data[i * depth.Channels]).ToArray();
        TestContext.Out.WriteLine($"depth range {values.Min():F2}..{values.Max():F2}, mean {values.Average():F2}");
        Assert.That(values.Max() - values.Min(), Is.GreaterThan(0.3f));
    }

    [Test, Explicit("Needs SPAWNGEN_MODELS, a GPU, and the network on first use (a 6.5 MB style). Look at the saved picture.")]
    public async Task Style_transfer_repaints_the_picture()
    {
        var files = await RunAsync("style", () =>
        {
            var g = new PromptGraph();
            g.Add("1", "LoadImage", ("image", "in.png"));
            g.Add("2", "StyleTransfer", ("image", PromptInput.Link("1")), ("style", "mosaic"));
            g.Add("3", "SaveImage", ("images", PromptInput.Link("2")), ("filename_prefix", "styled"));
            return g;
        }, TestPicture(512));
        Assert.That(files, Has.Length.EqualTo(1));
    }

    [Test, Explicit("Needs SPAWNGEN_MODELS, a GPU, SPAWNGEN_TEST_PICTURE (a photo of people) and the network on first use (MoveNet 9 MB). Look at the saved picture.")]
    public async Task Pose_estimation_draws_a_skeleton_and_lists_keypoints()
    {
        var files = await RunAsync("pose", () =>
        {
            var g = new PromptGraph();
            g.Add("1", "LoadImage", ("image", "in.png"));
            g.Add("2", "PoseEstimation", ("image", PromptInput.Link("1")), ("confidence", 0.3), ("overlay", true));
            g.Add("3", "SaveImage", ("images", PromptInput.Link("2")), ("filename_prefix", "pose"));
            g.Add("4", "CaptureText", ("text", PromptInput.Link("2", 1)));
            return g;
        }, TestPicture(512));
        TestContext.Out.WriteLine(CaptureText.Last);
        Assert.That(files, Has.Length.EqualTo(1));
        Assert.That(CaptureText.Last.Split(Environment.NewLine), Has.Length.EqualTo(17));
    }

    [Test, Explicit("Needs SPAWNGEN_MODELS, a GPU, SPAWNGEN_TEST_PICTURE (a photo with people or vehicles) and the network on first use (YOLOv8n 13 MB). Look at the saved picture.")]
    public async Task Object_detection_draws_boxes()
    {
        var files = await RunAsync("detect", () =>
        {
            var g = new PromptGraph();
            g.Add("1", "LoadImage", ("image", "in.png"));
            g.Add("2", "ObjectDetection", ("image", PromptInput.Link("1")), ("confidence", double.TryParse(Environment.GetEnvironmentVariable("DETECT_CONF"), out var conf) ? conf : 0.25));
            g.Add("3", "SaveImage", ("images", PromptInput.Link("2")), ("filename_prefix", "detect"));
            g.Add("4", "CaptureText", ("text", PromptInput.Link("2", 1)));
            return g;
        }, TestPicture(512));
        TestContext.Out.WriteLine(CaptureText.Last);
        Assert.That(files, Has.Length.EqualTo(1));
        Assert.That(CaptureText.Last, Does.Contain("bus").And.Contain("person"));
    }

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }
}
