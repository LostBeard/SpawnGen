using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Save Image with a picture on the GPU (PLANS "Tensors stay on the GPU", step e): Codecs' PNG encoder writes the file there, and it must hold
/// what PngWriter writes on the host for the same values: the pixels (ComfyUI's truncation, clamped), the colour type and the text chunks.
/// </summary>
public class DevicePictureTests
{
    sealed class Services(IResultStore store, MlRuntime runtime) : IServiceProvider
    {
        public IPictureEncoder? Encoder;
        public object? GetService(Type t) =>
            t == typeof(IResultStore) ? store : t == typeof(MlRuntime) ? runtime : t == typeof(IPictureEncoder) ? Encoder : null;
    }

    static NodeContext NodeCtx(IServiceProvider services) =>
        (NodeContext)Activator.CreateInstance(typeof(NodeContext), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, null,
            ["9", "SaveImage", services, CancellationToken.None, null, new Lazy<string>(() => "{\"9\":{\"class_type\":\"SaveImage\"}}"),
             new Dictionary<string, string> { ["workflow"] = "{\"nodes\":[],\"note\":\"é ✓\"}" }, null, null, 0], null)!;

    /// <summary>Values that hit every edge of the conversion: below 0, above 1, exact steps, just under a step, and NaN.</summary>
    static float[] Values(int count)
    {
        var rng = new Random(7);
        var edges = new[] { -0.5f, 0f, 1f / 255f, 0.5f, 254.999f / 255f, 1f, 1.5f, float.NaN, 0.99999f, 128f / 255f };
        return Enumerable.Range(0, count).Select(i => i % 5 == 0 ? edges[i / 5 % edges.Length] : (float)(rng.NextDouble() * 1.2 - 0.1)).ToArray();
    }

    [TestCase(3)]
    [TestCase(4)]
    public async Task Save_Image_encodes_a_GPU_picture_like_PngWriter_on_the_CPU_accelerator(int channels)
    {
        using var context = Context.Create(b => b.CPU());
        using var accelerator = context.CreateCPUAccelerator(0);
        await Check(accelerator, channels);
    }

    [TestCase(3)]
    [TestCase(4)]
    [Explicit("Needs a GPU (OpenCL / CUDA).")]
    public async Task Save_Image_encodes_a_GPU_picture_like_PngWriter_on_the_GPU(int channels)
    {
        using var runtime = new MlRuntime(new FolderModelSource(Path.GetTempPath()));
        TestContext.Out.WriteLine(runtime.Accelerator.Name);
        await Check(runtime.Accelerator, channels);
    }

    /// <summary>
    /// A real 512x512 picture (ComfyUI's SD 1.5 apple) at the size Save Image writes on the site: the 37x23 check above passed while the live site
    /// (a24eb38, Codecs 1.1.2) saved PNGs whose deflate stream did not decompress ("invalid bit length repeat"), so the encoder is checked at
    /// production scale too: the file must decode to exactly the picture that went in.
    /// </summary>
    [Test]
    public async Task Save_Image_encodes_a_real_512_picture_on_the_CPU_accelerator()
    {
        using var context = Context.Create(b => b.CPU());
        using var accelerator = context.CreateCPUAccelerator(0);
        await CheckPhoto(accelerator);
    }

    [Test]
    [Explicit("Needs a GPU (OpenCL / CUDA).")]
    public async Task Save_Image_encodes_a_real_512_picture_on_the_GPU()
    {
        using var runtime = new MlRuntime(new FolderModelSource(Path.GetTempPath()));
        TestContext.Out.WriteLine(runtime.Accelerator.Name);
        await CheckPhoto(runtime.Accelerator);
    }

    static async Task CheckPhoto(Accelerator accelerator)
    {
        var source = PngReader.Decode(File.ReadAllBytes(Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "canny", "apple.png")));
        using var runtime = new MlRuntime(new FolderModelSource(Path.GetTempPath()), accelerator);
        var store = new MemoryResultStore();
        var services = new Services(store, runtime);
        services.Encoder = new GpuPictureEncoder(services);
        using var images = ImageBatch.FromDevice(1, source.Height, source.Width, source.Channels, accelerator.Allocate1D(source.Data));
        var (_, ui) = await new SaveImage().ExecuteAsync(images, NodeCtx(services), services, "photo");
        Assert.That(images.OnHost, Is.False, "the picture never came to the host");
        var file = ((IEnumerable<object>)ui["images"]!).Cast<ResultFile>().Single();
        var png = store.Get(file.Type, file.Filename, file.Subfolder)!;
        var decoded = PngReader.Decode(png);
        Assert.That((decoded.Width, decoded.Height, decoded.Channels), Is.EqualTo((source.Width, source.Height, source.Channels)));
        Assert.That(decoded.Data, Is.EqualTo(source.Data), "the 8-bit picture comes back exactly");
    }

    static async Task Check(Accelerator accelerator, int channels)
    {
        const int batch = 2, h = 23, w = 37;
        var values = Values(batch * h * w * channels);
        using var runtime = new MlRuntime(new FolderModelSource(Path.GetTempPath()), accelerator);
        var store = new MemoryResultStore();
        var services = new Services(store, runtime);
        services.Encoder = new GpuPictureEncoder(services);
        var ctx = NodeCtx(services);

        using var images = ImageBatch.FromDevice(batch, h, w, channels, accelerator.Allocate1D(values));
        var (_, ui) = await new SaveImage().ExecuteAsync(images, ctx, services, "gpu");
        Assert.That(images.OnHost, Is.False, "the picture never came to the host");

        var host = new ImageBatch(batch, h, w, channels, values);
        var metadata = new Dictionary<string, string> { ["prompt"] = ctx.PromptJson, ["workflow"] = "{\"nodes\":[],\"note\":\"é ✓\"}" }; // what Save Image writes
        var files = ((IEnumerable<object>)ui["images"]!).Cast<ResultFile>().ToList();
        Assert.That(files.Select(f => f.Filename), Is.EqualTo(new[] { "gpu_00001_.png", "gpu_00002_.png" }));
        for (int i = 0; i < batch; i++)
        {
            var gpuPng = store.Get(files[i].Type, files[i].Filename, files[i].Subfolder)!;
            var hostPng = PngWriter.Encode(host, i, metadata);
            Assert.That(gpuPng[25], Is.EqualTo(hostPng[25]), "IHDR colour type (RGB 2 / RGBA 6, as Pillow writes for ComfyUI)");
            var gpuPixels = PngReader.Decode(gpuPng, out var gpuAlpha);
            var hostPixels = PngReader.Decode(hostPng, out var hostAlpha);
            Assert.That(gpuPixels.Data, Is.EqualTo(hostPixels.Data), $"picture {i}: the same 8-bit values");
            Assert.That(gpuAlpha, Is.EqualTo(hostAlpha), $"picture {i}: the same alpha");
            Assert.That(PngReader.ReadText(gpuPng), Is.EqualTo(PngReader.ReadText(hostPng)), "the prompt and the workflow, as text chunks");
        }
    }
}
