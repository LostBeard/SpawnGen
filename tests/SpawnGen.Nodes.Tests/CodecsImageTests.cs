using System.IO.Compression;
using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Load Image with WebP, GIF and BMP files (SpawnDev.Codecs) against Fixtures/codecs (make_load_image.py): the batch and mask layout of ComfyUI's own
/// Load Image, and the pixels of the reference decoders (libwebp and Pillow, via Pillow's RGBA of every frame). ComfyUI's PyAV path converts most of these
/// formats through swscale to float, 1 to 2.4 / 255 away from the decoded values, so its IMAGE is only checked to within 3 / 255 and its MASK to
/// within 1 / 255 (its alpha goes through the same conversion: a fully opaque pixel of a still WebP or a GIF gets a mask of 1/255, not 0).
/// </summary>
public class CodecsImageTests
{
    static readonly string[] Files = ["still_lossless.webp", "still_lossy.webp", "still_lossy_alpha.webp", "anim_lossless.webp", "anim.gif", "still.bmp"];

    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "codecs");

    static JsonElement Fixture(string name)
    {
        using var gz = new GZipStream(File.OpenRead(Path.Combine(Dir, "load_image.json.gz")), CompressionMode.Decompress);
        return JsonDocument.Parse(gz).RootElement.GetProperty(name).Clone();
    }

    static (int[] Shape, float[] Data) Tensor(JsonElement t) =>
        (t.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray(), t.GetProperty("data").EnumerateArray().Select(e => e.GetSingle()).ToArray());

    [TestCaseSource(nameof(Files))]
    public async Task Load_Image_reads_the_file_like_ComfyUI_on_the_CPU_accelerator(string file)
    {
        using var context = Context.Create(b => b.CPU());
        using var accelerator = context.CreateCPUAccelerator(0);
        await Check(accelerator, file);
    }

    [TestCaseSource(nameof(Files))]
    [Explicit("Needs a GPU (OpenCL / CUDA).")]
    public async Task Load_Image_reads_the_file_like_ComfyUI_on_the_GPU(string file)
    {
        using var runtime = new MlRuntime(new FolderModelSource(Path.GetTempPath()));
        TestContext.Out.WriteLine(runtime.Accelerator.Name);
        await Check(runtime.Accelerator, file);
    }

    static async Task Check(Accelerator accelerator, string file)
    {
        var fx = Fixture(file);
        var store = new MemoryInputStore();
        await store.SaveAsync(file, File.ReadAllBytes(Path.Combine(Dir, file)));
        var (savedStore, savedCodecs) = (InputFiles.Store, ImageDecoder.Codecs);
        InputFiles.Store = store;
        CodecsImages.Install(() => Task.FromResult(accelerator));
        try
        {
            var (image, mask) = await new LoadImage().ExecuteAsync(file);
            var (comfyImageShape, comfyImage) = Tensor(fx.GetProperty("image"));
            var (comfyMaskShape, comfyMask) = Tensor(fx.GetProperty("mask"));
            Assert.That(image.Shape, Is.EqualTo(comfyImageShape), "IMAGE shape");
            Assert.That(mask.Shape, Is.EqualTo(comfyMaskShape), "MASK shape");

            var pillow = fx.GetProperty("pillow_rgba").EnumerateArray().Select(f => f.EnumerateArray().Select(e => (byte)e.GetInt32()).ToArray()).ToArray();
            int n = image.Height * image.Width;
            bool hasMask = mask.Height == image.Height && mask.Width == image.Width;
            int rgbOff = 0, maskOff = 0;
            double worstComfy = 0;
            for (int f = 0; f < image.Batch; f++)
                for (int i = 0; i < n; i++)
                {
                    byte a = pillow[f][i * 4 + 3];
                    for (int c = 0; c < 3; c++)
                    {
                        float ours = image.Data[(f * n + i) * 3 + c];
                        // Under a fully transparent pixel the colour is whatever the decoder leaves there (Pillow keeps the palette colour, ffmpeg not).
                        if (a > 0 && Math.Abs(ours * 255 - pillow[f][i * 4 + c]) > 0.01) rgbOff++;
                        if (a > 0) worstComfy = Math.Max(worstComfy, Math.Abs(ours - comfyImage[(f * n + i) * 3 + c]) * 255);
                    }
                    if (hasMask)
                    {
                        float m = mask.Data[f * n + i];
                        if (Math.Abs(m - (1 - a / 255f)) > 1e-6) maskOff++;
                        if (Math.Abs(m - comfyMask[f * n + i]) > 1.01 / 255) maskOff++;
                    }
                }
            if (!hasMask) Assert.That(mask.Data, Is.All.EqualTo(0f), "ComfyUI's empty mask");
            TestContext.Out.WriteLine($"{file}: {image.Batch} frame(s); largest difference to ComfyUI's IMAGE {worstComfy:F2}/255");
            Assert.That(rgbOff, Is.Zero, "colour values that differ from the reference decoder");
            Assert.That(maskOff, Is.Zero, "mask values that differ");
            Assert.That(worstComfy, Is.LessThanOrEqualTo(3.0), "ComfyUI's IMAGE (PyAV) within 3/255");
        }
        finally { (InputFiles.Store, ImageDecoder.Codecs) = (savedStore, savedCodecs); }
    }
}
