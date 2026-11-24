using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnDev.ILGPU;
using SpawnDev.Codecs.Images;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// SpawnDev.Codecs' device decoders (the file on the accelerator, the pixels into a device buffer: the path Load Image takes to keep a picture
/// on the GPU) on the decode corpus, held to the same spec-correct reference as Load Image's host decoders (LoadImageCorpusTests): PNG exact,
/// JPEG within libjpeg's freedom. Rgba16 for 16-bit files so nothing is cut; the EXIF orientation from ImageInfo applied here, as Load Image
/// will. Each case is a measurement of where Codecs stands before Load Image moves onto it (PLANS: Input files, part B).
/// </summary>
public class CodecsDecodeCorpusTests
{
    static string Folder => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "decode_corpus");

    public static IEnumerable<string> Files() => LoadImageCorpusTests.Files();

    [TestCaseSource(nameof(Files))]
    public async Task Codecs_device_decode_on_the_CPU_accelerator(string file)
    {
        using var context = Context.Create(b => b.CPU());
        using var accelerator = context.CreateCPUAccelerator(0);
        await Check(accelerator, file);
    }

    [TestCaseSource(nameof(Files))]
    [Explicit("Needs a GPU (OpenCL / CUDA).")]
    public async Task Codecs_device_decode_on_the_GPU(string file)
    {
        using var runtime = new MlRuntime(new FolderModelSource(Path.GetTempPath()));
        TestContext.Out.WriteLine(runtime.Accelerator.Name);
        await Check(runtime.Accelerator, file);
    }

    static async Task Check(Accelerator accelerator, string file)
    {
        var bytes = File.ReadAllBytes(Path.Combine(Folder, file));
        using var decoder = new ImageDecoderGpu(accelerator);
        using var fileBuffer = accelerator.Allocate1D(bytes);
        var info = await decoder.ReadInfoAsync(fileBuffer.View, bytes.Length);
        TestContext.Out.WriteLine($"{file}: {info}");
        var format = info.BitDepth == 16 ? PixelFormat.Rgba16 : PixelFormat.Rgba8;
        using var pixels = accelerator.Allocate1D<byte>(info.GetPixelBufferSize(format));
        await decoder.DecodeAsync(fileBuffer.View, bytes.Length, pixels.View, format);
        var rgba = await pixels.View.CopyToHostAsync();

        int w = info.Width, h = info.Height, n = w * h;
        var image = new ImageBatch(1, h, w, 3);
        var alpha = new float[n];
        for (int i = 0; i < n; i++)
            for (int c = 0; c < 4; c++)
            {
                float v = format == PixelFormat.Rgba16 ? (rgba[i * 8 + c * 2] | (rgba[i * 8 + c * 2 + 1] << 8)) / 65535f : rgba[i * 4 + c] / 255f;
                if (c < 3) image.Data[i * 3 + c] = v; else alpha[i] = v;
            }
        if (info.Orientation != 1)
        {
            alpha = ExifOrientation.Apply(alpha, w, h, info.Orientation);
            image = ExifOrientation.Apply(image, info.Orientation);
        }
        // Load Image's mask: 1 - alpha at full size for a picture that can be transparent (a palette one always), else the empty 64 x 64 one.
        bool transparent = info.HasAlpha || bytes.Length > 25 && bytes[0] == 0x89 && bytes[25] == 3;
        var mask = transparent ? new MaskBatch(1, image.Height, image.Width) : new MaskBatch(1, 64, 64);
        if (transparent) for (int i = 0; i < n; i++) mask.Data[i] = 1f - alpha[i];
        LoadImageCorpusTests.Check(file, image, mask);
    }
}
