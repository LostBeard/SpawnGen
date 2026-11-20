using System.Runtime.CompilerServices;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.Codecs.Images.Png;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// Save Image's PNG on the GPU (AGENTS.md rule 4): the picture becomes RGBA8 in <see cref="ImageKernels.ToRgba8"/> (ComfyUI's truncation) and
/// Codecs' PNG encoder writes the file into a device buffer; the text chunks (the prompt, the workflow) are the only host-built bytes.
/// </summary>
public sealed class GpuPictureEncoder(IServiceProvider services) : IPictureEncoder
{
    static readonly ConditionalWeakTable<Accelerator, PngEncoderGpu> Encoders = new();

    public async Task<(MemoryBuffer1D<byte, Stride1D.Dense> File, long Length)?> EncodePngAsync(ImageBatch images, int index, IReadOnlyDictionary<string, string>? text)
    {
        if (images.Channels is not (3 or 4)) return null;
        var accelerator = (await GpuHelpers.RuntimeAsync(services)).Accelerator;
        int w = images.Width, h = images.Height, ch = images.Channels;
        long pixels = (long)w * h;
        var picture = (await images.ToDeviceAsync(accelerator)).SubView(index * pixels * ch, pixels * ch);
        using var rgba = accelerator.Allocate1D<int>(pixels);
        ImageKernels.ToRgba8(accelerator, picture, ch, rgba.View);
        ArrayView1D<byte, Stride1D.Dense> bytes = rgba.View.BaseView.Cast<byte>();
        var options = new PngEncodeOptions
        {
            ColorMode = ch == 3 ? PngColorMode.Rgb : PngColorMode.Rgba, // never Auto: Pillow writes RGB / RGBA for ComfyUI, even for a grey picture
            TextChunks = text,
        };
        var file = accelerator.Allocate1D<byte>(PngEncoderGpu.GetMaxEncodedSize(w, h, options));
        try
        {
            var encoder = Encoders.GetValue(accelerator, a => new PngEncoderGpu(a));
            int length = await encoder.EncodeAsync(bytes, w, h, file.View, options);
            return (file, length);
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }
}
