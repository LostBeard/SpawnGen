using ILGPU.Runtime;
using SpawnDev.Codecs.Images;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// WebP, GIF and BMP pictures for the Load Image nodes, decoded by SpawnDev.Codecs on the host's accelerator (OpenCL / CUDA on the desktop, WebGPU in the
/// browser): <see cref="Install"/> sets <see cref="ImageDecoder.Codecs"/>. Every frame comes back as the full canvas while it shows (libwebp's and the GIF
/// format's disposal and blending), which is what ComfyUI's Load Image gets per frame.
/// </summary>
public static class CodecsImages
{
    static readonly SemaphoreSlim Gate = new(1, 1);
    static ImageDecoderGpu? _decoder;
    static Accelerator? _accelerator;

    /// <summary>Lets Load Image read WebP, GIF and BMP on the accelerator <paramref name="accelerator"/> gives (asked for on the first such picture).</summary>
    public static void Install(Func<Task<Accelerator>> accelerator) =>
        ImageDecoder.Codecs = async data => ImageDecoderGpu.Detect(data) is ImageFormat.WebP or ImageFormat.Gif or ImageFormat.Bmp ? await DecodeAsync(await accelerator(), data) : null;

    public static async Task<ImageDecoder.DecodedFrames?> DecodeAsync(Accelerator accelerator, byte[] data)
    {
        var format = ImageDecoderGpu.Detect(data);
        if (format is not (ImageFormat.WebP or ImageFormat.Gif or ImageFormat.Bmp)) return null;
        await Gate.WaitAsync();
        try
        {
            if (_decoder is null || _accelerator != accelerator)
            {
                _decoder?.Dispose();
                _decoder = new ImageDecoderGpu(accelerator);
                _accelerator = accelerator;
            }
            var info = await _decoder.ReadInfoAsync(data);
            var frames = new List<byte[]>();
            for (int k = 0; k < Math.Max(1, info.FrameCount); k++)
            {
                var picture = format switch
                {
                    ImageFormat.WebP => await _decoder.WebP.DecodeFrameAsync(data, k, PixelFormat.Rgba8),
                    ImageFormat.Gif => await _decoder.Gif.DecodeFrameAsync(data, k, PixelFormat.Rgba8),
                    _ => await _decoder.DecodeAsync(data, PixelFormat.Rgba8),
                };
                frames.Add(picture.Pixels);
            }
            return new ImageDecoder.DecodedFrames(info.Width, info.Height, info.HasAlpha, frames);
        }
        finally { Gate.Release(); }
    }
}
