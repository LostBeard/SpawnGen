using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.Codecs.Images;
using SpawnDev.Codecs.Images.WebP;
using SpawnDev.Codecs.Images.WebP.Lossy;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

[Node(Category = "image", DisplayName = "Save Animated WEBP", IsOutput = true), DeviceTensors]
public sealed class SaveAnimatedWEBP
{
    public static string[] Methods() => ["default", "fastest", "slowest"];

    /// <summary>
    /// The whole batch as one animated WebP (<c>prefix_00001_.webp</c>), each frame <c>int(1000 / fps)</c> ms, looping, with ComfyUI's EXIF (the prompt in Model,
    /// the workflow in Make). Encoded on the GPU by SpawnDev.Codecs, which ports libwebp's WebPAnimEncoder for both modes (change rectangles, disposal,
    /// blending, key frames): lossy matches Pillow's file byte for byte (quality, method default = 4); lossless makes the same frame decisions with its own
    /// VP8L coder (the pictures decode exactly; the bytes differ). A single picture is saved as a still WebP, as Pillow does.
    /// The pictures are passed on. Everything runs on the device (AGENTS.md rule 4): the pictures (uploaded once if a host node made them) become
    /// RGBA8 in a kernel, the encoder writes the file into a device buffer, and the result store takes it from there (in the browser straight into JS).
    /// </summary>
    public async Task<(ImageBatch Images, UiOutput Ui)> ExecuteAsync(ImageBatch images, NodeContext context, IServiceProvider services,
        string filenamePrefix = "ComfyUI",
        [Input(Min = 0.01, Max = 1000, Step = 0.01)] double fps = 6.0,
        bool lossless = true,
        [Input(Min = 0, Max = 100)] int quality = 80,
        [Input(OptionsProvider = nameof(Methods))] string method = "default")
    {
        int m = method switch { "default" => 4, "fastest" => 0, "slowest" => 6, _ => throw new ArgumentException($"Unknown method '{method}'.", nameof(method)) };
        if (!lossless && m != 4)
            throw new NotSupportedException($"Lossy WebP is encoded with libwebp's method 4 (\"default\") on the GPU; method \"{method}\" is not available yet.");
        if (images.Channels is not (3 or 4)) throw new ArgumentException($"WebP needs 3 or 4 channels, got {images.Channels}.", nameof(images));
        var accelerator = (await GpuHelpers.RuntimeAsync(services)).Accelerator;
        var (file, length) = await EncodeOnDeviceAsync(images, fps, quality, accelerator, lossless, m, ExifWriter.ComfyMetadata(Results.Metadata(context)));
        int w = images.Width, h = images.Height;
        var prefix = SavePrefix.ComputeVars(filenamePrefix, w, h, DateTime.Now);
        ResultFile saved;
        using (file) saved = (await Results.For(services).SaveDeviceAsync("output", prefix, file, length, "webp")).AsImage(w, h);
        var ui = new UiOutput("images", new object[] { saved });
        if (images.Batch > 1) ui["animated"] = new object[] { true };
        return (images, ui);
    }

    /// <summary>The batch as a WebP file (animated for more than one picture), each frame int(1000 / fps) ms; <paramref name="method"/> is libwebp's 0-6.</summary>
    public static async Task<byte[]> EncodeAsync(ImageBatch images, double fps, int quality, Accelerator accelerator,
        bool lossless = true, int method = 4, byte[]? exif = null)
    {
        var (file, length) = await EncodeOnDeviceAsync(images, fps, quality, accelerator, lossless, method, exif);
        using (file) return await file.View.SubView(0, length).CopyToHostAsync();
    }

    // Output bounds until Codecs' exact-size device overloads (EncodeAnimationToDeviceAsync / ToUint8ArrayAsync, Geordi 2026-11-19) are out.
    // Lossless (Geordi: valid): per frame the encoder's own still bound, the EXIF and the container.
    static long LosslessBound(int w, int h, int frames, int exif) =>
        frames * (WebPLosslessEncoderGpu.GetMaxEncodedSize(w, h) + 64L) + Chunk(exif) + 1024;

    // Lossy: Geordi's formula for his GetMaxAnimationEncodedSize (a guarantee; the encoder fails rather than exceed partition 0's capacity):
    // VP8 <= 10 + (4096 + 64 mbs) + 7625 mbs + 1024, ALPH <= 1 + w h, each frame chunk(16 + chunk(ALPH) + chunk(VP8)), plus RIFF, VP8X, ANIM, EXIF.
    static long LossyBound(int w, int h, int frames, int exif)
    {
        long mbs = (long)((w + 15) / 16) * ((h + 15) / 16);
        long vp8 = 10 + (4096 + 64 * mbs) + 7625 * mbs + 1024, alph = 1 + (long)w * h;
        return 12 + Chunk(10) + Chunk(6) + frames * Chunk(16 + Chunk(alph) + Chunk(vp8)) + Chunk(exif);
    }

    static long Chunk(long payload) => 8 + payload + (payload & 1);

    /// <summary>
    /// <see cref="EncodeAsync"/> on the device: the file in a device buffer the caller disposes, and its length. The pictures go to RGBA8 in
    /// <see cref="ImageKernels.ToRgba8"/> (ComfyUI's truncation) and the encoder reads them there.
    /// </summary>
    public static async Task<(MemoryBuffer1D<byte, Stride1D.Dense> File, int Length)> EncodeOnDeviceAsync(ImageBatch images, double fps, int quality,
        Accelerator accelerator, bool lossless = true, int method = 4, byte[]? exif = null)
    {
        if (fps <= 0) throw new ArgumentOutOfRangeException(nameof(fps), "The frame rate must be above 0.");
        if (images.Channels is not (3 or 4)) throw new ArgumentException($"WebP needs 3 or 4 channels, got {images.Channels}.", nameof(images));
        exif ??= ExifWriter.ComfyMetadata(new Dictionary<string, string>());
        int w = images.Width, h = images.Height, n = images.Batch, duration = (int)(1000.0 / fps), m = method;
        long pixels = (long)w * h;
        using var rgba = accelerator.Allocate1D<int>(pixels * n);
        ImageKernels.ToRgba8(accelerator, await images.ToDeviceAsync(accelerator), images.Channels, rgba.View);
        ArrayView1D<byte, Stride1D.Dense> bytes = rgba.View.BaseView.Cast<byte>();
        var frames = Enumerable.Range(0, n).Select(i => bytes.SubView(i * pixels * 4, pixels * 4)).ToList();
        var file = accelerator.Allocate1D<byte>(lossless ? LosslessBound(w, h, n, exif.Length) : LossyBound(w, h, n, exif.Length));
        try
        {
            int length;
            if (lossless)
            {
                using var enc = new WebPLosslessEncoderGpu(accelerator);
                // Exact = false is libwebp's default (it may change the colour under fully transparent pixels).
                length = n == 1
                    ? await enc.EncodeAsync(frames[0], w, h, file.View, new WebPEncodeOptions { Quality = quality, Exact = false, Exif = exif })
                    : await enc.EncodeAnimationAsync(frames, w, h, [.. Enumerable.Repeat(duration, n)], file.View,
                        new WebPEncodeOptions { Quality = quality, Exact = false }, new WebPAnimEncoderOptions { Exif = exif });
            }
            else
            {
                using var enc = new WebPLossyEncoderGpu(accelerator);
                // A single picture takes Pillow's still path, which strips the "Exif\0\0" prefix before libwebp writes the chunk.
                length = n == 1
                    ? await enc.EncodeAsync(frames[0], w, h, file.View, new WebPLossyEncodeOptions { Quality = quality, Method = m, Exif = exif[6..] })
                    : await enc.EncodeAnimationAsync(frames, w, h, [.. Enumerable.Repeat(duration, n)], file.View,
                        new WebPLossyEncodeOptions { Quality = quality, Method = m }, new WebPAnimEncoderOptions { Exif = exif });
            }
            return (file, length);
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }
}
