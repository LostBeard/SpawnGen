using SpawnDev.Codecs.Images;
using SpawnDev.Codecs.Images.WebP;
using SpawnDev.Codecs.Images.WebP.Lossy;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

[Node(Category = "image", DisplayName = "Save Animated WEBP", IsOutput = true)]
public sealed class SaveAnimatedWEBP
{
    public static string[] Methods() => ["default", "fastest", "slowest"];

    /// <summary>
    /// The whole batch as one animated WebP (<c>prefix_00001_.webp</c>), each frame <c>int(1000 / fps)</c> ms, looping, with ComfyUI's EXIF (the prompt in Model,
    /// the workflow in Make). Encoded on the GPU by SpawnDev.Codecs, which ports libwebp's WebPAnimEncoder for both modes (change rectangles, disposal,
    /// blending, key frames): lossy matches Pillow's file byte for byte (quality, method default = 4); lossless makes the same frame decisions with its own
    /// VP8L coder (the pictures decode exactly; the bytes differ). A single picture is saved as a still WebP, as Pillow does.
    /// The pictures are passed on.
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
        var file = await EncodeAsync(images, fps, quality, accelerator, lossless, m, ExifWriter.ComfyMetadata(Results.Metadata(context)));
        int w = images.Width, h = images.Height;
        var prefix = SavePrefix.ComputeVars(filenamePrefix, w, h, DateTime.Now);
        var saved = (await Results.For(services).SaveAsync("output", prefix, file, "webp")).AsImage(w, h);
        var ui = new UiOutput("images", new object[] { saved });
        if (images.Batch > 1) ui["animated"] = new object[] { true };
        return (images, ui);
    }

    /// <summary>The batch as a WebP file (animated for more than one picture), each frame int(1000 / fps) ms; <paramref name="method"/> is libwebp's 0-6.</summary>
    public static async Task<byte[]> EncodeAsync(ImageBatch images, double fps, int quality, ILGPU.Runtime.Accelerator accelerator,
        bool lossless = true, int method = 4, byte[]? exif = null)
    {
        if (fps <= 0) throw new ArgumentOutOfRangeException(nameof(fps), "The frame rate must be above 0.");
        exif ??= ExifWriter.ComfyMetadata(new Dictionary<string, string>());
        int w = images.Width, h = images.Height, duration = (int)(1000.0 / fps), m = method;
        var frames = Enumerable.Range(0, images.Batch).Select(i => (ReadOnlyMemory<byte>)Rgba(images, i)).ToList();
        byte[] file;
        if (lossless)
        {
            using var enc = new WebPLosslessEncoderGpu(accelerator);
            // Exact = false is libwebp's default (it may change the colour under fully transparent pixels).
            file = frames.Count == 1
                ? await enc.EncodeAsync(frames[0], w, h, new WebPEncodeOptions { Quality = quality, Exact = false, Exif = exif })
                : await enc.EncodeAnimationAsync(frames, w, h, [.. Enumerable.Repeat(duration, frames.Count)],
                    new WebPEncodeOptions { Quality = quality, Exact = false }, new WebPAnimEncoderOptions { Exif = exif });
        }
        else
        {
            using var enc = new WebPLossyEncoderGpu(accelerator);
            // A single picture takes Pillow's still path, which strips the "Exif\0\0" prefix before libwebp writes the chunk.
            file = frames.Count == 1
                ? await enc.EncodeAsync(frames[0], w, h, new WebPLossyEncodeOptions { Quality = quality, Method = m, Exif = exif[6..] })
                : await enc.EncodeAnimationAsync(frames, w, h, [.. Enumerable.Repeat(duration, frames.Count)],
                    new WebPLossyEncodeOptions { Quality = quality, Method = m }, new WebPAnimEncoderOptions { Exif = exif });
        }
        return file;
    }

    /// <summary>One picture as RGBA8 the way ComfyUI hands it to Pillow: <c>np.clip(255 * x, 0, 255).astype(uint8)</c> (truncated), opaque without alpha.</summary>
    static byte[] Rgba(ImageBatch images, int index)
    {
        int n = images.Height * images.Width, ch = images.Channels;
        var px = new byte[n * 4];
        int o = index * n * ch;
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < 3; c++) px[i * 4 + c] = (byte)Math.Clamp(255f * images.Data[o + i * ch + c], 0f, 255f);
            px[i * 4 + 3] = ch == 4 ? (byte)Math.Clamp(255f * images.Data[o + i * ch + 3], 0f, 255f) : (byte)255;
        }
        return px;
    }
}
