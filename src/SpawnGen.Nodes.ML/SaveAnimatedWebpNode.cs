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
    /// the workflow in Make). Encoded on the GPU by SpawnDev.Codecs: lossy is libwebp's WebPAnimEncoder (quality, method default = 4); lossless writes every frame
    /// whole with its own VP8L coder (the pictures decode exactly; the file is larger than Pillow's). A single picture is saved as a still WebP, as Pillow does.
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
        int w = images.Width, h = images.Height, duration = (int)(1000.0 / fps);
        var frames = Enumerable.Range(0, images.Batch).Select(i => (ReadOnlyMemory<byte>)Rgba(images, i)).ToList();
        var exif = ExifWriter.ComfyMetadata(Results.Metadata(context));

        byte[] file;
        if (lossless)
        {
            using var enc = new WebPLosslessEncoderGpu(accelerator);
            var options = new WebPAnimationEncodeOptions { Quality = quality, Exif = exif };
            file = frames.Count == 1
                ? await enc.EncodeAsync(frames[0], w, h, options)
                : await enc.EncodeAnimationAsync(frames,
                    [.. Enumerable.Range(0, frames.Count).Select(i => new ImageFrameInfo(i, 0, 0, w, h, duration, FrameDispose.None, FrameBlend.Source))], w, h, options);
        }
        else
        {
            // A single lossy picture goes through the animation encoder too: the still lossy encoder takes no EXIF yet.
            using var enc = new WebPLossyEncoderGpu(accelerator);
            file = await enc.EncodeAnimationAsync(frames, w, h, [.. Enumerable.Repeat(duration, frames.Count)],
                new WebPLossyEncodeOptions { Quality = quality, Method = m }, new WebPLossyAnimationOptions { Exif = exif });
        }

        var prefix = SavePrefix.ComputeVars(filenamePrefix, w, h, DateTime.Now);
        var saved = await Results.For(services).SaveAsync("output", prefix, file, "webp");
        var ui = new UiOutput("images", new object[] { saved });
        if (images.Batch > 1) ui["animated"] = new object[] { true };
        return (images, ui);
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
