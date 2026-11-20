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
        int w = images.Width, h = images.Height;
        var prefix = SavePrefix.ComputeVars(filenamePrefix, w, h, DateTime.Now);
        var store = Results.For(services);
        var saved = (await EncodeOnDeviceAsync(images, fps, quality, accelerator, lossless, m, ExifWriter.ComfyMetadata(Results.Metadata(context)),
            file => store.SaveDeviceAsync("output", prefix, file, "webp"))).AsImage(w, h);
        var ui = new UiOutput("images", new object[] { saved });
        if (images.Batch > 1) ui["animated"] = new object[] { true };
        return (images, ui);
    }

    /// <summary>The batch as a WebP file (animated for more than one picture), each frame int(1000 / fps) ms; <paramref name="method"/> is libwebp's 0-6.</summary>
    public static Task<byte[]> EncodeAsync(ImageBatch images, double fps, int quality, Accelerator accelerator,
        bool lossless = true, int method = 4, byte[]? exif = null) =>
        EncodeOnDeviceAsync(images, fps, quality, accelerator, lossless, method, exif, file => file.CopyToHostAsync());

    /// <summary>
    /// <see cref="EncodeAsync"/> on the device: <paramref name="consume"/> gets exactly the file as a device view, valid while it runs (an animation
    /// is Codecs' EncodeAnimationToDeviceAsync: a view of the encoder's own buffer, no worst-case buffer; a single picture uses Codecs' bound).
    /// The pictures go to RGBA8 in <see cref="ImageKernels.ToRgba8"/> (ComfyUI's truncation) and the encoder reads them there.
    /// </summary>
    public static async Task<T> EncodeOnDeviceAsync<T>(ImageBatch images, double fps, int quality, Accelerator accelerator, bool lossless, int method,
        byte[]? exif, Func<ArrayView1D<byte, Stride1D.Dense>, Task<T>> consume)
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
        int[] durations = [.. Enumerable.Repeat(duration, n)];
        if (lossless)
        {
            using var enc = new WebPLosslessEncoderGpu(accelerator);
            // Exact = false is libwebp's default (it may change the colour under fully transparent pixels).
            if (n > 1)
                return await consume(await enc.EncodeAnimationToDeviceAsync(frames, w, h, durations,
                    new WebPEncodeOptions { Quality = quality, Exact = false }, new WebPAnimEncoderOptions { Exif = exif }));
            var still = new WebPEncodeOptions { Quality = quality, Exact = false, Exif = exif };
            using var file = accelerator.Allocate1D<byte>(WebPLosslessEncoderGpu.GetMaxEncodedSize(w, h, still));
            int length = await enc.EncodeAsync(frames[0], w, h, file.View, still);
            return await consume(file.View.SubView(0, length));
        }
        else
        {
            using var enc = new WebPLossyEncoderGpu(accelerator);
            if (n > 1)
                return await consume(await enc.EncodeAnimationToDeviceAsync(frames, w, h, durations,
                    new WebPLossyEncodeOptions { Quality = quality, Method = m }, new WebPAnimEncoderOptions { Exif = exif }));
            // A single picture takes Pillow's still path, which strips the "Exif\0\0" prefix before libwebp writes the chunk.
            var still = new WebPLossyEncodeOptions { Quality = quality, Method = m, Exif = exif[6..] };
            using var file = accelerator.Allocate1D<byte>(WebPLossyEncoderGpu.GetMaxEncodedSize(w, h, still));
            int length = await enc.EncodeAsync(frames[0], w, h, file.View, still);
            return await consume(file.View.SubView(0, length));
        }
    }
}
