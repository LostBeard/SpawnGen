using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md) from SpawnGen's tests and fixtures and black-box measurements of ComfyUI's Load Image. PNG, APNG and JPEG
// are decoded here (PngReader, JpegReader); WebP, GIF and BMP go through the Codecs hook (SpawnDev.Codecs, installed by SpawnGen.Nodes.ML).

/// <summary>
/// Turns picture files into SpawnGen images: every frame (Load Image), the first frame, or the first frame as Pillow's <c>convert("RGB")</c> makes it
/// (the dataset nodes). Alpha comes back separately, one value per pixel and frame (0 transparent .. 1 opaque), or null when the file has none.
/// </summary>
public static class ImageDecoder
{
    /// <summary>The file extensions the Load Image nodes list.</summary>
    public static readonly string[] Extensions = [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"];

    /// <summary>True for a file name with one of <see cref="Extensions"/> (any case).</summary>
    public static bool IsSupportedName(string name) =>
        Array.IndexOf(Extensions, Path.GetExtension(name).ToLowerInvariant()) >= 0;

    /// <summary>
    /// Decoder for the formats this assembly does not read itself (WebP, GIF, BMP): every frame as full-canvas RGBA8, or null when it does not
    /// handle the file. Null when nothing is installed.
    /// </summary>
    public static Func<byte[], Task<DecodedFrames?>>? Codecs { get; set; }

    /// <summary>Frames as the <see cref="Codecs"/> decoder returns them: each one the whole canvas in RGBA8, row by row.</summary>
    public sealed record DecodedFrames(int Width, int Height, bool HasAlpha, IReadOnlyList<byte[]> Rgba);

    /// <summary>
    /// Every frame of the picture (an animated PNG, WebP or GIF has several) and their alpha, frame after frame; alpha is null when the file has
    /// no transparency. A palette PNG always has alpha (fully opaque without a tRNS chunk): measured with ComfyUI, its Load Image gives such a
    /// picture a full-size mask, not the empty 64 x 64 one.
    /// </summary>
    public static async Task<(ImageBatch Images, float[]? Alpha)> DecodeFramesAsync(byte[] data)
    {
        if (IsPng(data))
        {
            if (PngReader.DecodeAnimation(data) is { } animation)
                return FromRgba(animation.Width, animation.Height, animation.HasAlpha, animation.Frames);
            var still = DecodePng(data, out var alpha);
            return (still, alpha);
        }
        if (JpegReader.IsJpeg(data)) return (JpegReader.Decode(data), null);
        if (Codecs is { } codecs && await codecs(data) is { } frames)
            return FromRgba(frames.Width, frames.Height, frames.HasAlpha, frames.Rgba);
        throw Unsupported(data, codecsMissing: Codecs is null);
    }

    /// <summary>The first frame and its alpha (null without transparency).</summary>
    public static async Task<(ImageBatch Image, float[]? Alpha)> DecodeFirstAsync(byte[] data)
    {
        var (images, alpha) = await DecodeFramesAsync(data);
        if (images.Batch == 1) return (images, alpha);
        int n = images.Height * images.Width;
        var first = new ImageBatch(1, images.Height, images.Width, images.Channels, images.Data[..(n * images.Channels)]);
        return (first, alpha?[..n]);
    }

    /// <summary>A PNG (its default image) or JPEG, without the Codecs decoder; <paramref name="alpha"/> as in <see cref="DecodeFramesAsync"/>.</summary>
    public static ImageBatch Decode(byte[] data, out float[]? alpha)
    {
        if (IsPng(data)) return DecodePng(data, out alpha);
        alpha = null;
        if (JpegReader.IsJpeg(data)) return JpegReader.Decode(data);
        throw Unsupported(data, codecsMissing: false);
    }

    /// <summary>A PNG or JPEG as RGB, its alpha dropped.</summary>
    public static ImageBatch Decode(byte[] data) => Decode(data, out _);

    /// <summary>
    /// The first frame as Pillow's <c>Image.open(...).convert("RGB")</c> gives it: alpha dropped (not composited), a 16-bit grey PNG clipped at 255
    /// (PngReader's Pillow mode), a JPEG as stored (Pillow's open does not apply the EXIF orientation).
    /// </summary>
    public static ImageBatch DecodePillowRgb(byte[] data)
    {
        if (IsPng(data)) return PngReader.Decode(data, out _, pillowGray16: true);
        if (JpegReader.IsJpeg(data)) return JpegReader.Decode(data, applyOrientation: false);
        throw Unsupported(data, codecsMissing: false);
    }

    /// <summary><see cref="DecodePillowRgb"/>, with WebP, GIF and BMP read by the <see cref="Codecs"/> decoder (first frame, alpha dropped).</summary>
    public static async Task<ImageBatch> DecodePillowRgbAsync(byte[] data)
    {
        if (IsPng(data) || JpegReader.IsJpeg(data)) return DecodePillowRgb(data);
        if (Codecs is { } codecs && await codecs(data) is { } frames)
            return FromRgba(frames.Width, frames.Height, false, [frames.Rgba[0]]).Images;
        throw Unsupported(data, codecsMissing: Codecs is null);
    }

    static readonly byte[] PngSignature = [0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A];

    static bool IsPng(byte[] data) => data.Length >= 8 && data.AsSpan(0, 8).SequenceEqual(PngSignature);

    /// <summary>A still PNG; a palette picture (IHDR colour type 3, PNG spec 11.2.2) without a tRNS chunk gets an opaque alpha.</summary>
    static ImageBatch DecodePng(byte[] data, out float[]? alpha)
    {
        var image = PngReader.Decode(data, out alpha);
        const int colourTypeOffset = 8 + 8 + 9; // signature, IHDR length and type, width, height, bit depth
        if (alpha is null && data.Length > colourTypeOffset && data[colourTypeOffset] == 3)
        {
            alpha = new float[image.Batch * image.Height * image.Width];
            Array.Fill(alpha, 1f);
        }
        return image;
    }

    static (ImageBatch Images, float[]? Alpha) FromRgba(int width, int height, bool hasAlpha, IReadOnlyList<byte[]> frames)
    {
        int n = width * height;
        var images = new ImageBatch(frames.Count, height, width, 3);
        var alpha = hasAlpha ? new float[frames.Count * n] : null;
        for (int f = 0; f < frames.Count; f++)
        {
            var rgba = frames[f];
            for (int i = 0; i < n; i++)
            {
                int o = (f * n + i) * 3;
                images.Data[o] = rgba[i * 4] / 255f;
                images.Data[o + 1] = rgba[i * 4 + 1] / 255f;
                images.Data[o + 2] = rgba[i * 4 + 2] / 255f;
                if (alpha is not null) alpha[f * n + i] = rgba[i * 4 + 3] / 255f;
            }
        }
        return (images, alpha);
    }

    static Exception Unsupported(byte[] data, bool codecsMissing)
    {
        string? format = data.Length >= 12 && data.AsSpan(0, 4).SequenceEqual("RIFF"u8) && data.AsSpan(8, 4).SequenceEqual("WEBP"u8) ? "WebP"
            : data.Length >= 4 && data.AsSpan(0, 4).SequenceEqual("GIF8"u8) ? "GIF"
            : data.Length >= 2 && data[0] == (byte)'B' && data[1] == (byte)'M' ? "BMP"
            : null;
        if (format is null) return new InvalidDataException("Not a picture SpawnGen can read (PNG, JPEG, WebP, GIF or BMP).");
        string message = $"{format} pictures are decoded by SpawnDev.Codecs, which is not available here.";
        return codecsMissing ? new NotSupportedException(message) : new InvalidDataException(message);
    }
}
