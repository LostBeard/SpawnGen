using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// Picture files the Load Image nodes read: PNG and JPEG here (exact to Pillow / libjpeg), and WebP, GIF and BMP through <see cref="Codecs"/>, which a host
/// with an accelerator sets (SpawnDev.Codecs).
/// </summary>
public static class ImageDecoder
{
    /// <summary>The file extensions the Load Image nodes list, lower case, with the dot.</summary>
    public static readonly string[] Extensions = [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"];

    public static bool IsSupportedName(string name) => Extensions.Contains(Path.GetExtension(name), StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// The decoder of the formats this class does not read itself (WebP, GIF, BMP): every frame of the file as the full canvas while it shows, RGBA8, or null
    /// when the data is not a format it knows. Set by the host (the server and the browser engine, on their accelerator).
    /// </summary>
    public static Func<byte[], Task<DecodedFrames?>>? Codecs { get; set; }

    /// <summary>The frames of a picture file: <see cref="Rgba"/> holds Width x Height x 4 bytes per frame.</summary>
    public sealed record DecodedFrames(int Width, int Height, bool HasAlpha, IReadOnlyList<byte[]> Rgba);

    /// <summary>
    /// Decodes by content, not by name: every frame as one picture of the batch (animated PNG, WebP and GIF), with its transparency (1 = opaque) or null for a
    /// file without it, as ComfyUI's Load Image reads frames.
    /// </summary>
    public static async Task<(ImageBatch Images, float[]? Alpha)> DecodeFramesAsync(byte[] data)
    {
        if (IsPng(data) && PngReader.DecodeAnimation(data) is { } apng)
            return FromFrames(new DecodedFrames(apng.Width, apng.Height, apng.HasAlpha, apng.Frames));
        if (JpegReader.IsJpeg(data) || IsPng(data)) return (Decode(data, out var alpha), alpha);
        var frames = Codecs is { } codecs ? await codecs(data) : null;
        if (frames is null)
        {
            if (Codecs is null && Format(data) is { } known)
                throw new NotSupportedException($"{known} pictures are decoded on the GPU (SpawnDev.Codecs); this host has not set a decoder for them.");
            throw new InvalidDataException("The file is not a PNG, JPEG, WebP, GIF or BMP picture.");
        }
        return FromFrames(frames);
    }

    static (ImageBatch Images, float[]? Alpha) FromFrames(DecodedFrames frames)
    {
        int w = frames.Width, h = frames.Height, n = frames.Rgba.Count;
        var images = new ImageBatch(n, h, w, 3);
        float[]? alphaOut = frames.HasAlpha ? new float[n * h * w] : null;
        for (int f = 0; f < n; f++)
        {
            var px = frames.Rgba[f];
            for (int i = 0; i < h * w; i++)
            {
                int o = (f * h * w + i) * 3;
                images.Data[o] = px[i * 4] / 255f;
                images.Data[o + 1] = px[i * 4 + 1] / 255f;
                images.Data[o + 2] = px[i * 4 + 2] / 255f;
                if (alphaOut is not null) alphaOut[f * h * w + i] = px[i * 4 + 3] / 255f;
            }
        }
        return (images, alphaOut);
    }

    /// <summary>The first frame only (Load Image (as Mask), like ComfyUI's, which reads one frame).</summary>
    public static async Task<(ImageBatch Image, float[]? Alpha)> DecodeFirstAsync(byte[] data)
    {
        var (images, alpha) = await DecodeFramesAsync(data);
        if (images.Batch == 1) return (images, alpha);
        int n = images.Height * images.Width;
        var first = new ImageBatch(1, images.Height, images.Width, 3);
        Array.Copy(images.Data, first.Data, n * 3);
        return (first, alpha?[..n]);
    }

    /// <summary>A PNG or JPEG, decoded here (no GPU needed). <paramref name="alpha"/> is null for pictures without transparency (all JPEGs).</summary>
    public static ImageBatch Decode(byte[] data, out float[]? alpha)
    {
        if (JpegReader.IsJpeg(data)) { alpha = null; return JpegReader.Decode(data); }
        if (IsPng(data)) return PngReader.Decode(data, out alpha);
        throw new InvalidDataException(Format(data) is { } known ? $"{known} pictures are decoded with DecodeFramesAsync (on the GPU)." : "The file is not a PNG or JPEG picture.");
    }

    public static ImageBatch Decode(byte[] data) => Decode(data, out _);

    /// <summary>
    /// A picture as Pillow's <c>Image.open(path).convert("RGB")</c> gives it (ComfyUI's dataset loaders, which do not use Load Image's path): the EXIF
    /// orientation is not applied, alpha is dropped, and a 16-bit grey PNG is clipped at 255 instead of scaled.
    /// </summary>
    public static ImageBatch DecodePillowRgb(byte[] data)
    {
        if (JpegReader.IsJpeg(data)) return JpegReader.Decode(data, applyOrientation: false);
        if (IsPng(data)) return PngReader.Decode(data, out _, pillowGray16: true);
        return Decode(data, out _); // the same refusal as Load Image for anything else
    }

    /// <summary><see cref="DecodePillowRgb"/>, with WebP (and GIF, BMP) through <see cref="Codecs"/>: the first frame, alpha dropped.</summary>
    public static async Task<ImageBatch> DecodePillowRgbAsync(byte[] data) =>
        JpegReader.IsJpeg(data) || IsPng(data) ? DecodePillowRgb(data) : (await DecodeFirstAsync(data)).Image;

    static bool IsPng(byte[] data) => data.Length >= 8 && data[0] == 0x89 && data[1] == 0x50;

    static string? Format(byte[] d) =>
        d.Length >= 12 && d[0] == 'R' && d[1] == 'I' && d[8] == 'W' && d[9] == 'E' ? "WebP"
        : d.Length >= 6 && d[0] == 'G' && d[1] == 'I' && d[2] == 'F' ? "GIF"
        : d.Length >= 2 && d[0] == 'B' && d[1] == 'M' ? "BMP"
        : null;
}
