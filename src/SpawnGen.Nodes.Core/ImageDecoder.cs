using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>Picture files the Load Image nodes read: PNG and JPEG (WebP and others are not supported yet).</summary>
public static class ImageDecoder
{
    /// <summary>The file extensions <see cref="Decode"/> reads, lower case, with the dot.</summary>
    public static readonly string[] Extensions = [".png", ".jpg", ".jpeg"];

    public static bool IsSupportedName(string name) => Extensions.Contains(Path.GetExtension(name), StringComparer.OrdinalIgnoreCase);

    /// <summary>Decodes by content, not by name. <paramref name="alpha"/> is null for pictures without transparency (all JPEGs).</summary>
    public static ImageBatch Decode(byte[] data, out float[]? alpha)
    {
        if (JpegReader.IsJpeg(data)) { alpha = null; return JpegReader.Decode(data); }
        if (data.Length >= 8 && data[0] == 0x89 && data[1] == 0x50) return PngReader.Decode(data, out alpha);
        if (data.Length >= 12 && data[0] == 'R' && data[1] == 'I' && data[8] == 'W' && data[9] == 'E') throw new NotSupportedException("WebP pictures cannot be loaded yet; use PNG or JPEG.");
        throw new InvalidDataException("The file is not a PNG or JPEG picture.");
    }

    public static ImageBatch Decode(byte[] data) => Decode(data, out _);

    /// <summary>
    /// A picture as Pillow's <c>Image.open(path).convert("RGB")</c> gives it (ComfyUI's dataset loaders, which do not use Load Image's path): the EXIF
    /// orientation is not applied, alpha is dropped, and a 16-bit grey PNG is clipped at 255 instead of scaled.
    /// </summary>
    public static ImageBatch DecodePillowRgb(byte[] data)
    {
        if (JpegReader.IsJpeg(data)) return JpegReader.Decode(data, applyOrientation: false);
        if (data.Length >= 8 && data[0] == 0x89 && data[1] == 0x50) return PngReader.Decode(data, out _, pillowGray16: true);
        return Decode(data, out _); // the same refusal as Load Image for anything else
    }
}
