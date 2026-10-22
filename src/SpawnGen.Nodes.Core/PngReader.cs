using System.Buffers.Binary;
using System.IO.Compression;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// Minimal PNG decoder (non-interlaced; grayscale, grayscale+alpha, RGB, RGBA and palette; 8 and 16 bit) so loading an image needs no imaging
/// dependency. The result is an RGB <see cref="ImageBatch"/> in 0..1, like ComfyUI's LoadImage IMAGE output; the alpha comes back separately (it becomes the MASK output).
/// </summary>
public static class PngReader
{
    static readonly byte[] Signature = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    /// <summary>
    /// The text entries of a PNG (tEXt, zTXt and iTXt chunks) by keyword. ComfyUI stores the run's <c>prompt</c> and the editor's <c>workflow</c>
    /// this way, so a generated picture carries the graph that made it. Returns an empty map for a file that is not a PNG or has none.
    /// </summary>
    public static IReadOnlyDictionary<string, string> ReadText(byte[] png)
    {
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        if (png.Length < 8 || !png.AsSpan(0, 8).SequenceEqual(Signature)) return result;
        for (int pos = 8; pos + 12 <= png.Length;)
        {
            int len = (int)BinaryPrimitives.ReadUInt32BigEndian(png.AsSpan(pos));
            if (len < 0 || pos + 12 + len > png.Length) break;
            string type = System.Text.Encoding.ASCII.GetString(png, pos + 4, 4);
            var data = png.AsSpan(pos + 8, len);
            switch (type)
            {
                case "tEXt":
                {
                    int z = data.IndexOf((byte)0);
                    if (z > 0) result[System.Text.Encoding.Latin1.GetString(data[..z])] = System.Text.Encoding.Latin1.GetString(data[(z + 1)..]);
                    break;
                }
                case "zTXt":
                {
                    int z = data.IndexOf((byte)0);
                    if (z > 0 && z + 2 <= data.Length && Inflate(data[(z + 2)..]) is { } bytes)
                        result[System.Text.Encoding.Latin1.GetString(data[..z])] = System.Text.Encoding.Latin1.GetString(bytes);
                    break;
                }
                case "iTXt":
                {
                    int z = data.IndexOf((byte)0);
                    if (z <= 0 || z + 3 > data.Length) break;
                    bool compressed = data[z + 1] == 1;
                    var rest = data[(z + 3)..];
                    int lang = rest.IndexOf((byte)0);
                    if (lang < 0) break;
                    rest = rest[(lang + 1)..];
                    int translated = rest.IndexOf((byte)0);
                    if (translated < 0) break;
                    var body = rest[(translated + 1)..];
                    var text = compressed ? (Inflate(body) is { } b ? System.Text.Encoding.UTF8.GetString(b) : null) : System.Text.Encoding.UTF8.GetString(body);
                    if (text is not null) result[System.Text.Encoding.UTF8.GetString(data[..z])] = text;
                    break;
                }
            }
            if (type == "IEND") break;
            pos += 12 + len;
        }
        return result;
    }

    static byte[]? Inflate(ReadOnlySpan<byte> compressed)
    {
        try
        {
            using var input = new MemoryStream(compressed.ToArray());
            using var z = new ZLibStream(input, CompressionMode.Decompress);
            using var output = new MemoryStream();
            z.CopyTo(output);
            return output.ToArray();
        }
        catch (InvalidDataException) { return null; }
    }

    public static ImageBatch Decode(byte[] png) => Decode(png, out _);

    /// <summary>Decodes the picture and its alpha (0 transparent .. 1 opaque, one value per pixel); <paramref name="alpha"/> is null when the file has none.</summary>
    public static ImageBatch Decode(byte[] png, out float[]? alpha)
    {
        if (png.Length < 8 || !png.AsSpan(0, 8).SequenceEqual(Signature))
            throw new InvalidDataException("Not a PNG file (JPEG and other formats are not supported yet).");

        int width = 0, height = 0, depth = 0, colorType = -1, interlace = 0;
        byte[]? palette = null, paletteAlpha = null;
        using var idat = new MemoryStream();
        for (int pos = 8; pos + 12 <= png.Length;)
        {
            int len = (int)BinaryPrimitives.ReadUInt32BigEndian(png.AsSpan(pos));
            string type = System.Text.Encoding.ASCII.GetString(png, pos + 4, 4);
            var data = png.AsSpan(pos + 8, len);
            switch (type)
            {
                case "IHDR":
                    width = (int)BinaryPrimitives.ReadUInt32BigEndian(data);
                    height = (int)BinaryPrimitives.ReadUInt32BigEndian(data[4..]);
                    depth = data[8]; colorType = data[9]; interlace = data[12];
                    break;
                case "PLTE": palette = data.ToArray(); break;
                case "tRNS": paletteAlpha = data.ToArray(); break;
                case "IDAT": idat.Write(data); break;
            }
            if (type == "IEND") break;
            pos += 12 + len;
        }

        if (width <= 0 || height <= 0 || colorType < 0) throw new InvalidDataException("PNG has no valid IHDR.");
        if (interlace != 0) throw new NotSupportedException("Interlaced PNG is not supported.");
        if (depth is not (8 or 16)) throw new NotSupportedException($"PNG bit depth {depth} is not supported (8 and 16 are).");
        int samples = colorType switch { 0 => 1, 2 => 3, 3 => 1, 4 => 2, 6 => 4, _ => throw new InvalidDataException($"Bad PNG color type {colorType}.") };
        if (colorType == 3 && (depth != 8 || palette is null)) throw new NotSupportedException("Only 8-bit palette PNGs with a PLTE chunk are supported.");

        int bytesPerPixel = samples * depth / 8, stride = width * bytesPerPixel;
        var raw = new byte[height * (stride + 1)];
        idat.Position = 0;
        using (var z = new ZLibStream(idat, CompressionMode.Decompress))
            z.ReadExactly(raw);

        // Undo the per-row filters in place, in a pixel buffer without the filter bytes.
        var px = new byte[height * stride];
        for (int y = 0; y < height; y++)
        {
            int filter = raw[y * (stride + 1)];
            int src = y * (stride + 1) + 1, dst = y * stride, prev = dst - stride;
            for (int x = 0; x < stride; x++)
            {
                int a = x >= bytesPerPixel ? px[dst + x - bytesPerPixel] : 0;
                int b = y > 0 ? px[prev + x] : 0;
                int c = x >= bytesPerPixel && y > 0 ? px[prev + x - bytesPerPixel] : 0;
                int v = raw[src + x];
                px[dst + x] = (byte)(filter switch
                {
                    0 => v,
                    1 => v + a,
                    2 => v + b,
                    3 => v + ((a + b) >> 1),
                    4 => v + Paeth(a, b, c),
                    _ => throw new InvalidDataException($"Bad PNG filter type {filter}."),
                });
            }
        }

        var img = new ImageBatch(1, height, width, 3);
        alpha = colorType is 4 or 6 || (colorType == 3 && paletteAlpha is not null) ? new float[width * height] : null;
        int step = depth / 8;
        for (int i = 0; i < width * height; i++)
        {
            int o = i * bytesPerPixel;
            float Sample(int k) => px[o + k * step] / 255f; // 16-bit: the high byte is enough for 0..1 floats
            float r, g, bl;
            switch (colorType)
            {
                case 0: case 4: r = g = bl = Sample(0); break;
                case 3: { int p = px[o] * 3; r = palette![p] / 255f; g = palette[p + 1] / 255f; bl = palette[p + 2] / 255f; break; }
                default: r = Sample(0); g = Sample(1); bl = Sample(2); break;
            }
            img.Data[i * 3] = r; img.Data[i * 3 + 1] = g; img.Data[i * 3 + 2] = bl;
            if (alpha is not null)
                alpha[i] = colorType == 3 ? (px[o] < paletteAlpha!.Length ? paletteAlpha[px[o]] / 255f : 1f) : Sample(samples - 1);
        }
        return img;
    }

    static int Paeth(int a, int b, int c)
    {
        int p = a + b - c, pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
        return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
    }
}
