using System.Buffers.Binary;
using System.IO.Compression;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// Minimal PNG decoder (non-interlaced; grayscale, grayscale+alpha, RGB, RGBA and palette; 1, 2, 4 (grey and palette), 8 and 16 bit) so loading an image needs no imaging
/// dependency. The result is an RGB <see cref="ImageBatch"/> in 0..1, like ComfyUI's LoadImage IMAGE output; the alpha comes back separately (it becomes the MASK output).
/// </summary>
public static partial class PngReader
{
    static readonly byte[] Signature = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    /// <summary>
    /// The text entries of a PNG (tEXt, zTXt and iTXt chunks) by keyword. ComfyUI stores the run's <c>prompt</c> and the editor's <c>workflow</c>
    /// this way, so a generated picture carries the graph that made it. Returns an empty map for a file that is not a PNG or has none.
    /// </summary>
    /// <summary>
    /// <see cref="ReadText"/> for a file read piece by piece (<paramref name="read"/>(offset, count) returns those bytes): only the chunk headers
    /// and the text chunks are read, never the pixels, so a browser File stays in JS when the page only looks for its workflow (AGENTS.md
    /// rule 4). Null when the file is not a PNG.
    /// </summary>
    public static async Task<IReadOnlyDictionary<string, string>?> ReadTextAsync(long length, Func<long, int, Task<byte[]>> read)
    {
        if (length < 8 || !(await read(0, 8)).AsSpan().SequenceEqual(Signature)) return null;
        using var text = new MemoryStream();
        text.Write(Signature);
        for (long pos = 8; pos + 12 <= length;)
        {
            var header = await read(pos, 8);
            long len = BinaryPrimitives.ReadUInt32BigEndian(header);
            if (pos + 12 + len > length) break;
            string type = System.Text.Encoding.ASCII.GetString(header, 4, 4);
            if (type is "tEXt" or "zTXt" or "iTXt")
            {
                text.Write(header);
                text.Write(await read(pos + 8, checked((int)len + 4))); // the data and its CRC
            }
            if (type == "IEND") break;
            pos += 12 + len;
        }
        return ReadText(text.ToArray());
    }

    /// <summary>
    /// A PNG's colour type (IHDR) and EXIF orientation (eXIf, 1 without one) read piece by piece as <see cref="ReadTextAsync"/> does: chunk
    /// headers, IHDR and eXIf only, never the pixels. Null when the file is not a PNG. For a GPU decode of a file kept in JS.
    /// </summary>
    public static async Task<(int ColorType, int Orientation)?> ReadLayoutAsync(long length, Func<long, int, Task<byte[]>> read)
    {
        if (length < 33 || !(await read(0, 8)).AsSpan().SequenceEqual(Signature)) return null;
        int colorType = -1, orientation = 1;
        for (long pos = 8; pos + 12 <= length;)
        {
            var header = await read(pos, 8);
            long len = BinaryPrimitives.ReadUInt32BigEndian(header);
            if (pos + 12 + len > length) break;
            string type = System.Text.Encoding.ASCII.GetString(header, 4, 4);
            if (type == "IHDR" && len >= 13) colorType = (await read(pos + 8, 13))[9];
            else if (type == "eXIf" && len > 0 && len < 1 << 20) orientation = ExifOrientation.Read(await read(pos + 8, (int)len));
            else if (type == "IEND") break;
            pos += 12 + len;
        }
        return colorType < 0 ? null : (colorType, orientation);
    }

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
    public static ImageBatch Decode(byte[] png, out float[]? alpha) => Decode(png, out alpha, pillowRgb: false);

    /// <summary>The seven Adam7 passes (PNG spec 8.2): first column, first row, column step, row step.</summary>
    static readonly (int X0, int Y0, int Dx, int Dy)[] Adam7 = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)];

    /// <summary>
    /// As <see cref="Decode(byte[], out float[])"/>. By default the picture is decoded as the PNG specification (ISO/IEC 15948) defines it
    /// (TJ 2026-11-20: picture decoding is held to correctness): a sample of bit depth b is s / (2^b - 1), so 16-bit pictures keep their 16 bits;
    /// interlaced (Adam7) pictures are read; a grey or RGB tRNS colour makes those pixels fully transparent; the eXIf orientation is applied.
    /// With <paramref name="pillowRgb"/> it comes out as Pillow's <c>Image.open(...).convert("RGB")</c> makes it (measured with ComfyUI's Load
    /// Image Dataset from Folder, which uses Pillow): a 16-bit grey sample clipped at 255 (Pillow's mode I;16 is not scaled), other 16-bit
    /// samples cut to their high byte, the orientation not applied.
    /// </summary>
    public static ImageBatch Decode(byte[] png, out float[]? alpha, bool pillowRgb)
    {
        if (png.Length < 8 || !png.AsSpan(0, 8).SequenceEqual(Signature))
            throw new InvalidDataException("Not a PNG file (JPEG and other formats are not supported yet).");

        int width = 0, height = 0, depth = 0, colorType = -1, interlace = 0, orientation = 1;
        byte[]? palette = null, trns = null;
        using var idat = new MemoryStream();
        for (int pos = 8; pos + 12 <= png.Length;)
        {
            int len = (int)BinaryPrimitives.ReadUInt32BigEndian(png.AsSpan(pos));
            if (len < 0 || pos + 12 + len > png.Length) break;
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
                case "tRNS": trns = data.ToArray(); break;
                case "IDAT": idat.Write(data); break;
                case "eXIf": orientation = ExifOrientation.Read(data); break;
            }
            if (type == "IEND") break;
            pos += 12 + len;
        }

        if (width <= 0 || height <= 0 || colorType < 0) throw new InvalidDataException("PNG has no valid IHDR.");
        if (interlace is not (0 or 1)) throw new InvalidDataException($"Bad PNG interlace method {interlace}.");
        bool packed = depth is 1 or 2 or 4;
        if (!packed && depth is not (8 or 16)) throw new NotSupportedException($"PNG bit depth {depth} is not supported (1, 2, 4, 8 and 16 are).");
        if (packed && colorType is not (0 or 3)) throw new InvalidDataException($"Bad PNG: bit depth {depth} with color type {colorType}.");
        int samples = colorType switch { 0 => 1, 2 => 3, 3 => 1, 4 => 2, 6 => 4, _ => throw new InvalidDataException($"Bad PNG color type {colorType}.") };
        if (colorType == 3 && palette is null) throw new InvalidDataException("Bad PNG: a palette picture without a PLTE chunk.");

        // Every pass (one without interlacing, the seven Adam7 passes with it) is unfiltered on its own rows (spec 7.2, 9.2: samples below
        // 8 bits are packed, leftmost pixel in the high bits, each row starting on a byte; filters work on whole bytes, at least 1) and its
        // samples are put at their place in the picture, at the file's bit depth.
        int bitsPerPixel = samples * depth, bytesPerPixel = Math.Max(1, bitsPerPixel / 8);
        (int X0, int Y0, int Dx, int Dy)[] passes = interlace == 1 ? Adam7 : [(0, 0, 1, 1)];
        int RowBytes(int w) => (w * bitsPerPixel + 7) / 8;
        static int Count(int start, int step, int size) => start >= size ? 0 : (size - start + step - 1) / step;
        long rawLength = 0;
        foreach (var p in passes)
        {
            int pw = Count(p.X0, p.Dx, width), ph = Count(p.Y0, p.Dy, height);
            if (pw > 0 && ph > 0) rawLength += (long)ph * (RowBytes(pw) + 1);
        }
        var raw = new byte[rawLength];
        idat.Position = 0;
        using (var z = new ZLibStream(idat, CompressionMode.Decompress))
            z.ReadExactly(raw);

        var s = new int[width * height * samples];
        long offset = 0;
        foreach (var (x0, y0, dx, dy) in passes)
        {
            int pw = Count(x0, dx, width), ph = Count(y0, dy, height);
            if (pw == 0 || ph == 0) continue;
            int stride = RowBytes(pw);
            var px = Unfilter(raw[(int)offset..(int)(offset + (long)ph * (stride + 1))], ph, stride, bytesPerPixel);
            offset += (long)ph * (stride + 1);
            for (int j = 0; j < ph; j++)
            {
                int row = j * stride, y = y0 + j * dy;
                for (int i = 0; i < pw; i++)
                {
                    int to = (y * width + x0 + i * dx) * samples;
                    for (int c = 0; c < samples; c++)
                        s[to + c] = depth switch
                        {
                            8 => px[row + i * samples + c],
                            16 => (px[row + 2 * (i * samples + c)] << 8) | px[row + 2 * (i * samples + c) + 1],
                            _ => (px[row + i * depth / 8] >> (8 - depth - i * depth % 8)) & ((1 << depth) - 1),
                        };
                }
            }
        }

        // A sample as 0..1: s / (2^depth - 1) (spec 13.12); Pillow's convert("RGB") keeps 8 bits (a 16-bit grey sample clipped at 255).
        float top = (1 << depth) - 1;
        float Level(int v) => !pillowRgb || depth != 16 ? v / top : (v >> 8) / 255f;
        float Grey(int v) => pillowRgb && depth == 16 ? Math.Min(v, 255) / 255f : v / top;
        int[]? key = colorType is 0 or 2 && trns is { } t && t.Length >= samples * 2
            ? [.. Enumerable.Range(0, samples).Select(k => (t[2 * k] << 8) | t[2 * k + 1])] : null;

        var img = new ImageBatch(1, height, width, 3);
        alpha = colorType is 4 or 6 || (colorType == 3 && trns is not null) || key is not null ? new float[width * height] : null; // ImageDecoder gives every palette picture one
        for (int i = 0; i < width * height; i++)
        {
            int o = i * samples;
            float r, g, bl;
            switch (colorType)
            {
                case 0: case 4: r = g = bl = Grey(s[o]); break;
                case 3:
                {
                    int p = s[o] * 3;
                    if (p + 2 >= palette!.Length) throw new InvalidDataException($"Bad PNG: palette index {s[o]} past the {palette.Length / 3} entries of PLTE.");
                    r = palette[p] / 255f; g = palette[p + 1] / 255f; bl = palette[p + 2] / 255f;
                    break;
                }
                default: r = Level(s[o]); g = Level(s[o + 1]); bl = Level(s[o + 2]); break;
            }
            img.Data[i * 3] = r; img.Data[i * 3 + 1] = g; img.Data[i * 3 + 2] = bl;
            if (alpha is null) continue;
            alpha[i] = colorType switch
            {
                3 => trns is not null && s[o] < trns.Length ? trns[s[o]] / 255f : 1f,
                4 or 6 => Level(s[o + samples - 1]),
                _ => s.AsSpan(o, samples).SequenceEqual(key) ? 0f : 1f,
            };
        }
        if (pillowRgb || orientation == 1) return img;
        if (alpha is not null) alpha = ExifOrientation.Apply(alpha, width, height, orientation);
        return ExifOrientation.Apply(img, orientation);
    }

    static int Paeth(int a, int b, int c)
    {
        int p = a + b - c, pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
        return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
    }
}
