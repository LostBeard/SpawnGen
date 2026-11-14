using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Written from the OpenEXR file layout documentation (openexr.com "OpenEXR File Layout" and "Technical Introduction", BSD-3-Clause project docs) and
// from files ComfyUI's Save Image (Advanced) wrote (2026-11-13, tests/SpawnGen.Nodes.Tests/Fixtures/image_save_advanced): single-part scan-line
// files, ZIP compression (16 scan lines per chunk), the channels in name order (A, B, G, R), half or float samples, the prompt and workflow as
// string attributes, and the gamut's chromaticities. The transfer functions are the published ones: IEC 61966-2-1 (sRGB) and ITU-R BT.2100 (HLG).

/// <summary>The colour space a picture's values are in, for an OpenEXR file (which always holds scene-linear light).</summary>
public enum ExrInput
{
    /// <summary>sRGB-encoded Rec.709: the inverse sRGB transfer is applied.</summary>
    Srgb,
    /// <summary>HLG-encoded Rec.2020 (BT.2100): the inverse HLG OETF is applied; the file's chromaticities are Rec.2020's.</summary>
    Hlg,
    /// <summary>Already scene-linear Rec.709: written as it is.</summary>
    Linear,
}

/// <summary>OpenEXR writer: one picture of an IMAGE batch as a scan-line file with ZIP compression.</summary>
public static class ExrWriter
{
    const int LinesPerChunk = 16; // ZIP_COMPRESSION

    static readonly float[] Rec709 = [0.64f, 0.33f, 0.30f, 0.60f, 0.15f, 0.06f, 0.3127f, 0.3290f];
    static readonly float[] Rec2020 = [0.708f, 0.292f, 0.170f, 0.797f, 0.131f, 0.046f, 0.3127f, 0.3290f];

    /// <summary>
    /// Picture <paramref name="index"/> of the batch (NHWC; 3 channels RGB or 4 RGBA) as an OpenEXR file. The colour channels go through
    /// <paramref name="input"/>'s inverse transfer in float32 (alpha is written as it is, colours are not premultiplied); values outside 0..1 are kept.
    /// <paramref name="half"/> writes 16-bit floats (rounded to nearest even), otherwise 32-bit. <paramref name="text"/> entries become string
    /// attributes (the prompt and the workflow, as ComfyUI writes them).
    /// </summary>
    public static byte[] Encode(ImageBatch images, int index, ExrInput input, bool half, IReadOnlyDictionary<string, string>? text = null)
    {
        int w = images.Width, h = images.Height, ch = images.Channels;
        if (ch is not (3 or 4)) throw new ArgumentException($"OpenEXR output needs 3 (RGB) or 4 (RGBA) channels, got {ch}.", nameof(images));
        if (w <= 0 || h <= 0) throw new ArgumentException("OpenEXR output needs a picture of at least 1 x 1.", nameof(images));

        // Channels in name order: (A,) B, G, R, as their index into the picture's pixel.
        int[] order = ch == 4 ? [3, 2, 1, 0] : [2, 1, 0];
        string[] names = ch == 4 ? ["A", "B", "G", "R"] : ["B", "G", "R"];
        int sample = half ? 2 : 4;

        var header = new MemoryStream();
        Span<byte> magic = stackalloc byte[8];
        BinaryPrimitives.WriteInt32LittleEndian(magic, 20000630);
        BinaryPrimitives.WriteInt32LittleEndian(magic[4..], 2); // version 2, single-part scan-line, short names
        header.Write(magic);

        var chlist = new MemoryStream();
        foreach (var name in names)
        {
            chlist.Write(Encoding.ASCII.GetBytes(name));
            chlist.WriteByte(0);
            WriteInt(chlist, half ? 1 : 2); // pixel type: HALF / FLOAT
            chlist.Write([0, 0, 0, 0]);    // pLinear, reserved
            WriteInt(chlist, 1);            // x sampling
            WriteInt(chlist, 1);            // y sampling
        }
        chlist.WriteByte(0);
        Attribute(header, "channels", "chlist", chlist.ToArray());
        Attribute(header, "compression", "compression", [3]); // ZIP_COMPRESSION
        var box = Ints(0, 0, w - 1, h - 1);
        Attribute(header, "dataWindow", "box2i", box);
        Attribute(header, "displayWindow", "box2i", box);
        Attribute(header, "lineOrder", "lineOrder", [0]); // INCREASING_Y
        // pixelAspectRatio is required by the file layout; ComfyUI's files (written through libavcodec) leave it out.
        Attribute(header, "pixelAspectRatio", "float", Floats(1f));
        Attribute(header, "screenWindowCenter", "v2f", Floats(0f, 0f));
        Attribute(header, "screenWindowWidth", "float", Floats(1f));
        Attribute(header, "gamma", "float", Floats(1f));
        Attribute(header, "writer", "string", Encoding.UTF8.GetBytes("SpawnGen"));
        if (text is not null)
            foreach (var (key, value) in text) Attribute(header, key, "string", Encoding.UTF8.GetBytes(value));
        Attribute(header, "chromaticities", "chromaticities", Floats(input == ExrInput.Hlg ? Rec2020 : Rec709));
        header.WriteByte(0);

        int chunks = (h + LinesPerChunk - 1) / LinesPerChunk;
        var blocks = new byte[chunks][];
        int lineBytes = w * names.Length * sample;
        var raw = new byte[LinesPerChunk * lineBytes];
        int plane = w * h * ch, start = index * plane;
        for (int c = 0; c < chunks; c++)
        {
            int y0 = c * LinesPerChunk, lines = Math.Min(LinesPerChunk, h - y0), size = lines * lineBytes, o = 0;
            for (int y = y0; y < y0 + lines; y++)
                for (int k = 0; k < names.Length; k++)
                {
                    int channel = order[k];
                    bool colour = channel < 3;
                    for (int x = 0; x < w; x++)
                    {
                        float v = images.Data[start + (y * w + x) * ch + channel];
                        if (colour) v = ToLinear(v, input);
                        if (half) { BinaryPrimitives.WriteHalfLittleEndian(raw.AsSpan(o), (Half)v); o += 2; }
                        else { BinaryPrimitives.WriteSingleLittleEndian(raw.AsSpan(o), v); o += 4; }
                    }
                }
            var packed = Zip(raw.AsSpan(0, size));
            var block = new byte[8 + Math.Min(packed.Length, size)];
            BinaryPrimitives.WriteInt32LittleEndian(block, y0);
            BinaryPrimitives.WriteInt32LittleEndian(block.AsSpan(4), block.Length - 8);
            // A chunk that does not get smaller is stored uncompressed (its size then equals the raw size, which tells a reader so).
            if (packed.Length < size) packed.CopyTo(block, 8);
            else raw.AsSpan(0, size).CopyTo(block.AsSpan(8));
            blocks[c] = block;
        }

        var file = new MemoryStream();
        header.Position = 0;
        header.CopyTo(file);
        long offset = file.Length + 8L * chunks;
        Span<byte> entry = stackalloc byte[8];
        foreach (var block in blocks)
        {
            BinaryPrimitives.WriteInt64LittleEndian(entry, offset);
            file.Write(entry);
            offset += block.Length;
        }
        foreach (var block in blocks) file.Write(block);
        return file.ToArray();
    }

    /// <summary>A colour value of <paramref name="input"/> as scene-linear light, in float32.</summary>
    public static float ToLinear(float v, ExrInput input) => input switch
    {
        // IEC 61966-2-1, extended to negatives and above 1 as ComfyUI's files show: the linear segment below 0.04045, the power curve above.
        ExrInput.Srgb => v <= 0.04045f ? v / 12.92f : MathF.Pow((v + 0.055f) / 1.055f, 2.4f),
        // ITU-R BT.2100 HLG inverse OETF (a = 0.17883277, b = 1 - 4a, c = 0.5 - a ln 4a): E^2 / 3 up to 0.5, (exp((E - c) / a) + b) / 12 above.
        ExrInput.Hlg => v <= 0.5f ? v * v / 3f : (MathF.Exp((v - 0.55991073f) / 0.17883277f) + 0.28466892f) / 12f,
        _ => v,
    };

    /// <summary>
    /// ZIP compression of one chunk: the bytes split into even and odd positions (first half, second half), each byte replaced by its difference
    /// from the one before plus 128, then zlib.
    /// </summary>
    static byte[] Zip(ReadOnlySpan<byte> raw)
    {
        var t = new byte[raw.Length];
        int half = (raw.Length + 1) / 2;
        for (int i = 0, a = 0, b = half; i < raw.Length; i++)
            if ((i & 1) == 0) t[a++] = raw[i]; else t[b++] = raw[i];
        int previous = t.Length > 0 ? t[0] : 0;
        for (int i = 1; i < t.Length; i++)
        {
            int current = t[i];
            t[i] = (byte)(current - previous + 128 + 256);
            previous = current;
        }
        using var output = new MemoryStream();
        using (var z = new ZLibStream(output, CompressionLevel.Optimal, leaveOpen: true)) z.Write(t);
        return output.ToArray();
    }

    static void Attribute(Stream s, string name, string type, byte[] value)
    {
        s.Write(Encoding.UTF8.GetBytes(name));
        s.WriteByte(0);
        s.Write(Encoding.ASCII.GetBytes(type));
        s.WriteByte(0);
        WriteInt(s, value.Length);
        s.Write(value);
    }

    static void WriteInt(Stream s, int v)
    {
        Span<byte> b = stackalloc byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(b, v);
        s.Write(b);
    }

    static byte[] Ints(params int[] values)
    {
        var b = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++) BinaryPrimitives.WriteInt32LittleEndian(b.AsSpan(i * 4), values[i]);
        return b;
    }

    static byte[] Floats(params float[] values)
    {
        var b = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++) BinaryPrimitives.WriteSingleLittleEndian(b.AsSpan(i * 4), values[i]);
        return b;
    }
}
