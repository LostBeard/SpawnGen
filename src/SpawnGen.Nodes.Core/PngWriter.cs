using System.Buffers.Binary;
using System.IO.Compression;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>Minimal 8-bit RGB PNG encoder, so saving an image needs no imaging dependency.</summary>
public static partial class PngWriter
{
    static readonly uint[] CrcTable = BuildCrcTable();

    /// <summary>Encodes image <paramref name="index"/> of the batch (NHWC, 0..1) as a PNG.</summary>
    public static byte[] Encode(ImageBatch images, int index = 0, IReadOnlyDictionary<string, string>? text = null)
    {
        int w = images.Width, h = images.Height, ch = images.Channels;
        if (ch is not (3 or 4)) throw new ArgumentException($"PNG output needs 3 or 4 channels, got {ch}.", nameof(images));
        var raw = new byte[h * (1 + w * ch)];
        int o = 0, src = index * h * w * ch;
        for (int y = 0; y < h; y++)
        {
            raw[o++] = 0; // filter: none
            for (int x = 0; x < w * ch; x++)
                raw[o++] = (byte)Math.Clamp(images.Data[src++] * 255f, 0f, 255f); // ComfyUI: np.clip(255. * i, 0, 255).astype(np.uint8), truncated, not rounded
        }

        using var idat = new MemoryStream();
        using (var z = new ZLibStream(idat, CompressionLevel.Fastest, leaveOpen: true))
            z.Write(raw);

        using var png = new MemoryStream();
        png.Write([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        var ihdr = new byte[13];
        BinaryPrimitives.WriteUInt32BigEndian(ihdr.AsSpan(0), (uint)w);
        BinaryPrimitives.WriteUInt32BigEndian(ihdr.AsSpan(4), (uint)h);
        ihdr[8] = 8;                      // bit depth
        ihdr[9] = (byte)(ch == 3 ? 2 : 6); // color type: RGB / RGBA
        WriteChunk(png, "IHDR", ihdr);
        if (text is not null)
            foreach (var (key, value) in text) WriteText(png, key, value);
        WriteChunk(png, "IDAT", idat.ToArray());
        WriteChunk(png, "IEND", []);
        return png.ToArray();
    }

    /// <summary>A text entry (ComfyUI keeps the prompt and the workflow this way): tEXt when it is Latin-1, else iTXt (UTF-8).</summary>
    static void WriteText(Stream s, string keyword, string value)
    {
        var key = System.Text.Encoding.Latin1.GetBytes(keyword);
        if (value.All(c => c <= 0xFF))
        {
            WriteChunk(s, "tEXt", [.. key, 0, .. System.Text.Encoding.Latin1.GetBytes(value)]);
            return;
        }
        // iTXt: keyword, 0, compression flag 0, method 0, empty language tag, 0, empty translated keyword, 0, text.
        WriteChunk(s, "iTXt", [.. key, 0, 0, 0, 0, 0, .. System.Text.Encoding.UTF8.GetBytes(value)]);
    }

    static void WriteChunk(Stream s, string type, byte[] data)
    {
        Span<byte> buf = stackalloc byte[4];
        BinaryPrimitives.WriteUInt32BigEndian(buf, (uint)data.Length);
        s.Write(buf);
        var typed = new byte[4 + data.Length];
        for (int i = 0; i < 4; i++) typed[i] = (byte)type[i];
        data.CopyTo(typed, 4);
        s.Write(typed);
        BinaryPrimitives.WriteUInt32BigEndian(buf, Crc(typed));
        s.Write(buf);
    }

    static uint Crc(byte[] data)
    {
        uint c = 0xFFFFFFFF;
        foreach (var b in data) c = CrcTable[(c ^ b) & 0xFF] ^ (c >> 8);
        return c ^ 0xFFFFFFFF;
    }

    static uint[] BuildCrcTable()
    {
        var t = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++) c = (c & 1) != 0 ? 0xEDB88320 ^ (c >> 1) : c >> 1;
            t[n] = c;
        }
        return t;
    }
}
