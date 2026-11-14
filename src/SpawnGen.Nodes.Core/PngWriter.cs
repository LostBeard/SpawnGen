using System.Buffers.Binary;
using System.IO.Compression;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>Minimal PNG encoder (grey + alpha, RGB, RGBA; 8 or 16 bits per sample), so saving an image needs no imaging dependency.</summary>
public static partial class PngWriter
{
    static readonly uint[] CrcTable = BuildCrcTable();

    /// <summary>
    /// Encodes image <paramref name="index"/> of the batch (NHWC, 0..1) as a PNG. Samples are value x 255 (or x 65535 for 16 bits) in float32,
    /// clamped and truncated, as ComfyUI's Save Image and Save Image (Advanced) write them (measured at every 16-bit level, 2026-11-13).
    /// <paramref name="aspectChunk"/> adds the pHYs chunk Save Image (Advanced)'s files carry (0 x 1 per unknown unit, measured), after the text.
    /// </summary>
    public static byte[] Encode(ImageBatch images, int index = 0, IReadOnlyDictionary<string, string>? text = null, int bitDepth = 8, bool aspectChunk = false)
    {
        int w = images.Width, h = images.Height, ch = images.Channels;
        if (ch is not (2 or 3 or 4)) throw new ArgumentException($"PNG output needs 2, 3 or 4 channels, got {ch}.", nameof(images));
        if (bitDepth is not (8 or 16)) throw new ArgumentOutOfRangeException(nameof(bitDepth), bitDepth, "PNG output is 8 or 16 bits per sample.");
        int bytes = bitDepth / 8;
        var raw = new byte[h * (1 + w * ch * bytes)];
        int o = 0, src = index * h * w * ch;
        for (int y = 0; y < h; y++)
        {
            raw[o++] = 0; // filter: none
            if (bitDepth == 8)
                for (int x = 0; x < w * ch; x++)
                    raw[o++] = (byte)Math.Clamp(images.Data[src++] * 255f, 0f, 255f); // ComfyUI: np.clip(255. * i, 0, 255).astype(np.uint8), truncated, not rounded
            else
                for (int x = 0; x < w * ch; x++)
                {
                    var v = (ushort)Math.Clamp(images.Data[src++] * 65535f, 0f, 65535f); // float32 product, truncated (measured: a float64 product differs at 32895 levels)
                    raw[o++] = (byte)(v >> 8);
                    raw[o++] = (byte)v;
                }
        }

        using var idat = new MemoryStream();
        using (var z = new ZLibStream(idat, CompressionLevel.Fastest, leaveOpen: true))
            z.Write(raw);

        using var png = new MemoryStream();
        png.Write([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        var ihdr = new byte[13];
        BinaryPrimitives.WriteUInt32BigEndian(ihdr.AsSpan(0), (uint)w);
        BinaryPrimitives.WriteUInt32BigEndian(ihdr.AsSpan(4), (uint)h);
        ihdr[8] = (byte)bitDepth;
        ihdr[9] = (byte)(ch switch { 2 => 4, 3 => 2, _ => 6 }); // color type: grey + alpha (Pillow's LA) / RGB / RGBA
        WriteChunk(png, "IHDR", ihdr);
        if (text is not null)
            foreach (var (key, value) in text) WriteText(png, key, value);
        if (aspectChunk) WriteChunk(png, "pHYs", [0, 0, 0, 0, 0, 0, 0, 1, 0]);
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
