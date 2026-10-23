using System.Buffers.Binary;
using System.IO.Compression;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>Animated PNG (APNG), written the way Pillow's <c>save(save_all=True)</c> does for ComfyUI's Save Animated PNG.</summary>
public static partial class PngWriter
{
    /// <summary>
    /// Every picture of the batch as one APNG frame showing for <paramref name="durationMs"/> (Pillow's rules: pixels are <c>255 * x</c> truncated like
    /// ComfyUI's PIL conversion; a frame equal to the one before is merged into it (the durations add up); later frames store only the rectangle that
    /// changed, drawn over the last one; a single remaining frame is written as a still PNG). <paramref name="chunksAfterImage"/> go after the image data
    /// (ComfyUI's private <c>comf</c> metadata chunks). The animation loops forever.
    /// </summary>
    public static byte[] EncodeAnimated(ImageBatch images, int durationMs, int compressLevel = 4, IReadOnlyList<(string Type, byte[] Data)>? chunksAfterImage = null)
    {
        int w = images.Width, h = images.Height, ch = images.Channels;
        if (ch is not (3 or 4)) throw new ArgumentException($"PNG output needs 3 or 4 channels, got {ch}.", nameof(images));
        if (compressLevel is < 0 or > 9) throw new ArgumentOutOfRangeException(nameof(compressLevel), "compress_level is 0 to 9.");
        int n = w * h * ch;

        var frames = new List<(byte[] Pixels, (int X, int Y, int W, int H)? Box, int Duration)>();
        for (int i = 0; i < images.Batch; i++)
        {
            var px = new byte[n];
            for (int k = 0; k < n; k++) px[k] = (byte)Math.Clamp(255f * images.Data[i * n + k], 0f, 255f);
            if (frames.Count > 0)
            {
                var box = ChangedBox(frames[^1].Pixels, px, w, h, ch);
                if (box is null) { frames[^1] = frames[^1] with { Duration = frames[^1].Duration + durationMs }; continue; }
                frames.Add((px, box, durationMs));
            }
            else frames.Add((px, null, durationMs));
        }

        using var png = new MemoryStream();
        png.Write([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        var ihdr = new byte[13];
        BinaryPrimitives.WriteUInt32BigEndian(ihdr.AsSpan(0), (uint)w);
        BinaryPrimitives.WriteUInt32BigEndian(ihdr.AsSpan(4), (uint)h);
        ihdr[8] = 8;
        ihdr[9] = (byte)(ch == 3 ? 2 : 6);
        WriteChunk(png, "IHDR", ihdr);

        if (frames.Count == 1) WriteChunk(png, "IDAT", Deflate(frames[0].Pixels, 0, 0, w, h, w, ch, compressLevel));
        else
        {
            var actl = new byte[8];
            BinaryPrimitives.WriteUInt32BigEndian(actl.AsSpan(0), (uint)frames.Count);
            BinaryPrimitives.WriteUInt32BigEndian(actl.AsSpan(4), 0); // loop forever
            WriteChunk(png, "acTL", actl);
            uint seq = 0;
            for (int f = 0; f < frames.Count; f++)
            {
                var (px, box, duration) = frames[f];
                var (bx, by, bw, bh) = box ?? (0, 0, w, h);
                if (duration > ushort.MaxValue) throw new ArgumentOutOfRangeException(nameof(durationMs), $"A frame lasts {duration} ms; APNG delays stop at {ushort.MaxValue} ms (raise the fps).");
                var fctl = new byte[26];
                BinaryPrimitives.WriteUInt32BigEndian(fctl.AsSpan(0), seq++);
                BinaryPrimitives.WriteUInt32BigEndian(fctl.AsSpan(4), (uint)bw);
                BinaryPrimitives.WriteUInt32BigEndian(fctl.AsSpan(8), (uint)bh);
                BinaryPrimitives.WriteUInt32BigEndian(fctl.AsSpan(12), (uint)bx);
                BinaryPrimitives.WriteUInt32BigEndian(fctl.AsSpan(16), (uint)by);
                BinaryPrimitives.WriteUInt16BigEndian(fctl.AsSpan(20), (ushort)duration);
                BinaryPrimitives.WriteUInt16BigEndian(fctl.AsSpan(22), 1000);
                fctl[24] = 0; // dispose: none
                fctl[25] = 0; // blend: source
                WriteChunk(png, "fcTL", fctl);
                var data = Deflate(px, bx, by, bw, bh, w, ch, compressLevel);
                if (f == 0) WriteChunk(png, "IDAT", data); // the first frame is the still picture too
                else
                {
                    var fdat = new byte[4 + data.Length];
                    BinaryPrimitives.WriteUInt32BigEndian(fdat, seq++);
                    data.CopyTo(fdat, 4);
                    WriteChunk(png, "fdAT", fdat);
                }
            }
        }
        foreach (var (type, data) in chunksAfterImage ?? []) WriteChunk(png, type, data);
        WriteChunk(png, "IEND", []);
        return png.ToArray();
    }

    /// <summary>The bounding box (x, y, width, height) of the pixels that differ, or null when the frames are equal (Pillow's getbbox of the difference).</summary>
    static (int X, int Y, int W, int H)? ChangedBox(byte[] a, byte[] b, int w, int h, int ch)
    {
        int x0 = w, y0 = h, x1 = -1, y1 = -1;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int p = (y * w + x) * ch;
                bool same = true;
                for (int k = 0; k < ch && same; k++) same = a[p + k] == b[p + k];
                if (same) continue;
                x0 = Math.Min(x0, x); x1 = Math.Max(x1, x); y0 = Math.Min(y0, y); y1 = Math.Max(y1, y);
            }
        return x1 < 0 ? null : (x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    }

    /// <summary>The zlib stream of a rectangle of the picture, each row with filter type 0.</summary>
    static byte[] Deflate(byte[] px, int bx, int by, int bw, int bh, int w, int ch, int level)
    {
        var raw = new byte[bh * (1 + bw * ch)];
        for (int y = 0; y < bh; y++) Array.Copy(px, ((by + y) * w + bx) * ch, raw, y * (1 + bw * ch) + 1, bw * ch);
        using var ms = new MemoryStream();
        using (var z = new ZLibStream(ms, new ZLibCompressionOptions { CompressionLevel = level }, leaveOpen: true)) z.Write(raw);
        return ms.ToArray();
    }
}
