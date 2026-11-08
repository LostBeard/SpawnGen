using System.Buffers.Binary;
using System.IO.Compression;

namespace SpawnGen.Nodes.Core;

public static partial class PngReader
{
    /// <summary>One animation frame's control (fcTL) and its compressed pixels (IDAT for the first frame when it is the default image, else fdAT).</summary>
    sealed record ApngFrame(int Width, int Height, int X, int Y, int Dispose, int Blend, MemoryStream Data);

    /// <summary>
    /// Every frame of an animated PNG as the full canvas while it shows, RGBA8, by the APNG specification with the frame choices ComfyUI's Load Image
    /// was measured to make: the canvas starts transparent black; before a frame, the one before is disposed (background: its rectangle cleared; previous:
    /// the canvas as it was before it); "source" frames, the first frame and frames disposed to the background replace their rectangle, other "over" frames
    /// are composited by the PNG specification's alpha compositing (<see cref="Draw"/>). Null for a PNG
    /// without an acTL chunk or with one frame (decode those with <see cref="Decode(byte[], out float[])"/>). <c>HasAlpha</c> is false for grey and RGB
    /// pictures without transparency (their alpha bytes are 255).
    /// </summary>
    public static (int Width, int Height, bool HasAlpha, List<byte[]> Frames)? DecodeAnimation(byte[] png)
    {
        if (png.Length < 8 || !png.AsSpan(0, 8).SequenceEqual(Signature)) return null;
        int width = 0, height = 0, depth = 0, colorType = -1, interlace = 0, numFrames = 0;
        byte[]? palette = null, paletteAlpha = null;
        var frames = new List<ApngFrame>();
        bool seenIdat = false;
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
                case "acTL": numFrames = (int)BinaryPrimitives.ReadUInt32BigEndian(data); break;
                case "fcTL":
                    frames.Add(new ApngFrame(
                        (int)BinaryPrimitives.ReadUInt32BigEndian(data[4..]), (int)BinaryPrimitives.ReadUInt32BigEndian(data[8..]),
                        (int)BinaryPrimitives.ReadUInt32BigEndian(data[12..]), (int)BinaryPrimitives.ReadUInt32BigEndian(data[16..]),
                        data[24], data[25], new MemoryStream()));
                    break;
                case "IDAT":
                    // The default image is the first frame only when its fcTL came before it.
                    if (frames.Count == 1 && !seenIdat) frames[0].Data.Write(data);
                    seenIdat = true;
                    break;
                case "fdAT":
                    if (frames.Count > 0) frames[^1].Data.Write(data[4..]); // after the sequence number
                    break;
            }
            if (type == "IEND") break;
            pos += 12 + len;
        }
        frames.RemoveAll(f => f.Data.Length == 0);
        if (numFrames <= 1 || frames.Count <= 1) return null;
        if (interlace != 0) throw new NotSupportedException("Interlaced PNG is not supported.");
        if (depth is not (8 or 16)) throw new NotSupportedException($"PNG bit depth {depth} is not supported (8 and 16 are).");
        bool hasAlpha = colorType is 4 or 6 || (colorType == 3 && paletteAlpha is not null);

        var canvas = new byte[width * height * 4];
        byte[]? saved = null;
        var output = new List<byte[]>();
        ApngFrame? previous = null;
        foreach (var f in frames)
        {
            if (f.X + f.Width > width || f.Y + f.Height > height) throw new InvalidDataException("An APNG frame lies outside the canvas.");
            if (previous is { Dispose: 1 }) Clear(canvas, width, previous);
            else if (previous is { Dispose: 2 } && saved is not null) Array.Copy(saved, canvas, canvas.Length);
            // A frame disposed to "previous" needs the canvas as it was before it (the first frame's "previous" means "background", as the format says).
            saved = f.Dispose == 2 && output.Count > 0 ? (byte[])canvas.Clone() : null;
            var rgba = Rgba8(f.Data.ToArray(), f.Width, f.Height, depth, colorType, palette, paletteAlpha);
            // Measured with ComfyUI's Load Image (PyAV) for every pair of disposals (Fixtures/codecs/apng, probe_*): the first frame is written as it is (its
            // colour kept under transparent pixels), and a frame that is itself disposed to the background is shown unblended.
            Draw(canvas, width, f, rgba, over: f.Blend == 1 && hasAlpha && output.Count > 0 && f.Dispose != 1);
            output.Add((byte[])canvas.Clone());
            previous = f.Dispose == 2 && output.Count == 1 ? f with { Dispose = 1 } : f;
        }
        return (width, height, hasAlpha, output);
    }

    static void Clear(byte[] canvas, int width, ApngFrame f)
    {
        for (int y = f.Y; y < f.Y + f.Height; y++) Array.Clear(canvas, (y * width + f.X) * 4, f.Width * 4);
    }

    /// <summary>
    /// Draws a frame's pixels onto the canvas: replaced (APNG_BLEND_OP_SOURCE), or composited over it (APNG_BLEND_OP_OVER), which the APNG specification
    /// defines as the PNG specification's alpha compositing: out_alpha = a + b_alpha * (1 - a), out_colour = (a * colour + b_alpha * (1 - a) * b_colour) /
    /// out_alpha, computed here in real arithmetic and rounded to the nearest 8-bit value.
    /// </summary>
    static void Draw(byte[] canvas, int width, ApngFrame f, byte[] rgba, bool over)
    {
        for (int y = 0; y < f.Height; y++)
            for (int x = 0; x < f.Width; x++)
            {
                int s = (y * f.Width + x) * 4, d = ((f.Y + y) * width + f.X + x) * 4;
                if (!over) { Array.Copy(rgba, s, canvas, d, 4); continue; }
                double a = rgba[s + 3] / 255.0, b = canvas[d + 3] / 255.0;
                if (a >= 1) { Array.Copy(rgba, s, canvas, d, 4); continue; }
                if (a <= 0) continue;
                double outAlpha = a + b * (1 - a);
                for (int c = 0; c < 3; c++)
                    canvas[d + c] = (byte)Math.Round((a * rgba[s + c] + b * (1 - a) * canvas[d + c]) / outAlpha, MidpointRounding.AwayFromZero);
                canvas[d + 3] = (byte)Math.Round(outAlpha * 255, MidpointRounding.AwayFromZero);
            }
    }

    /// <summary>One (sub-)image's pixels as RGBA8: filters undone, 16-bit samples by their high byte, palette and its tRNS applied, opaque otherwise.</summary>
    static byte[] Rgba8(byte[] zdata, int width, int height, int depth, int colorType, byte[]? palette, byte[]? paletteAlpha)
    {
        int samples = colorType switch { 0 => 1, 2 => 3, 3 => 1, 4 => 2, 6 => 4, _ => throw new InvalidDataException($"Bad PNG color type {colorType}.") };
        if (colorType == 3 && (depth != 8 || palette is null)) throw new NotSupportedException("Only 8-bit palette PNGs with a PLTE chunk are supported.");
        int bpp = samples * depth / 8, stride = width * bpp;
        var raw = new byte[height * (stride + 1)];
        using (var z = new ZLibStream(new MemoryStream(zdata), CompressionMode.Decompress)) z.ReadExactly(raw);
        var px = Unfilter(raw, height, stride, bpp);
        var rgba = new byte[width * height * 4];
        int step = depth / 8;
        for (int i = 0; i < width * height; i++)
        {
            int o = i * bpp, d = i * 4;
            byte S(int k) => px[o + k * step];
            switch (colorType)
            {
                case 0: rgba[d] = rgba[d + 1] = rgba[d + 2] = S(0); rgba[d + 3] = 255; break;
                case 4: rgba[d] = rgba[d + 1] = rgba[d + 2] = S(0); rgba[d + 3] = S(1); break;
                case 2: rgba[d] = S(0); rgba[d + 1] = S(1); rgba[d + 2] = S(2); rgba[d + 3] = 255; break;
                case 6: rgba[d] = S(0); rgba[d + 1] = S(1); rgba[d + 2] = S(2); rgba[d + 3] = S(3); break;
                case 3:
                    int p = px[o];
                    rgba[d] = palette![p * 3]; rgba[d + 1] = palette[p * 3 + 1]; rgba[d + 2] = palette[p * 3 + 2];
                    rgba[d + 3] = paletteAlpha is not null && p < paletteAlpha.Length ? paletteAlpha[p] : (byte)255;
                    break;
            }
        }
        return rgba;
    }

    /// <summary>Undoes the per-row filters, into a pixel buffer without the filter bytes.</summary>
    static byte[] Unfilter(byte[] raw, int height, int stride, int bytesPerPixel)
    {
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
        return px;
    }
}
