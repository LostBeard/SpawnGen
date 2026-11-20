using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// The EXIF orientation (TIFF tag 274, 0x0112) and turning a picture upright by it. JPEG carries it in APP1 ("Exif\0\0" then the TIFF data),
/// PNG in the eXIf chunk (the TIFF data alone). 1 = as stored; 2 mirror; 3 rotate 180; 4 flip; 5 transpose; 6 rotate 90 cw; 7 transverse;
/// 8 rotate 270 cw (the TIFF 6.0 / EXIF 2.32 definitions). Applied in full, mirrors too (TJ 2026-11-20: decoding is held to correctness;
/// ComfyUI's FFmpeg-based Load Image drops the mirrors, listed in docs/PARITY.md).
/// </summary>
public static class ExifOrientation
{
    /// <summary>The orientation in TIFF data (the "II*\0" / "MM\0*" header onwards), 1 when it is missing or not 1..8.</summary>
    public static int Read(ReadOnlySpan<byte> tiff)
    {
        if (tiff.Length < 8) return 1;
        bool little = tiff[0] == 'I';
        int U16(ReadOnlySpan<byte> d, int o) => little ? d[o] | (d[o + 1] << 8) : (d[o] << 8) | d[o + 1];
        long U32(ReadOnlySpan<byte> d, int o) => little
            ? (uint)(d[o] | (d[o + 1] << 8) | (d[o + 2] << 16) | (d[o + 3] << 24))
            : (uint)((d[o] << 24) | (d[o + 1] << 16) | (d[o + 2] << 8) | d[o + 3]);
        if (U16(tiff, 2) != 42) return 1;
        long ifd = U32(tiff, 4);
        if (ifd < 8 || ifd + 2 > tiff.Length) return 1;
        int entries = U16(tiff, (int)ifd);
        for (int i = 0; i < entries && ifd + 2 + (i + 1) * 12 <= tiff.Length; i++)
        {
            int e = (int)ifd + 2 + i * 12;
            if (U16(tiff, e) == 0x0112) { int v = U16(tiff, e + 8); return v is >= 1 and <= 8 ? v : 1; }
        }
        return 1;
    }

    /// <summary>Where the pixel at (x, y) of a w x h picture goes in the upright one, and the upright size.</summary>
    static (int X, int Y) Target(int orientation, int x, int y, int w, int h) => orientation switch
    {
        2 => (w - 1 - x, y),
        3 => (w - 1 - x, h - 1 - y),
        4 => (x, h - 1 - y),
        5 => (y, x),
        6 => (h - 1 - y, x),
        7 => (h - 1 - y, w - 1 - x),
        8 => (y, w - 1 - x),
        _ => (x, y),
    };

    /// <summary>The picture turned upright (a new batch; the same one for orientation 1).</summary>
    public static ImageBatch Apply(ImageBatch img, int orientation)
    {
        if (orientation is < 2 or > 8) return img;
        int w = img.Width, h = img.Height, c = img.Channels;
        bool swap = orientation >= 5;
        int nw = swap ? h : w, nh = swap ? w : h;
        var o = new ImageBatch(img.Batch, nh, nw, c);
        for (int b = 0; b < img.Batch; b++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    var (nx, ny) = Target(orientation, x, y, w, h);
                    Array.Copy(img.Data, ((long)b * h * w + y * w + x) * c, o.Data, ((long)b * nh * nw + ny * nw + nx) * c, c);
                }
        return o;
    }

    /// <summary>One value per pixel (alpha) turned upright the same way.</summary>
    public static float[] Apply(float[] plane, int width, int height, int orientation)
    {
        if (orientation is < 2 or > 8) return plane;
        bool swap = orientation >= 5;
        int nw = swap ? height : width;
        var o = new float[plane.Length];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                var (nx, ny) = Target(orientation, x, y, width, height);
                o[ny * nw + nx] = plane[y * width + x];
            }
        return o;
    }
}
