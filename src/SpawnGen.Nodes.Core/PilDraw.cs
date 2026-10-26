namespace SpawnGen.Nodes.Core;

/// <summary>
/// Pillow's <c>ImageDraw.rectangle</c> (libImaging Draw.c) on interleaved 8-bit pixels: coordinates are inclusive and truncated to ints, a filled rectangle
/// covers rows y0..y1, an outline of width w draws w rows at the top and bottom and w columns at each side, clipped to the picture.
/// </summary>
public static class PilDraw
{
    /// <summary>
    /// <c>ImageDraw.rectangle([x0, y0, x1, y1], fill=ink)</c> when <paramref name="fill"/>, else <c>outline=ink, width=width</c>. Pillow refuses x1 &lt; x0 or
    /// y1 &lt; y0 (ValueError).
    /// </summary>
    public static void Rectangle(byte[] px, int w, int h, int c, double x0, double y0, double x1, double y1, ReadOnlySpan<byte> ink, bool fill, int width = 1)
    {
        if (x1 < x0) throw new PyException("ValueError", "x1 must be greater than or equal to x0");
        if (y1 < y0) throw new PyException("ValueError", "y1 must be greater than or equal to y0");
        // _draw_rectangle hands (int)xy[i] to ImagingDrawRectangle: C truncation toward zero.
        Rectangle(px, w, h, c, (int)x0, (int)y0, (int)x1, (int)y1, ink, fill, width);
    }

    static void Rectangle(byte[] px, int w, int h, int c, int x0, int y0, int x1, int y1, ReadOnlySpan<byte> ink, bool fill, int width)
    {
        if (y0 > y1) (y0, y1) = (y1, y0);
        if (fill)
        {
            if (y0 < 0) y0 = 0;
            else if (y0 >= h) return;
            if (y1 < 0) return;
            else if (y1 > h) y1 = h;
            for (int y = y0; y <= y1; y++) HLine(px, w, h, c, x0, y, x1, ink);
            return;
        }
        if (width == 0) width = 1;
        for (int i = 0; i < width; i++)
        {
            HLine(px, w, h, c, x0, y0 + i, x1, ink);
            HLine(px, w, h, c, x0, y1 - i, x1, ink);
            VLine(px, w, h, c, x1 - i, y0 + width, y1 - width + 1, ink);
            VLine(px, w, h, c, x0 + i, y0 + width, y1 - width + 1, ink);
        }
    }

    static void HLine(byte[] px, int w, int h, int c, int x0, int y, int x1, ReadOnlySpan<byte> ink)
    {
        if (y < 0 || y >= h) return;
        if (x0 < 0) x0 = 0;
        else if (x0 >= w) return;
        if (x1 < 0) return;
        else if (x1 >= w) x1 = w - 1;
        for (int x = x0; x <= x1; x++) Set(px, w, c, x, y, ink);
    }

    /// <summary>Draw.c's <c>line</c> for a vertical line: |y1 - y0| points from y0 towards y1, the end point left out.</summary>
    static void VLine(byte[] px, int w, int h, int c, int x, int y0, int y1, ReadOnlySpan<byte> ink)
    {
        int dy = y1 - y0, ys = 1;
        if (dy < 0) { dy = -dy; ys = -1; }
        for (int i = 0; i < dy; i++, y0 += ys)
            if (x >= 0 && x < w && y0 >= 0 && y0 < h) Set(px, w, c, x, y0, ink);
    }

    static void Set(byte[] px, int w, int c, int x, int y, ReadOnlySpan<byte> ink)
    {
        int o = (y * w + x) * c;
        for (int k = 0; k < c; k++) px[o + k] = ink[k];
    }
}
