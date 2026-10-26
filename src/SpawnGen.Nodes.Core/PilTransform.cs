namespace SpawnGen.Nodes.Core;

/// <summary>
/// Pillow's geometry on interleaved 8-bit pictures: <c>transpose</c> (flips and quarter turns) and <c>rotate(angle, expand=True, resample=BICUBIC,
/// fillcolor=0)</c> for RGBA (resampled premultiplied, as Pillow does for RGBA), ported from Image.py and libImaging Geometry.c.
/// </summary>
public static class PilTransform
{
    public static byte[] FlipLeftRight(byte[] px, int w, int h, int c)
    {
        var o = new byte[px.Length];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                Array.Copy(px, (y * w + x) * c, o, (y * w + (w - 1 - x)) * c, c);
        return o;
    }

    public static byte[] FlipTopBottom(byte[] px, int w, int h, int c)
    {
        var o = new byte[px.Length];
        for (int y = 0; y < h; y++) Array.Copy(px, y * w * c, o, (h - 1 - y) * w * c, w * c);
        return o;
    }

    /// <summary>Transpose.ROTATE_90 / ROTATE_180 / ROTATE_270: a quarter turn counter-clockwise, a half turn, a quarter turn clockwise.</summary>
    public static (byte[] Px, int W, int H) Turn(byte[] px, int w, int h, int c, int degrees)
    {
        int ow = degrees == 180 ? w : h, oh = degrees == 180 ? h : w;
        var o = new byte[px.Length];
        for (int yo = 0; yo < oh; yo++)
            for (int xo = 0; xo < ow; xo++)
            {
                var (xi, yi) = degrees switch
                {
                    90 => (w - 1 - yo, xo),
                    180 => (w - 1 - xo, h - 1 - yo),
                    270 => (yo, h - 1 - xo),
                    _ => throw new ArgumentOutOfRangeException(nameof(degrees)),
                };
                Array.Copy(px, (yi * w + xi) * c, o, (yo * ow + xo) * c, c);
            }
        return (o, ow, oh);
    }

    // CPython's math.radians / math.degrees factors.
    internal const double DegToRad = Math.PI / 180.0, RadToDeg = 180.0 / Math.PI;

    /// <summary>Python's <c>x % m</c> for floats: the result has the divisor's sign.</summary>
    static double PyMod(double x, double m)
    {
        double r = x % m;
        if (r != 0) { if (r < 0 != m < 0) r += m; }
        else r = 0.0 * m; // copysign(0, m)
        return r;
    }

    /// <summary>
    /// <c>Image.rotate(angle, expand=True, resample=BICUBIC, fillcolor=(0, 0, 0, 0))</c> of an RGBA picture: counter-clockwise degrees, the canvas grown to hold
    /// the whole turned picture, uncovered pixels transparent black. Quarter and half turns are exact transposes, as in Pillow.
    /// </summary>
    public static (byte[] Px, int W, int H) RotateExpand(byte[] px, int w, int h, double angle)
    {
        angle = PyMod(angle, 360.0);
        if (angle == 0) return ((byte[])px.Clone(), w, h);
        if (angle == 180) return Turn(px, w, h, 4, 180);
        if (angle is 90 or 270) return Turn(px, w, h, 4, (int)angle);

        double cx = w / 2.0, cy = h / 2.0;
        double a = -(angle * DegToRad);
        var m = new[] { PyNumber.Round(Math.Cos(a), 15), PyNumber.Round(Math.Sin(a), 15), 0.0, PyNumber.Round(-Math.Sin(a), 15), PyNumber.Round(Math.Cos(a), 15), 0.0 };
        (double X, double Y) T(double x, double y) => (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]);
        (m[2], m[5]) = T(-cx - 0, -cy - 0);
        m[2] += cx;
        m[5] += cy;
        var corners = new[] { T(0, 0), T(w, 0), T(w, h), T(0, h) };
        int nw = (int)(Math.Ceiling(corners.Max(p => p.X)) - Math.Floor(corners.Min(p => p.X)));
        int nh = (int)(Math.Ceiling(corners.Max(p => p.Y)) - Math.Floor(corners.Min(p => p.Y)));
        (m[2], m[5]) = T(-(nw - w) / 2.0, -(nh - h) / 2.0);

        // Image.transform: RGBA goes through RGBa; the output starts as the fill colour; ImagingGenericTransform with the bicubic filter.
        var src = (byte[])px.Clone();
        PilLanczos.Premultiply(src, 4);
        var dst = new byte[nw * nh * 4];
        for (int y = 0; y < nh; y++)
            for (int x = 0; x < nw; x++)
            {
                double xin = m[0] * (x + 0.5) + m[1] * (y + 0.5) + m[2];
                double yin = m[3] * (x + 0.5) + m[4] * (y + 0.5) + m[5];
                Bicubic(src, w, h, xin, yin, dst, (y * nw + x) * 4);
            }
        PilLanczos.Unpremultiply(dst, 4);
        return (dst, nw, nh);
    }

    static int Floor(double v) => v < 0.0 ? (int)Math.Floor(v) : (int)v;

    static double Cubic(double v1, double v2, double v3, double v4, double d)
    {
        double p1 = v2, p2 = -v1 + v3, p3 = 2 * (v1 - v2) + v3 - v4, p4 = -v1 + v2 - v3 + v4;
        return p1 + d * (p2 + d * (p3 + d * p4));
    }

    /// <summary>Geometry.c's bicubic_filter32RGB on a 4-band picture: false (pixel left as it is) when the point is outside the picture.</summary>
    static void Bicubic(byte[] im, int xsize, int ysize, double xin, double yin, byte[] dst, int o)
    {
        if (xin < 0.0 || xin >= xsize || yin < 0.0 || yin >= ysize) return;
        xin -= 0.5;
        yin -= 0.5;
        int x = Floor(xin), y = Floor(yin);
        double dx = xin - x, dy = yin - y;
        x--;
        y--;
        int XClip(int v) => v < 0 ? 0 : v < xsize ? v : xsize - 1;
        int x0 = XClip(x), x1 = XClip(x + 1), x2 = XClip(x + 2), x3 = XClip(x + 3);
        int yc = y < 0 ? 0 : y < ysize ? y : ysize - 1;
        for (int b = 0; b < 4; b++)
        {
            double Row(int row) => Cubic(im[(row * xsize + x0) * 4 + b], im[(row * xsize + x1) * 4 + b], im[(row * xsize + x2) * 4 + b], im[(row * xsize + x3) * 4 + b], dx);
            double v1 = Row(yc);
            double v2 = y + 1 >= 0 && y + 1 < ysize ? Row(y + 1) : v1;
            double v3 = y + 2 >= 0 && y + 2 < ysize ? Row(y + 2) : v2;
            double v4 = y + 3 >= 0 && y + 3 < ysize ? Row(y + 3) : v3;
            double v = Cubic(v1, v2, v3, v4, dy);
            dst[o + b] = v <= 0.0 ? (byte)0 : v >= 255.0 ? (byte)255 : (byte)v;
        }
    }
}
