using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entry, SpawnGen's tests and black-box probes of ComfyUI (2026-11-08 and
// 2026-11-11); the TRELLIS.2 preprocessing (MIT) was read for orientation only. Pillow's lanczos and PyTorch's bilinear come from PilLanczos and
// TorchInterpolate.

/// <summary>
/// ComfyUI's Crop Image to Mask (the TRELLIS.2-style subject crop), written from black-box measurements of ComfyUI (tools/parity/make_crop_to_mask_fixtures.py,
/// fixture comfy_image_nodes_crop_to_mask.json.gz). Per picture of the batch (one mask may serve every picture; other batch sizes are an error):
/// <list type="number">
/// <item>A mask of another size is resized to the picture's with PyTorch bilinear (align_corners false, no antialiasing).</item>
/// <item>The mask is inverted (1 - m) when the mean of its four edges (corners counted twice) is above 0.5 and the mean of its centre half
/// [H/4, H - H/4) x [W/4, W - W/4) is below 0.5: a LoadImage-style mask (1 = background) becomes a subject mask.</item>
/// <item>Values below 0.05 become 0; grow_mask then takes the maximum (or, negative, the minimum) over a (2|g| + 1) square, clipped at the edges.</item>
/// <item>When max(width, height) is smaller than the picture's longer side, the picture and the mask are first shrunk by k = max(width, height) / that side
/// to int(W k) x int(H k) (Pillow lanczos on 8-bit pixels; the double product can truncate one short, e.g. 98 -> 63).</item>
/// <item>The subject box is the bounding box of the 8-bit mask values above 204. With no such value the shrunk mask is inverted; if that is empty too,
/// the whole composite is stretched to width x height.</item>
/// <item>The box grows to the output's aspect ratio and by pad_factor to sw x sh; each side becomes the even size 2 floor(s/2 + 1/2) placed at
/// trunc(centre - half), where the centre is the middle of the inclusive box.</item>
/// <item>The picture is composited over the background in float32 (rgb a + bg (1 - a)), made 8-bit (truncated), cropped to the window (outside the
/// picture is the background) and resized to width x height with Pillow lanczos.</item>
/// </list>
/// The output always has 3 channels: the first three of the picture, or its single channel repeated; 2-channel pictures are an error. The background
/// loses any leading '#' and must then be exactly six characters (otherwise black), each pair read as a Python base-16 integer (an unreadable pair is an
/// error).
/// </summary>
[Node(Category = "image/transform", DisplayName = "Crop Image to Mask", OutputNames = ["images"])]
public sealed class ImageCropToMask
{
    const byte Threshold = 204;

    public ImageBatch Execute(ImageBatch images, MaskBatch masks,
        [Input(Min = 64, Max = 4096, Step = 8, Tooltip = "Output width in pixels.")] int width = 1024,
        [Input(Min = 64, Max = 4096, Step = 8, Tooltip = "Output height in pixels.")] int height = 1024,
        [Input(Min = 1.0, Max = 2.0, Step = 0.01, Tooltip = "Extra margin around the mask bounding box as a multiplier.")] double padFactor = 1.0,
        [Input(Min = -32, Max = 32, Step = 1, Tooltip = "Grow or shrink the mask by this many pixels before cropping.")] int growMask = 0,
        [Input(Color = true, Tooltip = "Background color behind the masked subject.")] string background = "#000000")
    {
        if (masks.Batch != images.Batch && masks.Batch != 1)
            throw new ArgumentException($"Mask batch {masks.Batch} does not match image batch {images.Batch}");
        if (images.Channels == 2)
            throw new ArgumentException("Crop Image to Mask needs a picture of 1, 3 or 4 channels, not 2.", nameof(images));
        if (width <= 0 || height <= 0) throw new ArgumentException("height and width must be > 0");
        var bg = ParseBackground(background);
        var result = new ImageBatch(images.Batch, height, width, 3);
        int h = images.Height, w = images.Width;
        for (int b = 0; b < images.Batch; b++)
        {
            var rgb = Rgb(images, b);
            var mask = Mask(masks, Math.Min(b, masks.Batch - 1), h, w);
            var output = One(rgb, mask, h, w, width, height, padFactor, growMask, bg);
            Array.Copy(output, 0, result.Data, (long)b * width * height * 3, output.Length);
        }
        return result;
    }

    static float[] One(float[] rgb, float[] mask, int h, int w, int ow, int oh, double pad, int grow, int[] bg)
    {
        if (ShouldInvert(mask, h, w))
            for (int i = 0; i < mask.Length; i++) mask[i] = 1f - mask[i];
        for (int i = 0; i < mask.Length; i++) if (mask[i] < 0.05f) mask[i] = 0f;
        if (grow != 0) mask = Grow(mask, h, w, grow);

        double k = Math.Min(1.0, (double)Math.Max(ow, oh) / Math.Max(w, h));
        int nw = (int)(w * k), nh = (int)(h * k);
        float[] alpha = mask;
        if (nw != w || nh != h)
        {
            rgb = PilLanczos.Resize(rgb, w, h, 3, nw, nh);
            alpha = PilLanczos.Resize(mask, w, h, 1, nw, nh);
        }

        var box = Box(PilLanczos.ToBytes(alpha), nw, nh);
        if (box is null)
        {
            var inverted = new float[alpha.Length];
            for (int i = 0; i < alpha.Length; i++) inverted[i] = 1f - alpha[i];
            var invertedBox = Box(PilLanczos.ToBytes(inverted), nw, nh);
            if (invertedBox is not null) { alpha = inverted; box = invertedBox; }
        }

        var bgf = new float[3];
        for (int c = 0; c < 3; c++) bgf[c] = bg[c] / 255f;
        var composite = new float[nw * nh * 3];
        for (int i = 0; i < alpha.Length; i++)
        {
            float a = alpha[i], rest = 1f - a;
            for (int c = 0; c < 3; c++) composite[i * 3 + c] = rgb[i * 3 + c] * a + bgf[c] * rest;
        }
        var compositeBytes = PilLanczos.ToBytes(composite);
        if (box is not { } bb) return PilLanczos.FromBytes(PilLanczos.Resize(compositeBytes, nw, nh, 3, ow, oh));

        var (x0, y0, sw, sh) = Window(bb, ow, oh, pad);
        var canvas = new byte[(long)sw * sh * 3];
        for (int c = 0; c < 3; c++)
        {
            byte fill = (byte)Math.Clamp(bg[c], 0, 255);
            for (long i = c; i < canvas.Length; i += 3) canvas[i] = fill;
        }
        int sx0 = Math.Max(x0, 0), sy0 = Math.Max(y0, 0), sx1 = Math.Min(x0 + sw, nw), sy1 = Math.Min(y0 + sh, nh);
        if (sx1 > sx0)
            for (int y = sy0; y < sy1; y++)
                Array.Copy(compositeBytes, ((long)y * nw + sx0) * 3, canvas, (((long)(y - y0)) * sw + (sx0 - x0)) * 3, (long)(sx1 - sx0) * 3);
        return PilLanczos.FromBytes(PilLanczos.Resize(canvas, sw, sh, 3, ow, oh));
    }

    /// <summary>Edges (top, bottom, left, right rows; corners twice) mean above 0.5 and centre-half mean below 0.5.</summary>
    static bool ShouldInvert(float[] m, int h, int w)
    {
        double edges = 0;
        for (int x = 0; x < w; x++) edges += m[x] + m[(h - 1) * w + x];
        for (int y = 0; y < h; y++) edges += m[y * w] + m[y * w + w - 1];
        edges /= 2.0 * (w + h);
        int y0 = h / 4, y1 = h - h / 4, x0 = w / 4, x1 = w - w / 4;
        double centre = 0;
        for (int y = y0; y < y1; y++)
            for (int x = x0; x < x1; x++) centre += m[y * w + x];
        centre /= (double)(y1 - y0) * (x1 - x0);
        return edges > 0.5 && centre < 0.5;
    }

    /// <summary>Maximum (grow &gt; 0) or minimum (grow &lt; 0) over a (2|grow| + 1) square window clipped at the picture's edges.</summary>
    static float[] Grow(float[] m, int h, int w, int grow)
    {
        int r = Math.Abs(grow);
        bool max = grow > 0;
        var rows = new float[m.Length];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                float v = m[y * w + x];
                for (int xx = Math.Max(0, x - r); xx <= Math.Min(w - 1, x + r); xx++)
                    v = max ? Math.Max(v, m[y * w + xx]) : Math.Min(v, m[y * w + xx]);
                rows[y * w + x] = v;
            }
        var result = new float[m.Length];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                float v = rows[y * w + x];
                for (int yy = Math.Max(0, y - r); yy <= Math.Min(h - 1, y + r); yy++)
                    v = max ? Math.Max(v, rows[yy * w + x]) : Math.Min(v, rows[yy * w + x]);
                result[y * w + x] = v;
            }
        return result;
    }

    /// <summary>The inclusive bounding box (x0, x1, y0, y1) of the bytes above 204, or null.</summary>
    static (int X0, int X1, int Y0, int Y1)? Box(byte[] a, int w, int h)
    {
        int x0 = int.MaxValue, x1 = -1, y0 = int.MaxValue, y1 = -1;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                if (a[y * w + x] > Threshold)
                {
                    if (x < x0) x0 = x;
                    if (x > x1) x1 = x;
                    if (y < y0) y0 = y;
                    if (y > y1) y1 = y;
                }
        return x1 < 0 ? null : (x0, x1, y0, y1);
    }

    /// <summary>The crop window (left, top, width, height) for a box grown to the output aspect ratio and by <paramref name="pad"/>.</summary>
    static (int X, int Y, int W, int H) Window((int X0, int X1, int Y0, int Y1) box, int ow, int oh, double pad)
    {
        int bw = box.X1 - box.X0 + 1, bh = box.Y1 - box.Y0 + 1;
        double aspect = (double)ow / oh, sw, sh;
        if ((double)bw / bh > aspect) { sw = bw; sh = bw / aspect; }
        else { sw = bh * aspect; sh = bh; }
        sw *= pad;
        sh *= pad;
        int hw = (int)Math.Floor(sw / 2 + 0.5), hh = (int)Math.Floor(sh / 2 + 0.5);
        double cx = (box.X0 + box.X1) / 2.0, cy = (box.Y0 + box.Y1) / 2.0;
        return ((int)Math.Truncate(cx - hw), (int)Math.Truncate(cy - hh), 2 * hw, 2 * hh);
    }

    /// <summary>Picture b as interleaved RGB floats: the first three channels, or a single channel repeated.</summary>
    static float[] Rgb(ImageBatch images, int b)
    {
        int n = images.Height * images.Width, c = images.Channels;
        var rgb = new float[n * 3];
        long offset = (long)b * n * c;
        for (int i = 0; i < n; i++)
            for (int k = 0; k < 3; k++)
                rgb[i * 3 + k] = images.Data[offset + (long)i * c + (c == 1 ? 0 : k)];
        return rgb;
    }

    /// <summary>Mask b as a copy, resized to h x w with PyTorch bilinear when its size differs.</summary>
    static float[] Mask(MaskBatch masks, int b, int h, int w)
    {
        int n = masks.Height * masks.Width;
        var m = new float[n];
        Array.Copy(masks.Data, (long)b * n, m, 0, n);
        if (masks.Height == h && masks.Width == w) return m;
        return TorchInterpolate.Resize(m, 1, masks.Height, masks.Width, h, w, "bilinear");
    }

    /// <summary>
    /// The background as measured: leading '#' characters dropped, then exactly six characters (anything else is black), each pair a Python base-16
    /// integer (whitespace and a sign allowed, so values can fall outside 0..255); a pair that does not parse is an error.
    /// </summary>
    public static int[] ParseBackground(string? background)
    {
        string s = (background ?? "").TrimStart('#');
        if (s.Length != 6) return [0, 0, 0];
        var rgb = new int[3];
        for (int i = 0; i < 3; i++)
        {
            string pair = s.Substring(i * 2, 2);
            if (!TryParsePythonHex(pair, out rgb[i])) throw new ArgumentException($"invalid literal for int() with base 16: '{pair}'", nameof(background));
        }
        return rgb;
    }

    /// <summary>Python's <c>int(text, 16)</c> for a short string: surrounding whitespace, an optional sign, then hex digits.</summary>
    static bool TryParsePythonHex(string text, out int value)
    {
        value = 0;
        string t = text.Trim(' ', '\t', '\n', '\r', '\f', '\v');
        int sign = 1;
        if (t.Length > 0 && (t[0] == '+' || t[0] == '-')) { sign = t[0] == '-' ? -1 : 1; t = t[1..]; }
        if (t.Length == 0) return false;
        foreach (char ch in t)
        {
            int d = ch is >= '0' and <= '9' ? ch - '0' : ch is >= 'a' and <= 'f' ? ch - 'a' + 10 : ch is >= 'A' and <= 'F' ? ch - 'A' + 10 : -1;
            if (d < 0) return false;
            value = value * 16 + d;
        }
        value *= sign;
        return true;
    }
}
