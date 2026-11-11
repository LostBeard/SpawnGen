using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// Pad Image: grows the canvas on each side, by exact amounts, by a size multiplier or up to the next multiple, and fills the new area with a colour
/// (optionally blending the image edge into it) or with a mirrored copy of the picture. Behaviour measured against ComfyUI's node of the same id.
/// </summary>
[Node("PadImageNode", Category = "image/transform", DisplayName = "Pad Image",
    Description = "Pad an image with a solid color or a mirrored copy of its edges.", OutputNames = ["image"])]
public sealed class PadImageNode
{
    public static DynamicOption[] PaddingTypes() =>
    [
        new("pad dimensions",
            DynamicInput.Int("left", 0, 0, 16384, tooltip: "Pixels to add on the left."),
            DynamicInput.Int("top", 0, 0, 16384, tooltip: "Pixels to add on the top."),
            DynamicInput.Int("right", 0, 0, 16384, tooltip: "Pixels to add on the right."),
            DynamicInput.Int("bottom", 0, 0, 16384, tooltip: "Pixels to add on the bottom.")),
        new("pad by multiplier",
            DynamicInput.Float("multiplier", 1.0, 1.0, 8.0, 0.01,
                "Output size relative to the input (e.g., 1.5 adds 25% on each side). Padding is split evenly between opposite sides.")),
        new("pad to multiple",
            DynamicInput.Int("multiple", 32, 1, 16384,
                tooltip: "Pad so width and height are divisible by this number. Padding is split evenly between opposite sides, with the extra pixel going to the right/bottom.")),
    ];

    public static DynamicOption[] PaddingStyles() =>
    [
        new("color",
            new DynamicInput("color", new SocketType("COLOR"), new WidgetSpec(WidgetKind.Color) { Default = "#000000" },
                Tooltip: "Fill color for the padded area. Alpha is ignored."),
            DynamicInput.Int("feathering", 0, 0, 16384, tooltip: "Blend this many pixels of the image edge into the fill color on the padded sides.")),
        new("mirror image"),
    ];

    public ImageBatch Execute(ImageBatch image,
        [DynamicCombo(nameof(PaddingTypes)), Input(Tooltip = "Select how much padding to add: exact amounts per side, a size multiplier, or up to the next multiple.")]
        DynamicChoice paddingType,
        [DynamicCombo(nameof(PaddingStyles)), Input(Tooltip = "What to fill the padded area with.")] DynamicChoice paddingStyle)
    {
        var (left, top, right, bottom) = Amounts(paddingType, image.Width, image.Height);
        if (left == 0 && top == 0 && right == 0 && bottom == 0) return image; // measured: nothing to add gives the input back
        return paddingStyle.Key switch
        {
            "mirror image" => Mirror(image, left, top, right, bottom),
            "color" => Fill(image, left, top, right, bottom, ParseColor(paddingStyle.Get("color", "#000000")), (int)Math.Max(0, paddingStyle.Get("feathering", 0L))),
            _ => throw new ArgumentException($"Unknown padding style: {paddingStyle.Key}"),
        };
    }

    /// <summary>
    /// Pixels per side. Measured: a multiplier rounds the scaled size half to even (7 x 1.5 = 10.5 gives 10); a multiple pads by (-size) mod multiple; the
    /// added pixels are split with the odd one going to the right / bottom.
    /// </summary>
    static (int Left, int Top, int Right, int Bottom) Amounts(DynamicChoice type, int width, int height)
    {
        switch (type.Key)
        {
            case "pad dimensions":
                return (Side(type, "left"), Side(type, "top"), Side(type, "right"), Side(type, "bottom"));
            case "pad by multiplier":
            {
                double m = type.Get("multiplier", 1.0);
                int extraW = Math.Max(0, (int)Math.Round(width * m, MidpointRounding.ToEven) - width);
                int extraH = Math.Max(0, (int)Math.Round(height * m, MidpointRounding.ToEven) - height);
                return (extraW / 2, extraH / 2, extraW - extraW / 2, extraH - extraH / 2);
            }
            case "pad to multiple":
            {
                long k = Math.Max(1, type.Get("multiple", 32L));
                int padW = (int)(((-width) % k + k) % k), padH = (int)(((-height) % k + k) % k);
                return (padW / 2, padH / 2, padW - padW / 2, padH - padH / 2);
            }
            default:
                throw new ArgumentException($"Unknown padding type: {type.Key}");
        }
    }

    static int Side(DynamicChoice type, string name) => (int)Math.Clamp(type.Get(name, 0L), 0, 16384);

    /// <summary>
    /// The fill colour as measured: one leading '#' is dropped, at least six characters must remain, and each pair is read as a Python base-16 integer
    /// (so " 8" and "-1" are accepted, "-1" giving -1/255). Anything that does not parse is white. Extra characters (an alpha pair) are ignored.
    /// </summary>
    public static float[] ParseColor(string? color)
    {
        string s = color ?? "";
        if (s.StartsWith('#')) s = s[1..];
        var white = new[] { 1f, 1f, 1f };
        if (s.Length < 6) return white;
        var rgb = new float[3];
        for (int i = 0; i < 3; i++)
        {
            if (!TryParseHexPair(s.Substring(i * 2, 2), out int v)) return white;
            rgb[i] = (float)(v / 255.0);
        }
        return rgb;
    }

    /// <summary>Python's <c>int(text, 16)</c> for two characters: surrounding whitespace, an optional sign, then one or two hex digits.</summary>
    static bool TryParseHexPair(string text, out int value)
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

    /// <summary>Mirror padding as measured: reflection without repeating the edge pixel, continuing back and forth when the padding is wider than the picture.</summary>
    static ImageBatch Mirror(ImageBatch image, int left, int top, int right, int bottom)
    {
        int b = image.Batch, h = image.Height, w = image.Width, c = image.Channels;
        int oh = h + top + bottom, ow = w + left + right;
        var result = new ImageBatch(b, oh, ow, c);
        var xs = new int[ow];
        for (int x = 0; x < ow; x++) xs[x] = Reflect(x - left, w);
        for (int f = 0; f < b; f++)
            for (int y = 0; y < oh; y++)
            {
                int sy = Reflect(y - top, h);
                int src = (f * h + sy) * w, dst = (f * oh + y) * ow;
                for (int x = 0; x < ow; x++)
                    Array.Copy(image.Data, (src + xs[x]) * c, result.Data, (dst + x) * c, c);
            }
        return result;
    }

    static int Reflect(int i, int n)
    {
        if (n == 1) return 0;
        int period = 2 * (n - 1);
        int m = ((i % period) + period) % period;
        return m < n ? m : period - m;
    }

    /// <summary>
    /// Colour padding. The fill is the parsed RGB, then 1 for any further channel; a one- or two-channel picture takes the first channels of it. With
    /// feathering f, an image pixel at distance d (0 at the edge) from a padded side is blended toward the fill by ((f - d) / (f + 1))^2 in float32, the
    /// largest blend of the padded sides counting, through torch's lerp (see <see cref="TorchMath"/>; which of its two roundings applies was measured
    /// per element).
    /// </summary>
    static ImageBatch Fill(ImageBatch image, int left, int top, int right, int bottom, float[] rgb, int feathering)
    {
        int b = image.Batch, h = image.Height, w = image.Width, c = image.Channels;
        int oh = h + top + bottom, ow = w + left + right;
        var fill = new float[c];
        for (int k = 0; k < c; k++) fill[k] = k < 3 ? rgb[k] : 1f;
        var result = new ImageBatch(b, oh, ow, c);
        for (int p = 0; p < result.Data.Length; p += c) Array.Copy(fill, 0, result.Data, p, c);

        float[]? ax = feathering > 0 ? Blend(w, feathering, left > 0, right > 0) : null;
        float[]? ay = feathering > 0 ? Blend(h, feathering, top > 0, bottom > 0) : null;
        // Measured: torch's SIMD kernel (16 float32 lanes, fused) runs over the innermost run of elements (the channels, or for one channel a row) and its
        // scalar loop over the remainder.
        int run = c > 1 ? c : w, vectorized = run / 16 * 16;
        for (int f = 0; f < b; f++)
            for (int y = 0; y < h; y++)
            {
                int src = (f * h + y) * w * c, dst = ((f * oh + y + top) * ow + left) * c;
                Array.Copy(image.Data, src, result.Data, dst, w * c);
                if (ax is null) continue;
                for (int x = 0; x < w; x++)
                {
                    float a = Math.Max(ax[x], ay![y]);
                    if (a == 0f) continue;
                    for (int k = 0; k < c; k++)
                    {
                        int inner = c > 1 ? k : x;
                        float v = image.Data[src + x * c + k];
                        result.Data[dst + x * c + k] = inner < vectorized ? TorchMath.LerpFma(v, fill[k], a) : TorchMath.Lerp(v, fill[k], a);
                    }
                }
            }
        return result;
    }

    static float[] Blend(int n, int feathering, bool start, bool end)
    {
        var a = new float[n];
        float f = feathering;
        for (int i = 0; i < n; i++)
        {
            float v = 0f;
            if (start && i < feathering) v = Math.Max(v, Square((f - i) / (f + 1f)));
            int d = n - 1 - i;
            if (end && d < feathering) v = Math.Max(v, Square((f - d) / (f + 1f)));
            a[i] = v;
        }
        return a;
    }

    static float Square(float q) => q * q;
}

/// <summary>
/// <c>torch.lerp</c> on float32 as PyTorch documents and computes it (PyTorch, BSD): for |weight| &lt; 0.5 <c>start + weight * (end - start)</c>,
/// otherwise <c>end - (end - start) * (1 - weight)</c>, each step rounded; its SIMD kernel fuses the same formula into one multiply-add.
/// </summary>
public static class TorchMath
{
    /// <summary>The scalar path.</summary>
    public static float Lerp(float start, float end, float weight)
    {
        float diff = end - start;
        return Math.Abs(weight) < 0.5f ? start + weight * diff : end - diff * (1f - weight);
    }

    /// <summary>The vectorized path: <c>fma(coeff, end - start, base)</c> with coeff = weight or weight - 1 and base = start or end.</summary>
    public static float LerpFma(float start, float end, float weight)
    {
        float diff = end - start;
        return Math.Abs(weight) < 0.5f ? MathF.FusedMultiplyAdd(weight, diff, start) : MathF.FusedMultiplyAdd(weight - 1f, diff, end);
    }
}
