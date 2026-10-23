using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// ComfyUI's Pad Image (nodes_images.PadImageNode): grows the canvas by exact amounts per side, by a size multiplier or up to the next multiple, and fills
/// the new border with a colour (optionally feathering the picture's edge into it) or with a mirrored copy of the picture.
/// </summary>
[Node("PadImageNode", Category = "image/transform", DisplayName = "Pad Image",
    Description = "Pad an image with a solid color or a mirrored copy of its edges.", OutputNames = ["image"])]
public sealed class PadImageNode
{
    const int MaxResolution = 16384;

    public static DynamicOption[] PaddingTypes() =>
    [
        new("pad dimensions",
            DynamicInput.Int("left", 0, 0, MaxResolution, tooltip: "Pixels to add on the left."),
            DynamicInput.Int("top", 0, 0, MaxResolution, tooltip: "Pixels to add on the top."),
            DynamicInput.Int("right", 0, 0, MaxResolution, tooltip: "Pixels to add on the right."),
            DynamicInput.Int("bottom", 0, 0, MaxResolution, tooltip: "Pixels to add on the bottom.")),
        new("pad by multiplier", DynamicInput.Float("multiplier", 1.0, 1.0, 8.0,
            tooltip: "Output size relative to the input (e.g., 1.5 adds 25% on each side). Padding is split evenly between opposite sides.")),
        new("pad to multiple", DynamicInput.Int("multiple", 32, 1, MaxResolution,
            tooltip: "Pad so width and height are divisible by this number. Padding is split evenly between opposite sides, with the extra pixel going to the right/bottom.")),
    ];

    public static DynamicOption[] PaddingStyles() =>
    [
        new("color",
            new DynamicInput("color", new SocketType("COLOR"), new WidgetSpec(WidgetKind.Color) { Default = "#000000" }, Tooltip: "Fill color for the padded area. Alpha is ignored."),
            DynamicInput.Int("feathering", 0, 0, MaxResolution, tooltip: "Blend this many pixels of the image edge into the fill color on the padded sides.")),
        new("mirror image"),
    ];

    public ImageBatch Execute(ImageBatch image,
        [DynamicCombo(nameof(PaddingTypes)), Input(Tooltip = "Select how much padding to add: exact amounts per side, a size multiplier, or up to the next multiple.")] DynamicChoice paddingType,
        [DynamicCombo(nameof(PaddingStyles)), Input(Tooltip = "What to fill the padded area with.")] DynamicChoice paddingStyle)
    {
        int width = image.Width, height = image.Height;
        int left, top, right, bottom;
        switch (paddingType.Key)
        {
            case "pad dimensions":
                (left, top, right, bottom) = (paddingType.Get("left", 0), paddingType.Get("top", 0), paddingType.Get("right", 0), paddingType.Get("bottom", 0));
                break;
            case "pad by multiplier":
            {
                // Python's round (half to even) of the scaled size.
                double m = paddingType.Get("multiplier", 1.0);
                (left, right) = Split((int)Math.Round(width * m) - width);
                (top, bottom) = Split((int)Math.Round(height * m) - height);
                break;
            }
            case "pad to multiple":
            {
                int multiple = paddingType.Get("multiple", 32);
                (left, right) = Split(PyMod(-width, multiple));
                (top, bottom) = Split(PyMod(-height, multiple));
                break;
            }
            default: throw new ArgumentException($"Unsupported padding type: {paddingType.Key}");
        }
        if (left == 0 && top == 0 && right == 0 && bottom == 0) return image;
        return paddingStyle.Key switch
        {
            "mirror image" => Mirror(image, left, top, right, bottom),
            "color" => Fill(image, left, top, right, bottom, paddingStyle.Get("color", "#000000"), paddingStyle.Get("feathering", 0)),
            _ => throw new ArgumentException($"Unsupported padding style: {paddingStyle.Key}"),
        };
    }

    /// <summary>ComfyUI's split_padding: the smaller half before, the rest (the odd pixel) after.</summary>
    static (int Before, int After) Split(int total) => (FloorDiv(total, 2), total - FloorDiv(total, 2));

    static int FloorDiv(int a, int b) => (int)Math.Floor((double)a / b);

    static int PyMod(int a, int m) => ((a % m) + m) % m;

    /// <summary>Source index of every output row or column: reflected without repeating the edge pixel, the reflection repeating when the padding is wider than the picture.</summary>
    static int[] MirrorIndices(int size, int before, int after)
    {
        int period = Math.Max(2 * (size - 1), 1);
        var idx = new int[before + size + after];
        for (int i = 0; i < idx.Length; i++)
        {
            int k = Math.Abs(i - before) % period;
            idx[i] = k < size ? k : period - k;
        }
        return idx;
    }

    static ImageBatch Mirror(ImageBatch image, int left, int top, int right, int bottom)
    {
        int c = image.Channels, h = image.Height, w = image.Width;
        var rows = MirrorIndices(h, top, bottom);
        var cols = MirrorIndices(w, left, right);
        var result = new ImageBatch(image.Batch, rows.Length, cols.Length, c);
        for (int b = 0; b < image.Batch; b++)
            for (int y = 0; y < rows.Length; y++)
                for (int x = 0; x < cols.Length; x++)
                    Array.Copy(image.Data, ((b * h + rows[y]) * w + cols[x]) * c, result.Data, ((b * rows.Length + y) * cols.Length + x) * c, c);
        return result;
    }

    /// <summary>ComfyUI's colour_util.hex_to_rgb: "#rrggbb" (leading '#'s stripped) as 0..255 per channel, white for anything else.</summary>
    static (long R, long G, long B) HexToRgb(string value)
    {
        var h = value.TrimStart('#');
        if (h.Length == 6 && PyInt.TryParse(h[..2], 16, out var r) && PyInt.TryParse(h[2..4], 16, out var g) && PyInt.TryParse(h[4..6], 16, out var b)) return (r, g, b);
        return (255, 255, 255);
    }

    /// <summary>ComfyUI's feather_weight: ((feathering - distance to a padded side) / (feathering + 1))², the distance capped at the feathering.</summary>
    static float[] FeatherWeight(int size, int before, int after, int feathering)
    {
        var weight = new float[size];
        for (int i = 0; i < size; i++)
        {
            long dist = feathering;
            if (before > 0) dist = Math.Min(dist, i);
            if (after > 0) dist = Math.Min(dist, size - 1 - i);
            float t = (float)(feathering - dist) / (float)(feathering + 1L);
            weight[i] = t * t;
        }
        return weight;
    }

    static ImageBatch Fill(ImageBatch image, int left, int top, int right, int bottom, string color, int feathering)
    {
        int c = image.Channels, h = image.Height, w = image.Width;
        // The colour's first 7 characters ("#rrggbb", alpha ignored); channels past RGB are filled with 1, a grey picture takes red only.
        var (r, g, b) = HexToRgb(color.Length > 7 ? color[..7] : color);
        double[] rgba = [r / 255.0, g / 255.0, b / 255.0];
        var fill = new float[c];
        for (int k = 0; k < c; k++) fill[k] = k < 3 ? (float)rgba[k] : 1f;

        int oh = h + top + bottom, ow = w + left + right;
        var result = new ImageBatch(image.Batch, oh, ow, c);
        for (int p = 0; p < result.Data.Length; p += c) Array.Copy(fill, 0, result.Data, p, c);
        float[]? wy = feathering > 0 ? FeatherWeight(h, top, bottom, feathering) : null;
        float[]? wx = feathering > 0 ? FeatherWeight(w, left, right, feathering) : null;
        for (int bi = 0; bi < image.Batch; bi++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int src = ((bi * h + y) * w + x) * c, dst = ((bi * oh + y + top) * ow + x + left) * c;
                    if (wy is null) { Array.Copy(image.Data, src, result.Data, dst, c); continue; }
                    float weight = Math.Max(wy[y], wx![x]);
                    for (int k = 0; k < c; k++) result.Data[dst + k] = TorchMath.Lerp(image.Data[src + k], fill[k], weight);
                }
        return result;
    }
}

/// <summary>torch's float32 element functions where their exact form matters.</summary>
internal static class TorchMath
{
    /// <summary>torch.lerp: start + weight (end - start) for |weight| &lt; 0.5, else end - (end - start)(1 - weight), in float.</summary>
    public static float Lerp(float start, float end, float weight) =>
        Math.Abs(weight) < 0.5f ? start + weight * (end - start) : end - (end - start) * (1f - weight);

    /// <summary>torch.lerp as its vectorised CPU kernel computes it (contiguous operands): fused multiply-adds.</summary>
    public static float LerpFma(float start, float end, float weight) =>
        Math.Abs(weight) < 0.5f ? MathF.FusedMultiplyAdd(weight, end - start, start) : MathF.FusedMultiplyAdd(weight - 1f, end - start, end);
}
