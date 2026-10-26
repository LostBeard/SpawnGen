namespace SpawnGen.Nodes.Core;

/// <summary>
/// ComfyUI's layer blending (comfy_extras/compositor_blend.py) per pixel, in float32 as its numpy code computes it: sRGB transfer functions, the 26 blend
/// modes (each with its blend space: linear or perceptual, and its composite: union for normal, clip-to-backdrop for the rest), and a layer's placed bounds.
/// </summary>
public static class CompositorBlend
{
    const float Eps = 1e-6f;
    const float LumR = 0.2224884f, LumG = 0.71690369f, LumB = 0.06060791f;

    /// <summary>The blend modes in ComfyUI's order with their blend space and composite.</summary>
    public static readonly (string Name, bool Linear, bool Union)[] Modes =
    [
        ("normal", true, true), ("multiply", true, false), ("screen", false, false), ("overlay", false, false), ("darken", true, false),
        ("lighten", true, false), ("color-dodge", false, false), ("color-burn", false, false), ("hard-light", false, false), ("soft-light", false, false),
        ("difference", false, false), ("exclusion", false, false), ("linear-dodge", true, false), ("linear-burn", false, false), ("vivid-light", false, false),
        ("pin-light", false, false), ("linear-light", false, false), ("hard-mix", false, false), ("subtract", true, false), ("divide", true, false),
        ("grain-extract", false, false), ("grain-merge", false, false), ("hue", false, false), ("saturation", false, false), ("color", false, false),
        ("luminosity", true, false),
    ];

    public static readonly string[] ModeNames = Modes.Select(m => m.Name).ToArray();

    public static float SrgbToLinear(float c) => c <= 0.04045f ? c / 12.92f : MathF.Pow((MathF.Max(c, 0f) + 0.055f) / 1.055f, 2.4f);

    public static float LinearToSrgb(float c) => c <= 0.0031308f ? 12.92f * c : 1.055f * MathF.Pow(MathF.Max(c, 0f), (float)(1.0 / 2.4)) - 0.055f;

    static float SafeDiv(float a, float b) => MathF.Abs(b) >= Eps ? a / b : 0f;

    static float Channel(string blend, float i, float l) => blend switch
    {
        "multiply" => i * l,
        "screen" => 1 - (1 - i) * (1 - l),
        "overlay" => i < 0.5f ? 2 * i * l : 1 - 2 * (1 - l) * (1 - i),
        "darken" => MathF.Min(i, l),
        "lighten" => MathF.Max(i, l),
        "color-dodge" => i <= 0 ? 0f : 1 - l <= Eps ? 1f : MathF.Min(SafeDiv(i, 1 - l), 1f),
        "color-burn" => i >= 1 ? 1f : l <= Eps ? 0f : 1 - MathF.Min(SafeDiv(1 - i, l), 1f),
        "hard-light" => l > 0.5f ? MathF.Min(1 - (1 - i) * (1 - (l - 0.5f) * 2), 1f) : MathF.Min(i * (l * 2), 1f),
        "soft-light" => (1 - i) * (i * l) + i * (1 - (1 - i) * (1 - l)),
        "difference" => MathF.Abs(i - l),
        "exclusion" => 0.5f - 2 * (i - 0.5f) * (l - 0.5f),
        "linear-dodge" => i + l,
        "linear-burn" => i + l - 1,
        "vivid-light" => l <= 0.5f
            ? i >= 1 ? 1f : 2 * l <= Eps ? 0f : MathF.Max(1 - SafeDiv(1 - i, 2 * l), 0f)
            : i <= 0 ? 0f : 2 * (1 - l) <= Eps ? 1f : MathF.Min(SafeDiv(i, 2 * (1 - l)), 1f),
        "pin-light" => l > 0.5f ? MathF.Max(i, 2 * (l - 0.5f)) : MathF.Min(i, 2 * l),
        "linear-light" => i + 2 * l - 1,
        "hard-mix" => i + l < 1 ? 0f : 1f,
        "subtract" => MathF.Max(i - l, 0f),
        "divide" => Math.Clamp(i / MathF.Max(l, Eps), 0f, 1f),
        "grain-extract" => i - l + 0.5f,
        "grain-merge" => i + l - 0.5f,
        _ => l, // normal, and any name the table does not know
    };

    static float Max3(float a, float b, float c) => MathF.Max(MathF.Max(a, b), c);
    static float Min3(float a, float b, float c) => MathF.Min(MathF.Min(a, b), c);

    /// <summary>blend_pixel for one pixel: the HSL modes on the whole colour, the others channel by channel.</summary>
    public static void BlendPixel(string blend, ReadOnlySpan<float> i, ReadOnlySpan<float> l, Span<float> o)
    {
        switch (blend)
        {
            case "hue":
            {
                float srcMin = Min3(l[0], l[1], l[2]), srcMax = Max3(l[0], l[1], l[2]), srcDelta = srcMax - srcMin;
                bool achromatic = srcDelta <= Eps;
                float destMax = Max3(i[0], i[1], i[2]), destDelta = destMax - Min3(i[0], i[1], i[2]);
                float destS = destMax != 0 ? destDelta / destMax : 0f;
                float ratio = achromatic ? 0f : destS * destMax / srcDelta;
                float offset = destMax - srcMax * ratio;
                for (int k = 0; k < 3; k++) o[k] = achromatic ? i[k] : l[k] * ratio + offset;
                break;
            }
            case "saturation":
            {
                float destMax = Max3(i[0], i[1], i[2]), destDelta = destMax - Min3(i[0], i[1], i[2]);
                bool flat = destDelta <= Eps;
                float srcMax = Max3(l[0], l[1], l[2]), srcDelta = srcMax - Min3(l[0], l[1], l[2]);
                float srcS = srcMax != 0 ? srcDelta / srcMax : 0f;
                float ratio = flat ? 0f : srcS * destMax / destDelta;
                float offset = (1 - ratio) * destMax;
                for (int k = 0; k < 3; k++) o[k] = flat ? destMax : i[k] * ratio + offset;
                break;
            }
            case "color":
            {
                float destL = (Min3(i[0], i[1], i[2]) + Max3(i[0], i[1], i[2])) / 2;
                float srcL = (Min3(l[0], l[1], l[2]) + Max3(l[0], l[1], l[2])) / 2;
                bool gray = MathF.Abs(srcL) <= Eps || MathF.Abs(1 - srcL) <= Eps;
                float dl = MathF.Min(destL, 1 - destL), sl = MathF.Min(srcL, 1 - srcL);
                float ratio = dl / (gray ? 1f : sl);
                float offset = (destL > 0.5f ? 1 - 2 * dl : 0f) + (srcL > 0.5f ? 2 * dl - ratio : 0f);
                for (int k = 0; k < 3; k++) o[k] = gray ? destL : l[k] * ratio + offset;
                break;
            }
            case "luminosity":
            {
                float lumI = i[0] * LumR + i[1] * LumG + i[2] * LumB, lumL = l[0] * LumR + l[1] * LumG + l[2] * LumB;
                bool degenerate = lumI <= Eps;
                float ratio = degenerate ? 0f : lumL / lumI;
                for (int k = 0; k < 3; k++) o[k] = degenerate ? lumL : i[k] * ratio;
                break;
            }
            default:
                for (int k = 0; k < 3; k++) o[k] = Channel(blend, i[k], l[k]);
                break;
        }
    }

    /// <summary>resolve_mode: the blend space and composite of a mode name (an unknown name blends as normal).</summary>
    public static (bool Linear, bool Union) Resolve(string blend)
    {
        foreach (var m in Modes) if (m.Name == blend) return (m.Linear, m.Union);
        return (true, true);
    }

    /// <summary>
    /// blend_composite over a region: <paramref name="backdrop"/> (straight RGBA, linear light) is replaced by the layer blended onto it with the given
    /// opacity. Both are interleaved RGBA float32 of the same length.
    /// </summary>
    public static void Composite(string blend, float[] backdrop, int backdropOffset, int backdropStride, float[] layer, int layerOffset, int layerStride,
        int width, int height, double opacity)
    {
        var (linear, union) = Resolve(blend);
        float cov = (float)opacity;
        Span<float> inB = stackalloc float[3], layB = stackalloc float[3], comp = stackalloc float[3];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int bi = backdropOffset + y * backdropStride + x * 4, li = layerOffset + y * layerStride + x * 4;
                for (int k = 0; k < 3; k++)
                {
                    inB[k] = linear ? backdrop[bi + k] : LinearToSrgb(backdrop[bi + k]);
                    layB[k] = linear ? layer[li + k] : LinearToSrgb(layer[li + k]);
                }
                BlendPixel(blend, inB, layB, comp);
                if (!linear) for (int k = 0; k < 3; k++) comp[k] = SrgbToLinear(comp[k]);

                float inA = backdrop[bi + 3], layerA = layer[li + 3] * cov;
                if (union)
                {
                    float newA = layerA + (1 - layerA) * inA;
                    float ratio = newA != 0 ? layerA / newA : 0f;
                    bool keep = layerA == 0 || newA == 0;
                    for (int k = 0; k < 3; k++)
                    {
                        float blended = ratio * (inA * (comp[k] - layer[li + k]) + layer[li + k] - backdrop[bi + k]) + backdrop[bi + k];
                        backdrop[bi + k] = keep ? backdrop[bi + k] : inA == 0 ? layer[li + k] : blended;
                    }
                    backdrop[bi + 3] = newA;
                }
                else
                {
                    bool keep = inA == 0 || layerA == 0;
                    for (int k = 0; k < 3; k++)
                        if (!keep) backdrop[bi + k] = comp[k] * layerA + backdrop[bi + k] * (1 - layerA);
                    // clip-to-backdrop keeps the backdrop's alpha
                }
            }
    }

    /// <summary>placed_bounds: the whole-pixel box a w x h layer at (x, y) covers once turned by <paramref name="rotation"/> radians about its centre.</summary>
    public static (int X, int Y, int W, int H) PlacedBounds(double x, double y, double w, double h, double rotation)
    {
        double cx = x + w / 2, cy = y + h / 2, cos = Math.Cos(rotation), sin = Math.Sin(rotation), hw = w / 2, hh = h / 2;
        var corners = new[] { (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh) };
        var xs = corners.Select(p => cx + p.Item1 * cos - p.Item2 * sin).ToArray();
        var ys = corners.Select(p => cy + p.Item1 * sin + p.Item2 * cos).ToArray();
        int bx = (int)Math.Floor(xs.Min()), by = (int)Math.Floor(ys.Min());
        int bw = Math.Max(1, (int)Math.Ceiling(xs.Max()) - bx), bh = Math.Max(1, (int)Math.Ceiling(ys.Max()) - by);
        return (bx, by, bw, bh);
    }
}
