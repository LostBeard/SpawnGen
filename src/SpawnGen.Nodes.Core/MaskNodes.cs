using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

[Node(Category = "mask", DisplayName = "SolidMask")]
public sealed class SolidMask
{
    public MaskBatch Execute([Input(Min = 0, Max = 1, Step = 0.01)] float value = 1f,
        [Input(Min = 1, Max = 16384)] int width = 512, [Input(Min = 1, Max = 16384)] int height = 512) =>
        MaskBatch.Filled(1, height, width, value);
}

[Node(Category = "mask", DisplayName = "InvertMask")]
public sealed class InvertMask
{
    public MaskBatch Execute(MaskBatch mask)
    {
        var r = new MaskBatch(mask.Batch, mask.Height, mask.Width);
        for (int i = 0; i < r.Data.Length; i++) r.Data[i] = 1f - mask.Data[i];
        return r;
    }
}

[Node(Category = "mask", DisplayName = "CropMask")]
public sealed class CropMask
{
    public MaskBatch Execute(MaskBatch mask, [Input(Min = 0, Max = 16384)] int x = 0, [Input(Min = 0, Max = 16384)] int y = 0,
        [Input(Min = 1, Max = 16384)] int width = 512, [Input(Min = 1, Max = 16384)] int height = 512)
    {
        x = Math.Min(x, mask.Width - 1);
        y = Math.Min(y, mask.Height - 1);
        int w = Math.Min(width, mask.Width - x), h = Math.Min(height, mask.Height - y);
        var r = new MaskBatch(mask.Batch, h, w);
        for (int b = 0; b < mask.Batch; b++)
            for (int row = 0; row < h; row++)
                Array.Copy(mask.Data, (b * mask.Height + y + row) * mask.Width + x, r.Data, (b * h + row) * w, w);
        return r;
    }
}

[Node(Category = "mask", DisplayName = "MaskComposite")]
public sealed class MaskComposite
{
    public static string[] Operations() => ["multiply", "add", "subtract", "and", "or", "xor"];

    /// <summary>Combines <paramref name="source"/> into <paramref name="destination"/> where they overlap, with the source's top-left corner at (x, y).</summary>
    public MaskBatch Execute(MaskBatch destination, MaskBatch source, [Input(Min = 0, Max = 16384)] int x = 0, [Input(Min = 0, Max = 16384)] int y = 0,
        [Input(OptionsProvider = nameof(Operations))] string operation = "multiply")
    {
        if (!Operations().Contains(operation)) throw new ArgumentException($"Unknown operation '{operation}'.", nameof(operation));
        var r = new MaskBatch(destination.Batch, destination.Height, destination.Width);
        Array.Copy(destination.Data, r.Data, destination.Data.Length);
        for (int b = 0; b < destination.Batch; b++)
        {
            int sb = b % source.Batch;
            for (int sy = 0; sy < source.Height; sy++)
            {
                int dy = y + sy;
                if (dy >= destination.Height) break;
                for (int sx = 0; sx < source.Width; sx++)
                {
                    int dx = x + sx;
                    if (dx >= destination.Width) break;
                    int di = (b * destination.Height + dy) * destination.Width + dx;
                    float d = r.Data[di], s = source.Data[(sb * source.Height + sy) * source.Width + sx];
                    r.Data[di] = Math.Clamp(operation switch
                    {
                        "multiply" => d * s,
                        "add" => d + s,
                        "subtract" => d - s,
                        "and" => MathF.Round(d) != 0 && MathF.Round(s) != 0 ? 1f : 0f,
                        "or" => MathF.Round(d) != 0 || MathF.Round(s) != 0 ? 1f : 0f,
                        _ => (MathF.Round(d) != 0) ^ (MathF.Round(s) != 0) ? 1f : 0f,
                    }, 0f, 1f);
                }
            }
        }
        return r;
    }
}

[Node(Category = "mask", DisplayName = "FeatherMask")]
public sealed class FeatherMask
{
    /// <summary>Fades the mask to 0 towards each edge over the given number of pixels (a linear ramp that reaches 1 at the inner end).</summary>
    public MaskBatch Execute(MaskBatch mask, [Input(Min = 0, Max = 16384)] int left = 0, [Input(Min = 0, Max = 16384)] int top = 0,
        [Input(Min = 0, Max = 16384)] int right = 0, [Input(Min = 0, Max = 16384)] int bottom = 0)
    {
        left = Math.Min(left, mask.Width); right = Math.Min(right, mask.Width);
        top = Math.Min(top, mask.Height); bottom = Math.Min(bottom, mask.Height);
        var r = new MaskBatch(mask.Batch, mask.Height, mask.Width);
        Array.Copy(mask.Data, r.Data, mask.Data.Length);
        for (int b = 0; b < mask.Batch; b++)
        {
            int o = b * mask.Height * mask.Width;
            for (int x = 0; x < left; x++) for (int y = 0; y < mask.Height; y++) r.Data[o + y * mask.Width + x] *= (x + 1f) / left;
            for (int x = 0; x < right; x++) for (int y = 0; y < mask.Height; y++) r.Data[o + y * mask.Width + (mask.Width - 1 - x)] *= (x + 1f) / right;
            for (int y = 0; y < top; y++) for (int x = 0; x < mask.Width; x++) r.Data[o + y * mask.Width + x] *= (y + 1f) / top;
            for (int y = 0; y < bottom; y++) for (int x = 0; x < mask.Width; x++) r.Data[o + (mask.Height - 1 - y) * mask.Width + x] *= (y + 1f) / bottom;
        }
        return r;
    }
}

[Node(Category = "mask", DisplayName = "GrowMask")]
public sealed class GrowMask
{
    /// <summary>Grows (positive) or shrinks (negative) the selected area by <c>expand</c> pixels. Tapered corners use a plus-shaped step (rounder results), otherwise a full 3x3 square.</summary>
    public MaskBatch Execute(MaskBatch mask, [Input(Min = -16384, Max = 16384)] int expand = 0, bool taperedCorners = true)
    {
        var r = new MaskBatch(mask.Batch, mask.Height, mask.Width);
        Array.Copy(mask.Data, r.Data, mask.Data.Length);
        if (expand == 0) return r;
        bool dilate = expand > 0;
        int w = mask.Width, h = mask.Height;
        for (int b = 0; b < mask.Batch; b++)
        {
            var cur = new float[w * h];
            Array.Copy(mask.Data, b * w * h, cur, 0, w * h);
            for (int step = 0; step < Math.Abs(expand); step++)
            {
                var next = new float[w * h];
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        float best = cur[y * w + x];
                        for (int dy = -1; dy <= 1; dy++)
                            for (int dx = -1; dx <= 1; dx++)
                            {
                                if (taperedCorners && dx != 0 && dy != 0) continue;
                                int nx = Math.Clamp(x + dx, 0, w - 1), ny = Math.Clamp(y + dy, 0, h - 1);
                                float v = cur[ny * w + nx];
                                best = dilate ? Math.Max(best, v) : Math.Min(best, v);
                            }
                        next[y * w + x] = best;
                    }
                cur = next;
            }
            Array.Copy(cur, 0, r.Data, b * w * h, w * h);
        }
        return r;
    }
}

[Node(Category = "mask", DisplayName = "ThresholdMask")]
public sealed class ThresholdMask
{
    public MaskBatch Execute(MaskBatch mask, [Input(Min = 0, Max = 1, Step = 0.01)] float value = 0.5f)
    {
        var r = new MaskBatch(mask.Batch, mask.Height, mask.Width);
        for (int i = 0; i < r.Data.Length; i++) r.Data[i] = mask.Data[i] > value ? 1f : 0f;
        return r;
    }
}
