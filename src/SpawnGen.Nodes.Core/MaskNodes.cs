using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// The mask nodes. Masks are (batch, height, width) float32. Behavior measured with ComfyUI (2026-11-08) by running each node on chosen tensors;
// arithmetic is single precision throughout, as the measurements require.

/// <summary>A mask of one value: one picture of height x width, the value stored as a float.</summary>
[Node(Category = "image/mask", DisplayName = "Create Solid Mask")]
public sealed class SolidMask
{
    public MaskBatch Execute([Input(Min = 0.0, Max = 1.0, Step = 0.01)] float value = 1.0f,
        [Input(Min = 1, Max = 16384, Step = 1)] int width = 512,
        [Input(Min = 1, Max = 16384, Step = 1)] int height = 512) =>
        MaskBatch.Filled(1, height, width, value);
}

/// <summary>1 - mask for every value; values outside 0..1 are not clamped (measured: -0.5 gives 1.5).</summary>
[Node(Category = "image/mask", DisplayName = "Invert Mask")]
public sealed class InvertMask
{
    public MaskBatch Execute(MaskBatch mask)
    {
        var result = new MaskBatch(mask.Batch, mask.Height, mask.Width);
        for (int i = 0; i < mask.Data.Length; i++) result.Data[i] = 1f - mask.Data[i];
        return result;
    }
}

/// <summary>
/// The rows y .. y + height and columns x .. x + width of every mask, each range cut at the mask's edge like a Python slice. A corner past the
/// edge gives an empty (zero-sized) mask rather than being moved back inside (measured).
/// </summary>
[Node(Category = "image/mask", DisplayName = "Crop Mask")]
public sealed class CropMask
{
    public MaskBatch Execute(MaskBatch mask,
        [Input(Min = 0, Max = 16384, Step = 1)] int x = 0,
        [Input(Min = 0, Max = 16384, Step = 1)] int y = 0,
        [Input(Min = 1, Max = 16384, Step = 1)] int width = 512,
        [Input(Min = 1, Max = 16384, Step = 1)] int height = 512)
    {
        var (x0, x1) = SliceBounds(x, (long)x + width, mask.Width);
        var (y0, y1) = SliceBounds(y, (long)y + height, mask.Height);
        int w = x1 - x0, h = y1 - y0;
        var result = new MaskBatch(mask.Batch, h, w);
        for (int b = 0; b < mask.Batch; b++)
            for (int row = 0; row < h; row++)
                Array.Copy(mask.Data, (b * mask.Height + y0 + row) * mask.Width + x0, result.Data, (b * h + row) * w, w);
        return result;
    }

    /// <summary>Python's slice start:stop on a sequence of <paramref name="length"/> items, for non-negative start and stop.</summary>
    static (int Start, int Stop) SliceBounds(long start, long stop, int length)
    {
        int s = (int)Math.Min(start, length), e = (int)Math.Min(stop, length);
        return (s, Math.Max(s, e));
    }
}

/// <summary>
/// Pastes <paramref name="source"/> onto <paramref name="destination"/> at (x, y) with an operation, over the part of the source that falls inside the
/// destination, then clamps the whole result to 0..1. Measured with ComfyUI:
/// <list type="bullet">
/// <item>multiply, add, subtract: destination * source, destination + source, destination - source in float.</item>
/// <item>and, or, xor: both values are rounded half to even and the operation is logical (a value is true when its rounding is not 0), giving
/// 0 or 1: and(1.5, 1) = 1, and(0.5, 1) = 0, and(-0.6, 1) = 1; NaN counts as true.</item>
/// <item>The clamp covers the destination outside the pasted area too; NaN stays NaN.</item>
/// <item>The source batch must be 1 (used for every destination mask) or the destination's batch; anything else fails, even when nothing overlaps.</item>
/// </list>
/// </summary>
[Node(Category = "image/mask", DisplayName = "Combine Masks")]
public sealed class MaskComposite
{
    public MaskBatch Execute(MaskBatch destination, MaskBatch source,
        [Input(Min = 0, Max = 16384, Step = 1)] int x = 0,
        [Input(Min = 0, Max = 16384, Step = 1)] int y = 0,
        [Input(Options = ["multiply", "add", "subtract", "and", "or", "xor"])] string operation = "multiply")
    {
        Func<float, float, float> op = operation switch
        {
            "multiply" => (d, s) => d * s,
            "add" => (d, s) => d + s,
            "subtract" => (d, s) => d - s,
            "and" => (d, s) => Truth(d) && Truth(s) ? 1f : 0f,
            "or" => (d, s) => Truth(d) || Truth(s) ? 1f : 0f,
            "xor" => (d, s) => Truth(d) != Truth(s) ? 1f : 0f,
            _ => throw new ArgumentException($"Unknown operation '{operation}'.", nameof(operation)),
        };

        int dh = destination.Height, dw = destination.Width, sh = source.Height, sw = source.Width;
        // The destination area is rows y..y+sh and columns x..x+sw, cut at the edge; the source part used is its first (height - y) rows and
        // (width - x) columns as Python slices, so past the edge that count is negative and counts from the source's end. Measured with ComfyUI:
        // the two areas must have the same size (or the source part one row / column), checked last dimension first, and the source batch must be
        // 1 or the destination's, with these PyTorch messages; inside the edges both areas are the same size.
        int[] target = [destination.Batch, Span(y, sh, dh), Span(x, sw, dw)];
        int[] part = [source.Batch, Head(sh, (long)dh - y), Head(sw, (long)dw - x)];
        var combined = new int[3];
        for (int k = 2; k >= 0; k--)
        {
            if (target[k] != part[k] && target[k] != 1 && part[k] != 1)
                throw new ArgumentException($"The size of tensor a ({target[k]}) must match the size of tensor b ({part[k]}) at non-singleton dimension {k}");
            combined[k] = target[k] == 1 ? part[k] : target[k];
        }
        for (int k = 0; k < 3; k++)
            if (combined[k] != target[k])
                throw new ArgumentException($"The expanded size of the tensor ({target[k]}) must match the existing size ({combined[k]}) at non-singleton dimension {k}.  " +
                    $"Target sizes: [{string.Join(", ", target)}].  Tensor sizes: [{string.Join(", ", combined)}]");
        int h = target[1], w = target[2];

        var result = new MaskBatch(destination.Batch, dh, dw, (float[])destination.Data.Clone());
        for (int b = 0; b < destination.Batch; b++)
        {
            int sb = source.Batch == 1 ? 0 : b;
            for (int row = 0; row < h; row++)
            {
                int di = (b * dh + y + row) * dw + x, si = (sb * sh + row) * sw;
                for (int col = 0; col < w; col++) result.Data[di + col] = op(result.Data[di + col], source.Data[si + col]);
            }
        }
        for (int i = 0; i < result.Data.Length; i++) result.Data[i] = Math.Clamp(result.Data[i], 0f, 1f);
        return result;
    }

    /// <summary>The value rounded half to even, as a boolean: true unless it rounds to 0 (NaN is true).</summary>
    static bool Truth(float v) => MathF.Round(v, MidpointRounding.ToEven) != 0f;

    /// <summary>The length of the Python slice [start:start + count] of <paramref name="length"/> items (start, count non-negative).</summary>
    static int Span(int start, int count, int length) => (int)Math.Max(0, Math.Min(length, (long)start + count) - Math.Min(start, length));

    /// <summary>The length of the Python slice [:stop] of <paramref name="length"/> items; a negative stop counts from the end.</summary>
    static int Head(int length, long stop) => (int)(stop >= 0 ? Math.Min(length, stop) : Math.Max(0, length + stop));
}

/// <summary>
/// Fades the mask towards its edges: on each side the first <c>n</c> rows or columns (n = that side's amount, at most the mask's size) are
/// multiplied by (distance from the edge + 1) / n, so the outermost one gets 1/n. Measured with ComfyUI: the factor is rounded to float, the mask
/// is multiplied in float, left then right then top then bottom, and corners take the product of both sides.
/// </summary>
[Node(Category = "image/mask", DisplayName = "Feather Mask")]
public sealed class FeatherMask
{
    public MaskBatch Execute(MaskBatch mask,
        [Input(Min = 0, Max = 16384, Step = 1)] int left = 0,
        [Input(Min = 0, Max = 16384, Step = 1)] int top = 0,
        [Input(Min = 0, Max = 16384, Step = 1)] int right = 0,
        [Input(Min = 0, Max = 16384, Step = 1)] int bottom = 0)
    {
        int h = mask.Height, w = mask.Width;
        var result = new MaskBatch(mask.Batch, h, w, (float[])mask.Data.Clone());
        var d = result.Data;
        int l = Math.Min(left, w), r = Math.Min(right, w), t = Math.Min(top, h), bt = Math.Min(bottom, h);

        for (int b = 0; b < mask.Batch; b++)
        {
            int frame = b * h * w;
            for (int i = 0; i < l; i++) ScaleColumn(d, frame, w, h, i, Factor(i, l));
            for (int i = 0; i < r; i++) ScaleColumn(d, frame, w, h, w - 1 - i, Factor(i, r));
            for (int i = 0; i < t; i++) ScaleRow(d, frame, w, i, Factor(i, t));
            for (int i = 0; i < bt; i++) ScaleRow(d, frame, w, h - 1 - i, Factor(i, bt));
        }
        return result;
    }

    static float Factor(int distance, int feather) => (float)((distance + 1) / (double)feather);

    static void ScaleColumn(float[] d, int frame, int w, int h, int x, float f)
    {
        for (int y = 0; y < h; y++) d[frame + y * w + x] *= f;
    }

    static void ScaleRow(float[] d, int frame, int w, int y, float f)
    {
        for (int x = 0; x < w; x++) d[frame + y * w + x] *= f;
    }
}

/// <summary>
/// Grows (expand &gt; 0) or shrinks (expand &lt; 0) the mask by |expand| steps of grey-scale dilation or erosion: each step takes the maximum
/// (or minimum) over the pixel and its 4 neighbours (tapered corners) or its 8 neighbours (square). Past the border the edge pixel repeats.
/// Values are kept, not thresholded. Measured with ComfyUI (bit-exact on random masks, expand -3..4 and 50): a NaN pixel stays NaN and does not
/// spread, a NaN neighbour is ignored.
/// </summary>
[Node(Category = "image/mask", DisplayName = "Grow Mask")]
public sealed class GrowMask
{
    public MaskBatch Execute(MaskBatch mask,
        [Input(Min = -16384, Max = 16384, Step = 1)] int expand = 0,
        bool taperedCorners = true)
    {
        int h = mask.Height, w = mask.Width, n = h * w;
        var result = new MaskBatch(mask.Batch, h, w, (float[])mask.Data.Clone());
        if (expand == 0 || n == 0) return result;

        bool grow = expand > 0;
        int steps = Math.Abs(expand);
        (int Dy, int Dx)[] neighbours = taperedCorners
            ? [(-1, 0), (1, 0), (0, -1), (0, 1)]
            : [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)];

        var cur = new float[n];
        var next = new float[n];
        for (int b = 0; b < mask.Batch; b++)
        {
            Array.Copy(result.Data, b * n, cur, 0, n);
            for (int step = 0; step < steps; step++)
            {
                bool changed = false;
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        float v = cur[y * w + x];
                        foreach (var (dy, dx) in neighbours)
                        {
                            float u = cur[Math.Clamp(y + dy, 0, h - 1) * w + Math.Clamp(x + dx, 0, w - 1)];
                            if (grow ? u > v : u < v) { v = u; changed = true; }
                        }
                        next[y * w + x] = v;
                    }
                (cur, next) = (next, cur);
                if (!changed) break; // a step that changes nothing: every further step gives the same mask
            }
            Array.Copy(cur, 0, result.Data, b * n, n);
        }
        return result;
    }
}

/// <summary>1 where the mask is greater than <paramref name="value"/>, else 0. The comparison is in float (measured: a mask value of 0.2f is not above 0.2).</summary>
[Node(Category = "image/mask", DisplayName = "Threshold Mask")]
public sealed class ThresholdMask
{
    public MaskBatch Execute(MaskBatch mask, [Input(Min = 0.0, Max = 1.0, Step = 0.01)] float value = 0.5f)
    {
        var result = new MaskBatch(mask.Batch, mask.Height, mask.Width);
        for (int i = 0; i < mask.Data.Length; i++) result.Data[i] = mask.Data[i] > value ? 1f : 0f;
        return result;
    }
}
