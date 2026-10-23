using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// Joining, cutting and measuring latents along an axis (ComfyUI's <c>nodes_latent.py</c> / <c>nodes_images.py</c>). The axis names are ComfyUI's: x is the width,
/// y the height and t the third axis from the end, which is the frames of a 5-D video latent but the channels of SpawnGen's 4-D (batch, channels, height, width)
/// latent, exactly as ComfyUI treats a picture latent.
/// </summary>
internal static class LatentAxes
{
    /// <summary>The axis ComfyUI's dim option picks in a 4-D latent: x = 3 (width), y = 2 (height), t = 1 (ndim - 3).</summary>
    public static int Axis(string dim) =>
        dim.Contains('x') ? 3 : dim.Contains('y') ? 2 : dim.Contains('t') ? 1 : throw new ArgumentException($"Unknown dim '{dim}'.", nameof(dim));

    /// <summary>The latent as (outer, axis, inner): everything before the axis, the axis, everything after.</summary>
    public static (int Outer, int Length, int Inner) Split(int[] shape, int axis)
    {
        int outer = 1, inner = 1;
        for (int i = 0; i < axis; i++) outer *= shape[i];
        for (int i = axis + 1; i < shape.Length; i++) inner *= shape[i];
        return (outer, shape[axis], inner);
    }

    /// <summary><c>torch.narrow(t, axis, start, length)</c> (start already non-negative).</summary>
    public static (int[] Shape, float[] Data) Narrow(HostTensor t, int axis, int start, int length)
    {
        var (outer, size, inner) = Split(t.Shape, axis);
        var data = new float[outer * length * inner];
        for (int o = 0; o < outer; o++) Array.Copy(t.Data, (o * size + start) * inner, data, o * length * inner, length * inner);
        var shape = (int[])t.Shape.Clone();
        shape[axis] = length;
        return (shape, data);
    }
}

[Node(Category = "model/latent/advanced", DisplayName = "LatentConcat")]
public sealed class LatentConcat
{
    public static string[] Dims() => ["x", "-x", "y", "-y", "t", "-t"];

    /// <summary>
    /// Joins two latents along the width (x), the height (y) or the third axis from the end (t: the channels of a picture latent). With "-" the second comes first.
    /// The second is repeated or cut to the first's batch size; the other sizes must match.
    /// </summary>
    public LatentBatch Execute(LatentBatch samples1, LatentBatch samples2, [Input(OptionsProvider = nameof(Dims))] string dim = "x")
    {
        var s2 = Latents.RepeatToBatch(samples2, samples1.Batch);
        var (a, b) = dim.Contains('-') ? (s2, samples1) : (samples1, s2);
        int axis = LatentAxes.Axis(dim);
        for (int i = 0; i < 4; i++)
            if (i != axis && a.Shape[i] != b.Shape[i])
                throw new ArgumentException($"Sizes of tensors must match except in dimension {axis - 4}: expected {a.Shape[i]} but got {b.Shape[i]} in dimension {i}.");
        var (outer, la, inner) = LatentAxes.Split(a.Shape, axis);
        int lb = b.Shape[axis];
        var data = new float[outer * (la + lb) * inner];
        for (int o = 0; o < outer; o++)
        {
            Array.Copy(a.Data, o * la * inner, data, o * (la + lb) * inner, la * inner);
            Array.Copy(b.Data, o * lb * inner, data, (o * (la + lb) + la) * inner, lb * inner);
        }
        var shape = (int[])a.Shape.Clone();
        shape[axis] = la + lb;
        return Latents.Like(samples1, shape[0], shape[1], shape[2], shape[3], data);
    }
}

[Node(Category = "model/latent/advanced", DisplayName = "LatentCut")]
public sealed class LatentCut
{
    public static string[] Dims() => ["x", "y", "t"];

    /// <summary>
    /// <c>amount</c> cells from <c>index</c> along x, y or t (the channels of a picture latent). A negative index counts from the end; both are clamped to what is there.
    /// </summary>
    public LatentBatch Execute(LatentBatch samples, [Input(OptionsProvider = nameof(Dims))] string dim = "x",
        [Input(Min = -16384, Max = 16384, Step = 1)] int index = 0, [Input(Min = 1, Max = 16384, Step = 1)] int amount = 1)
    {
        int axis = LatentAxes.Axis(dim), size = samples.Shape[axis];
        if (index >= 0)
        {
            index = Math.Min(index, size - 1);
            amount = Math.Min(size - index, amount);
        }
        else
        {
            index = Math.Max(index, -size);
            amount = Math.Min(-index, amount);
        }
        var (shape, data) = LatentAxes.Narrow(samples, axis, index < 0 ? index + size : index, amount);
        return Latents.Like(samples, shape[0], shape[1], shape[2], shape[3], data);
    }
}

[Node(Category = "model/latent/advanced", DisplayName = "LatentCutToBatch")]
public sealed class LatentCutToBatch
{
    public static string[] Dims() => ["t", "x", "y"];

    /// <summary>
    /// Cuts the latent into slices of <c>slice_size</c> along x or y and stacks them as a batch (picture by picture, slice by slice); a remainder that does not fill
    /// a slice is dropped. t is the channel axis of a picture latent, which ComfyUI leaves alone, so the latent passes through.
    /// </summary>
    public LatentBatch Execute(LatentBatch samples, [Input(OptionsProvider = nameof(Dims))] string dim = "t", [Input(Min = 1, Max = 16384, Step = 1)] int sliceSize = 1)
    {
        int axis = LatentAxes.Axis(dim);
        if (axis < 2) return samples;
        int length = samples.Shape[axis];
        if (length < sliceSize) sliceSize = length;
        int slices = length / sliceSize; // a remainder is cut off
        int nb = samples.Batch * slices, c = samples.Channels;
        int h = axis == 2 ? sliceSize : samples.Height, w = axis == 3 ? sliceSize : samples.Width;
        var data = new float[nb * c * h * w];
        for (int b = 0; b < samples.Batch; b++)
            for (int j = 0; j < slices; j++)
                for (int ch = 0; ch < c; ch++)
                    for (int y = 0; y < h; y++)
                        for (int x = 0; x < w; x++)
                        {
                            int sy = axis == 2 ? j * sliceSize + y : y, sx = axis == 3 ? j * sliceSize + x : x;
                            data[(((b * slices + j) * c + ch) * h + y) * w + x] = samples.Data[((b * c + ch) * samples.Height + sy) * samples.Width + sx];
                        }
        return Latents.Like(samples, nb, c, h, w, data);
    }
}

[Node(Category = "latent", DisplayName = "Get Image/Video Latent Size", Description = "Returns the width, height, frame count, and batch size of an image or video latent.")]
public sealed class GetLatentSize
{
    /// <summary>The latent's size in latent cells. A picture latent has one frame. The numbers are also shown on the node.</summary>
    public (int Width, int Height, int Frames, int BatchSize) Execute([Input(Tooltip = "The image or video latent to inspect.")] LatentBatch latent, NodeContext context)
    {
        int width = latent.Width, height = latent.Height, frames = 1, batchSize = latent.Batch;
        context.Status($"width: {width}, height: {height}\n frames: {frames}, batch size: {batchSize}");
        return (width, height, frames, batchSize);
    }
}
