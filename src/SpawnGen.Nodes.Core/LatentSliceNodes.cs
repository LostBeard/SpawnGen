using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// The named axes of the slicing nodes on a (batch, channels, height, width) latent: x is the width, y the height and t the third axis from the end,
/// which for a picture latent is the channels.
/// </summary>
public static class LatentAxes
{
    /// <summary>The axis index (1..3) of "x", "y" or "t".</summary>
    public static int Of(string dim) => dim switch
    {
        "x" => 3,
        "y" => 2,
        "t" => 1,
        _ => throw new ArgumentException($"Unknown dimension '{dim}': expected x, y or t.", nameof(dim)),
    };

    /// <summary>The cells <paramref name="start"/> .. start + count - 1 along <paramref name="axis"/>.</summary>
    public static LatentBatch Narrow(LatentBatch l, int axis, int start, int count)
    {
        var shape = (int[])l.Shape.Clone();
        shape[axis] = count;
        var r = new LatentBatch(shape[0], shape[1], shape[2], shape[3]);
        int outer = 1, inner = 1;
        for (int d = 0; d < axis; d++) outer *= l.Shape[d];
        for (int d = axis + 1; d < 4; d++) inner *= l.Shape[d];
        int size = l.Shape[axis];
        for (int o = 0; o < outer; o++)
            Array.Copy(l.Data, (o * size + start) * inner, r.Data, o * count * inner, count * inner);
        return r;
    }

    /// <summary>
    /// The two latents joined along <paramref name="axis"/>, <paramref name="a"/> first. Every other axis must agree (PyTorch's torch.cat message,
    /// measured).
    /// </summary>
    public static LatentBatch Cat(LatentBatch a, LatentBatch b, int axis)
    {
        for (int d = 0; d < 4; d++)
            if (d != axis && a.Shape[d] != b.Shape[d])
                throw new ArgumentException($"Sizes of tensors must match except in dimension {axis}. Expected size {a.Shape[d]} but got size {b.Shape[d]} for tensor number 1 in the list.");
        var shape = (int[])a.Shape.Clone();
        shape[axis] += b.Shape[axis];
        var r = new LatentBatch(shape[0], shape[1], shape[2], shape[3]);
        int outer = 1, inner = 1;
        for (int d = 0; d < axis; d++) outer *= a.Shape[d];
        for (int d = axis + 1; d < 4; d++) inner *= a.Shape[d];
        int na = a.Shape[axis] * inner, nb = b.Shape[axis] * inner;
        for (int o = 0; o < outer; o++)
        {
            Array.Copy(a.Data, o * na, r.Data, o * (na + nb), na);
            Array.Copy(b.Data, o * nb, r.Data, o * (na + nb) + na, nb);
        }
        return r;
    }
}

[Node(Category = "latent", DisplayName = "Get Image/Video Latent Size", OutputNames = ["width", "height", "frames", "batch_size"],
    Description = "Returns the width, height, frame count, and batch size of an image or video latent.")]
public sealed class GetLatentSize
{
    /// <summary>The latent's width and height in cells, its frame count (1 for a picture latent) and the number of pictures.</summary>
    public (int Width, int Height, int Frames, int BatchSize) Execute([Input(Tooltip = "The image or video latent to inspect.")] LatentBatch latent) =>
        (latent.Width, latent.Height, 1, latent.Batch);
}

[Node(Category = "model/latent/advanced", DisplayName = "LatentConcat")]
public sealed class LatentConcat
{
    /// <summary>
    /// Joins the two latents along x, y or t; with "-" the second comes first. Measured with ComfyUI: the second latent's batch is read around or cut to
    /// the first's (batch 2 with 3 gives 2 pictures, 3 with 2 gives 3); the other sizes must match.
    /// </summary>
    public LatentBatch Execute(LatentBatch samples1, LatentBatch samples2, [Input(Options = ["x", "-x", "y", "-y", "t", "-t"])] string dim = "x")
    {
        bool reversed = dim.StartsWith('-');
        int axis = LatentAxes.Of(reversed ? dim[1..] : dim);
        var second = Latents.RepeatToBatch(samples2, samples1.Batch);
        var r = reversed ? LatentAxes.Cat(second, samples1, axis) : LatentAxes.Cat(samples1, second, axis);
        return Latents.Like(samples1, r.Batch, r.Channels, r.Height, r.Width, r.Data);
    }
}

[Node(Category = "model/latent/advanced", DisplayName = "LatentCut")]
public sealed class LatentCut
{
    /// <summary>
    /// Keeps <paramref name="amount"/> cells along x, y or t from <paramref name="index"/>. A negative index counts from the end (at most the whole axis back)
    /// and then keeps at most up to the end; an index past the end is the last cell.
    /// </summary>
    public LatentBatch Execute(LatentBatch samples, [Input(Options = ["x", "y", "t"])] string dim = "x",
        [Input(Min = -16384, Max = 16384)] int index = 0, [Input(Min = 1, Max = 16384)] int amount = 1)
    {
        int axis = LatentAxes.Of(dim), size = samples.Shape[axis];
        int start;
        if (index >= 0)
        {
            start = Math.Min(index, size - 1);
            amount = Math.Min(size - start, amount);
        }
        else
        {
            index = Math.Max(index, -size);
            start = size + index;
            amount = Math.Min(-index, amount);
        }
        var r = LatentAxes.Narrow(samples, axis, start, amount);
        return Latents.Like(samples, r.Batch, r.Channels, r.Height, r.Width, r.Data);
    }
}

[Node(Category = "model/latent/advanced", DisplayName = "LatentCutToBatch")]
public sealed class LatentCutToBatch
{
    /// <summary>
    /// Cuts the latent along x or y into slices of <paramref name="sliceSize"/> cells (at most the whole axis), drops what is left over, and makes each
    /// slice a picture: picture b's slices in order, then the next picture's. "t" leaves a picture latent as it is.
    /// </summary>
    public LatentBatch Execute(LatentBatch samples, [Input(Options = ["t", "x", "y"])] string dim = "t", [Input(Min = 1, Max = 16384)] int sliceSize = 1)
    {
        int axis = LatentAxes.Of(dim);
        if (axis == 1) return samples;
        int size = samples.Shape[axis], slice = Math.Min(sliceSize, size), count = size / slice;
        int h = axis == 2 ? slice : samples.Height, w = axis == 3 ? slice : samples.Width, c = samples.Channels;
        var r = new LatentBatch(samples.Batch * count, c, h, w);
        int one = c * h * w;
        for (int b = 0; b < samples.Batch; b++)
        {
            var picture = LatentAxes.Narrow(samples, 0, b, 1);
            for (int s = 0; s < count; s++)
                Array.Copy(LatentAxes.Narrow(picture, axis, s * slice, slice).Data, 0, r.Data, (b * count + s) * one, one);
        }
        return r;
    }
}
