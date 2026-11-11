using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// The latent nodes. Behavior measured with ComfyUI (2026-11-08) by running each node on chosen and random latents. Unless a node says otherwise,
// its result keeps the first latent's noise mask and batch index unchanged (measured: every node here returns the first input's noise_mask and
// batch_index as they were, even where the samples change size).

/// <summary>Helpers the latent nodes share.</summary>
public static class Latents
{
    /// <summary>A latent with new samples that keeps <paramref name="template"/>'s noise mask and batch index.</summary>
    public static LatentBatch Like(LatentBatch template, int batch, int channels, int height, int width, float[] data) =>
        new(batch, channels, height, width, data) { NoiseMask = template.NoiseMask, BatchIndex = template.BatchIndex };

    /// <summary>
    /// The samples with <paramref name="batch"/> pictures: picture i is picture i mod n of the source (a longer batch is cut, a shorter one read around).
    /// Only the samples: no noise mask or batch index. The latent itself when the count already matches.
    /// </summary>
    public static LatentBatch RepeatToBatch(LatentBatch latent, int batch)
    {
        if (latent.Batch == batch) return latent;
        return new LatentBatch(batch, latent.Channels, latent.Height, latent.Width, Repeat(latent.Data, latent.Batch, batch));
    }

    /// <summary>The masks with <paramref name="batch"/> entries, read around as <see cref="RepeatToBatch(LatentBatch, int)"/> does.</summary>
    public static MaskBatch RepeatToBatch(MaskBatch mask, int batch)
    {
        if (mask.Batch == batch) return mask;
        return new MaskBatch(batch, mask.Height, mask.Width, Repeat(mask.Data, mask.Batch, batch));
    }

    static float[] Repeat(float[] data, int count, int batch)
    {
        if (count == 0) throw new ArgumentException("Cannot repeat an empty batch.");
        int size = data.Length / count;
        var r = new float[batch * size];
        for (int i = 0; i < batch; i++) Array.Copy(data, i % count * size, r, i * size, size);
        return r;
    }

    /// <summary>Python's <c>seq[start:stop]</c> on a sequence of <paramref name="length"/>: negative ends count from the end, both are clamped.</summary>
    internal static (int Start, int Count) Slice(int start, int stop, int length)
    {
        static int Fix(int i, int n) => i < 0 ? Math.Max(0, i + n) : Math.Min(i, n);
        int a = Fix(start, length), b = Fix(stop, length);
        return (a, Math.Max(0, b - a));
    }

    /// <summary>
    /// The second latent brought to the first one's shape for the latent arithmetic nodes (Add, Subtract, Interpolate). Measured: another height or
    /// width is cut to the first's aspect ratio around its centre and resized bilinearly to the first's size (a stretch without the cut differs by
    /// whole values on random latents); then the batch is read around or cut to the first's.
    /// </summary>
    internal static LatentBatch ShapedLike(LatentBatch second, LatentBatch first)
    {
        if (second.Height != first.Height || second.Width != first.Width)
        {
            var (data, h, w) = TorchInterpolate.CommonUpscale(second.Data, second.Batch, second.Channels, second.Height, second.Width, first.Width, first.Height, "bilinear", "center");
            second = new LatentBatch(second.Batch, second.Channels, h, w, data);
        }
        return RepeatToBatch(second, first.Batch);
    }

    /// <summary>
    /// PyTorch broadcasting of two sizes along one dimension: equal, or one of them 1. Otherwise PyTorch's error for an element-wise operation
    /// (measured message).
    /// </summary>
    internal static int Broadcast(int a, int b, int dim) =>
        a == b || b == 1 ? a : a == 1 ? b : throw new ArgumentException($"The size of tensor a ({a}) must match the size of tensor b ({b}) at non-singleton dimension {dim}");

    /// <summary>
    /// <paramref name="f"/> applied element by element to two latents of the same height and width, broadcasting batch and channels as PyTorch does.
    /// </summary>
    internal static (float[] Data, int Batch, int Channels) Combine(LatentBatch a, LatentBatch b, Func<float, float, float> f)
    {
        int n = Broadcast(a.Batch, b.Batch, 0), c = Broadcast(a.Channels, b.Channels, 1), plane = a.Height * a.Width;
        var r = new float[n * c * plane];
        for (int i = 0; i < n; i++)
            for (int k = 0; k < c; k++)
            {
                int ia = ((a.Batch == 1 ? 0 : i) * a.Channels + (a.Channels == 1 ? 0 : k)) * plane;
                int ib = ((b.Batch == 1 ? 0 : i) * b.Channels + (b.Channels == 1 ? 0 : k)) * plane;
                int o = (i * c + k) * plane;
                for (int p = 0; p < plane; p++) r[o + p] = f(a.Data[ia + p], b.Data[ib + p]);
            }
        return (r, n, c);
    }

    /// <summary>
    /// The check made when a block of samples is written into another latent: each of its sizes must equal the target's or be 1 (then it is spread).
    /// The message is PyTorch's (measured); like PyTorch it lists the block's sizes without its leading 1s.
    /// </summary>
    internal static void CheckPaste(int[] target, int[] block)
    {
        for (int d = 0; d < target.Length; d++)
            if (block[d] != target[d] && block[d] != 1)
            {
                int lead = 0;
                while (lead < block.Length - 1 && block[lead] == 1) lead++;
                throw new ArgumentException($"The expanded size of the tensor ({target[d]}) must match the existing size ({block[d]}) at non-singleton dimension {d}.  " +
                    $"Target sizes: [{string.Join(", ", target)}].  Tensor sizes: [{string.Join(", ", block.Skip(lead))}]");
            }
    }
}

[Node(Category = "model/latent", DisplayName = "Upscale Latent")]
public sealed class LatentUpscale
{
    public static string[] Methods() => TorchInterpolate.LatentMethods;

    /// <summary>
    /// Resizes the latent to <paramref name="width"/> x <paramref name="height"/> pixels (an eighth of that in cells, rounded down). Measured with ComfyUI:
    /// <list type="bullet">
    /// <item>Both 0: the latent is returned as it is.</item>
    /// <item>Each given side is raised to at least 64 pixels.</item>
    /// <item>One side 0: it follows the latent's aspect ratio from the other side after that side was raised to 64, as
    /// <c>round(cells_w * height / cells_h)</c> in double precision, rounded half to even, then also raised to 64 (latent 7 x 14 at height 495 gives 31 cells
    /// wide, which other orders of that arithmetic do not).</item>
    /// <item>The noise mask is not resized.</item>
    /// </list>
    /// </summary>
    public LatentBatch Execute(LatentBatch samples, [Input(OptionsProvider = nameof(Methods))] string upscaleMethod = "nearest-exact",
        [Input(Min = 0, Max = 16384, Step = 8)] int width = 512, [Input(Min = 0, Max = 16384, Step = 8)] int height = 512,
        [Input(Options = ["disabled", "center"])] string crop = "disabled")
    {
        if (width == 0 && height == 0) return samples;
        if (width == 0)
        {
            height = Math.Max(64, height);
            width = Math.Max(64, (int)Math.Round((double)samples.Width * height / samples.Height, MidpointRounding.ToEven));
        }
        else if (height == 0)
        {
            width = Math.Max(64, width);
            height = Math.Max(64, (int)Math.Round((double)samples.Height * width / samples.Width, MidpointRounding.ToEven));
        }
        else
        {
            width = Math.Max(64, width);
            height = Math.Max(64, height);
        }
        return Resized(samples, width / 8, height / 8, upscaleMethod, crop);
    }

    /// <summary>The samples resized to <paramref name="w"/> x <paramref name="h"/> cells; the noise mask and batch index are kept.</summary>
    internal static LatentBatch Resized(LatentBatch samples, int w, int h, string method, string crop)
    {
        if (w <= 0 || h <= 0)
        {
            // Measured: "area" gives an empty latent; the other methods fail with PyTorch's message (bislerp resizes a one-row strip first).
            if (method == "area") return Latents.Like(samples, samples.Batch, samples.Channels, Math.Max(0, h), Math.Max(0, w), []);
            // bislerp resizes along the width, then along the height, each as a one-row strip.
            var (inH, inW, outH, outW) = method != "bislerp" ? (samples.Height, samples.Width, h, w)
                : w <= 0 ? (1, samples.Width, 1, w) : (1, samples.Height, 1, h);
            throw new ArgumentException($"Input and output sizes should be greater than 0, but got input (H: {inH}, W: {inW}) output (H: {outH}, W: {outW})");
        }
        var (data, height, width) = TorchInterpolate.CommonUpscale(samples.Data, samples.Batch, samples.Channels, samples.Height, samples.Width, w, h, method, crop);
        return Latents.Like(samples, samples.Batch, samples.Channels, height, width, data);
    }
}

[Node(Category = "model/latent", DisplayName = "Upscale Latent By")]
public sealed class LatentUpscaleBy
{
    public static string[] Methods() => TorchInterpolate.LatentMethods;

    /// <summary>
    /// Resizes the latent by <paramref name="scaleBy"/>. Measured with ComfyUI: each side is <c>round(cells * scale_by)</c> (double precision, half to even:
    /// 3 x 1.5 = 4.5 gives 4, 5 x 1.5 = 7.5 gives 8), with no minimum; the noise mask is not resized.
    /// </summary>
    public LatentBatch Execute(LatentBatch samples, [Input(OptionsProvider = nameof(Methods))] string upscaleMethod = "nearest-exact",
        [Input(Min = 0.01, Max = 8.0, Step = 0.01)] double scaleBy = 1.5)
    {
        int w = (int)Math.Round(samples.Width * scaleBy, MidpointRounding.ToEven), h = (int)Math.Round(samples.Height * scaleBy, MidpointRounding.ToEven);
        return LatentUpscale.Resized(samples, w, h, upscaleMethod, "disabled");
    }
}

[Node(Category = "model/latent/batch", DisplayName = "Get Latent From Batch")]
public sealed class LatentFromBatch
{
    /// <summary>
    /// <paramref name="length"/> pictures from <paramref name="batchIndex"/> on. Measured with ComfyUI:
    /// <list type="bullet">
    /// <item>A negative index counts from the end; the start is clamped to the batch (-10 of 4 is 0, 10 of 4 is 3), the length to what is left.</item>
    /// <item>The batch index is the latent's own (or 0, 1, ...), cut the same way, so each picture keeps its noise.</item>
    /// <item>A noise mask of one picture is kept whole; otherwise the masks are read around (or cut) to the batch size and cut the same way.</item>
    /// </list>
    /// </summary>
    public LatentBatch Execute(LatentBatch samples, [Input(Min = -16384, Max = 16384)] int batchIndex = 0, [Input(Min = 1, Max = 64)] int length = 1)
    {
        int count = samples.Batch;
        int start = batchIndex < 0 ? Math.Max(0, count + batchIndex) : Math.Min(count - 1, batchIndex);
        int take = Math.Min(count - start, length);
        int size = samples.Channels * samples.Height * samples.Width;
        var data = new float[take * size];
        Array.Copy(samples.Data, start * size, data, 0, take * size);

        var mask = samples.NoiseMask;
        if (mask is not null && mask.Batch != 1)
        {
            var all = Latents.RepeatToBatch(mask, count);
            int plane = mask.Height * mask.Width;
            var cut = new float[take * plane];
            Array.Copy(all.Data, start * plane, cut, 0, take * plane);
            mask = new MaskBatch(take, mask.Height, mask.Width, cut);
        }
        int[] index = (samples.BatchIndex ?? Enumerable.Range(0, count).ToArray()).Skip(start).Take(take).ToArray();
        return new LatentBatch(take, samples.Channels, samples.Height, samples.Width, data) { NoiseMask = mask, BatchIndex = index };
    }
}

[Node(Category = "model/latent/batch", DisplayName = "Repeat Latent Batch")]
public sealed class RepeatLatentBatch
{
    /// <summary>
    /// The whole batch <paramref name="amount"/> times over. Measured with ComfyUI: a noise mask of more than one picture is repeated the same number of
    /// times (whatever its count), a single one is kept; a batch index continues, each copy shifted by (largest - smallest + 1) of the original
    /// (<c>[4, 1, 2]</c> twice is <c>[4, 1, 2, 8, 5, 6]</c>); without one none is made.
    /// </summary>
    public LatentBatch Execute(LatentBatch samples, [Input(Min = 1, Max = 64)] int amount = 1)
    {
        var data = new float[samples.Data.Length * amount];
        for (int k = 0; k < amount; k++) Array.Copy(samples.Data, 0, data, k * samples.Data.Length, samples.Data.Length);

        var mask = samples.NoiseMask;
        if (mask is not null && mask.Batch > 1)
        {
            var m = new float[mask.Data.Length * amount];
            for (int k = 0; k < amount; k++) Array.Copy(mask.Data, 0, m, k * mask.Data.Length, mask.Data.Length);
            mask = new MaskBatch(mask.Batch * amount, mask.Height, mask.Width, m);
        }
        int[]? index = samples.BatchIndex;
        if (index is { Length: > 0 } && amount > 1)
        {
            int offset = index.Max() - index.Min() + 1;
            index = Enumerable.Range(0, amount).SelectMany(k => index.Select(i => i + k * offset)).ToArray();
        }
        return new LatentBatch(samples.Batch * amount, samples.Channels, samples.Height, samples.Width, data) { NoiseMask = mask, BatchIndex = index };
    }
}

[Node(Category = "model/latent", DisplayName = "Latent Composite")]
public sealed class LatentComposite
{
    /// <summary>
    /// Pastes <paramref name="samplesFrom"/> into <paramref name="samplesTo"/> at (x, y) pixels (x / 8, y / 8 cells, rounded down). Measured with ComfyUI:
    /// <list type="bullet">
    /// <item>The target area is the target's rows and columns from (y, x) on, as many as the pasted latent has and the target still has; the pasted
    /// latent is cut to <c>[:height - y, :width - x]</c> as a Python slice cuts (a negative end counts from its end). The cut is then written into the
    /// area, spread where it has size 1 (a batch or channel count of 1 is used for all; one leftover row or column beyond the target is spread over an
    /// empty area, changing nothing); other sizes that differ fail with PyTorch's message.</item>
    /// <item>With <paramref name="feather"/> (f = feather / 8 cells) a weight the size of the cut fades the edges that lie inside the target
    /// (top when y &gt; 0, bottom when y + its height &lt; the target's height, and so for left and right): for t = 0 .. f - 1, row or column t from
    /// such an edge is multiplied by (t + 1) / f (single precision, edges in the order top, bottom, left, right for each t). The far edges are counted
    /// the way a Python slice <c>[h - 1 - t : h - t]</c> counts, so when f is larger than the paste a weight can land near the other edge (f = 4 over 2
    /// rows weights both rows .25 x .5). The area becomes from x weight + to x (1 - weight). A paste that starts beyond the target and leaves a cut of
    /// two or more rows or columns fails (with or without feather; ComfyUI's feathered message is not always reproduced).</item>
    /// </list>
    /// </summary>
    public LatentBatch Execute(LatentBatch samplesTo, LatentBatch samplesFrom, [Input(Min = 0, Max = 16384, Step = 8)] int x = 0,
        [Input(Min = 0, Max = 16384, Step = 8)] int y = 0, [Input(Min = 0, Max = 16384, Step = 8)] int feather = 0)
    {
        x /= 8;
        y /= 8;
        int f = feather / 8;
        var to = samplesTo;
        var from = samplesFrom;
        int H = to.Height, W = to.Width, fh = from.Height, fw = from.Width;
        int[] target = [to.Batch, to.Channels, Latents.Slice(y, y + fh, H).Count, Latents.Slice(x, x + fw, W).Count];
        int[] cut = [from.Batch, from.Channels, Latents.Slice(0, H - y, fh).Count, Latents.Slice(0, W - x, fw).Count];
        int ch = cut[2], cw = cut[3];
        int[] weightShape = [1, 1, ch, cw];

        // Shapes of from x weight (a), to x (1 - weight) (b) and their sum (r), checked in that order as PyTorch checks them.
        int[] a = cut, b = target, r = cut;
        if (f > 0)
        {
            a = Shape(cut, weightShape);
            b = Shape(target, weightShape);
            r = Shape(a, b);
        }
        Latents.CheckPaste(target, r);

        float[]? weight = null;
        if (f > 0)
        {
            weight = new float[ch * cw];
            Array.Fill(weight, 1f);
            for (int t = 0; t < f; t++)
            {
                float factor = (float)((1.0 + t) / f);
                if (y != 0) Rows(weight, ch, cw, t, t + 1, factor);
                if (y + ch < H) Rows(weight, ch, cw, ch - 1 - t, ch - t, factor);
                if (x != 0) Columns(weight, ch, cw, t, t + 1, factor);
                if (x + cw < W) Columns(weight, ch, cw, cw - 1 - t, cw - t, factor);
            }
        }

        // Per axis and output position: where it reads the cut (F), the weight for from (M1), the target (T) and the weight for to (M2).
        var maps = new (int[] F, int[] M1, int[] T, int[] M2)[4];
        for (int d = 0; d < 4; d++)
        {
            int n = target[d];
            var map = (F: new int[n], M1: new int[n], T: new int[n], M2: new int[n]);
            for (int i = 0; i < n; i++)
            {
                int ir = r[d] == 1 ? 0 : i;
                int ia = a[d] == 1 ? 0 : ir, ib = b[d] == 1 ? 0 : ir;
                map.F[i] = cut[d] == 1 ? 0 : ia;
                map.M1[i] = weightShape[d] == 1 ? 0 : ia;
                map.T[i] = target[d] == 1 ? 0 : ib;
                map.M2[i] = weightShape[d] == 1 ? 0 : ib;
            }
            maps[d] = map;
        }

        var data = (float[])to.Data.Clone();
        for (int n = 0; n < target[0]; n++)
            for (int c = 0; c < target[1]; c++)
            {
                int dst = (n * to.Channels + c) * H * W;
                int src = (maps[0].F[n] * from.Channels + maps[1].F[c]) * fh * fw;
                int old = (maps[0].T[n] * to.Channels + maps[1].T[c]) * H * W;
                for (int row = 0; row < target[2]; row++)
                    for (int col = 0; col < target[3]; col++)
                    {
                        float v = from.Data[src + maps[2].F[row] * fw + maps[3].F[col]];
                        if (weight is not null)
                        {
                            float m1 = weight[maps[2].M1[row] * cw + maps[3].M1[col]], m2 = weight[maps[2].M2[row] * cw + maps[3].M2[col]];
                            v = v * m1 + to.Data[old + (y + maps[2].T[row]) * W + x + maps[3].T[col]] * (1f - m2);
                        }
                        data[dst + (y + row) * W + x + col] = v;
                    }
            }
        return Latents.Like(to, to.Batch, to.Channels, H, W, data);
    }

    /// <summary>The broadcast shape of two 4-D shapes, the last dimension checked first (PyTorch's order and message).</summary>
    static int[] Shape(int[] a, int[] b)
    {
        var r = new int[4];
        for (int d = 3; d >= 0; d--) r[d] = Latents.Broadcast(a[d], b[d], d);
        return r;
    }

    static void Rows(float[] weight, int h, int w, int start, int stop, float factor)
    {
        var (s, n) = Latents.Slice(start, stop, h);
        for (int r = s; r < s + n; r++)
            for (int q = 0; q < w; q++) weight[r * w + q] *= factor;
    }

    static void Columns(float[] weight, int h, int w, int start, int stop, float factor)
    {
        var (s, n) = Latents.Slice(start, stop, w);
        for (int r = 0; r < h; r++)
            for (int q = s; q < s + n; q++) weight[r * w + q] *= factor;
    }
}

[Node(Category = "model/latent", DisplayName = "Latent Composite Masked")]
public sealed class LatentCompositeMasked
{
    /// <summary>
    /// Pastes <paramref name="source"/> into <paramref name="destination"/> at (x, y) pixels (x / 8, y / 8 cells), through <paramref name="mask"/>.
    /// Measured with ComfyUI:
    /// <list type="bullet">
    /// <item><paramref name="resizeSource"/>: the source is first stretched bilinearly to the destination's size (no cropping).</item>
    /// <item>The source batch is read around or cut to the destination's; its channels must equal the destination's or be 1.</item>
    /// <item>The mask is resized bilinearly to the source's size, read around or cut to the destination's batch, and cut with the source to what
    /// fits. Its values are used as they are (not clamped): result = source x mask + destination x (1 - mask). No mask is a mask of 1 (measured: an infinite destination value still turns into NaN, inf x 0).</item>
    /// </list>
    /// </summary>
    public LatentBatch Execute(LatentBatch destination, LatentBatch source, [Input(Min = 0, Max = 16384, Step = 8)] int x = 0,
        [Input(Min = 0, Max = 16384, Step = 8)] int y = 0, bool resizeSource = false, MaskBatch? mask = null)
    {
        x /= 8;
        y /= 8;
        var dst = destination;
        var src = source;
        if (resizeSource && (src.Height != dst.Height || src.Width != dst.Width))
            src = new LatentBatch(src.Batch, src.Channels, dst.Height, dst.Width,
                TorchInterpolate.Resize(src.Data, src.Batch * src.Channels, src.Height, src.Width, dst.Height, dst.Width, "bilinear"));
        src = Latents.RepeatToBatch(src, dst.Batch);

        var data = (float[])dst.Data.Clone();
        int rh = Math.Max(0, Math.Min(src.Height, dst.Height - y)), rw = Math.Max(0, Math.Min(src.Width, dst.Width - x));
        if (rh == 0 || rw == 0) return Latents.Like(dst, dst.Batch, dst.Channels, dst.Height, dst.Width, data);
        Latents.CheckPaste([dst.Batch, dst.Channels, rh, rw], [src.Batch, src.Channels, rh, rw]);

        MaskBatch? m = null;
        if (mask is not null)
        {
            m = mask;
            if (m.Height != src.Height || m.Width != src.Width)
                m = new MaskBatch(m.Batch, src.Height, src.Width, TorchInterpolate.Resize(m.Data, m.Batch, m.Height, m.Width, src.Height, src.Width, "bilinear"));
            m = Latents.RepeatToBatch(m, dst.Batch);
        }

        int plane = dst.Height * dst.Width, srcPlane = src.Height * src.Width;
        for (int n = 0; n < dst.Batch; n++)
            for (int c = 0; c < dst.Channels; c++)
            {
                int to = (n * dst.Channels + c) * plane, from = (n * src.Channels + (src.Channels == 1 ? 0 : c)) * srcPlane;
                for (int r = 0; r < rh; r++)
                    for (int q = 0; q < rw; q++)
                    {
                        int o = to + (y + r) * dst.Width + x + q;
                        float v = src.Data[from + r * src.Width + q];
                        float w = m is null ? 1f : m.Data[n * srcPlane + r * src.Width + q];
                        data[o] = v * w + data[o] * (1f - w);
                    }
            }
        return Latents.Like(dst, dst.Batch, dst.Channels, dst.Height, dst.Width, data);
    }
}

[Node(Category = "model/latent/transform", DisplayName = "Flip Latent")]
public sealed class LatentFlip
{
    /// <summary>"x-axis: vertically" turns the rows upside down, "y-axis: horizontally" mirrors left and right (measured with ComfyUI).</summary>
    public LatentBatch Execute(LatentBatch samples, [Input(Options = ["x-axis: vertically", "y-axis: horizontally"])] string flipMethod = "x-axis: vertically")
    {
        bool vertical = flipMethod == "x-axis: vertically";
        if (!vertical && flipMethod != "y-axis: horizontally") return samples; // measured: another value leaves the latent as it is
        int h = samples.Height, w = samples.Width, planes = samples.Batch * samples.Channels;
        var data = new float[samples.Data.Length];
        for (int p = 0; p < planes; p++)
            for (int r = 0; r < h; r++)
                for (int q = 0; q < w; q++)
                    data[(p * h + r) * w + q] = samples.Data[(p * h + (vertical ? h - 1 - r : r)) * w + (vertical ? q : w - 1 - q)];
        return Latents.Like(samples, samples.Batch, samples.Channels, h, w, data);
    }
}

[Node(Category = "model/latent/transform", DisplayName = "Rotate Latent")]
public sealed class LatentRotate
{
    /// <summary>Rotates clockwise by the chosen angle (measured with ComfyUI: 90 degrees turns [[0, 1, 2], [3, 4, 5]] into [[3, 0], [4, 1], [5, 2]]).</summary>
    public LatentBatch Execute(LatentBatch samples, [Input(Options = ["none", "90 degrees", "180 degrees", "270 degrees"])] string rotation = "none")
    {
        int turns = rotation switch { "90 degrees" => 1, "180 degrees" => 2, "270 degrees" => 3, _ => 0 };
        if (turns == 0) return samples; // measured: "none" (or another value) leaves the samples as they are
        int h = samples.Height, w = samples.Width, planes = samples.Batch * samples.Channels;
        int nh = turns == 2 ? h : w, nw = turns == 2 ? w : h;
        var data = new float[samples.Data.Length];
        for (int p = 0; p < planes; p++)
            for (int r = 0; r < nh; r++)
                for (int q = 0; q < nw; q++)
                {
                    // Output (r, q) comes from source (sr, sq).
                    var (sr, sq) = turns switch
                    {
                        1 => (h - 1 - q, r),
                        2 => (h - 1 - r, w - 1 - q),
                        _ => (q, w - 1 - r),
                    };
                    data[(p * nh + r) * nw + q] = samples.Data[(p * h + sr) * w + sq];
                }
        return Latents.Like(samples, samples.Batch, samples.Channels, nh, nw, data);
    }
}

[Node(Category = "model/latent/transform", DisplayName = "Crop Latent")]
public sealed class LatentCrop
{
    /// <summary>
    /// The width x height pixel area at (x, y), all divided by 8 (rounded down). Measured with ComfyUI: a start beyond (size - 8) cells is pulled back
    /// to size - 8, and the rows and columns are then taken as a Python slice <c>[start : start + size]</c> takes them, so a start that became negative
    /// counts from the end (a 4-cell latent cropped at 1 keeps all 4). The noise mask is kept as it is (not cropped).
    /// </summary>
    public LatentBatch Execute(LatentBatch samples, [Input(Min = 64, Max = 16384, Step = 8)] int width = 512, [Input(Min = 64, Max = 16384, Step = 8)] int height = 512,
        [Input(Min = 0, Max = 16384, Step = 8)] int x = 0, [Input(Min = 0, Max = 16384, Step = 8)] int y = 0)
    {
        int h = samples.Height, w = samples.Width;
        int cx = x / 8, cy = y / 8, cw = width / 8, ch = height / 8;
        if (cx > w - 8) cx = w - 8;
        if (cy > h - 8) cy = h - 8;
        var (r0, rows) = Latents.Slice(cy, cy + ch, h);
        var (q0, cols) = Latents.Slice(cx, cx + cw, w);
        int planes = samples.Batch * samples.Channels;
        var data = new float[planes * rows * cols];
        for (int p = 0; p < planes; p++)
            for (int r = 0; r < rows; r++)
                Array.Copy(samples.Data, (p * h + r0 + r) * w + q0, data, (p * rows + r) * cols, cols);
        return Latents.Like(samples, samples.Batch, samples.Channels, rows, cols, data);
    }
}

[Node(Category = "experimental", DisplayName = "Latent Blend")]
public sealed class LatentBlend
{
    /// <summary>
    /// samples1 x blend_factor + samples2 x (1 - blend_factor). Measured with ComfyUI: when the shapes differ at all, samples2 is first brought to samples1's
    /// height and width (cut to its aspect ratio around the centre, bicubic; the same size is left as it is); batch and channels then broadcast as
    /// PyTorch does (one picture is used for all). The arithmetic is single precision.
    /// </summary>
    public LatentBatch Execute(LatentBatch samples1, LatentBatch samples2, [Input(Min = 0, Max = 1, Step = 0.01)] double blendFactor = 0.5)
    {
        var s2 = samples2;
        if (s2.Height != samples1.Height || s2.Width != samples1.Width)
        {
            var (resized, h, w) = TorchInterpolate.CommonUpscale(s2.Data, s2.Batch, s2.Channels, s2.Height, s2.Width, samples1.Width, samples1.Height, "bicubic", "center");
            s2 = new LatentBatch(s2.Batch, s2.Channels, h, w, resized);
        }
        float a = (float)blendFactor, b = (float)(1 - blendFactor);
        var (data, n, c) = Latents.Combine(samples1, s2, (p, q) => p * a + q * b);
        return Latents.Like(samples1, n, c, samples1.Height, samples1.Width, data);
    }
}

[Node(Category = "model/latent", DisplayName = "Set Latent Noise Mask")]
public sealed class SetLatentNoiseMask
{
    /// <summary>
    /// The latent with <paramref name="mask"/> as its noise mask (1 = regenerate, 0 = keep), at the mask's own size and batch (measured with ComfyUI: the mask is
    /// stored as given; the sampler fits it to the latent). The batch index is kept.
    /// </summary>
    public LatentBatch Execute(LatentBatch samples, MaskBatch mask) =>
        new(samples.Batch, samples.Channels, samples.Height, samples.Width, (float[])samples.Data.Clone()) { NoiseMask = mask, BatchIndex = samples.BatchIndex };
}

[Node(Category = "model/latent/advanced", DisplayName = "LatentAdd")]
public sealed class LatentAdd
{
    /// <summary>samples1 + samples2, samples2 first brought to samples1's shape (centre cut and bilinear resize, batch read around or cut).</summary>
    public LatentBatch Execute(LatentBatch samples1, LatentBatch samples2)
    {
        var (data, n, c) = Latents.Combine(samples1, Latents.ShapedLike(samples2, samples1), (a, b) => a + b);
        return Latents.Like(samples1, n, c, samples1.Height, samples1.Width, data);
    }
}

[Node(Category = "model/latent/advanced", DisplayName = "LatentSubtract")]
public sealed class LatentSubtract
{
    /// <summary>samples1 - samples2, samples2 first brought to samples1's shape (centre cut and bilinear resize, batch read around or cut).</summary>
    public LatentBatch Execute(LatentBatch samples1, LatentBatch samples2)
    {
        var (data, n, c) = Latents.Combine(samples1, Latents.ShapedLike(samples2, samples1), (a, b) => a - b);
        return Latents.Like(samples1, n, c, samples1.Height, samples1.Width, data);
    }
}

[Node(Category = "model/latent/advanced", DisplayName = "LatentMultiply")]
public sealed class LatentMultiply
{
    /// <summary>Every value times <paramref name="multiplier"/> (single precision).</summary>
    public LatentBatch Execute(LatentBatch samples, [Input(Min = -10.0, Max = 10.0, Step = 0.01)] double multiplier = 1.0)
    {
        float m = (float)multiplier;
        var data = new float[samples.Data.Length];
        for (int i = 0; i < data.Length; i++) data[i] = samples.Data[i] * m;
        return Latents.Like(samples, samples.Batch, samples.Channels, samples.Height, samples.Width, data);
    }
}

[Node(Category = "model/latent/advanced", DisplayName = "LatentInterpolate")]
public sealed class LatentInterpolate
{
    /// <summary>
    /// Interpolates direction and length separately: at each position the channel vectors are normalised, mixed with <paramref name="ratio"/> (1 = samples1),
    /// normalised again, and given the mixed length. samples2 is first brought to samples1's shape (centre cut and bilinear resize, batch read around or cut).
    /// Measured with ComfyUI: the per-position lengths have the shape (batch, height, width) and are broadcast against (batch, channels, height, width)
    /// as PyTorch broadcasts, so in a batch of several pictures channel k is divided by picture k's length; batch and channel counts that are neither
    /// equal nor 1 fail. After each division by a length, NaN becomes 0 and an infinity the largest finite float (a zero vector mixes as 0).
    /// </summary>
    public LatentBatch Execute(LatentBatch samples1, LatentBatch samples2, [Input(Min = 0.0, Max = 1.0, Step = 0.01)] double ratio = 1.0)
    {
        var s1 = samples1;
        var s2 = Latents.ShapedLike(samples2, samples1);
        int b = s1.Batch, c = s1.Channels, c2 = s2.Channels, plane = s1.Height * s1.Width;
        // The lengths (b, plane) line up with the channel axis: channel k uses length row (b == 1 ? 0 : k).
        int k = Latents.Broadcast(Latents.Broadcast(c, b, 1), Latents.Broadcast(c2, b, 1), 1);
        float r = (float)ratio, q = (float)(1 - ratio);
        var m1 = Norms(s1.Data, b, c, plane);
        var m2 = Norms(s2.Data, b, c2, plane);

        var t = new float[b * k * plane];
        for (int n = 0; n < b; n++)
            for (int ch = 0; ch < k; ch++)
            {
                int i1 = (n * c + (c == 1 ? 0 : ch)) * plane, i2 = (n * c2 + (c2 == 1 ? 0 : ch)) * plane;
                int len = (b == 1 ? 0 : ch) * plane, o = (n * k + ch) * plane;
                for (int p = 0; p < plane; p++)
                {
                    float a = Finite(s1.Data[i1 + p] / m1[len + p]), d = Finite(s2.Data[i2 + p] / m2[len + p]);
                    t[o + p] = a * r + d * q;
                }
            }
        var mt = Norms(t, b, k, plane);
        var result = new float[t.Length];
        for (int n = 0; n < b; n++)
            for (int ch = 0; ch < k; ch++)
            {
                int len = (b == 1 ? 0 : ch) * plane, o = (n * k + ch) * plane;
                for (int p = 0; p < plane; p++)
                {
                    float mixed = m1[len + p] * r + m2[len + p] * q;
                    result[o + p] = Finite(t[o + p] / mt[len + p]) * mixed;
                }
            }
        return Latents.Like(samples1, b, k, s1.Height, s1.Width, result);
    }

    /// <summary>torch.nan_to_num: NaN (0 / 0) becomes 0, an infinity (x / 0) the largest finite float of its sign (measured).</summary>
    static float Finite(float v) => float.IsNaN(v) ? 0f : float.IsPositiveInfinity(v) ? float.MaxValue : float.IsNegativeInfinity(v) ? float.MinValue : v;

    /// <summary>The length of the channel vector at each position, per picture: (b, plane).</summary>
    static float[] Norms(float[] data, int b, int c, int plane)
    {
        var r = new float[b * plane];
        for (int n = 0; n < b; n++)
            for (int p = 0; p < plane; p++)
            {
                float sum = 0;
                for (int ch = 0; ch < c; ch++)
                {
                    float v = data[(n * c + ch) * plane + p];
                    sum += v * v;
                }
                r[n * plane + p] = MathF.Sqrt(sum);
            }
        return r;
    }
}

[Node("LatentBatch", Category = "model/latent/batch", DisplayName = "Batch Latents (DEPRECATED)", Aliases = ["LatentBatchNode"])]
public sealed class LatentBatchNode
{
    /// <summary>
    /// Both latents in one batch, measured to behave as Batch Latents (<see cref="Batching.Latents"/>): samples2 brought to samples1's size, the noise mask
    /// of samples1, the batch indexes of both one after the other.
    /// </summary>
    public LatentBatch Execute(LatentBatch samples1, LatentBatch samples2) => Batching.Latents([samples1, samples2]);
}

[Node(Category = "model/latent/advanced", DisplayName = "LatentBatchSeedBehavior")]
public sealed class LatentBatchSeedBehavior
{
    /// <summary>
    /// "fixed": every picture gets the first picture's noise (batch index = the first entry of the latent's own, or 0, for every picture); "random": the
    /// batch index is removed, so picture i gets noise i. The noise mask is kept (measured with ComfyUI).
    /// </summary>
    public LatentBatch Execute(LatentBatch samples, [Input(Options = ["random", "fixed"])] string seedBehavior = "fixed")
    {
        int[]? index = seedBehavior switch
        {
            "random" => null,
            "fixed" => Enumerable.Repeat(samples.BatchIndex is { Length: > 0 } own ? own[0] : 0, samples.Batch).ToArray(),
            _ => samples.BatchIndex,
        };
        return new LatentBatch(samples.Batch, samples.Channels, samples.Height, samples.Width, (float[])samples.Data.Clone())
        {
            NoiseMask = samples.NoiseMask,
            BatchIndex = index,
        };
    }
}
