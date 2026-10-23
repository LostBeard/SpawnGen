using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

internal static class Latents
{
    public static float[] Frame(LatentBatch l, int index)
    {
        int n = l.Channels * l.Height * l.Width;
        var f = new float[n];
        Array.Copy(l.Data, index * n, f, 0, n);
        return f;
    }

    public static LatentBatch From(IReadOnlyList<float[]> frames, int c, int h, int w, MaskBatch? mask = null, int[]? batchIndex = null)
    {
        var l = new LatentBatch(frames.Count, c, h, w) { NoiseMask = mask, BatchIndex = batchIndex };
        for (int i = 0; i < frames.Count; i++) Array.Copy(frames[i], 0, l.Data, i * c * h * w, c * h * w);
        return l;
    }

    /// <summary>
    /// A new latent that keeps <paramref name="like"/>'s noise mask and batch index: what ComfyUI's <c>samples.copy()</c> followed by a new <c>"samples"</c> does
    /// (the mask is not resized or cut with the samples, exactly as there; the sampler fits it to the latent).
    /// </summary>
    public static LatentBatch Like(LatentBatch like, int b, int c, int h, int w, float[] data) =>
        new(b, c, h, w, data) { NoiseMask = like.NoiseMask, BatchIndex = like.BatchIndex };

    /// <summary><c>comfy.utils.common_upscale</c> of a latent to (width, height) in latent cells; the noise mask and batch index come along unchanged.</summary>
    public static LatentBatch Upscale(LatentBatch s, int width, int height, string method, string crop)
    {
        var (data, h, w) = TorchInterpolate.CommonUpscale(s.Data, s.Batch, s.Channels, s.Height, s.Width, width, height, method, crop);
        return Like(s, s.Batch, s.Channels, h, w, data);
    }

    /// <summary><c>comfy.utils.repeat_to_batch_size</c>: picture i of the result is picture i mod n of the latent (cut short when it has more).</summary>
    public static LatentBatch RepeatToBatch(LatentBatch l, int batch)
    {
        if (l.Batch == batch) return l;
        int per = l.Channels * l.Height * l.Width;
        var data = new float[batch * per];
        for (int i = 0; i < batch; i++) Array.Copy(l.Data, (i % l.Batch) * per, data, i * per, per);
        return Like(l, batch, l.Channels, l.Height, l.Width, data);
    }

    /// <summary>
    /// ComfyUI's <c>reshape_latent_to</c>: a latent whose channels, height or width differ from the target's is resized to the target's size (bilinear,
    /// centre crop), then (unless told not to) repeated or cut to the target's batch size.
    /// </summary>
    public static LatentBatch ReshapeTo(LatentBatch target, LatentBatch latent, bool repeatBatch = true)
    {
        if (latent.Channels != target.Channels || latent.Height != target.Height || latent.Width != target.Width)
            latent = Upscale(latent, target.Width, target.Height, "bilinear", "center");
        return repeatBatch ? RepeatToBatch(latent, target.Batch) : latent;
    }

    /// <summary>Python's <c>a[start:stop]</c> along an axis of <paramref name="length"/>: negative indices count from the end and both ends are clamped.</summary>
    public static (int Start, int Count) PySlice(int length, int start, int stop)
    {
        static int Norm(int i, int len) => i < 0 ? Math.Max(0, i + len) : Math.Min(i, len);
        int s = Norm(start, length), e = Norm(stop, length);
        return (s, Math.Max(0, e - s));
    }

    /// <summary>Python's <c>a // b</c> (rounds towards minus infinity).</summary>
    public static int FloorDiv(int a, int b) => (int)Math.Floor((double)a / b);

    /// <summary>PyTorch broadcasting of one dimension: equal sizes, or a size 1 stretched to the other; anything else is the error PyTorch raises.</summary>
    public static int Broadcast(int a, int b, int dim) =>
        a == b || b == 1 ? a : a == 1 ? b
            : throw new ArgumentException($"The size of tensor a ({a}) must match the size of tensor b ({b}) at non-singleton dimension {dim}.");

    /// <summary>An element-wise operation with PyTorch broadcasting over (batch, channels, height, width).</summary>
    public static (int B, int C, int H, int W, float[] Data) Elementwise(LatentBatch a, LatentBatch b, Func<float, float, float> f)
    {
        int nb = Broadcast(a.Batch, b.Batch, 0), nc = Broadcast(a.Channels, b.Channels, 1), nh = Broadcast(a.Height, b.Height, 2), nw = Broadcast(a.Width, b.Width, 3);
        static int At(LatentBatch t, int n, int c, int y, int x) =>
            (((t.Batch == 1 ? 0 : n) * t.Channels + (t.Channels == 1 ? 0 : c)) * t.Height + (t.Height == 1 ? 0 : y)) * t.Width + (t.Width == 1 ? 0 : x);
        var r = new float[nb * nc * nh * nw];
        int i = 0;
        for (int n = 0; n < nb; n++)
            for (int c = 0; c < nc; c++)
                for (int y = 0; y < nh; y++)
                    for (int x = 0; x < nw; x++)
                        r[i++] = f(a.Data[At(a, n, c, y, x)], b.Data[At(b, n, c, y, x)]);
        return (nb, nc, nh, nw, r);
    }

    /// <summary><c>torch.nan_to_num</c>: NaN to 0, infinities to the largest finite float.</summary>
    public static float NanToNum(float v) => float.IsNaN(v) ? 0f : float.IsPositiveInfinity(v) ? float.MaxValue : float.IsNegativeInfinity(v) ? float.MinValue : v;

    /// <summary>A mask batch's picture <paramref name="index"/> as its own array.</summary>
    public static float[] MaskFrame(MaskBatch m, int index)
    {
        var one = new float[m.Height * m.Width];
        Array.Copy(m.Data, index * one.Length, one, 0, one.Length);
        return one;
    }
}

/// <summary>Latent sizes are given in pixels in the node widgets and are 1/8 of that in the latent, like ComfyUI.</summary>
[Node(Category = "model/latent", DisplayName = "Upscale Latent")]
public sealed class LatentUpscale
{
    public static string[] Methods() => TorchInterpolate.LatentMethods;
    public static string[] Crops() => ["disabled", "center"];

    /// <summary>
    /// Resizes to (width, height) pixels, at least 64 each; a 0 side follows the aspect ratio and both 0 leave the latent alone. "center" crops the middle part
    /// with the new aspect ratio before resizing. A noise mask is kept as it is (the sampler fits it to the new size), as in ComfyUI.
    /// </summary>
    public LatentBatch Execute(LatentBatch samples, [Input(OptionsProvider = nameof(Methods))] string upscaleMethod = "nearest-exact",
        [Input(Min = 0, Max = 16384, Step = 8)] int width = 512, [Input(Min = 0, Max = 16384, Step = 8)] int height = 512,
        [Input(OptionsProvider = nameof(Crops))] string crop = "disabled")
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
        return Latents.Upscale(samples, width / 8, height / 8, upscaleMethod, crop);
    }
}

[Node(Category = "model/latent", DisplayName = "Upscale Latent By")]
public sealed class LatentUpscaleBy
{
    public static string[] Methods() => TorchInterpolate.LatentMethods;

    /// <summary>Scales the latent's own size by <c>scale_by</c> (Python rounding, no minimum), without cropping.</summary>
    public LatentBatch Execute(LatentBatch samples, [Input(OptionsProvider = nameof(Methods))] string upscaleMethod = "nearest-exact",
        [Input(Min = 0.01, Max = 8, Step = 0.01)] double scaleBy = 1.5) =>
        Latents.Upscale(samples, (int)Math.Round(samples.Width * scaleBy, MidpointRounding.ToEven), (int)Math.Round(samples.Height * scaleBy, MidpointRounding.ToEven),
            upscaleMethod, "disabled");
}

[Node(Category = "model/latent/batch", DisplayName = "Get Latent From Batch")]
public sealed class LatentFromBatch
{
    /// <summary>
    /// <c>length</c> pictures from <c>batch_index</c> on (negative counts from the end; both are clamped to the batch). The pictures keep their place in the
    /// batch as their batch index, so a sampler gives them the same noise they would have had in the whole batch.
    /// </summary>
    public LatentBatch Execute(LatentBatch samples, [Input(Min = -16384, Max = 16384)] int batchIndex = 0, [Input(Min = 1, Max = 64)] int length = 1)
    {
        int count = samples.Batch;
        if (batchIndex < 0) batchIndex += count;
        batchIndex = Math.Max(0, Math.Min(count - 1, batchIndex));
        length = Math.Min(count - batchIndex, length);

        MaskBatch? mask = samples.NoiseMask;
        if (mask is { Batch: > 1 } m)
        {
            // A mask batch shorter than the latent's is repeated to its length first, then the same pictures are taken.
            int available = Math.Max(m.Batch, count);
            var (start, take) = Latents.PySlice(available, batchIndex, batchIndex + length);
            mask = new MaskBatch(take, m.Height, m.Width,
                Enumerable.Range(start, take).SelectMany(i => Latents.MaskFrame(m, i % m.Batch)).ToArray());
        }
        int[] batch;
        if (samples.BatchIndex is { } existing)
        {
            var (start, take) = Latents.PySlice(existing.Length, batchIndex, batchIndex + length);
            batch = existing.Skip(start).Take(take).ToArray();
        }
        else batch = Enumerable.Range(batchIndex, length).ToArray();
        return Latents.From(Enumerable.Range(batchIndex, length).Select(i => Latents.Frame(samples, i)).ToList(), samples.Channels, samples.Height, samples.Width, mask, batch);
    }
}

[Node(Category = "model/latent/batch", DisplayName = "Repeat Latent Batch")]
public sealed class RepeatLatentBatch
{
    /// <summary>The whole batch <c>amount</c> times over. A batch index is continued past its range for every copy, so each copy gets new noise.</summary>
    public LatentBatch Execute(LatentBatch samples, [Input(Min = 1, Max = 64)] int amount = 1)
    {
        var mask = samples.NoiseMask;
        if (mask is { Batch: > 1 } m) mask = new MaskBatch(m.Batch * amount, m.Height, m.Width, Enumerable.Range(0, amount).SelectMany(_ => m.Data).ToArray());
        int[]? batch = samples.BatchIndex;
        if (batch is { Length: > 0 } bi)
        {
            int offset = bi.Max() - bi.Min() + 1;
            batch = [.. bi, .. Enumerable.Range(1, Math.Max(0, amount - 1)).SelectMany(i => bi.Select(x => x + i * offset))];
        }
        return Latents.From(Enumerable.Range(0, amount).SelectMany(_ => Enumerable.Range(0, samples.Batch).Select(i => Latents.Frame(samples, i))).ToList(),
            samples.Channels, samples.Height, samples.Width, mask, batch);
    }
}

[Node(Category = "model/latent", DisplayName = "Latent Composite")]
public sealed class LatentComposite
{
    /// <summary>
    /// Pastes <c>samples_from</c> into <c>samples_to</c> at (x, y) pixels (cut to what fits). With <c>feather</c> the pasted part fades in over that many pixels
    /// from each of its edges that is not on the border of the target.
    /// </summary>
    public LatentBatch Execute(LatentBatch samplesTo, LatentBatch samplesFrom, [Input(Min = 0, Max = 16384, Step = 8)] int x = 0,
        [Input(Min = 0, Max = 16384, Step = 8)] int y = 0, [Input(Min = 0, Max = 16384, Step = 8)] int feather = 0)
    {
        x = Latents.FloorDiv(x, 8); y = Latents.FloorDiv(y, 8); feather = Latents.FloorDiv(feather, 8);
        int b = samplesTo.Batch, c = samplesTo.Channels, h = samplesTo.Height, w = samplesTo.Width;
        var result = Latents.Like(samplesTo, b, c, h, w, (float[])samplesTo.Data.Clone());

        // samples_from[:, :, :H - y, :W - x], pasted at s[:, :, y:y + rows, x:x + cols].
        var (_, fh) = Latents.PySlice(samplesFrom.Height, 0, h - y);
        var (_, fw) = Latents.PySlice(samplesFrom.Width, 0, w - x);
        var (ty, th) = Latents.PySlice(h, y, y + (feather == 0 ? samplesFrom.Height : fh));
        var (tx, tw) = Latents.PySlice(w, x, x + (feather == 0 ? samplesFrom.Width : fw));
        if (th != fh || tw != fw)
            throw new ArgumentException($"Latent Composite: the part of samples_from that fits ({fw}x{fh}) does not match the place it goes ({tw}x{th}).");
        // The pasted values broadcast into the target: a batch or channel count of 1 is stretched, any other must match.
        if (samplesFrom.Batch != b && samplesFrom.Batch != 1) throw new ArgumentException($"Latent Composite: samples_from has {samplesFrom.Batch} pictures and samples_to {b}.");
        if (samplesFrom.Channels != c && samplesFrom.Channels != 1) throw new ArgumentException($"Latent Composite: samples_from has {samplesFrom.Channels} channels and samples_to {c}.");
        if (fh == 0 || fw == 0) return result;

        float[]? mask = null;
        if (feather != 0)
        {
            mask = new float[fh * fw];
            Array.Fill(mask, 1f);
            void Rows(int start, int stop, float f)
            {
                var (s, n) = Latents.PySlice(fh, start, stop);
                for (int r = s; r < s + n; r++) for (int col = 0; col < fw; col++) mask[r * fw + col] *= f;
            }
            void Cols(int start, int stop, float f)
            {
                var (s, n) = Latents.PySlice(fw, start, stop);
                for (int col = s; col < s + n; col++) for (int r = 0; r < fh; r++) mask[r * fw + col] *= f;
            }
            for (int t = 0; t < feather; t++)
            {
                float f = (float)(1.0 / feather * (t + 1));
                if (y != 0) Rows(t, 1 + t, f);
                if (y + fh < h) Rows(fh - 1 - t, fh - t, f);
                if (x != 0) Cols(t, 1 + t, f);
                if (x + fw < w) Cols(fw - 1 - t, fw - t, f);
            }
        }
        for (int n = 0; n < b; n++)
        {
            int fb = samplesFrom.Batch == 1 ? 0 : n;
            for (int ch = 0; ch < c; ch++)
            {
                int fc = samplesFrom.Channels == 1 ? 0 : ch;
                for (int r = 0; r < fh; r++)
                    for (int col = 0; col < fw; col++)
                    {
                        float s = samplesFrom.Data[((fb * samplesFrom.Channels + fc) * samplesFrom.Height + r) * samplesFrom.Width + col];
                        int ti = ((n * c + ch) * h + ty + r) * w + tx + col;
                        if (mask is null) result.Data[ti] = s;
                        else
                        {
                            float m = mask[r * fw + col];
                            result.Data[ti] = s * m + result.Data[ti] * (1f - m);
                        }
                    }
            }
        }
        return result;
    }
}

[Node(Category = "model/latent/transform", DisplayName = "Flip Latent")]
public sealed class LatentFlip
{
    public static string[] Methods() => ["x-axis: vertically", "y-axis: horizontally"];

    public LatentBatch Execute(LatentBatch samples, [Input(OptionsProvider = nameof(Methods))] string flipMethod = "x-axis: vertically")
    {
        bool vertical = flipMethod.StartsWith('x');
        if (!vertical && !flipMethod.StartsWith('y')) return samples;
        var r = Latents.Like(samples, samples.Batch, samples.Channels, samples.Height, samples.Width, new float[samples.Data.Length]);
        int w = samples.Width, h = samples.Height;
        for (int plane = 0; plane < samples.Batch * samples.Channels; plane++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    r.Data[(plane * h + y) * w + x] = samples.Data[(plane * h + (vertical ? h - 1 - y : y)) * w + (vertical ? x : w - 1 - x)];
        return r;
    }
}

[Node(Category = "model/latent/transform", DisplayName = "Rotate Latent")]
public sealed class LatentRotate
{
    public static string[] Rotations() => ["none", "90 degrees", "180 degrees", "270 degrees"];

    /// <summary>Rotates clockwise (<c>torch.rot90(dims=[3, 2])</c>). 90 and 270 swap width and height; the noise mask is left as it is, as in ComfyUI.</summary>
    public LatentBatch Execute(LatentBatch samples, [Input(OptionsProvider = nameof(Rotations))] string rotation = "none")
    {
        int quarter = rotation.StartsWith("90") ? 1 : rotation.StartsWith("180") ? 2 : rotation.StartsWith("270") ? 3 : 0;
        if (quarter == 0) return samples;
        int h = samples.Height, w = samples.Width;
        int nh = quarter % 2 == 1 ? w : h, nw = quarter % 2 == 1 ? h : w;
        var r = Latents.Like(samples, samples.Batch, samples.Channels, nh, nw, new float[samples.Data.Length]);
        for (int plane = 0; plane < samples.Batch * samples.Channels; plane++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    // Clockwise: (x, y) -> (h-1-y, x) for 90, (w-1-x, h-1-y) for 180, (y, w-1-x) for 270.
                    var (ny, nx) = quarter switch { 1 => (x, h - 1 - y), 2 => (h - 1 - y, w - 1 - x), _ => (w - 1 - x, y) };
                    r.Data[(plane * nh + ny) * nw + nx] = samples.Data[(plane * h + y) * w + x];
                }
        return r;
    }
}

[Node(Category = "model/latent/transform", DisplayName = "Crop Latent")]
public sealed class LatentCrop
{
    /// <summary>The (width, height) pixel part at (x, y). The start is pulled back so at least 8 latent cells (64 pixels) remain; the size is cut to what is there.</summary>
    public LatentBatch Execute(LatentBatch samples, [Input(Min = 64, Max = 16384, Step = 8)] int width = 512, [Input(Min = 64, Max = 16384, Step = 8)] int height = 512,
        [Input(Min = 0, Max = 16384, Step = 8)] int x = 0, [Input(Min = 0, Max = 16384, Step = 8)] int y = 0)
    {
        x = Latents.FloorDiv(x, 8); y = Latents.FloorDiv(y, 8);
        if (x > samples.Width - 8) x = samples.Width - 8;
        if (y > samples.Height - 8) y = samples.Height - 8;
        var (sx, cw) = Latents.PySlice(samples.Width, x, x + Latents.FloorDiv(width, 8));
        var (sy, ch) = Latents.PySlice(samples.Height, y, y + Latents.FloorDiv(height, 8));
        var r = Latents.Like(samples, samples.Batch, samples.Channels, ch, cw, new float[samples.Batch * samples.Channels * ch * cw]);
        for (int plane = 0; plane < samples.Batch * samples.Channels; plane++)
            for (int row = 0; row < ch; row++)
                Array.Copy(samples.Data, (plane * samples.Height + sy + row) * samples.Width + sx, r.Data, (plane * ch + row) * cw, cw);
        return r;
    }
}

[Node(Category = "experimental", DisplayName = "Latent Blend")]
public sealed class LatentBlend
{
    /// <summary><c>samples1 * blend_factor + samples2 * (1 - blend_factor)</c>; a second latent of another shape is first resized to the first's size (bicubic, centre crop).</summary>
    public LatentBatch Execute(LatentBatch samples1, LatentBatch samples2, [Input(Min = 0, Max = 1, Step = 0.01)] double blendFactor = 0.5)
    {
        if (samples1.Batch != samples2.Batch || samples1.Channels != samples2.Channels || samples1.Height != samples2.Height || samples1.Width != samples2.Width)
            samples2 = Latents.Upscale(samples2, samples1.Width, samples1.Height, "bicubic", "center");
        float f1 = (float)blendFactor, f2 = (float)(1 - blendFactor);
        var (b, c, h, w, data) = Latents.Elementwise(samples1, samples2, (a, s) => a * f1 + s * f2);
        return Latents.Like(samples1, b, c, h, w, data);
    }
}

[Node(Category = "model/latent", DisplayName = "Set Latent Noise Mask")]
public sealed class SetLatentNoiseMask
{
    /// <summary>Marks the part of the latent that sampling may change (mask = 1); the rest is held at the original. The mask may be any size: the sampler resizes it to the latent.</summary>
    public LatentBatch Execute(LatentBatch samples, MaskBatch mask) =>
        new(samples.Batch, samples.Channels, samples.Height, samples.Width, (float[])samples.Data.Clone()) { NoiseMask = mask, BatchIndex = samples.BatchIndex };
}

[Node(Category = "model/latent/advanced", DisplayName = "LatentAdd")]
public sealed class LatentAdd
{
    /// <summary>The second latent is brought to the first one's shape (bilinear resize with centre crop, then repeated or cut to its batch size) and added.</summary>
    public LatentBatch Execute(LatentBatch samples1, LatentBatch samples2)
    {
        var (b, c, h, w, data) = Latents.Elementwise(samples1, Latents.ReshapeTo(samples1, samples2), (x, y) => x + y);
        return Latents.Like(samples1, b, c, h, w, data);
    }
}

[Node(Category = "model/latent/advanced", DisplayName = "LatentSubtract")]
public sealed class LatentSubtract
{
    public LatentBatch Execute(LatentBatch samples1, LatentBatch samples2)
    {
        var (b, c, h, w, data) = Latents.Elementwise(samples1, Latents.ReshapeTo(samples1, samples2), (x, y) => x - y);
        return Latents.Like(samples1, b, c, h, w, data);
    }
}

[Node(Category = "model/latent/advanced", DisplayName = "LatentMultiply")]
public sealed class LatentMultiply
{
    public LatentBatch Execute(LatentBatch samples, [Input(Min = -10, Max = 10, Step = 0.01)] double multiplier = 1.0)
    {
        float m = (float)multiplier;
        var r = Latents.Like(samples, samples.Batch, samples.Channels, samples.Height, samples.Width, new float[samples.Data.Length]);
        for (int i = 0; i < r.Data.Length; i++) r.Data[i] = samples.Data[i] * m;
        return r;
    }
}

[Node(Category = "model/latent/advanced", DisplayName = "LatentInterpolate")]
public sealed class LatentInterpolate
{
    /// <summary>
    /// Mixes two latents by <c>ratio</c> (1 = all of the first) while keeping the strength: at every position the direction over the channels is interpolated and the length
    /// (the norm over the channels) is interpolated separately, so a half-way mix is not washed out like a plain average.
    /// ComfyUI divides by the per-position norms without keeping their dimension, so PyTorch broadcasting lines them up with the channels: with a batch of one this
    /// is the norm of the same picture, with a batch as large as the channel count picture <i>c</i>'s norm scales channel <i>c</i>, and any other batch size is an
    /// error there. All three are reproduced.
    /// </summary>
    public LatentBatch Execute(LatentBatch samples1, LatentBatch samples2, [Input(Min = 0, Max = 1, Step = 0.01)] double ratio = 1.0)
    {
        var s2 = Latents.ReshapeTo(samples1, samples2);
        int nb = samples1.Batch, c = samples1.Channels, h = samples1.Height, w = samples1.Width, plane = h * w, per = c * plane;
        if (s2.Channels != c) throw new ArgumentException($"The latents have {c} and {s2.Channels} channels and cannot be interpolated.");
        if (nb != 1 && nb != c)
            throw new ArgumentException($"The size of tensor a ({c}) must match the size of tensor b ({nb}) at non-singleton dimension 1 (ComfyUI's LatentInterpolate only handles a batch of 1 or of {c}).");

        static float[] Norms(float[] data, int nb, int c, int plane)
        {
            var m = new float[nb * plane];
            for (int n = 0; n < nb; n++)
                for (int p = 0; p < plane; p++)
                {
                    double sum = 0;
                    for (int ch = 0; ch < c; ch++) { double v = data[(n * c + ch) * plane + p]; sum += v * v; }
                    m[n * plane + p] = (float)Math.Sqrt(sum);
                }
            return m;
        }
        int K(int n, int ch) => nb == 1 ? 0 : ch; // which norm divides element (n, ch): the broadcast described above

        var m1 = Norms(samples1.Data, nb, c, plane);
        var m2 = Norms(s2.Data, nb, c, plane);
        float r1 = (float)ratio, r2 = (float)(1.0 - ratio);
        var t = new float[samples1.Data.Length];
        for (int n = 0; n < nb; n++)
            for (int ch = 0; ch < c; ch++)
                for (int p = 0; p < plane; p++)
                {
                    int i = n * per + ch * plane + p, k = K(n, ch) * plane + p;
                    t[i] = Latents.NanToNum(samples1.Data[i] / m1[k]) * r1 + Latents.NanToNum(s2.Data[i] / m2[k]) * r2;
                }
        var mt = Norms(t, nb, c, plane);
        var result = Latents.Like(samples1, nb, c, h, w, new float[t.Length]);
        for (int n = 0; n < nb; n++)
            for (int ch = 0; ch < c; ch++)
                for (int p = 0; p < plane; p++)
                {
                    int i = n * per + ch * plane + p, k = K(n, ch) * plane + p;
                    result.Data[i] = Latents.NanToNum(t[i] / mt[k]) * (m1[k] * r1 + m2[k] * r2);
                }
        return result;
    }
}

[Node("LatentBatch", Category = "model/latent/batch", DisplayName = "Batch Latents (DEPRECATED)", Aliases = ["LatentBatchNode"])]
public sealed class LatentBatchNode
{
    /// <summary>
    /// Stacks two latents into one batch; a second latent of another size is resized (bilinear, centre crop) to the first's. The batch indexes are joined, so
    /// the second latent's pictures keep their own noise (index 0 on); the first latent's noise mask is kept.
    /// </summary>
    public LatentBatch Execute(LatentBatch samples1, LatentBatch samples2)
    {
        var second = Latents.ReshapeTo(samples1, samples2, repeatBatch: false);
        if (second.Channels != samples1.Channels) throw new ArgumentException($"The latents have {samples1.Channels} and {second.Channels} channels and cannot be batched.");
        var frames = Enumerable.Range(0, samples1.Batch).Select(i => Latents.Frame(samples1, i))
            .Concat(Enumerable.Range(0, second.Batch).Select(i => Latents.Frame(second, i))).ToList();
        int[] batch = [.. samples1.BatchIndex ?? Enumerable.Range(0, samples1.Batch).ToArray(), .. samples2.BatchIndex ?? Enumerable.Range(0, samples2.Batch).ToArray()];
        return Latents.From(frames, samples1.Channels, samples1.Height, samples1.Width, samples1.NoiseMask, batch);
    }
}

[Node(Category = "model/latent/advanced", DisplayName = "LatentBatchSeedBehavior")]
public sealed class LatentBatchSeedBehavior
{
    public static string[] Behaviors() => ["random", "fixed"];

    /// <summary>"fixed" gives every picture the noise of the first one (so the batch varies only by prompt or settings); "random" gives each its own.</summary>
    public LatentBatch Execute(LatentBatch samples, [Input(OptionsProvider = nameof(Behaviors))] string seedBehavior = "fixed") => seedBehavior switch
    {
        "random" => new(samples.Batch, samples.Channels, samples.Height, samples.Width, samples.Data) { NoiseMask = samples.NoiseMask },
        "fixed" => new(samples.Batch, samples.Channels, samples.Height, samples.Width, samples.Data)
        {
            NoiseMask = samples.NoiseMask, BatchIndex = Enumerable.Repeat(samples.BatchIndex is { Length: > 0 } bi ? bi[0] : 0, samples.Batch).ToArray(),
        },
        _ => samples,
    };
}

[Node(Category = "model/latent", DisplayName = "Latent Composite Masked")]
public sealed class LatentCompositeMasked
{
    /// <summary>
    /// Pastes <c>source</c> into <c>destination</c> at (x, y) pixels through an optional mask (resized bilinearly to the source); with <c>resize_source</c> the source is
    /// first stretched to the destination's size. A source batch or mask batch that is shorter than the destination's is repeated.
    /// </summary>
    public LatentBatch Execute(LatentBatch destination, LatentBatch source, [Input(Min = 0, Max = 16384, Step = 8)] int x = 0, [Input(Min = 0, Max = 16384, Step = 8)] int y = 0,
        bool resizeSource = false, MaskBatch? mask = null)
    {
        int db = destination.Batch, c = destination.Channels, dh = destination.Height, dw = destination.Width;
        if (resizeSource)
            source = Latents.Like(source, source.Batch, source.Channels, dh, dw,
                TorchInterpolate.Resize(source.Data, source.Batch * source.Channels, source.Height, source.Width, dh, dw, "bilinear"));
        source = Latents.RepeatToBatch(source, db);
        if (source.Channels != c && source.Channels != 1) throw new ArgumentException($"The latents have {c} and {source.Channels} channels and cannot be composited.");
        int sh = source.Height, sw = source.Width;

        x = Math.Max(-sw * 8, Math.Min(x, dw * 8));
        y = Math.Max(-sh * 8, Math.Min(y, dh * 8));
        int left = Latents.FloorDiv(x, 8), top = Latents.FloorDiv(y, 8), right = left + sw, bottom = top + sh;

        float[]? m = null; // (batch, sh, sw), already repeated to the destination's batch
        if (mask is not null)
        {
            var resized = TorchInterpolate.Resize(mask.Data, mask.Batch, mask.Height, mask.Width, sh, sw, "bilinear");
            m = new float[db * sh * sw];
            for (int b = 0; b < db; b++) Array.Copy(resized, (b % mask.Batch) * sh * sw, m, b * sh * sw, sh * sw);
        }

        // Only the part of the source that lands on the destination is used.
        int visibleW = dw - left + Math.Min(0, x), visibleH = dh - top + Math.Min(0, y);
        var (_, vh) = Latents.PySlice(sh, 0, visibleH);
        var (_, vw) = Latents.PySlice(sw, 0, visibleW);
        var (ty, th) = Latents.PySlice(dh, top, bottom);
        var (tx, tw) = Latents.PySlice(dw, left, right);
        if (th != vh || tw != vw)
            throw new ArgumentException($"Latent Composite Masked: the visible part of the source ({vw}x{vh}) does not match the place it goes ({tw}x{th}).");

        var result = Latents.Like(destination, db, c, dh, dw, (float[])destination.Data.Clone());
        for (int b = 0; b < db; b++)
            for (int ch = 0; ch < c; ch++)
            {
                int sc = source.Channels == 1 ? 0 : ch;
                for (int yy = 0; yy < vh; yy++)
                    for (int xx = 0; xx < vw; xx++)
                    {
                        float wgt = m is null ? 1f : m[(b * sh + yy) * sw + xx];
                        int di = ((b * c + ch) * dh + ty + yy) * dw + tx + xx;
                        result.Data[di] = wgt * source.Data[((b * source.Channels + sc) * sh + yy) * sw + xx] + (1f - wgt) * result.Data[di];
                    }
            }
        return result;
    }
}
