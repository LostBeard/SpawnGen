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

    public static LatentBatch From(IReadOnlyList<float[]> frames, int c, int h, int w, MaskBatch? mask = null)
    {
        var l = new LatentBatch(frames.Count, c, h, w) { NoiseMask = mask };
        for (int i = 0; i < frames.Count; i++) Array.Copy(frames[i], 0, l.Data, i * c * h * w, c * h * w);
        return l;
    }

    /// <summary>NCHW planes to interleaved (HWC) and back, so the image resampler (which works on interleaved channels) can be reused.</summary>
    public static float[] ToHwc(float[] chw, int c, int h, int w)
    {
        var r = new float[chw.Length];
        for (int ch = 0; ch < c; ch++) for (int i = 0; i < h * w; i++) r[i * c + ch] = chw[ch * h * w + i];
        return r;
    }

    public static float[] ToChw(float[] hwc, int c, int h, int w)
    {
        var r = new float[hwc.Length];
        for (int ch = 0; ch < c; ch++) for (int i = 0; i < h * w; i++) r[ch * h * w + i] = hwc[i * c + ch];
        return r;
    }

    public static float[] Resize(float[] chw, int c, int h, int w, int nh, int nw, ResampleMethod m) =>
        ToChw(Resample.Resize(ToHwc(chw, c, h, w), w, h, c, nw, nh, m), c, nh, nw);

    /// <summary>Applies a function to the mask the same way the latent was changed, so an inpainting mask keeps lining up.</summary>
    public static MaskBatch? Map(MaskBatch? mask, Func<MaskBatch, MaskBatch> op) => mask is null ? null : op(mask);
}

/// <summary>Latent sizes are given in pixels in the node widgets and are 1/8 of that in the latent, like ComfyUI.</summary>
[Node(Category = "latent", DisplayName = "Upscale Latent")]
public sealed class LatentUpscale
{
    public static string[] Methods() => Resample.MethodNames;
    public static string[] Crops() => ["disabled", "center"];

    public LatentBatch Execute(LatentBatch samples, [Input(OptionsProvider = nameof(Methods))] string upscaleMethod = "nearest-exact",
        [Input(Min = 0, Max = 16384, Step = 8)] int width = 512, [Input(Min = 0, Max = 16384, Step = 8)] int height = 512,
        [Input(OptionsProvider = nameof(Crops))] string crop = "disabled")
    {
        var method = Resample.Parse(upscaleMethod);
        var (pw, ph) = Pixels.TargetSize(samples.Width * 8, samples.Height * 8, width, height);
        int nw = Math.Max(1, pw / 8), nh = Math.Max(1, ph / 8);
        if (nw == samples.Width && nh == samples.Height) return samples;

        int sw = nw, sh = nh, x0 = 0, y0 = 0;
        if (crop == "center")
        {
            double scale = Math.Max((double)nw / samples.Width, (double)nh / samples.Height);
            sw = Math.Max(nw, (int)Math.Round(samples.Width * scale));
            sh = Math.Max(nh, (int)Math.Round(samples.Height * scale));
            x0 = (sw - nw) / 2; y0 = (sh - nh) / 2;
        }
        float[] Cut(float[] chw, int c, int w, int h)
        {
            if (sw == nw && sh == nh) return chw;
            var cut = new float[c * nh * nw];
            for (int ch = 0; ch < c; ch++) for (int y = 0; y < nh; y++) Array.Copy(chw, ch * h * w + (y0 + y) * w + x0, cut, ch * nh * nw + y * nw, nw);
            return cut;
        }
        var frames = Enumerable.Range(0, samples.Batch)
            .Select(i => Cut(Latents.Resize(Latents.Frame(samples, i), samples.Channels, samples.Height, samples.Width, sh, sw, method), samples.Channels, sw, sh)).ToList();
        var mask = Latents.Map(samples.NoiseMask, m => new MaskBatch(m.Batch, nh, nw,
            Enumerable.Range(0, m.Batch).SelectMany(b =>
            {
                var one = new float[m.Height * m.Width];
                Array.Copy(m.Data, b * one.Length, one, 0, one.Length);
                var big = Resample.Resize(one, m.Width, m.Height, 1, sw, sh, ResampleMethod.Bilinear);
                if (sw == nw && sh == nh) return big;
                var cut = new float[nh * nw];
                for (int y = 0; y < nh; y++) Array.Copy(big, (y0 + y) * sw + x0, cut, y * nw, nw);
                return cut;
            }).ToArray()));
        return Latents.From(frames, samples.Channels, nh, nw, mask);
    }
}

[Node(Category = "latent", DisplayName = "Upscale Latent By")]
public sealed class LatentUpscaleBy
{
    public static string[] Methods() => Resample.MethodNames;

    public LatentBatch Execute(LatentBatch samples, [Input(OptionsProvider = "Methods")] string upscaleMethod = "nearest-exact",
        [Input(Min = 0.01, Max = 8, Step = 0.01)] float scaleBy = 1.5f) =>
        new LatentUpscale().Execute(samples, upscaleMethod, Math.Max(1, (int)Math.Round(samples.Width * scaleBy)) * 8, Math.Max(1, (int)Math.Round(samples.Height * scaleBy)) * 8);
}

[Node(Category = "latent/batch", DisplayName = "Latent From Batch")]
public sealed class LatentFromBatch
{
    public LatentBatch Execute(LatentBatch samples, [Input(Min = 0, Max = 63)] int batchIndex = 0, [Input(Min = 1, Max = 64)] int length = 1)
    {
        int start = Math.Min(batchIndex, samples.Batch - 1), count = Math.Min(length, samples.Batch - start);
        var mask = Latents.Map(samples.NoiseMask, m => new MaskBatch(count, m.Height, m.Width,
            m.Data.Skip(Math.Min(start, m.Batch - 1) * m.Height * m.Width).Take(count * m.Height * m.Width).ToArray()));
        return Latents.From(Enumerable.Range(start, count).Select(i => Latents.Frame(samples, i)).ToList(), samples.Channels, samples.Height, samples.Width, mask);
    }
}

[Node(Category = "latent/batch", DisplayName = "Repeat Latent Batch")]
public sealed class RepeatLatentBatch
{
    public LatentBatch Execute(LatentBatch samples, [Input(Min = 1, Max = 64)] int amount = 1) =>
        Latents.From(Enumerable.Range(0, amount).SelectMany(_ => Enumerable.Range(0, samples.Batch).Select(i => Latents.Frame(samples, i))).ToList(),
            samples.Channels, samples.Height, samples.Width,
            Latents.Map(samples.NoiseMask, m => new MaskBatch(m.Batch * amount, m.Height, m.Width, Enumerable.Range(0, amount).SelectMany(_ => m.Data).ToArray())));
}

[Node(Category = "latent", DisplayName = "LatentComposite")]
public sealed class LatentComposite
{
    /// <summary>Pastes <c>samples_from</c> into <c>samples_to</c> at (x, y) pixels, fading the edges over <c>feather</c> pixels (only on edges that are inside the target).</summary>
    public LatentBatch Execute(LatentBatch samplesTo, LatentBatch samplesFrom, [Input(Min = 0, Max = 16384, Step = 8)] int x = 0,
        [Input(Min = 0, Max = 16384, Step = 8)] int y = 0, [Input(Min = 0, Max = 16384, Step = 8)] int feather = 0)
    {
        int lx = x / 8, ly = y / 8, f = feather / 8;
        var r = new LatentBatch(samplesTo.Batch, samplesTo.Channels, samplesTo.Height, samplesTo.Width) { NoiseMask = samplesTo.NoiseMask };
        Array.Copy(samplesTo.Data, r.Data, samplesTo.Data.Length);
        int h = Math.Min(samplesFrom.Height, samplesTo.Height - ly), w = Math.Min(samplesFrom.Width, samplesTo.Width - lx);
        if (h <= 0 || w <= 0) return r;
        for (int b = 0; b < samplesTo.Batch; b++)
        {
            int fb = b % samplesFrom.Batch;
            for (int c = 0; c < samplesTo.Channels; c++)
                for (int yy = 0; yy < h; yy++)
                    for (int xx = 0; xx < w; xx++)
                    {
                        float m = 1f;
                        if (f > 0)
                        {
                            // Ramp up from the edges that are not on the border of the target.
                            if (ly > 0) m *= Math.Min(1f, (yy + 1f) / f);
                            if (lx > 0) m *= Math.Min(1f, (xx + 1f) / f);
                            if (ly + h < samplesTo.Height) m *= Math.Min(1f, (h - yy) / (float)f);
                            if (lx + w < samplesTo.Width) m *= Math.Min(1f, (w - xx) / (float)f);
                        }
                        int ti = ((b * samplesTo.Channels + c) * samplesTo.Height + ly + yy) * samplesTo.Width + lx + xx;
                        float s = samplesFrom.Data[((fb * samplesFrom.Channels + c) * samplesFrom.Height + yy) * samplesFrom.Width + xx];
                        r.Data[ti] = s * m + r.Data[ti] * (1 - m);
                    }
        }
        return r;
    }
}

[Node(Category = "latent/transform", DisplayName = "Flip Latent")]
public sealed class LatentFlip
{
    public static string[] Methods() => ["x-axis: vertically", "y-axis: horizontally"];

    public LatentBatch Execute(LatentBatch samples, [Input(OptionsProvider = nameof(Methods))] string flipMethod = "x-axis: vertically")
    {
        bool vertical = flipMethod.StartsWith("x-axis");
        var r = new LatentBatch(samples.Batch, samples.Channels, samples.Height, samples.Width) { NoiseMask = samples.NoiseMask };
        int w = samples.Width, h = samples.Height;
        for (int plane = 0; plane < samples.Batch * samples.Channels; plane++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    r.Data[(plane * h + y) * w + x] = samples.Data[(plane * h + (vertical ? h - 1 - y : y)) * w + (vertical ? x : w - 1 - x)];
        return r;
    }
}

[Node(Category = "latent/transform", DisplayName = "Rotate Latent")]
public sealed class LatentRotate
{
    public static string[] Rotations() => ["none", "90 degrees", "180 degrees", "270 degrees"];

    /// <summary>Rotates clockwise. 90 and 270 swap width and height (the latent's channels are rotated as pictures, which is what ComfyUI does).</summary>
    public LatentBatch Execute(LatentBatch samples, [Input(OptionsProvider = nameof(Rotations))] string rotation = "none")
    {
        int quarter = Array.IndexOf(Rotations(), rotation);
        if (quarter < 0) throw new ArgumentException($"Unknown rotation '{rotation}'.", nameof(rotation));
        if (quarter == 0) return samples;
        int h = samples.Height, w = samples.Width;
        int nh = quarter % 2 == 1 ? w : h, nw = quarter % 2 == 1 ? h : w;
        var r = new LatentBatch(samples.Batch, samples.Channels, nh, nw);
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

[Node(Category = "latent/transform", DisplayName = "Crop Latent")]
public sealed class LatentCrop
{
    public LatentBatch Execute(LatentBatch samples, [Input(Min = 64, Max = 16384, Step = 8)] int width = 512, [Input(Min = 64, Max = 16384, Step = 8)] int height = 512,
        [Input(Min = 0, Max = 16384, Step = 8)] int x = 0, [Input(Min = 0, Max = 16384, Step = 8)] int y = 0)
    {
        int lx = Math.Min(x / 8, samples.Width - 1), ly = Math.Min(y / 8, samples.Height - 1);
        int w = Math.Min(width / 8, samples.Width - lx), h = Math.Min(height / 8, samples.Height - ly);
        var r = new LatentBatch(samples.Batch, samples.Channels, h, w);
        for (int plane = 0; plane < samples.Batch * samples.Channels; plane++)
            for (int row = 0; row < h; row++)
                Array.Copy(samples.Data, (plane * samples.Height + ly + row) * samples.Width + lx, r.Data, (plane * h + row) * w, w);
        return r;
    }
}

[Node(Category = "_for_testing", DisplayName = "LatentBlend")]
public sealed class LatentBlend
{
    public LatentBatch Execute(LatentBatch samples1, LatentBatch samples2, [Input(Min = 0, Max = 1, Step = 0.01)] float blendFactor = 0.5f)
    {
        if (samples2.Width != samples1.Width || samples2.Height != samples1.Height)
            samples2 = new LatentUpscale().Execute(samples2, "bicubic", samples1.Width * 8, samples1.Height * 8, "center");
        var r = new LatentBatch(samples1.Batch, samples1.Channels, samples1.Height, samples1.Width) { NoiseMask = samples1.NoiseMask };
        for (int i = 0; i < r.Data.Length; i++) r.Data[i] = samples1.Data[i] * blendFactor + samples2.Data[i % samples2.Data.Length] * (1 - blendFactor);
        return r;
    }
}

[Node(Category = "latent/inpaint", DisplayName = "Set Latent Noise Mask")]
public sealed class SetLatentNoiseMask
{
    /// <summary>Marks the part of the latent that sampling may change (mask = 1); the rest is held at the original. The mask may be any size: the sampler resizes it to the latent.</summary>
    public LatentBatch Execute(LatentBatch samples, MaskBatch mask) =>
        new(samples.Batch, samples.Channels, samples.Height, samples.Width, (float[])samples.Data.Clone()) { NoiseMask = mask };
}
