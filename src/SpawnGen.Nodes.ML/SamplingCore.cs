using ILGPU.Runtime;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md). The run's bookkeeping follows ComfyUI's documented behaviour, checked by the tests: the starting noise is
// torch.randn of the latent from a CPU generator seeded with the seed (Fixtures/sampling/comfyui_sd15); a conditioning's entries are evaluated where
// they are active (timestep ranges inclusive, SamplingSetupTests) and combined as a weighted average over their areas and masks; inpainting
// re-noises the kept area to the step's sigma on the way into the model and pins it on the way out (SamplerParityTests' inpaint cases).
// A video latent [C, F, h, w] is laid out like a picture latent [C, F x h, w], so the run treats it as one (areas and masks over the stacked
// frames: a video noise mask is a MaskBatch of F x h rows); only the model call is told the frames (UNetRun.Frames).

/// <summary>A sampling run's settings: seed, sampler, whether the start is noised, the sampler's options.</summary>
internal sealed record SamplingSettings(long Seed, string Sampler, bool AddNoise = true, SamplerOptions? Options = null);

/// <summary>Runs a guider's model over a latent batch on the GPU, picture by picture.</summary>
internal static class SamplingCore
{
    public static async Task<LatentBatch> RunAsync(Accelerator acc, GuiderChoice guider, LatentBatch latent, float[] sigmas, SamplingSettings settings, NodeContext context) =>
        (await RunWithDenoisedAsync(acc, guider, latent, sigmas, settings, context)).Output;

    /// <summary>
    /// The starting noise of picture <paramref name="index"/>: torch.randn of the whole latent batch from a CPU generator seeded with the seed;
    /// with batch indexes, one picture-sized draw per index in turn, the picture taking its index's draw.
    /// </summary>
    internal static float[] InitialNoise(long seed, int index, LatentBatch latent)
    {
        int per = latent.ItemSize;
        if (latent.BatchIndex is null) return TorchRandom.Randn(seed, per * latent.Batch).AsSpan(index * per, per).ToArray();
        int want = NoiseIndex(latent, index);
        var g = new TorchRandom(seed);
        float[] draw = [];
        for (int i = 0; i <= want; i++) draw = g.Randn(per);
        return draw;
    }

    /// <summary>Which noise picture <paramref name="index"/> takes (its batch index, else its position).</summary>
    internal static int NoiseIndex(LatentBatch latent, int index) => latent.BatchIndex is { } bi && index < bi.Length ? bi[index] : index;

    /// <summary>The sampled latent and the last denoised estimate (both in the latent's own units).</summary>
    public static async Task<(LatentBatch Output, LatentBatch Denoised)> RunWithDenoisedAsync(Accelerator acc, GuiderChoice guider, LatentBatch latent,
        float[] sigmas, SamplingSettings settings, NodeContext context)
    {
        if (sigmas.Length < 2) return (latent, latent);
        var model = guider.Model;
        var diffusion = model.Model ?? throw new InvalidOperationException("The guider has no diffusion model to run.");
        var sampling = model.Sampling;
        var format = model.LatentFormat;
        latent = ToModelChannels(latent, format);
        int batch = latent.Batch, ch = latent.Channels, h = StackedHeight(latent), w = latent.Width, plane = h * w, per = ch * plane;
        var runSigmas = sigmas.Select(v => (double)v).ToArray();
        var output = new float[latent.Data.Length];
        var denoisedOut = new float[latent.Data.Length];
        var previewer = context.WantsPreview && !latent.IsVideo && format.RgbFactors.Length > 0 ? await LatentPreviewer.CreateAsync(context.PreviewMethod, format, context.Services) : null;
        var ct = context.CancellationToken;
        int steps = sigmas.Length - 1, done = 0, total = steps * batch;
        using var unet = diffusion.BeginRun(acc);
        unet.Frames = latent.Frames;
        for (int b = 0; b < batch; b++)
        {
            var original = format.ProcessIn(latent.Data.AsSpan(b * per, per).ToArray());
            var noise = settings.AddNoise ? InitialNoise(settings.Seed, b, latent) : new float[per];
            var x0 = sampling.NoiseScaling(sigmas[0], noise, original, sampling.IsMaxDenoise(sigmas[0]));
            var mask = ResizedMask(latent.NoiseMask, b, h, w);
            unet.FirstFrameClean = latent.IsVideo && FirstFrameOnlyKept(mask, latent.Frames, plane / latent.Frames);
            var run = new GuidanceRun(guider, ch, h, w, Samplers.DisablesCfg1Optimization(settings.Sampler)) { BatchIndex = b, BatchSize = batch };
            int item = b;
            Samplers.DenoiseFn denoise = (x, sigma) => InpaintedCall(x, sigma, mask, model.Patches?.DenoiseMask, runSigmas, sampling, original, noise, plane,
                input => run.PredictAsync(input, sigma,
                    (cond, scheduled, options) => EvaluateAsync(unet, cond, input, sigma, scheduled, options, sampling, format, latent, original, item, ct),
                    (cond, at, options) => EvaluateAsync(unet, cond, at, sigma, true, options, sampling, format, latent, original, item, ct),
                    async () => await unet.ReadMiddleAttentionAsync() ?? throw new NotSupportedException("This model cannot record its middle attention.")));
            var noiseSource = new Samplers.NoiseSource(settings.Seed, batch, b, ch, h, w);
            float[]? last = null;
            var x = await Samplers.RunAsync(settings.Sampler, x0, sigmas, denoise, noiseSource,
                onStep: _ => context.Progress(++done, total), ct: ct, uncond: () => run.Last, options: settings.Options, sampling: sampling,
                onDenoised: async d =>
                {
                    last = d;
                    if (previewer is not null) context.Preview(await previewer.PngAsync(d, h, w, ct));
                });
            x = sampling.InverseNoiseScaling(sigmas[^1], x);
            format.ProcessOut(x).CopyTo(output, b * per);
            format.ProcessOut(last ?? x).CopyTo(denoisedOut, b * per);
        }
        return (LatentBatch.Shaped(latent, batch, output).With(latent), LatentBatch.Shaped(latent, batch, denoisedOut).With(latent));
    }

    /// <summary>
    /// An all-zero latent (an empty latent) with another channel count than the model's becomes the model's: zeros of the same size, with the noise
    /// mask and batch indexes kept. Measured with ComfyUI (2026-11-13): SD 1.5 sampled from Empty SD3 Latent Image (16 channels) gives a 4-channel
    /// result identical to the one from Empty Latent Image. A latent with any non-zero value is passed on unchanged.
    /// </summary>
    internal static LatentBatch ToModelChannels(LatentBatch latent, LatentFormat format)
    {
        if (latent.Channels == format.Channels || !Array.TrueForAll(latent.Data, v => v == 0f)) return latent;
        var zeros = latent.IsVideo
            ? new LatentBatch(latent.Batch, format.Channels, latent.Frames, latent.Height, latent.Width)
            : new LatentBatch(latent.Batch, format.Channels, latent.Height, latent.Width);
        return zeros.With(latent);
    }

    /// <summary>The latent's rows as the run sees them: a video latent's frames stacked (F x h), a picture latent's height.</summary>
    internal static int StackedHeight(LatentBatch latent) => latent.Frames * latent.Height;

    /// <summary>
    /// Whether an image-to-video mask keeps the first frame (0 there, 1 everywhere else, as Wan22ImageToVideoLatent makes it): the model then runs that
    /// frame at timestep 0. No mask, or one that keeps nothing, is ordinary sampling; one that keeps everything (a one-frame video from a picture) too,
    /// since every frame is pinned to the original anyway. Any other kept region is not something the video models take yet.
    /// </summary>
    internal static bool FirstFrameOnlyKept(float[]? mask, int frames, int framePlane)
    {
        if (mask is null || Array.TrueForAll(mask, v => v == 1f)) return false;
        if (Array.TrueForAll(mask, v => v == 0f)) return true;
        for (int i = 0; i < mask.Length; i++)
            if (mask[i] != (i < framePlane ? 0f : 1f))
                throw new NotSupportedException("This video latent's noise mask keeps something other than the first frame; SpawnGen's video models take a start picture only.");
        return true;
    }

    /// <summary>
    /// A conditioning's denoised estimate at <paramref name="x"/>: each active entry evaluated over its area (the model sees the cropped latent),
    /// weighted by its strength and mask, the weighted estimates averaged per position (0 where nothing applies).
    /// </summary>
    static async Task<float[]> EvaluateAsync(UNetRun unet, Conditioning cond, float[] x, double sigma, bool scheduled, UNetOptions options,
        ModelSampling sampling, LatentFormat format, LatentBatch latent, float[] original, int item, CancellationToken ct)
    {
        int ch = latent.Channels, h = StackedHeight(latent), w = latent.Width, plane = h * w;
        var sum = new double[x.Length];
        var count = new double[plane];
        float t = (float)sampling.Timestep(sigma);
        var input = sampling.CalculateInput(sigma, x);
        foreach (var e in cond.Entries)
        {
            if (scheduled && !IsActive(e, sigma, sampling)) continue;
            var weight = Weight(e, h, w, item);
            var (ax, ay, aw, ah) = e.Area is { } area ? Clip(area.Resolve(w, h), w, h) : (0, 0, w, h);
            if (aw <= 0 || ah <= 0) continue;
            var cropIn = Crop(input, ch, h, w, ax, ay, aw, ah);
            var cropX = Crop(x, ch, h, w, ax, ay, aw, ah);
            float[]? concat = null;
            if (unet.ConcatChannels > 0)
            {
                var full = InpaintConcat(e, original, latent.NoiseMask, format, item, latent.Batch, h, w);
                concat = Crop(full, full.Length / plane, h, w, ax, ay, aw, ah);
            }
            List<ControlCall>? controls = null;
            if (e.Control is { } control)
            {
                foreach (var use in control.Active(sigma, sampling))
                {
                    controls ??= [];
                    var u = use;
                    controls.Add(new ControlCall(u.ControlNet.Net,
                        () => aw == w && ah == h ? u.Hint(item, 8 * h, 8 * w) : Crop(u.Hint(item, 8 * h, 8 * w), 3, 8 * h, 8 * w, 8 * ax, 8 * ay, 8 * aw, 8 * ah),
                        (u, item, ax, ay, aw, ah), u.Strength));
                }
            }
            var outp = await unet.PredictAsync(cropIn, ah, aw, t, e, controls, options, concat, ct);
            var den = sampling.CalculateDenoised(sigma, outp, cropX);
            for (int yy = 0; yy < ah; yy++)
                for (int xx = 0; xx < aw; xx++)
                {
                    int p = (ay + yy) * w + ax + xx;
                    double wt = weight[p];
                    if (wt == 0) continue;
                    count[p] += wt;
                    for (int c = 0; c < ch; c++) sum[c * plane + p] += den[c * ah * aw + yy * aw + xx] * wt;
                }
        }
        var r = new float[x.Length];
        for (int c = 0; c < ch; c++)
            for (int p = 0; p < plane; p++)
                r[c * plane + p] = count[p] > 0 ? (float)(sum[c * plane + p] / count[p]) : 0f;
        return r;
    }

    static (int X, int Y, int Width, int Height) Clip((int X, int Y, int Width, int Height) a, int w, int h)
    {
        int x0 = Math.Clamp(a.X, 0, w), y0 = Math.Clamp(a.Y, 0, h);
        return (x0, y0, Math.Clamp(a.Width, 0, w - x0), Math.Clamp(a.Height, 0, h - y0));
    }

    static float[] Crop(float[] src, int ch, int h, int w, int x0, int y0, int cw, int chh)
    {
        if (x0 == 0 && y0 == 0 && cw == w && chh == h) return src;
        var r = new float[ch * cw * chh];
        for (int c = 0; c < ch; c++)
            for (int y = 0; y < chh; y++)
                Array.Copy(src, c * h * w + (y0 + y) * w + x0, r, c * cw * chh + y * cw, cw);
        return r;
    }

    /// <summary>
    /// One model call with inpainting: the kept area (mask 0) replaced by the original noised to this sigma on the way in, and by the original
    /// itself on the way out. Differential Diffusion's function, when set, turns the mask into the step's mask first.
    /// </summary>
    internal static async Task<float[]> InpaintedCall(float[] sample, double sigma, float[]? mask, Func<double, float[], double[], ModelSampling, float[]>? denoiseMask,
        double[] runSigmas, ModelSampling sampling, float[] original, float[] inpaintNoise, int plane, Func<float[], Task<float[]>> predict)
    {
        if (mask is null) return await predict(sample);
        var m = denoiseMask is null ? mask : denoiseMask(sigma, mask, runSigmas, sampling);
        var noised = sampling.NoiseScaling(sigma, inpaintNoise, original);
        var result = await predict(Blend(sample, noised, m, plane));
        return Blend(result, original, m, plane);
    }

    /// <summary>a m + b (1 - m), the mask per position (shared by the channels) or per value.</summary>
    internal static float[] Blend(float[] a, float[] b, float[] m, int plane)
    {
        var r = new float[a.Length];
        bool full = m.Length == a.Length;
        for (int i = 0; i < r.Length; i++)
        {
            float k = full ? m[i] : m[i % plane];
            r[i] = a[i] * k + b[i] * (1 - k);
        }
        return r;
    }

    /// <summary>
    /// What an inpainting UNet gets besides the latent: the mask at latent size, then the masked picture's latent in the model's units (the entry's
    /// from Inpaint Model Conditioning, else the run's latent and noise mask; no mask means all of it).
    /// </summary>
    internal static float[] InpaintConcat(CondEntry e, float[] latentImage, MaskBatch? noiseMask, LatentFormat format, int item, int batch, int h, int w)
    {
        int plane = h * w;
        var mask = ResizedMask(e.ConcatMask ?? noiseMask, item, h, w) ?? Enumerable.Repeat(1f, plane).ToArray();
        float[] masked;
        if (e.ConcatLatent is { } cl)
        {
            int per = cl.Channels * cl.Height * cl.Width, slot = BatchSlot(cl.Batch, batch, item);
            var picture = cl.Data.AsSpan(slot * per, per).ToArray();
            if (cl.Height != h || cl.Width != w) picture = TorchInterpolate.Resize(picture, cl.Channels, cl.Height, cl.Width, h, w, "bilinear");
            masked = format.ProcessIn(picture);
        }
        else masked = latentImage;
        var r = new float[plane + masked.Length];
        Array.Copy(mask, 0, r, 0, plane);
        Array.Copy(masked, 0, r, plane, masked.Length);
        return r;
    }

    /// <summary>Which of <paramref name="count"/> conditioning pictures goes with picture <paramref name="index"/> of a batch (repeated when shorter).</summary>
    internal static int BatchSlot(int count, int batch, int index) => count <= 1 ? 0 : index % count;

    /// <summary>Whether an entry applies at <paramref name="sigma"/>: its end sigma &lt;= sigma &lt;= its start sigma.</summary>
    internal static bool IsActive(CondEntry e, double sigma, ModelSampling sampling) =>
        sigma <= PercentToSigma(e.StartPercent, sampling) && sigma >= PercentToSigma(e.EndPercent, sampling);

    internal static double PercentToSigma(float percent, ModelSampling sampling) => sampling.PercentToSigma(percent);

    /// <summary>An entry's weight per latent position: its strength inside its area (everywhere without one), times its mask resized to the latent.</summary>
    internal static float[] Weight(CondEntry e, int h, int w) => Weight(e, h, w, 0);

    static float[] Weight(CondEntry e, int h, int w, int item)
    {
        var r = new float[h * w];
        var (ax, ay, aw, ah) = e.Area is { } area ? Clip(area.Resolve(w, h), w, h) : (0, 0, w, h);
        var mask = ResizedMask(e.Mask, item, h, w);
        for (int y = ay; y < ay + ah; y++)
            for (int x = ax; x < ax + aw; x++)
            {
                int p = y * w + x;
                r[p] = e.Strength * (mask is null ? 1f : mask[p]);
            }
        return r;
    }

    /// <summary>Picture <paramref name="index"/> of a mask batch (repeated when shorter) at h x w (bilinear), or null without a mask.</summary>
    internal static float[]? ResizedMask(MaskBatch? mask, int index, int h, int w)
    {
        if (mask is null) return null;
        int plane = mask.Height * mask.Width, slot = BatchSlot(mask.Batch, Math.Max(mask.Batch, index + 1), index);
        var frame = mask.Data.AsSpan(slot * plane, plane).ToArray();
        return mask.Height == h && mask.Width == w ? frame : TorchInterpolate.Resize(frame, 1, mask.Height, mask.Width, h, w, "bilinear");
    }
}
