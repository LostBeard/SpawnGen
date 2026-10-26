using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

internal sealed record SamplingSettings(long Seed, string Sampler, bool AddNoise = true, SamplerOptions? Options = null);

/// <summary>
/// One sampling run over a latent batch: noise it to the first sigma, then step down the given sigmas with the chosen sampler, asking the guider for
/// guided predictions (CFG over possibly several conditioning entries with areas, masks and timestep ranges, with the model's guidance patches) and holding the
/// unmasked part of an inpainting latent at the original. The maths follows ComfyUI (KSAMPLER, KSamplerX0Inpaint, CFGGuider) so results and workflows behave the same.
/// </summary>
internal static class SamplingCore
{
    public static async Task<LatentBatch> RunAsync(Accelerator acc, GuiderChoice guider, LatentBatch latent, float[] sigmas, SamplingSettings settings, NodeContext context) =>
        (await RunWithDenoisedAsync(acc, guider, latent, sigmas, settings, context)).Output;

    /// <summary>
    /// The noise a run starts from for picture <paramref name="index"/> of a batch (also AddNoise's). ComfyUI's batch_index picks which of the seed's noises a
    /// picture gets: Get Latent From Batch keeps a picture's own, "fixed" seed behaviour gives every picture the first one's.
    /// </summary>
    internal static float[] InitialNoise(long seed, int index, LatentBatch latent) =>
        DiffusionScheduler.GenerateNoise(latent.Channels, latent.Height, latent.Width, unchecked((int)(seed + NoiseIndex(latent, index))));

    internal static int NoiseIndex(LatentBatch latent, int index) => latent.BatchIndex is { } batchIndex && index < batchIndex.Length ? batchIndex[index] : index;

    /// <summary>Like <see cref="RunAsync"/>, also returning the model's last denoised estimate (SamplerCustom's <c>denoised_output</c>).</summary>
    public static async Task<(LatentBatch Output, LatentBatch Denoised)> RunWithDenoisedAsync(Accelerator acc, GuiderChoice guider, LatentBatch latent,
        float[] sigmas, SamplingSettings settings, NodeContext context)
    {
        var ckpt = guider.Model.Model;
        var sampling = guider.Model.Sampling;
        var format = guider.Model.LatentFormat;
        if (latent.Channels != Checkpoint.LatentChannels)
            throw new ArgumentException($"Latent has {latent.Channels} channels; this checkpoint expects {Checkpoint.LatentChannels}.", nameof(latent));
        int steps = sigmas.Length - 1, h = latent.Height, w = latent.Width, perImage = Checkpoint.LatentChannels * h * w;
        if (sigmas.Length == 0) return (latent, latent); // nothing to sample; a single sigma only noises the latent, as in ComfyUI

        // The model's LoRAs are merged for this run; the embeddings go to the GPU once per run, not once per step.
        await ckpt.UseLorasAsync(guider.Model.Loras, context.CancellationToken);
        // The old Apply ControlNet's controls reach the other conditionings at sampling time (ComfyUI's apply_empty_x_to_equal_area on 'control').
        var positive = guider.Conditionings.First();
        var filled = guider.Conditionings.Skip(1).Distinct(ReferenceEqualityComparer.Instance).Cast<Conditioning>()
            .ToDictionary(c => c, c => ControlFill.Fill(positive, c), ReferenceEqualityComparer.Instance);
        Conditioning Filled(Conditioning c) => filled.TryGetValue(c, out var f) ? f : c;
        var previewer = context.WantsPreview ? await LatentPreviewer.CreateAsync(context.PreviewMethod, format, context.Services) : null;
        using (var unet = ckpt.BeginRun(acc))
        {
            var result = new LatentBatch(latent.Batch, latent.Channels, h, w) { NoiseMask = latent.NoiseMask, BatchIndex = latent.BatchIndex };
            var denoisedResult = new LatentBatch(latent.Batch, latent.Channels, h, w) { NoiseMask = latent.NoiseMask, BatchIndex = latent.BatchIndex };
            bool maxDenoise = sampling.IsMaxDenoise(sigmas[0]);
            var runSigmas = sigmas.Select(v => (double)v).ToArray();

            for (int b = 0; b < latent.Batch; b++)
            {
                // A LATENT holds the VAE's values; the sampler works in the model's (ComfyUI's process_latent_in / process_latent_out).
                var original = format.ProcessIn(latent.Data[(b * perImage)..((b + 1) * perImage)]);
                var mask = ResizedMask(latent.NoiseMask, b, h, w);

                int noiseIndex = NoiseIndex(latent, b);
                int itemSeed = unchecked((int)(settings.Seed + noiseIndex));
                var noise = settings.AddNoise ? InitialNoise(settings.Seed, b, latent) : new float[perImage];
                // ddim is euler with fresh noise for the kept area of an inpainting run (ComfyUI's inpaint_options "random").
                var inpaintNoise = settings.Sampler == "ddim" && mask is not null
                    ? DiffusionScheduler.GenerateNoise(Checkpoint.LatentChannels, h, w, itemSeed ^ 0x1A1D) : noise;
                var x = sampling.NoiseScaling(sigmas[0], noise, original, maxDenoise);
                float[]? lastDenoised = null; // ComfyUI's x0_output: the last callback's estimate, else the output itself

                var guidance = new GuidanceRun(guider, Checkpoint.LatentChannels, h, w, Samplers.DisablesCfg1Optimization(settings.Sampler)) { BatchIndex = b, BatchSize = latent.Batch };
                int item = b;

                Task<float[]> Denoise(float[] sample, float sigma) =>
                    InpaintedCall(sample, sigma, mask, guider.Model.Patches.DenoiseMask, runSigmas, sampling, original, inpaintNoise, h * w,
                        input => guidance.PredictAsync(input, sigma,
                            (cond, scheduled) => GuidedAsync(Filled(cond).Entries, input, sigma, sampling, unet, h, w, scheduled, item, context.CancellationToken)));

                x = await Samplers.RunAsync(settings.Sampler, x, sigmas, Denoise,
                    new Samplers.NoiseSource(itemSeed ^ 0x5EED, Checkpoint.LatentChannels, h, w),
                    done => context.Progress(item * steps + Math.Min(done, steps), Math.Max(1, latent.Batch * steps)),
                    context.CancellationToken, () => guidance.Last, settings.Options, sampling,
                    async d =>
                    {
                        lastDenoised = d;
                        if (item == 0 && previewer is not null) context.Preview(await previewer.PngAsync(d, h, w, context.CancellationToken));
                    });
                x = sampling.InverseNoiseScaling(sigmas[^1], x);
                Array.Copy(format.ProcessOut(x), 0, result.Data, b * perImage, perImage);
                Array.Copy(format.ProcessOut(lastDenoised ?? x), 0, denoisedResult.Data, b * perImage, perImage);
            }
            return (result, denoisedResult);
        }
    }

    /// <summary>
    /// ComfyUI's KSamplerX0Inpaint around one model call: with an inpainting mask (possibly reshaped per step by a denoise mask function such as Differential
    /// Diffusion's) the kept area of the input is the original latent noised to this sigma with the run's noise, and the kept area of the output is the original itself.
    /// </summary>
    internal static async Task<float[]> InpaintedCall(float[] sample, double sigma, float[]? mask, Func<double, float[], double[], ModelSampling, float[]>? denoiseMask,
        double[] runSigmas, ModelSampling sampling, float[] original, float[] inpaintNoise, int plane, Func<float[], Task<float[]>> predict)
    {
        if (mask is null) return await predict(sample);
        var m = denoiseMask is null ? mask : denoiseMask(sigma, mask, runSigmas, sampling);
        var input = Blend(sample, sampling.NoiseScaling(sigma, inpaintNoise, original), m, plane);
        return Blend(await predict(input), original, m, plane);
    }

    /// <summary>a * m + b * (1 - m) with a per-pixel mask broadcast over the channels.</summary>
    internal static float[] Blend(float[] a, float[] b, float[] m, int plane)
    {
        var r = new float[a.Length];
        for (int i = 0; i < r.Length; i++)
        {
            float mi = m[i % plane];
            r[i] = a[i] * mi + b[i] * (1f - mi);
        }
        return r;
    }

    /// <summary>
    /// ComfyUI's calc_cond_batch for one conditioning: each active entry's denoised estimate, blended by its weight (strength x area x mask) over a count that starts at
    /// 1e-37, so pixels no entry covers come out 0. <paramref name="scheduled"/> false ignores the entries' timestep ranges.
    /// </summary>
    static async Task<float[]> GuidedAsync(IReadOnlyList<CondEntry> entries, float[] x, double sigma, ModelSampling sampling, UNetRun unet,
        int h, int w, bool scheduled, int batchIndex, CancellationToken cancellationToken)
    {
        int plane = h * w, perImage = Checkpoint.LatentChannels * plane;
        var active = entries.Where(e => !scheduled || IsActive(e, sigma, sampling)).ToList();
        if (active.Count == 0) return new float[perImage];

        var input = sampling.CalculateInput(sigma, x);
        float timestep = sampling.Timestep(sigma);
        async Task<float[]> Predict(CondEntry e) =>
            sampling.CalculateDenoised(sigma, await unet.PredictAsync(input, h, w, timestep, e, Controls(e, sigma, sampling, batchIndex, h, w), cancellationToken), x);

        if (active.Count == 1 && active[0].Area is null && active[0].Mask is null && active[0].Strength != 0f)
            return await Predict(active[0]); // the common case: one plain prompt (strength cancels in the blend)

        var sum = new float[perImage];
        var weightSum = new float[plane];
        Array.Fill(weightSum, 1e-37f);
        foreach (var entry in active)
        {
            var den = await Predict(entry);
            var weight = Weight(entry, h, w);
            for (int i = 0; i < plane; i++)
            {
                if (weight[i] == 0f) continue;
                weightSum[i] += weight[i];
                for (int c = 0; c < Checkpoint.LatentChannels; c++) sum[c * plane + i] += den[c * plane + i] * weight[i];
            }
        }
        for (int i = 0; i < plane; i++)
            for (int c = 0; c < Checkpoint.LatentChannels; c++) sum[c * plane + i] /= weightSum[i];
        return sum;
    }

    /// <summary>The ControlNets of an entry's chain active at this sigma, with their hints for this picture.</summary>
    static List<ControlCall>? Controls(CondEntry e, double sigma, ModelSampling sampling, int batchIndex, int h, int w)
    {
        if (e.Control is null) return null;
        var calls = new List<ControlCall>();
        foreach (var c in e.Control.Active(sigma, sampling))
        {
            var use = c;
            calls.Add(new ControlCall(use.ControlNet.Net, () => use.Hint(batchIndex, 8 * h, 8 * w), (use, batchIndex % use.Image.Batch), use.Strength));
        }
        return calls;
    }

    /// <summary>An entry is active while the run's sigma is inside its percent range: end sigma &lt;= sigma &lt;= start sigma (ComfyUI's timestep_start / timestep_end).</summary>
    internal static bool IsActive(CondEntry e, double sigma, ModelSampling sampling) =>
        sigma <= PercentToSigma(e.StartPercent, sampling) && sigma >= PercentToSigma(e.EndPercent, sampling);

    internal static double PercentToSigma(float percent, ModelSampling sampling) => sampling.PercentToSigma(percent);

    /// <summary>Where and how much one entry counts, over the latent: strength x (1 inside its area) x its mask.</summary>
    internal static float[] Weight(CondEntry e, int h, int w)
    {
        var weight = new float[h * w];
        Array.Fill(weight, e.Strength);
        if (e.Area is { } area)
        {
            var (ax, ay, aw, ah) = area.Resolve(w, h);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    if (x < ax || x >= ax + aw || y < ay || y >= ay + ah) weight[y * w + x] = 0f;
        }
        if (ResizedMask(e.Mask, 0, h, w) is { } mask)
            for (int i = 0; i < weight.Length; i++) weight[i] *= mask[i];
        return weight;
    }

    /// <summary>Frame <paramref name="index"/> (wrapping) of a mask of any size, resized to the latent with bilinear filtering; null when there is no mask.</summary>
    internal static float[]? ResizedMask(MaskBatch? mask, int index, int h, int w)
    {
        if (mask is null) return null;
        var one = new float[mask.Height * mask.Width];
        Array.Copy(mask.Data, (index % mask.Batch) * one.Length, one, 0, one.Length);
        return mask.Width == w && mask.Height == h ? one : Resample.Resize(one, mask.Width, mask.Height, 1, w, h, ResampleMethod.Bilinear);
    }

}
