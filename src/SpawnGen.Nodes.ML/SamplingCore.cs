using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

internal sealed record SamplingSettings(long Seed, float Cfg, string Sampler, bool AddNoise = true);

/// <summary>
/// One sampling run over a latent batch: noise it to the first sigma, then step down the given sigmas with the chosen sampler, asking the UNet for
/// guided predictions (CFG over possibly several conditioning entries with areas, masks and timestep ranges) and holding the unmasked part of an
/// inpainting latent at the original. The maths follows ComfyUI so results and workflows behave the same.
/// </summary>
internal static class SamplingCore
{
    public static async Task<LatentBatch> RunAsync(Accelerator acc, Checkpoint ckpt, Conditioning positive, Conditioning negative, LatentBatch latent,
        float[] sigmas, SamplingSettings settings, NodeContext context) =>
        (await RunWithDenoisedAsync(acc, ckpt, positive, negative, latent, sigmas, settings, context)).Output;

    /// <summary>Like <see cref="RunAsync"/>, also returning the model's last denoised estimate (SamplerCustom's <c>denoised_output</c>).</summary>
    public static async Task<(LatentBatch Output, LatentBatch Denoised)> RunWithDenoisedAsync(Accelerator acc, Checkpoint ckpt, Conditioning positive, Conditioning negative, LatentBatch latent,
        float[] sigmas, SamplingSettings settings, NodeContext context)
    {
        if (latent.Channels != Checkpoint.LatentChannels)
            throw new ArgumentException($"Latent has {latent.Channels} channels; this checkpoint expects {Checkpoint.LatentChannels}.", nameof(latent));
        int steps = sigmas.Length - 1, h = latent.Height, w = latent.Width, perImage = Checkpoint.LatentChannels * h * w;
        if (steps < 1) return (latent, latent); // nothing to sample (a start step past the end, for instance)

        var shape = new[] { 1, Checkpoint.LatentChannels, h, w };
        var uploaded = new List<(MemoryBuffer1D<float, Stride1D.Dense> Buffer, Tensor Tensor)>();
        try
        {
            // The embeddings go to the GPU once per run, not once per step.
            Tensor Upload(CondEntry e)
            {
                var (buf, tensor) = GpuHelpers.Upload(acc, e.Embeddings, e.Shape);
                uploaded.Add((buf, tensor));
                return tensor;
            }
            var pos = positive.Entries.Select(e => (Entry: e, Tensor: Upload(e))).ToList();
            var neg = negative.Entries.Select(e => (Entry: e, Tensor: Upload(e))).ToList();

            var result = new LatentBatch(latent.Batch, latent.Channels, h, w) { NoiseMask = latent.NoiseMask };
            var denoisedResult = new LatentBatch(latent.Batch, latent.Channels, h, w) { NoiseMask = latent.NoiseMask };
            bool maxDenoise = MathF.Abs(sigmas[0] - ckpt.Sigmas.SigmaMax) <= 1e-5f * ckpt.Sigmas.SigmaMax || sigmas[0] > ckpt.Sigmas.SigmaMax;

            for (int b = 0; b < latent.Batch; b++)
            {
                var original = new float[perImage];
                Array.Copy(latent.Data, b * perImage, original, 0, perImage);
                var mask = ResizedMask(latent.NoiseMask, b, h, w);

                // Start: the latent plus noise at the first sigma. From the top of the schedule the noise is scaled so the model's input has unit variance.
                var noise = settings.AddNoise ? DiffusionScheduler.GenerateNoise(Checkpoint.LatentChannels, h, w, unchecked((int)(settings.Seed + b))) : new float[perImage];
                float noiseScale = maxDenoise ? MathF.Sqrt(1f + sigmas[0] * sigmas[0]) : sigmas[0];
                var x = new float[perImage];
                for (int i = 0; i < perImage; i++) x[i] = original[i] + noise[i] * noiseScale;
                var lastDenoised = x;

                async Task<float[]> Denoise(float[] sample, float sigma)
                {
                    float scale = 1f / MathF.Sqrt(sigma * sigma + 1f), timestep = ckpt.Sigmas.Timestep(sigma);
                    var scaled = new float[perImage];
                    for (int i = 0; i < perImage; i++) scaled[i] = sample[i] * scale;

                    // Run progress as the share of sigma range covered, for the timestep ranges of conditioning entries.
                    var cond = await GuidedAsync(pos, scaled, shape, timestep, sigma, ckpt, acc, h, w);
                    float[] eps = cond;
                    if (settings.Cfg != 1f)
                    {
                        var uncond = await GuidedAsync(neg, scaled, shape, timestep, sigma, ckpt, acc, h, w);
                        eps = new float[perImage];
                        for (int i = 0; i < perImage; i++) eps[i] = uncond[i] + settings.Cfg * (cond[i] - uncond[i]);
                    }
                    var denoised = new float[perImage];
                    for (int i = 0; i < perImage; i++) denoised[i] = sample[i] - sigma * eps[i];
                    if (mask is not null)
                    {
                        // Inpainting: the part outside the mask is held at the original latent.
                        int plane = h * w;
                        for (int c = 0; c < Checkpoint.LatentChannels; c++)
                            for (int i = 0; i < plane; i++)
                                denoised[c * plane + i] = denoised[c * plane + i] * mask[i] + original[c * plane + i] * (1f - mask[i]);
                    }
                    lastDenoised = denoised;
                    return denoised;
                }

                int item = b;
                x = await Samplers.RunAsync(settings.Sampler, x, sigmas, Denoise, new Samplers.NoiseSource(unchecked((int)(settings.Seed + b) ^ 0x5EED), Checkpoint.LatentChannels, h, w),
                    done => context.Progress(item * steps + done, latent.Batch * steps), context.CancellationToken);
                Array.Copy(x, 0, result.Data, b * perImage, perImage);
                Array.Copy(lastDenoised, 0, denoisedResult.Data, b * perImage, perImage);
            }
            return (result, denoisedResult);
        }
        finally
        {
            foreach (var (buf, _) in uploaded) buf.Dispose();
        }
    }

    /// <summary>
    /// The UNet's prediction for a set of conditioning entries: each active entry is evaluated and blended by its area weight (strength x area x mask),
    /// like ComfyUI. Pixels no entry covers get 0. A single entry that applies everywhere is just its own prediction.
    /// </summary>
    static async Task<float[]> GuidedAsync(List<(CondEntry Entry, Tensor Tensor)> entries, float[] scaled, int[] shape, float timestep, float sigma,
        Checkpoint ckpt, Accelerator acc, int h, int w)
    {
        int plane = h * w, perImage = Checkpoint.LatentChannels * plane;
        var active = entries.Where(e => IsActive(e.Entry, sigma, ckpt.Sigmas)).ToList();
        if (active.Count == 0) return new float[perImage];

        if (active.Count == 1 && active[0].Entry.Area is null && active[0].Entry.Mask is null)
            return await PredictAsync(acc, ckpt.Unet, scaled, shape, timestep, active[0].Tensor); // the common case: one plain prompt

        var sum = new float[perImage];
        var weightSum = new float[plane];
        foreach (var (entry, tensor) in active)
        {
            var eps = await PredictAsync(acc, ckpt.Unet, scaled, shape, timestep, tensor);
            var weight = Weight(entry, h, w);
            for (int i = 0; i < plane; i++)
            {
                if (weight[i] == 0f) continue;
                weightSum[i] += weight[i];
                for (int c = 0; c < Checkpoint.LatentChannels; c++) sum[c * plane + i] += eps[c * plane + i] * weight[i];
            }
        }
        for (int i = 0; i < plane; i++)
            for (int c = 0; c < Checkpoint.LatentChannels; c++) sum[c * plane + i] = weightSum[i] > 0f ? sum[c * plane + i] / weightSum[i] : 0f;
        return sum;
    }

    /// <summary>An entry is active while the run's sigma is inside its percent range (start 0 = from the very beginning, end 1 = to the very end).</summary>
    internal static bool IsActive(CondEntry e, float sigma, ModelSigmas sigmas)
    {
        float startSigma = PercentToSigma(e.StartPercent, sigmas), endSigma = PercentToSigma(e.EndPercent, sigmas);
        return sigma <= startSigma && sigma > endSigma;
    }

    internal static float PercentToSigma(float percent, ModelSigmas sigmas) =>
        percent <= 0f ? float.PositiveInfinity : percent >= 1f ? 0f : sigmas.Sigma((1f - percent) * (sigmas.TrainingSteps - 1));

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

    static async Task<float[]> PredictAsync(Accelerator acc, InferenceSession unet, float[] sample, int[] shape, float timestep, Tensor conditioning)
    {
        var (sampleBuf, sampleTensor) = GpuHelpers.Upload(acc, sample, shape);
        var (tBuf, tTensor) = GpuHelpers.Upload(acc, [timestep], [1]);
        using (sampleBuf)
        using (tBuf)
        {
            var outputs = await unet.RunAsync(new Dictionary<string, Tensor>
            {
                [unet.InputNames[0]] = sampleTensor,
                [unet.InputNames[1]] = tTensor,
                [unet.InputNames[2]] = conditioning,
            });
            return await GpuHelpers.ReadAsync(acc, outputs[unet.OutputNames[0]], sample.Length);
        }
    }
}
