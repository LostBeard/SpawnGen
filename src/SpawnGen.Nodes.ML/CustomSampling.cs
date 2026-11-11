using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries for the custom sampling nodes (noise, guiders, SamplerCustom and
// SamplerCustomAdvanced) and the behaviour SamplingNodeSchemaTests and CustomSamplingTests pin down (AddNoise's noise in the model's latent space,
// SamplerCustom matching KSampler on the same sigmas and seed).

/// <summary>NOISE: the starting noise of a run, from a seed, or none.</summary>
[Socket("NOISE")]
public sealed record NoiseChoice(long Seed, bool Enabled);

[Node("RandomNoise", Category = "model/sampling/noise", DisplayName = "RandomNoise")]
public sealed class RandomNoise
{
    public NoiseChoice Execute([Input(Min = 0, Max = 18446744073709551615d)] long noiseSeed = 0) => new(noiseSeed, true);
}

[Node("DisableNoise", Category = "model/sampling/noise", DisplayName = "DisableNoise")]
public sealed class DisableNoise
{
    public NoiseChoice Execute() => new(0, false);
}

[Node("AddNoise", Category = "model/sampling/noise", DisplayName = "AddNoise")]
public sealed class AddNoise
{
    /// <summary>
    /// The latent with the seed's noise added in the model's latent space at the noise level the sigmas span (|first - last|, or the one sigma);
    /// each picture gets its own noise. No sigmas: the latent itself; disabled noise: the latent unchanged.
    /// </summary>
    public LatentBatch Execute(ModelHandle model, NoiseChoice noise, Sigmas sigmas, LatentBatch latentImage)
    {
        var s = sigmas.Values;
        if (s.Length == 0) return latentImage;
        if (!noise.Enabled) return Copy(latentImage, (float[])latentImage.Data.Clone());
        double scale = s.Length > 1 ? Math.Abs(s[0] - s[^1]) : s[0];
        var format = model.LatentFormat;
        int per = latentImage.Data.Length / Math.Max(latentImage.Batch, 1);
        var r = new float[latentImage.Data.Length];
        for (int b = 0; b < latentImage.Batch; b++)
        {
            var picture = latentImage.Data.AsSpan(b * per, per).ToArray();
            var n = SamplingCore.InitialNoise(noise.Seed, b, latentImage);
            var noisy = model.Sampling.NoiseScaling(scale, n, format.ProcessIn(picture));
            format.ProcessOut(noisy).CopyTo(r, b * per);
        }
        return Copy(latentImage, r);
    }

    static LatentBatch Copy(LatentBatch like, float[] data) =>
        new(like.Batch, like.Channels, like.Height, like.Width, data) { NoiseMask = like.NoiseMask, BatchIndex = like.BatchIndex };
}

[Node("BasicGuider", Category = "model/sampling/guiders", DisplayName = "Basic Guider")]
public sealed class BasicGuider
{
    public GuiderChoice Execute(ModelHandle model, Conditioning conditioning) => new BasicGuiderChoice(model, conditioning);
}

[Node("CFGGuider", Category = "model/sampling/guiders", DisplayName = "CFG Guider")]
public sealed class CFGGuider
{
    public GuiderChoice Execute(ModelHandle model, Conditioning positive, Conditioning negative, [Input(Min = 0.0, Max = 100.0, Step = 0.1)] double cfg = 8.0) =>
        new CfgGuider(model, positive, negative, cfg);
}

[Node("SamplerCustom", Category = "model/sampling/custom", DisplayName = "SamplerCustom", OutputNames = ["output", "denoised_output"])]
public sealed class SamplerCustom
{
    /// <summary>The sampler run over the sigmas with classifier-free guidance; also the last denoised estimate.</summary>
    public async Task<(LatentBatch Output, LatentBatch DenoisedOutput)> ExecuteAsync(ModelHandle model, Conditioning positive, Conditioning negative,
        SamplerChoice sampler, Sigmas sigmas, LatentBatch latentImage, NodeContext context, IServiceProvider services, bool addNoise = true,
        [Input(Min = 0, Max = 18446744073709551615d)] long noiseSeed = 0, [Input(Min = 0.0, Max = 100.0, Step = 0.1)] double cfg = 8.0) =>
        await SamplerCustomAdvanced.RunAsync(new CfgGuider(model, positive, negative, cfg), new NoiseChoice(noiseSeed, addNoise), sampler, sigmas,
            latentImage, context, services);
}

[Node("SamplerCustomAdvanced", Category = "model/sampling/custom", DisplayName = "SamplerCustomAdvanced", OutputNames = ["output", "denoised_output"])]
public sealed class SamplerCustomAdvanced
{
    public async Task<(LatentBatch Output, LatentBatch DenoisedOutput)> ExecuteAsync(NoiseChoice noise, GuiderChoice guider, SamplerChoice sampler, Sigmas sigmas,
        LatentBatch latentImage, NodeContext context, IServiceProvider services) =>
        await RunAsync(guider, noise, sampler, sigmas, latentImage, context, services);

    internal static async Task<(LatentBatch Output, LatentBatch DenoisedOutput)> RunAsync(GuiderChoice guider, NoiseChoice noise, SamplerChoice sampler,
        Sigmas sigmas, LatentBatch latent, NodeContext context, IServiceProvider services)
    {
        var model = guider.Model;
        if (model.Model is { } m) await m.UseWeightsAsync(model.Weights, context.CancellationToken);
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var settings = new SamplingSettings(noise.Seed, sampler.Name, noise.Enabled, sampler.Options);
        return await SamplingCore.RunWithDenoisedAsync(runtime.Accelerator, guider, latent, sigmas.Values, settings, context);
    }
}
