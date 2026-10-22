using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

/// <summary>The noise levels a custom sampling run steps through (length steps + 1, ending in 0): ComfyUI's SIGMAS.</summary>
[Socket("SIGMAS")]
public sealed record Sigmas(float[] Values);

/// <summary>A sampler picked by name (ComfyUI's SAMPLER).</summary>
[Socket("SAMPLER")]
public sealed record SamplerChoice(string Name);

/// <summary>The sigma schedules of k-diffusion, from their formulas (a final 0 is appended like ComfyUI does).</summary>
public static class SigmaSchedules
{
    static IEnumerable<double> Ramp(int n, double from, double to) => Enumerable.Range(0, n).Select(i => n == 1 ? from : from + (to - from) * i / (n - 1));

    static float[] WithZero(IEnumerable<double> values) => [.. values.Select(v => (float)v), 0f];

    public static float[] Karras(int steps, double sigmaMax, double sigmaMin, double rho)
    {
        double minInv = Math.Pow(sigmaMin, 1 / rho), maxInv = Math.Pow(sigmaMax, 1 / rho);
        return WithZero(Ramp(steps, 0, 1).Select(t => Math.Pow(maxInv + t * (minInv - maxInv), rho)));
    }

    public static float[] Exponential(int steps, double sigmaMax, double sigmaMin) =>
        WithZero(Ramp(steps, Math.Log(sigmaMax), Math.Log(sigmaMin)).Select(Math.Exp));

    public static float[] Polyexponential(int steps, double sigmaMax, double sigmaMin, double rho) =>
        WithZero(Ramp(steps, 1, 0).Select(t => Math.Exp(Math.Pow(t, rho) * (Math.Log(sigmaMax) - Math.Log(sigmaMin)) + Math.Log(sigmaMin))));

    /// <summary>Variance-preserving schedule (Song et al.): sigma(t) = sqrt(exp(beta_d t^2 / 2 + beta_min t) - 1) for t from 1 down to eps_s.</summary>
    public static float[] VariancePreserving(int steps, double betaD, double betaMin, double epsS) =>
        WithZero(Ramp(steps, 1, epsS).Select(t => Math.Sqrt(Math.Exp(betaD * t * t / 2 + betaMin * t) - 1)));

    /// <summary>SD-Turbo's few-step schedule: the model's sigmas at timesteps 999, 899, ... 99, starting <c>10 - 10 * denoise</c> entries in.</summary>
    public static float[] SdTurbo(ModelSigmas model, int steps, float denoise)
    {
        int start = 10 - (int)(10 * denoise);
        var timesteps = Enumerable.Range(0, 10).Select(i => 999 - 100 * i).Skip(start).Take(steps);
        return WithZero(timesteps.Select(t => (double)model.Sigma(t)));
    }
}

[Node(Category = "sampling/custom_sampling/samplers", DisplayName = "KSamplerSelect")]
public sealed class KSamplerSelect
{
    public static string[] SamplerNames() => Samplers.Names;

    public SamplerChoice Execute([Input(OptionsProvider = nameof(SamplerNames))] string samplerName = "euler") =>
        Samplers.Names.Contains(samplerName) ? new SamplerChoice(samplerName)
            : throw new ArgumentException($"Unknown sampler '{samplerName}'. Known: {string.Join(", ", Samplers.Names)}.", nameof(samplerName));
}

[Node(Category = "sampling/custom_sampling/schedulers", DisplayName = "BasicScheduler")]
public sealed class BasicScheduler
{
    public static string[] SchedulerNames() => ModelSigmas.SchedulerNames;

    /// <summary>The sigmas of one of the model's own schedulers; with denoise below 1 only the tail of a longer schedule is used (as KSampler does).</summary>
    public Sigmas Execute(ModelHandle model, [Input(OptionsProvider = nameof(SchedulerNames))] string scheduler = "simple",
        [Input(Min = 1, Max = 10000)] int steps = 20, [Input(Min = 0, Max = 1, Step = 0.01)] float denoise = 1f)
    {
        if (denoise <= 0f) return new Sigmas([]);
        return new Sigmas(model.Checkpoint.Sigmas.ScheduleForDenoise(scheduler, steps, Math.Min(denoise, 1f)));
    }
}

[Node(Category = "sampling/custom_sampling/schedulers", DisplayName = "KarrasScheduler")]
public sealed class KarrasScheduler
{
    public Sigmas Execute([Input(Min = 1, Max = 10000)] int steps = 20, [Input(Min = 0, Max = 5000, Step = 0.01)] float sigmaMax = 14.614642f,
        [Input(Min = 0, Max = 5000, Step = 0.01)] float sigmaMin = 0.0291675f, [Input(Min = 0, Max = 100, Step = 0.01)] float rho = 7f) =>
        new(SigmaSchedules.Karras(steps, sigmaMax, sigmaMin, rho));
}

[Node(Category = "sampling/custom_sampling/schedulers", DisplayName = "ExponentialScheduler")]
public sealed class ExponentialScheduler
{
    public Sigmas Execute([Input(Min = 1, Max = 10000)] int steps = 20, [Input(Min = 0, Max = 5000, Step = 0.01)] float sigmaMax = 14.614642f,
        [Input(Min = 0, Max = 5000, Step = 0.01)] float sigmaMin = 0.0291675f) =>
        new(SigmaSchedules.Exponential(steps, sigmaMax, sigmaMin));
}

[Node(Category = "sampling/custom_sampling/schedulers", DisplayName = "PolyexponentialScheduler")]
public sealed class PolyexponentialScheduler
{
    public Sigmas Execute([Input(Min = 1, Max = 10000)] int steps = 20, [Input(Min = 0, Max = 5000, Step = 0.01)] float sigmaMax = 14.614642f,
        [Input(Min = 0, Max = 5000, Step = 0.01)] float sigmaMin = 0.0291675f, [Input(Min = 0, Max = 100, Step = 0.01)] float rho = 1f) =>
        new(SigmaSchedules.Polyexponential(steps, sigmaMax, sigmaMin, rho));
}

[Node(Category = "sampling/custom_sampling/schedulers", DisplayName = "VPScheduler")]
public sealed class VPScheduler
{
    public Sigmas Execute([Input(Min = 1, Max = 10000)] int steps = 20, [Input(Min = 0, Max = 5000, Step = 0.01)] float betaD = 19.9f,
        [Input(Min = 0, Max = 5000, Step = 0.01)] float betaMin = 0.1f, [Input(Min = 0, Max = 1, Step = 0.0001)] float epsS = 0.001f) =>
        new(SigmaSchedules.VariancePreserving(steps, betaD, betaMin, epsS));
}

[Node(Category = "sampling/custom_sampling/schedulers", DisplayName = "SDTurboScheduler")]
public sealed class SDTurboScheduler
{
    public Sigmas Execute(ModelHandle model, [Input(Min = 1, Max = 10)] int steps = 1, [Input(Min = 0, Max = 1, Step = 0.01)] float denoise = 1f) =>
        new(SigmaSchedules.SdTurbo(model.Checkpoint.Sigmas, steps, denoise));
}

[Node(Category = "sampling/custom_sampling/sigmas", DisplayName = "SplitSigmas")]
public sealed class SplitSigmas
{
    /// <summary>Splits at step <paramref name="step"/>: high is sigmas[0..step], low is sigmas[step..] (they share the sigma at the split).</summary>
    public (Sigmas High, Sigmas Low) Execute(Sigmas sigmas, [Input(Min = 0, Max = 10000)] int step = 0)
    {
        step = Math.Clamp(step, 0, sigmas.Values.Length);
        var v = sigmas.Values;
        return (new Sigmas(v[..Math.Min(step + 1, v.Length)]), new Sigmas(v[step..]));
    }
}

[Node(Category = "sampling/custom_sampling/sigmas", DisplayName = "SplitSigmasDenoise")]
public sealed class SplitSigmasDenoise
{
    /// <summary>Splits so the low part holds the last <c>denoise</c> share of the steps (an image-to-image run uses it); the high part is what remains before it.</summary>
    public (Sigmas High, Sigmas Low) Execute(Sigmas sigmas, [Input(Min = 0, Max = 1, Step = 0.01)] float denoise = 1f)
    {
        var v = sigmas.Values;
        int keep = Math.Min((int)Math.Round((v.Length - 1) * denoise) + 1, v.Length);
        return (new Sigmas(v[..(v.Length - keep)]), new Sigmas(v[(v.Length - keep)..]));
    }
}

[Node(Category = "sampling/custom_sampling/sigmas", DisplayName = "FlipSigmas")]
public sealed class FlipSigmas
{
    /// <summary>Reverses the schedule (to run it upwards, for inversion); a leading 0 becomes a tiny sigma so the first step is defined.</summary>
    public Sigmas Execute(Sigmas sigmas)
    {
        if (sigmas.Values.Length == 0) return sigmas;
        var flipped = sigmas.Values.Reverse().ToArray();
        if (flipped[0] == 0f) flipped[0] = 1e-4f;
        return new Sigmas(flipped);
    }
}

[Node(Category = "sampling/custom_sampling/sigmas", DisplayName = "SetFirstSigma")]
public sealed class SetFirstSigma
{
    public Sigmas Execute(Sigmas sigmas, [Input(Min = 0, Max = 20000, Step = 0.001)] float sigma = 136f)
    {
        var v = (float[])sigmas.Values.Clone();
        if (v.Length > 0) v[0] = sigma;
        return new Sigmas(v);
    }
}

[Node(Category = "sampling/custom_sampling", DisplayName = "SamplerCustom")]
public sealed class SamplerCustom
{
    /// <summary>
    /// Sampling with the pieces chosen separately: a sampler, a sigma schedule, and (here, as in ComfyUI's node) classifier-free guidance with a cfg scale.
    /// <c>output</c> is the sampled latent, <c>denoised_output</c> the model's last clean estimate (they differ when the schedule ends above 0).
    /// </summary>
    public async Task<(LatentBatch Output, LatentBatch DenoisedOutput)> ExecuteAsync(
        ModelHandle model, Conditioning positive, Conditioning negative, SamplerChoice sampler, Sigmas sigmas, LatentBatch latentImage,
        NodeContext context, IServiceProvider services,
        bool addNoise = true,
        long noiseSeed = 0,
        [Input(Min = 0, Max = 100, Step = 0.1)] float cfg = 8f)
    {
        if (sigmas.Values.Length < 2) return (latentImage, latentImage);
        var acc = (await GpuHelpers.RuntimeAsync(services)).Accelerator;
        return await SamplingCore.RunWithDenoisedAsync(acc, model.Checkpoint, positive, negative, latentImage, sigmas.Values,
            new SamplingSettings(noiseSeed, cfg, sampler.Name, AddNoise: addNoise), context);
    }
}

/// <summary>The noise a custom sampling run starts from (ComfyUI's NOISE): a seed, or none.</summary>
[Socket("NOISE")]
public sealed record NoiseChoice(long Seed, bool Enabled);

/// <summary>How the model is steered at each step (ComfyUI's GUIDER): the model, what to aim for, what to avoid and how hard.</summary>
[Socket("GUIDER")]
public sealed record GuiderChoice(ModelHandle Model, Conditioning Positive, Conditioning Negative, float Cfg);

[Node(Category = "sampling/custom_sampling/noise", DisplayName = "RandomNoise")]
public sealed class RandomNoise
{
    public NoiseChoice Execute(long noiseSeed = 0) => new(noiseSeed, true);
}

[Node(Category = "sampling/custom_sampling/noise", DisplayName = "DisableNoise")]
public sealed class DisableNoise
{
    /// <summary>No fresh noise: the run continues from the latent it is given (what a second pass of a split schedule wants).</summary>
    public NoiseChoice Execute() => new(0, false);
}

[Node(Category = "sampling/custom_sampling/guiders", DisplayName = "BasicGuider")]
public sealed class BasicGuider
{
    /// <summary>Guidance with the prompt alone (no negative prompt, cfg 1): what distilled models like SD-Turbo and FLUX expect.</summary>
    public GuiderChoice Execute(ModelHandle model, Conditioning conditioning) => new(model, conditioning, conditioning, 1f);
}

[Node(Category = "sampling/custom_sampling/guiders", DisplayName = "CFGGuider")]
public sealed class CFGGuider
{
    public GuiderChoice Execute(ModelHandle model, Conditioning positive, Conditioning negative, [Input(Min = 0, Max = 100, Step = 0.1)] float cfg = 8f) =>
        new(model, positive, negative, cfg);
}

[Node(Category = "sampling/custom_sampling", DisplayName = "SamplerCustomAdvanced")]
public sealed class SamplerCustomAdvanced
{
    /// <summary>
    /// SamplerCustom with every piece a node of its own: noise, guider, sampler, sigmas. <c>output</c> is the sampled latent, <c>denoised_output</c> the model's last clean estimate.
    /// </summary>
    public async Task<(LatentBatch Output, LatentBatch DenoisedOutput)> ExecuteAsync(NoiseChoice noise, GuiderChoice guider, SamplerChoice sampler, Sigmas sigmas, LatentBatch latentImage,
        NodeContext context, IServiceProvider services)
    {
        if (sigmas.Values.Length < 2) return (latentImage, latentImage);
        var acc = (await GpuHelpers.RuntimeAsync(services)).Accelerator;
        return await SamplingCore.RunWithDenoisedAsync(acc, guider.Model.Checkpoint, guider.Positive, guider.Negative, latentImage, sigmas.Values,
            new SamplingSettings(noise.Seed, guider.Cfg, sampler.Name, AddNoise: noise.Enabled), context);
    }
}

[Node(Category = "sampling/custom_sampling/schedulers", DisplayName = "BetaSamplingScheduler")]
public sealed class BetaSamplingScheduler
{
    public Sigmas Execute(ModelHandle model, [Input(Min = 1, Max = 10000)] int steps = 20, [Input(Min = 0, Max = 50, Step = 0.01)] float alpha = 0.6f, [Input(Min = 0, Max = 50, Step = 0.01)] float beta = 0.6f) =>
        new(model.Checkpoint.Sigmas.Beta(steps, Math.Max(alpha, 0.001), Math.Max(beta, 0.001)));
}

[Node(Category = "sampling/custom_sampling/schedulers", DisplayName = "LinearQuadraticScheduler")]
public sealed class LinearQuadraticScheduler
{
    public Sigmas Execute(ModelHandle model, [Input(Min = 1, Max = 10000)] int steps = 20, [Input(Min = 0, Max = 1, Step = 0.001)] float thresholdNoise = 0.025f, [Input(Min = 0, Max = 10000)] int linearSteps = 0) =>
        new(model.Checkpoint.Sigmas.LinearQuadratic(steps, thresholdNoise, linearSteps > 0 ? linearSteps : null));
}
