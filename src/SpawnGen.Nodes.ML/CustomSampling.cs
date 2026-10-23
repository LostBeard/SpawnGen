using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

/// <summary>The noise levels a custom sampling run steps through (length steps + 1, ending in 0): ComfyUI's SIGMAS.</summary>
[Socket("SIGMAS")]
public sealed record Sigmas(float[] Values);

/// <summary>A sampler picked by name, with the settings a sampler node chose (ComfyUI's SAMPLER: a KSAMPLER with its extra_options).</summary>
[Socket("SAMPLER")]
public sealed record SamplerChoice(string Name, SamplerOptions? Options = null);

/// <summary>The sigma schedules of k-diffusion, from their formulas (a final 0 is appended like ComfyUI does).</summary>
public static class SigmaSchedules
{
    static IEnumerable<double> Ramp(int n, double from, double to) => ModelSigmas.Linspace(from, to, n);

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

    /// <summary>
    /// The Laplace schedule (Tiankai et al. 2024): log sigma = mu - beta * sign(0.5 - x) * log(1 - 2|0.5 - x| + 1e-5) for x from 0 to 1, clamped to [sigma_min, sigma_max].
    /// ComfyUI appends no final 0 here.
    /// </summary>
    public static float[] Laplace(int steps, double sigmaMax, double sigmaMin, double mu, double beta) =>
        [.. Ramp(steps, 0, 1).Select(x =>
        {
            double lmb = mu - beta * Math.Sign(0.5 - x) * Math.Log(1 - 2 * Math.Abs(0.5 - x) + 1e-5);
            return (float)Math.Clamp(Math.Exp(lmb), sigmaMin, sigmaMax);
        })];

    /// <summary>SD-Turbo's few-step schedule: the model's sigmas at timesteps 999, 899, ... 99, starting <c>10 - int(10 * denoise)</c> entries in.</summary>
    public static float[] SdTurbo(ModelSigmas model, int steps, double denoise)
    {
        int start = 10 - (int)(10 * denoise);
        var timesteps = Enumerable.Range(0, 10).Select(i => 999 - 100 * i).Skip(start).Take(steps);
        return WithZero(timesteps.Select(t => (double)model.Sigma(t)));
    }

    /// <summary>numpy-style log-linear interpolation of a decreasing schedule to <paramref name="count"/> points (Align Your Steps, GITS, OptimalSteps).</summary>
    public static double[] LogLinearInterp(double[] tSteps, int count)
    {
        int n = tSteps.Length;
        var ys = tSteps.Reverse().Select(v => Math.Log(v)).ToArray();
        var r = new double[count];
        for (int i = 0; i < count; i++)
        {
            double x = count == 1 ? 0 : (double)i / (count - 1), pos = x * (n - 1);
            int lo = Math.Min((int)Math.Floor(pos), n - 2);
            double y = n == 1 ? ys[0] : ys[lo] + (ys[lo + 1] - ys[lo]) * (pos - lo);
            r[count - 1 - i] = Math.Exp(y);
        }
        return r;
    }
}

[Node(Category = "model/sampling/samplers", DisplayName = "KSamplerSelect")]
public sealed class KSamplerSelect
{
    public static string[] SamplerNames() => Samplers.Names;

    public SamplerChoice Execute([Input(OptionsProvider = nameof(SamplerNames))] string samplerName = "euler") =>
        Samplers.Names.Contains(samplerName) ? new SamplerChoice(samplerName)
            : throw new ArgumentException($"Unknown sampler '{samplerName}'. Known: {string.Join(", ", Samplers.Names)}.", nameof(samplerName));
}

[Node(Category = "model/sampling/schedulers", DisplayName = "BasicScheduler")]
public sealed class BasicScheduler
{
    public static string[] SchedulerNames() => ModelSigmas.SchedulerNames;

    /// <summary>The sigmas of one of the model's own schedulers; with denoise below 1 only the tail of a longer schedule is used (as KSampler does).</summary>
    public Sigmas Execute(ModelHandle model, [Input(OptionsProvider = nameof(SchedulerNames))] string scheduler = "simple",
        [Input(Min = 1, Max = 10000)] int steps = 20, [Input(Min = 0, Max = 1, Step = 0.01)] double denoise = 1.0)
    {
        if (denoise <= 0) return new Sigmas([]);
        return new Sigmas(model.Sampling.Sigmas.ScheduleForDenoise(scheduler, steps, Math.Min(denoise, 1.0)));
    }
}

[Node(Category = "model/sampling/schedulers", DisplayName = "KarrasScheduler")]
public sealed class KarrasScheduler
{
    public Sigmas Execute([Input(Min = 1, Max = 10000)] int steps = 20, [Input(Min = 0, Max = 5000, Step = 0.01)] double sigmaMax = 14.614642,
        [Input(Min = 0, Max = 5000, Step = 0.01)] double sigmaMin = 0.0291675, [Input(Min = 0, Max = 100, Step = 0.01)] double rho = 7.0) =>
        new(SigmaSchedules.Karras(steps, sigmaMax, sigmaMin, rho));
}

[Node(Category = "model/sampling/schedulers", DisplayName = "ExponentialScheduler")]
public sealed class ExponentialScheduler
{
    public Sigmas Execute([Input(Min = 1, Max = 10000)] int steps = 20, [Input(Min = 0, Max = 5000, Step = 0.01)] double sigmaMax = 14.614642,
        [Input(Min = 0, Max = 5000, Step = 0.01)] double sigmaMin = 0.0291675) =>
        new(SigmaSchedules.Exponential(steps, sigmaMax, sigmaMin));
}

[Node(Category = "model/sampling/schedulers", DisplayName = "PolyexponentialScheduler")]
public sealed class PolyexponentialScheduler
{
    public Sigmas Execute([Input(Min = 1, Max = 10000)] int steps = 20, [Input(Min = 0, Max = 5000, Step = 0.01)] double sigmaMax = 14.614642,
        [Input(Min = 0, Max = 5000, Step = 0.01)] double sigmaMin = 0.0291675, [Input(Min = 0, Max = 100, Step = 0.01)] double rho = 1.0) =>
        new(SigmaSchedules.Polyexponential(steps, sigmaMax, sigmaMin, rho));
}

[Node(Category = "model/sampling/schedulers", DisplayName = "VPScheduler")]
public sealed class VPScheduler
{
    public Sigmas Execute([Input(Min = 1, Max = 10000)] int steps = 20, [Input(Min = 0, Max = 5000, Step = 0.01)] double betaD = 19.9,
        [Input(Min = 0, Max = 5000, Step = 0.01)] double betaMin = 0.1, [Input(Min = 0, Max = 1, Step = 0.0001)] double epsS = 0.001) =>
        new(SigmaSchedules.VariancePreserving(steps, betaD, betaMin, epsS));
}

[Node(Category = "model/sampling/schedulers", DisplayName = "SDTurboScheduler")]
public sealed class SDTurboScheduler
{
    public Sigmas Execute(ModelHandle model, [Input(Min = 1, Max = 10)] int steps = 1, [Input(Min = 0, Max = 1, Step = 0.01)] double denoise = 1.0) =>
        new(SigmaSchedules.SdTurbo(model.Sampling.Sigmas, steps, denoise));
}

[Node(Category = "model/sampling/sigmas", DisplayName = "SplitSigmas", OutputNames = ["high_sigmas", "low_sigmas"])]
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

[Node(Category = "model/sampling/sigmas", DisplayName = "SplitSigmasDenoise", OutputNames = ["high_sigmas", "low_sigmas"])]
public sealed class SplitSigmasDenoise
{
    /// <summary>Splits so the low part holds the last <c>round(steps * denoise)</c> steps (an image-to-image run uses it); the high part is what remains before it.</summary>
    public (Sigmas High, Sigmas Low) Execute(Sigmas sigmas, [Input(Min = 0, Max = 1, Step = 0.01)] double denoise = 1.0)
    {
        var v = sigmas.Values;
        int steps = Math.Max(v.Length - 1, 0), total = (int)Math.Round(steps * denoise, MidpointRounding.ToEven);
        // Python slicing: sigmas[:-0] is empty, and a negative start past the beginning is the beginning.
        var high = total == 0 ? [] : v[..Math.Max(0, v.Length - total)];
        var low = v[Math.Max(0, v.Length - (total + 1))..];
        return (new Sigmas(high), new Sigmas(low));
    }
}

[Node(Category = "model/sampling/sigmas", DisplayName = "FlipSigmas")]
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

[Node(Category = "model/sampling/sigmas", DisplayName = "SetFirstSigma")]
public sealed class SetFirstSigma
{
    public Sigmas Execute(Sigmas sigmas, [Input(Min = 0, Max = 20000, Step = 0.001)] double sigma = 136.0)
    {
        var v = (float[])sigmas.Values.Clone();
        if (v.Length > 0) v[0] = (float)sigma;
        return new Sigmas(v);
    }
}

[Node(Category = "model/sampling/custom", DisplayName = "SamplerCustom", OutputNames = ["output", "denoised_output"])]
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
        [Input(Min = 0, Max = 100, Step = 0.1)] double cfg = 8.0)
    {
        if (sigmas.Values.Length == 0) return (latentImage, latentImage);
        var acc = (await GpuHelpers.RuntimeAsync(services)).Accelerator;
        return await SamplingCore.RunWithDenoisedAsync(acc, new CfgGuider(model, positive, negative, cfg), latentImage, sigmas.Values,
            new SamplingSettings(noiseSeed, sampler.Name, AddNoise: addNoise, Options: sampler.Options), context);
    }
}

/// <summary>The noise a custom sampling run starts from (ComfyUI's NOISE): a seed, or none.</summary>
[Socket("NOISE")]
public sealed record NoiseChoice(long Seed, bool Enabled);

[Node(Category = "model/sampling/noise", DisplayName = "RandomNoise")]
public sealed class RandomNoise
{
    public NoiseChoice Execute(long noiseSeed = 0) => new(noiseSeed, true);
}

[Node(Category = "model/sampling/noise", DisplayName = "DisableNoise")]
public sealed class DisableNoise
{
    /// <summary>No fresh noise: the run continues from the latent it is given (what a second pass of a split schedule wants).</summary>
    public NoiseChoice Execute() => new(0, false);
}

[Node(Category = "model/sampling/guiders", DisplayName = "Basic Guider")]
public sealed class BasicGuider
{
    /// <summary>Guidance with the prompt alone (no negative prompt, cfg 1): what distilled models like SD-Turbo and FLUX expect.</summary>
    public GuiderChoice Execute(ModelHandle model, Conditioning conditioning) => new BasicGuiderChoice(model, conditioning);
}

[Node(Category = "model/sampling/guiders", DisplayName = "CFG Guider")]
public sealed class CFGGuider
{
    public GuiderChoice Execute(ModelHandle model, Conditioning positive, Conditioning negative, [Input(Min = 0, Max = 100, Step = 0.1)] double cfg = 8.0) =>
        new CfgGuider(model, positive, negative, cfg);
}

[Node(Category = "model/sampling/custom", DisplayName = "SamplerCustomAdvanced", OutputNames = ["output", "denoised_output"])]
public sealed class SamplerCustomAdvanced
{
    /// <summary>
    /// SamplerCustom with every piece a node of its own: noise, guider, sampler, sigmas. <c>output</c> is the sampled latent, <c>denoised_output</c> the model's last clean estimate.
    /// </summary>
    public async Task<(LatentBatch Output, LatentBatch DenoisedOutput)> ExecuteAsync(NoiseChoice noise, GuiderChoice guider, SamplerChoice sampler, Sigmas sigmas, LatentBatch latentImage,
        NodeContext context, IServiceProvider services)
    {
        if (sigmas.Values.Length == 0) return (latentImage, latentImage);
        var acc = (await GpuHelpers.RuntimeAsync(services)).Accelerator;
        return await SamplingCore.RunWithDenoisedAsync(acc, guider, latentImage, sigmas.Values,
            new SamplingSettings(noise.Seed, sampler.Name, AddNoise: noise.Enabled, Options: sampler.Options), context);
    }
}

[Node(Category = "model/sampling/schedulers", DisplayName = "BetaSamplingScheduler")]
public sealed class BetaSamplingScheduler
{
    public Sigmas Execute(ModelHandle model, [Input(Min = 1, Max = 10000)] int steps = 20, [Input(Min = 0, Max = 50, Step = 0.01)] double alpha = 0.6, [Input(Min = 0, Max = 50, Step = 0.01)] double beta = 0.6) =>
        alpha > 0 && beta > 0 ? new(model.Sampling.Sigmas.Beta(steps, alpha, beta))
            : throw new ArgumentException("alpha and beta must be above 0 (the Beta distribution's quantiles are undefined otherwise, and ComfyUI fails there too).");
}

/// <summary>Not a ComfyUI node any more (its schedule is the "linear_quadratic" scheduler); kept so SpawnGen workflows that use it still open.</summary>
[Node(Category = "model/sampling/schedulers", DisplayName = "LinearQuadraticScheduler")]
public sealed class LinearQuadraticScheduler
{
    public Sigmas Execute(ModelHandle model, [Input(Min = 1, Max = 10000)] int steps = 20, [Input(Min = 0, Max = 1, Step = 0.001)] double thresholdNoise = 0.025, [Input(Min = 0, Max = 10000)] int linearSteps = 0) =>
        new(model.Sampling.Sigmas.LinearQuadratic(steps, thresholdNoise, linearSteps > 0 ? linearSteps : null));
}
