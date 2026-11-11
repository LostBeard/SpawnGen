using System.Globalization;
using System.Text.RegularExpressions;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries and Fixtures/sampling/reference.json ("tables": every scheduler and
// sigma node against ComfyUI's values). Sources: k-diffusion (MIT) for the Karras, exponential, polyexponential and VP schedules; NVIDIA's Align
// Your Steps tables and log-linear interpolation; the Optimal Steps tables; GITS (measured, GitsTable); the Laplace schedule of Hang et al. 2024.

/// <summary>SIGMAS: a run's noise levels, from the most noise down.</summary>
[Socket("SIGMAS")]
public sealed record Sigmas(float[] Values);

/// <summary>SAMPLER: a sampler by name, with its options.</summary>
[Socket("SAMPLER")]
public sealed record SamplerChoice(string Name, SamplerOptions? Options = null);

/// <summary>The scheduler node formulas.</summary>
public static class SigmaSchedules
{
    static float[] WithZero(IEnumerable<double> v) => [.. v.Select(x => (float)x), 0f];

    public static float[] Karras(int steps, double sigmaMax, double sigmaMin, double rho) => ModelSigmas.Karras(steps, sigmaMin, sigmaMax, rho);

    public static float[] Exponential(int steps, double sigmaMax, double sigmaMin) => ModelSigmas.Exponential(steps, sigmaMin, sigmaMax);

    /// <summary>k-diffusion's polyexponential: log sigma moves from max to min along (1 - x)^rho.</summary>
    public static float[] Polyexponential(int steps, double sigmaMax, double sigmaMin, double rho)
    {
        double lmax = Math.Log(sigmaMax), lmin = Math.Log(sigmaMin);
        return WithZero(ModelSigmas.Linspace(1, 0, steps).Select(r => Math.Exp(Math.Pow(r, rho) * (lmax - lmin) + lmin)));
    }

    /// <summary>k-diffusion's variance-preserving schedule: sigma(t) = sqrt(exp(beta_d t^2 / 2 + beta_min t) - 1), t from 1 down to eps_s.</summary>
    public static float[] VariancePreserving(int steps, double betaD, double betaMin, double epsS) =>
        WithZero(ModelSigmas.Linspace(1, epsS, steps).Select(t => Math.Sqrt(Math.Exp(betaD * t * t / 2 + betaMin * t) - 1)));

    /// <summary>
    /// The Laplace schedule (Hang et al. 2024): log sigma = mu - beta sign(0.5 - x) log(1 - 2 |0.5 - x|), clamped to the range; <paramref name="steps"/>
    /// values with no 0 appended (measured).
    /// </summary>
    public static float[] Laplace(int steps, double sigmaMax, double sigmaMin, double mu, double beta)
    {
        const double eps = 1e-5;
        return [.. ModelSigmas.Linspace(0, 1, steps).Select(x =>
        {
            double d = 0.5 - x;
            double lmb = mu - beta * Math.Sign(d) * Math.Log(1 - 2 * Math.Abs(d) + eps);
            return (float)Math.Clamp(Math.Exp(lmb), sigmaMin, sigmaMax);
        })];
    }

    /// <summary>SD-Turbo's timesteps 999, 899, ..., 99, from the one the denoise starts at, as sigmas.</summary>
    public static float[] SdTurbo(ModelSigmas model, int steps, double denoise)
    {
        int start = 10 - (int)(10 * denoise);
        var timesteps = Enumerable.Range(1, 10).Reverse().Select(i => i * 100 - 1).Skip(start).Take(steps);
        return WithZero(timesteps.Select(t => (double)model.Sigma(t)));
    }

    /// <summary>Align Your Steps' log-linear interpolation of a (falling) table to <paramref name="count"/> values.</summary>
    public static double[] LogLinearInterp(double[] tSteps, int count)
    {
        var ys = tSteps.Reverse().Select(v => Math.Log(v)).ToArray();
        var xs = ModelSigmas.Linspace(0, 1, ys.Length);
        var nx = ModelSigmas.Linspace(0, 1, count);
        var r = new double[count];
        for (int i = 0; i < count; i++)
        {
            double x = nx[i];
            int j = Math.Min(Math.Max(0, Array.BinarySearch(xs, x) is var k && k >= 0 ? k : ~k - 1), ys.Length - 2);
            double t = (x - xs[j]) / (xs[j + 1] - xs[j]);
            r[i] = Math.Exp(ys[j] + (ys[j + 1] - ys[j]) * t);
        }
        Array.Reverse(r);
        return r;
    }
}

/// <summary>The table schedulers' data and their denoise rule.</summary>
public static class TableSchedules
{
    /// <summary>NVIDIA's Align Your Steps 10-step tables (SD 1.5, SDXL, SVD).</summary>
    public static readonly Dictionary<string, double[]> AlignYourSteps = new()
    {
        ["SD1"] = [14.6146412293, 6.4745760956, 3.8636745985, 2.694615152, 1.8841921177, 1.3943805092, 0.9642583904, 0.6523686016, 0.3977456272, 0.1515232662, 0.0291671582],
        ["SDXL"] = [14.6146412293, 6.3184485287, 3.7681790315, 2.1811480769, 1.3405244945, 0.8620721141, 0.5550693289, 0.3798540708, 0.2332364134, 0.1114188177, 0.0291671582],
        ["SVD"] = [700, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.002],
    };

    /// <summary>The Optimal Steps tables (FLUX, Wan, Chroma), fractions of the noise.</summary>
    public static readonly Dictionary<string, double[]> OptimalSteps = new()
    {
        ["FLUX"] = [0.9968, 0.9886, 0.9819, 0.975, 0.966, 0.9471, 0.9158, 0.8287, 0.5512, 0.2808, 0.001],
        ["Wan"] = [1, 0.997, 0.995, 0.993, 0.991, 0.989, 0.987, 0.985, 0.98, 0.975, 0.973, 0.968, 0.96, 0.946, 0.927, 0.902, 0.864, 0.776, 0.539, 0.208, 0.001],
        ["Chroma"] = [0.992, 0.99, 0.988, 0.985, 0.982, 0.978, 0.973, 0.968, 0.961, 0.953, 0.943, 0.931, 0.917, 0.9, 0.881, 0.858, 0.832, 0.802, 0.769, 0.731, 0.69, 0.646, 0.599, 0.55, 0.501, 0.451, 0.402, 0.355, 0.311, 0.27, 0.232, 0.199, 0.169, 0.143, 0.12, 0.101, 0.084, 0.07, 0.058, 0.048, 0.001],
    };

    /// <summary>
    /// A table schedule for <paramref name="steps"/> at <paramref name="denoise"/> (measured): the table for <paramref name="steps"/> steps (log-linearly
    /// interpolated when its length differs), its last round(steps x denoise) + 1 values, the last set to 0; nothing at denoise 0.
    /// </summary>
    public static float[] Tail(double[] table, int steps, double denoise)
    {
        if (denoise <= 0) return [];
        var s = table.Length == steps + 1 ? (double[])table.Clone() : SigmaSchedules.LogLinearInterp(table, steps + 1);
        return Last(s, steps, denoise);
    }

    /// <summary>The last round(steps x denoise) + 1 values of a steps + 1 schedule, the last set to 0.</summary>
    internal static float[] Last(double[] schedule, int steps, double denoise)
    {
        int keep = denoise < 1 ? (int)Math.Round(steps * denoise, MidpointRounding.ToEven) : steps;
        var r = schedule[^(keep + 1)..].Select(v => (float)v).ToArray();
        r[^1] = 0;
        return r;
    }

    /// <summary>GITS' sigmas for a coeff (the nearest measured one): 2 to 20 steps.</summary>
    public static double[][] Gits(double coeff) =>
        GitsTable.Levels.OrderBy(l => Math.Abs(l.Coeff - coeff)).First().Levels;
}

[Node("BasicScheduler", Category = "model/sampling/schedulers", DisplayName = "BasicScheduler")]
public sealed class BasicScheduler
{
    public static string[] SchedulerNames() => ModelSigmas.SchedulerNames;

    public Sigmas Execute(ModelHandle model, [Input(OptionsProvider = nameof(SchedulerNames))] string scheduler = "simple",
        [Input(Min = 1, Max = 10000)] int steps = 20, [Input(Min = 0.0, Max = 1.0, Step = 0.01)] double denoise = 1.0)
    {
        if (denoise <= 0) return new Sigmas([]);
        var s = model.Sampling.Sigmas;
        if (denoise >= 1) return new Sigmas(s.Schedule(scheduler, steps));
        var full = s.Schedule(scheduler, (int)(steps / denoise));
        return new Sigmas(full[Math.Max(0, full.Length - (steps + 1))..]);
    }
}

[Node("KarrasScheduler", Category = "model/sampling/schedulers", DisplayName = "KarrasScheduler")]
public sealed class KarrasScheduler
{
    public Sigmas Execute([Input(Min = 1, Max = 10000)] int steps = 20, [Input(Min = 0.0, Max = 5000.0, Step = 0.01)] double sigmaMax = 14.614642,
        [Input(Min = 0.0, Max = 5000.0, Step = 0.01)] double sigmaMin = 0.0291675, [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double rho = 7.0) =>
        new(SigmaSchedules.Karras(steps, sigmaMax, sigmaMin, rho));
}

[Node("ExponentialScheduler", Category = "model/sampling/schedulers", DisplayName = "ExponentialScheduler")]
public sealed class ExponentialScheduler
{
    public Sigmas Execute([Input(Min = 1, Max = 10000)] int steps = 20, [Input(Min = 0.0, Max = 5000.0, Step = 0.01)] double sigmaMax = 14.614642,
        [Input(Min = 0.0, Max = 5000.0, Step = 0.01)] double sigmaMin = 0.0291675) =>
        new(SigmaSchedules.Exponential(steps, sigmaMax, sigmaMin));
}

[Node("PolyexponentialScheduler", Category = "model/sampling/schedulers", DisplayName = "PolyexponentialScheduler")]
public sealed class PolyexponentialScheduler
{
    public Sigmas Execute([Input(Min = 1, Max = 10000)] int steps = 20, [Input(Min = 0.0, Max = 5000.0, Step = 0.01)] double sigmaMax = 14.614642,
        [Input(Min = 0.0, Max = 5000.0, Step = 0.01)] double sigmaMin = 0.0291675, [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double rho = 1.0) =>
        new(SigmaSchedules.Polyexponential(steps, sigmaMax, sigmaMin, rho));
}

[Node("LaplaceScheduler", Category = "model/sampling/schedulers", DisplayName = "LaplaceScheduler")]
public sealed class LaplaceScheduler
{
    public Sigmas Execute([Input(Min = 1, Max = 10000)] int steps = 20, [Input(Min = 0.0, Max = 5000.0, Step = 0.01)] double sigmaMax = 14.614642,
        [Input(Min = 0.0, Max = 5000.0, Step = 0.01)] double sigmaMin = 0.0291675, [Input(Min = -10.0, Max = 10.0, Step = 0.1)] double mu = 0.0,
        [Input(Min = 0.0, Max = 10.0, Step = 0.1)] double beta = 0.5) =>
        new(SigmaSchedules.Laplace(steps, sigmaMax, sigmaMin, mu, beta));
}

[Node("VPScheduler", Category = "model/sampling/schedulers", DisplayName = "VPScheduler")]
public sealed class VPScheduler
{
    public Sigmas Execute([Input(Min = 1, Max = 10000)] int steps = 20, [Input(Min = 0.0, Max = 5000.0, Step = 0.01)] double betaD = 19.9,
        [Input(Min = 0.0, Max = 5000.0, Step = 0.01)] double betaMin = 0.1, [Input(Min = 0.0, Max = 1.0, Step = 0.0001)] double epsS = 0.001) =>
        new(SigmaSchedules.VariancePreserving(steps, betaD, betaMin, epsS));
}

[Node("BetaSamplingScheduler", Category = "model/sampling/schedulers", DisplayName = "BetaSamplingScheduler")]
public sealed class BetaSamplingScheduler
{
    public Sigmas Execute(ModelHandle model, [Input(Min = 1, Max = 10000)] int steps = 20,
        [Input(Min = 0.0, Max = 50.0, Step = 0.01)] double alpha = 0.6, [Input(Min = 0.0, Max = 50.0, Step = 0.01)] double beta = 0.6) =>
        new(model.Sampling.Sigmas.Beta(steps, alpha, beta));
}

[Node("LinearQuadraticScheduler", Category = "model/sampling/schedulers", DisplayName = "LinearQuadraticScheduler")]
public sealed class LinearQuadraticScheduler
{
    public Sigmas Execute(ModelHandle model, [Input(Min = 1, Max = 10000)] int steps = 20,
        [Input(Min = 0.0, Max = 1.0, Step = 0.001)] double thresholdNoise = 0.025, [Input(Min = 0, Max = 10000)] int linearSteps = 0) =>
        new(model.Sampling.Sigmas.LinearQuadratic(steps, thresholdNoise, linearSteps == 0 ? null : linearSteps));
}

[Node("SDTurboScheduler", Category = "model/sampling/schedulers", DisplayName = "SDTurboScheduler")]
public sealed class SDTurboScheduler
{
    public Sigmas Execute(ModelHandle model, [Input(Min = 1, Max = 10)] int steps = 1, [Input(Min = 0.0, Max = 1.0, Step = 0.01)] double denoise = 1.0) =>
        new(SigmaSchedules.SdTurbo(model.Sampling.Sigmas, steps, denoise));
}

[Node("AlignYourStepsScheduler", Category = "model/sampling/schedulers", DisplayName = "AlignYourStepsScheduler")]
public sealed class AlignYourStepsScheduler
{
    public Sigmas Execute([Input(Options = ["SD1", "SDXL", "SVD"])] string modelType = "SD1", [Input(Min = 1, Max = 10000)] int steps = 10,
        [Input(Min = 0.0, Max = 1.0, Step = 0.01)] double denoise = 1.0) =>
        new(TableSchedules.Tail(TableSchedules.AlignYourSteps[modelType], steps, denoise));
}

[Node("OptimalStepsScheduler", Category = "model/sampling/schedulers", DisplayName = "OptimalStepsScheduler")]
public sealed class OptimalStepsScheduler
{
    public Sigmas Execute([Input(Options = ["FLUX", "Wan", "Chroma"])] string modelType = "FLUX", [Input(Min = 3, Max = 1000)] int steps = 20,
        [Input(Min = 0.0, Max = 1.0, Step = 0.01)] double denoise = 1.0) =>
        new(TableSchedules.Tail(TableSchedules.OptimalSteps[modelType], steps, denoise));
}

[Node("GITSScheduler", Category = "model/sampling/schedulers", DisplayName = "GITSScheduler")]
public sealed class GITSScheduler
{
    /// <summary>SD 1.x's lowest sigma, which GITS' rows end at before the last is set to 0 (measured from the interpolated schedules).</summary>
    const double RowEnd = 0.0291675;

    /// <summary>
    /// GITS' measured sigmas for up to 20 steps; for more, its 20-step row (ending at <see cref="RowEnd"/>) interpolated log-linearly; then the
    /// table schedulers' denoise rule.
    /// </summary>
    public Sigmas Execute([Input(Min = 0.8, Max = 1.5, Step = 0.05)] double coeff = 1.2, [Input(Min = 2, Max = 1000)] int steps = 10,
        [Input(Min = 0.0, Max = 1.0, Step = 0.01)] double denoise = 1.0)
    {
        if (denoise <= 0) return new Sigmas([]);
        var rows = TableSchedules.Gits(coeff);
        double[] s;
        if (steps <= 20) s = rows[steps - 2];
        else
        {
            var last = (double[])rows[^1].Clone();
            last[^1] = RowEnd;
            s = SigmaSchedules.LogLinearInterp(last, steps + 1);
        }
        return new Sigmas(TableSchedules.Last(s, steps, denoise));
    }
}

[Node("ExtendIntermediateSigmas", Category = "model/sampling/sigmas", DisplayName = "ExtendIntermediateSigmas")]
public sealed class ExtendIntermediateSigmas
{
    /// <summary>
    /// Each pair of neighbours whose upper sigma is in [end_at_sigma, start_at_sigma] (start -1: no upper limit) split into <paramref name="steps"/>
    /// parts, spaced linearly or along a curve; the fractions from the upper sigma (measured): "cosine" sin(x pi / 2), "sine" 1 - cos(x pi / 2).
    /// </summary>
    public Sigmas Execute(Sigmas sigmas, [Input(Min = 1, Max = 100)] int steps = 2,
        [Input(Min = -1.0, Max = 20000.0, Step = 0.01)] double startAtSigma = -1.0, [Input(Min = 0.0, Max = 20000.0, Step = 0.01)] double endAtSigma = 12.0,
        [Input(Options = ["linear", "cosine", "sine"])] string spacing = "linear")
    {
        double start = startAtSigma < 0 ? double.PositiveInfinity : startAtSigma;
        var points = ModelSigmas.Linspace(0, 1, steps + 1)[1..^1].Select(x => spacing switch
        {
            "cosine" => Math.Sin(x * Math.PI / 2),
            "sine" => 1 - Math.Cos(x * Math.PI / 2),
            _ => x,
        }).ToArray();
        var v = sigmas.Values;
        var r = new List<float>();
        for (int i = 0; i < v.Length; i++)
        {
            r.Add(v[i]);
            if (i + 1 >= v.Length) break;
            float hi = v[i], lo = v[i + 1];
            if (hi <= start && hi >= endAtSigma)
                foreach (var p in points) r.Add((float)(hi + (lo - hi) * p));
        }
        return new Sigmas([.. r]);
    }
}

[Node("FlipSigmas", Category = "model/sampling/sigmas", DisplayName = "FlipSigmas")]
public sealed class FlipSigmas
{
    /// <summary>The sigmas in reverse order; a leading 0 becomes 0.0001.</summary>
    public Sigmas Execute(Sigmas sigmas)
    {
        if (sigmas.Values.Length == 0) return sigmas;
        var r = sigmas.Values.Reverse().ToArray();
        if (r[0] == 0) r[0] = 0.0001f;
        return new Sigmas(r);
    }
}

[Node("SetFirstSigma", Category = "model/sampling/sigmas", DisplayName = "SetFirstSigma")]
public sealed class SetFirstSigma
{
    public Sigmas Execute(Sigmas sigmas, [Input(Min = 0.0, Max = 20000.0, Step = 0.001)] double sigma = 136.0)
    {
        var r = (float[])sigmas.Values.Clone();
        if (r.Length > 0) r[0] = (float)sigma;
        return new Sigmas(r);
    }
}

[Node("SplitSigmas", Category = "model/sampling/sigmas", DisplayName = "SplitSigmas", OutputNames = ["high_sigmas", "low_sigmas"])]
public sealed class SplitSigmas
{
    /// <summary>The sigmas up to and including index <paramref name="step"/>, and from it on.</summary>
    public (Sigmas High, Sigmas Low) Execute(Sigmas sigmas, [Input(Min = 0, Max = 10000)] int step = 0)
    {
        var v = sigmas.Values;
        int s = Math.Min(step, v.Length);
        return (new Sigmas(v[..Math.Min(s + 1, v.Length)]), new Sigmas(v[s..]));
    }
}

[Node("SplitSigmasDenoise", Category = "model/sampling/sigmas", DisplayName = "SplitSigmasDenoise", OutputNames = ["high_sigmas", "low_sigmas"])]
public sealed class SplitSigmasDenoise
{
    /// <summary>
    /// The last round(steps x denoise) steps as the low part and the sigmas before its last step as the high part (measured: with no low steps, no
    /// high sigmas and the last sigma as the low part).
    /// </summary>
    public (Sigmas High, Sigmas Low) Execute(Sigmas sigmas, [Input(Min = 0.0, Max = 1.0, Step = 0.01)] double denoise = 1.0)
    {
        var v = sigmas.Values;
        int steps = Math.Max(v.Length - 1, 0);
        int low = (int)Math.Round(steps * denoise, MidpointRounding.ToEven);
        float[] high = low == 0 ? [] : v[..Math.Max(v.Length - low, 0)];
        return (new Sigmas(high), new Sigmas(v[Math.Max(v.Length - low - 1, 0)..]));
    }
}

[Node("ManualSigmas", Category = "model/sampling/sigmas", DisplayName = "ManualSigmas")]
public sealed partial class ManualSigmas
{
    [GeneratedRegex(@"[-+]?(?:\d+\.?\d*|\.\d+)")]
    private static partial Regex Number();

    /// <summary>The plain decimal numbers in the text, in order (measured: no exponents, so "1e-3" reads as 1 and -3); none gives no sigmas.</summary>
    public Sigmas Execute(string sigmas = "1, 0.5") =>
        new([.. Number().Matches(sigmas).Select(m => float.Parse(m.Value, NumberStyles.Float, CultureInfo.InvariantCulture))]);
}

[Node("SamplingPercentToSigma", Category = "model/sampling/sigmas", DisplayName = "SamplingPercentToSigma", OutputNames = ["sigma_value"])]
public sealed class SamplingPercentToSigma
{
    /// <summary>The sigma at a fraction of the run; at 0 and 1 the model's own highest and lowest sigma when asked for the actual value.</summary>
    public double Execute(ModelHandle model, [Input(Min = 0.0, Max = 1.0, Step = 0.0001)] double samplingPercent = 0.0,
        [Input(Tooltip = "Return the actual sigma value instead of the value used for interval checks.\nThis only affects results at 0.0 and 1.0.")] bool returnActualSigma = false)
    {
        var s = model.Sampling;
        if (returnActualSigma && samplingPercent <= 0) return s.SigmaMax;
        if (returnActualSigma && samplingPercent >= 1) return s.SigmaMin;
        return s.PercentToSigma(samplingPercent);
    }
}

[Node("KSamplerSelect", Category = "model/sampling/samplers", DisplayName = "KSamplerSelect")]
public sealed class KSamplerSelect
{
    public static string[] SamplerNames() => KSampler.SamplerNames();

    public SamplerChoice Execute([Input(OptionsProvider = nameof(SamplerNames))] string samplerName = "euler") =>
        Samplers.Names.Contains(samplerName) ? new(samplerName) : throw new ArgumentException($"Unknown sampler '{samplerName}'.", nameof(samplerName));
}

[Node("SamplerEulerAncestral", Category = "model/sampling/samplers", DisplayName = "SamplerEulerAncestral")]
public sealed class SamplerEulerAncestral
{
    public SamplerChoice Execute([Input(Min = 0.0, Max = 100.0, Step = 0.01)] double eta = 1.0, [Input(Name = "s_noise", Min = 0.0, Max = 100.0, Step = 0.01)] double sNoise = 1.0) =>
        new("euler_ancestral", new SamplerOptions { Eta = eta, SNoise = sNoise });
}

[Node("SamplerEulerAncestralCFGPP", Category = "model/sampling/samplers", DisplayName = "SamplerEulerAncestralCFG++")]
public sealed class SamplerEulerAncestralCFGPP
{
    public SamplerChoice Execute([Input(Min = 0.0, Max = 1.0, Step = 0.01)] double eta = 1.0, [Input(Name = "s_noise", Min = 0.0, Max = 10.0, Step = 0.01)] double sNoise = 1.0) =>
        new("euler_ancestral_cfg_pp", new SamplerOptions { Eta = eta, SNoise = sNoise });
}

[Node("SamplerEulerCFGpp", Category = "experimental", DisplayName = "SamplerEulerCFG++")]
public sealed class SamplerEulerCFGpp
{
    /// <summary>"regular" is euler_cfg_pp; "alternative" the other CFG++ Euler form, <see cref="Samplers.EulerPP"/>.</summary>
    public SamplerChoice Execute([Input(Options = ["regular", "alternative"])] string version = "regular") =>
        new(version == "alternative" ? Samplers.EulerPP : "euler_cfg_pp");
}

[Node("SamplerLMS", Category = "model/sampling/samplers", DisplayName = "SamplerLMS")]
public sealed class SamplerLMS
{
    public SamplerChoice Execute([Input(Min = 1, Max = 100)] int order = 4) => new("lms", new SamplerOptions { Order = order });
}

[Node("SamplerDPMPP_2S_Ancestral", Category = "model/sampling/samplers", DisplayName = "SamplerDPMPP_2S_Ancestral")]
public sealed class SamplerDPMPP_2S_Ancestral
{
    public SamplerChoice Execute([Input(Min = 0.0, Max = 100.0, Step = 0.01)] double eta = 1.0, [Input(Name = "s_noise", Min = 0.0, Max = 100.0, Step = 0.01)] double sNoise = 1.0) =>
        new("dpmpp_2s_ancestral", new SamplerOptions { Eta = eta, SNoise = sNoise });
}

[Node("SamplerDPMPP_SDE", Category = "model/sampling/samplers", DisplayName = "SamplerDPMPP_SDE")]
public sealed class SamplerDPMPP_SDE
{
    public SamplerChoice Execute([Input(Min = 0.0, Max = 100.0, Step = 0.01)] double eta = 1.0, [Input(Name = "s_noise", Min = 0.0, Max = 100.0, Step = 0.01)] double sNoise = 1.0,
        [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double r = 0.5, [Input(Options = ["gpu", "cpu"])] string noiseDevice = "gpu") =>
        new(noiseDevice == "cpu" ? "dpmpp_sde" : "dpmpp_sde_gpu", new SamplerOptions { Eta = eta, SNoise = sNoise, R = r });
}

[Node("SamplerDPMPP_2M_SDE", Category = "model/sampling/samplers", DisplayName = "SamplerDPMPP_2M_SDE")]
public sealed class SamplerDPMPP_2M_SDE
{
    public SamplerChoice Execute([Input(Options = ["midpoint", "heun"])] string solverType = "midpoint", [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double eta = 1.0,
        [Input(Name = "s_noise", Min = 0.0, Max = 100.0, Step = 0.01)] double sNoise = 1.0, [Input(Options = ["gpu", "cpu"])] string noiseDevice = "gpu") =>
        new(noiseDevice == "cpu" ? "dpmpp_2m_sde" : "dpmpp_2m_sde_gpu", new SamplerOptions { SolverType = solverType, Eta = eta, SNoise = sNoise });
}

[Node("SamplerDPMPP_3M_SDE", Category = "model/sampling/samplers", DisplayName = "SamplerDPMPP_3M_SDE")]
public sealed class SamplerDPMPP_3M_SDE
{
    public SamplerChoice Execute([Input(Min = 0.0, Max = 100.0, Step = 0.01)] double eta = 1.0, [Input(Name = "s_noise", Min = 0.0, Max = 100.0, Step = 0.01)] double sNoise = 1.0,
        [Input(Options = ["gpu", "cpu"])] string noiseDevice = "gpu") =>
        new(noiseDevice == "cpu" ? "dpmpp_3m_sde" : "dpmpp_3m_sde_gpu", new SamplerOptions { Eta = eta, SNoise = sNoise });
}

[Node("SamplerDPMAdaptative", Category = "model/sampling/samplers", DisplayName = "SamplerDPMAdaptative")]
public sealed class SamplerDPMAdaptative
{
    public SamplerChoice Execute([Input(Min = 2, Max = 3)] int order = 3, [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double rtol = 0.05,
        [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double atol = 0.0078, [Input(Name = "h_init", Min = 0.0, Max = 100.0, Step = 0.01)] double hInit = 0.05,
        [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double pcoeff = 0.0, [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double icoeff = 1.0,
        [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double dcoeff = 0.0, [Input(Name = "accept_safety", Min = 0.0, Max = 100.0, Step = 0.01)] double acceptSafety = 0.81,
        [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double eta = 0.0, [Input(Name = "s_noise", Min = 0.0, Max = 100.0, Step = 0.01)] double sNoise = 1.0) =>
        new("dpm_adaptive", new SamplerOptions
        {
            Order = order, Rtol = rtol, Atol = atol, HInit = hInit, PCoeff = pcoeff, ICoeff = icoeff, DCoeff = dcoeff, AcceptSafety = acceptSafety, Eta = eta, SNoise = sNoise,
        });
}

[Node("SamplerER_SDE", Category = "model/sampling/samplers", DisplayName = "SamplerER_SDE")]
public sealed class SamplerER_SDE
{
    /// <summary>
    /// ER-SDE-Solver (Cui et al. 2023). The noise scaler is x * g(x)^eta, g being the paper's ER-SDE choice exp(x^0.3) + 10 or, for the
    /// reverse-time SDE, x; at eta 0 (or for the ODE) it is x itself and there is no noise.
    /// </summary>
    public SamplerChoice Execute([Input(Options = ["ER-SDE", "Reverse-time SDE", "ODE"])] string solverType = "ER-SDE",
        [Input(Min = 1, Max = 3)] int maxStage = 3,
        [Input(Min = 0.0, Max = 10.0, Step = 0.01, Tooltip = "Stochastic strength of SDEs.\nWhen eta=0, they reduce to deterministic ODE.\nLarge eta may cause invalid outputs. If this occurs, try decreasing this value.")] double eta = 1.0,
        [Input(Name = "s_noise", Min = 0.0, Max = 100.0, Step = 0.01)] double sNoise = 1.0)
    {
        bool ode = solverType == "ODE" || eta == 0;
        Func<double, double> scaler = ode ? x => x
            : solverType == "Reverse-time SDE" ? x => x * Math.Pow(x, eta)
            : x => x * Math.Pow(Math.Exp(Math.Pow(x, 0.3)) + 10, eta);
        return new("er_sde", new SamplerOptions { MaxStage = maxStage, SNoise = ode ? 0 : sNoise, NoiseScaler = scaler });
    }
}

[Node("SamplerSASolver", Category = "model/sampling/samplers", DisplayName = "SamplerSASolver")]
public sealed class SamplerSASolver
{
    /// <summary>SA-Solver (Xue et al. 2023): stochastic (tau = eta) between the two fractions of the run, deterministic elsewhere.</summary>
    public SamplerChoice Execute(ModelHandle model, [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double eta = 1.0,
        [Input(Min = 0.0, Max = 1.0, Step = 0.001)] double sdeStartPercent = 0.2, [Input(Min = 0.0, Max = 1.0, Step = 0.001)] double sdeEndPercent = 0.8,
        [Input(Name = "s_noise", Min = 0.0, Max = 100.0, Step = 0.01)] double sNoise = 1.0, [Input(Min = 1, Max = 6)] int predictorOrder = 3,
        [Input(Min = 0, Max = 6)] int correctorOrder = 4, bool usePece = false, [Input(Name = "simple_order_2")] bool simpleOrder2 = false)
    {
        double start = model.Sampling.PercentToSigma(sdeStartPercent), end = model.Sampling.PercentToSigma(sdeEndPercent);
        return new(usePece ? "sa_solver_pece" : "sa_solver", new SamplerOptions
        {
            Eta = eta, SNoise = sNoise, PredictorOrder = predictorOrder, CorrectorOrder = correctorOrder, UsePece = usePece, SimpleOrder2 = simpleOrder2,
            TauFunc = sigma => sigma <= start && sigma >= end ? eta : 0,
        });
    }
}

[Node("SamplerSEEDS2", Category = "model/sampling/samplers", DisplayName = "SamplerSEEDS2",
    Description = "This sampler node can represent multiple samplers:\n\nseeds_2\n- default setting\n\nexp_heun_2_x0\n- solver_type=phi_2, r=1.0, eta=0.0\n\nexp_heun_2_x0_sde\n- solver_type=phi_2, r=1.0, eta=1.0, s_noise=1.0")]
public sealed class SamplerSEEDS2
{
    public SamplerChoice Execute([Input(Options = ["phi_1", "phi_2"])] string solverType = "phi_1",
        [Input(Min = 0.0, Max = 100.0, Step = 0.01, Tooltip = "Stochastic strength")] double eta = 1.0,
        [Input(Name = "s_noise", Min = 0.0, Max = 100.0, Step = 0.01, Tooltip = "SDE noise multiplier")] double sNoise = 1.0,
        [Input(Min = 0.01, Max = 1.0, Step = 0.01, Tooltip = "Relative step size for the intermediate stage (c2 node)")] double r = 0.5) =>
        new("seeds_2", new SamplerOptions { SolverType = solverType, Eta = eta, SNoise = sNoise, R = r });
}

[Node("SamplerLCM", Category = "model/sampling/samplers", DisplayName = "SamplerLCM",
    Description = "LCM sampler with tunable per-step noise. s_noise is a multiplier on the model's training noise scale")]
public sealed class SamplerLCM
{
    public SamplerChoice Execute(
        [Input(Name = "s_noise", Min = 0.0, Max = 64.0, Step = 0.01, Tooltip = "Per-step noise multiplier at the first step (1.0 = match training).")] double sNoise = 1.0,
        [Input(Name = "s_noise_end", Min = 0.0, Max = 64.0, Step = 0.01, Tooltip = "Per-step noise multiplier at the last step. Set equal to s_noise for a constant schedule.")] double sNoiseEnd = 1.0,
        [Input(Min = 0.0, Max = 10.0, Step = 0.01, Tooltip = "Clamp per-step noise to +/- N*std. 0 disables.")] double noiseClipStd = 0.0) =>
        new("lcm", new SamplerOptions { SNoise = sNoise, SNoiseEnd = sNoiseEnd, NoiseClipStd = noiseClipStd });
}
