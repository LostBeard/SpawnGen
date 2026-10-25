using System.Globalization;
using System.Text.RegularExpressions;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

// The sampler nodes of ComfyUI's comfy_extras/nodes_custom_sampler.py and nodes_advanced_samplers.py: each picks a k-diffusion sampler and sets its options.

[Node(Category = "model/sampling/samplers", DisplayName = "SamplerEulerAncestral")]
public sealed class SamplerEulerAncestral
{
    public SamplerChoice Execute([Input(Min = 0, Max = 100, Step = 0.01)] double eta = 1.0, [Input(Name = "s_noise", Min = 0, Max = 100, Step = 0.01)] double sNoise = 1.0) =>
        new("euler_ancestral", new SamplerOptions { Eta = eta, SNoise = sNoise });
}

[Node(Category = "model/sampling/samplers", DisplayName = "SamplerEulerAncestralCFG++")]
public sealed class SamplerEulerAncestralCFGPP
{
    public SamplerChoice Execute([Input(Min = 0, Max = 1, Step = 0.01)] double eta = 1.0, [Input(Name = "s_noise", Min = 0, Max = 10, Step = 0.01)] double sNoise = 1.0) =>
        new("euler_ancestral_cfg_pp", new SamplerOptions { Eta = eta, SNoise = sNoise });
}

[Node(Category = "model/sampling/samplers", DisplayName = "SamplerLMS")]
public sealed class SamplerLMS
{
    public SamplerChoice Execute([Input(Min = 1, Max = 100)] int order = 4) => new("lms", new SamplerOptions { Order = order });
}

[Node(Category = "model/sampling/samplers", DisplayName = "SamplerDPMPP_3M_SDE")]
public sealed class SamplerDPMPP_3M_SDE
{
    /// <summary><c>noise_device</c> only says where ComfyUI keeps its Brownian tree; the noise here is the same either way.</summary>
    public SamplerChoice Execute([Input(Min = 0, Max = 100, Step = 0.01)] double eta = 1.0, [Input(Name = "s_noise", Min = 0, Max = 100, Step = 0.01)] double sNoise = 1.0,
        [Input(Options = ["gpu", "cpu"])] string noiseDevice = "gpu") =>
        new(noiseDevice == "cpu" ? "dpmpp_3m_sde" : "dpmpp_3m_sde_gpu", new SamplerOptions { Eta = eta, SNoise = sNoise });
}

[Node(Category = "model/sampling/samplers", DisplayName = "SamplerDPMPP_2M_SDE")]
public sealed class SamplerDPMPP_2M_SDE
{
    public SamplerChoice Execute([Input(Options = ["midpoint", "heun"])] string solverType = "midpoint",
        [Input(Min = 0, Max = 100, Step = 0.01)] double eta = 1.0, [Input(Name = "s_noise", Min = 0, Max = 100, Step = 0.01)] double sNoise = 1.0,
        [Input(Options = ["gpu", "cpu"])] string noiseDevice = "gpu") =>
        new(noiseDevice == "cpu" ? "dpmpp_2m_sde" : "dpmpp_2m_sde_gpu", new SamplerOptions { Eta = eta, SNoise = sNoise, SolverType = solverType });
}

[Node(Category = "model/sampling/samplers", DisplayName = "SamplerDPMPP_SDE")]
public sealed class SamplerDPMPP_SDE
{
    public SamplerChoice Execute([Input(Min = 0, Max = 100, Step = 0.01)] double eta = 1.0, [Input(Name = "s_noise", Min = 0, Max = 100, Step = 0.01)] double sNoise = 1.0,
        [Input(Min = 0, Max = 100, Step = 0.01)] double r = 0.5, [Input(Options = ["gpu", "cpu"])] string noiseDevice = "gpu") =>
        new(noiseDevice == "cpu" ? "dpmpp_sde" : "dpmpp_sde_gpu", new SamplerOptions { Eta = eta, SNoise = sNoise, R = r });
}

[Node(Category = "model/sampling/samplers", DisplayName = "SamplerDPMPP_2S_Ancestral")]
public sealed class SamplerDPMPP_2S_Ancestral
{
    public SamplerChoice Execute([Input(Min = 0, Max = 100, Step = 0.01)] double eta = 1.0, [Input(Name = "s_noise", Min = 0, Max = 100, Step = 0.01)] double sNoise = 1.0) =>
        new("dpmpp_2s_ancestral", new SamplerOptions { Eta = eta, SNoise = sNoise });
}

[Node(Category = "model/sampling/samplers", DisplayName = "SamplerDPMAdaptative")]
public sealed class SamplerDPMAdaptative
{
    /// <summary>DPM-Solver with adaptive step size: it picks its own steps (only the first and last sigma of the schedule count) until the error is within rtol / atol.</summary>
    public SamplerChoice Execute(
        [Input(Min = 2, Max = 3)] int order = 3,
        [Input(Min = 0, Max = 100, Step = 0.01)] double rtol = 0.05,
        [Input(Min = 0, Max = 100, Step = 0.01)] double atol = 0.0078,
        [Input(Name = "h_init", Min = 0, Max = 100, Step = 0.01)] double hInit = 0.05,
        [Input(Min = 0, Max = 100, Step = 0.01)] double pcoeff = 0.0,
        [Input(Min = 0, Max = 100, Step = 0.01)] double icoeff = 1.0,
        [Input(Min = 0, Max = 100, Step = 0.01)] double dcoeff = 0.0,
        [Input(Name = "accept_safety", Min = 0, Max = 100, Step = 0.01)] double acceptSafety = 0.81,
        [Input(Min = 0, Max = 100, Step = 0.01)] double eta = 0.0,
        [Input(Name = "s_noise", Min = 0, Max = 100, Step = 0.01)] double sNoise = 1.0) =>
        new("dpm_adaptive", new SamplerOptions
        {
            Order = order, Rtol = rtol, Atol = atol, HInit = hInit, PCoeff = pcoeff, ICoeff = icoeff, DCoeff = dcoeff, AcceptSafety = acceptSafety, Eta = eta, SNoise = sNoise,
        });
}

[Node(Category = "model/sampling/samplers", DisplayName = "SamplerER_SDE")]
public sealed class SamplerER_SDE
{
    /// <summary>
    /// The ER-SDE solver with its noise scaler: ER-SDE x (e^(x^0.3) + 10)^eta, reverse-time SDE x^(eta + 1), or ODE x (no noise; also what eta 0 means).
    /// </summary>
    public SamplerChoice Execute([Input(Options = ["ER-SDE", "Reverse-time SDE", "ODE"])] string solverType = "ER-SDE",
        [Input(Min = 1, Max = 3)] int maxStage = 3,
        [Input(Min = 0, Max = 10, Step = 0.01, Tooltip = "Stochastic strength of SDEs.\nWhen eta=0, they reduce to deterministic ODE.\nLarge eta may cause invalid outputs. If this occurs, try decreasing this value.")] double eta = 1.0,
        [Input(Name = "s_noise", Min = 0, Max = 100, Step = 0.01)] double sNoise = 1.0)
    {
        if (solverType == "ODE" || eta == 0)
        {
            sNoise = 0.0;
            solverType = "ODE";
        }
        Func<double, double> scaler = solverType switch
        {
            "ER-SDE" => x => x * Math.Pow(Math.Exp(Math.Pow(x, 0.3)) + 10.0, eta),
            "Reverse-time SDE" => x => Math.Pow(x, eta + 1),
            _ => x => x,
        };
        return new("er_sde", new SamplerOptions { SNoise = sNoise, NoiseScaler = scaler, MaxStage = maxStage });
    }
}

[Node(Category = "model/sampling/samplers", DisplayName = "SamplerSASolver")]
public sealed class SamplerSASolver
{
    /// <summary>Stochastic Adams solver; the noise is added only between sde_start_percent and sde_end_percent of the run.</summary>
    public SamplerChoice Execute(ModelHandle model,
        [Input(Min = 0, Max = 10, Step = 0.01)] double eta = 1.0,
        [Input(Min = 0, Max = 1, Step = 0.001)] double sdeStartPercent = 0.2,
        [Input(Min = 0, Max = 1, Step = 0.001)] double sdeEndPercent = 0.8,
        [Input(Name = "s_noise", Min = 0, Max = 100, Step = 0.01)] double sNoise = 1.0,
        [Input(Min = 1, Max = 6)] int predictorOrder = 3,
        [Input(Min = 0, Max = 6)] int correctorOrder = 4,
        bool usePece = false,
        [Input(Name = "simple_order_2")] bool simpleOrder2 = false)
    {
        var tau = Samplers.TauInterval(model.Sampling.PercentToSigma(sdeStartPercent), model.Sampling.PercentToSigma(sdeEndPercent), eta);
        return new("sa_solver", new SamplerOptions
        {
            TauFunc = tau, SNoise = sNoise, PredictorOrder = predictorOrder, CorrectorOrder = correctorOrder, UsePece = usePece, SimpleOrder2 = simpleOrder2,
        });
    }
}

[Node(Category = "model/sampling/samplers", DisplayName = "SamplerSEEDS2",
    Description = "This sampler node can represent multiple samplers:\n\nseeds_2\n- default setting\n\nexp_heun_2_x0\n- solver_type=phi_2, r=1.0, eta=0.0\n\nexp_heun_2_x0_sde\n- solver_type=phi_2, r=1.0, eta=1.0, s_noise=1.0")]
public sealed class SamplerSEEDS2
{
    public SamplerChoice Execute([Input(Options = ["phi_1", "phi_2"])] string solverType = "phi_1",
        [Input(Min = 0, Max = 100, Step = 0.01, Tooltip = "Stochastic strength")] double eta = 1.0,
        [Input(Name = "s_noise", Min = 0, Max = 100, Step = 0.01, Tooltip = "SDE noise multiplier")] double sNoise = 1.0,
        [Input(Min = 0.01, Max = 1, Step = 0.01, Tooltip = "Relative step size for the intermediate stage (c2 node)")] double r = 0.5) =>
        new("seeds_2", new SamplerOptions { Eta = eta, SNoise = sNoise, R = r, SolverType = solverType });
}

[Node(Category = "model/sampling/samplers", DisplayName = "SamplerLCM",
    Description = "LCM sampler with tunable per-step noise. s_noise is a multiplier on the model's training noise scale")]
public sealed class SamplerLCM
{
    public SamplerChoice Execute(
        [Input(Name = "s_noise", Min = 0, Max = 64, Step = 0.01, Tooltip = "Per-step noise multiplier at the first step (1.0 = match training).")] double sNoise = 1.0,
        [Input(Name = "s_noise_end", Min = 0, Max = 64, Step = 0.01, Tooltip = "Per-step noise multiplier at the last step. Set equal to s_noise for a constant schedule.")] double sNoiseEnd = 1.0,
        [Input(Min = 0, Max = 10, Step = 0.01, Tooltip = "Clamp per-step noise to +/- N*std. 0 disables.")] double noiseClipStd = 0.0) =>
        new("lcm", new SamplerOptions { SNoise = sNoise, SNoiseEnd = sNoiseEnd, NoiseClipStd = noiseClipStd });
}

[Node(Category = "experimental", DisplayName = "SamplerEulerCFG++")]
public sealed class SamplerEulerCFGpp
{
    /// <summary>CFG++ Euler: "regular" is euler_cfg_pp; "alternative" steps along (x - 2 * denoised + uncond) / sigma (ComfyUI's sample_euler_pp).</summary>
    public SamplerChoice Execute([Input(Options = ["regular", "alternative"])] string version = "regular") =>
        new(version == "alternative" ? Samplers.EulerPP : "euler_cfg_pp");
}

// --- schedulers -------------------------------------------------------------------------------------------------------------------------------

[Node(Category = "model/sampling/schedulers", DisplayName = "LaplaceScheduler")]
public sealed class LaplaceScheduler
{
    public Sigmas Execute([Input(Min = 1, Max = 10000)] int steps = 20, [Input(Min = 0, Max = 5000, Step = 0.01)] double sigmaMax = 14.614642,
        [Input(Min = 0, Max = 5000, Step = 0.01)] double sigmaMin = 0.0291675, [Input(Min = -10, Max = 10, Step = 0.1)] double mu = 0.0,
        [Input(Min = 0, Max = 10, Step = 0.1)] double beta = 0.5) =>
        new(SigmaSchedules.Laplace(steps, sigmaMax, sigmaMin, mu, beta));
}

/// <summary>The fixed schedules of Align Your Steps, GITS and OptimalSteps: a table, log-linearly resampled to the step count, the tail kept for denoise below 1.</summary>
public static class TableSchedules
{
    /// <summary>ComfyUI's rule for these nodes: total = round(steps * denoise); the last total + 1 values of the (resampled) table, the final one set to 0.</summary>
    internal static float[] Tail(double[] table, int steps, double denoise)
    {
        int total = steps;
        if (denoise < 1.0)
        {
            if (denoise <= 0.0) return [];
            total = (int)Math.Round(steps * denoise, MidpointRounding.ToEven);
        }
        var sigmas = steps + 1 != table.Length ? SigmaSchedules.LogLinearInterp(table, steps + 1) : table;
        var tail = sigmas[Math.Max(0, sigmas.Length - (total + 1))..].Select(v => (float)v).ToArray();
        tail[^1] = 0f;
        return tail;
    }

    // From NVIDIA's Align Your Steps (comfy_extras/nodes_align_your_steps.py).
    internal static readonly Dictionary<string, double[]> AlignYourSteps = new()
    {
        ["SD1"] = [14.6146412293, 6.4745760956, 3.8636745985, 2.6946151520, 1.8841921177, 1.3943805092, 0.9642583904, 0.6523686016, 0.3977456272, 0.1515232662, 0.0291671582],
        ["SDXL"] = [14.6146412293, 6.3184485287, 3.7681790315, 2.1811480769, 1.3405244945, 0.8620721141, 0.5550693289, 0.3798540708, 0.2332364134, 0.1114188177, 0.0291671582],
        ["SVD"] = [700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.002],
    };

    // From OptimalSteps (comfy_extras/nodes_optimalsteps.py).
    internal static readonly Dictionary<string, double[]> OptimalSteps = new()
    {
        ["FLUX"] = [0.9968, 0.9886, 0.9819, 0.975, 0.966, 0.9471, 0.9158, 0.8287, 0.5512, 0.2808, 0.001],
        ["Wan"] = [1.0, 0.997, 0.995, 0.993, 0.991, 0.989, 0.987, 0.985, 0.98, 0.975, 0.973, 0.968, 0.96, 0.946, 0.927, 0.902, 0.864, 0.776, 0.539, 0.208, 0.001],
        ["Chroma"] = [0.992, 0.99, 0.988, 0.985, 0.982, 0.978, 0.973, 0.968, 0.961, 0.953, 0.943, 0.931, 0.917, 0.9, 0.881, 0.858, 0.832, 0.802, 0.769, 0.731, 0.69, 0.646, 0.599, 0.55, 0.501, 0.451, 0.402, 0.355, 0.311, 0.27, 0.232, 0.199, 0.169, 0.143, 0.12, 0.101, 0.084, 0.07, 0.058, 0.048, 0.001],
    };

    /// <summary>The GITS table for a coefficient (rounded to 2 decimals; only the multiples of 0.05 from 0.80 to 1.50 exist, as in ComfyUI).</summary>
    internal static double[][] Gits(double coeff)
    {
        double key = Math.Round(coeff, 2, MidpointRounding.ToEven);
        foreach (var (c, levels) in GitsLevels)
            if (Math.Abs(c - key) < 1e-9) return levels;
        throw new ArgumentException($"GITS has no schedules for coeff {key} (0.80 to 1.50 in steps of 0.05).", nameof(coeff));
    }

    // Ported from ComfyUI's comfy_extras/nodes_gits.py NOISE_LEVELS: per coefficient, the GITS schedules for 2 to 20 steps (index steps - 2).
    static readonly (double Coeff, double[][] Levels)[] GitsLevels = GitsTable.Levels;
}

[Node(Category = "model/sampling/schedulers", DisplayName = "AlignYourStepsScheduler")]
public sealed class AlignYourStepsScheduler
{
    public Sigmas Execute([Input(Options = ["SD1", "SDXL", "SVD"])] string modelType = "SD1", [Input(Min = 1, Max = 10000)] int steps = 10,
        [Input(Min = 0, Max = 1, Step = 0.01)] double denoise = 1.0) =>
        new(TableSchedules.Tail(TableSchedules.AlignYourSteps[modelType], steps, denoise));
}

[Node(Category = "model/sampling/schedulers", DisplayName = "GITSScheduler")]
public sealed class GITSScheduler
{
    public Sigmas Execute([Input(Min = 0.80, Max = 1.50, Step = 0.05)] double coeff = 1.20, [Input(Min = 2, Max = 1000)] int steps = 10,
        [Input(Min = 0, Max = 1, Step = 0.01)] double denoise = 1.0)
    {
        if (denoise < 1.0 && denoise <= 0.0) return new Sigmas([]);
        var levels = TableSchedules.Gits(coeff);
        // Up to 20 steps the table has the exact schedule; beyond, its 20-step one is resampled (Tail does that when the length differs).
        var table = steps <= 20 ? levels[steps - 2] : levels[^1];
        return new(TableSchedules.Tail(table, steps, denoise));
    }
}

[Node(Category = "model/sampling/schedulers", DisplayName = "OptimalStepsScheduler")]
public sealed class OptimalStepsScheduler
{
    public Sigmas Execute([Input(Options = ["FLUX", "Wan", "Chroma"])] string modelType = "FLUX", [Input(Min = 3, Max = 1000)] int steps = 20,
        [Input(Min = 0, Max = 1, Step = 0.01)] double denoise = 1.0) =>
        new(TableSchedules.Tail(TableSchedules.OptimalSteps[modelType], steps, denoise));
}

// --- sigmas -----------------------------------------------------------------------------------------------------------------------------------

[Node(Category = "model/sampling/sigmas", DisplayName = "ManualSigmas")]
public sealed class ManualSigmas
{
    static readonly Regex Number = new(@"[-+]?(?:\d*\.*\d+)", RegexOptions.CultureInvariant);

    /// <summary>The numbers in the text, in order (ComfyUI's regex, so "1, 0.5" is two sigmas; there is no final 0 unless you type one).</summary>
    public Sigmas Execute(string sigmas = "1, 0.5") =>
        new([.. Number.Matches(sigmas).Select(m => (float)double.Parse(m.Value, NumberStyles.Float, CultureInfo.InvariantCulture))]);
}

[Node(Category = "model/sampling/sigmas", DisplayName = "ExtendIntermediateSigmas")]
public sealed class ExtendIntermediateSigmas
{
    /// <summary>
    /// Inserts <c>steps - 1</c> sigmas between each pair whose upper sigma lies in [end_at_sigma, start_at_sigma] (start -1 = no upper limit), spaced linearly or
    /// along a cosine / sine curve.
    /// </summary>
    public Sigmas Execute(Sigmas sigmas, [Input(Min = 1, Max = 100)] int steps = 2,
        [Input(Min = -1, Max = 20000, Step = 0.01)] double startAtSigma = -1.0, [Input(Min = 0, Max = 20000, Step = 0.01)] double endAtSigma = 12.0,
        [Input(Options = ["linear", "cosine", "sine"])] string spacing = "linear")
    {
        if (startAtSigma < 0) startAtSigma = double.PositiveInfinity;
        var x = ModelSigmas.Linspace(0, 1, steps + 1);
        var inner = x.Length > 2 ? x[1..^1] : [];
        var computed = inner.Select(v => spacing switch
        {
            "cosine" => (double)MathF.Sin((float)(v * Math.PI / 2)),
            "sine" => 1 - (double)MathF.Cos((float)(v * Math.PI / 2)),
            _ => v,
        }).ToArray();
        var v0 = sigmas.Values;
        var extended = new List<float>();
        for (int i = 0; i < v0.Length - 1; i++)
        {
            float current = v0[i], next = v0[i + 1];
            extended.Add(current);
            if (endAtSigma <= current && current <= startAtSigma)
                foreach (var c in computed) extended.Add((float)(c * (next - current) + current));
        }
        if (v0.Length > 0) extended.Add(v0[^1]);
        return new([.. extended]);
    }
}

[Node(Category = "model/sampling/sigmas", DisplayName = "SamplingPercentToSigma", OutputNames = ["sigma_value"])]
public sealed class SamplingPercentToSigma
{
    /// <summary>The sigma at which a run is the given fraction done. At 0 and 1 the interval-check values are a huge number and 0, unless the model's actual range is asked for.</summary>
    public double Execute(ModelHandle model, [Input(Min = 0, Max = 1, Step = 0.0001)] double samplingPercent = 0.0,
        [Input(Tooltip = "Return the actual sigma value instead of the value used for interval checks.\nThis only affects results at 0.0 and 1.0.")] bool returnActualSigma = false)
    {
        var sampling = model.Sampling;
        double sigma = sampling.PercentToSigma(samplingPercent);
        if (returnActualSigma)
        {
            if (samplingPercent == 0.0) sigma = sampling.SigmaMax;
            else if (samplingPercent == 1.0) sigma = sampling.SigmaMin;
        }
        return sigma;
    }
}

// --- noise ------------------------------------------------------------------------------------------------------------------------------------

[Node(Category = "model/sampling/noise", DisplayName = "AddNoise")]
public sealed class AddNoise
{
    /// <summary>
    /// Adds the noise to the latent at the noise level the sigmas span (|first - last|, or the only sigma), as a sampler would at its start. NaN and infinities become 0.
    /// </summary>
    public LatentBatch Execute(ModelHandle model, NoiseChoice noise, Sigmas sigmas, LatentBatch latentImage)
    {
        var v = sigmas.Values;
        if (v.Length == 0) return latentImage;
        double scale = v.Length > 1 ? Math.Abs(v[0] - v[^1]) : v[0];
        int perImage = latentImage.Channels * latentImage.Height * latentImage.Width;
        var result = new LatentBatch(latentImage.Batch, latentImage.Channels, latentImage.Height, latentImage.Width)
        {
            NoiseMask = latentImage.NoiseMask, BatchIndex = latentImage.BatchIndex,
        };
        for (int b = 0; b < latentImage.Batch; b++)
        {
            // In the model's space, as the sampler would add it, then back to the VAE's (the empty latent is not scaled).
            var latent = model.LatentFormat.ProcessIn(latentImage.Data[(b * perImage)..((b + 1) * perImage)]);
            var fresh = noise.Enabled ? SamplingCore.InitialNoise(noise.Seed, b, latentImage) : new float[perImage];
            var noisy = model.LatentFormat.ProcessOut(model.Sampling.NoiseScaling(scale, fresh, latent));
            for (int i = 0; i < perImage; i++) result.Data[b * perImage + i] = float.IsFinite(noisy[i]) ? noisy[i] : 0f;
        }
        return result;
    }
}
