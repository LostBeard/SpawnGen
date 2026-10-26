using SpawnDev.ILGPU.ML.Preprocessing;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// The settings a sampler node passes (ComfyUI's <c>extra_options</c>). A null field keeps the sampler's own default, so the same record serves every sampler.
/// </summary>
public sealed record SamplerOptions
{
    public double? Eta { get; init; }
    public double? SNoise { get; init; }
    /// <summary>The intermediate stage position (dpmpp_sde, seeds_2).</summary>
    public double? R { get; init; }
    /// <summary>midpoint / heun (dpmpp_2m_sde), phi_1 / phi_2 (seeds_2).</summary>
    public string? SolverType { get; init; }
    /// <summary>lms order; dpm_adaptive order (2 or 3).</summary>
    public int? Order { get; init; }
    public double? Rtol { get; init; }
    public double? Atol { get; init; }
    public double? HInit { get; init; }
    public double? PCoeff { get; init; }
    public double? ICoeff { get; init; }
    public double? DCoeff { get; init; }
    public double? AcceptSafety { get; init; }
    /// <summary>er_sde: how many history stages (1 to 3) and its noise scaler (a function of lambda = sigma / alpha).</summary>
    public int? MaxStage { get; init; }
    public Func<double, double>? NoiseScaler { get; init; }
    /// <summary>sa_solver: stochasticity tau as a function of sigma, and its predictor / corrector orders.</summary>
    public Func<double, double>? TauFunc { get; init; }
    public int? PredictorOrder { get; init; }
    public int? CorrectorOrder { get; init; }
    public bool? UsePece { get; init; }
    public bool? SimpleOrder2 { get; init; }
    /// <summary>lcm: the per-step noise multiplier at the last step (s_noise is the first), and a clamp of the fresh noise to +/- N standard deviations (0 = off).</summary>
    public double? SNoiseEnd { get; init; }
    public double? NoiseClipStd { get; init; }
}

/// <summary>The predictions behind the model call just made: the conditional and unconditional denoised estimates (the latter zero when it was not computed).</summary>
public sealed record GuidanceOutput(float[] CondDenoised, float[] UncondDenoised, bool HasUncond);

/// <summary>
/// Sigma-space samplers, ported line by line from k-diffusion as ComfyUI ships it (<c>comfy/k_diffusion/sampling.py</c>, <c>uni_pc.py</c>, <c>deis.py</c>,
/// <c>sa_solver.py</c>): the sample <c>x</c> lives at noise level sigma, the model returns its denoised estimate, and each step moves <c>x</c> to the next sigma.
/// Pure host maths over small latents; the model call is the only part that touches the GPU. Fresh noise comes from a seeded <see cref="NoiseSource"/> instead
/// of torch's generator, so a seed gives a different (equally distributed) picture than in ComfyUI.
/// </summary>
public static partial class Samplers
{
    /// <summary>ComfyUI's SAMPLER_NAMES (KSAMPLER_NAMES then ddim, uni_pc, uni_pc_bh2), in its order.</summary>
    public static readonly string[] Names =
    [
        "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2", "exp_heun_2_x0", "exp_heun_2_x0_sde", "dpm_2", "dpm_2_ancestral",
        "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
        "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",
        "ipndm", "ipndm_v", "deis", "cfgpp_ud10_ab", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp",
        "gradient_estimation", "gradient_estimation_cfg_pp", "er_sde", "seeds_2", "seeds_3", "sa_solver", "sa_solver_pece",
        "ddim", "uni_pc", "uni_pc_bh2",
    ];

    /// <summary>SamplerEulerCFGpp's "alternative" version (ComfyUI's <c>sample_euler_pp</c>): not a KSampler name, only reachable through that node.</summary>
    public const string EulerPP = "euler_pp";

    /// <summary>KSampler computes one step more for these and drops the second to last sigma (ComfyUI's DISCARD_PENULTIMATE_SIGMA_SAMPLERS).</summary>
    public static bool DiscardsPenultimateSigma(string sampler) => sampler is "dpm_2" or "dpm_2_ancestral" or "uni_pc" or "uni_pc_bh2";

    /// <summary>The model's denoised estimate for the sample <c>x</c> at noise level <c>sigma</c> (the callee applies input scaling, guidance and masks).</summary>
    public delegate Task<float[]> DenoiseFn(float[] x, float sigma);

    /// <summary>The samplers that read the unconditional denoised estimate (ComfyUI's CFG++ family).</summary>
    public static bool UsesUncond(string sampler) => sampler.EndsWith("_cfg_pp", StringComparison.Ordinal) || sampler is "cfgpp_ud10_ab" or EulerPP;

    /// <summary>The samplers that want the unconditional estimate even at cfg 1 (they set ComfyUI's disable_cfg1_optimization).</summary>
    public static bool DisablesCfg1Optimization(string sampler) => UsesUncond(sampler) && sampler != "cfgpp_ud10_ab";

    /// <summary>
    /// Where a stochastic sampler's fresh noise comes from: standard normal draws, deterministic per (seed, draw number) so a seed reproduces a picture.
    /// Each draw is a new well-mixed seed of the library generator. Tests override <see cref="At"/> to feed known draws.
    /// </summary>
    public class NoiseSource(int seed, int channels, int height, int width)
    {
        int _draws;

        /// <summary>Draw number <paramref name="index"/> (independent of the others).</summary>
        public virtual float[] At(int index) => DiffusionScheduler.GenerateNoise(channels, height, width, Mix(seed, index));

        /// <summary>The next draw (ComfyUI's <c>noise_sampler(sigma, sigma_next)</c> for the default sampler, which ignores its arguments).</summary>
        public float[] Next() => At(_draws++);

        // A well-mixed seed per draw: nearby seeds give correlated streams in the library generator, and correlated noise would add up coherently.
        static int Mix(int seed, int step)
        {
            unchecked
            {
                uint z = (uint)seed * 0x9E3779B1u + (uint)(step + 1) * 0x85EBCA6Bu;
                z ^= z >> 16; z *= 0x7FEB352Du; z ^= z >> 15; z *= 0x846CA68Bu; z ^= z >> 16;
                return (int)z;
            }
        }
    }

    /// <summary>
    /// Runs <paramref name="sampler"/> over <paramref name="sigmas"/> (length steps + 1, usually ending in 0), starting from the already-noised <paramref name="x"/>.
    /// <paramref name="uncond"/> gives the guidance behind the model call just made (the CFG++ samplers need its unconditional estimate);
    /// <paramref name="onStep"/> reports steps done and <paramref name="onDenoised"/> the denoised estimate ComfyUI's callback sees (the preview and denoised_output).
    /// <paramref name="sampling"/> is the model's sampling type (eps SD when null).
    /// </summary>
    public static Task<float[]> RunAsync(string sampler, float[] x, float[] sigmas, DenoiseFn model, NoiseSource noise,
        Action<int>? onStep = null, CancellationToken ct = default, Func<GuidanceOutput?>? uncond = null, SamplerOptions? options = null,
        ModelSampling? sampling = null, Func<float[], Task>? onDenoised = null)
    {
        if (!Names.Contains(sampler) && sampler != EulerPP) throw new ArgumentException($"Unknown sampler '{sampler}'. Known: {string.Join(", ", Names)}.", nameof(sampler));
        var run = new Run
        {
            Model = model, Noise = noise, OnStep = onStep, OnDenoised = onDenoised, Ct = ct, Uncond = uncond,
            Options = options ?? new SamplerOptions(), Sampling = sampling ?? DefaultSampling.Value,
        };
        var s = sigmas.Select(v => (double)v).ToArray();
        x = (float[])x.Clone();
        var o = run.Options;
        return sampler switch
        {
            "euler" or "ddim" => Euler(run, x, s),
            "euler_cfg_pp" => EulerAncestralCfgPP(run, x, s, 0.0, 0.0),
            "euler_ancestral" => EulerAncestral(run, x, s, o.Eta ?? 1, o.SNoise ?? 1),
            "euler_ancestral_cfg_pp" => EulerAncestralCfgPP(run, x, s, o.Eta ?? 1, o.SNoise ?? 1),
            "heun" => Heun(run, x, s),
            "heunpp2" => HeunPP2(run, x, s),
            "exp_heun_2_x0" => Seeds2(run, x, s, 0.0, 0.0, 1.0, o.SolverType ?? "phi_2"),
            "exp_heun_2_x0_sde" => Seeds2(run, x, s, o.Eta ?? 1, o.SNoise ?? 1, 1.0, o.SolverType ?? "phi_2"),
            "dpm_2" => Dpm2(run, x, s),
            "dpm_2_ancestral" => Dpm2Ancestral(run, x, s, o.Eta ?? 1, o.SNoise ?? 1),
            "lms" => Lms(run, x, s, o.Order ?? 4),
            "dpm_fast" => DpmFast(run, x, s),
            "dpm_adaptive" => DpmAdaptive(run, x, s),
            "dpmpp_2s_ancestral" => Dpmpp2SAncestral(run, x, s, o.Eta ?? 1, o.SNoise ?? 1),
            "dpmpp_2s_ancestral_cfg_pp" => Dpmpp2SAncestralCfgPP(run, x, s, o.Eta ?? 1, o.SNoise ?? 1),
            "dpmpp_sde" or "dpmpp_sde_gpu" => DpmppSde(run, x, s, o.Eta ?? 1, o.SNoise ?? 1, o.R ?? 0.5),
            "dpmpp_2m" => Dpmpp2M(run, x, s),
            "dpmpp_2m_cfg_pp" => Dpmpp2MCfgPP(run, x, s),
            "dpmpp_2m_sde" or "dpmpp_2m_sde_gpu" => Dpmpp2MSde(run, x, s, o.Eta ?? 1, o.SNoise ?? 1, o.SolverType ?? "midpoint"),
            "dpmpp_2m_sde_heun" or "dpmpp_2m_sde_heun_gpu" => Dpmpp2MSde(run, x, s, o.Eta ?? 1, o.SNoise ?? 1, o.SolverType ?? "heun"),
            "dpmpp_3m_sde" or "dpmpp_3m_sde_gpu" => Dpmpp3MSde(run, x, s, o.Eta ?? 1, o.SNoise ?? 1),
            "ddpm" => Ddpm(run, x, s),
            "lcm" => Lcm(run, x, s, o.SNoise ?? 1, o.SNoiseEnd, o.NoiseClipStd ?? 0),
            "ipndm" => Ipndm(run, x, s),
            "ipndm_v" => IpndmV(run, x, s),
            "deis" => Deis(run, x, s),
            "cfgpp_ud10_ab" => CfgppHistory(run, x, s, 0.25, 1.0, 1, 0.1),
            "res_multistep" => ResMultistep(run, x, s, o.SNoise ?? 1, 0.0, cfgPP: false),
            "res_multistep_cfg_pp" => ResMultistep(run, x, s, o.SNoise ?? 1, 0.0, cfgPP: true),
            "res_multistep_ancestral" => ResMultistep(run, x, s, o.SNoise ?? 1, o.Eta ?? 1, cfgPP: false),
            "res_multistep_ancestral_cfg_pp" => ResMultistep(run, x, s, o.SNoise ?? 1, o.Eta ?? 1, cfgPP: true),
            "gradient_estimation" => GradientEstimation(run, x, s, 2.0, cfgPP: false),
            "gradient_estimation_cfg_pp" => GradientEstimation(run, x, s, 2.0, cfgPP: true),
            "er_sde" => ErSde(run, x, s, o.SNoise ?? 1, o.NoiseScaler, o.MaxStage ?? 3),
            "seeds_2" => Seeds2(run, x, s, o.Eta ?? 1, o.SNoise ?? 1, o.R ?? 0.5, o.SolverType ?? "phi_1"),
            "seeds_3" => Seeds3(run, x, s, o.Eta ?? 1, o.SNoise ?? 1),
            "sa_solver" => SaSolver(run, x, s, usePece: o.UsePece ?? false),
            "sa_solver_pece" => SaSolver(run, x, s, usePece: true),
            "uni_pc" => UniPC(run, x, s, "bh1"),
            "uni_pc_bh2" => UniPC(run, x, s, "bh2"),
            EulerPP => EulerPPSampler(run, x, s),
            _ => throw new ArgumentException($"Unknown sampler '{sampler}'.", nameof(sampler)),
        };
    }

    static readonly Lazy<ModelSampling> DefaultSampling = new(() => new ModelSampling(ModelSigmas.LinearSchedule()));

    /// <summary>One sampling run's plumbing: the model, its noise, callbacks, and the model sampling's half-log-SNR maps.</summary>
    sealed class Run
    {
        public required DenoiseFn Model { get; init; }
        public required NoiseSource Noise { get; init; }
        public required SamplerOptions Options { get; init; }
        public required ModelSampling Sampling { get; init; }
        public Action<int>? OnStep { get; init; }
        public Func<float[], Task>? OnDenoised { get; init; }
        public Func<GuidanceOutput?>? Uncond { get; init; }
        public CancellationToken Ct { get; init; }

        public Task<float[]> Call(float[] x, double sigma)
        {
            Ct.ThrowIfCancellationRequested();
            return Model(x, (float)sigma);
        }

        /// <summary>ComfyUI's per-step callback: the preview / denoised_output estimate, and progress.</summary>
        public async Task CallbackAsync(int i, float[] denoised)
        {
            if (OnDenoised is { } onDenoised) await onDenoised(denoised);
            OnStep?.Invoke(i + 1);
        }

        /// <summary>The unconditional estimate of the last model call (zero when guidance skipped it, like ComfyUI's empty batch slot).</summary>
        public float[] UncondDenoised(float[] fallback) => Uncond?.Invoke()?.UncondDenoised ?? fallback;

        /// <summary>The last call's guidance, for cfgpp_ud10_ab: the unconditional estimate, or the conditional one when there was none.</summary>
        public float[] UncondOrCond(float[] fallback) => Uncond?.Invoke() is { } g ? (g.HasUncond ? g.UncondDenoised : g.CondDenoised) : fallback;

        public float[] NextNoise() => Noise.Next();

        /// <summary>Half-log-SNR lambda = log(alpha / sigma): -log(sigma), or log((1 - sigma) / sigma) for a CONST (flow) sampling.</summary>
        public double Lambda(double sigma) => Sampling.IsConst ? -Logit(sigma) : -Math.Log(sigma);

        /// <summary>The inverse of <see cref="Lambda"/>.</summary>
        public double SigmaOfLambda(double lambda) => Sampling.IsConst ? 1.0 / (1.0 + Math.Exp(lambda)) : Math.Exp(-lambda);

        /// <summary>ComfyUI's offset_first_sigma_for_snr: a flow schedule starting at sigma 1 (infinite logSNR) starts at percent 1e-4 instead.</summary>
        public double[] OffsetFirstSigma(double[] sigmas)
        {
            if (sigmas.Length <= 1 || !Sampling.IsConst || sigmas[0] < 1) return sigmas;
            var r = (double[])sigmas.Clone();
            r[0] = Sampling.PercentToSigma(1e-4);
            return r;
        }

        static double Logit(double p) => Math.Log(p / (1 - p));
    }

    // --- vector helpers: x is float, the coefficients are double ----------------------------------------------------------------------------

    static float[] ToD(float[] x, double sigma, float[] denoised)
    {
        var d = new float[x.Length];
        for (int k = 0; k < d.Length; k++) d[k] = (float)((x[k] - denoised[k]) / sigma);
        return d;
    }

    /// <summary>a * ca + b * cb.</summary>
    static float[] Lin(float[] a, double ca, float[] b, double cb)
    {
        var r = new float[a.Length];
        for (int k = 0; k < r.Length; k++) r[k] = (float)(a[k] * ca + b[k] * cb);
        return r;
    }

    /// <summary>x += b * cb, in place.</summary>
    static void AddScaled(float[] x, float[] b, double cb)
    {
        for (int k = 0; k < x.Length; k++) x[k] = (float)(x[k] + b[k] * cb);
    }

    internal static float[] Subtract(float[] a, float[] b) => Sub(a, b);

    internal static float[] LinCombine(float[] a, double ca, float[] b, double cb) => Lin(a, ca, b, cb);

    /// <summary>Classifier-free guidance as ComfyUI writes it: uncond + (cond - uncond) * scale.</summary>
    internal static float[] Guide(float[] uncond, float[] cond, double scale)
    {
        var r = new float[uncond.Length];
        for (int k = 0; k < r.Length; k++) r[k] = (float)(uncond[k] + (float)(cond[k] - uncond[k]) * scale);
        return r;
    }

    static float[] Sub(float[] a, float[] b)
    {
        var r = new float[a.Length];
        for (int k = 0; k < r.Length; k++) r[k] = a[k] - b[k];
        return r;
    }

    /// <summary>exp(x) - 1 without losing precision for small x.</summary>
    static double Expm1(double x) => Math.Abs(x) < 1e-5 ? x + 0.5 * x * x + x * x * x / 6.0 : Math.Exp(x) - 1.0;

    /// <summary>How far a step goes down the ODE (<c>down</c>) and how much fresh noise is added (<c>up</c>) so the total noise level is <paramref name="next"/> (k-diffusion's get_ancestral_step).</summary>
    static (double Down, double Up) AncestralStep(double sigma, double next, double eta = 1.0)
    {
        if (eta == 0) return (next, 0.0);
        double up = Math.Min(next, eta * Math.Sqrt(next * next * (sigma * sigma - next * next) / (sigma * sigma)));
        return (Math.Sqrt(next * next - up * up), up);
    }

    // --- Euler family ---------------------------------------------------------------------------------------------------------------------

    static async Task<float[]> Euler(Run run, float[] x, double[] s)
    {
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            var d = ToD(x, s[i], denoised);
            await run.CallbackAsync(i, denoised);
            AddScaled(x, d, s[i + 1] - s[i]);
        }
        return x;
    }

    static async Task<float[]> EulerAncestral(Run run, float[] x, double[] s, double eta, double sNoise)
    {
        if (run.Sampling.IsConst) return await EulerAncestralRF(run, x, s, eta, sNoise);
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            var (down, up) = AncestralStep(s[i], s[i + 1], eta);
            await run.CallbackAsync(i, denoised);
            if (down == 0) x = denoised;
            else
            {
                var d = ToD(x, s[i], denoised);
                AddScaled(x, d, down - s[i]);
                AddScaled(x, run.NextNoise(), sNoise * up);
            }
        }
        return x;
    }

    /// <summary>Flow-model ancestral steps (ComfyUI's sample_euler_ancestral_RF).</summary>
    static async Task<float[]> EulerAncestralRF(Run run, float[] x, double[] s, double eta, double sNoise)
    {
        sNoise *= run.Sampling.NoiseScale;
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            await run.CallbackAsync(i, denoised);
            if (s[i + 1] == 0) x = denoised;
            else
            {
                double downstepRatio = 1 + (s[i + 1] / s[i] - 1) * eta, down = s[i + 1] * downstepRatio;
                double alphaIp1 = 1 - s[i + 1], alphaDown = 1 - down;
                double renoise = Math.Sqrt(s[i + 1] * s[i + 1] - down * down * alphaIp1 * alphaIp1 / (alphaDown * alphaDown));
                double ratio = down / s[i];
                x = Lin(x, ratio, denoised, 1 - ratio);
                if (eta > 0) x = Lin(x, alphaIp1 / alphaDown, run.NextNoise(), sNoise * renoise);
            }
        }
        return x;
    }

    /// <summary>CFG++ ancestral Euler (euler_cfg_pp is eta 0, s_noise 0): the step leaves from the guided estimate along the unconditional direction.</summary>
    static async Task<float[]> EulerAncestralCfgPP(Run run, float[] x, double[] s, double eta, double sNoise)
    {
        sNoise *= run.Sampling.NoiseScale;
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            await run.CallbackAsync(i, denoised);
            if (s[i + 1] == 0) { x = denoised; continue; }
            var uncond = run.UncondDenoised(denoised);
            double alphaS = s[i] * Math.Exp(run.Lambda(s[i])), alphaT = s[i + 1] * Math.Exp(run.Lambda(s[i + 1]));
            var d = new float[x.Length];
            for (int k = 0; k < d.Length; k++) d[k] = (float)((x[k] - alphaS * uncond[k]) / s[i]);
            var (down, up) = AncestralStep(s[i] / alphaS, s[i + 1] / alphaT, eta);
            down *= alphaT;
            x = Lin(denoised, alphaT, d, down);
            if (eta > 0 && sNoise > 0) AddScaled(x, run.NextNoise(), alphaT * sNoise * up);
        }
        return x;
    }

    /// <summary>SamplerEulerCFGpp's "alternative": Euler along (x - 2 denoised + uncond) / sigma.</summary>
    static async Task<float[]> EulerPPSampler(Run run, float[] x, double[] s)
    {
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            var uncond = run.UncondDenoised(denoised);
            var d = new float[x.Length];
            for (int k = 0; k < d.Length; k++) d[k] = (float)((x[k] - denoised[k] + uncond[k] - denoised[k]) / s[i]);
            await run.CallbackAsync(i, denoised);
            AddScaled(x, d, s[i + 1] - s[i]);
        }
        return x;
    }

    static async Task<float[]> Heun(Run run, float[] x, double[] s)
    {
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            var d = ToD(x, s[i], denoised);
            await run.CallbackAsync(i, denoised);
            double dt = s[i + 1] - s[i];
            if (s[i + 1] == 0) AddScaled(x, d, dt);
            else
            {
                var x2 = Lin(x, 1, d, dt);
                var denoised2 = await run.Call(x2, s[i + 1]);
                var d2 = ToD(x2, s[i + 1], denoised2);
                for (int k = 0; k < x.Length; k++) x[k] = (float)(x[k] + (d[k] + d2[k]) / 2 * dt);
            }
        }
        return x;
    }

    /// <summary>Heun++ (sample_heunpp2): up to three derivatives per step, weighted by how far down the run their sigmas are.</summary>
    static async Task<float[]> HeunPP2(Run run, float[] x, double[] s)
    {
        double sEnd = s[^1];
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            var d = ToD(x, s[i], denoised);
            await run.CallbackAsync(i, denoised);
            double dt = s[i + 1] - s[i];
            if (s[i + 1] == sEnd) AddScaled(x, d, dt);
            else if (s[i + 2] == sEnd)
            {
                var x2 = Lin(x, 1, d, dt);
                var d2 = ToD(x2, s[i + 1], await run.Call(x2, s[i + 1]));
                double w = 2 * s[0], w2 = s[i + 1] / w, w1 = 1 - w2;
                for (int k = 0; k < x.Length; k++) x[k] = (float)(x[k] + (d[k] * w1 + d2[k] * w2) * dt);
            }
            else
            {
                var x2 = Lin(x, 1, d, dt);
                var d2 = ToD(x2, s[i + 1], await run.Call(x2, s[i + 1]));
                double dt2 = s[i + 2] - s[i + 1];
                var x3 = Lin(x2, 1, d2, dt2);
                var d3 = ToD(x3, s[i + 2], await run.Call(x3, s[i + 2]));
                double w = 3 * s[0], w2 = s[i + 1] / w, w3 = s[i + 2] / w, w1 = 1 - w2 - w3;
                for (int k = 0; k < x.Length; k++) x[k] = (float)(x[k] + (w1 * d[k] + w2 * d2[k] + w3 * d3[k]) * dt);
            }
        }
        return x;
    }

    static async Task<float[]> Dpm2(Run run, float[] x, double[] s)
    {
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            var d = ToD(x, s[i], denoised);
            await run.CallbackAsync(i, denoised);
            if (s[i + 1] == 0) AddScaled(x, d, s[i + 1] - s[i]);
            else
            {
                double logS = Math.Log(s[i]), mid = Math.Exp(logS + 0.5 * (Math.Log(s[i + 1]) - logS));
                var x2 = Lin(x, 1, d, mid - s[i]);
                var d2 = ToD(x2, mid, await run.Call(x2, mid));
                AddScaled(x, d2, s[i + 1] - s[i]);
            }
        }
        return x;
    }

    static async Task<float[]> Dpm2Ancestral(Run run, float[] x, double[] s, double eta, double sNoise)
    {
        bool rf = run.Sampling.IsConst;
        if (rf) sNoise *= run.Sampling.NoiseScale;
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            double down, up = 0, alphaIp1 = 0, alphaDown = 0, renoise = 0;
            if (rf)
            {
                down = s[i + 1] * (1 + (s[i + 1] / s[i] - 1) * eta);
                alphaIp1 = 1 - s[i + 1]; alphaDown = 1 - down;
                renoise = Math.Sqrt(s[i + 1] * s[i + 1] - down * down * alphaIp1 * alphaIp1 / (alphaDown * alphaDown));
            }
            else (down, up) = AncestralStep(s[i], s[i + 1], eta);
            await run.CallbackAsync(i, denoised);
            var d = ToD(x, s[i], denoised);
            if (down == 0) AddScaled(x, d, down - s[i]);
            else
            {
                double logS = Math.Log(s[i]), mid = Math.Exp(logS + 0.5 * (Math.Log(down) - logS));
                var x2 = Lin(x, 1, d, mid - s[i]);
                var d2 = ToD(x2, mid, await run.Call(x2, mid));
                AddScaled(x, d2, down - s[i]);
                if (rf) x = Lin(x, alphaIp1 / alphaDown, run.NextNoise(), sNoise * renoise);
                else AddScaled(x, run.NextNoise(), sNoise * up);
            }
        }
        return x;
    }

    /// <summary>The weight of the derivative <paramref name="j"/> steps back in the order-<paramref name="order"/> linear multistep step from t[i] to t[i+1] (integral of its Lagrange basis polynomial).</summary>
    internal static double LinearMultistepCoeff(int order, double[] t, int i, int j)
    {
        if (order - 1 > i) throw new ArgumentException($"Order {order} too high for step {i}");
        double Basis(double tau)
        {
            double prod = 1;
            for (int k = 0; k < order; k++)
            {
                if (k == j) continue;
                prod *= (tau - t[i - k]) / (t[i - j] - t[i - k]);
            }
            return prod;
        }
        // The integrand is a polynomial of degree order - 1, so an order-point Gauss-Legendre rule is exact (scipy's quad gets it to 1e-4 or better).
        var (nodes, weights) = GaussLegendre(Math.Max(order, 2));
        double a = t[i], b = t[i + 1], half = (b - a) / 2, mid = (a + b) / 2, sum = 0;
        for (int q = 0; q < nodes.Length; q++) sum += weights[q] * Basis(mid + half * nodes[q]);
        return sum * half;
    }

    static readonly Dictionary<int, (double[] Nodes, double[] Weights)> GaussLegendreCache = [];

    /// <summary>The n-point Gauss-Legendre rule on [-1, 1] (nodes by Newton's method on the Legendre polynomial).</summary>
    static (double[] Nodes, double[] Weights) GaussLegendre(int n)
    {
        lock (GaussLegendreCache)
        {
            if (GaussLegendreCache.TryGetValue(n, out var cached)) return cached;
            var nodes = new double[n];
            var weights = new double[n];
            for (int i = 0; i < n; i++)
            {
                double z = Math.Cos(Math.PI * (i + 0.75) / (n + 0.5)), dp = 0;
                for (int iter = 0; iter < 100; iter++)
                {
                    double p0 = 1, p1 = z;
                    for (int k = 2; k <= n; k++) { double p2 = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
                    double pn = n == 1 ? z : p1, pn1 = n == 1 ? 1 : p0;
                    dp = n * (z * pn - pn1) / (z * z - 1);
                    double dz = pn / dp;
                    z -= dz;
                    if (Math.Abs(dz) < 1e-15) break;
                }
                nodes[i] = z;
                weights[i] = 2 / ((1 - z * z) * dp * dp);
            }
            return GaussLegendreCache[n] = (nodes, weights);
        }
    }

    static async Task<float[]> Lms(Run run, float[] x, double[] s, int order)
    {
        var ds = new List<float[]>();
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            ds.Add(ToD(x, s[i], denoised));
            if (ds.Count > order) ds.RemoveAt(0);
            await run.CallbackAsync(i, denoised);
            if (s[i + 1] == 0) x = denoised;
            else
            {
                int cur = Math.Min(i + 1, order);
                for (int j = 0; j < cur; j++) AddScaled(x, ds[ds.Count - 1 - j], LinearMultistepCoeff(cur, s, i, j));
            }
        }
        return x;
    }

    // --- DPM-Solver++ family ---------------------------------------------------------------------------------------------------------------

    static async Task<float[]> Dpmpp2SAncestral(Run run, float[] x, double[] s, double eta, double sNoise)
    {
        if (run.Sampling.IsConst) return await Dpmpp2SAncestralRF(run, x, s, eta, sNoise);
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            var (down, up) = AncestralStep(s[i], s[i + 1], eta);
            await run.CallbackAsync(i, denoised);
            if (down == 0) AddScaled(x, ToD(x, s[i], denoised), down - s[i]);
            else
            {
                // DPM-Solver++(2S) in t = -log(sigma): a midpoint evaluation, then the full step with its result.
                double t = -Math.Log(s[i]), tNext = -Math.Log(down), h = tNext - t, sMid = t + 0.5 * h;
                var x2 = Lin(x, Math.Exp(-sMid) / Math.Exp(-t), denoised, -Expm1(-h * 0.5));
                var denoised2 = await run.Call(x2, Math.Exp(-sMid));
                x = Lin(x, Math.Exp(-tNext) / Math.Exp(-t), denoised2, -Expm1(-h));
            }
            if (s[i + 1] > 0) AddScaled(x, run.NextNoise(), sNoise * up);
        }
        return x;
    }

    static async Task<float[]> Dpmpp2SAncestralRF(Run run, float[] x, double[] s, double eta, double sNoise)
    {
        sNoise *= run.Sampling.NoiseScale;
        static double SigmaFn(double l) => 1.0 / (Math.Exp(l) + 1);
        static double LambdaFn(double sigma) => Math.Log((1 - sigma) / sigma);
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            double down = s[i + 1] * (1 + (s[i + 1] / s[i] - 1) * eta);
            double alphaIp1 = 1 - s[i + 1], alphaDown = 1 - down;
            double renoise = Math.Sqrt(s[i + 1] * s[i + 1] - down * down * alphaIp1 * alphaIp1 / (alphaDown * alphaDown));
            await run.CallbackAsync(i, denoised);
            if (s[i + 1] == 0) AddScaled(x, ToD(x, s[i], denoised), down - s[i]);
            else
            {
                double sigmaS;
                if (s[i] == 1.0) sigmaS = 0.9999;
                else
                {
                    double ti = LambdaFn(s[i]), tDown = LambdaFn(down), h = tDown - ti;
                    sigmaS = SigmaFn(ti + 0.5 * h);
                }
                double sRatio = sigmaS / s[i];
                var u = Lin(x, sRatio, denoised, 1 - sRatio);
                var di = await run.Call(u, sigmaS);
                double dRatio = down / s[i];
                x = Lin(x, dRatio, di, 1 - dRatio);
            }
            if (s[i + 1] > 0 && eta > 0) x = Lin(x, alphaIp1 / alphaDown, run.NextNoise(), sNoise * renoise);
        }
        return x;
    }

    static async Task<float[]> Dpmpp2SAncestralCfgPP(Run run, float[] x, double[] s, double eta, double sNoise)
    {
        sNoise *= run.Sampling.NoiseScale;
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            var (down, up) = AncestralStep(s[i], s[i + 1], eta);
            await run.CallbackAsync(i, denoised);
            if (down == 0)
            {
                var d = ToD(x, s[i], run.UncondDenoised(denoised));
                x = Lin(denoised, 1, d, down);
            }
            else
            {
                double t = -Math.Log(s[i]), tNext = -Math.Log(down), h = tNext - t, sMid = t + 0.5 * h;
                var uncond = run.UncondDenoised(denoised);
                var shifted = new float[x.Length];
                for (int k = 0; k < x.Length; k++) shifted[k] = x[k] + (denoised[k] - uncond[k]);
                var x2 = Lin(shifted, Math.Exp(-sMid) / Math.Exp(-t), denoised, -Expm1(-h * 0.5));
                var denoised2 = await run.Call(x2, Math.Exp(-sMid));
                uncond = run.UncondDenoised(denoised2); // the capture is of the latest call, as in ComfyUI
                for (int k = 0; k < x.Length; k++) shifted[k] = x[k] + (denoised[k] - uncond[k]);
                x = Lin(shifted, Math.Exp(-tNext) / Math.Exp(-t), denoised2, -Expm1(-h));
            }
            if (s[i + 1] > 0) AddScaled(x, run.NextNoise(), sNoise * up);
        }
        return x;
    }

    /// <summary>
    /// DPM-Solver++ (stochastic), sample_dpmpp_sde: two noisy stages per step. Its noise is a Brownian path over sigma, so the second stage's noise is correlated with the
    /// first's like k-diffusion's Brownian tree (see <see cref="BrownianNoise"/>).
    /// </summary>
    static async Task<float[]> DpmppSde(Run run, float[] x, double[] s, double eta, double sNoise, double r)
    {
        if (s.Length <= 1) return x;
        double sigmaMin = s.Where(v => v > 0).Min(), sigmaMax = s.Max();
        var brownian = new BrownianNoise(run.Noise, sigmaMin, sigmaMax);
        s = run.OffsetFirstSigma(s);
        sNoise *= run.Sampling.NoiseScale;
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            await run.CallbackAsync(i, denoised);
            if (s[i + 1] == 0) { x = denoised; continue; }
            double lambdaS = run.Lambda(s[i]), lambdaT = run.Lambda(s[i + 1]), h = lambdaT - lambdaS;
            double lambdaS1 = lambdaS + r * h, fac = 1 / (2 * r);
            double sigmaS1 = run.SigmaOfLambda(lambdaS1);
            double alphaS = s[i] * Math.Exp(lambdaS), alphaS1 = sigmaS1 * Math.Exp(lambdaS1), alphaT = s[i + 1] * Math.Exp(lambdaT);

            // Step 1
            var (sd, su) = AncestralStep(Math.Exp(-lambdaS), Math.Exp(-lambdaS1), eta);
            double h_ = -Math.Log(sd) - lambdaS;
            var x2 = Lin(x, alphaS1 / alphaS * Math.Exp(-h_), denoised, -alphaS1 * Expm1(-h_));
            if (eta > 0 && sNoise > 0) AddScaled(x2, brownian.Increment(s[i], sigmaS1), alphaS1 * sNoise * su);
            var denoised2 = await run.Call(x2, sigmaS1);

            // Step 2
            (sd, su) = AncestralStep(Math.Exp(-lambdaS), Math.Exp(-lambdaT), eta);
            h_ = -Math.Log(sd) - lambdaS;
            var denoisedD = Lin(denoised, 1 - fac, denoised2, fac);
            x = Lin(x, alphaT / alphaS * Math.Exp(-h_), denoisedD, -alphaT * Expm1(-h_));
            if (eta > 0 && sNoise > 0) AddScaled(x, brownian.Increment(s[i], s[i + 1]), alphaT * sNoise * su);
        }
        return x;
    }

    static async Task<float[]> Dpmpp2M(Run run, float[] x, double[] s)
    {
        float[]? oldDenoised = null;
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            await run.CallbackAsync(i, denoised);
            double t = -Math.Log(s[i]), tNext = -Math.Log(s[i + 1]), h = tNext - t;
            double scale = Math.Exp(-tNext) / Math.Exp(-t), coeff = -Expm1(-h);
            if (oldDenoised is null || s[i + 1] == 0) x = Lin(x, scale, denoised, coeff);
            else
            {
                double hLast = t - (-Math.Log(s[i - 1])), r = hLast / h;
                var denoisedD = Lin(denoised, 1 + 1 / (2 * r), oldDenoised, -1 / (2 * r));
                x = Lin(x, scale, denoisedD, coeff);
            }
            oldDenoised = denoised;
        }
        return x;
    }

    static async Task<float[]> Dpmpp2MCfgPP(Run run, float[] x, double[] s)
    {
        float[]? oldUncond = null;
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            var uncond = run.UncondDenoised(denoised);
            await run.CallbackAsync(i, denoised);
            double t = -Math.Log(s[i]), tNext = -Math.Log(s[i + 1]), h = tNext - t, keep = Math.Exp(-h);
            var next = new float[x.Length];
            if (oldUncond is null || s[i + 1] == 0)
                for (int k = 0; k < x.Length; k++) next[k] = (float)(denoised[k] - keep * uncond[k] + keep * x[k]);
            else
            {
                double hLast = t - (-Math.Log(s[i - 1])), r = hLast / h, c = Expm1(-h) * (1 / (2 * r));
                for (int k = 0; k < x.Length; k++)
                    next[k] = (float)(denoised[k] + (-keep * uncond[k] - c * (denoised[k] - oldUncond[k])) + keep * x[k]);
            }
            x = next;
            oldUncond = uncond;
        }
        return x;
    }

    static async Task<float[]> Dpmpp2MSde(Run run, float[] x, double[] s, double eta, double sNoise, string solverType)
    {
        if (s.Length <= 1) return x;
        if (solverType is not ("heun" or "midpoint")) throw new ArgumentException("solver_type must be 'heun' or 'midpoint'");
        s = run.OffsetFirstSigma(s);
        sNoise *= run.Sampling.NoiseScale;
        float[]? oldDenoised = null;
        double h = 0, hLast = 0;
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            await run.CallbackAsync(i, denoised);
            if (s[i + 1] == 0) x = denoised;
            else
            {
                double lambdaS = run.Lambda(s[i]), lambdaT = run.Lambda(s[i + 1]);
                h = lambdaT - lambdaS;
                double hEta = h * (eta + 1), alphaT = s[i + 1] * Math.Exp(lambdaT), mix = -Expm1(-hEta);
                x = Lin(x, s[i + 1] / s[i] * Math.Exp(-h * eta), denoised, alphaT * mix);
                if (oldDenoised is not null)
                {
                    double r = hLast / h;
                    var diff = Sub(denoised, oldDenoised);
                    if (solverType == "heun") AddScaled(x, diff, alphaT * (mix / (-hEta) + 1) * (1 / r));
                    else AddScaled(x, diff, 0.5 * alphaT * mix * (1 / r));
                }
                // The noise intervals of consecutive steps do not overlap, so independent draws have the Brownian tree's distribution exactly.
                if (eta > 0 && sNoise > 0) AddScaled(x, run.NextNoise(), s[i + 1] * Math.Sqrt(-Expm1(-2 * h * eta)) * sNoise);
            }
            oldDenoised = denoised;
            hLast = h;
        }
        return x;
    }

    static async Task<float[]> Dpmpp3MSde(Run run, float[] x, double[] s, double eta, double sNoise)
    {
        if (s.Length <= 1) return x;
        s = run.OffsetFirstSigma(s);
        sNoise *= run.Sampling.NoiseScale;
        float[]? denoised1 = null, denoised2 = null;
        double? h = null, h1 = null, h2 = null;
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            await run.CallbackAsync(i, denoised);
            if (s[i + 1] == 0) x = denoised;
            else
            {
                double lambdaS = run.Lambda(s[i]), lambdaT = run.Lambda(s[i + 1]), hh = lambdaT - lambdaS;
                h = hh;
                double hEta = hh * (eta + 1), alphaT = s[i + 1] * Math.Exp(lambdaT);
                x = Lin(x, s[i + 1] / s[i] * Math.Exp(-hh * eta), denoised, alphaT * -Expm1(-hEta));
                if (h2 is not null)
                {
                    double r0 = h1!.Value / hh, r1 = h2.Value / hh;
                    double phi2 = Expm1(-hEta) / hEta + 1, phi3 = phi2 / hEta - 0.5;
                    for (int k = 0; k < x.Length; k++)
                    {
                        double d10 = (denoised[k] - denoised1![k]) / r0, d11 = (denoised1[k] - denoised2![k]) / r1;
                        double d1 = d10 + (d10 - d11) * r0 / (r0 + r1), d2 = (d10 - d11) / (r0 + r1);
                        x[k] = (float)(x[k] + alphaT * phi2 * d1 - alphaT * phi3 * d2);
                    }
                }
                else if (h1 is not null)
                {
                    double r = h1.Value / hh, phi2 = Expm1(-hEta) / hEta + 1;
                    for (int k = 0; k < x.Length; k++) x[k] = (float)(x[k] + alphaT * phi2 * ((denoised[k] - denoised1![k]) / r));
                }
                if (eta > 0 && sNoise > 0) AddScaled(x, run.NextNoise(), s[i + 1] * Math.Sqrt(-Expm1(-2 * hh * eta)) * sNoise);
            }
            (denoised1, denoised2) = (denoised, denoised1);
            (h1, h2) = (h, h1);
        }
        return x;
    }

    // --- DDPM, LCM ------------------------------------------------------------------------------------------------------------------------

    /// <summary>The ancestral DDPM step in the variance-preserving space (generic_step_sampler with DDPMSampler_step).</summary>
    static async Task<float[]> Ddpm(Run run, float[] x, double[] s)
    {
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            await run.CallbackAsync(i, denoised);
            double sigma = s[i], prev = s[i + 1];
            double ac = 1 / (sigma * sigma + 1), acPrev = 1 / (prev * prev + 1), alpha = ac / acPrev;
            double inScale = 1 / Math.Sqrt(1.0 + sigma * sigma), muScale = Math.Sqrt(1.0 / alpha), noiseCoeff = (1 - alpha) / Math.Sqrt(1 - ac);
            var mu = new float[x.Length];
            for (int k = 0; k < x.Length; k++)
            {
                double eps = (x[k] - denoised[k]) / sigma;
                mu[k] = (float)(muScale * (x[k] * inScale - noiseCoeff * eps));
            }
            if (prev > 0) AddScaled(mu, run.NextNoise(), Math.Sqrt((1 - alpha) * (1 - acPrev) / (1 - ac)));
            x = mu;
            if (prev != 0)
                for (int k = 0; k < x.Length; k++) x[k] = (float)(x[k] * Math.Sqrt(1.0 + prev * prev));
        }
        return x;
    }

    /// <summary>Latent consistency sampling: jump to the denoised estimate, then re-noise to the next level (s_noise may ramp to s_noise_end; the noise may be clamped).</summary>
    static async Task<float[]> Lcm(Run run, float[] x, double[] s, double sNoise, double? sNoiseEnd, double noiseClipStd)
    {
        int nSteps = Math.Max(1, s.Length - 1);
        double sStart = sNoise, sEnd = sNoiseEnd ?? sStart;
        for (int i = 0; i < nSteps; i++)
        {
            var denoised = await run.Call(x, s[i]);
            await run.CallbackAsync(i, denoised);
            x = denoised;
            if (s[i + 1] > 0)
            {
                var noise = run.NextNoise();
                if (noiseClipStd > 0)
                {
                    double mean = noise.Average(v => (double)v), var = noise.Sum(v => (v - mean) * (v - mean)) / (noise.Length - 1);
                    float clip = (float)(noiseClipStd * Math.Sqrt(var));
                    noise = noise.Select(v => Math.Clamp(v, -clip, clip)).ToArray();
                }
                double t = nSteps > 1 ? (double)i / (nSteps - 1) : 0.0, si = sStart + (sEnd - sStart) * t;
                if (si != 1.0) noise = noise.Select(v => (float)(v * si)).ToArray();
                x = run.Sampling.NoiseScaling(s[i + 1], noise, x);
            }
        }
        return x;
    }

    // --- PNDM / multistep -----------------------------------------------------------------------------------------------------------------

    /// <summary>Improved PNDM (sample_ipndm): Adams-Bashforth of order up to 4 on the derivative with fixed coefficients.</summary>
    static async Task<float[]> Ipndm(Run run, float[] x, double[] s)
    {
        const int maxOrder = 4;
        var buffer = new List<float[]>();
        for (int i = 0; i < s.Length - 1; i++)
        {
            double tCur = s[i], tNext = s[i + 1];
            var denoised = await run.Call(x, tCur);
            await run.CallbackAsync(i, denoised);
            var d = ToD(x, tCur, denoised);
            int order = Math.Min(maxOrder, i + 1);
            double dt = tNext - tCur;
            if (tNext == 0) x = denoised;
            else
            {
                var next = new float[x.Length];
                for (int k = 0; k < x.Length; k++)
                {
                    double step = order switch
                    {
                        1 => d[k],
                        2 => (3.0 * d[k] - buffer[^1][k]) / 2,
                        3 => (23.0 * d[k] - 16.0 * buffer[^1][k] + 5.0 * buffer[^2][k]) / 12,
                        _ => (55.0 * d[k] - 59.0 * buffer[^1][k] + 37.0 * buffer[^2][k] - 9.0 * buffer[^3][k]) / 24,
                    };
                    next[k] = (float)(x[k] + dt * step);
                }
                x = next;
            }
            PushHistory(buffer, d, maxOrder - 1);
        }
        return x;
    }

    /// <summary>The history buffer of ipndm / deis: the last <paramref name="capacity"/> derivatives, oldest first.</summary>
    static void PushHistory(List<float[]> buffer, float[] d, int capacity)
    {
        if (buffer.Count == capacity) { buffer.RemoveAt(0); buffer.Add(d); }
        else buffer.Add(d);
    }

    /// <summary>Improved PNDM with variable step sizes (sample_ipndm_v).</summary>
    static async Task<float[]> IpndmV(Run run, float[] x, double[] s)
    {
        const int maxOrder = 4;
        var buffer = new List<float[]>();
        for (int i = 0; i < s.Length - 1; i++)
        {
            double tCur = s[i], tNext = s[i + 1];
            var denoised = await run.Call(x, tCur);
            await run.CallbackAsync(i, denoised);
            var d = ToD(x, tCur, denoised);
            int order = Math.Min(maxOrder, i + 1);
            if (tNext == 0) x = denoised;
            else
            {
                double hn = tNext - tCur, c1, c2 = 0, c3 = 0, c4 = 0;
                if (order == 1) c1 = 1;
                else if (order == 2)
                {
                    double hn1 = tCur - s[i - 1];
                    c1 = (2 + hn / hn1) / 2;
                    c2 = -(hn / hn1) / 2;
                }
                else if (order == 3)
                {
                    double hn1 = tCur - s[i - 1], hn2 = s[i - 1] - s[i - 2];
                    double temp = (1 - hn / (3 * (hn + hn1)) * (hn * (hn + hn1)) / (hn1 * (hn1 + hn2))) / 2;
                    c1 = (2 + hn / hn1) / 2 + temp;
                    c2 = -(hn / hn1) / 2 - (1 + hn1 / hn2) * temp;
                    c3 = temp * hn1 / hn2;
                }
                else
                {
                    double hn1 = tCur - s[i - 1], hn2 = s[i - 1] - s[i - 2], hn3 = s[i - 2] - s[i - 3];
                    double temp1 = (1 - hn / (3 * (hn + hn1)) * (hn * (hn + hn1)) / (hn1 * (hn1 + hn2))) / 2;
                    double temp2 = ((1 - hn / (3 * (hn + hn1))) / 2 + (1 - hn / (2 * (hn + hn1))) * hn / (6 * (hn + hn1 + hn2)))
                        * (hn * (hn + hn1) * (hn + hn1 + hn2)) / (hn1 * (hn1 + hn2) * (hn1 + hn2 + hn3));
                    c1 = (2 + hn / hn1) / 2 + temp1 + temp2;
                    c2 = -(hn / hn1) / 2 - (1 + hn1 / hn2) * temp1 - (1 + hn1 / hn2 + hn1 * (hn1 + hn2) / (hn2 * (hn2 + hn3))) * temp2;
                    c3 = temp1 * hn1 / hn2 + (hn1 / hn2 + hn1 * (hn1 + hn2) / (hn2 * (hn2 + hn3)) * (1 + hn2 / hn3)) * temp2;
                    c4 = -temp2 * (hn1 * (hn1 + hn2) / (hn2 * (hn2 + hn3))) * hn1 / hn2;
                }
                var next = new float[x.Length];
                for (int k = 0; k < x.Length; k++)
                {
                    double sum = c1 * d[k];
                    if (order >= 2) sum += c2 * buffer[^1][k];
                    if (order >= 3) sum += c3 * buffer[^2][k];
                    if (order >= 4) sum += c4 * buffer[^3][k];
                    next[k] = (float)(x[k] + hn * sum);
                }
                x = next;
            }
            PushHistory(buffer, d, maxOrder - 1);
        }
        return x;
    }

    /// <summary>Second-order multistep exponential integrator (RES, arXiv 2308.02157; ComfyUI's res_multistep and its ancestral / CFG++ forms).</summary>
    static async Task<float[]> ResMultistep(Run run, float[] x, double[] s, double sNoise, double eta, bool cfgPP)
    {
        sNoise *= run.Sampling.NoiseScale;
        double oldSigmaDown = 0;
        float[]? oldDenoised = null;
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            var (down, up) = AncestralStep(s[i], s[i + 1], eta);
            await run.CallbackAsync(i, denoised);
            float[]? uncond = cfgPP ? run.UncondDenoised(denoised) : null;
            if (down == 0 || oldDenoised is null)
            {
                if (cfgPP) x = Lin(denoised, 1, ToD(x, s[i], uncond!), down);
                else AddScaled(x, ToD(x, s[i], denoised), down - s[i]);
            }
            else
            {
                double t = -Math.Log(s[i]), tOld = -Math.Log(oldSigmaDown), tNext = -Math.Log(down), tPrev = -Math.Log(s[i - 1]);
                double h = tNext - t, c2 = (tPrev - tOld) / h;
                double phi1 = Expm1(-h) / -h, phi2 = (phi1 - 1.0) / -h;
                double b1 = phi1 - phi2 / c2, b2 = phi2 / c2;
                if (double.IsNaN(b1)) b1 = 0;
                if (double.IsNaN(b2)) b2 = 0;
                double keep = Math.Exp(-h);
                var source = denoised;
                if (cfgPP)
                {
                    for (int k = 0; k < x.Length; k++) x[k] = x[k] + (denoised[k] - uncond![k]);
                    source = uncond!;
                }
                var next = new float[x.Length];
                for (int k = 0; k < x.Length; k++) next[k] = (float)(keep * x[k] + h * (b1 * source[k] + b2 * oldDenoised[k]));
                x = next;
            }
            if (up > 0) AddScaled(x, run.NextNoise(), sNoise * up);
            oldDenoised = cfgPP ? uncond : denoised;
            oldSigmaDown = down;
        }
        return x;
    }

    /// <summary>Gradient estimation (sample_gradient_estimation, gamma 2): an Euler step plus an extrapolation of how the derivative changed.</summary>
    static async Task<float[]> GradientEstimation(Run run, float[] x, double[] s, double gamma, bool cfgPP)
    {
        float[]? oldD = null;
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            var d = ToD(x, s[i], cfgPP ? run.UncondDenoised(denoised) : denoised);
            await run.CallbackAsync(i, denoised);
            double dt = s[i + 1] - s[i];
            if (s[i + 1] == 0) x = denoised;
            else
            {
                if (cfgPP) x = Lin(denoised, 1, d, s[i + 1]);
                else AddScaled(x, d, dt);
                if (i >= 1)
                    for (int k = 0; k < x.Length; k++) x[k] = (float)(x[k] + (gamma - 1) * (d[k] - oldD![k]) * dt);
            }
            oldD = d;
        }
        return x;
    }
}
