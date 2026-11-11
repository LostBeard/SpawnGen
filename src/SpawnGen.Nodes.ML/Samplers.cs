using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md). The k-diffusion samplers here follow Katherine Crowson's k-diffusion (MIT, see THIRD-PARTY-NOTICES.md)
// and its papers: Karras et al. 2022 (Euler, Heun, churn), Lu et al. 2022 (DPM-Solver, DPM-Solver++ 2S / 2M / SDE), the linear multistep method.
// Draw order, call counts and last-step details were fitted to Fixtures/sampling/reference.json (ComfyUI's samplers on a stand-in denoiser with
// recorded noise). The rest of the samplers are in SamplersAdvanced.cs.

/// <summary>The samplers: each runs a denoiser from the first sigma down the schedule.</summary>
public static partial class Samplers
{
    /// <summary>The sampler names, in ComfyUI's order.</summary>
    public static readonly string[] Names =
    [
        "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2", "exp_heun_2_x0", "exp_heun_2_x0_sde", "dpm_2",
        "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m",
        "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm",
        "lcm", "ipndm", "ipndm_v", "deis", "cfgpp_ud10_ab", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral",
        "res_multistep_ancestral_cfg_pp", "gradient_estimation", "gradient_estimation_cfg_pp", "er_sde", "seeds_2", "seeds_3", "sa_solver",
        "sa_solver_pece", "ddim", "uni_pc", "uni_pc_bh2",
    ];

    /// <summary>The "alternative" Euler CFG++ of SamplerEulerCFGpp (not in the sampler list).</summary>
    public const string EulerPP = "euler_pp";

    /// <summary>Samplers that KSampler runs one step longer with the penultimate sigma dropped.</summary>
    public static bool DiscardsPenultimateSigma(string sampler) => sampler is "dpm_2" or "dpm_2_ancestral" or "uni_pc" or "uni_pc_bh2";

    /// <summary>The CFG++ samplers, which read the unconditional estimate.</summary>
    public static bool UsesUncond(string sampler) => sampler.EndsWith("_cfg_pp", StringComparison.Ordinal) || sampler is "cfgpp_ud10_ab" or EulerPP;

    /// <summary>Samplers that need the negative evaluated even at cfg 1 (the CFG++ ones).</summary>
    public static bool DisablesCfg1Optimization(string sampler) => UsesUncond(sampler);

    /// <summary>The denoiser: the denoised estimate of <paramref name="x"/> at <paramref name="sigma"/>.</summary>
    public delegate Task<float[]> DenoiseFn(float[] x, float sigma);

    /// <summary>
    /// A seeded source of standard normal draws, one per call, each the size of a picture (PyTorch's CPU generator: the draws continue one
    /// stream). A batch shares the stream: draw i is the slice of the i-th whole-batch draw.
    /// </summary>
    public class NoiseSource
    {
        readonly TorchRandom? _random;
        readonly int _size, _batch, _index;
        readonly List<float[]> _draws = [];
        int _next;

        public NoiseSource(int seed, int channels, int height, int width) : this(seed, 1, 0, channels, height, width) { }

        internal NoiseSource(long seed, int batch, int index, int channels, int height, int width)
        {
            _size = channels * height * width;
            _batch = Math.Max(batch, 1);
            _index = index;
            if (_size > 0) _random = new TorchRandom(seed);
        }

        /// <summary>Draw <paramref name="index"/> (0 based).</summary>
        public virtual float[] At(int index)
        {
            while (_draws.Count <= index)
            {
                var all = _random!.Randn(_size * _batch);
                _draws.Add(_batch == 1 ? all : all.AsSpan(_index * _size, _size).ToArray());
            }
            return _draws[index];
        }

        /// <summary>The next draw.</summary>
        public float[] Next() => At(_next++);
    }

    /// <summary>
    /// A Brownian path over sigma for the SDE samplers (a stand-in for torchsde's Brownian tree): W(sigma_min) = 0, W(sigma_max) from the first
    /// draw, any other point drawn on the bridge between its known neighbours (or beyond the ends), so increments over nested intervals correlate.
    /// </summary>
    internal sealed class BrownianNoise
    {
        readonly NoiseSource _source;
        readonly SortedList<double, float[]> _points = new();

        public BrownianNoise(NoiseSource source, double sigmaMin, double sigmaMax)
        {
            _source = source;
            var first = source.Next();
            _points[sigmaMin] = new float[first.Length];
            if (sigmaMax > sigmaMin) _points[sigmaMax] = Scale(first, Math.Sqrt(sigmaMax - sigmaMin));
        }

        /// <summary>W(tau).</summary>
        public float[] At(double tau)
        {
            var keys = _points.Keys;
            foreach (var k in keys)
                if (Math.Abs(k - tau) <= 1e-6 * Math.Max(Math.Abs(k), 1e-12)) return _points[k];
            var z = _source.Next();
            int hi = 0;
            while (hi < keys.Count && keys[hi] < tau) hi++;
            float[] w;
            if (hi == 0) w = Add(_points[keys[0]], z, Math.Sqrt(keys[0] - tau));
            else if (hi == keys.Count) w = Add(_points[keys[^1]], z, Math.Sqrt(tau - keys[^1]));
            else
            {
                double a = keys[hi - 1], b = keys[hi];
                float frac = (float)((tau - a) / (b - a)), std = (float)Math.Sqrt((tau - a) * (b - tau) / (b - a));
                float[] pa = _points[a], pb = _points[b];
                w = new float[pa.Length];
                for (int i = 0; i < w.Length; i++) w[i] = pa[i] + frac * (pb[i] - pa[i]) + std * z[i];
            }
            _points[tau] = w;
            return w;
        }

        /// <summary>The normalised increment (W(next) - W(sigma)) / sqrt(|next - sigma|): a standard normal draw.</summary>
        public float[] Increment(double sigma, double sigmaNext)
        {
            var w0 = At(sigma);
            var w1 = At(sigmaNext);
            float s = (float)Math.Sqrt(Math.Abs(sigmaNext - sigma));
            var r = new float[w0.Length];
            for (int i = 0; i < r.Length; i++) r[i] = (w1[i] - w0[i]) / s;
            return r;
        }

        static float[] Scale(float[] a, double k)
        {
            float f = (float)k;
            var r = new float[a.Length];
            for (int i = 0; i < r.Length; i++) r[i] = a[i] * f;
            return r;
        }

        static float[] Add(float[] a, float[] z, double k)
        {
            float f = (float)k;
            var r = new float[a.Length];
            for (int i = 0; i < r.Length; i++) r[i] = a[i] + z[i] * f;
            return r;
        }
    }

    /// <summary>What a sampler works with: the denoiser (counting calls), the noise, the callbacks and the options.</summary>
    internal sealed class Run(DenoiseFn model, NoiseSource noise, Action<int>? onStep, CancellationToken ct, Func<GuidanceOutput?>? uncond,
        SamplerOptions options, ModelSampling sampling, Func<float[], Task>? onDenoised)
    {
        public NoiseSource Noise { get; } = noise;
        public SamplerOptions Options { get; } = options;
        public ModelSampling Sampling { get; } = sampling;
        public CancellationToken Token { get; } = ct;

        public Task<float[]> Denoise(float[] x, double sigma) => model(x, (float)sigma);

        /// <summary>The unconditional estimate of the last prediction (zeros when there was none: a CFG++ sampler then acts like its plain form).</summary>
        public float[] Uncond(float[] denoised) => uncond?.Invoke() is { } u ? u.UncondDenoised : denoised;

        public Task Callback(float[] denoised) => onDenoised?.Invoke(denoised) ?? Task.CompletedTask;

        public void Step(int i) => onStep?.Invoke(i + 1);

        public void Check() => ct.ThrowIfCancellationRequested();

        public double Eta(double fallback = 1.0) => Options.Eta ?? fallback;
        public double SNoise => Options.SNoise ?? 1.0;
    }

    /// <summary>
    /// Runs <paramref name="sampler"/> from <paramref name="x"/> (already noised to the first sigma) down <paramref name="sigmas"/>.
    /// <paramref name="onStep"/> gets 1, 2, ... after each step, <paramref name="onDenoised"/> each step's denoised estimate.
    /// </summary>
    public static async Task<float[]> RunAsync(string sampler, float[] x, float[] sigmas, DenoiseFn model, NoiseSource noise, Action<int>? onStep = null,
        CancellationToken ct = default, Func<GuidanceOutput?>? uncond = null, SamplerOptions? options = null, ModelSampling? sampling = null,
        Func<float[], Task>? onDenoised = null)
    {
        if (!Names.Contains(sampler) && sampler != EulerPP)
            throw new ArgumentException($"Unknown sampler '{sampler}'. Known: {string.Join(", ", Names)}.", nameof(sampler));
        ct.ThrowIfCancellationRequested();
        var run = new Run(model, noise, onStep, ct, uncond, options ?? new SamplerOptions(), sampling ?? DefaultSampling.Value, onDenoised);
        x = (float[])x.Clone();
        if (sigmas.Length < 2) return x;
        var s = sigmas.Select(v => (double)v).ToArray();
        return sampler switch
        {
            "euler" or "ddim" => await Euler(run, x, s),
            "euler_ancestral" => await EulerAncestral(run, x, s, cfgPP: false),
            "euler_ancestral_cfg_pp" => await EulerAncestral(run, x, s, cfgPP: true),
            "heun" => await Heun(run, x, s),
            "dpm_2" => await Dpm2(run, x, s),
            "dpm_2_ancestral" => await Dpm2Ancestral(run, x, s),
            "lms" => await Lms(run, x, s, run.Options.Order ?? 4),
            "dpm_fast" => await DpmFast(run, x, s),
            "dpm_adaptive" => await DpmAdaptive(run, x, s),
            "dpmpp_2s_ancestral" => await Dpmpp2SAncestral(run, x, s, cfgPP: false),
            "dpmpp_2s_ancestral_cfg_pp" => await Dpmpp2SAncestral(run, x, s, cfgPP: true),
            "dpmpp_sde" or "dpmpp_sde_gpu" => await DpmppSde(run, x, s),
            "dpmpp_2m" => await Dpmpp2M(run, x, s, cfgPP: false),
            "dpmpp_2m_cfg_pp" => await Dpmpp2M(run, x, s, cfgPP: true),
            "dpmpp_2m_sde" or "dpmpp_2m_sde_gpu" => await Dpmpp2MSde(run, x, s, run.Options.SolverType ?? "midpoint"),
            "dpmpp_2m_sde_heun" or "dpmpp_2m_sde_heun_gpu" => await Dpmpp2MSde(run, x, s, run.Options.SolverType ?? "heun"),
            "dpmpp_3m_sde" or "dpmpp_3m_sde_gpu" => await Dpmpp3MSde(run, x, s),
            _ => await RunAdvanced(sampler, run, x, s),
        };
    }

    static readonly Lazy<ModelSampling> DefaultSampling = new(() => new ModelSampling(ModelSigmas.LinearSchedule(), PredictionType.Eps));

    // ---- array helpers (float32 like the tensors they stand for) ----

    public static float[] Subtract(float[] a, float[] b)
    {
        var r = new float[a.Length];
        for (int i = 0; i < r.Length; i++) r[i] = a[i] - b[i];
        return r;
    }

    /// <summary>a ca + b cb.</summary>
    public static float[] LinCombine(float[] a, double ca, float[] b, double cb)
    {
        float fa = (float)ca, fb = (float)cb;
        var r = new float[a.Length];
        for (int i = 0; i < r.Length; i++) r[i] = a[i] * fa + b[i] * fb;
        return r;
    }

    /// <summary>Classifier-free guidance: uncond + (cond - uncond) scale.</summary>
    public static float[] Guide(float[] uncond, float[] cond, double scale)
    {
        float k = (float)scale;
        var r = new float[cond.Length];
        for (int i = 0; i < r.Length; i++) r[i] = uncond[i] + (cond[i] - uncond[i]) * k;
        return r;
    }

    /// <summary>The derivative (x - denoised) / sigma.</summary>
    static float[] ToD(float[] x, double sigma, float[] denoised)
    {
        float s = (float)sigma;
        var r = new float[x.Length];
        for (int i = 0; i < r.Length; i++) r[i] = (x[i] - denoised[i]) / s;
        return r;
    }

    /// <summary>x + d dt.</summary>
    static float[] Step(float[] x, float[] d, double dt)
    {
        float f = (float)dt;
        var r = new float[x.Length];
        for (int i = 0; i < r.Length; i++) r[i] = x[i] + d[i] * f;
        return r;
    }

    /// <summary>x + noise k (in place).</summary>
    static void AddNoise(float[] x, float[] noise, double k)
    {
        float f = (float)k;
        for (int i = 0; i < x.Length; i++) x[i] += noise[i] * f;
    }

    /// <summary>k-diffusion's ancestral split of a step: (sigma_down, sigma_up) with sigma_down^2 + sigma_up^2 = sigma_to^2.</summary>
    internal static (double Down, double Up) AncestralStep(double sigmaFrom, double sigmaTo, double eta = 1.0)
    {
        if (eta == 0) return (sigmaTo, 0);
        float from = (float)sigmaFrom, to = (float)sigmaTo;
        float up = MathF.Min(to, (float)eta * MathF.Sqrt(to * to * (from * from - to * to) / (from * from)));
        float down = MathF.Sqrt(to * to - up * up);
        return (down, up);
    }

    static float Log(double v) => MathF.Log((float)v);
    static float Exp(double v) => MathF.Exp((float)v);

    // ---- the k-diffusion samplers ----

    static async Task<float[]> Euler(Run run, float[] x, double[] s)
    {
        for (int i = 0; i < s.Length - 1; i++)
        {
            run.Check();
            var denoised = await run.Denoise(x, s[i]);
            var d = ToD(x, s[i], denoised);
            await run.Callback(denoised);
            x = Step(x, d, (float)s[i + 1] - (float)s[i]);
            run.Step(i);
        }
        return x;
    }

    static async Task<float[]> EulerAncestral(Run run, float[] x, double[] s, bool cfgPP)
    {
        double eta = run.Eta();
        for (int i = 0; i < s.Length - 1; i++)
        {
            run.Check();
            var denoised = await run.Denoise(x, s[i]);
            await run.Callback(denoised);
            if (run.Sampling.IsConst)
            {
                x = await FlowAncestralStep(run, x, denoised, s[i], s[i + 1], eta, cfgPP ? run.Uncond(denoised) : null);
                run.Step(i);
                continue;
            }
            var (down, up) = AncestralStep(s[i], s[i + 1], eta);
            if (cfgPP)
            {
                var u = run.Uncond(denoised);
                var d = ToD(x, s[i], u);
                x = Step(denoised, d, down);
            }
            else x = Step(x, ToD(x, s[i], denoised), down - (float)s[i]);
            if (s[i + 1] > 0) AddNoise(x, run.Noise.Next(), run.SNoise * up);
            run.Step(i);
        }
        return x;
    }

    static async Task<float[]> Heun(Run run, float[] x, double[] s)
    {
        for (int i = 0; i < s.Length - 1; i++)
        {
            run.Check();
            var denoised = await run.Denoise(x, s[i]);
            var d = ToD(x, s[i], denoised);
            await run.Callback(denoised);
            float dt = (float)s[i + 1] - (float)s[i];
            if (s[i + 1] == 0) x = Step(x, d, dt);
            else
            {
                var x2 = Step(x, d, dt);
                var denoised2 = await run.Denoise(x2, s[i + 1]);
                var d2 = ToD(x2, s[i + 1], denoised2);
                var dPrime = new float[d.Length];
                for (int k = 0; k < d.Length; k++) dPrime[k] = (d[k] + d2[k]) / 2;
                x = Step(x, dPrime, dt);
            }
            run.Step(i);
        }
        return x;
    }

    static async Task<float[]> Dpm2(Run run, float[] x, double[] s)
    {
        for (int i = 0; i < s.Length - 1; i++)
        {
            run.Check();
            var denoised = await run.Denoise(x, s[i]);
            var d = ToD(x, s[i], denoised);
            await run.Callback(denoised);
            if (s[i + 1] == 0) x = Step(x, d, (float)s[i + 1] - (float)s[i]);
            else
            {
                float mid = MathF.Exp(Lerp(Log(s[i]), Log(s[i + 1]), 0.5f));
                var x2 = Step(x, d, mid - (float)s[i]);
                var denoised2 = await run.Denoise(x2, mid);
                var d2 = ToD(x2, mid, denoised2);
                x = Step(x, d2, (float)s[i + 1] - (float)s[i]);
            }
            run.Step(i);
        }
        return x;
    }

    static async Task<float[]> Dpm2Ancestral(Run run, float[] x, double[] s)
    {
        double eta = run.Eta();
        for (int i = 0; i < s.Length - 1; i++)
        {
            run.Check();
            var denoised = await run.Denoise(x, s[i]);
            await run.Callback(denoised);
            if (run.Sampling.IsConst)
            {
                x = await FlowDpm2AncestralStep(run, x, denoised, s[i], s[i + 1], eta);
                run.Step(i);
                continue;
            }
            var (down, up) = AncestralStep(s[i], s[i + 1], eta);
            var d = ToD(x, s[i], denoised);
            if (down == 0) x = Step(x, d, down - (float)s[i]);
            else
            {
                float mid = MathF.Exp(Lerp(Log(s[i]), Log(down), 0.5f));
                var x2 = Step(x, d, mid - (float)s[i]);
                var denoised2 = await run.Denoise(x2, mid);
                var d2 = ToD(x2, mid, denoised2);
                x = Step(x, d2, (float)down - (float)s[i]);
            }
            if (s[i + 1] > 0) AddNoise(x, run.Noise.Next(), run.SNoise * up);
            run.Step(i);
        }
        return x;
    }

    /// <summary>torch.lerp's float formula (from the nearer end).</summary>
    static float Lerp(float a, float b, float w) => w < 0.5f ? a + w * (b - a) : b - (b - a) * (1 - w);

    /// <summary>The weight of derivative j in an order-<paramref name="order"/> linear multistep step from t[i] to t[i + 1] (the Lagrange basis integrated).</summary>
    public static double LinearMultistepCoeff(int order, double[] t, int i, int j)
    {
        if (order - 1 > i) throw new ArgumentException($"Order {order} is too high for step {i}.");
        double Basis(double tau)
        {
            double prod = 1;
            for (int k = 0; k < order; k++)
                if (k != j) prod *= (tau - t[i - k]) / (t[i - j] - t[i - k]);
            return prod;
        }
        return GaussLegendre(Basis, t[i], t[i + 1], Math.Max(order, 2));
    }

    /// <summary>The integral of <paramref name="f"/> over [a, b] by n-point Gauss-Legendre (exact for polynomials of degree 2n - 1).</summary>
    internal static double GaussLegendre(Func<double, double> f, double a, double b, int n)
    {
        double sum = 0, half = (b - a) / 2, mid = (a + b) / 2;
        for (int k = 1; k <= n; k++)
        {
            // Newton's method on P_n from the Chebyshev-like first guess.
            double z = Math.Cos(Math.PI * (k - 0.25) / (n + 0.5)), dp;
            for (int it = 0; ; it++)
            {
                double p0 = 1, p1 = z;
                for (int m = 2; m <= n; m++) { double p2 = ((2 * m - 1) * z * p1 - (m - 1) * p0) / m; p0 = p1; p1 = p2; }
                double pn = n == 1 ? z : p1, pm = n == 1 ? 1 : p0;
                dp = n * (z * pn - pm) / (z * z - 1);
                double dz = pn / dp;
                z -= dz;
                if (Math.Abs(dz) < 1e-15 || it > 100) break;
            }
            double w = 2 / ((1 - z * z) * dp * dp);
            sum += w * f(mid + half * z);
        }
        return sum * half;
    }

    static async Task<float[]> Lms(Run run, float[] x, double[] s, int order)
    {
        // The sigmas as float32 values, the coefficients in double (as scipy's quad over a float32 array).
        var t = s.Select(v => (double)(float)v).ToArray();
        var ds = new List<float[]>();
        for (int i = 0; i < s.Length - 1; i++)
        {
            run.Check();
            var denoised = await run.Denoise(x, s[i]);
            var d = ToD(x, s[i], denoised);
            ds.Add(d);
            if (ds.Count > order) ds.RemoveAt(0);
            await run.Callback(denoised);
            if (s[i + 1] == 0)
            {
                x = denoised;
                run.Step(i);
                continue;
            }
            int cur = Math.Min(i + 1, order);
            var sum = new float[x.Length];
            for (int j = 0; j < cur; j++)
            {
                float c = (float)LinearMultistepCoeff(cur, t, i, j);
                var dj = ds[ds.Count - 1 - j];
                for (int k = 0; k < sum.Length; k++) sum[k] += c * dj[k];
            }
            for (int k = 0; k < x.Length; k++) x[k] += sum[k];
            run.Step(i);
        }
        return x;
    }

    static async Task<float[]> Dpmpp2SAncestral(Run run, float[] x, double[] s, bool cfgPP)
    {
        double eta = run.Eta();
        for (int i = 0; i < s.Length - 1; i++)
        {
            run.Check();
            var denoised = await run.Denoise(x, s[i]);
            await run.Callback(denoised);
            if (run.Sampling.IsConst)
            {
                x = await FlowDpmpp2SAncestralStep(run, x, denoised, s[i], s[i + 1], eta);
                run.Step(i);
                continue;
            }
            var (down, up) = AncestralStep(s[i], s[i + 1], eta);
            var u = cfgPP ? run.Uncond(denoised) : denoised;
            if (down == 0)
            {
                var d = ToD(x, s[i], u);
                x = cfgPP ? Step(denoised, d, down) : Step(x, d, down - (float)s[i]);
            }
            else
            {
                float t = -Log(s[i]), tNext = -Log(down), h = tNext - t, sMid = t + 0.5f * h;
                float sigT = MathF.Exp(-t), sigS = MathF.Exp(-sMid), sigNext = MathF.Exp(-tNext);
                if (cfgPP)
                {
                    // CFG++: both updates start from x + (D - U); the final one pairs the first guided estimate with the latest unconditional
                    // one (measured).
                    var shifted = new float[x.Length];
                    for (int k = 0; k < x.Length; k++) shifted[k] = x[k] + (denoised[k] - u[k]);
                    var x2 = LinCombine(shifted, sigS / sigT, denoised, -ExpM1(-h * 0.5f));
                    var denoised2 = await run.Denoise(x2, sigS);
                    var u2 = run.Uncond(denoised2);
                    for (int k = 0; k < x.Length; k++) shifted[k] = x[k] + (denoised[k] - u2[k]);
                    x = LinCombine(shifted, sigNext / sigT, denoised2, -ExpM1(-h));
                }
                else
                {
                    var x2 = LinCombine(x, sigS / sigT, denoised, -ExpM1(-h * 0.5f));
                    var denoised2 = await run.Denoise(x2, sigS);
                    x = LinCombine(x, sigNext / sigT, denoised2, -ExpM1(-h));
                }
            }
            if (s[i + 1] > 0) AddNoise(x, run.Noise.Next(), run.SNoise * up);
            run.Step(i);
        }
        return x;
    }

    /// <summary>e^v - 1 without cancellation, rounded to float.</summary>
    internal static float ExpM1(float v) => (float)(Math.Abs(v) < 1e-5 ? v + v * (double)v / 2 + v * (double)v * v / 6 : Math.Exp(v) - 1);

    static async Task<float[]> DpmppSde(Run run, float[] x, double[] s)
    {
        double eta = run.Eta(), r = run.Options.R ?? 0.5;
        double min = s.Where(v => v > 0).Min(), max = s.Max();
        var brownian = new BrownianNoise(run.Noise, (float)min, (float)max);
        for (int i = 0; i < s.Length - 1; i++)
        {
            run.Check();
            var denoised = await run.Denoise(x, s[i]);
            await run.Callback(denoised);
            if (s[i + 1] == 0)
            {
                x = Step(x, ToD(x, s[i], denoised), (float)s[i + 1] - (float)s[i]);
                run.Step(i);
                continue;
            }
            if (run.Sampling.IsConst)
            {
                x = await FlowDpmppSdeStep(run, x, denoised, s[i], s[i + 1], eta, r, brownian);
                run.Step(i);
                continue;
            }
            float t = -Log(s[i]), tNext = -Log(s[i + 1]), h = tNext - t, sMid = t + h * (float)r;
            float fac = (float)(1 / (2 * r));
            float sigT = MathF.Exp(-t), sigS = MathF.Exp(-sMid), sigNext = MathF.Exp(-tNext);
            // Step 1: to the intermediate point, with its share of the noise.
            var (sd, su) = AncestralStep(sigT, sigS, eta);
            float s_ = -Log(sd);
            var x2 = LinCombine(x, MathF.Exp(-s_) / sigT, denoised, -ExpM1(t - s_));
            AddNoise(x2, brownian.Increment(sigT, sigS), run.SNoise * su);
            var denoised2 = await run.Denoise(x2, sigS);
            // Step 2: the whole step from a blend of the two estimates.
            (sd, su) = AncestralStep(sigT, sigNext, eta);
            float tNext_ = -Log(sd);
            var dd = LinCombine(denoised, 1 - fac, denoised2, fac);
            x = LinCombine(x, MathF.Exp(-tNext_) / sigT, dd, -ExpM1(t - tNext_));
            AddNoise(x, brownian.Increment(sigT, sigNext), run.SNoise * su);
            run.Step(i);
        }
        return x;
    }

    static async Task<float[]> Dpmpp2M(Run run, float[] x, double[] s, bool cfgPP)
    {
        float[]? old = null;
        for (int i = 0; i < s.Length - 1; i++)
        {
            run.Check();
            var denoised = await run.Denoise(x, s[i]);
            await run.Callback(denoised);
            float t = -Log(s[i]), tNext = -Log(s[i + 1]), h = tNext - t;
            float ratio = MathF.Exp(-tNext) / MathF.Exp(-t), k = -ExpM1(-h);
            if (cfgPP)
            {
                x = await Dpmpp2MCfgPPStep(run, x, denoised, old, s, i);
                old = run.Uncond(denoised);
                run.Step(i);
                continue;
            }
            if (old is null || s[i + 1] == 0) x = LinCombine(x, ratio, denoised, k);
            else
            {
                float hLast = t - -Log(s[i - 1]), r = hLast / h;
                var dd = LinCombine(denoised, 1 + 1 / (2 * r), old, -(1 / (2 * r)));
                x = LinCombine(x, ratio, dd, k);
            }
            old = denoised;
            run.Step(i);
        }
        return x;
    }

    static async Task<float[]> Dpmpp2MSde(Run run, float[] x, double[] s, string solverType)
    {
        if (solverType is not ("heun" or "midpoint")) throw new ArgumentException("solver_type must be 'heun' or 'midpoint'.");
        double eta = run.Eta();
        float[]? old = null;
        float hLast = 0;
        for (int i = 0; i < s.Length - 1; i++)
        {
            run.Check();
            var denoised = await run.Denoise(x, s[i]);
            await run.Callback(denoised);
            if (s[i + 1] == 0) x = denoised;
            else if (run.Sampling.IsConst)
            {
                (x, hLast) = await FlowDpmpp2MSdeStep(run, x, denoised, old, hLast, s[i], s[i + 1], eta, solverType);
            }
            else
            {
                float t = -Log(s[i]), sNext = -Log(s[i + 1]), h = sNext - t, etaH = (float)eta * h;
                x = LinCombine(x, (float)s[i + 1] / (float)s[i] * MathF.Exp(-etaH), denoised, -ExpM1(-h - etaH));
                if (old is not null)
                {
                    float r = hLast / h, phi = -ExpM1(-h - etaH);
                    float c = solverType == "heun" ? (phi / (-h - etaH) + 1) * (1 / r) : 0.5f * phi * (1 / r);
                    var diff = Subtract(denoised, old);
                    for (int k = 0; k < x.Length; k++) x[k] += c * diff[k];
                }
                if (eta != 0) AddNoise(x, run.Noise.Next(), (float)s[i + 1] * MathF.Sqrt(-ExpM1(-2 * etaH)) * (float)run.SNoise);
                hLast = h;
            }
            old = denoised;
            run.Step(i);
        }
        return x;
    }

    static async Task<float[]> Dpmpp3MSde(Run run, float[] x, double[] s)
    {
        double eta = run.Eta();
        float[]? d1Prev = null, d2Prev = null;
        float? h1 = null, h2 = null;
        for (int i = 0; i < s.Length - 1; i++)
        {
            run.Check();
            var denoised = await run.Denoise(x, s[i]);
            await run.Callback(denoised);
            if (s[i + 1] == 0) x = denoised;
            else
            {
                float t = -Log(s[i]), sNext = -Log(s[i + 1]), h = sNext - t, hEta = h * ((float)eta + 1);
                x = LinCombine(x, MathF.Exp(-hEta), denoised, -ExpM1(-hEta));
                if (h2 is not null)
                {
                    float r0 = h1!.Value / h, r1 = h2.Value / h;
                    var d10 = Scale(Subtract(denoised, d1Prev!), 1 / r0);
                    var d11 = Scale(Subtract(d1Prev!, d2Prev!), 1 / r1);
                    float phi2 = ExpM1(-hEta) / hEta + 1, phi3 = phi2 / hEta - 0.5f;
                    for (int k = 0; k < x.Length; k++)
                    {
                        float d1 = d10[k] + (d10[k] - d11[k]) * r0 / (r0 + r1);
                        float d2 = (d10[k] - d11[k]) / (r0 + r1);
                        x[k] = x[k] + phi2 * d1 - phi3 * d2;
                    }
                }
                else if (h1 is not null)
                {
                    float r = h1.Value / h, phi2 = ExpM1(-hEta) / hEta + 1;
                    var dd = Subtract(denoised, d1Prev!);
                    for (int k = 0; k < x.Length; k++) x[k] += phi2 * (dd[k] / r);
                }
                if (eta != 0) AddNoise(x, run.Noise.Next(), (float)s[i + 1] * MathF.Sqrt(-ExpM1(-2 * h * (float)eta)) * (float)run.SNoise);
                (h1, h2) = (h, h1);
            }
            (d1Prev, d2Prev) = (denoised, d1Prev);
            run.Step(i);
        }
        return x;
    }

    static float[] Scale(float[] a, float k)
    {
        var r = new float[a.Length];
        for (int i = 0; i < r.Length; i++) r[i] = a[i] * k;
        return r;
    }
}
