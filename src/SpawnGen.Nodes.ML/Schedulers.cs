namespace SpawnGen.Nodes.ML;

/// <summary>
/// The noise levels of a discrete-timestep model (SD-style: 1000 training steps, sigma_t = sqrt((1 - alphaBar_t) / alphaBar_t)) and the schedulers that
/// pick which of them a run steps through. Same formulas as ComfyUI's <c>ModelSamplingDiscrete</c> and <c>calculate_sigmas</c>.
/// </summary>
public sealed class ModelSigmas
{
    /// <summary>ComfyUI's SCHEDULER_NAMES, in its order (a saved workflow stores the name, the combo shows this order).</summary>
    public static readonly string[] SchedulerNames = ["simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal", "linear_quadratic", "kl_optimal"];

    readonly double[] _sigmas;     // ascending, one per (distilled) training timestep
    readonly double[] _logSigmas;
    readonly int _skip;            // timesteps per entry: 1, or 20 for an LCM-distilled table of 50
    readonly Continuity _continuity;

    /// <summary>Discrete training timesteps (SD), or a continuous sigma (ComfyUI's ModelSamplingContinuousEDM, ModelSamplingContinuousV).</summary>
    public enum Continuity { Discrete, ContinuousEdm, ContinuousV }

    /// <summary>Discrete, or how a continuous model is told its noise level (0.25 ln sigma for EDM, atan(sigma) 2 / pi for continuous V).</summary>
    public Continuity Kind => _continuity;

    /// <summary>
    /// A continuous-sigma model's table as ComfyUI builds it (set_parameters): 1000 sigmas from <paramref name="sigmaMin"/> to <paramref name="sigmaMax"/>, evenly
    /// spaced in log-sigma (torch.linspace of the logs, then exp, in float32); the table is used by the schedulers that pick from it.
    /// </summary>
    public static ModelSigmas Continuous(double sigmaMin, double sigmaMax, bool v)
    {
        const int n = 1000;
        float a = (float)Math.Log(sigmaMin), b = (float)Math.Log(sigmaMax);
        double step = ((double)b - a) / (n - 1);
        var sigmas = new double[n];
        for (int i = 0; i < n; i++)
        {
            double log = i < n / 2 ? a + step * i : b - step * (n - 1 - i); // torch.linspace's two halves
            sigmas[i] = MathF.Exp((float)log);
        }
        return new ModelSigmas(sigmas, 1, v ? Continuity.ContinuousV : Continuity.ContinuousEdm);
    }

    public ModelSigmas(IReadOnlyList<float> alphasCumprod)
        : this(alphasCumprod.Select((float a) => Math.Sqrt((1 - (double)a) / a)).ToArray(), 1) { }

    /// <summary>
    /// SD's own table as ComfyUI builds it (make_beta_schedule "linear": betas = linspace(sqrt(start), sqrt(end), n)^2 in float64, then the cumulative product),
    /// the same for SD 1.x, 2.x and SD-Turbo.
    /// </summary>
    public static ModelSigmas LinearSchedule(double linearStart = 0.00085, double linearEnd = 0.012, int timesteps = 1000)
    {
        double a = Math.Sqrt(linearStart), b = Math.Sqrt(linearEnd), step = (b - a) / (timesteps - 1), cum = 1;
        var sigmas = new double[timesteps];
        for (int i = 0; i < timesteps; i++)
        {
            double root = i < timesteps / 2 ? a + step * i : b - step * (timesteps - 1 - i); // torch.linspace's two halves
            cum *= 1 - root * root;
            sigmas[i] = Math.Sqrt((1 - cum) / cum);
        }
        return new ModelSigmas(sigmas, 1);
    }

    ModelSigmas(double[] sigmas, int skip, Continuity continuity = Continuity.Discrete)
    {
        // ComfyUI keeps the table in float32.
        _sigmas = sigmas.Select(v => (double)(float)v).ToArray();
        _logSigmas = _sigmas.Select(v => (double)(float)Math.Log(v)).ToArray();
        _skip = skip;
        _continuity = continuity;
    }

    /// <summary>The table's length (1000, or 50 for an LCM-distilled one).</summary>
    public int TrainingSteps => _sigmas.Length;
    public float SigmaMin => (float)_sigmas[0];
    public float SigmaMax => (float)_sigmas[^1];

    /// <summary>The table entry <paramref name="index"/> (ascending): what the simple and beta schedulers pick from.</summary>
    public double TableSigma(int index) => _sigmas[index];

    /// <summary>
    /// The same schedule with zero terminal SNR (ComfyUI's <c>rescale_zero_terminal_snr_sigmas</c>, the zsnr switch): sqrt(alphaBar) is shifted so the last timestep
    /// is pure noise and rescaled so the first is unchanged; the last alphaBar is set to 4.8973451890853435e-08 to keep sigma finite.
    /// </summary>
    public ModelSigmas ZeroTerminalSnr()
    {
        if (_continuity != Continuity.Discrete) throw new NotSupportedException("Zero terminal SNR applies to discrete-timestep models.");
        int n = _sigmas.Length;
        var sqrtBar = _sigmas.Select(s => Math.Sqrt(1 / (s * s + 1))).ToArray();
        double first = sqrtBar[0], last = sqrtBar[^1];
        for (int i = 0; i < n; i++) sqrtBar[i] = (sqrtBar[i] - last) * (first / (first - last));
        var bar = sqrtBar.Select(v => v * v).ToArray();
        bar[^1] = 4.8973451890853435e-08;
        return new ModelSigmas(bar.Select(a => Math.Sqrt((1 - a) / a)).ToArray(), _skip);
    }

    /// <summary>
    /// The table of a model distilled to <paramref name="originalTimesteps"/> steps (ComfyUI's <c>ModelSamplingDiscreteDistilled</c>, used by LCM): every
    /// (count / originalTimesteps)-th sigma from the top, and timesteps told to the model as index * skip + skip - 1.
    /// </summary>
    public ModelSigmas Distilled(int originalTimesteps = 50)
    {
        if (_continuity != Continuity.Discrete) throw new NotSupportedException("A distilled timestep table applies to discrete-timestep models.");
        int n = _sigmas.Length, skip = n / originalTimesteps;
        var valid = new double[originalTimesteps];
        for (int x = 0; x < originalTimesteps; x++) valid[originalTimesteps - 1 - x] = _sigmas[n - 1 - x * skip];
        return new ModelSigmas(valid, skip);
    }

    /// <summary>The sigma at a (possibly fractional) timestep: linear in log-sigma between the neighbouring table entries.</summary>
    public float Sigma(double timestep)
    {
        // Continuous models: exp(t / 0.25) (EDM), tan(t pi / 2) (V), in float32.
        if (_continuity == Continuity.ContinuousEdm) return MathF.Exp((float)timestep / 0.25f);
        if (_continuity == Continuity.ContinuousV) return MathF.Tan((float)timestep * MathF.PI / 2f);
        double t = Math.Clamp((timestep - (_skip - 1)) / _skip, 0, _sigmas.Length - 1);
        int low = (int)Math.Floor(t), high = (int)Math.Ceiling(t);
        double frac = t - low;
        return (float)Math.Exp((1 - frac) * _logSigmas[low] + frac * _logSigmas[high]);
    }

    /// <summary>
    /// The timestep the model is told for <paramref name="sigma"/>: the table entry nearest in log-sigma (the first on a tie), as ComfyUI's <c>timestep</c> does,
    /// so the model sees whole training timesteps even when a schedule falls between them.
    /// </summary>
    public float Timestep(double sigma)
    {
        // Continuous models: 0.25 ln(sigma) (EDM), atan(sigma) / pi * 2 (V), in float32.
        if (_continuity == Continuity.ContinuousEdm) return 0.25f * MathF.Log((float)sigma);
        if (_continuity == Continuity.ContinuousV) return MathF.Atan((float)sigma) / MathF.PI * 2f;
        double log = Math.Log(sigma);
        int best = 0;
        double bestDist = double.PositiveInfinity;
        for (int i = 0; i < _logSigmas.Length; i++)
        {
            double d = Math.Abs(log - _logSigmas[i]);
            if (d < bestDist) { bestDist = d; best = i; }
        }
        return best * _skip + (_skip - 1);
    }

    /// <summary>ComfyUI's <c>percent_to_sigma</c>: 0 (the start) is a huge sigma, 1 (the end) is 0, in between the sigma at timestep (1 - percent) * 999.</summary>
    public double PercentToSigma(double percent)
    {
        if (percent <= 0.0) return 999999999.9;
        if (percent >= 1.0) return 0.0;
        if (_continuity != Continuity.Discrete)
        {
            // Linear in log-sigma between the table's ends, in float64 as ComfyUI's math.log / math.exp.
            double logMin = Math.Log(SigmaMin);
            return Math.Exp((Math.Log(SigmaMax) - logMin) * (1.0 - percent) + logMin);
        }
        return Sigma((float)((1.0 - percent) * 999.0));
    }

    /// <summary>The sigmas of a run: <paramref name="steps"/> noise levels from high to low, then 0 (length steps + 1, fewer for beta when steps merge).</summary>
    public float[] Schedule(string scheduler, int steps)
    {
        if (steps < 1) throw new ArgumentOutOfRangeException(nameof(steps));
        return scheduler switch
        {
            "simple" => Simple(steps),
            "sgm_uniform" => Normal(steps, sgm: true),
            "karras" => SigmaSchedules.Karras(steps, SigmaMax, SigmaMin, 7.0),
            "exponential" => SigmaSchedules.Exponential(steps, SigmaMax, SigmaMin),
            "ddim_uniform" => DdimUniform(steps),
            "beta" => Beta(steps, 0.6, 0.6),
            "normal" => Normal(steps, sgm: false),
            "linear_quadratic" => LinearQuadratic(steps),
            "kl_optimal" => KlOptimal(steps, SigmaMin, SigmaMax),
            _ => throw new ArgumentException($"Unknown scheduler '{scheduler}'. Known: {string.Join(", ", SchedulerNames)}.", nameof(scheduler)),
        };
    }

    /// <summary>
    /// ComfyUI's beta scheduler (Lee et al., "Beta Sampling is All You Need"): the training timesteps at the quantiles of a Beta(alpha, beta) distribution, from the noisiest down,
    /// each timestep used once (steps that land on the same one are merged, so there can be fewer than <paramref name="steps"/> + 1 values).
    /// </summary>
    public float[] Beta(int steps, double alpha, double beta)
    {
        if (steps < 1) throw new ArgumentOutOfRangeException(nameof(steps));
        int total = _sigmas.Length - 1;
        var result = new List<float>();
        double last = -1;
        for (int i = 0; i < steps; i++)
        {
            double q = 1.0 - (double)i / steps;            // 1 - linspace(0, 1, steps, endpoint=False)
            double t = Math.Round(BetaPpf(q, alpha, beta) * total, MidpointRounding.ToEven);
            if (t != last) result.Add((float)_sigmas[(int)t]);
            last = t;
        }
        result.Add(0f);
        return [.. result];
    }

    /// <summary>
    /// The Beta quantile as scipy returns it. At the median of a symmetric Beta(a, a) the exact answer is 0.5, where a timestep can land exactly half-way between two
    /// and the rounding decides; scipy (boost's ibeta_inv) is an ulp off there in a direction that depends on a. For the scheduler's default 0.6 it is one ulp above,
    /// which is reproduced so the default beta schedule matches on every table; other symmetric shapes use 0.5.
    /// </summary>
    static double BetaPpf(double q, double alpha, double beta) =>
        q == 0.5 && alpha == beta ? (alpha == 0.6 ? Math.BitIncrement(0.5) : 0.5) : BetaMath.Ppf(q, alpha, beta);

    /// <summary>
    /// ComfyUI's linear-quadratic schedule (from Mochi): the first half of the steps falls linearly, the rest along a quadratic that meets it, scaled to the model's largest sigma.
    /// </summary>
    public float[] LinearQuadratic(int steps, double thresholdNoise = 0.025, int? linearSteps = null)
    {
        if (steps < 1) throw new ArgumentOutOfRangeException(nameof(steps));
        if (steps == 1) return [SigmaMax, 0f];
        int linear = linearSteps ?? steps / 2;
        var schedule = new List<double>();
        for (int i = 0; i < linear; i++) schedule.Add(i * thresholdNoise / linear);
        double diff = linear - thresholdNoise * steps;
        int quadraticSteps = steps - linear;
        if (quadraticSteps > 0)
        {
            double quadraticCoef = diff / (linear * (double)quadraticSteps * quadraticSteps);
            double linearCoef = thresholdNoise / linear - 2 * diff / ((double)quadraticSteps * quadraticSteps);
            double constant = quadraticCoef * linear * linear;
            for (int i = linear; i < steps; i++) schedule.Add(quadraticCoef * i * i + linearCoef * i + constant);
        }
        schedule.Add(1.0);
        return [.. schedule.Select(x => (float)((float)(1.0 - x) * SigmaMax))];
    }

    /// <summary>
    /// ComfyUI's kl_optimal schedule (AUTOMATIC1111 PR 15608): sigmas evenly spaced in arctan(sigma) from sigma_max to sigma_min, then 0. One step divides by zero there
    /// and gives NaN, as in ComfyUI.
    /// </summary>
    public static float[] KlOptimal(int steps, double sigmaMin, double sigmaMax)
    {
        var r = new float[steps + 1];
        for (int i = 0; i < steps; i++)
        {
            float adj = (float)i / (steps - 1);
            r[i] = MathF.Tan((float)(adj * Math.Atan(sigmaMin) + (1 - adj) * Math.Atan(sigmaMax)));
        }
        return r;
    }

    /// <summary>
    /// The sigmas for a run that starts part-way down: with <paramref name="denoise"/> below 1 the full schedule has <c>int(steps / denoise)</c> steps and only its
    /// last <paramref name="steps"/> are run (ComfyUI's rule), so sampling starts from a noised input instead of pure noise.
    /// </summary>
    public float[] ScheduleForDenoise(string scheduler, int steps, double denoise) => ScheduleForDenoise(steps, denoise, n => Schedule(scheduler, n));

    /// <summary>ScheduleForDenoise over any schedule function (KSampler's, which may discard the penultimate sigma).</summary>
    public static float[] ScheduleForDenoise(int steps, double denoise, Func<int, float[]> schedule)
    {
        if (denoise is <= 0 or > 1) throw new ArgumentOutOfRangeException(nameof(denoise), "denoise must be in (0, 1].");
        if (denoise >= 1) return schedule(steps);
        int total = (int)(steps / denoise);
        var full = schedule(total);
        return full[Math.Max(0, full.Length - (steps + 1))..]; // a scheduler that merges steps (beta) can return fewer than asked: take what there is, like Python slicing
    }

    float[] Normal(int steps, bool sgm)
    {
        double start = Timestep(SigmaMax), end = Timestep(SigmaMin);
        bool appendZero = true;
        double[] timesteps;
        if (sgm)
            timesteps = Linspace(start, end, steps + 1)[..^1];
        else
        {
            if (Math.Abs(Sigma(end)) <= 0.00001) { steps += 1; appendZero = false; }
            timesteps = Linspace(start, end, steps);
        }
        var sigs = timesteps.Select(t => Sigma(t)).ToList();
        if (appendZero) sigs.Add(0f);
        return [.. sigs];
    }

    /// <summary>torch.linspace in float32: start + i * step for the first half, end - (n - 1 - i) * step for the second.</summary>
    internal static double[] Linspace(double start, double end, int n)
    {
        var r = new double[n];
        if (n == 1) { r[0] = (float)start; return r; }
        float step = (float)((end - start) / (n - 1));
        int half = n / 2;
        for (int i = 0; i < n; i++) r[i] = i < half ? (float)((float)start + step * i) : (float)((float)end - step * (n - 1 - i));
        return r;
    }

    float[] Simple(int steps)
    {
        // Every (count / steps)-th table sigma, from the top.
        double stride = (double)_sigmas.Length / steps;
        var r = new float[steps + 1];
        for (int i = 0; i < steps; i++) r[i] = (float)_sigmas[^(1 + (int)(i * stride))];
        return r;
    }

    float[] DdimUniform(int steps)
    {
        var sigs = new List<float>();
        if (Math.Abs(_sigmas[1]) <= 0.00001) steps += 1;
        else sigs.Add(0f);
        int stride = Math.Max(_sigmas.Length / steps, 1);
        for (int x = 1; x < _sigmas.Length; x += stride) sigs.Add((float)_sigmas[x]);
        sigs.Reverse();
        return [.. sigs];
    }
}
