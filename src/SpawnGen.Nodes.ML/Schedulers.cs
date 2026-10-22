namespace SpawnGen.Nodes.ML;

/// <summary>
/// The noise levels of a discrete-timestep model (SD-style: 1000 training steps, sigma_t = sqrt((1 - alphaBar_t) / alphaBar_t)) and the schedulers that
/// pick which of them a run steps through. Same formulas as ComfyUI's <c>calculate_sigmas</c>.
/// </summary>
public sealed class ModelSigmas
{
    public static readonly string[] SchedulerNames = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta", "linear_quadratic"];

    readonly double[] _sigmas;     // ascending, one per training timestep
    readonly double[] _logSigmas;

    public ModelSigmas(IReadOnlyList<float> alphasCumprod)
    {
        _sigmas = alphasCumprod.Select((float a) => Math.Sqrt((1 - (double)a) / a)).ToArray();
        _logSigmas = _sigmas.Select(v => Math.Log(v)).ToArray();
    }

    public int TrainingSteps => _sigmas.Length;
    public float SigmaMin => (float)_sigmas[0];
    public float SigmaMax => (float)_sigmas[^1];

    /// <summary>The sigma at a (possibly fractional) timestep: linear in log-sigma between the neighbouring training steps.</summary>
    public float Sigma(double timestep)
    {
        double t = Math.Clamp(timestep, 0, _sigmas.Length - 1);
        int low = (int)Math.Floor(t), high = Math.Min(low + 1, _sigmas.Length - 1);
        double frac = t - low;
        return (float)Math.Exp(_logSigmas[low] * (1 - frac) + _logSigmas[high] * frac);
    }

    /// <summary>The (fractional) timestep whose sigma is <paramref name="sigma"/>: the inverse of <see cref="Sigma"/>. What the model is told as "t".</summary>
    public float Timestep(float sigma)
    {
        double log = Math.Log(Math.Max(sigma, 1e-10));
        if (log <= _logSigmas[0]) return 0;
        if (log >= _logSigmas[^1]) return _sigmas.Length - 1;
        int lo = 0, hi = _logSigmas.Length - 1;
        while (hi - lo > 1) { int mid = (lo + hi) / 2; if (_logSigmas[mid] <= log) lo = mid; else hi = mid; }
        double w = (log - _logSigmas[lo]) / (_logSigmas[lo + 1] - _logSigmas[lo]);
        return (float)(lo + w);
    }

    /// <summary>The sigmas of a run: <paramref name="steps"/> noise levels from high to low, then 0 (length steps + 1).</summary>
    public float[] Schedule(string scheduler, int steps)
    {
        if (steps < 1) throw new ArgumentOutOfRangeException(nameof(steps));
        if (scheduler == "beta") return Beta(steps, 0.6, 0.6);
        if (scheduler == "linear_quadratic") return LinearQuadratic(steps);
        double[] values = scheduler switch
        {
            "normal" => Linspace(steps).Select(t => (double)Sigma(t)).ToArray(),
            "sgm_uniform" => LinspaceExcludingEnd(steps).Select(t => (double)Sigma(t)).ToArray(),
            "simple" => Simple(steps),
            "ddim_uniform" => DdimUniform(steps),
            "karras" => Karras(steps),
            "exponential" => Exponential(steps),
            _ => throw new ArgumentException($"Unknown scheduler '{scheduler}'. Known: {string.Join(", ", SchedulerNames)}.", nameof(scheduler)),
        };
        return [.. values.Select(v => (float)v), 0f];
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
            double t = Math.Round(BetaMath.Ppf(q, alpha, beta) * total, MidpointRounding.ToEven);
            if (t != last) result.Add((float)_sigmas[(int)t]);
            last = t;
        }
        result.Add(0f);
        return [.. result];
    }

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
        return [.. schedule.Select(x => (float)((1.0 - x) * SigmaMax))];
    }

    /// <summary>
    /// The sigmas for a run that starts part-way down: with <paramref name="denoise"/> below 1 the full schedule has <c>steps / denoise</c> steps and only its
    /// last <paramref name="steps"/> are run (ComfyUI's rule), so sampling starts from a noised input instead of pure noise.
    /// </summary>
    public float[] ScheduleForDenoise(string scheduler, int steps, float denoise)
    {
        if (denoise is <= 0f or > 1f) throw new ArgumentOutOfRangeException(nameof(denoise), "denoise must be in (0, 1].");
        if (denoise >= 1f) return Schedule(scheduler, steps);
        int total = (int)(steps / denoise);
        var full = Schedule(scheduler, total);
        return full[Math.Max(0, full.Length - (steps + 1))..]; // a scheduler that merges steps (beta) can return fewer than asked: take what there is, like Python slicing
    }

    // The training timesteps from the noisiest (TrainingSteps - 1) to the cleanest (0).
    IEnumerable<double> Linspace(int n)
    {
        double start = _sigmas.Length - 1;
        for (int i = 0; i < n; i++) yield return n == 1 ? start : start + (0 - start) * i / (n - 1);
    }

    IEnumerable<double> LinspaceExcludingEnd(int n)
    {
        double start = _sigmas.Length - 1;
        for (int i = 0; i < n; i++) yield return start + (0 - start) * i / n; // linspace(start, 0, n + 1) without its last point
    }

    double[] Simple(int steps)
    {
        // Every (count / steps)-th training sigma, from the top.
        double stride = (double)_sigmas.Length / steps;
        return Enumerable.Range(0, steps).Select(i => _sigmas[^(1 + (int)(i * stride))]).ToArray();
    }

    double[] DdimUniform(int steps)
    {
        int stride = Math.Max(_sigmas.Length / steps, 1);
        var picked = new List<double>();
        for (int x = 1; x < _sigmas.Length; x += stride) picked.Add(_sigmas[x]);
        picked.Reverse();
        return [.. picked];
    }

    double[] Karras(int steps)
    {
        const double rho = 7;
        double max = _sigmas[^1], min = _sigmas[0];
        double minInv = Math.Pow(min, 1 / rho), maxInv = Math.Pow(max, 1 / rho);
        return Enumerable.Range(0, steps).Select(i => Math.Pow(maxInv + (steps == 1 ? 0 : (double)i / (steps - 1)) * (minInv - maxInv), rho)).ToArray();
    }

    double[] Exponential(int steps)
    {
        double max = Math.Log(_sigmas[^1]), min = Math.Log(_sigmas[0]);
        return Enumerable.Range(0, steps).Select(i => Math.Exp(max + (steps == 1 ? 0 : (double)i / (steps - 1)) * (min - max))).ToArray();
    }
}
