namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md). Sources: the scaled-linear beta schedule of Stable Diffusion (as published with the model and in diffusers,
// Apache-2.0), Karras et al. 2022 "Elucidating the Design Space of Diffusion-Based Generative Models" and k-diffusion (MIT) for the karras and
// exponential schedules and the continuous EDM sigmas, Lin et al. 2023 "Common Diffusion Noise Schedules and Sample Steps are Flawed" for zero
// terminal SNR, the LCM paper for the distilled table, Movie Gen / genmo's mochi (Apache-2.0) for linear_quadratic. ComfyUI's own rules (nearest
// timestep, simple, ddim_uniform, sgm_uniform, beta, kl_optimal and where float32 is used) were fitted to Fixtures/sampling/reference.json and
// Fixtures/model_sampling/continuous.json, outputs of ComfyUI.

/// <summary>
/// A model's noise levels: the sigma for each training timestep (discrete, 1000 or a distilled subset) or a continuous range, the mapping between
/// sigma and timestep, and the schedulers that pick a run's sigmas from them.
/// </summary>
public sealed class ModelSigmas
{
    public enum Continuity { Discrete, ContinuousEdm, ContinuousV }

    /// <summary>The scheduler names, in ComfyUI's order.</summary>
    public static string[] SchedulerNames { get; } = ["simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal", "linear_quadratic", "kl_optimal"];

    readonly float[] _sigmas;     // rising: index 0 is the least noise
    readonly float[] _logSigmas;
    readonly double[]? _alphasCumprod;
    readonly int _timestepStride; // distilled tables: table index i is training timestep (i + 1) * stride - 1
    readonly double _continuousMin, _continuousMax;

    public Continuity Kind { get; }

    ModelSigmas(float[] sigmas, double[]? alphasCumprod, Continuity kind, int stride = 1, double min = 0, double max = 0)
    {
        _sigmas = sigmas;
        _logSigmas = [.. sigmas.Select(s => MathF.Log(s))];
        _alphasCumprod = alphasCumprod;
        Kind = kind;
        _timestepStride = stride;
        _continuousMin = min;
        _continuousMax = max;
    }

    /// <summary>A discrete table from the cumulative products of alpha, one per training timestep: sigma = sqrt((1 - a) / a).</summary>
    public ModelSigmas(float[] alphasCumprod) : this(Sigmas([.. alphasCumprod.Select(a => (double)a)]), [.. alphasCumprod.Select(a => (double)a)], Continuity.Discrete) { }

    static float[] Sigmas(double[] alphasCumprod) => [.. alphasCumprod.Select(a => (float)Math.Sqrt((1 - a) / a))];

    /// <summary>
    /// Stable Diffusion's "scaled linear" schedule: betas evenly spaced in square-root space from <paramref name="start"/> to <paramref name="end"/>,
    /// squared, and alpha's cumulative product taken in double precision.
    /// </summary>
    public static ModelSigmas LinearSchedule(double start = 0.00085, double end = 0.012, int timesteps = 1000)
    {
        var cumprod = new double[timesteps];
        double cum = 1, a = Math.Sqrt(start), b = Math.Sqrt(end);
        for (int t = 0; t < timesteps; t++)
        {
            double sq = timesteps == 1 ? a : a + (b - a) * t / (timesteps - 1);
            cum *= 1 - sq * sq;
            cumprod[t] = cum;
        }
        return new ModelSigmas(Sigmas(cumprod), cumprod, Continuity.Discrete);
    }

    /// <summary>
    /// A continuous model's 1000-entry table, sigmas evenly spaced in log space from <paramref name="min"/> to <paramref name="max"/>. EDM models take
    /// timestep = log(sigma) / 4 (Karras et al.); <paramref name="v"/> models take timestep = atan(sigma) * 2 / pi.
    /// </summary>
    public static ModelSigmas Continuous(double min, double max, bool v)
    {
        var table = new float[1000];
        double lmin = Math.Log(min), lmax = Math.Log(max);
        for (int i = 0; i < table.Length; i++) table[i] = (float)Math.Exp(lmin + (lmax - lmin) * i / (table.Length - 1));
        return new ModelSigmas(table, null, v ? Continuity.ContinuousV : Continuity.ContinuousEdm, 1, min, max);
    }

    /// <summary>The LCM table: every 20th training timestep (19, 39, ..., 999), 50 entries.</summary>
    public ModelSigmas Distilled(int originalSteps = 50)
    {
        if (Kind != Continuity.Discrete) throw new NotSupportedException("Only a discrete schedule can be distilled.");
        int stride = _sigmas.Length / originalSteps;
        return new ModelSigmas([.. Enumerable.Range(1, originalSteps).Select(i => _sigmas[i * stride - 1])], null, Continuity.Discrete, stride);
    }

    /// <summary>
    /// The same schedule rescaled to zero terminal SNR (Lin et al. 2023, algorithm 1, on sqrt of alpha's cumulative product), with the last alpha
    /// kept just above zero so its sigma stays finite (measured: sigma_max 4518.76).
    /// </summary>
    public ModelSigmas ZeroTerminalSnr()
    {
        if (_alphasCumprod is null || Kind != Continuity.Discrete || _timestepStride != 1) throw new NotSupportedException("Zero terminal SNR needs a model's full discrete schedule.");
        var a = _alphasCumprod.Select(Math.Sqrt).ToArray();
        double a0 = a[0], aT = a[^1];
        var cumprod = a.Select(v => Math.Pow((v - aT) * a0 / (a0 - aT), 2)).ToArray();
        cumprod[^1] = 4.8973451890853435e-08;
        return new ModelSigmas(Sigmas(cumprod), cumprod, Continuity.Discrete);
    }

    /// <summary>The number of training timesteps in the table (1000, or 50 for a distilled one).</summary>
    public int TrainingSteps => _sigmas.Length;
    public float SigmaMin => _sigmas[0];
    public float SigmaMax => _sigmas[^1];

    /// <summary>The table's sigma at index <paramref name="index"/>.</summary>
    public float TableSigma(int index) => _sigmas[index];

    /// <summary>
    /// The timestep the model is told for a sigma: for a discrete table the training timestep whose log sigma is nearest (measured: no interpolation),
    /// for continuous models their formula.
    /// </summary>
    public float Timestep(double sigma)
    {
        switch (Kind)
        {
            case Continuity.ContinuousEdm: return 0.25f * MathF.Log((float)sigma);
            case Continuity.ContinuousV: return MathF.Atan((float)sigma) / MathF.PI * 2f;
        }
        float log = MathF.Log((float)sigma);
        int best = 0;
        float bestDistance = float.PositiveInfinity;
        for (int i = 0; i < _logSigmas.Length; i++)
        {
            float d = MathF.Abs(log - _logSigmas[i]);
            if (d < bestDistance) { bestDistance = d; best = i; }
        }
        return (best + 1) * _timestepStride - 1f;
    }

    /// <summary>The sigma at a (fractional) timestep: log-linear between the table's neighbours, or the continuous formula.</summary>
    public float Sigma(double timestep)
    {
        switch (Kind)
        {
            case Continuity.ContinuousEdm: return MathF.Exp((float)timestep / 0.25f);
            case Continuity.ContinuousV: return MathF.Tan((float)timestep * MathF.PI / 2f);
        }
        double t = Math.Clamp((timestep + 1) / _timestepStride - 1, 0, _sigmas.Length - 1);
        int low = (int)Math.Floor(t), high = (int)Math.Ceiling(t);
        float w = (float)(t - low);
        return MathF.Exp((1 - w) * _logSigmas[low] + w * _logSigmas[high]);
    }

    /// <summary>
    /// The sigma at a fraction of the run (0 = start, 1 = end): above every sigma at 0, zero at 1, otherwise the timestep (1 - percent) of the way
    /// (continuous models: log-linear between their sigma range).
    /// </summary>
    public double PercentToSigma(double percent)
    {
        if (percent <= 0) return 999999999.9;
        if (percent >= 1) return 0;
        if (Kind != Continuity.Discrete)
            return Math.Exp(Math.Log(_continuousMin) + (Math.Log(_continuousMax) - Math.Log(_continuousMin)) * (1 - percent));
        return Sigma((1 - percent) * (_sigmas.Length * _timestepStride - 1));
    }

    /// <summary>numpy's linspace: <paramref name="n"/> values from <paramref name="start"/> to <paramref name="stop"/>, both included.</summary>
    public static double[] Linspace(double start, double stop, int n)
    {
        if (n == 1) return [start];
        var v = new double[Math.Max(n, 0)];
        double step = (stop - start) / (n - 1);
        for (int i = 0; i < v.Length; i++) v[i] = start + step * i;
        if (n > 1) v[^1] = stop;
        return v;
    }

    /// <summary>The run's sigmas for a scheduler, from the most noise down, ending in 0.</summary>
    public float[] Schedule(string scheduler, int steps)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(steps, 1);
        return scheduler switch
        {
            "simple" => Simple(steps),
            "sgm_uniform" => Uniform(steps, sgm: true),
            "karras" => Karras(steps, SigmaMin, SigmaMax),
            "exponential" => Exponential(steps, SigmaMin, SigmaMax),
            "ddim_uniform" => DdimUniform(steps),
            "beta" => Beta(steps, 0.6, 0.6),
            "normal" => Uniform(steps, sgm: false),
            "linear_quadratic" => LinearQuadratic(steps),
            "kl_optimal" => KlOptimal(steps, SigmaMin, SigmaMax),
            _ => throw new ArgumentException($"Unknown scheduler '{scheduler}'. The schedulers are: {string.Join(", ", SchedulerNames)}.", nameof(scheduler)),
        };
    }

    /// <summary>
    /// The last <paramref name="steps"/> + 1 sigmas of the schedule for int(steps / denoise) steps: a partial denoise starts part-way down.
    /// </summary>
    public float[] ScheduleForDenoise(string scheduler, int steps, double denoise)
    {
        if (denoise <= 0 || denoise > 1) throw new ArgumentOutOfRangeException(nameof(denoise), denoise, "Denoise must be above 0 and at most 1.");
        ArgumentOutOfRangeException.ThrowIfLessThan(steps, 1);
        var full = Schedule(scheduler, (int)(steps / denoise));
        return full[Math.Max(0, full.Length - (steps + 1))..];
    }

    static float[] WithZero(IEnumerable<float> sigmas) => [.. sigmas, 0f];

    /// <summary>Every (count / steps)-th table sigma from the top.</summary>
    float[] Simple(int steps)
    {
        double stride = (double)_sigmas.Length / steps;
        return WithZero(Enumerable.Range(0, steps).Select(x => _sigmas[_sigmas.Length - 1 - (int)(x * stride)]));
    }

    /// <summary>
    /// Timesteps evenly spaced from sigma_max's to sigma_min's ("normal"), or that range cut into steps + 1 points with the last dropped
    /// ("sgm_uniform"), each turned back into a sigma.
    /// </summary>
    float[] Uniform(int steps, bool sgm)
    {
        double start = Timestep(SigmaMax), end = Timestep(SigmaMin);
        var t = sgm ? Linspace(start, end, steps + 1)[..^1] : Linspace(start, end, steps);
        return WithZero(t.Select(v => Sigma(v)));
    }

    /// <summary>Table sigmas at every (count / steps)-th index from 1 upwards, from the top down (measured: one step starts from the lowest).</summary>
    float[] DdimUniform(int steps)
    {
        int stride = Math.Max(_sigmas.Length / steps, 1);
        var picked = new List<float>();
        for (int x = 1; x < _sigmas.Length; x += stride) picked.Add(_sigmas[x]);
        picked.Reverse();
        return WithZero(picked);
    }

    /// <summary>Karras et al. 2022: sigma^(1/rho) evenly spaced from sigma_max to sigma_min (float32, as measured).</summary>
    public static float[] Karras(int steps, double sigmaMin, double sigmaMax, double rho = 7.0)
    {
        float minInv = (float)Math.Pow(sigmaMin, 1 / rho), maxInv = (float)Math.Pow(sigmaMax, 1 / rho), r = (float)rho;
        var ramp = Linspace(0, 1, steps);
        return WithZero(ramp.Select(x => MathF.Pow(maxInv + (float)x * (minInv - maxInv), r)));
    }

    /// <summary>Sigmas evenly spaced in log space from sigma_max to sigma_min (float32).</summary>
    public static float[] Exponential(int steps, double sigmaMin, double sigmaMax) =>
        WithZero(Linspace(Math.Log(sigmaMax), Math.Log(sigmaMin), steps).Select(v => MathF.Exp((float)v)));

    /// <summary>
    /// The beta schedule: timesteps at the Beta(alpha, beta) quantiles of 1, 1 - 1/steps, ..., rounded to whole timesteps, a timestep equal to the one
    /// before skipped (so there can be fewer steps), each taken from the table.
    /// </summary>
    public float[] Beta(int steps, double alpha, double beta)
    {
        int total = _sigmas.Length - 1;
        var picked = new List<float>();
        double? last = null;
        for (int i = 0; i < steps; i++)
        {
            double q = 1 - (double)i / steps;
            double t = Math.Round(Quantile(q, alpha, beta) * total, MidpointRounding.ToEven);
            if (t != last) picked.Add(_sigmas[(int)t]);
            last = t;
        }
        return WithZero(picked);
    }

    /// <summary>
    /// The Beta quantile as ComfyUI gets it from scipy. At the median of a symmetric shape, timestep = 0.5 * an odd count is a rounding tie that scipy's
    /// last-bit error decides (measured with scipy: 0.1, 0.6 and 1.5 come out just above 0.5, 0.5 and 2.0 just below, others exact); other shapes
    /// use the exact median, so their median step can differ from ComfyUI's by one timestep.
    /// </summary>
    static double Quantile(double q, double alpha, double beta)
    {
        if (alpha == beta && q == 0.5)
            return alpha is 0.1 or 0.6 or 1.5 ? Math.BitIncrement(0.5) : alpha is 0.5 or 2.0 ? Math.BitDecrement(0.5) : 0.5;
        return BetaMath.Ppf(q, alpha, beta);
    }

    /// <summary>
    /// Movie Gen's linear-quadratic schedule (as in genmo's mochi, Apache-2.0): the noise fraction falls linearly by <paramref name="threshold"/> over
    /// the first half of the steps, then quadratically to 1; sigma = (1 - fraction) * sigma_max.
    /// </summary>
    public float[] LinearQuadratic(int steps, double threshold = 0.025, int? linearSteps = null)
    {
        if (steps == 1) return [SigmaMax, 0f];
        int ls = linearSteps ?? steps / 2;
        var schedule = Enumerable.Range(0, ls).Select(i => i * threshold / ls).ToList();
        double diff = ls - threshold * steps;
        int qs = steps - ls;
        double qc = diff / (ls * (double)qs * qs), lc = threshold / ls - 2 * diff / ((double)qs * qs), c = qc * ls * ls;
        for (int i = ls; i < steps; i++) schedule.Add(qc * i * i + lc * i + c);
        schedule.Add(1.0);
        return [.. schedule.Select(f => (float)((1 - f) * SigmaMax))];
    }

    /// <summary>
    /// The KL-optimal schedule: tan of the angle moved evenly from atan(sigma_max) to atan(sigma_min), in float32 (measured; one step divides by zero,
    /// giving NaN, as ComfyUI does).
    /// </summary>
    public static float[] KlOptimal(int steps, double sigmaMin, double sigmaMax)
    {
        float aMin = (float)Math.Atan(sigmaMin), aMax = (float)Math.Atan(sigmaMax), denominator = steps - 1;
        return WithZero(Enumerable.Range(0, steps).Select(i =>
        {
            float a = i / denominator;
            return MathF.Tan(a * aMin + (1 - a) * aMax);
        }));
    }
}
