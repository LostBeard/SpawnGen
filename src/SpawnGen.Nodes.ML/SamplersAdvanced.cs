namespace SpawnGen.Nodes.ML;

/// <summary>
/// The less common k-diffusion samplers ComfyUI ships: DPM-Solver fast / adaptive, DEIS, the CFG++ history sampler, ER-SDE, SEEDS, SA-Solver and UniPC.
/// Same conventions as <see cref="Samplers"/>.
/// </summary>
public static partial class Samplers
{
    /// <summary>
    /// A Brownian motion W over sigma, built on demand from independent normal draws, standing in for k-diffusion's BrownianTreeNoiseSampler: the noise for
    /// (sigma, sigma_next) is (W(sigma_next) - W(sigma)) / sqrt(|sigma_next - sigma|). A new point between known ones is drawn from the Brownian bridge, one outside
    /// them extends the path, so overlapping intervals get correctly correlated noise (dpmpp_sde's two stages share part of their interval).
    /// </summary>
    internal sealed class BrownianNoise
    {
        readonly NoiseSource _source;
        readonly SortedList<double, float[]> _points = [];

        public BrownianNoise(NoiseSource source, double sigmaMin, double sigmaMax)
        {
            _source = source;
            var first = source.Next();
            _points[sigmaMin] = new float[first.Length];
            if (sigmaMax > sigmaMin)
            {
                double scale = Math.Sqrt(sigmaMax - sigmaMin);
                _points[sigmaMax] = first.Select(v => (float)(v * scale)).ToArray();
            }
        }

        /// <summary>W at <paramref name="tau"/>.</summary>
        public float[] At(double tau)
        {
            if (_points.TryGetValue(tau, out var known)) return known;
            var keys = _points.Keys;
            int hi = 0;
            while (hi < keys.Count && keys[hi] < tau) hi++;
            var z = _source.Next();
            var w = new float[z.Length];
            if (hi == 0)
            {
                // Below every known point: an independent increment downwards.
                double lo = keys[0], scale = Math.Sqrt(lo - tau);
                var basis = _points.Values[0];
                for (int k = 0; k < w.Length; k++) w[k] = (float)(basis[k] + z[k] * scale);
            }
            else if (hi == keys.Count)
            {
                double top = keys[^1], scale = Math.Sqrt(tau - top);
                var basis = _points.Values[^1];
                for (int k = 0; k < w.Length; k++) w[k] = (float)(basis[k] + z[k] * scale);
            }
            else
            {
                double a = keys[hi - 1], b = keys[hi], frac = (tau - a) / (b - a), std = Math.Sqrt((tau - a) * (b - tau) / (b - a));
                var wa = _points.Values[hi - 1];
                var wb = _points.Values[hi];
                for (int k = 0; k < w.Length; k++) w[k] = (float)(wa[k] + frac * (wb[k] - wa[k]) + std * z[k]);
            }
            _points[tau] = w;
            return w;
        }

        /// <summary>The normalised increment from <paramref name="sigma"/> to <paramref name="sigmaNext"/>: a standard normal draw per element.</summary>
        public float[] Increment(double sigma, double sigmaNext)
        {
            var w0 = At(sigma);
            var w1 = At(sigmaNext);
            double scale = 1 / Math.Sqrt(Math.Abs(sigmaNext - sigma));
            var r = new float[w0.Length];
            for (int k = 0; k < r.Length; k++) r[k] = (float)((w1[k] - w0[k]) * scale);
            return r;
        }
    }

    // --- DPM-Solver (fast and adaptive), in t = -log(sigma) and eps space ------------------------------------------------------------------

    sealed class DpmSolver(Run run)
    {
        public static double T(double sigma) => -Math.Log(sigma);
        public static double Sigma(double t) => Math.Exp(-t);

        public async Task<float[]> Eps(Dictionary<string, float[]> cache, string key, float[] x, double t)
        {
            if (cache.TryGetValue(key, out var hit)) return hit;
            double sigma = Sigma(t);
            var denoised = await run.Call(x, sigma);
            var eps = new float[x.Length];
            for (int k = 0; k < eps.Length; k++) eps[k] = (float)((x[k] - denoised[k]) / sigma);
            cache[key] = eps;
            return eps;
        }

        public async Task<float[]> Step1(float[] x, double t, double tNext, Dictionary<string, float[]> cache)
        {
            double h = tNext - t;
            var eps = await Eps(cache, "eps", x, t);
            return Lin(x, 1, eps, -Sigma(tNext) * Expm1(h));
        }

        public async Task<float[]> Step2(float[] x, double t, double tNext, Dictionary<string, float[]> cache, double r1 = 0.5)
        {
            double h = tNext - t;
            var eps = await Eps(cache, "eps", x, t);
            double s1 = t + r1 * h;
            var u1 = Lin(x, 1, eps, -Sigma(s1) * Expm1(r1 * h));
            var epsR1 = await Eps(cache, "eps_r1", u1, s1);
            var r = new float[x.Length];
            double a = Sigma(tNext) * Expm1(h), b = Sigma(tNext) / (2 * r1) * Expm1(h);
            for (int k = 0; k < r.Length; k++) r[k] = (float)(x[k] - a * eps[k] - b * (epsR1[k] - eps[k]));
            return r;
        }

        public async Task<float[]> Step3(float[] x, double t, double tNext, Dictionary<string, float[]> cache, double r1 = 1.0 / 3, double r2 = 2.0 / 3)
        {
            double h = tNext - t;
            var eps = await Eps(cache, "eps", x, t);
            double s1 = t + r1 * h, s2 = t + r2 * h;
            var u1 = Lin(x, 1, eps, -Sigma(s1) * Expm1(r1 * h));
            var epsR1 = await Eps(cache, "eps_r1", u1, s1);
            var u2 = new float[x.Length];
            double c1 = Sigma(s2) * Expm1(r2 * h), c2 = Sigma(s2) * (r2 / r1) * (Expm1(r2 * h) / (r2 * h) - 1);
            for (int k = 0; k < u2.Length; k++) u2[k] = (float)(x[k] - c1 * eps[k] - c2 * (epsR1[k] - eps[k]));
            var epsR2 = await Eps(cache, "eps_r2", u2, s2);
            var r = new float[x.Length];
            double a = Sigma(tNext) * Expm1(h), b = Sigma(tNext) / r2 * (Expm1(h) / h - 1);
            for (int k = 0; k < r.Length; k++) r[k] = (float)(x[k] - a * eps[k] - b * (epsR2[k] - eps[k]));
            return r;
        }
    }

    /// <summary>The lowest sigma dpm_fast / dpm_adaptive aim for: the last one, or the one before when the schedule ends in 0 (ComfyUI's wrappers).</summary>
    static double DpmSigmaMin(double[] s) => s[^1] == 0 ? s[^2] : s[^1];

    /// <summary>DPM-Solver-Fast: fixed steps in t of orders 3, 3, ... chosen so the model is called as many times as the schedule has steps.</summary>
    static async Task<float[]> DpmFast(Run run, float[] x, double[] s)
    {
        if (s.Length <= 1) return x;
        double sigmaMin = DpmSigmaMin(s), sigmaMax = s[0];
        int nfe = s.Length - 1;
        if (sigmaMin <= 0 || sigmaMax <= 0) throw new ArgumentException("sigma_min and sigma_max must not be 0");
        var solver = new DpmSolver(run);
        double tStart = DpmSolver.T(sigmaMax), tEnd = DpmSolver.T(sigmaMin);
        int m = nfe / 3 + 1;
        var ts = ModelSigmas.Linspace(tStart, tEnd, m + 1);
        var orders = nfe % 3 == 0 ? [.. Enumerable.Repeat(3, m - 2), 2, 1] : Enumerable.Repeat(3, m - 1).Append(nfe % 3).ToArray();
        for (int i = 0; i < orders.Length; i++)
        {
            var cache = new Dictionary<string, float[]>();
            double t = ts[i], tNext = ts[i + 1];
            var eps = await solver.Eps(cache, "eps", x, t);
            run.Callback(i, Lin(x, 1, eps, -DpmSolver.Sigma(t)));
            x = orders[i] switch
            {
                1 => await solver.Step1(x, t, tNext, cache),
                2 => await solver.Step2(x, t, tNext, cache),
                _ => await solver.Step3(x, t, tNext, cache),
            };
            // eta is 0 here, so no fresh noise is added (ComfyUI draws it and multiplies it by 0).
        }
        return x;
    }

    /// <summary>DPM-Solver-12 / 23 with adaptive step size: a PID controller on the difference between an order-(k-1) and an order-k step.</summary>
    static async Task<float[]> DpmAdaptive(Run run, float[] x, double[] s)
    {
        if (s.Length <= 1) return x;
        var o = run.Options;
        int order = o.Order ?? 3;
        double rtol = o.Rtol ?? 0.05, atol = o.Atol ?? 0.0078, hInit = o.HInit ?? 0.05, pcoeff = o.PCoeff ?? 0, icoeff = o.ICoeff ?? 1, dcoeff = o.DCoeff ?? 0;
        double acceptSafety = o.AcceptSafety ?? 0.81, eta = o.Eta ?? 0, sNoise = o.SNoise ?? 1;
        double sigmaMin = DpmSigmaMin(s), sigmaMax = s[0];
        if (sigmaMin <= 0 || sigmaMax <= 0) throw new ArgumentException("sigma_min and sigma_max must not be 0");
        if (order is not (2 or 3)) throw new ArgumentException("order should be 2 or 3");
        var solver = new DpmSolver(run);
        double tStart = DpmSolver.T(sigmaMax), tEnd = DpmSolver.T(sigmaMin);
        bool forward = tEnd > tStart;
        if (!forward && eta != 0) throw new ArgumentException("eta must be 0 for reverse sampling");
        hInit = Math.Abs(hInit) * (forward ? 1 : -1);
        double sCur = tStart;
        var xPrev = x;
        var pid = new PidStepSizeController(hInit, pcoeff, icoeff, dcoeff, eta != 0 ? 1.5 : order, acceptSafety);
        int steps = 0;
        while (forward ? sCur < tEnd - 1e-5 : sCur > tEnd + 1e-5)
        {
            var cache = new Dictionary<string, float[]>();
            double t = forward ? Math.Min(tEnd, sCur + pid.H) : Math.Max(tEnd, sCur + pid.H), tUse, su;
            if (eta != 0)
            {
                var (sd, up) = AncestralStep(DpmSolver.Sigma(sCur), DpmSolver.Sigma(t), eta);
                tUse = Math.Min(tEnd, DpmSolver.T(sd));
                su = Math.Sqrt(DpmSolver.Sigma(t) * DpmSolver.Sigma(t) - DpmSolver.Sigma(tUse) * DpmSolver.Sigma(tUse));
            }
            else (tUse, su) = (t, 0.0);

            var eps = await solver.Eps(cache, "eps", x, sCur);
            var denoised = Lin(x, 1, eps, -DpmSolver.Sigma(sCur));
            float[] xLow, xHigh;
            if (order == 2)
            {
                xLow = await solver.Step1(x, sCur, tUse, cache);
                xHigh = await solver.Step2(x, sCur, tUse, cache);
            }
            else
            {
                xLow = await solver.Step2(x, sCur, tUse, cache, r1: 1.0 / 3);
                xHigh = await solver.Step3(x, sCur, tUse, cache);
            }
            double sum = 0;
            for (int k = 0; k < x.Length; k++)
            {
                double delta = Math.Max(atol, rtol * Math.Max(Math.Abs(xLow[k]), Math.Abs(xPrev[k])));
                double e = (xLow[k] - xHigh[k]) / delta;
                sum += e * e;
            }
            double error = Math.Sqrt(sum) / Math.Sqrt(x.Length);
            if (pid.ProposeStep(error))
            {
                xPrev = xLow;
                x = xHigh;
                if (eta != 0) AddScaled(x, run.NextNoise(), su * sNoise);
                sCur = t;
            }
            steps++;
            run.OnDenoised?.Invoke(denoised);
            run.OnStep?.Invoke(Math.Min(steps, s.Length - 1));
            run.Ct.ThrowIfCancellationRequested();
        }
        return x;
    }

    sealed class PidStepSizeController(double h, double pcoeff, double icoeff, double dcoeff, double order, double acceptSafety, double eps = 1e-8)
    {
        readonly double _b1 = (pcoeff + icoeff + dcoeff) / order, _b2 = -(pcoeff + 2 * dcoeff) / order, _b3 = dcoeff / order;
        double[]? _errs;
        public double H { get; private set; } = h;

        static double Limiter(double x) => 1 + Math.Atan(x - 1);

        public bool ProposeStep(double error)
        {
            double inv = 1 / ((float)error + eps);
            _errs ??= [inv, inv, inv];
            _errs[0] = inv;
            double factor = Limiter(Math.Pow(_errs[0], _b1) * Math.Pow(_errs[1], _b2) * Math.Pow(_errs[2], _b3));
            bool accept = factor >= acceptSafety;
            if (accept) { _errs[2] = _errs[1]; _errs[1] = _errs[0]; }
            H *= factor;
            return accept;
        }
    }

    // --- DEIS ------------------------------------------------------------------------------------------------------------------------------

    /// <summary>DEIS in its 'tab' mode (sample_deis, max order 3): multistep coefficients from integrating the Lagrange basis against the VP integrand.</summary>
    static async Task<float[]> Deis(Run run, float[] x, double[] s)
    {
        const int maxOrder = 3;
        var coeffs = DeisCoefficients(s, maxOrder);
        var buffer = new List<float[]>();
        for (int i = 0; i < s.Length - 1; i++)
        {
            double tCur = s[i], tNext = s[i + 1];
            var denoised = await run.Call(x, tCur);
            run.Callback(i, denoised);
            var d = ToD(x, tCur, denoised);
            int order = Math.Min(maxOrder, i + 1);
            if (tNext <= 0) order = 1;
            if (order == 1) x = Lin(x, 1, d, tNext - tCur);
            else
            {
                var c = coeffs[i];
                var next = new float[x.Length];
                for (int k = 0; k < x.Length; k++)
                {
                    double v = x[k] + c[0] * d[k];
                    for (int j = 1; j < order; j++) v += c[j] * buffer[^j][k];
                    next[k] = (float)v;
                }
                x = next;
            }
            PushHistory(buffer, d, maxOrder - 1);
        }
        return x;
    }

    /// <summary>deis.get_deis_coeff_list in 'tab' mode: EDM sigmas to VP time (sigma range 0.002 to 80, eps_s 1e-3), then a 10000-point Riemann sum per coefficient.</summary>
    internal static double[][] DeisCoefficients(double[] sigmas, int maxOrder, int n = 10000)
    {
        const double epsilonS = 1e-3, sigmaMin = 0.002, sigmaMax = 80;
        double betaD = 2 * (Math.Log(sigmaMin * sigmaMin + 1) / epsilonS - Math.Log(sigmaMax * sigmaMax + 1)) / (epsilonS - 1);
        double betaMin = Math.Log(sigmaMax * sigmaMax + 1) - 0.5 * betaD;
        double beta0 = betaMin, beta1 = betaD + betaMin;
        var t = sigmas.Select(sg => (Math.Sqrt(betaMin * betaMin + 2 * betaD * Math.Log(sg * sg + 1)) - betaMin) / betaD).ToArray();
        var result = new double[t.Length - 1][];
        for (int i = 0; i < t.Length - 1; i++)
        {
            int order = Math.Min(i + 1, maxOrder);
            if (order == 1) { result[i] = []; continue; }
            double tCur = t[i], tNext = t[i + 1], dtau = (tNext - tCur) / n;
            var prev = Enumerable.Range(0, order).Select(k => t[i - k]).ToArray();
            var c = new double[order];
            for (int q = 0; q < n; q++)
            {
                double tau = n == 1 ? tCur : tCur + (tNext - tCur) * q / (n - 1);
                double alpha = Math.Exp(-0.5 * tau * tau * (beta1 - beta0) - tau * beta0);
                double dLogAlpha = -tau * (beta1 - beta0) - beta0;
                double integrand = -0.5 * dLogAlpha / Math.Sqrt(alpha * (1 - alpha));
                for (int j = 0; j < order; j++)
                {
                    double poly = 1;
                    for (int k = 0; k < order; k++)
                        if (k != j) poly *= (tau - prev[k]) / (prev[j] - prev[k]);
                    c[j] += integrand * poly;
                }
            }
            for (int j = 0; j < order; j++) c[j] *= dtau;
            result[i] = c;
        }
        return result;
    }

    // --- CFG++ with history (cfgpp_ud10_ab) ------------------------------------------------------------------------------------------------

    /// <summary>
    /// ComfyUI's _sample_cfgpp_history: CFG++ Euler with a variable-step AB2 history of the step, a short history on the unconditional derivative, and an
    /// extrapolation of the denoised estimate to sigma 0 on the last step.
    /// </summary>
    static async Task<float[]> CfgppHistory(Run run, float[] x, double[] s, double historyWeight, double? zeroWeight, int zeroOrder, double uncondHistoryWeight)
    {
        var derivatives = new List<float[]>();
        var denoisedHistory = new List<float[]>();
        float[]? oldUncondD = null;
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            var uncond = run.UncondOrCond(denoised);
            run.Callback(i, denoised);
            double alphaS = s[i] * Math.Exp(run.Lambda(s[i])), alphaT = s[i + 1] != 0 ? s[i + 1] * Math.Exp(run.Lambda(s[i + 1])) : 1.0;
            var currentUncondD = new float[x.Length];
            for (int k = 0; k < x.Length; k++) currentUncondD[k] = (float)((x[k] - alphaS * uncond[k]) / s[i]);
            var uncondD = currentUncondD;
            double dt = s[i + 1] - s[i];
            if (i > 0 && uncondHistoryWeight != 0)
            {
                double stepRatio = dt / (s[i] - s[i - 1]);
                uncondD = Lin(currentUncondD, 1, Sub(currentUncondD, oldUncondD!), uncondHistoryWeight * stepRatio);
            }
            var eulerStep = new float[x.Length];
            for (int k = 0; k < x.Length; k++) eulerStep[k] = (float)(alphaT * denoised[k] + s[i + 1] * uncondD[k] - x[k]);
            derivatives.Add(eulerStep.Select(v => (float)(v / dt)).ToArray());
            if (derivatives.Count > 2) derivatives.RemoveAt(0);
            float[] step;
            if (derivatives.Count == 1) step = eulerStep;
            else
            {
                double c0 = LinearMultistepCoeff(2, s, i, 0), c1 = LinearMultistepCoeff(2, s, i, 1);
                var historyStep = Lin(derivatives[^1], c0, derivatives[^2], c1);
                step = Lin(eulerStep, 1 - historyWeight, historyStep, historyWeight);
            }
            AddScaled(x, step, 1);
            if (s[i + 1] == 0 && zeroWeight is { } zw && denoisedHistory.Count > 0)
            {
                float[] zeroPrediction;
                if (zeroOrder == 2 && denoisedHistory.Count > 1)
                {
                    double s0 = s[i - 2], s1 = s[i - 1], s2 = s[i];
                    double w0 = s1 * s2 / ((s0 - s1) * (s0 - s2)), w1 = s0 * s2 / ((s1 - s0) * (s1 - s2)), w2 = s0 * s1 / ((s2 - s0) * (s2 - s1));
                    zeroPrediction = new float[x.Length];
                    for (int k = 0; k < x.Length; k++) zeroPrediction[k] = (float)(w0 * denoisedHistory[^2][k] + w1 * denoisedHistory[^1][k] + w2 * denoised[k]);
                }
                else
                {
                    zeroPrediction = new float[x.Length];
                    for (int k = 0; k < x.Length; k++)
                    {
                        double slope = (denoised[k] - denoisedHistory[^1][k]) / (s[i] - s[i - 1]);
                        zeroPrediction[k] = (float)(denoised[k] - s[i] * slope);
                    }
                }
                x = Lin(x, 1 - zw, zeroPrediction, zw);
            }
            denoisedHistory.Add(denoised);
            if (denoisedHistory.Count > 2) denoisedHistory.RemoveAt(0);
            oldUncondD = currentUncondD;
        }
        return x;
    }

    // --- ER-SDE ----------------------------------------------------------------------------------------------------------------------------

    /// <summary>ER-SDE-Solver-3 (sample_er_sde, arXiv 2309.06169): up to three stages of history, with the noise scaler choosing between ER-SDE, reverse-time SDE and ODE.</summary>
    static async Task<float[]> ErSde(Run run, float[] x, double[] s, double sNoise, Func<double, double>? noiseScaler, int maxStage)
    {
        sNoise *= run.Sampling.NoiseScale;
        noiseScaler ??= v => v * (Math.Exp(Math.Pow(v, 0.3)) + 10.0);
        const double numPoints = 200.0;
        s = run.OffsetFirstSigma(s);
        var erLambdas = s.Select(v => Math.Exp(-run.Lambda(v))).ToArray();
        float[]? oldDenoised = null, oldDenoisedD = null;
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            run.Callback(i, denoised);
            int stage = Math.Min(maxStage, i + 1);
            if (s[i + 1] == 0) x = denoised;
            else
            {
                double ls = erLambdas[i], lt = erLambdas[i + 1];
                double alphaS = s[i] / ls, alphaT = s[i + 1] / lt, rAlpha = alphaT / alphaS;
                double r = noiseScaler(lt) / noiseScaler(ls);
                x = Lin(x, rAlpha * r, denoised, alphaT * (1 - r));
                if (stage >= 2)
                {
                    double dt = lt - ls, stepSize = -dt / numPoints, sum1 = 0, sumU = 0;
                    for (int p = 0; p < (int)numPoints; p++)
                    {
                        // ComfyUI forms these positions in float32.
                        double pos = (float)(lt + (float)p * (float)stepSize), scaled = noiseScaler(pos);
                        sum1 += 1 / scaled;
                        sumU += (pos - ls) / scaled;
                    }
                    double sInt = sum1 * stepSize;
                    var denoisedD = new float[x.Length];
                    for (int k = 0; k < x.Length; k++) denoisedD[k] = (float)((denoised[k] - oldDenoised![k]) / (ls - erLambdas[i - 1]));
                    AddScaled(x, denoisedD, alphaT * (dt + sInt * noiseScaler(lt)));
                    if (stage >= 3)
                    {
                        double sU = sumU * stepSize;
                        var denoisedU = new float[x.Length];
                        for (int k = 0; k < x.Length; k++) denoisedU[k] = (float)((denoisedD[k] - oldDenoisedD![k]) / ((ls - erLambdas[i - 2]) / 2));
                        AddScaled(x, denoisedU, alphaT * (dt * dt / 2 + sU * noiseScaler(lt)));
                    }
                    oldDenoisedD = denoisedD;
                }
                if (sNoise > 0)
                {
                    double amount = Math.Sqrt(lt * lt - ls * ls * r * r);
                    if (double.IsNaN(amount)) amount = 0;
                    AddScaled(x, run.NextNoise(), alphaT * sNoise * amount);
                }
            }
            oldDenoised = denoised;
        }
        return x;
    }

    // --- SEEDS -----------------------------------------------------------------------------------------------------------------------------

    static double PhiH1(double h) => Expm1(h);
    static double PhiH2(double h) => (Expm1(h) - h) / h;

    /// <summary>SEEDS-2 (sample_seeds_2, arXiv 2305.14267); exp_heun_2_x0 (_sde) is its r = 1, phi_2 form.</summary>
    static async Task<float[]> Seeds2(Run run, float[] x, double[] s, double eta, double sNoise, double r, string solverType)
    {
        if (solverType is not ("phi_1" or "phi_2")) throw new ArgumentException("solver_type must be 'phi_1' or 'phi_2'");
        sNoise *= run.Sampling.NoiseScale;
        bool injectNoise = eta > 0 && sNoise > 0;
        s = run.OffsetFirstSigma(s);
        double fac = 1 / (2 * r);
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            run.Callback(i, denoised);
            if (s[i + 1] == 0) { x = denoised; continue; }
            double lambdaS = run.Lambda(s[i]), lambdaT = run.Lambda(s[i + 1]), h = lambdaT - lambdaS, hEta = h * (eta + 1);
            double lambdaS1 = lambdaS + r * (lambdaT - lambdaS), sigmaS1 = run.SigmaOfLambda(lambdaS1);
            double alphaS1 = sigmaS1 * Math.Exp(lambdaS1), alphaT = s[i + 1] * Math.Exp(lambdaT);

            // Step 1
            var x2 = Lin(x, sigmaS1 / s[i] * Math.Exp(-r * h * eta), denoised, -alphaS1 * PhiH1(-r * hEta));
            float[]? sdeNoise = null;
            if (injectNoise)
            {
                sdeNoise = run.NextNoise().Select(v => (float)(Math.Sqrt(-Expm1(-2 * r * h * eta)) * v)).ToArray();
                AddScaled(x2, sdeNoise, sigmaS1 * sNoise);
            }
            var denoised2 = await run.Call(x2, sigmaS1);

            // Step 2
            if (solverType == "phi_1")
            {
                var denoisedD = Lin(denoised, 1 - fac, denoised2, fac);
                x = Lin(x, s[i + 1] / s[i] * Math.Exp(-h * eta), denoisedD, -alphaT * PhiH1(-hEta));
            }
            else
            {
                double b2 = PhiH2(-hEta) / r, b1 = PhiH1(-hEta) - b2;
                var next = new float[x.Length];
                double keep = s[i + 1] / s[i] * Math.Exp(-h * eta);
                for (int k = 0; k < x.Length; k++) next[k] = (float)(keep * x[k] - alphaT * (b1 * denoised[k] + b2 * denoised2[k]));
                x = next;
            }
            if (injectNoise)
            {
                double segment = (r - 1) * h * eta;
                var fresh = run.NextNoise();
                for (int k = 0; k < x.Length; k++) sdeNoise![k] = (float)(sdeNoise[k] * Math.Exp(segment) + Math.Sqrt(-Expm1(2 * segment)) * fresh[k]);
                AddScaled(x, sdeNoise!, s[i + 1] * sNoise);
            }
        }
        return x;
    }

    /// <summary>SEEDS-3 (sample_seeds_3): three stages at r_1 = 1/3 and r_2 = 2/3 of the step in logSNR.</summary>
    static async Task<float[]> Seeds3(Run run, float[] x, double[] s, double eta, double sNoise, double r1 = 1.0 / 3, double r2 = 2.0 / 3)
    {
        sNoise *= run.Sampling.NoiseScale;
        bool injectNoise = eta > 0 && sNoise > 0;
        s = run.OffsetFirstSigma(s);
        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(x, s[i]);
            run.Callback(i, denoised);
            if (s[i + 1] == 0) { x = denoised; continue; }
            double lambdaS = run.Lambda(s[i]), lambdaT = run.Lambda(s[i + 1]), h = lambdaT - lambdaS, hEta = h * (eta + 1);
            double lambdaS1 = lambdaS + r1 * (lambdaT - lambdaS), lambdaS2 = lambdaS + r2 * (lambdaT - lambdaS);
            double sigmaS1 = run.SigmaOfLambda(lambdaS1), sigmaS2 = run.SigmaOfLambda(lambdaS2);
            double alphaS1 = sigmaS1 * Math.Exp(lambdaS1), alphaS2 = sigmaS2 * Math.Exp(lambdaS2), alphaT = s[i + 1] * Math.Exp(lambdaT);

            // Step 1
            var x2 = Lin(x, sigmaS1 / s[i] * Math.Exp(-r1 * h * eta), denoised, -alphaS1 * PhiH1(-r1 * hEta));
            float[]? sdeNoise = null;
            if (injectNoise)
            {
                sdeNoise = run.NextNoise().Select(v => (float)(Math.Sqrt(-Expm1(-2 * r1 * h * eta)) * v)).ToArray();
                AddScaled(x2, sdeNoise, sigmaS1 * sNoise);
            }
            var denoised2 = await run.Call(x2, sigmaS1);

            // Step 2
            double a32 = r2 / r1 * PhiH2(-r2 * hEta), a31 = PhiH1(-r2 * hEta) - a32;
            var x3 = new float[x.Length];
            double keep3 = sigmaS2 / s[i] * Math.Exp(-r2 * h * eta);
            for (int k = 0; k < x.Length; k++) x3[k] = (float)(keep3 * x[k] - alphaS2 * (a31 * denoised[k] + a32 * denoised2[k]));
            if (injectNoise)
            {
                double segment = (r1 - r2) * h * eta;
                var fresh = run.NextNoise();
                for (int k = 0; k < x.Length; k++) sdeNoise![k] = (float)(sdeNoise[k] * Math.Exp(segment) + Math.Sqrt(-Expm1(2 * segment)) * fresh[k]);
                AddScaled(x3, sdeNoise!, sigmaS2 * sNoise);
            }
            var denoised3 = await run.Call(x3, sigmaS2);

            // Step 3
            double b3 = PhiH2(-hEta) / r2, b1 = PhiH1(-hEta) - b3;
            var next = new float[x.Length];
            double keep = s[i + 1] / s[i] * Math.Exp(-h * eta);
            for (int k = 0; k < x.Length; k++) next[k] = (float)(keep * x[k] - alphaT * (b1 * denoised[k] + b3 * denoised3[k]));
            x = next;
            if (injectNoise)
            {
                double segment = (r2 - 1) * h * eta;
                var fresh = run.NextNoise();
                for (int k = 0; k < x.Length; k++) sdeNoise![k] = (float)(sdeNoise[k] * Math.Exp(segment) + Math.Sqrt(-Expm1(2 * segment)) * fresh[k]);
                AddScaled(x, sdeNoise!, s[i + 1] * sNoise);
            }
        }
        return x;
    }

    // --- SA-Solver -------------------------------------------------------------------------------------------------------------------------

    /// <summary>sa_solver.get_tau_interval_func: the stochasticity is <paramref name="eta"/> while sigma is within [end, start], else 0 (an ODE step).</summary>
    public static Func<double, double> TauInterval(double startSigma, double endSigma, double eta = 1.0) =>
        sigma => eta <= 0 ? 0.0 : (startSigma >= sigma && sigma >= endSigma ? eta : 0.0);

    /// <summary>Stochastic Adams solver with predictor and corrector (sample_sa_solver, NeurIPS 2023); PECE re-evaluates the corrected state.</summary>
    static async Task<float[]> SaSolver(Run run, float[] x, double[] s, bool usePece)
    {
        if (s.Length <= 1) return x;
        var o = run.Options;
        double sNoise = (o.SNoise ?? 1) * run.Sampling.NoiseScale;
        int predictorOrder = o.PredictorOrder ?? 3, correctorOrder = o.CorrectorOrder ?? 4;
        bool simpleOrder2 = o.SimpleOrder2 ?? false;
        s = run.OffsetFirstSigma(s);
        var lambdas = s.Select(run.Lambda).ToArray();
        var tauFunc = o.TauFunc ?? TauInterval(run.Sampling.PercentToSigma(0.2), run.Sampling.PercentToSigma(0.8), 1.0);
        int maxUsedOrder = Math.Max(predictorOrder, correctorOrder);
        var xPred = x;
        double h = 0, tauT = 0;
        float[]? noise = null;
        var predList = new List<float[]>();
        bool lowerOrderToEnd = s[^1] == 0;

        for (int i = 0; i < s.Length - 1; i++)
        {
            var denoised = await run.Call(xPred, s[i]);
            run.Callback(i, denoised);
            predList.Add(denoised);
            if (predList.Count > maxUsedOrder) predList.RemoveRange(0, predList.Count - maxUsedOrder);

            int predictorUsed = Math.Min(predictorOrder, predList.Count);
            int correctorUsed = i == 0 || (s[i + 1] == 0 && !usePece) ? 0 : Math.Min(correctorOrder, predList.Count);
            if (lowerOrderToEnd)
            {
                predictorUsed = Math.Min(predictorUsed, s.Length - 2 - i);
                correctorUsed = Math.Min(correctorUsed, s.Length - 1 - i);
            }

            // Corrector
            if (correctorUsed == 0) x = xPred;
            else
            {
                var currLambdas = lambdas[(i - correctorUsed + 1)..(i + 1)];
                var b = StochasticAdamsCoefficients(s[i], currLambdas, lambdas[i - 1], lambdas[i], tauT, simpleOrder2, isCorrector: true);
                var corr = Combine(predList, correctorUsed, b);
                x = Lin(x, s[i] / s[i - 1] * Math.Exp(-(tauT * tauT) * h), corr, 1);
                if (tauT > 0 && sNoise > 0) AddScaled(x, noise!, 1);
                if (usePece)
                {
                    denoised = await run.Call(x, s[i]);
                    predList[^1] = denoised;
                }
            }

            // Predictor
            if (s[i + 1] == 0) xPred = denoised;
            else
            {
                tauT = tauFunc(s[i + 1]);
                var currLambdas = lambdas[(i - predictorUsed + 1)..(i + 1)];
                var b = StochasticAdamsCoefficients(s[i + 1], currLambdas, lambdas[i], lambdas[i + 1], tauT, simpleOrder2, isCorrector: false);
                var pred = Combine(predList, predictorUsed, b);
                h = lambdas[i + 1] - lambdas[i];
                xPred = Lin(x, s[i + 1] / s[i] * Math.Exp(-(tauT * tauT) * h), pred, 1);
                if (tauT > 0 && sNoise > 0)
                {
                    double amount = s[i + 1] * Math.Sqrt(-Expm1(-2 * tauT * tauT * h)) * sNoise;
                    noise = run.NextNoise().Select(v => (float)(v * amount)).ToArray();
                    AddScaled(xPred, noise, 1);
                }
            }
        }
        return xPred;
    }

    /// <summary>sum_k b[k] * pred[^count + k]: the last <paramref name="count"/> predictions, oldest first, against the coefficients.</summary>
    static float[] Combine(List<float[]> preds, int count, double[] b)
    {
        var r = new float[preds[0].Length];
        for (int k = 0; k < r.Length; k++)
        {
            double v = 0;
            for (int j = 0; j < count; j++) v += preds[preds.Count - count + j][k] * b[j];
            r[k] = (float)v;
        }
        return r;
    }

    /// <summary>sa_solver.compute_exponential_coeffs: (1 + tau^2) * the integral of exp((1 + tau^2) x) x^p from s to t, with exp((1 + tau^2) t) factored out, p = 0 .. order - 1.</summary>
    static double[] ExponentialCoefficients(double s, double t, int order, double tau)
    {
        double tauMul = 1 + tau * tau, h = t - s;
        var product = new double[order];
        var logFactorial = new double[order];
        for (int p = 0; p < order; p++)
        {
            product[p] = Math.Pow(t, p) - Math.Pow(s, p) * Math.Exp(-tauMul * h);
            logFactorial[p] = LogGamma(p + 1);
        }
        var r = new double[order];
        for (int a = 0; a < order; a++)
            for (int b = 0; b <= a; b++)
            {
                int depth = a - b;
                double c = logFactorial[a] - logFactorial[b];
                if (tau > 0) c -= depth * Math.Log(tauMul);
                r[a] += Math.Exp(c) * (depth % 2 == 0 ? 1 : -1) * product[b];
            }
        return r;
    }

    static double LogGamma(double x)
    {
        // ln((x - 1)!) for the small whole numbers the solver orders need.
        double r = 0;
        for (int k = 2; k < (int)Math.Round(x); k++) r += Math.Log(k);
        return r;
    }

    /// <summary>sa_solver.compute_stochastic_adams_b_coeffs: the b_i of eqs. 15 and 18, from a Vandermonde solve (or the simple order-2 forms).</summary>
    static double[] StochasticAdamsCoefficients(double sigmaNext, double[] currLambdas, double lambdaS, double lambdaT, double tau, bool simpleOrder2, bool isCorrector)
    {
        int n = currLambdas.Length;
        double alphaT = sigmaNext * Math.Exp(lambdaT);
        if (simpleOrder2 && n == 2)
        {
            double tauMul = 1 + tau * tau, h = lambdaT - lambdaS;
            double b1, b2;
            if (isCorrector)
            {
                b1 = alphaT * (0.5 * tauMul * h);
                b2 = alphaT * -Expm1(-h * tauMul) - b1;
            }
            else
            {
                b2 = alphaT * (0.5 * tauMul * h * h) / (currLambdas[^2] - lambdaS);
                b1 = alphaT * -Expm1(-h * tauMul) - b2;
            }
            return [b2, b1];
        }
        var rhs = ExponentialCoefficients(lambdaS, lambdaT, n, tau);
        // vander(curr_lambdas, n, increasing=True).T: row p holds lambda_k^p.
        var m = new double[n, n];
        for (int p = 0; p < n; p++)
            for (int k = 0; k < n; k++) m[p, k] = Math.Pow(currLambdas[k], p);
        var sol = Solve(m, rhs);
        return sol.Select(v => alphaT * v).ToArray();
    }

    /// <summary>Gaussian elimination with partial pivoting (torch.linalg.solve's LU).</summary>
    internal static double[] Solve(double[,] a, double[] b)
    {
        int n = b.Length;
        var m = (double[,])a.Clone();
        var x = (double[])b.Clone();
        for (int col = 0; col < n; col++)
        {
            int pivot = col;
            for (int r = col + 1; r < n; r++) if (Math.Abs(m[r, col]) > Math.Abs(m[pivot, col])) pivot = r;
            if (pivot != col)
            {
                for (int c = 0; c < n; c++) (m[col, c], m[pivot, c]) = (m[pivot, c], m[col, c]);
                (x[col], x[pivot]) = (x[pivot], x[col]);
            }
            for (int r = col + 1; r < n; r++)
            {
                double f = m[r, col] / m[col, col];
                if (f == 0) continue;
                for (int c = col; c < n; c++) m[r, c] -= f * m[col, c];
                x[r] -= f * x[col];
            }
        }
        for (int r = n - 1; r >= 0; r--)
        {
            double v = x[r];
            for (int c = r + 1; c < n; c++) v -= m[r, c] * x[c];
            x[r] = v / m[r, r];
        }
        return x;
    }

    // --- UniPC -----------------------------------------------------------------------------------------------------------------------------

    /// <summary>
    /// UniPC (multistep, data prediction, order up to 3, lower order at the end; variant bh1 or bh2), as ComfyUI's sample_unipc runs it: in the variance-preserving
    /// space with sigma as its time, the last sigma 0 replaced by 0.001 (so the result keeps that much noise), and no corrector on the final step.
    /// </summary>
    static async Task<float[]> UniPC(Run run, float[] x, double[] sigmas, string variant)
    {
        var ts = (double[])sigmas.Clone();
        if (ts[^1] == 0) ts[^1] = 0.001;
        static double LogMean(double t) => 0.5 * Math.Log(1 / (t * t + 1));
        static double Alpha(double t) => Math.Exp(LogMean(t));
        static double Std(double t) => Math.Sqrt(1.0 - Math.Exp(2.0 * LogMean(t)));
        static double Lambda(double t) { double lm = LogMean(t); return lm - 0.5 * Math.Log(1.0 - Math.Exp(2.0 * lm)); }

        // The data prediction: back to k-diffusion space for the model, then x0 from the noise prediction.
        async Task<float[]> ModelFn(float[] xv, double t)
        {
            double up = Math.Sqrt(t * t + 1.0);
            var input = xv.Select(v => (float)(v * up)).ToArray();
            var denoised = await run.Call(input, t);
            double a = Alpha(t), sd = Std(t);
            var x0 = new float[xv.Length];
            for (int k = 0; k < x0.Length; k++)
            {
                double eps = (input[k] - denoised[k]) / t;
                x0[k] = (float)((xv[k] - sd * eps) / a);
            }
            return x0;
        }

        async Task<(float[] X, float[]? Model)> Update(float[] xv, List<float[]> modelPrev, List<double> tPrev, double t, int order, bool useCorrector)
        {
            double tPrev0 = tPrev[^1], lambdaPrev0 = Lambda(tPrev0), lambdaT = Lambda(t);
            var modelPrev0 = modelPrev[^1];
            double sigmaPrev0 = Std(tPrev0), sigmaT = Std(t), alphaT = Math.Exp(LogMean(t));
            double h = lambdaT - lambdaPrev0;
            var rks = new List<double>();
            var d1s = new List<float[]>();
            for (int i = 1; i < order; i++)
            {
                double rk = (Lambda(tPrev[^(i + 1)]) - lambdaPrev0) / h;
                rks.Add(rk);
                var mi = modelPrev[^(i + 1)];
                d1s.Add(mi.Select((v, k) => (float)((v - modelPrev0[k]) / rk)).ToArray());
            }
            rks.Add(1.0);
            double hh = -h, hPhi1 = Expm1(hh), hPhiK = hPhi1 / hh - 1, factorial = 1;
            double bH = variant == "bh1" ? hh : Expm1(hh);
            var rMat = new double[order, order];
            var b = new double[order];
            for (int i = 1; i <= order; i++)
            {
                for (int k = 0; k < order; k++) rMat[i - 1, k] = Math.Pow(rks[k], i - 1);
                b[i - 1] = hPhiK * factorial / bH;
                factorial *= i + 1;
                hPhiK = hPhiK / hh - 1 / factorial;
            }
            double[]? rhosP = null;
            if (d1s.Count > 0)
            {
                if (order == 2) rhosP = [0.5];
                else
                {
                    var sub = new double[order - 1, order - 1];
                    for (int r = 0; r < order - 1; r++) for (int c = 0; c < order - 1; c++) sub[r, c] = rMat[r, c];
                    rhosP = Solve(sub, b[..(order - 1)]);
                }
            }
            double[]? rhosC = null;
            if (useCorrector) rhosC = order == 1 ? [0.5] : Solve(rMat, b);

            var xT_ = new float[xv.Length];
            for (int k = 0; k < xv.Length; k++) xT_[k] = (float)(sigmaT / sigmaPrev0 * xv[k] - alphaT * hPhi1 * modelPrev0[k]);
            var xT = (float[])xT_.Clone();
            if (d1s.Count > 0)
                for (int k = 0; k < xv.Length; k++)
                {
                    double pred = 0;
                    for (int j = 0; j < d1s.Count; j++) pred += d1s[j][k] * rhosP![j];
                    xT[k] = (float)(xT_[k] - alphaT * bH * pred);
                }
            float[]? modelT = null;
            if (useCorrector)
            {
                modelT = await ModelFn(xT, t);
                for (int k = 0; k < xv.Length; k++)
                {
                    double corr = 0;
                    for (int j = 0; j < d1s.Count; j++) corr += d1s[j][k] * rhosC![j];
                    double d1t = modelT[k] - modelPrev0[k];
                    xT[k] = (float)(xT_[k] - alphaT * bH * (corr + rhosC![^1] * d1t));
                }
            }
            return (xT, modelT);
        }

        int steps = ts.Length - 1;
        int uniOrder = Math.Min(3, ts.Length - 2);
        var xs = x.Select(v => (float)(v / Math.Sqrt(1.0 + ts[0] * ts[0]))).ToArray();
        var modelPrevList = new List<float[]>();
        var tPrevList = new List<double>();
        for (int stepIndex = 0; stepIndex < steps; stepIndex++)
        {
            if (stepIndex == 0)
            {
                modelPrevList.Add(await ModelFn(xs, ts[0]));
                tPrevList.Add(ts[0]);
            }
            else if (stepIndex < uniOrder)
            {
                int initOrder = stepIndex;
                var (nx, mx) = await Update(xs, modelPrevList, tPrevList, ts[initOrder], initOrder, useCorrector: true);
                xs = nx;
                modelPrevList.Add(mx ?? await ModelFn(xs, ts[initOrder]));
                tPrevList.Add(ts[initOrder]);
            }
            else
            {
                int extra = stepIndex == steps - 1 ? 1 : 0;
                for (int step = stepIndex; step <= stepIndex + extra; step++)
                {
                    int stepOrder = Math.Min(uniOrder, steps + 1 - step);
                    bool useCorrector = step != steps;
                    var (nx, mx) = await Update(xs, modelPrevList, tPrevList, ts[step], stepOrder, useCorrector);
                    xs = nx;
                    for (int i = 0; i < uniOrder - 1; i++)
                    {
                        tPrevList[i] = tPrevList[i + 1];
                        modelPrevList[i] = modelPrevList[i + 1];
                    }
                    tPrevList[^1] = ts[step];
                    if (step < steps) modelPrevList[^1] = mx ?? await ModelFn(xs, ts[step]);
                }
            }
            run.Callback(stepIndex, modelPrevList[^1]);
        }
        double last = Alpha(ts[^1]);
        return xs.Select(v => (float)(v / last)).ToArray();
    }
}
