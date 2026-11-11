namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md). DPM-Solver fast / adaptive follow k-diffusion (MIT) and Lu et al. 2022; Heun++ (heunpp2) follows
// Carzit's sd-webui-samplers-scheduler (MIT). The other samplers come from their papers (cited at each), with the details ComfyUI chose fitted
// to Fixtures/sampling/reference.json and measured by calling ComfyUI's samplers as black boxes.

/// <summary>A sampler's options (what the sampler nodes set).</summary>
public sealed record SamplerOptions
{
    public double? Eta { get; init; }
    public double? SNoise { get; init; }
    public double? R { get; init; }
    public string? SolverType { get; init; }
    public int? Order { get; init; }
    public double? Rtol { get; init; }
    public double? Atol { get; init; }
    public double? HInit { get; init; }
    public double? PCoeff { get; init; }
    public double? ICoeff { get; init; }
    public double? DCoeff { get; init; }
    public double? AcceptSafety { get; init; }
    public int? MaxStage { get; init; }
    public Func<double, double>? NoiseScaler { get; init; }
    public Func<double, double>? TauFunc { get; init; }
    public int? PredictorOrder { get; init; }
    public int? CorrectorOrder { get; init; }
    public bool? UsePece { get; init; }
    public bool? SimpleOrder2 { get; init; }
    public double? SNoiseEnd { get; init; }
    public double? NoiseClipStd { get; init; }
}

public static partial class Samplers
{
    static Task<float[]> RunAdvanced(string sampler, Run run, float[] x, double[] s) => sampler switch
    {
        "heunpp2" => Heunpp2(run, x, s),
        "euler_cfg_pp" => EulerCfgPP(run, x, s),
        "gradient_estimation" => GradientEstimation(run, x, s, cfgPP: false),
        "gradient_estimation_cfg_pp" => GradientEstimation(run, x, s, cfgPP: true),
        "res_multistep" => ResMultistep(run, x, s, cfgPP: false),
        "res_multistep_cfg_pp" => ResMultistep(run, x, s, cfgPP: true),
        "res_multistep_ancestral" => ResMultistep(run, x, s, cfgPP: false, ancestral: true),
        "res_multistep_ancestral_cfg_pp" => ResMultistep(run, x, s, cfgPP: true, ancestral: true),
        "ddpm" => Ddpm(run, x, s),
        "er_sde" => ErSde(run, x, s),
        "uni_pc" => UniPc(run, x, s, bh2: false),
        "sa_solver" => SaSolver(run, x, s, pece: false),
        "sa_solver_pece" => SaSolver(run, x, s, pece: true),
        "uni_pc_bh2" => UniPc(run, x, s, bh2: true),
        "seeds_2" => Seeds(run, x, s, 2, run.Options.SolverType ?? "phi_1", run.Eta(), run.SNoise, run.Options.R ?? 0.5),
        "seeds_3" => Seeds(run, x, s, 3, "phi_2", run.Eta(), run.SNoise, 1.0 / 3),
        "exp_heun_2_x0" => Seeds(run, x, s, 2, "phi_2", 0, run.SNoise, 1.0),
        "exp_heun_2_x0_sde" => Seeds(run, x, s, 2, "phi_2", run.Eta(), run.SNoise, 1.0),
        "lcm" => Lcm(run, x, s),
        "ipndm" => Ipndm(run, x, s),
        "ipndm_v" => IpndmV(run, x, s),
        "deis" => Deis(run, x, s),
        _ => throw new NotImplementedException(sampler),
    };

    // ---- CFG++ (Chung et al. 2024): move to the guided estimate, keep the unconditional estimate's noise ----

    static async Task<float[]> EulerCfgPP(Run run, float[] x, double[] s)
    {
        for (int i = 0; i < s.Length - 1; i++)
        {
            run.Check();
            var denoised = await run.Denoise(x, s[i]);
            var u = run.Uncond(denoised);
            await run.Callback(denoised);
            x = Step(denoised, ToD(x, s[i], u), s[i + 1]);
            run.Step(i);
        }
        return x;
    }

    /// <summary>DPM-Solver++(2M) with CFG++: the previous step's unconditional estimate in the extrapolation, plus ratio (D - U) (measured).</summary>
    static Task<float[]> Dpmpp2MCfgPPStep(Run run, float[] x, float[] denoised, float[]? oldUncond, double[] s, int i)
    {
        var u = run.Uncond(denoised);
        float t = -Log(s[i]), tNext = -Log(s[i + 1]), h = tNext - t;
        float ratio = s[i + 1] == 0 ? 0 : MathF.Exp(-tNext) / MathF.Exp(-t);
        float[] dd = denoised;
        if (oldUncond is not null && s[i + 1] != 0)
        {
            float hLast = t - -Log(s[i - 1]), r = hLast / h;
            dd = LinCombine(denoised, 1 + 1 / (2 * r), oldUncond, -(1 / (2 * r)));
        }
        float k = s[i + 1] == 0 ? 1 : -ExpM1(-h);
        var r2 = new float[x.Length];
        for (int j = 0; j < r2.Length; j++) r2[j] = ratio * x[j] + k * dd[j] + ratio * (denoised[j] - u[j]);
        return Task.FromResult(r2);
    }

    // ---- Gradient estimation (Permenter & Yuan 2023): gamma = 2 extrapolation of the derivative ----

    static async Task<float[]> GradientEstimation(Run run, float[] x, double[] s, bool cfgPP)
    {
        const float gamma = 2f;
        float[]? old = null;
        for (int i = 0; i < s.Length - 1; i++)
        {
            run.Check();
            var denoised = await run.Denoise(x, s[i]);
            var u = cfgPP ? run.Uncond(denoised) : denoised;
            await run.Callback(denoised);
            var d = ToD(x, s[i], u);
            float dt = (float)s[i + 1] - (float)s[i];
            if (s[i + 1] == 0) x = denoised;
            else if (old is null) x = cfgPP ? Step(denoised, d, s[i + 1]) : Step(x, d, dt);
            else
            {
                var r = new float[x.Length];
                for (int k = 0; k < r.Length; k++)
                {
                    float dBar = gamma * d[k] + (1 - gamma) * old[k];
                    r[k] = x[k] + dBar * dt + (cfgPP ? denoised[k] - u[k] : 0);
                }
                x = r;
            }
            old = d;
            run.Step(i);
        }
        return x;
    }

    // ---- RES multistep (Zhang, Tao & Chen 2023): second-order exponential multistep with phi functions ----

    static async Task<float[]> ResMultistep(Run run, float[] x, double[] s, bool cfgPP, bool ancestral = false)
    {
        double eta = ancestral ? run.Eta() : 0;
        float[]? old = null;
        double hLast = 0;
        for (int i = 0; i < s.Length - 1; i++)
        {
            run.Check();
            var denoised = await run.Denoise(x, s[i]);
            var u = cfgPP ? run.Uncond(denoised) : denoised;
            await run.Callback(denoised);
            var (down, up) = ancestral ? AncestralStep(s[i], s[i + 1], eta) : (s[i + 1], 0.0);
            float ratio = (float)down / (float)s[i];
            if (old is null || down == 0)
            {
                // First order: x' = ratio x + (1 - ratio) D; CFG++ keeps the unconditional noise.
                x = cfgPP ? LinCombine(denoised, 1, Subtract(x, u), ratio) : LinCombine(x, ratio, denoised, 1 - ratio);
                hLast = down == 0 ? 0 : Math.Log((float)s[i] / (float)down);
            }
            else
            {
                // The previous point is the previous step's own exponential step back (measured for the ancestral form; the same thing without eta).
                double h = Math.Log((float)s[i] / (float)down), c2 = -hLast / h, z = -h;
                double phi1 = (Math.Abs(z) < 1e-5 ? z + z * z / 2 : Math.Exp(z) - 1) / z, phi2 = (phi1 - 1) / z;
                double b1 = phi1 - phi2 / c2, b2 = phi2 / c2;
                float e = (float)Math.Exp(-h), w1 = (float)(h * b1), w2 = (float)(h * b2);
                var r = new float[x.Length];
                for (int k = 0; k < r.Length; k++)
                    r[k] = e * x[k] + w1 * u[k] + w2 * old[k] + (cfgPP ? ratio * (denoised[k] - u[k]) : 0);
                x = r;
                hLast = h;
            }
            if (up > 0 && s[i + 1] > 0) AddNoise(x, run.Noise.Next(), run.SNoise * up);
            old = u;
            run.Step(i);
        }
        return x;
    }

    // ---- SEEDS (Gonzalez et al. 2023): exponential SDE stages in data prediction, with Brownian noise shared between the stages ----

    static double Expm1D(double z) => Math.Abs(z) < 1e-5 ? z + z * z / 2 + z * z * z / 6 : Math.Exp(z) - 1;

    static double Phi1(double z) => Math.Abs(z) < 1e-8 ? 1 + z / 2 : Expm1D(z) / z;
    static double Phi2(double z) => Math.Abs(z) < 1e-5 ? 0.5 + z / 6 : (Phi1(z) - 1) / z;

    /// <summary>
    /// SEEDS-2 (stages at r) and SEEDS-3 (stages at 1/3, 2/3). With eta the deterministic part decays over h_eta = (1 + eta) h; each stage's
    /// noise is the Brownian increment of its sub-interval, carried to later stages with the factor e^(-eta (remaining h)). phi_1 uses the last
    /// stage's estimate alone, phi_2 adds the difference to the first (h_eta phi_2(-h_eta) / r).
    /// </summary>
    static async Task<float[]> Seeds(Run run, float[] x, double[] s, int stages, string solver, double eta, double sNoise, double r)
    {
        double[] fr = stages == 2 ? [r, 1] : [1.0 / 3, 2.0 / 3, 1];
        for (int i = 0; i < s.Length - 1; i++)
        {
            run.Check();
            var d0 = await run.Denoise(x, s[i]);
            await run.Callback(d0);
            if (s[i + 1] == 0)
            {
                x = d0;
                run.Step(i);
                continue;
            }
            double sig = (float)s[i], h = Math.Log(sig / (float)s[i + 1]), hEta = (1 + eta) * h;
            var noises = new List<float[]>();
            var estimates = new List<float[]> { d0 };
            float[] StageX(double c, bool final)
            {
                double sc = sig * Math.Exp(-c * h);
                double keepX = sc / sig * Math.Exp(-eta * c * h), total = -Expm1D(-c * hEta);
                var dw = new double[estimates.Count];
                dw[0] = total;
                if (estimates.Count > 1)
                {
                    // Second-order correction from the latest estimate (phi_2 form), or the latest estimate alone (phi_1 form, final only).
                    int last = estimates.Count - 1;
                    double cl = fr[last - 1];
                    if (final && solver == "phi_1") { dw[0] = 0; dw[last] = total; }
                    else
                    {
                        double w = c * hEta * Phi2(-c * hEta) / (cl / c);
                        dw[0] -= w;
                        dw[last] += w;
                    }
                }
                var nx = new float[x.Length];
                for (int k = 0; k < nx.Length; k++)
                {
                    double v = keepX * x[k];
                    for (int j = 0; j < dw.Length; j++) v += dw[j] * estimates[j][k];
                    nx[k] = (float)v;
                }
                if (eta != 0)
                {
                    // Noise of each earlier sub-interval, decayed to this stage, plus this stage's own.
                    double start = 0;
                    for (int j = 0; j < noises.Count; j++)
                    {
                        double end = fr[j], len = end - start;
                        double coef = sc * Math.Exp(-eta * (c - end) * h) * Math.Sqrt(-Expm1D(-2 * eta * len * h)) * sNoise;
                        AddNoise(nx, noises[j], coef);
                        start = end;
                    }
                }
                return nx;
            }
            for (int st = 0; st < stages - 1; st++)
            {
                if (eta != 0) noises.Add(run.Noise.Next());
                var xs = StageX(fr[st], final: false);
                estimates.Add(await run.Denoise(xs, sig * Math.Exp(-fr[st] * h)));
            }
            if (eta != 0) noises.Add(run.Noise.Next());
            x = StageX(1, final: true);
            run.Step(i);
        }
        return x;
    }

    // ---- ER-SDE-Solver (Cui et al. 2023): up to three stages, the integrals by a 200-interval left Riemann sum (measured) ----

    static async Task<float[]> ErSde(Run run, float[] x, double[] s)
    {
        var o = run.Options;
        int maxStage = o.MaxStage ?? 3;
        double sNoise = run.SNoise;
        var f = o.NoiseScaler ?? (v => v * (Math.Exp(Math.Pow(v, 0.3)) + 10));
        const int n = 200;
        float[]? old = null, oldD1 = null;
        double lPrev = 0, lPrev2 = 0;
        for (int i = 0; i < s.Length - 1; i++)
        {
            run.Check();
            var denoised = await run.Denoise(x, s[i]);
            await run.Callback(denoised);
            if (s[i + 1] == 0)
            {
                x = denoised;
                run.Step(i);
                continue;
            }
            double ls = (float)s[i], lt = (float)s[i + 1], r = f(lt) / f(ls);
            var nx = new double[x.Length];
            for (int k = 0; k < nx.Length; k++) nx[k] = r * x[k] + (1 - r) * denoised[k];
            int stage = Math.Min(maxStage, i + 1);
            float[]? d1 = null;
            if (stage >= 2 && old is not null)
            {
                double span = ls - lt, sInt = 0, suInt = 0;
                for (int m = 0; m < n; m++)
                {
                    double tau = lt + span * m / n, ft = f(tau);
                    sInt += 1 / ft;
                    suInt += (tau - ls) / ft;
                }
                sInt *= span / n;
                suInt *= span / n;
                double dl = lt - ls, a = dl + sInt * f(lt);
                d1 = new float[x.Length];
                for (int k = 0; k < nx.Length; k++)
                {
                    d1[k] = (float)((denoised[k] - old[k]) / (ls - lPrev));
                    nx[k] += a * d1[k];
                }
                if (stage >= 3 && oldD1 is not null)
                {
                    double b = dl * dl / 2 + suInt * f(lt), half = (ls - lPrev2) / 2;
                    for (int k = 0; k < nx.Length; k++) nx[k] += b * ((d1[k] - oldD1[k]) / half);
                }
            }
            x = [.. nx.Select(v => (float)v)];
            if (sNoise != 0) AddNoise(x, run.Noise.Next(), Math.Sqrt(Math.Max(lt * lt - ls * ls * r * r, 0)) * sNoise);
            oldD1 = d1;
            old = denoised;
            lPrev2 = lPrev;
            lPrev = ls;
            run.Step(i);
        }
        return x;
    }

    // ---- UniPC (Zhao et al. 2023): multistep predictor (UniP) and corrector (UniC), data prediction, in VP form ----

    /// <summary>
    /// UniPC with B(h) = h (bh1, "uni_pc") or e^h - 1 (bh2). Order min(3, sigmas - 2), warmed up with lower orders, the corrector after every
    /// step but the last (reusing the step's model call), a lower order near the end; a final sigma of 0 is run to 0.001.
    /// </summary>
    static async Task<float[]> UniPc(Run run, float[] x0, double[] sigmas, bool bh2)
    {
        var ts = sigmas.Select(v => (double)(float)v).ToArray();
        if (ts[^1] == 0) ts[^1] = 0.001;
        static double Alpha(double s) => 1 / Math.Sqrt(1 + s * s);
        static double SigVp(double s) => s / Math.Sqrt(1 + s * s);
        static double Lam(double s) => -Math.Log(s);
        int steps = ts.Length - 1, order = Math.Min(3, ts.Length - 2), callStep = 0;
        var x = x0.Select(v => (double)v * Alpha(ts[0])).ToArray();
        async Task<double[]> Model(double[] xv, double t)
        {
            run.Check();
            var input = xv.Select(v => (float)(v / Alpha(t))).ToArray();
            var d = await run.Denoise(input, t);
            await run.Callback(d);
            run.Step(callStep++);
            return [.. d.Select(v => (double)v)];
        }
        var models = new List<double[]> { await Model(x, ts[0]) };
        var times = new List<double> { ts[0] };
        if (order < 1) return [.. x.Select(v => (float)(v / Alpha(ts[^1])))];

        async Task<(double[] X, double[]? M)> Update(double[] xs, int ord, double t, bool corrector)
        {
            double s0 = times[^1];
            var m0 = models[^1];
            double h = Lam(t) - Lam(s0), hh = -h;
            var rks = new List<double>();
            var d1s = new List<double[]>();
            for (int i = 1; i < ord; i++)
            {
                double si = times[^(i + 1)];
                var mi = models[^(i + 1)];
                double rk = (Lam(si) - Lam(s0)) / h;
                rks.Add(rk);
                d1s.Add([.. mi.Select((v, k) => (v - m0[k]) / rk)]);
            }
            rks.Add(1.0);
            double hPhi1 = Expm1D(hh), hPhiK = hPhi1 / hh - 1, fac = 1, bH = bh2 ? Expm1D(hh) : hh;
            var rMat = new double[ord, ord];
            var b = new double[ord];
            for (int i = 1; i <= ord; i++)
            {
                for (int k = 0; k < ord; k++) rMat[i - 1, k] = Math.Pow(rks[k], i - 1);
                b[i - 1] = hPhiK * fac / bH;
                fac *= i + 1;
                hPhiK = hPhiK / hh - 1 / fac;
            }
            double[] rhosP = d1s.Count == 0 ? [] : ord == 2 ? [0.5] : SolveSub(rMat, b, ord - 1);
            double st = SigVp(t), at = Alpha(t), ss = SigVp(s0);
            var xt_ = new double[xs.Length];
            for (int k = 0; k < xs.Length; k++) xt_[k] = st / ss * xs[k] - at * hPhi1 * m0[k];
            var xt = new double[xs.Length];
            for (int k = 0; k < xs.Length; k++)
            {
                double pred = 0;
                for (int j = 0; j < d1s.Count; j++) pred += rhosP[j] * d1s[j][k];
                xt[k] = xt_[k] - at * bH * pred;
            }
            if (!corrector) return (xt, null);
            double[] rhosC = ord == 1 ? [0.5] : SolveSub(rMat, b, ord);
            var mt = await Model(xt, t);
            for (int k = 0; k < xs.Length; k++)
            {
                double corr = 0;
                for (int j = 0; j < d1s.Count; j++) corr += rhosC[j] * d1s[j][k];
                xt[k] = xt_[k] - at * bH * (corr + rhosC[^1] * (mt[k] - m0[k]));
            }
            return (xt, mt);
        }

        for (int step = 1; step < order; step++)
        {
            double t = ts[step];
            var (nx, mx) = await Update(x, step, t, true);
            x = nx;
            mx ??= await Model(x, t);
            times.Add(t);
            models.Add(mx);
        }
        for (int step = order; step <= steps; step++)
        {
            double t = ts[step];
            int so = Math.Min(order, steps + 1 - step);
            var (nx, mx) = await Update(x, so, t, step != steps);
            x = nx;
            times.Add(t);
            if (step < steps)
            {
                mx ??= await Model(x, t);
                models.Add(mx);
            }
            while (times.Count > order) times.RemoveAt(0);
            while (models.Count > order) models.RemoveAt(0);
        }
        return [.. x.Select(v => (float)(v / Alpha(ts[^1])))];
    }

    /// <summary>The solution of the leading n x n block of a against b's first n entries.</summary>
    static double[] SolveSub(double[,] a, double[] b, int n)
    {
        var flat = new double[n * n];
        for (int r = 0; r < n; r++) for (int c = 0; c < n; c++) flat[r * n + c] = a[r, c];
        return Solve(flat, b[..n]);
    }

    // ---- SA-Solver (Xue et al. 2023): stochastic Adams predictor and corrector in data prediction, lambda = -log sigma ----

    /// <summary>
    /// The weights of the estimates at <paramref name="lams"/> (Lagrange basis) in sigma_t (1 + tau^2) integral over [lambda_s, lambda_t] of
    /// e^(-tau^2 (lambda_t - lambda)) e^lambda L_j(lambda).
    /// </summary>
    static double[] SaWeights(double[] lams, double ls, double lt, double sigT, double tau2)
    {
        var w = new double[lams.Length];
        for (int j = 0; j < lams.Length; j++)
        {
            int jj = j;
            w[j] = sigT * (1 + tau2) * GaussLegendre(l =>
            {
                double p = 1;
                for (int k = 0; k < lams.Length; k++) if (k != jj) p *= (l - lams[k]) / (lams[jj] - lams[k]);
                return Math.Exp(-tau2 * (lt - l) + l) * p;
            }, ls, lt, 24);
        }
        return w;
    }

    /// <summary>
    /// SA-Solver: an order-p stochastic Adams predictor, then (except on the last step) an order-c corrector that includes the estimate at the
    /// predicted point; PECE evaluates the model again at the corrected point. Tau comes from the options (default: eta 1 between 20% and 80%
    /// of the run); a stochastic step draws one noise shared by predictor and corrector. The step to sigma 0 returns the last estimate.
    /// </summary>
    static async Task<float[]> SaSolver(Run run, float[] x0, double[] sigmas, bool pece)
    {
        var o = run.Options;
        int pOrder = o.PredictorOrder ?? 3, cOrder = o.CorrectorOrder ?? 4;
        bool simple2 = o.SimpleOrder2 ?? false;
        double sNoise = run.SNoise;
        var tauFunc = o.TauFunc ?? TauInterval(run.Sampling.PercentToSigma(0.2), run.Sampling.PercentToSigma(0.8), 1.0);
        var s = sigmas.Select(v => (double)(float)v).ToArray();
        static double Lam(double v) => -Math.Log(v);
        var x = x0.Select(v => (double)v).ToArray();
        async Task<double[]> Model(double[] xv, double sigma, bool report = true)
        {
            run.Check();
            var d = await run.Denoise([.. xv.Select(v => (float)v)], sigma);
            if (report) await run.Callback(d);
            return [.. d.Select(v => (double)v)];
        }
        var ds = new List<double[]> { await Model(x, s[0]) };
        var lams = new List<double> { Lam(s[0]) };
        for (int i = 0; i < s.Length - 1; i++)
        {
            run.Check();
            if (s[i + 1] == 0)
            {
                x = ds[^1];
                run.Step(i);
                break;
            }
            double ls = lams[^1], lt = Lam(s[i + 1]), h = lt - ls, tau = tauFunc(s[i + 1]), tau2 = tau * tau;
            double keep = s[i + 1] / s[i] * Math.Exp(-tau2 * h), noiseScale = s[i + 1] * Math.Sqrt(-Expm1D(-2 * tau2 * h)) * sNoise;
            float[]? z = tau != 0 ? run.Noise.Next() : null;
            double[] Combine(double[] w, IList<double[]> est, bool withNoise = true)
            {
                var r = new double[x.Length];
                for (int k = 0; k < r.Length; k++)
                {
                    double v = keep * x[k];
                    for (int j = 0; j < w.Length; j++) v += w[j] * est[j][k];
                    if (z is not null && withNoise) v += noiseScale * z[k];
                    r[k] = v;
                }
                return r;
            }
            // Ending at sigma 0, the orders drop to the steps left before the last (measured).
            int cap = s[^1] == 0 ? Math.Max(s.Length - 2 - i, 1) : int.MaxValue;
            int po = Math.Min(Math.Min(ds.Count, pOrder), cap);
            var pEst = Enumerable.Range(0, po).Select(j => ds[^(j + 1)]).ToList();
            var pLam = Enumerable.Range(0, po).Select(j => lams[^(j + 1)]).ToArray();
            bool corrects = i + 2 < s.Length && cOrder > 0;
            var xp = Combine(SaWeightsFor(pLam, ls, lt, s[i + 1], tau2, simple2), pEst);
            if (corrects)
            {
                var dn = await Model(xp, s[i + 1]);
                int co = Math.Min(Math.Min(ds.Count + 1, cOrder), cap);
                var cEst = new List<double[]> { dn };
                var cLam = new List<double> { lt };
                for (int j = 0; j < co - 1; j++) { cEst.Add(ds[^(j + 1)]); cLam.Add(lams[^(j + 1)]); }
                x = Combine(SaWeightsFor([.. cLam], ls, lt, s[i + 1], tau2, simple2), cEst);
                if (pece) dn = await Model(x, s[i + 1], report: false);
                ds.Add(dn);
            }
            else
            {
                x = xp;
                if (i + 2 < s.Length) ds.Add(await Model(x, s[i + 1]));
            }
            lams.Add(lt);
            run.Step(i);
        }
        return [.. x.Select(v => (float)v)];
    }

    /// <summary>
    /// The weights for a step; with simple_order_2, two-point steps use UniPC's second-order forms instead (measured): the corrector adds
    /// h/2 (D_t - D_s), the predictor h/2 (D_prev - D_s) / r with r = (lambda_prev - lambda_s) / h, h taken as (1 + tau^2) h.
    /// </summary>
    static double[] SaWeightsFor(double[] lams, double ls, double lt, double sigT, double tau2, bool simple2)
    {
        if (!simple2 || lams.Length != 2) return SaWeights(lams, ls, lt, sigT, tau2);
        double h = lt - ls, hEff = (1 + tau2) * h, total = -Expm1D(-hEff);
        if (lams[0] == lt) return [0.5 * hEff, total - 0.5 * hEff];
        double c = 0.5 * hEff / ((lams[1] - ls) / h);
        return [total - c, c];
    }

    // ---- DDPM (Ho et al. 2020): the posterior step in sigma form ----

    static async Task<float[]> Ddpm(Run run, float[] x, double[] s)
    {
        for (int i = 0; i < s.Length - 1; i++)
        {
            run.Check();
            var denoised = await run.Denoise(x, s[i]);
            await run.Callback(denoised);
            float sig = (float)s[i], next = (float)s[i + 1], keep = next * next / (sig * sig);
            var r = new float[x.Length];
            for (int k = 0; k < r.Length; k++) r[k] = denoised[k] + keep * (x[k] - denoised[k]);
            x = r;
            if (next > 0) AddNoise(x, run.Noise.Next(), next * MathF.Sqrt(1 - keep));
            run.Step(i);
        }
        return x;
    }

    // ---- LCM (Luo et al. 2023): the denoised estimate, re-noised to the next sigma ----

    static async Task<float[]> Lcm(Run run, float[] x, double[] s)
    {
        var o = run.Options;
        double start = o.SNoise ?? 1, end = o.SNoiseEnd ?? start, clip = o.NoiseClipStd ?? 0;
        int noised = Math.Max(s.Count(v => v > 0) - 1, 1);
        for (int i = 0; i < s.Length - 1; i++)
        {
            run.Check();
            var denoised = await run.Denoise(x, s[i]);
            await run.Callback(denoised);
            x = denoised;
            if (s[i + 1] > 0)
            {
                var z = run.Noise.Next();
                double scale = noised <= 1 ? start : start + (end - start) * i / (noised - 1);
                var n = new float[z.Length];
                float std = clip > 0 ? Std(z) : 0;
                for (int k = 0; k < n.Length; k++)
                {
                    float v = z[k];
                    if (clip > 0) v = Math.Clamp(v, (float)(-clip * std), (float)(clip * std));
                    n[k] = v * (float)scale;
                }
                x = run.Sampling.NoiseScaling(s[i + 1], n, x);
            }
            run.Step(i);
        }
        return x;
    }

    /// <summary>The sample standard deviation (torch.std's default, n - 1).</summary>
    static float Std(float[] v)
    {
        double mean = v.Average(a => (double)a), sum = 0;
        foreach (var a in v) sum += (a - mean) * (a - mean);
        return (float)Math.Sqrt(sum / Math.Max(v.Length - 1, 1));
    }

    // ---- iPNDM (Zhang & Chen 2022): Adams-Bashforth up to order 4 on the derivative, fixed or step-size-aware coefficients ----

    static async Task<float[]> Ipndm(Run run, float[] x, double[] s)
    {
        var ds = new List<float[]>();
        for (int i = 0; i < s.Length - 1; i++)
        {
            run.Check();
            var denoised = await run.Denoise(x, s[i]);
            await run.Callback(denoised);
            ds.Insert(0, ToD(x, s[i], denoised));
            if (ds.Count > 4) ds.RemoveAt(4);
            float dt = (float)s[i + 1] - (float)s[i];
            if (s[i + 1] == 0)
            {
                x = denoised;
                run.Step(i);
                continue;
            }
            float[] w = ds.Count switch
            {
                1 => [1f],
                2 => [3f / 2, -1f / 2],
                3 => [23f / 12, -16f / 12, 5f / 12],
                _ => [55f / 24, -59f / 24, 37f / 24, -9f / 24],
            };
            x = Multistep(x, ds, w, dt);
            run.Step(i);
        }
        return x;
    }

    /// <summary>x + dt sum w_j d_j.</summary>
    static float[] Multistep(float[] x, List<float[]> ds, float[] w, float dt)
    {
        var r = new float[x.Length];
        for (int k = 0; k < r.Length; k++)
        {
            float sum = 0;
            for (int j = 0; j < w.Length; j++) sum += w[j] * ds[j][k];
            r[k] = x[k] + dt * sum;
        }
        return r;
    }

    /// <summary>
    /// iPNDM with variable steps (Zhou et al. 2024): AB2 for the step sizes, then their third- and fourth-order corrections. The weights were
    /// measured from ComfyUI's sampler; the fourth-order term's oldest derivative carries an extra factor h1 h3 / h2^2 there, kept for parity.
    /// </summary>
    static async Task<float[]> IpndmV(Run run, float[] x, double[] s)
    {
        var ds = new List<float[]>();
        for (int i = 0; i < s.Length - 1; i++)
        {
            run.Check();
            var denoised = await run.Denoise(x, s[i]);
            await run.Callback(denoised);
            ds.Insert(0, ToD(x, s[i], denoised));
            if (ds.Count > 4) ds.RemoveAt(4);
            double h = (float)s[i + 1] - (float)s[i];
            if (s[i + 1] == 0)
            {
                x = denoised;
                run.Step(i);
                continue;
            }
            double[] w;
            if (ds.Count == 1) w = [1];
            else
            {
                double h1 = (float)s[i] - (float)s[i - 1];
                w = [1 + h / (2 * h1), -h / (2 * h1), 0, 0];
                if (ds.Count >= 3)
                {
                    double h2 = (float)s[i - 1] - (float)s[i - 2];
                    double t1 = (1 - h / (3 * (h + h1)) * (h * (h + h1)) / (h1 * (h1 + h2))) / 2;
                    // t1 times (d0 - d1) - (h1 / h2)(d1 - d2)
                    w[0] += t1; w[1] -= t1 * (1 + h1 / h2); w[2] += t1 * h1 / h2;
                    if (ds.Count == 4)
                    {
                        double h3 = (float)s[i - 2] - (float)s[i - 3];
                        double t2 = ((1 - h / (3 * (h + h1))) / 2 + (1 - h / (2 * (h + h1))) * h / (6 * (h + h1 + h2)))
                            * (h * (h + h1) * (h + h1 + h2)) / (h1 * (h1 + h2) * (h1 + h2 + h3));
                        // The third divided difference over the four nodes, scaled so d0's weight is 1.
                        double[] node = [0, -h1, -h1 - h2, -h1 - h2 - h3];
                        var dd = new double[4];
                        for (int j = 0; j < 4; j++)
                        {
                            double p = 1;
                            for (int k = 0; k < 4; k++) if (k != j) p *= node[j] - node[k];
                            dd[j] = 1 / p;
                        }
                        double norm = dd[0];
                        for (int j = 0; j < 4; j++) dd[j] /= norm;
                        dd[3] *= h1 * h3 / (h2 * h2);
                        for (int j = 0; j < 4; j++) w[j] += t2 * dd[j];
                    }
                }
            }
            var r = new float[x.Length];
            for (int k = 0; k < r.Length; k++)
            {
                double sum = 0;
                for (int j = 0; j < ds.Count; j++) sum += w[j] * ds[j][k];
                r[k] = x[k] + (float)(h * sum);
            }
            x = r;
            run.Step(i);
        }
        return x;
    }

    // ---- DEIS (Zhang & Chen 2022), time-AB form: Lagrange extrapolation of the noise in VP time, integrated over sigma ----

    // EDM's VP parametrization of sigma (Karras et al. 2022, appendix): sigma_min 0.002, sigma_max 80, epsilon_s 1e-3 fix beta_d and beta_min.
    static readonly double VpBetaD = 2 * (Math.Log(0.002 * 0.002 + 1) / 1e-3 - Math.Log(80.0 * 80 + 1)) / (1e-3 - 1);
    static readonly double VpBetaMin = Math.Log(80.0 * 80 + 1) - 0.5 * VpBetaD;

    /// <summary>VP time of a sigma: sigma^2 = exp(beta_d t^2 / 2 + beta_min t) - 1.</summary>
    static double VpTime(double sigma) =>
        (-VpBetaMin + Math.Sqrt(VpBetaMin * VpBetaMin + 2 * VpBetaD * Math.Log(sigma * sigma + 1))) / VpBetaD;

    /// <summary>d sigma / d t at VP time t.</summary>
    static double VpSigmaRate(double t)
    {
        double e = Math.Exp(0.5 * VpBetaD * t * t + VpBetaMin * t);
        return (VpBetaD * t + VpBetaMin) * e / (2 * Math.Sqrt(e - 1));
    }

    /// <summary>
    /// DEIS's coefficients: for each step, the weights of the last (up to <paramref name="maxOrder"/>) noise estimates, newest first: the Lagrange
    /// basis over the previous steps' VP times integrated over the step in sigma. The integral is the sum over <paramref name="n"/> evenly spaced
    /// times (both ends included) times (t_next - t) / n, as ComfyUI evaluates it (measured).
    /// </summary>
    public static double[][] DeisCoefficients(double[] sigmas, int maxOrder, int n = 10000)
    {
        var r = new double[sigmas.Length - 1][];
        for (int i = 0; i < r.Length; i++)
        {
            int order = Math.Min(i + 1, maxOrder);
            if (order == 1) { r[i] = [sigmas[i + 1] - sigmas[i]]; continue; }
            var nodes = Enumerable.Range(0, order).Select(k => VpTime(sigmas[i - k])).ToArray();
            double t0 = VpTime(sigmas[i]), t1 = VpTime(sigmas[i + 1]), dt = (t1 - t0) / n;
            r[i] = new double[order];
            for (int m = 0; m < n; m++)
            {
                double tau = t0 + (t1 - t0) * m / (n - 1), rate = VpSigmaRate(tau);
                for (int j = 0; j < order; j++)
                {
                    double p = 1;
                    for (int k = 0; k < order; k++) if (k != j) p *= (tau - nodes[k]) / (nodes[j] - nodes[k]);
                    r[i][j] += rate * p;
                }
            }
            for (int j = 0; j < order; j++) r[i][j] *= dt;
        }
        return r;
    }

    static async Task<float[]> Deis(Run run, float[] x, double[] s)
    {
        var coeffs = DeisCoefficients(s.Select(v => (double)(float)v).ToArray(), 3);
        var eps = new List<float[]>();
        for (int i = 0; i < s.Length - 1; i++)
        {
            run.Check();
            var denoised = await run.Denoise(x, s[i]);
            await run.Callback(denoised);
            eps.Insert(0, ToD(x, s[i], denoised));
            if (s[i + 1] == 0)
            {
                x = denoised;
                run.Step(i);
                continue;
            }
            var c = coeffs[i];
            var r = new float[x.Length];
            for (int k = 0; k < r.Length; k++)
            {
                double sum = 0;
                for (int j = 0; j < c.Length; j++) sum += c[j] * eps[j][k];
                r[k] = x[k] + (float)sum;
            }
            x = r;
            run.Step(i);
        }
        return x;
    }

    // ---- DPM-Solver (Lu et al. 2022) in k-diffusion's form: t = -log sigma ----

    /// <summary>dpm_fast and dpm_adaptive end at the last sigma above 0.</summary>
    static double DpmSigmaMin(double[] s) => s[^1] == 0 ? s[^2] : s[^1];

    internal sealed class DpmSolver(Run run)
    {
        public static double T(double sigma) => -MathF.Log((float)sigma);
        public static double Sigma(double t) => MathF.Exp(-(float)t);

        public float[]? LastDenoised;

        public async Task<float[]> Eps(Dictionary<string, float[]> cache, string key, float[] x, double t)
        {
            if (cache.TryGetValue(key, out var e)) return e;
            float sigma = (float)Sigma(t);
            var denoised = await run.Denoise(x, sigma);
            var eps = new float[x.Length];
            for (int i = 0; i < eps.Length; i++) eps[i] = (x[i] - denoised[i]) / sigma;
            cache[key] = eps;
            return eps;
        }

        static float F(double v) => (float)v;

        public async Task<float[]> Step1(float[] x, double t, double tNext, Dictionary<string, float[]> cache)
        {
            float h = F(tNext) - F(t);
            var eps = await Eps(cache, "eps", x, t);
            return LinCombine(x, 1, eps, -(float)Sigma(tNext) * ExpM1(h));
        }

        public async Task<float[]> Step2(float[] x, double t, double tNext, Dictionary<string, float[]> cache, double r1 = 0.5)
        {
            float h = F(tNext) - F(t), fr1 = (float)r1;
            var eps = await Eps(cache, "eps", x, t);
            float s1 = F(t) + fr1 * h;
            var u1 = LinCombine(x, 1, eps, -(float)Sigma(s1) * ExpM1(fr1 * h));
            var epsR1 = await Eps(cache, "eps_r1", u1, s1);
            float sn = (float)Sigma(tNext), e = ExpM1(h);
            var r = new float[x.Length];
            for (int i = 0; i < r.Length; i++) r[i] = x[i] - sn * e * eps[i] - sn / (2 * fr1) * e * (epsR1[i] - eps[i]);
            return r;
        }

        public async Task<float[]> Step3(float[] x, double t, double tNext, Dictionary<string, float[]> cache, double r1 = 1.0 / 3, double r2 = 2.0 / 3)
        {
            float h = F(tNext) - F(t), fr1 = (float)r1, fr2 = (float)r2;
            var eps = await Eps(cache, "eps", x, t);
            float s1 = F(t) + fr1 * h, s2 = F(t) + fr2 * h;
            var u1 = LinCombine(x, 1, eps, -(float)Sigma(s1) * ExpM1(fr1 * h));
            var epsR1 = await Eps(cache, "eps_r1", u1, s1);
            float sig2 = (float)Sigma(s2), e2 = ExpM1(fr2 * h), c2 = sig2 * (fr2 / fr1) * (e2 / (fr2 * h) - 1);
            var u2 = new float[x.Length];
            for (int i = 0; i < u2.Length; i++) u2[i] = x[i] - sig2 * e2 * eps[i] - c2 * (epsR1[i] - eps[i]);
            var epsR2 = await Eps(cache, "eps_r2", u2, s2);
            float sn = (float)Sigma(tNext), e = ExpM1(h), c3 = sn / fr2 * (e / h - 1);
            var r = new float[x.Length];
            for (int i = 0; i < r.Length; i++) r[i] = x[i] - sn * e * eps[i] - c3 * (epsR2[i] - eps[i]);
            return r;
        }
    }

    static async Task<float[]> DpmFast(Run run, float[] x, double[] s)
    {
        double sigmaMin = DpmSigmaMin(s), sigmaMax = s[0];
        int n = s.Length - 1;
        double eta = run.Options.Eta ?? 0;
        var solver = new DpmSolver(run);
        float tStart = (float)DpmSolver.T(sigmaMax), tEnd = (float)DpmSolver.T(sigmaMin);
        int m = n / 3 + 1;
        var ts = TorchLinspace(tStart, tEnd, m + 1);
        int[] orders = n % 3 == 0 ? [.. Enumerable.Repeat(3, m - 2), 2, 1] : [.. Enumerable.Repeat(3, m - 1), n % 3];
        for (int i = 0; i < orders.Length; i++)
        {
            run.Check();
            var cache = new Dictionary<string, float[]>();
            float t = ts[i], tNext = ts[i + 1], tNext_ = tNext, su = 0;
            if (eta != 0)
            {
                var (sd, up) = AncestralStep(DpmSolver.Sigma(t), DpmSolver.Sigma(tNext), eta);
                tNext_ = MathF.Min(tEnd, (float)DpmSolver.T(sd));
                float a = (float)DpmSolver.Sigma(tNext), b = (float)DpmSolver.Sigma(tNext_);
                su = MathF.Sqrt(a * a - b * b);
            }
            var eps = await solver.Eps(cache, "eps", x, t);
            float sig = (float)DpmSolver.Sigma(t);
            await run.Callback(LinCombine(x, 1, eps, -sig));
            x = orders[i] switch
            {
                1 => await solver.Step1(x, t, tNext_, cache),
                2 => await solver.Step2(x, t, tNext_, cache),
                _ => await solver.Step3(x, t, tNext_, cache),
            };
            if (eta != 0) AddNoise(x, run.Noise.Next(), su * run.SNoise);
            run.Step(i);
        }
        return x;
    }

    /// <summary>torch.linspace in float32: from the start for the first half, from the end for the second.</summary>
    internal static float[] TorchLinspace(float start, float end, int steps)
    {
        var r = new float[steps];
        if (steps == 1) { r[0] = start; return r; }
        float step = (end - start) / (steps - 1);
        int half = steps / 2;
        for (int i = 0; i < steps; i++) r[i] = i < half ? start + i * step : end - (steps - 1 - i) * step;
        return r;
    }

    /// <summary>k-diffusion's PID step size controller for the adaptive solver.</summary>
    internal sealed class PidStepSizeController(double h, double pcoeff, double icoeff, double dcoeff, double order, double acceptSafety, double eps = 1e-8)
    {
        readonly double _b1 = (pcoeff + icoeff + dcoeff) / order, _b2 = -(pcoeff + 2 * dcoeff) / order, _b3 = dcoeff / order;
        double[]? _errs;

        public double H { get; private set; } = h;

        public bool ProposeStep(double error)
        {
            double inv = 1 / (error + eps);
            _errs ??= [inv, inv, inv];
            _errs[0] = inv;
            double factor = Math.Pow(_errs[0], _b1) * Math.Pow(_errs[1], _b2) * Math.Pow(_errs[2], _b3);
            factor = 1 + Math.Atan(factor - 1);
            bool accept = factor >= acceptSafety;
            if (accept) { _errs[2] = _errs[1]; _errs[1] = _errs[0]; }
            H *= factor;
            return accept;
        }
    }

    static async Task<float[]> DpmAdaptive(Run run, float[] x, double[] s)
    {
        var o = run.Options;
        int order = o.Order ?? 3;
        if (order is not (2 or 3)) throw new ArgumentException("dpm_adaptive's order must be 2 or 3.");
        double eta = o.Eta ?? 0, sNoise = run.SNoise;
        float tStart = (float)DpmSolver.T(s[0]), tEnd = (float)DpmSolver.T(DpmSigmaMin(s));
        bool forward = tEnd > tStart;
        double hInit = Math.Abs(o.HInit ?? 0.05) * (forward ? 1 : -1);
        float atol = (float)(o.Atol ?? 0.0078), rtol = (float)(o.Rtol ?? 0.05);
        var pid = new PidStepSizeController(hInit, o.PCoeff ?? 0, o.ICoeff ?? 1, o.DCoeff ?? 0, eta != 0 ? 1.5 : order, o.AcceptSafety ?? 0.81);
        var solver = new DpmSolver(run);
        float sT = tStart;
        var xPrev = x;
        int steps = 0;
        while (forward ? sT < tEnd - 1e-5f : sT > tEnd + 1e-5f)
        {
            run.Check();
            var cache = new Dictionary<string, float[]>();
            float t = forward ? MathF.Min(tEnd, sT + (float)pid.H) : MathF.Max(tEnd, sT + (float)pid.H), t_ = t, su = 0;
            if (eta != 0)
            {
                var (sd, up) = AncestralStep(DpmSolver.Sigma(sT), DpmSolver.Sigma(t), eta);
                t_ = MathF.Min(tEnd, (float)DpmSolver.T(sd));
                float a = (float)DpmSolver.Sigma(t), b = (float)DpmSolver.Sigma(t_);
                su = MathF.Sqrt(a * a - b * b);
            }
            var eps = await solver.Eps(cache, "eps", x, sT);
            var denoised = LinCombine(x, 1, eps, -(float)DpmSolver.Sigma(sT));
            float[] low, high;
            if (order == 2)
            {
                low = await solver.Step1(x, sT, t_, cache);
                high = await solver.Step2(x, sT, t_, cache);
            }
            else
            {
                low = await solver.Step2(x, sT, t_, cache, r1: 1.0 / 3);
                high = await solver.Step3(x, sT, t_, cache);
            }
            double sum = 0;
            for (int i = 0; i < x.Length; i++)
            {
                float delta = MathF.Max(atol, rtol * MathF.Max(MathF.Abs(low[i]), MathF.Abs(xPrev[i])));
                float q = (low[i] - high[i]) / delta;
                sum += (double)q * q;
            }
            float error = (float)Math.Sqrt(sum) / MathF.Sqrt(x.Length);
            bool accept = pid.ProposeStep(error);
            if (accept)
            {
                xPrev = low;
                x = high;
                AddNoise(x, run.Noise.Next(), su * sNoise);
                sT = t;
            }
            await run.Callback(denoised);
            run.Step(steps++);
        }
        return x;
    }

    // ---- Heun++ (Carzit, MIT): Heun with a third evaluation where two more sigmas remain ----

    static async Task<float[]> Heunpp2(Run run, float[] x, double[] s)
    {
        double end = s[^1];
        for (int i = 0; i < s.Length - 1; i++)
        {
            run.Check();
            var denoised = await run.Denoise(x, s[i]);
            var d = ToD(x, s[i], denoised);
            await run.Callback(denoised);
            float dt = (float)s[i + 1] - (float)s[i];
            if (s[i + 1] == end) x = Step(x, d, dt);
            else if (s[i + 2] == end)
            {
                var x2 = Step(x, d, dt);
                var d2 = ToD(x2, s[i + 1], await run.Denoise(x2, s[i + 1]));
                float w = 2 * (float)s[0], w2 = (float)s[i + 1] / w, w1 = 1 - w2;
                var dp = new float[d.Length];
                for (int k = 0; k < dp.Length; k++) dp[k] = d[k] * w1 + d2[k] * w2;
                x = Step(x, dp, dt);
            }
            else
            {
                var x2 = Step(x, d, dt);
                var d2 = ToD(x2, s[i + 1], await run.Denoise(x2, s[i + 1]));
                var x3 = Step(x2, d2, (float)s[i + 2] - (float)s[i + 1]);
                var d3 = ToD(x3, s[i + 2], await run.Denoise(x3, s[i + 2]));
                float w = 3 * (float)s[0], w2 = (float)s[i + 1] / w, w3 = (float)s[i + 2] / w, w1 = 1 - w2 - w3;
                var dp = new float[d.Length];
                for (int k = 0; k < dp.Length; k++) dp[k] = w1 * d[k] + w2 * d2[k] + w3 * d3[k];
                x = Step(x, dp, dt);
            }
            run.Step(i);
        }
        return x;
    }

    // ---- CFG++ (Chung et al. 2024) helpers and the flow (rectified-flow) ancestral steps: fitted below ----


    static Task<float[]> FlowAncestralStep(Run run, float[] x, float[] denoised, double sigma, double sigmaNext, double eta, float[]? uncond) =>
        throw new NotImplementedException("flow ancestral");

    static Task<float[]> FlowDpm2AncestralStep(Run run, float[] x, float[] denoised, double sigma, double sigmaNext, double eta) =>
        throw new NotImplementedException("flow dpm_2_ancestral");

    static Task<float[]> FlowDpmpp2SAncestralStep(Run run, float[] x, float[] denoised, double sigma, double sigmaNext, double eta) =>
        throw new NotImplementedException("flow dpmpp_2s_ancestral");

    static Task<float[]> FlowDpmppSdeStep(Run run, float[] x, float[] denoised, double sigma, double sigmaNext, double eta, double r, BrownianNoise brownian) =>
        throw new NotImplementedException("flow dpmpp_sde");

    static Task<(float[], float)> FlowDpmpp2MSdeStep(Run run, float[] x, float[] denoised, float[]? old, float hLast, double sigma, double sigmaNext, double eta,
        string solverType) => throw new NotImplementedException("flow dpmpp_2m_sde");

    /// <summary>SA-Solver's stochasticity: eta between the two sigmas (inclusive), 0 elsewhere.</summary>
    public static Func<double, double> TauInterval(double startSigma, double endSigma, double eta = 1.0) =>
        sigma => sigma <= startSigma && sigma >= endSigma ? eta : 0;

    /// <summary>The solution of a x = b for a square a (row-major), by Gaussian elimination with partial pivoting.</summary>
    public static double[] Solve(double[] a, double[] b)
    {
        int n = b.Length;
        var m = (double[])a.Clone();
        var v = (double[])b.Clone();
        for (int c = 0; c < n; c++)
        {
            int p = c;
            for (int r = c + 1; r < n; r++) if (Math.Abs(m[r * n + c]) > Math.Abs(m[p * n + c])) p = r;
            if (p != c)
            {
                for (int k = 0; k < n; k++) (m[c * n + k], m[p * n + k]) = (m[p * n + k], m[c * n + k]);
                (v[c], v[p]) = (v[p], v[c]);
            }
            for (int r = c + 1; r < n; r++)
            {
                double f = m[r * n + c] / m[c * n + c];
                for (int k = c; k < n; k++) m[r * n + k] -= f * m[c * n + k];
                v[r] -= f * v[c];
            }
        }
        var xs = new double[n];
        for (int r = n - 1; r >= 0; r--)
        {
            double sum = v[r];
            for (int k = r + 1; k < n; k++) sum -= m[r * n + k] * xs[k];
            xs[r] = sum / m[r * n + r];
        }
        return xs;
    }
}
