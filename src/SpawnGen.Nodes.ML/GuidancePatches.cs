using System.Numerics;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// The maths of ComfyUI's guidance patches (comfy_extras nodes_cfg, nodes_apg, nodes_mahiro, nodes_tcfg, nodes_perpneg, nodes_fresca, nodes_eps, nodes_lumina2,
/// nodes_model_advanced), over one picture's channel-first latent. Reductions ComfyUI takes over a whole batch (Mahiro's mean, Perp-Neg's sums) are taken over the one picture.
/// </summary>
public static class GuidancePatches
{
    static double Dot(float[] a, float[] b) { double s = 0; for (int i = 0; i < a.Length; i++) s += (double)a[i] * b[i]; return s; }

    /// <summary>Sample standard deviation (torch.std, n - 1).</summary>
    static double Std(double[] v)
    {
        double mean = v.Average(), sum = 0;
        foreach (var x in v) sum += (x - mean) * (x - mean);
        return Math.Sqrt(sum / Math.Max(1, v.Length - 1));
    }

    static double Std(float[] v) => Std(v.Select(x => (double)x).ToArray());

    /// <summary>The L2 norm over the channels at each pixel (torch.norm(dim=1)).</summary>
    static double[] ChannelNorms(float[] v, int channels, int plane)
    {
        var r = new double[plane];
        for (int c = 0; c < channels; c++)
            for (int i = 0; i < plane; i++) r[i] += (double)v[c * plane + i] * v[c * plane + i];
        for (int i = 0; i < plane; i++) r[i] = Math.Sqrt(r[i]);
        return r;
    }

    /// <summary>
    /// RescaleCFG's cfg function (arXiv 2305.08891): CFG in v-space, rescaled to the spread of the prompt's own prediction and blended back by
    /// <paramref name="multiplier"/>. Flow models are rescaled directly in x0 space. Returns the cfg-function value (the guided estimate is x minus it).
    /// </summary>
    public static float[] RescaleCfg(CfgFunctionArgs a, double multiplier, bool isFlow)
    {
        int n = a.Input.Length;
        double s = a.CondScale;
        if (isFlow)
        {
            var cfg = new double[n];
            for (int i = 0; i < n; i++) cfg[i] = a.UncondDenoised[i] + s * (a.CondDenoised[i] - a.UncondDenoised[i]);
            double roPos = Std(a.CondDenoised), roCfg = Math.Max(Std(cfg), 1e-8);
            var r = new float[n];
            for (int i = 0; i < n; i++)
            {
                double fin = multiplier * (cfg[i] * (roPos / roCfg)) + (1.0 - multiplier) * cfg[i];
                r[i] = (float)(a.Input[i] - fin);
            }
            return r;
        }
        double sigma = a.Sigma, k = Math.Sqrt(sigma * sigma + 1.0);
        var x = new double[n];
        var cond = new double[n];
        var xCfg = new double[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = a.Input[i] / (sigma * sigma + 1.0);
            cond[i] = (x[i] - (a.Input[i] - a.Cond[i])) * k / sigma;
            double uncond = (x[i] - (a.Input[i] - a.Uncond[i])) * k / sigma;
            xCfg[i] = uncond + s * (cond[i] - uncond);
        }
        double ratio = Std(cond) / Std(xCfg);
        var result = new float[n];
        for (int i = 0; i < n; i++)
        {
            double fin = multiplier * (xCfg[i] * ratio) + (1.0 - multiplier) * xCfg[i];
            result[i] = (float)(a.Input[i] - (x[i] - fin * sigma / k));
        }
        return result;
    }

    /// <summary>CFGNorm (post-cfg): scale the guided estimate per pixel so its channel norm does not exceed the prompt's own, times <paramref name="strength"/>.</summary>
    public static float[] CfgNormPost(PostCfgArgs a, double strength)
    {
        int c = a.Context.Channels, plane = a.Context.Height * a.Context.Width;
        var condNorm = ChannelNorms(a.CondDenoised, c, plane);
        var predNorm = ChannelNorms(a.Denoised, c, plane);
        var r = new float[a.Denoised.Length];
        for (int i = 0; i < r.Length; i++)
        {
            int p = i % plane;
            double scale = Math.Clamp(condNorm[p] / (predNorm[p] + 1e-8), 0.0, 1.0);
            r[i] = (float)(a.Denoised[i] * scale * strength);
        }
        return r;
    }

    /// <summary>CFGNorm with pre_cfg (a cfg function): the combined noise is rescaled per pixel to the prompt noise's channel norm, without clamping.</summary>
    public static float[] CfgNormPre(CfgFunctionArgs a, double strength)
    {
        int c = a.Context.Channels, plane = a.Context.Height * a.Context.Width, n = a.Input.Length;
        var comb = new float[n];
        for (int i = 0; i < n; i++) comb[i] = (float)(a.Uncond[i] + a.CondScale * (a.Cond[i] - a.Uncond[i]));
        var condNorm = ChannelNorms(a.Cond, c, plane);
        var combNorm = ChannelNorms(comb, c, plane);
        var r = new float[n];
        for (int i = 0; i < n; i++)
        {
            int p = i % plane;
            double rescale = combNorm[p] > 0 ? condNorm[p] / Math.Max(combNorm[p], 1e-12) : 1.0;
            double v = comb[i] * rescale;
            if (strength != 1.0) v = strength * v + (1.0 - strength) * comb[i];
            r[i] = (float)v;
        }
        return r;
    }

    /// <summary>CFG-Zero* (post-cfg): the unconditional estimate is scaled by the projection of the prompt's noise on the unconditional noise.</summary>
    public static float[] CfgZeroStar(PostCfgArgs a)
    {
        int n = a.Input.Length;
        var pos = Samplers.Subtract(a.Input, a.CondDenoised);
        var neg = Samplers.Subtract(a.Input, a.UncondDenoised);
        double alpha = Dot(pos, neg) / (Dot(neg, neg) + 1e-8);
        var r = new float[n];
        for (int i = 0; i < n; i++)
            r[i] = (float)(a.Denoised[i] + a.UncondDenoised[i] * (alpha - 1.0) + a.CondScale * a.UncondDenoised[i] * (1.0 - alpha));
        return r;
    }

    /// <summary>Adaptive Projected Guidance (pre-cfg, a fresh running average per run): the guidance is clipped in norm and its part along the prompt scaled by eta.</summary>
    public static Func<PreCfgArgs, float[][]> Apg(double eta, double normThreshold, double momentum)
    {
        float[]? runningAvg = null;
        double? prevSigma = null;
        return a =>
        {
            if (a.CondsOut.Length == 1) return a.CondsOut;
            var cond = a.CondsOut[0];
            var uncond = a.CondsOut[1];
            double sigma = a.Sigma;
            if (prevSigma is not null && sigma > prevSigma) runningAvg = null;
            prevSigma = sigma;
            var guidance = Samplers.Subtract(cond, uncond);
            if (momentum != 0)
            {
                if (runningAvg is null) runningAvg = guidance;
                else
                {
                    var avg = new float[guidance.Length];
                    for (int i = 0; i < avg.Length; i++) avg[i] = (float)(momentum * runningAvg[i] + guidance[i]);
                    runningAvg = avg;
                }
                guidance = runningAvg;
            }
            if (normThreshold > 0)
            {
                double norm = Math.Sqrt(Dot(guidance, guidance)), scale = Math.Min(1.0, normThreshold / norm);
                guidance = guidance.Select(v => (float)(v * scale)).ToArray();
            }
            double condNorm = Math.Max(Math.Sqrt(Dot(cond, cond)), 1e-12);
            var v1 = cond.Select(v => (float)(v / condNorm)).ToArray();
            double along = Dot(guidance, v1);
            var modifiedCond = new float[cond.Length];
            for (int i = 0; i < cond.Length; i++)
            {
                double parallel = along * v1[i], orthogonal = guidance[i] - parallel;
                double modified = orthogonal + eta * parallel;
                modifiedCond[i] = (float)((uncond[i] + modified) + (cond[i] - uncond[i]) / a.CondScale);
            }
            return [modifiedCond, uncond, .. a.CondsOut.Skip(2)];
        };
    }

    /// <summary>Mahiro, positive-biased guidance (post-cfg): blend the guided estimate towards cond * scale by how similar the (square-rooted) leaps are.</summary>
    public static float[] Mahiro(PostCfgArgs a)
    {
        int c = a.Context.Channels, plane = a.Context.Height * a.Context.Width, n = a.Denoised.Length;
        double scale = a.CondScale;
        var leap = new double[n];
        var normu = new double[n];
        var normm = new double[n];
        for (int i = 0; i < n; i++)
        {
            leap[i] = a.CondDenoised[i] * scale;
            double uLeap = a.UncondDenoised[i] * scale, merge = (leap[i] + a.Denoised[i]) / 2;
            normu[i] = Math.Sqrt(Math.Abs(uLeap)) * Math.Sign(uLeap);
            normm[i] = Math.Sqrt(Math.Abs(merge)) * Math.Sign(merge);
        }
        // F.cosine_similarity over the channels (each side divided by its norm clamped to 1e-8), then the mean over the pixels.
        double simSum = 0;
        for (int p = 0; p < plane; p++)
        {
            double nu = 0, nm = 0, dot = 0;
            for (int ch = 0; ch < c; ch++)
            {
                double u = normu[ch * plane + p], m = normm[ch * plane + p];
                nu += u * u; nm += m * m; dot += u * m;
            }
            simSum += dot / (Math.Max(Math.Sqrt(nu), 1e-8) * Math.Max(Math.Sqrt(nm), 1e-8));
        }
        double sim = simSum / plane, simsc = 2 * (sim + 1);
        var r = new float[n];
        for (int i = 0; i < n; i++) r[i] = (float)((simsc * a.Denoised[i] + (4 - simsc) * leap[i]) / 4);
        return r;
    }

    /// <summary>
    /// TCFG's tangential damping: the unconditional score keeps only its component along the first right singular vector of [uncond; cond] (the dominant shared direction).
    /// </summary>
    public static float[] TangentialDamping(float[] condScore, float[] uncondScore)
    {
        double uu = Dot(uncondScore, uncondScore), uc = Dot(uncondScore, condScore), cc = Dot(condScore, condScore);
        // The largest eigenpair of the 2 x 2 Gram matrix [[uu, uc], [uc, cc]] gives v1 = (e0 * u + e1 * c) / sqrt(lambda1).
        double mean = (uu + cc) / 2, diff = (uu - cc) / 2, lambda1 = mean + Math.Sqrt(diff * diff + uc * uc);
        if (lambda1 <= 0) return new float[uncondScore.Length];
        double e0, e1;
        if (Math.Abs(lambda1 - cc) >= Math.Abs(lambda1 - uu)) (e0, e1) = (lambda1 - cc, uc);
        else (e0, e1) = (uc, lambda1 - uu);
        double en = Math.Sqrt(e0 * e0 + e1 * e1);
        if (en == 0) (e0, e1, en) = (1, 0, 1);
        e0 /= en; e1 /= en;
        double inv = 1 / Math.Sqrt(lambda1);
        var v1 = new double[uncondScore.Length];
        double along = 0;
        for (int i = 0; i < v1.Length; i++)
        {
            v1[i] = (e0 * uncondScore[i] + e1 * condScore[i]) * inv;
            along += uncondScore[i] * v1[i];
        }
        return v1.Select(v => (float)(along * v)).ToArray();
    }

    /// <summary>TCFG (pre-cfg): the unconditional estimate is replaced by its tangentially damped form; skipped when either conditioning is missing.</summary>
    public static float[][] Tcfg(PreCfgArgs a)
    {
        if (a.CondsOut.Length <= 1 || a.CondsPresent.Take(2).Contains(false)) return a.CondsOut;
        var x = a.Input;
        var td = TangentialDamping(Samplers.Subtract(x, a.CondsOut[0]), Samplers.Subtract(x, a.CondsOut[1]));
        return [a.CondsOut[0], Samplers.Subtract(x, td), .. a.CondsOut.Skip(2)];
    }

    /// <summary>Perp-Neg: nocond + cfg * (pos - neg_scale * the part of neg perpendicular to pos), all relative to the empty prompt's estimate.</summary>
    public static float[] PerpNeg(float[] pos, float[] neg, float[] nocond, double negScale, double condScale)
    {
        var p = Samplers.Subtract(pos, nocond);
        var q = Samplers.Subtract(neg, nocond);
        double coef = Dot(q, p) / Dot(p, p);
        var r = new float[pos.Length];
        for (int i = 0; i < r.Length; i++)
        {
            double perp = q[i] - coef * p[i];
            r[i] = (float)(nocond[i] + condScale * (p[i] - perp * negScale));
        }
        return r;
    }

    /// <summary>
    /// FreSca's Fourier filter: per channel, a 2D FFT with the low frequencies (within <paramref name="cutoff"/> of the centre after fftshift) scaled by
    /// <paramref name="scaleLow"/> and the rest by <paramref name="scaleHigh"/>, back to space (real part).
    /// </summary>
    public static float[] FourierFilter(float[] x, int channels, int height, int width, double scaleLow, double scaleHigh, int cutoff)
    {
        var r = new float[x.Length];
        int plane = height * width, ch = height / 2, cw = width / 2, fh = Math.Min(cutoff, ch), fw = Math.Min(cutoff, cw);
        var dftH = DftMatrix(height, -1);
        var dftW = DftMatrix(width, -1);
        var idftH = DftMatrix(height, 1);
        var idftW = DftMatrix(width, 1);
        for (int c = 0; c < channels; c++)
        {
            var g = new Complex[height, width];
            for (int y = 0; y < height; y++)
                for (int xx = 0; xx < width; xx++) g[y, xx] = x[c * plane + y * width + xx];
            var f = Transform(g, dftH, dftW);
            for (int ky = 0; ky < height; ky++)
                for (int kx = 0; kx < width; kx++)
                {
                    // Position after fftshift: index k moves to (k + n / 2) mod n.
                    int sy = (ky + ch) % height, sx = (kx + cw) % width;
                    bool low = sy >= ch - fh && sy < ch + fh && sx >= cw - fw && sx < cw + fw;
                    f[ky, kx] *= low ? scaleLow : scaleHigh;
                }
            var back = Transform(f, idftH, idftW);
            for (int y = 0; y < height; y++)
                for (int xx = 0; xx < width; xx++) r[c * plane + y * width + xx] = (float)(back[y, xx].Real / plane);
        }
        return r;
    }

    static Complex[,] DftMatrix(int n, int sign)
    {
        var m = new Complex[n, n];
        for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++) m[k, j] = Complex.FromPolarCoordinates(1, sign * 2 * Math.PI * ((long)k * j % n) / n);
        return m;
    }

    /// <summary>Rows then columns through the given DFT matrices.</summary>
    static Complex[,] Transform(Complex[,] g, Complex[,] mh, Complex[,] mw)
    {
        int h = g.GetLength(0), w = g.GetLength(1);
        var rows = new Complex[h, w];
        for (int y = 0; y < h; y++)
            for (int k = 0; k < w; k++)
            {
                Complex s = 0;
                for (int j = 0; j < w; j++) s += mw[k, j] * g[y, j];
                rows[y, k] = s;
            }
        var r = new Complex[h, w];
        for (int k = 0; k < h; k++)
            for (int xx = 0; xx < w; xx++)
            {
                Complex s = 0;
                for (int j = 0; j < h; j++) s += mh[k, j] * rows[j, xx];
                r[k, xx] = s;
            }
        return r;
    }

    /// <summary>FreSca (pre-cfg): the guidance cond - uncond is Fourier-filtered; skipped when either conditioning is missing.</summary>
    public static float[][] FreSca(PreCfgArgs a, double scaleLow, double scaleHigh, int cutoff)
    {
        if (a.CondsOut.Length <= 1 || a.CondsPresent.Take(2).Contains(false)) return a.CondsOut;
        var cond = a.CondsOut[0];
        var uncond = a.CondsOut[1];
        var filtered = FourierFilter(Samplers.Subtract(cond, uncond), a.Context.Channels, a.Context.Height, a.Context.Width, scaleLow, scaleHigh, cutoff);
        var filteredCond = new float[cond.Length];
        for (int i = 0; i < cond.Length; i++) filteredCond[i] = filtered[i] + uncond[i];
        return [filteredCond, uncond, .. a.CondsOut.Skip(2)];
    }

    /// <summary>Epsilon Scaling (post-cfg): the predicted noise x - denoised is divided by the factor.</summary>
    public static float[] EpsilonScaling(PostCfgArgs a, double factor)
    {
        if (factor == 0) factor = 1e-9;
        var r = new float[a.Denoised.Length];
        for (int i = 0; i < r.Length; i++) r[i] = (float)(a.Input[i] - (a.Input[i] - a.Denoised[i]) / factor);
        return r;
    }

    /// <summary>Temporal Score Rescaling (post-cfg, arXiv 2510.01184): lerp from x / alpha to the estimate by r = (snr * var + 1) / (snr * var / k + 1).</summary>
    public static float[] TemporalScoreRescaling(PostCfgArgs a, double k, double variance)
    {
        double sigma = a.Sigma;
        if (k == 1 || sigma == 0) return a.Denoised;
        var sampling = a.Context.Sampling;
        double halfLogSnr = sampling.IsConst ? Math.Log((1 - sigma) / sigma) : -Math.Log(sigma);
        double snr = Math.Exp(2 * halfLogSnr);
        if (snr == 0) return a.Denoised;
        double rr = double.IsPositiveInfinity(snr) ? k : (snr * variance + 1) / (snr * variance / k + 1);
        double alpha = sigma * Math.Exp(halfLogSnr);
        var r = new float[a.Denoised.Length];
        for (int i = 0; i < r.Length; i++)
        {
            double start = a.Input[i] / alpha;
            r[i] = (float)(start + rr * (a.Denoised[i] - start));
        }
        return r;
    }

    /// <summary>RenormCFG's cfg function (Lumina 2): below sigma cfg_trunc plain CFG whose norm is capped at renorm_cfg times the prompt's, above it the prompt alone.</summary>
    public static float[] RenormCfg(CfgFunctionArgs a, double cfgTrunc, double renormCfg)
    {
        int n = a.Input.Length;
        var half = new double[n];
        if (a.Sigma < cfgTrunc)
        {
            for (int i = 0; i < n; i++) half[i] = a.UncondDenoised[i] + a.CondScale * (a.CondDenoised[i] - a.UncondDenoised[i]);
            if (renormCfg > 0.0)
            {
                double oriPos = Math.Sqrt(Dot(a.CondDenoised, a.CondDenoised)), maxNew = oriPos * renormCfg;
                double newPos = Math.Sqrt(half.Sum(v => v * v));
                if (newPos >= maxNew)
                    for (int i = 0; i < n; i++) half[i] *= maxNew / newPos;
            }
        }
        else for (int i = 0; i < n; i++) half[i] = a.CondDenoised[i];
        var r = new float[n];
        for (int i = 0; i < n; i++) r[i] = (float)(a.Input[i] - (float)half[i]);
        return r;
    }

    /// <summary>
    /// Differential Diffusion's denoise mask: at each step only the pixels whose mask value is at least how far the run still has to go (in timesteps) are
    /// regenerated, so a soft mask becomes a per-pixel denoise strength. Below strength 1 the binary mask is blended with the soft one.
    /// </summary>
    public static float[] DifferentialDiffusionMask(double sigma, float[] mask, double[] stepSigmas, ModelSampling sampling, double strength)
    {
        double sigmaTo = sampling.SigmaMin;
        if (stepSigmas[^1] > sigmaTo) sigmaTo = stepSigmas[^1];
        double sigmaFrom = stepSigmas[0];
        double tsFrom = sampling.Timestep(sigmaFrom), tsTo = sampling.Timestep(sigmaTo), current = sampling.Timestep(sigma);
        double threshold = (current - tsTo) / (tsFrom - tsTo);
        var r = new float[mask.Length];
        bool blend = strength != 0 && strength < 1;
        for (int i = 0; i < r.Length; i++)
        {
            float binary = mask[i] >= threshold ? 1f : 0f;
            r[i] = blend ? (float)(strength * binary + (1 - strength) * mask[i]) : binary;
        }
        return r;
    }

    /// <summary>
    /// VideoLinearCFGGuidance's cfg function: the batch (a video's frames) gets cfg rising from <paramref name="minCfg"/> on the first frame to the sampler's cfg on the
    /// last, <c>uncond + scale * (cond - uncond)</c>, all in float32 like torch.
    /// </summary>
    public static float[] VideoLinearCfg(CfgFunctionArgs a, double minCfg) =>
        ScaledCfg(a, Linspace((float)minCfg, (float)a.CondScale, a.Context.BatchSize)[a.Context.BatchIndex]);

    /// <summary>
    /// VideoTriangleCFGGuidance's cfg function: cfg goes from <paramref name="minCfg"/> at both ends of the batch up to the sampler's cfg in its middle
    /// (a triangle wave over linspace(0, 1) with period 1).
    /// </summary>
    public static float[] VideoTriangleCfg(CfgFunctionArgs a, double minCfg)
    {
        float v = Linspace(0f, 1f, a.Context.BatchSize)[a.Context.BatchIndex];
        v = 2f * MathF.Abs(v / 1f - MathF.Floor(v / 1f + 0.5f));
        float scale = v * (float)(a.CondScale - minCfg) + (float)minCfg;
        return ScaledCfg(a, scale);
    }

    static float[] ScaledCfg(CfgFunctionArgs a, float scale)
    {
        var r = new float[a.Cond.Length];
        for (int i = 0; i < r.Length; i++) r[i] = a.Uncond[i] + scale * (a.Cond[i] - a.Uncond[i]);
        return r;
    }

    /// <summary>torch.linspace in float32 (its CPU kernel: the first half counts up from the start, the second half down from the end).</summary>
    internal static float[] Linspace(float start, float end, int steps)
    {
        var r = new float[steps];
        if (steps == 1) { r[0] = start; return r; }
        float step = (end - start) / (steps - 1);
        int halfway = steps / 2;
        for (int i = 0; i < steps; i++) r[i] = i < halfway ? start + step * i : end - step * (steps - i - 1);
        return r;
    }
}
