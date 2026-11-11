namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md). Each patch follows its paper; where ComfyUI's node made its own choices (which space, which norm, which
// constants), they were measured by calling the nodes as black boxes on random tensors and checked against Fixtures/sampling/reference.json:
// RescaleCFG (Lin et al. 2023), CFG-Zero* (Fan et al. 2025), CFGNorm, APG (Sadat et al. 2024), Mahiro, TCFG (Kwon et al. 2025), Perp-Neg
// (Armandpour et al. 2023), FreSca (Huang et al. 2025), Epsilon Scaling (Ning et al. 2023), TSR (temporal score rescaling), RenormCFG
// (Lumina-Image 2.0), Differential Diffusion (Levin & Fried 2023) and the video cfg ramps.

/// <summary>The guidance patches' maths. Arrays are one picture, channels x height x width.</summary>
public static class GuidancePatches
{
    static double Dot(float[] a, float[] b)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++) s += (double)a[i] * b[i];
        return s;
    }

    static double Norm(float[] a) => Math.Sqrt(Dot(a, a));

    static double SampleStd(float[] v)
    {
        double mean = 0;
        foreach (var a in v) mean += a;
        mean /= v.Length;
        double s = 0;
        foreach (var a in v) s += (a - mean) * (a - mean);
        return Math.Sqrt(s / Math.Max(v.Length - 1, 1));
    }

    /// <summary>
    /// RescaleCFG: classifier-free guidance on v-predictions, its spread rescaled to the prompt's own (std ratio) by <paramref name="multiplier"/>.
    /// Returns the cfg function's value (input - denoised).
    /// </summary>
    public static float[] RescaleCfg(CfgFunctionArgs a, double multiplier, bool isFlow)
    {
        double s = a.Sigma, w = a.CondScale;
        var x = a.Input;
        int n = x.Length;
        var vc = new double[n];
        var vu = new double[n];
        double alpha = 1 / Math.Sqrt(1 + s * s);
        for (int i = 0; i < n; i++)
        {
            double c = a.CondDenoised[i], u = a.UncondDenoised[i];
            if (isFlow) { vc[i] = (x[i] - c) / s; vu[i] = (x[i] - u) / s; }
            else { vc[i] = alpha * ((x[i] - c) / s - s * c); vu[i] = alpha * ((x[i] - u) / s - s * u); }
        }
        var vcfg = new float[n];
        for (int i = 0; i < n; i++) vcfg[i] = (float)(vu[i] + w * (vc[i] - vu[i]));
        double factor = SampleStd([.. vc.Select(v => (float)v)]) / SampleStd(vcfg);
        double k = multiplier * factor + (1 - multiplier);
        var r = new float[n];
        for (int i = 0; i < n; i++)
        {
            double v = vcfg[i] * k;
            double denoised = isFlow ? x[i] - s * v : alpha * alpha * x[i] - s * alpha * v;
            r[i] = (float)(x[i] - denoised);
        }
        return r;
    }

    /// <summary>CFGNorm after cfg: the guided estimate scaled per pixel by min(|cond| / |guided|, 1) over the channels, times the strength.</summary>
    public static float[] CfgNormPost(PostCfgArgs a, double strength)
    {
        var ctx = a.Context;
        int plane = ctx.Height * ctx.Width, ch = a.Denoised.Length / Math.Max(plane, 1);
        var r = new float[a.Denoised.Length];
        for (int p = 0; p < plane; p++)
        {
            double nc = 0, nd = 0;
            for (int c = 0; c < ch; c++)
            {
                double vc = a.CondDenoised[c * plane + p], vd = a.Denoised[c * plane + p];
                nc += vc * vc;
                nd += vd * vd;
            }
            double k = Math.Min(Math.Sqrt(nc) / Math.Sqrt(nd), 1) * strength;
            for (int c = 0; c < ch; c++) r[c * plane + p] = (float)(a.Denoised[c * plane + p] * k);
        }
        return r;
    }

    /// <summary>CFGNorm as the cfg function: the guided noise blended towards its rescaling to the prompt's noise norm per pixel, by the strength.</summary>
    public static float[] CfgNormPre(CfgFunctionArgs a, double strength)
    {
        var ctx = a.Context;
        int plane = ctx.Height * ctx.Width, ch = a.Cond.Length / Math.Max(plane, 1);
        var r = new float[a.Cond.Length];
        float w = (float)a.CondScale;
        for (int p = 0; p < plane; p++)
        {
            double nc = 0, ng = 0;
            var g = new float[ch];
            for (int c = 0; c < ch; c++)
            {
                int i = c * plane + p;
                g[c] = a.Uncond[i] + (a.Cond[i] - a.Uncond[i]) * w;
                nc += (double)a.Cond[i] * a.Cond[i];
                ng += (double)g[c] * g[c];
            }
            double k = (1 - strength) + strength * Math.Sqrt(nc) / Math.Sqrt(ng);
            for (int c = 0; c < ch; c++) r[c * plane + p] = (float)(g[c] * k);
        }
        return r;
    }

    /// <summary>
    /// CFG-Zero*: the unconditional estimate's share rescaled by the noise predictions' projection ratio alpha: the incoming estimate plus
    /// (alpha - 1)(1 - w) u (for plain guidance, w c + (1 - w) alpha u; a guider's own estimate is kept otherwise, measured).
    /// </summary>
    public static float[] CfgZeroStar(PostCfgArgs a)
    {
        var x = a.Input;
        int n = x.Length;
        double num = 0, den = 0;
        for (int i = 0; i < n; i++)
        {
            double ec = x[i] - a.CondDenoised[i], eu = x[i] - a.UncondDenoised[i];
            num += ec * eu;
            den += eu * eu;
        }
        double alpha = num / (den + 1e-8), w = a.CondScale;
        var r = new float[n];
        for (int i = 0; i < n; i++) r[i] = (float)(a.Denoised[i] + (alpha - 1) * (1 - w) * a.UncondDenoised[i]);
        return r;
    }

    /// <summary>
    /// APG: the guidance difference (with momentum, then capped to the norm threshold) split into its parts along and across the prompt's estimate,
    /// the parallel part scaled by eta; the result is the prompt's estimate plus cfg times that update (measured), returned as the cond that plain
    /// guidance turns into it.
    /// </summary>
    public static Func<PreCfgArgs, float[][]> Apg(double eta, double normThreshold, double momentum)
    {
        float[]? running = null;
        return a =>
        {
            // At cfg 1 the negative is skipped and its slot holds zeros; APG still applies (measured).
            var outs = a.CondsOut;
            if (outs.Length < 2) return outs;
            float[] c = outs[0], u = outs[1];
            int n = c.Length;
            var diff = new float[n];
            for (int i = 0; i < n; i++) diff[i] = c[i] - u[i];
            if (momentum != 0)
            {
                if (running is null) running = diff;
                else
                {
                    var next = new float[n];
                    for (int i = 0; i < n; i++) next[i] = diff[i] + (float)momentum * running[i];
                    running = next;
                }
                diff = (float[])running.Clone();
            }
            if (normThreshold > 0)
            {
                double scale = Math.Min(1, normThreshold / Norm(diff));
                for (int i = 0; i < n; i++) diff[i] = (float)(diff[i] * scale);
            }
            double cn = Norm(c), p = 0;
            for (int i = 0; i < n; i++) p += diff[i] * (c[i] / cn);
            double w = a.CondScale;
            var cond = new float[n];
            for (int i = 0; i < n; i++)
            {
                double par = p * c[i] / cn, orth = diff[i] - par;
                double guided = c[i] + w * (orth + eta * par);
                cond[i] = (float)(u[i] + (guided - u[i]) / w);
            }
            return [cond, u, .. outs.Skip(2)];
        };
    }

    static float SignedSqrt(float v) => MathF.Sign(v) * MathF.Sqrt(MathF.Abs(v));

    /// <summary>
    /// Mahiro: the guided estimate blended with cfg times the prompt's estimate, by how similar (per-pixel cosine over the channels of signed square
    /// roots, averaged) the unconditional leap is to the merged one.
    /// </summary>
    public static float[] Mahiro(PostCfgArgs a)
    {
        var ctx = a.Context;
        int plane = ctx.Height * ctx.Width, ch = a.Denoised.Length / Math.Max(plane, 1);
        float w = (float)a.CondScale;
        double simSum = 0;
        for (int p = 0; p < plane; p++)
        {
            double dot = 0, n1 = 0, n2 = 0;
            for (int c = 0; c < ch; c++)
            {
                int i = c * plane + p;
                float ul = SignedSqrt(a.UncondDenoised[i] * w), merge = SignedSqrt((a.CondDenoised[i] * w + a.Denoised[i]) / 2);
                dot += ul * merge;
                n1 += ul * ul;
                n2 += merge * merge;
            }
            simSum += dot / Math.Max(Math.Sqrt(n1) * Math.Sqrt(n2), 1e-8);
        }
        double sim = simSum / plane, s = 2 * (sim + 1);
        var r = new float[a.Denoised.Length];
        for (int i = 0; i < r.Length; i++) r[i] = (float)((s * a.Denoised[i] + (4 - s) * a.CondDenoised[i] * w) / 4);
        return r;
    }

    /// <summary>TCFG's damping: the unconditional score projected onto the first right singular vector of [cond; uncond].</summary>
    public static float[] TangentialDamping(float[] condScore, float[] uncondScore)
    {
        double g11 = Dot(condScore, condScore), g12 = Dot(condScore, uncondScore), g22 = Dot(uncondScore, uncondScore);
        // The top eigenvector (e1, e2) of the 2 x 2 Gram matrix gives v = (e1 cond + e2 uncond) / |...|.
        double tr = g11 + g22, det = g11 * g22 - g12 * g12, l1 = tr / 2 + Math.Sqrt(Math.Max(tr * tr / 4 - det, 0));
        double e1, e2;
        if (Math.Abs(g12) > 1e-30) { e1 = g12; e2 = l1 - g11; }
        else if (g11 >= g22) { e1 = 1; e2 = 0; }
        else { e1 = 0; e2 = 1; }
        int n = condScore.Length;
        var v = new double[n];
        double vn = 0;
        for (int i = 0; i < n; i++) { v[i] = e1 * condScore[i] + e2 * uncondScore[i]; vn += v[i] * v[i]; }
        vn = Math.Sqrt(vn);
        double proj = 0;
        for (int i = 0; i < n; i++) proj += uncondScore[i] * v[i] / vn;
        var r = new float[n];
        for (int i = 0; i < n; i++) r[i] = (float)(proj * v[i] / vn);
        return r;
    }

    /// <summary>TCFG before cfg: the unconditional estimate's noise damped to its tangential part (measured: in noise space, input - denoised).</summary>
    public static float[][] Tcfg(PreCfgArgs a)
    {
        var outs = a.CondsOut;
        if (outs.Length < 2 || !a.CondsPresent[0] || !a.CondsPresent[1]) return outs;
        var x = a.Input;
        var ec = Samplers.Subtract(x, outs[0]);
        var eu = Samplers.Subtract(x, outs[1]);
        var damped = TangentialDamping(ec, eu);
        return [outs[0], Samplers.Subtract(x, damped), .. outs.Skip(2)];
    }

    /// <summary>Perp-Neg: the empty prompt plus cfg times (positive - negScale x the part of the negative perpendicular to the positive), both from the empty prompt.</summary>
    public static float[] PerpNeg(float[] pos, float[] neg, float[] nocond, double negScale, double condScale)
    {
        int n = pos.Length;
        var p = new float[n];
        var q = new float[n];
        for (int i = 0; i < n; i++) { p[i] = pos[i] - nocond[i]; q[i] = neg[i] - nocond[i]; }
        float k = (float)(Dot(q, p) / Dot(p, p));
        var r = new float[n];
        for (int i = 0; i < n; i++)
        {
            float perp = q[i] - k * p[i];
            r[i] = nocond[i] + (float)condScale * (p[i] - perp * (float)negScale);
        }
        return r;
    }

    /// <summary>
    /// FreSca's filter: each channel's 2-D Fourier transform (centred), the frequencies within <paramref name="cutoff"/> of the centre (the window
    /// clamped to the picture) scaled by <paramref name="scaleLow"/>, the rest by <paramref name="scaleHigh"/>, transformed back (real part).
    /// </summary>
    public static float[] FourierFilter(float[] x, int channels, int height, int width, double scaleLow, double scaleHigh, int cutoff)
    {
        int plane = height * width;
        var r = new float[x.Length];
        int cr = height / 2, cc = width / 2;
        // The window ends at twice the centre index (measured: for an odd size the last row / column stays high).
        int r0 = Math.Max(cr - cutoff, 0), r1 = Math.Min(cr + cutoff, 2 * cr), c0 = Math.Max(cc - cutoff, 0), c1 = Math.Min(cc + cutoff, 2 * cc);
        // Shifted index s holds frequency (s - h / 2) mod h, so frequency f sits at (f + h / 2) mod h.
        double Scale(int fy, int fx)
        {
            int sy = (fy + height / 2) % height, sx = (fx + width / 2) % width;
            return sy >= r0 && sy < r1 && sx >= c0 && sx < c1 ? scaleLow : scaleHigh;
        }
        var re = new double[plane];
        var im = new double[plane];
        for (int ch = 0; ch < channels; ch++)
        {
            int o = ch * plane;
            for (int fy = 0; fy < height; fy++)
                for (int fx = 0; fx < width; fx++)
                {
                    double sr = 0, si = 0;
                    for (int y = 0; y < height; y++)
                        for (int xx = 0; xx < width; xx++)
                        {
                            double ang = -2 * Math.PI * ((double)fy * y / height + (double)fx * xx / width);
                            double v = x[o + y * width + xx];
                            sr += v * Math.Cos(ang);
                            si += v * Math.Sin(ang);
                        }
                    double k = Scale(fy, fx);
                    re[fy * width + fx] = sr * k;
                    im[fy * width + fx] = si * k;
                }
            for (int y = 0; y < height; y++)
                for (int xx = 0; xx < width; xx++)
                {
                    double sum = 0;
                    for (int fy = 0; fy < height; fy++)
                        for (int fx = 0; fx < width; fx++)
                        {
                            double ang = 2 * Math.PI * ((double)fy * y / height + (double)fx * xx / width);
                            sum += re[fy * width + fx] * Math.Cos(ang) - im[fy * width + fx] * Math.Sin(ang);
                        }
                    r[o + y * width + xx] = (float)(sum / plane);
                }
        }
        return r;
    }

    /// <summary>FreSca: the guidance difference filtered in frequency; the cond becomes uncond + the filtered difference.</summary>
    public static float[][] FreSca(PreCfgArgs a, double scaleLow, double scaleHigh, int cutoff)
    {
        var outs = a.CondsOut;
        if (outs.Length < 2 || !a.CondsPresent[0] || !a.CondsPresent[1]) return outs;
        var ctx = a.Context;
        var diff = Samplers.Subtract(outs[0], outs[1]);
        var f = FourierFilter(diff, diff.Length / (ctx.Height * ctx.Width), ctx.Height, ctx.Width, scaleLow, scaleHigh, cutoff);
        var cond = new float[f.Length];
        for (int i = 0; i < f.Length; i++) cond[i] = outs[1][i] + f[i];
        return [cond, outs[1], .. outs.Skip(2)];
    }

    /// <summary>Epsilon Scaling: the noise prediction divided by the factor.</summary>
    public static float[] EpsilonScaling(PostCfgArgs a, double factor)
    {
        var x = a.Input;
        var r = new float[x.Length];
        for (int i = 0; i < r.Length; i++) r[i] = (float)(x[i] - (x[i] - a.Denoised[i]) / factor);
        return r;
    }

    /// <summary>Temporal score rescaling: the noise prediction scaled by (sigma^2 + v) / (sigma^2 + v / k).</summary>
    public static float[] TemporalScoreRescaling(PostCfgArgs a, double k, double variance)
    {
        double s2 = a.Sigma * a.Sigma, ratio = (s2 + variance) / (s2 + variance / k);
        var x = a.Input;
        var r = new float[x.Length];
        for (int i = 0; i < r.Length; i++) r[i] = (float)(x[i] - ratio * (x[i] - a.Denoised[i]));
        return r;
    }

    /// <summary>
    /// RenormCFG (measured): below <paramref name="cfgTrunc"/> (in sigma) the guided estimate, its norm capped at renorm_cfg times the prompt's;
    /// from there on the prompt alone. Returns input - estimate.
    /// </summary>
    public static float[] RenormCfg(CfgFunctionArgs a, double cfgTrunc, double renormCfg)
    {
        var x = a.Input;
        if (!(a.Sigma < cfgTrunc)) return (float[])a.Cond.Clone();
        var g = Samplers.Guide(a.UncondDenoised, a.CondDenoised, a.CondScale);
        double k = 1;
        if (renormCfg > 0)
        {
            double maxNorm = Norm(a.CondDenoised) * renormCfg, gn = Norm(g);
            if (gn >= maxNorm) k = maxNorm / gn;
        }
        var r = new float[x.Length];
        for (int i = 0; i < r.Length; i++) r[i] = (float)(x[i] - g[i] * k);
        return r;
    }

    /// <summary>
    /// Differential Diffusion: the inpainting mask made binary at the run's progress (in model timesteps from the run's first sigma down to its last
    /// sigma, or the model's lowest when the run ends below it); a strength strictly between 0 and 1 blends the original mask back in (measured).
    /// </summary>
    public static float[] DifferentialDiffusionMask(double sigma, float[] mask, double[] stepSigmas, ModelSampling sampling, double strength)
    {
        double sigmaTo = Math.Max(sampling.SigmaMin, stepSigmas[^1]);
        double tsFrom = sampling.Timestep(stepSigmas[0]), tsTo = sampling.Timestep(sigmaTo), current = sampling.Timestep(sigma);
        double threshold = (current - tsTo) / (tsFrom - tsTo);
        var r = new float[mask.Length];
        bool blend = strength > 0 && strength < 1;
        for (int i = 0; i < r.Length; i++)
        {
            float binary = mask[i] >= threshold ? 1f : 0f;
            r[i] = blend ? (float)(strength * binary + (1 - strength) * mask[i]) : binary;
        }
        return r;
    }

    /// <summary>Video Linear CFG: frame b of the batch guided at linspace(min_cfg, cfg, frames)[b] (on the noise predictions).</summary>
    public static float[] VideoLinearCfg(CfgFunctionArgs a, double minCfg)
    {
        var scales = Linspace((float)minCfg, (float)a.CondScale, Math.Max(a.Context.BatchSize, 1));
        return Samplers.Guide(a.Uncond, a.Cond, scales[Math.Min(a.Context.BatchIndex, scales.Length - 1)]);
    }

    /// <summary>Video Triangle CFG: the scale rises and falls along the frames, min_cfg at the ends, cfg in the middle.</summary>
    public static float[] VideoTriangleCfg(CfgFunctionArgs a, double minCfg)
    {
        int frames = Math.Max(a.Context.BatchSize, 1);
        var t = Linspace(0f, 1f, frames);
        float v = t[Math.Min(a.Context.BatchIndex, frames - 1)];
        float tri = 2 * MathF.Abs(v - MathF.Floor(v + 0.5f));
        float scale = tri * ((float)a.CondScale - (float)minCfg) + (float)minCfg;
        return Samplers.Guide(a.Uncond, a.Cond, scale);
    }

    /// <summary>torch.linspace in float32.</summary>
    public static float[] Linspace(float start, float end, int steps) => Samplers.TorchLinspace(start, end, steps);
}
