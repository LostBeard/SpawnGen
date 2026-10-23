using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.ML;

/// <summary>What a guidance patch works on: one picture's latent (channels, height, width, channel-first like the sampler), the model sampling, and a way to ask the model about another conditioning.</summary>
public sealed class PatchContext
{
    public required int Channels { get; init; }
    public required int Height { get; init; }
    public required int Width { get; init; }
    public required ModelSampling Sampling { get; init; }
    /// <summary>The model's denoised estimate for a conditioning at the current sample and sigma, ignoring its timestep range (PerpNeg's empty prompt).</summary>
    public required Func<Conditioning, Task<float[]>> EvaluateUnscheduled { get; init; }
}

/// <summary>ComfyUI's sampler_cfg_function arguments: <c>Cond</c> / <c>Uncond</c> are x minus the denoised estimates; the result r gives the guided estimate x - r.</summary>
public sealed record CfgFunctionArgs(float[] Cond, float[] Uncond, double CondScale, double Sigma, float[] Input, float[] CondDenoised, float[] UncondDenoised,
    bool HasCond, bool HasUncond, PatchContext Context);

/// <summary>ComfyUI's sampler_post_cfg_function arguments: the guided estimate so far and what it came from (HasCond / HasUncond: whether those conditionings were used).</summary>
public sealed record PostCfgArgs(float[] Denoised, bool HasCond, bool HasUncond, double CondScale, float[] CondDenoised, float[] UncondDenoised, double Sigma, float[] Input,
    PatchContext Context);

/// <summary>ComfyUI's sampler_pre_cfg_function arguments: the denoised estimate of each conditioning (zero for one not used, see CondsPresent) before they are combined.</summary>
public sealed record PreCfgArgs(bool[] CondsPresent, float[][] CondsOut, double CondScale, double Sigma, float[] Input, PatchContext Context);

/// <summary>A cfg override from CFGOverride: <c>Cfg</c> replaces the guider's cfg while sigma is within [SigmaLow, SigmaHigh].</summary>
public sealed record CfgOverride(double Cfg, double SigmaHigh, double SigmaLow);

/// <summary>
/// The guidance patches on a model (ComfyUI's model_options entries). Each patch is a factory, called once per sampling run, so a patch that keeps state across
/// steps (APG's running average) starts fresh. Patches are added in order; a newer cfg function replaces the older one, as in ComfyUI.
/// </summary>
public sealed record ModelPatches
{
    public static readonly ModelPatches None = new();

    public IReadOnlyList<Func<Func<PreCfgArgs, float[][]>>> PreCfg { get; init; } = [];
    public Func<Func<CfgFunctionArgs, Task<float[]>>>? CfgFunction { get; init; }
    public IReadOnlyList<Func<Func<PostCfgArgs, float[]>>> PostCfg { get; init; } = [];
    public bool DisableCfg1Optimization { get; init; }
    /// <summary>ComfyUI's denoise_mask_function: (sigma, inpaint mask per pixel, the run's sigmas, model sampling) to the mask used for this model call.</summary>
    public Func<double, float[], double[], ModelSampling, float[]>? DenoiseMask { get; init; }
    public IReadOnlyList<CfgOverride> CfgOverrides { get; init; } = [];

    public ModelPatches WithPreCfg(Func<Func<PreCfgArgs, float[][]>> patch, bool disableCfg1 = false) =>
        this with { PreCfg = [.. PreCfg, patch], DisableCfg1Optimization = DisableCfg1Optimization || disableCfg1 };

    public ModelPatches WithPostCfg(Func<Func<PostCfgArgs, float[]>> patch, bool disableCfg1 = false) =>
        this with { PostCfg = [.. PostCfg, patch], DisableCfg1Optimization = DisableCfg1Optimization || disableCfg1 };

    public ModelPatches WithCfgFunction(Func<Func<CfgFunctionArgs, Task<float[]>>> patch, bool disableCfg1 = false) =>
        this with { CfgFunction = patch, DisableCfg1Optimization = DisableCfg1Optimization || disableCfg1 };

    /// <summary>A stateless patch as a factory.</summary>
    public static Func<T> Stateless<T>(T function) => () => function;
}

/// <summary>
/// How the model is steered at each step (ComfyUI's GUIDER): which conditionings are evaluated and how their denoised estimates are combined. <see cref="Cfg"/> is the
/// guider's own cfg (the one CFGOverride replaces); guiders with two scales keep theirs separately.
/// </summary>
[Socket("GUIDER")]
public abstract class GuiderChoice(ModelHandle model, double cfg)
{
    public ModelHandle Model { get; } = model;
    public double Cfg { get; } = cfg;

    /// <summary>Every conditioning the guider may evaluate (their embeddings go to the GPU once per run).</summary>
    public abstract IEnumerable<Conditioning> Conditionings { get; }

    /// <summary>ComfyUI's predict_noise: the guided denoised estimate at the step's sample and sigma, with <paramref name="cfg"/> after any override.</summary>
    internal abstract Task<float[]> PredictAsync(GuidanceStep step, double cfg);
}

/// <summary>Classifier-free guidance: uncond + cfg * (cond - uncond), the negative skipped at cfg 1 (CFGGuider, and what KSampler and SamplerCustom use).</summary>
public sealed class CfgGuider(ModelHandle model, Conditioning positive, Conditioning negative, double cfg) : GuiderChoice(model, cfg)
{
    public Conditioning Positive { get; } = positive;
    public Conditioning Negative { get; } = negative;
    public override IEnumerable<Conditioning> Conditionings => [Positive, Negative];
    internal override Task<float[]> PredictAsync(GuidanceStep step, double cfg) => step.SamplingFunction(Positive, Negative, cfg);
}

/// <summary>The prompt alone (BasicGuider): there is no negative, so the unconditional estimate is zero, at the guider's cfg of 1.</summary>
public sealed class BasicGuiderChoice(ModelHandle model, Conditioning conditioning) : GuiderChoice(model, 1.0)
{
    public Conditioning Positive { get; } = conditioning;
    public override IEnumerable<Conditioning> Conditionings => [Positive];
    internal override Task<float[]> PredictAsync(GuidanceStep step, double cfg) => step.SamplingFunction(Positive, null, cfg);
}

/// <summary>
/// Two prompts and a negative (DualCFGGuider): "regular" is cfg_function(cond2 vs negative, cfg_cond2_negative) + (cond1 - cond2) * cfg_conds; "nested" guides cond1 against
/// cond2 first and the result against the negative.
/// </summary>
public sealed class DualCfgGuiderChoice(ModelHandle model, Conditioning cond1, Conditioning cond2, Conditioning negative, double cfgConds, double cfgCond2Negative, bool nested)
    : GuiderChoice(model, 1.0)
{
    public override IEnumerable<Conditioning> Conditionings => [cond1, cond2, negative];

    internal override async Task<float[]> PredictAsync(GuidanceStep step, double cfg)
    {
        Conditioning? negativeCond = negative, middleCond = cond2;
        if (nested)
        {
            var o = await step.CalcCondBatch(negativeCond, middleCond, cond1);
            var predText = await step.CfgFunction(o[2], o[1], cfgConds, cond1, middleCond);
            return Samplers.Guide(o[0], predText, cfgCond2Negative);
        }
        if (!step.DisableCfg1Optimization && cfgCond2Negative == 1.0)
        {
            negativeCond = null;
            if (cfgConds == 1.0) middleCond = null;
        }
        var outs = await step.CalcCondBatch(negativeCond, middleCond, cond1);
        var guided = await step.CfgFunction(outs[1], outs[0], cfgCond2Negative, middleCond, negativeCond);
        var r = new float[guided.Length];
        for (int k = 0; k < r.Length; k++) r[k] = (float)(guided[k] + (outs[2][k] - outs[1][k]) * cfgConds);
        return r;
    }
}

/// <summary>Perp-Neg (PerpNegGuider): the negative only pushes along the part of it perpendicular to the prompt, both measured from an empty prompt.</summary>
public sealed class PerpNegGuiderChoice(ModelHandle model, Conditioning positive, Conditioning negative, Conditioning empty, double cfg, double negScale) : GuiderChoice(model, cfg)
{
    public override IEnumerable<Conditioning> Conditionings => [positive, negative, empty];

    internal override async Task<float[]> PredictAsync(GuidanceStep step, double cfg)
    {
        Conditioning? negativeCond = negative, emptyCond = empty;
        if (!step.DisableCfg1Optimization && negScale == 0.0)
        {
            negativeCond = null;
            if (cfg == 1.0) emptyCond = null;
        }
        var outs = step.ApplyPreCfg([positive, negativeCond, emptyCond], await step.CalcCondBatch(positive, negativeCond, emptyCond), cfg);
        var result = GuidancePatches.PerpNeg(outs[0], outs[1], outs[2], negScale, cfg);
        return step.ApplyPostCfg(result, outs[0], outs[1], cfg, true, negativeCond is not null);
    }
}

/// <summary>
/// One sampling run's guidance: the guider plus fresh instances of the model's patches. <see cref="PredictAsync"/> is one model call (ComfyUI's CFGGuider.predict_noise
/// with its wrappers); <see cref="Last"/> is what the CFG++ samplers read afterwards.
/// </summary>
public sealed class GuidanceRun
{
    internal readonly GuiderChoice Guider;
    internal readonly Func<PreCfgArgs, float[][]>[] PreCfg;
    internal readonly Func<CfgFunctionArgs, Task<float[]>>? CfgFn;
    internal readonly Func<PostCfgArgs, float[]>[] PostCfg;
    internal readonly bool DisableCfg1;
    readonly int _channels, _height, _width;

    public GuidanceRun(GuiderChoice guider, int channels, int height, int width, bool samplerDisablesCfg1)
    {
        Guider = guider;
        var p = guider.Model.Patches;
        PreCfg = [.. p.PreCfg.Select(f => f())];
        CfgFn = p.CfgFunction?.Invoke();
        PostCfg = [.. p.PostCfg.Select(f => f())];
        DisableCfg1 = p.DisableCfg1Optimization || samplerDisablesCfg1;
        (_channels, _height, _width) = (channels, height, width);
    }

    /// <summary>The conditional / unconditional estimates of the last cfg combination (ComfyUI's post-cfg capture of the CFG++ samplers).</summary>
    public GuidanceOutput? Last { get; internal set; }

    /// <summary>The guided denoised estimate at <paramref name="x"/>, <paramref name="sigma"/>. <paramref name="evaluate"/> is calc_cond_batch for one conditioning (null: zero).</summary>
    public Task<float[]> PredictAsync(float[] x, double sigma, Func<Conditioning, bool, Task<float[]>> evaluate)
    {
        double cfg = Guider.Cfg;
        foreach (var o in Guider.Model.Patches.CfgOverrides)
            if (o.SigmaLow <= sigma && sigma <= o.SigmaHigh) cfg = o.Cfg; // the override nearest the sampler (added last) wins
        var step = new GuidanceStep(this, x, sigma, evaluate, new PatchContext
        {
            Channels = _channels, Height = _height, Width = _width, Sampling = Guider.Model.Sampling,
            EvaluateUnscheduled = c => evaluate(c, false),
        });
        return Guider.PredictAsync(step, cfg);
    }
}

/// <summary>One model call of a <see cref="GuidanceRun"/>: ComfyUI's sampling_function, cfg_function and calc_cond_batch at a fixed sample and sigma.</summary>
public sealed class GuidanceStep
{
    readonly GuidanceRun _run;
    readonly Func<Conditioning, bool, Task<float[]>> _evaluate;

    internal GuidanceStep(GuidanceRun run, float[] x, double sigma, Func<Conditioning, bool, Task<float[]>> evaluate, PatchContext context)
    {
        _run = run;
        X = x;
        Sigma = sigma;
        _evaluate = evaluate;
        Context = context;
    }

    public float[] X { get; }
    public double Sigma { get; }
    public PatchContext Context { get; }
    public bool DisableCfg1Optimization => _run.DisableCfg1;

    /// <summary>The denoised estimate of each conditioning; a missing one is zero, like ComfyUI's empty batch slot.</summary>
    public async Task<float[][]> CalcCondBatch(params Conditioning?[] conds)
    {
        var r = new float[conds.Length][];
        for (int i = 0; i < conds.Length; i++) r[i] = conds[i] is { } c ? await _evaluate(c, true) : new float[X.Length];
        return r;
    }

    /// <summary>ComfyUI's sampling_function: the prompt and (unless cfg is 1) the negative, the pre-cfg patches, then <see cref="CfgFunction"/>.</summary>
    public async Task<float[]> SamplingFunction(Conditioning? cond, Conditioning? uncond, double cfg)
    {
        var uncondUsed = cfg == 1.0 && !DisableCfg1Optimization ? null : uncond;
        var outs = ApplyPreCfg([cond, uncondUsed], await CalcCondBatch(cond, uncondUsed), cfg);
        return await CfgFunction(outs[0], outs[1], cfg, cond, uncondUsed);
    }

    /// <summary>The pre-cfg patches over the per-conditioning estimates.</summary>
    public float[][] ApplyPreCfg(Conditioning?[] conds, float[][] outs, double cfg)
    {
        var present = conds.Select(c => c is not null).ToArray();
        foreach (var fn in _run.PreCfg) outs = fn(new PreCfgArgs(present, outs, cfg, Sigma, X, Context));
        return outs;
    }

    /// <summary>ComfyUI's cfg_function: the cfg function patch (or plain CFG), then the post-cfg patches; records the estimates for the CFG++ samplers.</summary>
    public async Task<float[]> CfgFunction(float[] condPred, float[] uncondPred, double cfg, Conditioning? cond, Conditioning? uncond)
    {
        float[] result;
        if (_run.CfgFn is { } fn)
        {
            var noise = await fn(new CfgFunctionArgs(Samplers.Subtract(X, condPred), Samplers.Subtract(X, uncondPred), cfg, Sigma, X, condPred, uncondPred,
                cond is not null, uncond is not null, Context));
            result = Samplers.Subtract(X, noise);
        }
        else result = Samplers.Guide(uncondPred, condPred, cfg);
        return ApplyPostCfg(result, condPred, uncondPred, cfg, cond is not null, uncond is not null);
    }

    /// <summary>The post-cfg patches (and the CFG++ capture, which ComfyUI's samplers add as the last post-cfg function).</summary>
    public float[] ApplyPostCfg(float[] denoised, float[] condPred, float[] uncondPred, double cfg, bool hasCond, bool hasUncond)
    {
        foreach (var fn in _run.PostCfg) denoised = fn(new PostCfgArgs(denoised, hasCond, hasUncond, cfg, condPred, uncondPred, Sigma, X, Context));
        _run.Last = new GuidanceOutput(condPred, uncondPred, hasUncond);
        return denoised;
    }
}
