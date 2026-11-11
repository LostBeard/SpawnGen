using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md). The patch points follow ComfyUI's documented model options (pre-cfg functions that see every
// conditioning's denoised estimate, one cfg function that replaces the plain classifier-free guidance, post-cfg functions that adjust the result)
// and the behaviour SamplingNodeSchemaTests, RescaleCfgTests, LatentOperationTests and SamplerParityTests pin down; classifier-free guidance is
// Ho & Salimans 2022.

/// <summary>A cfg value used instead of the guider's between two sigmas (CFG Override); bounds are inclusive.</summary>
public sealed record CfgOverride(double Cfg, double SigmaHigh, double SigmaLow);

/// <summary>Deep Shrink (PatchModelAddDownscale) and the sigmas it acts between (inclusive).</summary>
public sealed record DeepShrinkWindow(SdDeepShrink Shrink, double SigmaStart, double SigmaEnd);

/// <summary>Options for the UNet itself, set by the UNet patch nodes.</summary>
public sealed record UNetOptions
{
    public static readonly UNetOptions None = new();

    public SdFreeU? FreeU { get; init; }
    public bool PerturbMiddleSelfAttention { get; init; }
    public bool RecordMiddleSelfAttention { get; init; }
    public SdHyperTile? HyperTile { get; init; }
    public SdTokenMerge? TokenMerge { get; init; }
    public DeepShrinkWindow? DeepShrink { get; init; }

    /// <summary>True when nothing is set (the plain UNet).</summary>
    public bool IsNone => this == None;

    /// <summary>The options in force at <paramref name="sigma"/>: Deep Shrink only inside its window.</summary>
    public UNetOptions At(double sigma) =>
        DeepShrink is { } d && !(sigma <= d.SigmaStart && sigma >= d.SigmaEnd) ? this with { DeepShrink = null } : this;
}

/// <summary>What a patch can see and ask of the run besides its arguments.</summary>
public sealed class PatchContext
{
    public int Channels { get; init; }
    public int Height { get; init; }
    public int Width { get; init; }
    public ModelSampling Sampling { get; init; } = null!;

    /// <summary>Which picture of the batch this is, and the batch size (the sampler runs picture by picture).</summary>
    public int BatchIndex { get; init; }
    public int BatchSize { get; init; } = 1;

    /// <summary>A conditioning's denoised estimate at this step, ignoring its timestep range.</summary>
    public Func<Conditioning, Task<float[]>> EvaluateUnscheduled { get; init; } = null!;

    /// <summary>The UNet options in force at this step.</summary>
    public UNetOptions? UNet { get; init; }

    /// <summary>A conditioning's denoised estimate with other UNet options.</summary>
    public Func<Conditioning, UNetOptions, Task<float[]>>? EvaluateWith { get; init; }

    /// <summary>A conditioning's denoised estimate at another input (same sigma).</summary>
    public Func<Conditioning, float[], Task<float[]>>? EvaluateAt { get; init; }

    /// <summary>The middle self-attention recorded during the last unconditional evaluation, when one was asked for.</summary>
    public Func<Task<float[]>>? ReadMiddleAttention { get; init; }
}

/// <summary>A pre-cfg patch's input: every conditioning's denoised estimate (zeros where one is absent).</summary>
public sealed record PreCfgArgs(bool[] CondsPresent, float[][] CondsOut, double CondScale, double Sigma, float[] Input, PatchContext Context);

/// <summary>A cfg function's input. Cond and Uncond are input - denoised (the noise-like predictions); the function returns that kind of value.</summary>
public sealed record CfgFunctionArgs(float[] Cond, float[] Uncond, double CondScale, double Sigma, float[] Input, float[] CondDenoised, float[] UncondDenoised,
    bool HasCond, bool HasUncond, PatchContext Context);

/// <summary>A post-cfg patch's input: the guided estimate and the two it came from.</summary>
public sealed record PostCfgArgs(float[] Denoised, bool HasCond, bool HasUncond, double CondScale, float[] CondDenoised, float[] UncondDenoised, double Sigma,
    float[] Input, PatchContext Context)
{
    public Conditioning? Cond { get; init; }
    public Conditioning? Uncond { get; init; }
}

/// <summary>The two estimates of the last guided prediction (what the CFG++ samplers read).</summary>
public sealed record GuidanceOutput(float[] CondDenoised, float[] UncondDenoised, bool HasUncond);

/// <summary>
/// The guidance patches on a model. Patches are stored as factories so each sampling run starts with fresh state (APG's momentum); a newer cfg
/// function replaces an older one, pre- and post-cfg patches stack in the order they were added.
/// </summary>
public sealed record ModelPatches
{
    public static readonly ModelPatches None = new();

    public IReadOnlyList<Func<Func<PreCfgArgs, float[][]>>> PreCfg { get; init; } = [];
    public Func<Func<CfgFunctionArgs, Task<float[]>>>? CfgFunction { get; init; }
    public IReadOnlyList<Func<Func<PostCfgArgs, Task<float[]>>>> PostCfg { get; init; } = [];

    /// <summary>Evaluate the negative even at cfg 1 (a patch that needs it).</summary>
    public bool DisableCfg1Optimization { get; init; }

    /// <summary>Differential Diffusion: the inpainting mask to use at a sigma, from the mask, the run's sigmas and the model sampling.</summary>
    public Func<double, float[], double[], ModelSampling, float[]>? DenoiseMask { get; init; }

    public IReadOnlyList<CfgOverride> CfgOverrides { get; init; } = [];
    public UNetOptions? UNet { get; init; }

    /// <summary>Record the middle self-attention during the unconditional evaluation (Self-Attention Guidance).</summary>
    public bool RecordUncondMiddleAttention { get; init; }

    public ModelPatches WithPreCfg(Func<Func<PreCfgArgs, float[][]>> patch, bool disableCfg1 = false) =>
        this with { PreCfg = [.. PreCfg, patch], DisableCfg1Optimization = DisableCfg1Optimization || disableCfg1 };

    public ModelPatches WithPostCfg(Func<Func<PostCfgArgs, float[]>> patch, bool disableCfg1 = false) =>
        WithPostCfgAsync(() =>
        {
            var f = patch();
            return a => Task.FromResult(f(a));
        }, disableCfg1);

    public ModelPatches WithPostCfgAsync(Func<Func<PostCfgArgs, Task<float[]>>> patch, bool disableCfg1 = false) =>
        this with { PostCfg = [.. PostCfg, patch], DisableCfg1Optimization = DisableCfg1Optimization || disableCfg1 };

    public ModelPatches WithCfgFunction(Func<Func<CfgFunctionArgs, Task<float[]>>> patch, bool disableCfg1 = false) =>
        this with { CfgFunction = patch, DisableCfg1Optimization = DisableCfg1Optimization || disableCfg1 };

    /// <summary>A patch without state, as a factory that hands out the same function.</summary>
    public static Func<T> Stateless<T>(T function) => () => function;
}

/// <summary>
/// GUIDER: how the model is steered at each step, which conditionings are evaluated and how their estimates are combined. <see cref="Cfg"/> is the
/// guider's own scale (the one CFG Override replaces).
/// </summary>
[Socket("GUIDER")]
public abstract class GuiderChoice(ModelHandle model, double cfg)
{
    public ModelHandle Model { get; } = model;
    public double Cfg { get; } = cfg;

    /// <summary>Every conditioning the guider may evaluate.</summary>
    public abstract IEnumerable<Conditioning> Conditionings { get; }

    /// <summary>The guided denoised estimate for the step, with <paramref name="cfg"/> after any override.</summary>
    public abstract Task<float[]> PredictAsync(GuidanceStep step, double cfg);
}

/// <summary>Classifier-free guidance (CFG Guider, KSampler, SamplerCustom): uncond + cfg (cond - uncond).</summary>
public sealed class CfgGuider(ModelHandle model, Conditioning positive, Conditioning negative, double cfg) : GuiderChoice(model, cfg)
{
    public Conditioning Positive { get; } = positive;
    public Conditioning Negative { get; } = negative;
    public override IEnumerable<Conditioning> Conditionings => [Positive, Negative];
    public override Task<float[]> PredictAsync(GuidanceStep step, double cfg) => step.SamplingFunction(Positive, Negative, cfg);
}

/// <summary>The prompt alone (Basic Guider): no negative, so the unconditional estimate is zero, at cfg 1.</summary>
public sealed class BasicGuiderChoice(ModelHandle model, Conditioning conditioning) : GuiderChoice(model, 1.0)
{
    public Conditioning Positive { get; } = conditioning;
    public override IEnumerable<Conditioning> Conditionings => [Positive];
    public override Task<float[]> PredictAsync(GuidanceStep step, double cfg) => step.SamplingFunction(Positive, null, cfg);
}

/// <summary>
/// Two prompts and a negative (Dual CFG Guider). "regular": the second prompt guided against the negative by cfg_cond2_negative, plus cfg_conds
/// times (first - second). "nested": the first prompt guided against the second by cfg_conds, that result against the negative by
/// cfg_cond2_negative.
/// </summary>
public sealed class DualCfgGuiderChoice(ModelHandle model, Conditioning cond1, Conditioning cond2, Conditioning negative, double cfgConds, double cfgCond2Negative,
    bool nested) : GuiderChoice(model, 1.0)
{
    public override IEnumerable<Conditioning> Conditionings => [cond1, cond2, negative];

    public override async Task<float[]> PredictAsync(GuidanceStep step, double cfg)
    {
        var outs = await step.CalcCondBatch(negative, cond2, cond1);
        float[] neg = outs[0], second = outs[1], first = outs[2];
        if (nested)
        {
            var inner = Samplers.Guide(second, first, cfgConds);
            return Samplers.Guide(neg, inner, cfgCond2Negative);
        }
        var guided = Samplers.Guide(neg, second, cfgCond2Negative);
        var r = new float[guided.Length];
        float k = (float)cfgConds;
        for (int i = 0; i < r.Length; i++) r[i] = guided[i] + (first[i] - second[i]) * k;
        return r;
    }
}

/// <summary>Perp-Neg (Armandpour et al. 2023) as a guider: the negative pushes only along its part perpendicular to the prompt, both measured from an empty prompt.</summary>
public sealed class PerpNegGuiderChoice(ModelHandle model, Conditioning positive, Conditioning negative, Conditioning empty, double cfg, double negScale)
    : GuiderChoice(model, cfg)
{
    public override IEnumerable<Conditioning> Conditionings => [positive, negative, empty];

    public override async Task<float[]> PredictAsync(GuidanceStep step, double cfg)
    {
        var conds = new Conditioning?[] { positive, negative, empty };
        var outs = step.ApplyPreCfg(conds, await step.CalcCondBatch(conds), cfg);
        var result = GuidancePatches.PerpNeg(outs[0], outs[1], outs[2], negScale, cfg);
        return await step.ApplyPostCfg(result, outs[0], outs[1], cfg, positive, negative);
    }
}

/// <summary>One guided prediction: the step's input and sigma, and the stages of guidance a guider combines.</summary>
public sealed class GuidanceStep(GuidanceRun run, float[] x, double sigma, Func<Conditioning, bool, Task<float[]>> evaluate, PatchContext context)
{
    public float[] X { get; } = x;
    public double Sigma { get; } = sigma;
    public PatchContext Context { get; } = context;
    public bool DisableCfg1Optimization => run.DisableCfg1;

    /// <summary>Each conditioning's denoised estimate (zeros for a missing one); the second is the unconditional slot.</summary>
    public async Task<float[][]> CalcCondBatch(params Conditioning?[] conds)
    {
        var outs = new float[conds.Length][];
        for (int i = 0; i < conds.Length; i++)
            outs[i] = conds[i] is { } c ? await evaluate(c, i == 1) : new float[X.Length];
        return outs;
    }

    /// <summary>The standard pipeline: estimates, pre-cfg patches, the cfg (function), post-cfg patches; at cfg 1 the negative is skipped unless disabled.</summary>
    public async Task<float[]> SamplingFunction(Conditioning cond, Conditioning? uncond, double cfg)
    {
        if (!DisableCfg1Optimization && Math.Abs(cfg - 1.0) <= 1e-9) uncond = null;
        var conds = new[] { cond, uncond };
        var outs = ApplyPreCfg(conds, await CalcCondBatch(conds), cfg);
        var guided = await CfgFunction(outs[0], outs[1], cfg, cond, uncond);
        return await ApplyPostCfg(guided, outs[0], outs[1], cfg, cond, uncond);
    }

    public float[][] ApplyPreCfg(Conditioning?[] conds, float[][] outs, double cfg)
    {
        if (run.PreCfg.Length == 0) return outs;
        var present = conds.Select(c => c is not null).ToArray();
        foreach (var patch in run.PreCfg) outs = patch(new PreCfgArgs(present, outs, cfg, Sigma, X, Context));
        return outs;
    }

    public async Task<float[]> CfgFunction(float[] condPred, float[] uncondPred, double cfg, Conditioning? cond, Conditioning? uncond)
    {
        if (run.CfgFn is null) return Samplers.Guide(uncondPred, condPred, cfg);
        var r = await run.CfgFn(new CfgFunctionArgs(Samplers.Subtract(X, condPred), Samplers.Subtract(X, uncondPred), cfg, Sigma, X, condPred, uncondPred,
            cond is not null, uncond is not null, Context));
        return Samplers.Subtract(X, r);
    }

    public async Task<float[]> ApplyPostCfg(float[] denoised, float[] condPred, float[] uncondPred, double cfg, Conditioning? cond, Conditioning? uncond)
    {
        run.Last = new GuidanceOutput(condPred, uncondPred, uncond is not null);
        foreach (var patch in run.PostCfg)
            denoised = await patch(new PostCfgArgs(denoised, cond is not null, uncond is not null, cfg, condPred, uncondPred, Sigma, X, Context) { Cond = cond, Uncond = uncond });
        return denoised;
    }
}

/// <summary>A sampling run's guidance for one picture: the guider with its model's patches instantiated once for the run.</summary>
public sealed class GuidanceRun
{
    public readonly GuiderChoice Guider;
    public readonly Func<PreCfgArgs, float[][]>[] PreCfg;
    public readonly Func<CfgFunctionArgs, Task<float[]>>? CfgFn;
    public readonly Func<PostCfgArgs, Task<float[]>>[] PostCfg;
    public readonly bool DisableCfg1;
    readonly int _channels, _height, _width;
    readonly ModelPatches _patches;

    public GuidanceRun(GuiderChoice guider, int channels, int height, int width, bool samplerDisablesCfg1)
    {
        Guider = guider;
        _patches = guider.Model.Patches ?? ModelPatches.None;
        PreCfg = [.. _patches.PreCfg.Select(f => f())];
        CfgFn = _patches.CfgFunction?.Invoke();
        PostCfg = [.. _patches.PostCfg.Select(f => f())];
        DisableCfg1 = samplerDisablesCfg1 || _patches.DisableCfg1Optimization;
        (_channels, _height, _width) = (channels, height, width);
    }

    public int BatchIndex { get; init; }
    public int BatchSize { get; init; } = 1;

    /// <summary>The estimates of the last prediction.</summary>
    public GuidanceOutput? Last { get; set; }

    /// <summary>
    /// The guided denoised estimate at <paramref name="x"/>, <paramref name="sigma"/>. <paramref name="evaluate"/> gives a conditioning's denoised
    /// estimate (whether its timestep range applies, the UNet options); the optional functions serve Self-Attention Guidance.
    /// </summary>
    public Task<float[]> PredictAsync(float[] x, double sigma, Func<Conditioning, bool, UNetOptions, Task<float[]>> evaluate,
        Func<Conditioning, float[], UNetOptions, Task<float[]>>? evaluateAt = null, Func<Task<float[]>>? readMiddleAttention = null)
    {
        var options = (_patches.UNet ?? UNetOptions.None).At(sigma);
        double cfg = Guider.Cfg;
        foreach (var o in _patches.CfgOverrides)
            if (sigma <= o.SigmaHigh && sigma >= o.SigmaLow) cfg = o.Cfg;
        var context = new PatchContext
        {
            Channels = _channels, Height = _height, Width = _width, Sampling = Guider.Model.Sampling, BatchIndex = BatchIndex, BatchSize = BatchSize,
            EvaluateUnscheduled = c => evaluate(c, false, options),
            UNet = options,
            EvaluateWith = (c, o) => evaluate(c, true, o),
            EvaluateAt = evaluateAt is null ? null : (c, at) => evaluateAt(c, at, options),
            ReadMiddleAttention = readMiddleAttention,
        };
        var recording = _patches.RecordUncondMiddleAttention ? options with { RecordMiddleSelfAttention = true } : options;
        var step = new GuidanceStep(this, x, sigma, (c, isUncond) => evaluate(c, true, isUncond ? recording : options), context);
        return Guider.PredictAsync(step, cfg);
    }
}
