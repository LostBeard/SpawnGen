using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.ML;

// ComfyUI's guidance patches and guiders: each patch node returns a copy of the model with a function added to its guidance (see GuidancePatches for the maths).

[Node(Category = "advanced/guidance", DisplayName = "CFGNorm", OutputNames = ["patched_model"])]
public sealed class CFGNorm
{
    public ModelHandle Execute(ModelHandle model, [Input(Min = 0, Max = 100, Step = 0.01)] double strength = 1.0,
        [Input(Optional = true, Tooltip = "If true, rescale the combined noise BEFORE the sampler's CFG combine, without clamping (can amplify). Matches the norm-scaled CFG used by models like Lens. Default false keeps the original post-CFG x0-space attenuate-only behavior.")]
        bool preCfg = false) =>
        preCfg
            ? model.WithPatches(p => p.WithCfgFunction(ModelPatches.Stateless<Func<CfgFunctionArgs, Task<float[]>>>(a => Task.FromResult(GuidancePatches.CfgNormPre(a, strength)))))
            : model.WithPatches(p => p.WithPostCfg(ModelPatches.Stateless<Func<PostCfgArgs, float[]>>(a => GuidancePatches.CfgNormPost(a, strength))));
}

[Node(Category = "advanced/guidance", DisplayName = "CFGZeroStar", OutputNames = ["patched_model"])]
public sealed class CFGZeroStar
{
    public ModelHandle Execute(ModelHandle model) =>
        model.WithPatches(p => p.WithPostCfg(ModelPatches.Stateless<Func<PostCfgArgs, float[]>>(GuidancePatches.CfgZeroStar)));
}

[Node(Category = "model/sampling/custom", DisplayName = "Adaptive Projected Guidance")]
public sealed class APG
{
    public ModelHandle Execute(ModelHandle model,
        [Input(Min = -10, Max = 10, Step = 0.01, Tooltip = "Controls the scale of the parallel guidance vector. Default CFG behavior at a setting of 1.")] double eta = 1.0,
        [Input(Min = 0, Max = 50, Step = 0.1, Tooltip = "Normalize guidance vector to this value, normalization disable at a setting of 0.")] double normThreshold = 5.0,
        [Input(Min = -5, Max = 1, Step = 0.01, Tooltip = "Controls a running average of guidance during diffusion, disabled at a setting of 0.")] double momentum = 0.0) =>
        model.WithPatches(p => p.WithPreCfg(() => GuidancePatches.Apg(eta, normThreshold, momentum)));
}

[Node(Category = "experimental", DisplayName = "Positive-Biased Guidance", OutputNames = ["patched_model"],
    Description = "Modify the guidance to scale more on the 'direction' of the positive prompt rather than the difference between the negative prompt.")]
public sealed class Mahiro
{
    public ModelHandle Execute(ModelHandle model) =>
        model.WithPatches(p => p.WithPostCfg(ModelPatches.Stateless<Func<PostCfgArgs, float[]>>(GuidancePatches.Mahiro)));
}

[Node(Category = "advanced/guidance", DisplayName = "Tangential Damping CFG", OutputNames = ["patched_model"],
    Description = "TCFG – Tangential Damping CFG (2503.18137)\n\nRefine the uncond (negative) to align with the cond (positive) for improving quality.")]
public sealed class TCFG
{
    public ModelHandle Execute(ModelHandle model) =>
        model.WithPatches(p => p.WithPreCfg(ModelPatches.Stateless<Func<PreCfgArgs, float[][]>>(GuidancePatches.Tcfg)));
}

[Node(Category = "experimental", DisplayName = "Perp-Neg (DEPRECATED by Perp-Neg Guider)")]
public sealed class PerpNeg
{
    /// <summary>The old Perp-Neg patch: a cfg function that also asks the model about the empty prompt at every step. Perp-Neg Guider does the same in one pass.</summary>
    public ModelHandle Execute(ModelHandle model, Conditioning emptyConditioning, [Input(Min = 0, Max = 100, Step = 0.01)] double negScale = 1.0) =>
        model.WithPatches(p => p.WithCfgFunction(ModelPatches.Stateless<Func<CfgFunctionArgs, Task<float[]>>>(async a =>
        {
            var nocond = await a.Context.EvaluateUnscheduled(emptyConditioning);
            return Samplers.Subtract(a.Input, GuidancePatches.PerpNeg(a.CondDenoised, a.UncondDenoised, nocond, negScale, a.CondScale));
        })));
}

[Node(Category = "experimental", DisplayName = "Perp-Neg Guider")]
public sealed class PerpNegGuider
{
    public GuiderChoice Execute(ModelHandle model, Conditioning positive, Conditioning negative, Conditioning emptyConditioning,
        [Input(Min = 0, Max = 100, Step = 0.1)] double cfg = 8.0, [Input(Min = 0, Max = 100, Step = 0.01)] double negScale = 1.0) =>
        new PerpNegGuiderChoice(model, positive, negative, emptyConditioning, cfg, negScale);
}

[Node(Category = "experimental", DisplayName = "FreSca", Description = "Applies frequency-dependent scaling to the guidance")]
public sealed class FreSca
{
    public ModelHandle Execute(ModelHandle model,
        [Input(Min = 0, Max = 10, Step = 0.01, Tooltip = "Scaling factor for low-frequency components")] double scaleLow = 1.0,
        [Input(Min = 0, Max = 10, Step = 0.01, Tooltip = "Scaling factor for high-frequency components")] double scaleHigh = 1.25,
        [Input(Min = 1, Max = 10000, Step = 1, Tooltip = "Number of frequency indices around center to consider as low-frequency")] int freqCutoff = 20) =>
        model.WithPatches(p => p.WithPreCfg(ModelPatches.Stateless<Func<PreCfgArgs, float[][]>>(a => GuidancePatches.FreSca(a, scaleLow, scaleHigh, freqCutoff))));
}

[Node("Epsilon Scaling", Category = "model/patch/unet", DisplayName = "Epsilon Scaling")]
public sealed class EpsilonScaling
{
    /// <summary>Epsilon Scaling (arXiv 2308.15321, uniform schedule): the predicted noise is divided by the factor after guidance, against exposure bias.</summary>
    public ModelHandle Execute(ModelHandle model, [Input(Min = 0.5, Max = 1.5, Step = 0.001)] double scalingFactor = 1.005) =>
        model.WithPatches(p => p.WithPostCfg(ModelPatches.Stateless<Func<PostCfgArgs, float[]>>(a => GuidancePatches.EpsilonScaling(a, scalingFactor))));
}

[Node(Category = "model/patch/unet", DisplayName = "TSR - Temporal Score Rescaling", OutputNames = ["patched_model"],
    Description = "[Post-CFG Function]\nTSR - Temporal Score Rescaling (2510.01184)\n\nRescaling the model's score or noise to steer the sampling diversity.\n")]
public sealed class TemporalScoreRescaling
{
    public ModelHandle Execute(ModelHandle model,
        [Input(Name = "tsr_k", Min = 0.01, Max = 100, Step = 0.001, Tooltip = "Controls the rescaling strength.\nLower k produces more detailed results; higher k produces smoother results in image generation. Setting k = 1 disables rescaling.")] double tsrK = 0.95,
        [Input(Name = "tsr_sigma", Min = 0.01, Max = 100, Step = 0.001, Tooltip = "Controls how early rescaling takes effect.\nLarger values take effect earlier.")] double tsrSigma = 1.0)
    {
        double variance = tsrSigma * tsrSigma;
        return model.WithPatches(p => p.WithPostCfg(ModelPatches.Stateless<Func<PostCfgArgs, float[]>>(a => GuidancePatches.TemporalScoreRescaling(a, tsrK, variance))));
    }
}

[Node(Category = "model/patch", DisplayName = "RenormCFG")]
public sealed class RenormCFG
{
    public ModelHandle Execute(ModelHandle model, [Input(Min = 0, Max = 100, Step = 0.01)] double cfgTrunc = 100, [Input(Min = 0, Max = 100, Step = 0.01)] double renormCfg = 1.0) =>
        model.WithPatches(p => p.WithCfgFunction(ModelPatches.Stateless<Func<CfgFunctionArgs, Task<float[]>>>(a => Task.FromResult(GuidancePatches.RenormCfg(a, cfgTrunc, renormCfg)))));
}

[Node(Category = "experimental", DisplayName = "Differential Diffusion")]
public sealed class DifferentialDiffusion
{
    /// <summary>Makes a soft inpainting mask a per-pixel denoise strength: a pixel is regenerated only while the run is past its mask value.</summary>
    public ModelHandle Execute(ModelHandle model, [Input(Optional = true, Min = 0, Max = 1, Step = 0.01)] double strength = 1.0) =>
        model.WithPatches(p => p with { DenoiseMask = (sigma, mask, sigmas, sampling) => GuidancePatches.DifferentialDiffusionMask(sigma, mask, sigmas, sampling, strength) });
}

[Node(Category = "model/sampling/guiders", DisplayName = "CFG Override",
    Description = "Override cfg to a fixed value over a [start, end] percent (sigma) range. With multiple overrides, the one nearest the sampler wins on overlap.")]
public sealed class CFGOverride
{
    public ModelHandle Execute(ModelHandle model, [Input(Min = 0, Max = 100, Step = 0.1)] double cfg = 1.0,
        [Input(Min = 0, Max = 1, Step = 0.001)] double startPercent = 0.0, [Input(Min = 0, Max = 1, Step = 0.001)] double endPercent = 1.0)
    {
        double hi = model.Sampling.PercentToSigma(startPercent), lo = model.Sampling.PercentToSigma(endPercent);
        return model.WithPatches(p => p with { CfgOverrides = [.. p.CfgOverrides, new CfgOverride(cfg, hi, lo)] });
    }
}

[Node(Category = "model/sampling/guiders", DisplayName = "Dual CFG Guider")]
public sealed class DualCFGGuider
{
    public GuiderChoice Execute(ModelHandle model, Conditioning cond1, Conditioning cond2, Conditioning negative,
        [Input(Min = 0, Max = 100, Step = 0.1)] double cfgConds = 8.0, [Input(Name = "cfg_cond2_negative", Min = 0, Max = 100, Step = 0.1)] double cfgCond2Negative = 8.0,
        [Input(Options = ["regular", "nested"])] string style = "regular") =>
        new DualCfgGuiderChoice(model, cond1, cond2, negative, cfgConds, cfgCond2Negative, style == "nested");
}

[Node(Category = "model/patch", DisplayName = "ModelSamplingDiscrete")]
public sealed class ModelSamplingDiscrete
{
    public static string[] SamplingTypes() => ["eps", "v_prediction", "lcm", "x0", "img_to_img", "img_to_img_flow"];

    /// <summary>
    /// Tells the sampler what the model predicts (noise, v, x0, LCM's boundary-conditioned eps on 50 distilled steps, ...), optionally with zero terminal SNR. The
    /// schedule is rebuilt from the checkpoint's own beta schedule, as ComfyUI rebuilds it from the model config.
    /// </summary>
    public ModelHandle Execute(ModelHandle model, [Input(OptionsProvider = nameof(SamplingTypes))] string sampling = "eps", bool zsnr = false)
    {
        var sigmas = model.Model?.Sampling.Sigmas ?? model.Sampling.Sigmas;
        if (zsnr) sigmas = sigmas.ZeroTerminalSnr();
        var prediction = sampling switch
        {
            "eps" => PredictionType.Eps,
            "v_prediction" => PredictionType.VPrediction,
            "lcm" => PredictionType.Lcm,
            "x0" => PredictionType.X0,
            "img_to_img" => PredictionType.ImgToImg,
            "img_to_img_flow" => PredictionType.ImgToImgFlow,
            _ => throw new ArgumentException($"Unknown sampling '{sampling}'. Known: {string.Join(", ", SamplingTypes())}.", nameof(sampling)),
        };
        if (prediction == PredictionType.Lcm) sigmas = sigmas.Distilled();
        return new ModelHandle(model.Model!, new ModelSampling(sigmas, prediction)) { Patches = model.Patches, Loras = model.Loras };
    }
}
