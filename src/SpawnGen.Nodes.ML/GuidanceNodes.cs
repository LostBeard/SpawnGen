using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries and SamplerParityTests / SamplingNodeSchemaTests /
// ContinuousSamplingTests (each patch and guider against ComfyUI's own results); the guidance maths is GuidancePatches' and the guiders' (Guidance).

static class Patch
{
    public static Func<T> Once<T>(T function) => ModelPatches.Stateless(function);

    public static ModelHandle WithSampling(ModelHandle model, ModelSampling sampling) =>
        new(model.Model, sampling) { Patches = model.Patches, Weights = model.Weights };
}

[Node("APG", Category = "model/sampling/custom", DisplayName = "Adaptive Projected Guidance")]
public sealed class APG
{
    /// <summary>Adaptive Projected Guidance (Sadat et al. 2024) as a pre-cfg patch; its momentum is kept per run.</summary>
    public ModelHandle Execute(ModelHandle model,
        [Input(Min = -10.0, Max = 10.0, Step = 0.01, Tooltip = "Controls the scale of the parallel guidance vector. Default CFG behavior at a setting of 1.")] double eta = 1.0,
        [Input(Min = 0.0, Max = 50.0, Step = 0.1, Tooltip = "Normalize guidance vector to this value, normalization disable at a setting of 0.")] double normThreshold = 5.0,
        [Input(Min = -5.0, Max = 1.0, Step = 0.01, Tooltip = "Controls a running average of guidance during diffusion, disabled at a setting of 0.")] double momentum = 0.0) =>
        model.WithPatches(p => p.WithPreCfg(() => GuidancePatches.Apg(eta, normThreshold, momentum)));
}

[Node("CFGNorm", Category = "advanced/guidance", DisplayName = "CFGNorm", OutputNames = ["patched_model"])]
public sealed class CFGNorm
{
    public ModelHandle Execute(ModelHandle model, [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double strength = 1.0,
        [Input(Tooltip = "If true, rescale the combined noise BEFORE the sampler's CFG combine, without clamping (can amplify). Matches the norm-scaled CFG used by models like Lens. Default false keeps the original post-CFG x0-space attenuate-only behavior.")]
        bool preCfg = false) =>
        preCfg
            ? model.WithPatches(p => p.WithCfgFunction(Patch.Once<Func<CfgFunctionArgs, Task<float[]>>>(a => Task.FromResult(GuidancePatches.CfgNormPre(a, strength)))))
            : model.WithPatches(p => p.WithPostCfg(Patch.Once<Func<PostCfgArgs, float[]>>(a => GuidancePatches.CfgNormPost(a, strength))));
}

[Node("CFGOverride", Category = "model/sampling/guiders", DisplayName = "CFG Override",
    Description = "Override cfg to a fixed value over a [start, end] percent (sigma) range. With multiple overrides, the one nearest the sampler wins on overlap.")]
public sealed class CFGOverride
{
    public ModelHandle Execute(ModelHandle model, [Input(Min = 0.0, Max = 100.0, Step = 0.1)] double cfg = 1.0,
        [Input(Min = 0.0, Max = 1.0, Step = 0.001)] double startPercent = 0.0, [Input(Min = 0.0, Max = 1.0, Step = 0.001)] double endPercent = 1.0)
    {
        var o = new CfgOverride(cfg, model.Sampling.PercentToSigma(startPercent), model.Sampling.PercentToSigma(endPercent));
        return model.WithPatches(p => p with { CfgOverrides = [.. p.CfgOverrides ?? [], o] });
    }
}

[Node("CFGZeroStar", Category = "advanced/guidance", DisplayName = "CFGZeroStar", OutputNames = ["patched_model"])]
public sealed class CFGZeroStar
{
    public ModelHandle Execute(ModelHandle model) =>
        model.WithPatches(p => p.WithPostCfg(Patch.Once<Func<PostCfgArgs, float[]>>(GuidancePatches.CfgZeroStar)));
}

[Node("DifferentialDiffusion", Category = "experimental", DisplayName = "Differential Diffusion")]
public sealed class DifferentialDiffusion
{
    /// <summary>The inpainting mask turned into a per-step threshold (Levin and Fried 2023): each area is denoised for as long as its mask value says.</summary>
    public ModelHandle Execute(ModelHandle model, [Input(Min = 0.0, Max = 1.0, Step = 0.01, Optional = true)] double strength = 1.0) =>
        model.WithPatches(p => p with { DenoiseMask = (sigma, mask, stepSigmas, sampling) => GuidancePatches.DifferentialDiffusionMask(sigma, mask, stepSigmas, sampling, strength) });
}

[Node("DualCFGGuider", Category = "model/sampling/guiders", DisplayName = "Dual CFG Guider")]
public sealed class DualCFGGuider
{
    public GuiderChoice Execute(ModelHandle model, Conditioning cond1, Conditioning cond2, Conditioning negative,
        [Input(Min = 0.0, Max = 100.0, Step = 0.1)] double cfgConds = 8.0,
        [Input(Name = "cfg_cond2_negative", Min = 0.0, Max = 100.0, Step = 0.1)] double cfgCond2Negative = 8.0,
        [Input(Options = ["regular", "nested"])] string style = "regular") =>
        new DualCfgGuiderChoice(model, cond1, cond2, negative, cfgConds, cfgCond2Negative, nested: style == "nested");
}

[Node("Epsilon Scaling", Category = "model/patch/unet", DisplayName = "Epsilon Scaling")]
public sealed class EpsilonScaling
{
    public ModelHandle Execute(ModelHandle model, [Input(Min = 0.5, Max = 1.5, Step = 0.001)] double scalingFactor = 1.005) =>
        model.WithPatches(p => p.WithPostCfg(Patch.Once<Func<PostCfgArgs, float[]>>(a => GuidancePatches.EpsilonScaling(a, scalingFactor))));
}

[Node("FreSca", Category = "experimental", DisplayName = "FreSca", Description = "Applies frequency-dependent scaling to the guidance")]
public sealed class FreSca
{
    public ModelHandle Execute(ModelHandle model,
        [Input(Min = 0.0, Max = 10.0, Step = 0.01, Tooltip = "Scaling factor for low-frequency components")] double scaleLow = 1.0,
        [Input(Min = 0.0, Max = 10.0, Step = 0.01, Tooltip = "Scaling factor for high-frequency components")] double scaleHigh = 1.25,
        [Input(Min = 1, Max = 10000, Step = 1, Tooltip = "Number of frequency indices around center to consider as low-frequency")] int freqCutoff = 20) =>
        model.WithPatches(p => p.WithPreCfg(Patch.Once<Func<PreCfgArgs, float[][]>>(a => GuidancePatches.FreSca(a, scaleLow, scaleHigh, freqCutoff))));
}

[Node("Mahiro", Category = "experimental", DisplayName = "Positive-Biased Guidance", OutputNames = ["patched_model"],
    Description = "Modify the guidance to scale more on the 'direction' of the positive prompt rather than the difference between the negative prompt.")]
public sealed class Mahiro
{
    public ModelHandle Execute(ModelHandle model) =>
        model.WithPatches(p => p.WithPostCfg(Patch.Once<Func<PostCfgArgs, float[]>>(GuidancePatches.Mahiro)));
}

[Node("ModelSamplingDiscrete", Category = "model/patch", DisplayName = "ModelSamplingDiscrete")]
public sealed class ModelSamplingDiscrete
{
    public static string[] SamplingTypes() => ["eps", "v_prediction", "lcm", "x0", "img_to_img", "img_to_img_flow"];

    /// <summary>The model's sampling replaced: SD's discrete schedule (zero terminal SNR when asked), LCM's distilled table, or another prediction type.</summary>
    public ModelHandle Execute(ModelHandle model, [Input(OptionsProvider = nameof(SamplingTypes))] string sampling = "eps", bool zsnr = false)
    {
        var sigmas = ModelSigmas.LinearSchedule();
        if (zsnr) sigmas = sigmas.ZeroTerminalSnr();
        var s = sampling switch
        {
            "eps" => new ModelSampling(sigmas, PredictionType.Eps),
            "v_prediction" => new ModelSampling(sigmas, PredictionType.VPrediction),
            "lcm" => new ModelSampling(sigmas.Distilled(), PredictionType.Lcm),
            "x0" => new ModelSampling(sigmas, PredictionType.X0),
            "img_to_img" => new ModelSampling(sigmas, PredictionType.ImgToImg),
            "img_to_img_flow" => new ModelSampling(sigmas, PredictionType.ImgToImgFlow),
            _ => throw new ArgumentException($"Unknown sampling '{sampling}'. The choices are: {string.Join(", ", SamplingTypes())}.", nameof(sampling)),
        };
        return Patch.WithSampling(model, s);
    }
}

[Node("ModelSamplingContinuousEDM", Category = "model/patch", DisplayName = "ModelSamplingContinuousEDM")]
public sealed class ModelSamplingContinuousEDM
{
    public static string[] SamplingTypes() => ["v_prediction", "edm", "edm_playground_v2.5", "eps", "cosmos_rflow"];

    /// <summary>A continuous EDM schedule between the two sigmas (Karras et al. 2022); "edm" predicts with sigma_data 0.5.</summary>
    public ModelHandle Execute(ModelHandle model, [Input(OptionsProvider = nameof(SamplingTypes))] string sampling,
        [Input(Min = 0.0, Max = 1000.0, Step = 0.001, Default = 120.0)] double sigmaMax,
        [Input(Min = 0.0, Max = 1000.0, Step = 0.001, Default = 0.002)] double sigmaMin)
    {
        var sigmas = ModelSigmas.Continuous(sigmaMin, sigmaMax, v: false);
        var s = sampling switch
        {
            "v_prediction" => new ModelSampling(sigmas, PredictionType.VPrediction),
            "edm" => new ModelSampling(sigmas, PredictionType.Edm, 0.5),
            "eps" => new ModelSampling(sigmas, PredictionType.Eps),
            "edm_playground_v2.5" or "cosmos_rflow" => throw new NotSupportedException($"Sampling '{sampling}' needs a latent format or prediction SpawnGen does not have."),
            _ => throw new ArgumentException($"Unknown sampling '{sampling}'.", nameof(sampling)),
        };
        return Patch.WithSampling(model, s);
    }
}

/// <summary>ModelSamplingSD3: a flow-matching model's sampling with the SD3 timestep shift (Wan 2.x uses it with shift 8, SD3 with 3).</summary>
[Node("ModelSamplingSD3", Category = "model/patch/stable diffusion", DisplayName = "ModelSamplingSD3")]
public sealed class ModelSamplingSD3
{
    public ModelHandle Execute(ModelHandle model, [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double shift = 3.0) =>
        Patch.WithSampling(model, new ModelSampling(ModelSigmas.Flow(ModelSigmas.Sd3Shift(shift), 1000), PredictionType.Const));
}

/// <summary>ModelSamplingAuraFlow: the same shift with the model told sigma itself as its timestep; img_to_img_velocity for image-to-image flow models.</summary>
[Node("ModelSamplingAuraFlow", Category = "model/patch", DisplayName = "ModelSamplingAuraFlow")]
public sealed class ModelSamplingAuraFlow
{
    public static string[] SamplingTypes() => ["flow", "img_to_img_velocity"];

    public ModelHandle Execute(ModelHandle model, [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double shift = 1.73,
        [Input(Optional = true, OptionsProvider = nameof(SamplingTypes))] string sampling = "flow") =>
        Patch.WithSampling(model, new ModelSampling(ModelSigmas.Flow(ModelSigmas.Sd3Shift(shift), 1),
            sampling == "img_to_img_velocity" ? PredictionType.ImgToImgFlow : PredictionType.Const));
}

/// <summary>ModelSamplingFlux: Flux's resolution-dependent shift (more for bigger pictures).</summary>
[Node("ModelSamplingFlux", Category = "model/patch/flux", DisplayName = "ModelSamplingFlux")]
public sealed class ModelSamplingFlux
{
    public ModelHandle Execute(ModelHandle model,
        [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double maxShift = 1.15,
        [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double baseShift = 0.5,
        [Input(Min = 16, Max = 16384, Step = 8)] int width = 1024,
        [Input(Min = 16, Max = 16384, Step = 8)] int height = 1024) =>
        Patch.WithSampling(model, new ModelSampling(ModelSigmas.Flow(ModelSigmas.FluxShift(maxShift, baseShift, width, height), 1, entries: 10000), PredictionType.Const));
}

[Node("ModelSamplingContinuousV", Category = "model/patch", DisplayName = "ModelSamplingContinuousV")]
public sealed class ModelSamplingContinuousV
{
    public static string[] SamplingTypes() => ["v_prediction"];

    public ModelHandle Execute(ModelHandle model, [Input(OptionsProvider = nameof(SamplingTypes))] string sampling,
        [Input(Min = 0.0, Max = 1000.0, Step = 0.001, Default = 500.0)] double sigmaMax,
        [Input(Min = 0.0, Max = 1000.0, Step = 0.001, Default = 0.03)] double sigmaMin)
    {
        if (sampling != "v_prediction") throw new ArgumentException($"Unknown sampling '{sampling}'.", nameof(sampling));
        return Patch.WithSampling(model, new ModelSampling(ModelSigmas.Continuous(sigmaMin, sigmaMax, v: true), PredictionType.VPrediction));
    }
}

[Node("PerpNeg", Category = "experimental", DisplayName = "Perp-Neg (DEPRECATED by Perp-Neg Guider)")]
public sealed class PerpNeg
{
    /// <summary>Perp-Neg (Armandpour et al. 2023) as a cfg function: the negative is projected out along the prompt, against the empty conditioning.</summary>
    public ModelHandle Execute(ModelHandle model, Conditioning emptyConditioning, [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double negScale = 1.0) =>
        model.WithPatches(p => p.WithCfgFunction(Patch.Once<Func<CfgFunctionArgs, Task<float[]>>>(async a =>
        {
            var empty = await a.Context.EvaluateUnscheduled(emptyConditioning);
            var nocond = new float[a.Input.Length];
            for (int i = 0; i < nocond.Length; i++) nocond[i] = a.Input[i] - empty[i];
            return GuidancePatches.PerpNeg(a.Cond, a.Uncond, nocond, negScale, a.CondScale);
        }), disableCfg1: true));
}

[Node("PerpNegGuider", Category = "experimental", DisplayName = "Perp-Neg Guider")]
public sealed class PerpNegGuider
{
    public GuiderChoice Execute(ModelHandle model, Conditioning positive, Conditioning negative, Conditioning emptyConditioning,
        [Input(Min = 0.0, Max = 100.0, Step = 0.1)] double cfg = 8.0, [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double negScale = 1.0) =>
        new PerpNegGuiderChoice(model, positive, negative, emptyConditioning, cfg, negScale);
}

[Node("RenormCFG", Category = "model/patch", DisplayName = "RenormCFG")]
public sealed class RenormCFG
{
    public ModelHandle Execute(ModelHandle model, [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double cfgTrunc = 100.0,
        [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double renormCfg = 1.0) =>
        model.WithPatches(p => p.WithCfgFunction(Patch.Once<Func<CfgFunctionArgs, Task<float[]>>>(a => Task.FromResult(GuidancePatches.RenormCfg(a, cfgTrunc, renormCfg)))));
}

[Node("TCFG", Category = "advanced/guidance", DisplayName = "Tangential Damping CFG", OutputNames = ["patched_model"],
    Description = "TCFG – Tangential Damping CFG (2503.18137)\n\nRefine the uncond (negative) to align with the cond (positive) for improving quality.")]
public sealed class TCFG
{
    public ModelHandle Execute(ModelHandle model) =>
        model.WithPatches(p => p.WithPreCfg(Patch.Once<Func<PreCfgArgs, float[][]>>(GuidancePatches.Tcfg)));
}

[Node("TemporalScoreRescaling", Category = "model/patch/unet", DisplayName = "TSR - Temporal Score Rescaling", OutputNames = ["patched_model"],
    Description = "[Post-CFG Function]\nTSR - Temporal Score Rescaling (2510.01184)\n\nRescaling the model's score or noise to steer the sampling diversity.\n")]
public sealed class TemporalScoreRescaling
{
    public ModelHandle Execute(ModelHandle model,
        [Input(Name = "tsr_k", Min = 0.01, Max = 100.0, Step = 0.001, Tooltip = "Controls the rescaling strength.\nLower k produces more detailed results; higher k produces smoother results in image generation. Setting k = 1 disables rescaling.")] double tsrK = 0.95,
        [Input(Name = "tsr_sigma", Min = 0.01, Max = 100.0, Step = 0.001, Tooltip = "Controls how early rescaling takes effect.\nLarger values take effect earlier.")] double tsrSigma = 1.0) =>
        model.WithPatches(p => p.WithPostCfg(Patch.Once<Func<PostCfgArgs, float[]>>(a => GuidancePatches.TemporalScoreRescaling(a, tsrK, tsrSigma * tsrSigma))));
}

[Node("VideoLinearCFGGuidance", Category = "model/sampling/guiders", DisplayName = "Video Linear CFG Guidance")]
public sealed class VideoLinearCFGGuidance
{
    public ModelHandle Execute(ModelHandle model, [Input(Min = 0.0, Max = 100.0, Step = 0.5)] double minCfg = 1.0) =>
        model.WithPatches(p => p.WithCfgFunction(Patch.Once<Func<CfgFunctionArgs, Task<float[]>>>(a => Task.FromResult(GuidancePatches.VideoLinearCfg(a, minCfg)))));
}

[Node("VideoTriangleCFGGuidance", Category = "model/sampling/guiders", DisplayName = "Video Triangle CFG Guidance")]
public sealed class VideoTriangleCFGGuidance
{
    public ModelHandle Execute(ModelHandle model, [Input(Min = 0.0, Max = 100.0, Step = 0.5)] double minCfg = 1.0) =>
        model.WithPatches(p => p.WithCfgFunction(Patch.Once<Func<CfgFunctionArgs, Task<float[]>>>(a => Task.FromResult(GuidancePatches.VideoTriangleCfg(a, minCfg)))));
}
