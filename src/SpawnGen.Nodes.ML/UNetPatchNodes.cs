using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries and the papers the patches come from: FreeU (Si et al. 2023), HyperTile,
// Kohya's deep shrink, ToMe for SD (Bolya and Hoffman 2023), Perturbed-Attention Guidance (Ahn et al. 2024) and Self-Attention Guidance (Hong et al.
// 2023). The UNet work is SpawnDev.ILGPU.ML's (SdFreeU, SdHyperTile, SdDeepShrink, SdTokenMerge, SdGuidance.SagBlur, as its docs describe them);
// ComfyUIRenderParityTests compares each patch's render with ComfyUI's.

static class UNetPatch
{
    /// <summary>A copy of the model with its UNet options changed.</summary>
    public static ModelHandle With(ModelHandle model, Func<UNetOptions, UNetOptions> change) =>
        model.WithPatches(p => p with { UNet = change(p.UNet ?? UNetOptions.None) });
}

[Node("FreeU", Category = "model/patch/unet", DisplayName = "FreeU")]
public sealed class FreeU
{
    /// <summary>FreeU: the backbone features of the output blocks scaled by b1 / b2 and the skip features' low frequencies by s1 / s2.</summary>
    public ModelHandle Execute(ModelHandle model,
        [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double b1 = 1.1, [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double b2 = 1.2,
        [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double s1 = 0.9, [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double s2 = 0.2) =>
        UNetPatch.With(model, o => o with { FreeU = new SdFreeU((float)b1, (float)b2, (float)s1, (float)s2) });
}

[Node("FreeU_V2", Category = "model/patch/unet", DisplayName = "FreeU_V2")]
public sealed class FreeUV2
{
    /// <summary>FreeU's second version: the backbone scale follows each pixel's feature mean.</summary>
    public ModelHandle Execute(ModelHandle model,
        [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double b1 = 1.3, [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double b2 = 1.4,
        [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double s1 = 0.9, [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double s2 = 0.2) =>
        UNetPatch.With(model, o => o with { FreeU = new SdFreeU((float)b1, (float)b2, (float)s1, (float)s2, V2: true) });
}

[Node("HyperTile", Category = "model/patch/unet", DisplayName = "HyperTile")]
public sealed class HyperTile
{
    /// <summary>Self-attention computed in tiles of about <paramref name="tile_size"/> pixels at the outer levels.</summary>
    public ModelHandle Execute(ModelHandle model,
        [Input(Min = 1, Max = 2048)] int tile_size = 256, [Input(Min = 1, Max = 128)] int swap_size = 2,
        [Input(Min = 0, Max = 10)] int max_depth = 0, bool scale_depth = false) =>
        UNetPatch.With(model, o => o with { HyperTile = new SdHyperTile(tile_size, swap_size, max_depth, scale_depth) });
}

[Node("PatchModelAddDownscale", Category = "model/patch/unet", DisplayName = "PatchModelAddDownscale (Kohya Deep Shrink)")]
public sealed class PatchModelAddDownscale
{
    public static string[] Methods() => ["bicubic", "nearest-exact", "bilinear", "area", "bislerp"];

    static SdResize Resize(string method) => method switch
    {
        "bicubic" => SdResize.Bicubic,
        "nearest-exact" => SdResize.NearestExact,
        "bilinear" => SdResize.Bilinear,
        "area" => SdResize.Area,
        "bislerp" => SdResize.Bislerp,
        _ => throw new ArgumentException($"Unknown resize method '{method}'.", nameof(method)),
    };

    /// <summary>Deep shrink: the features after input block <paramref name="block_number"/> shrunk by the factor while sigma is in the window, grown back after.</summary>
    public ModelHandle Execute(ModelHandle model,
        [Input(Min = 1, Max = 32, Step = 1)] int block_number = 3,
        [Input(Min = 0.1, Max = 9.0, Step = 0.001)] double downscale_factor = 2.0,
        [Input(Min = 0.0, Max = 1.0, Step = 0.001)] double start_percent = 0.0,
        [Input(Min = 0.0, Max = 1.0, Step = 0.001)] double end_percent = 0.35,
        bool downscale_after_skip = true,
        [Input(OptionsProvider = nameof(Methods))] string downscale_method = "bicubic",
        [Input(OptionsProvider = nameof(Methods))] string upscale_method = "bicubic")
    {
        var shrink = new SdDeepShrink(block_number, downscale_factor, downscale_after_skip, Resize(downscale_method), Resize(upscale_method));
        var window = new DeepShrinkWindow(shrink, model.Sampling.PercentToSigma(start_percent), model.Sampling.PercentToSigma(end_percent));
        return UNetPatch.With(model, o => o with { DeepShrink = window });
    }
}

[Node("TomePatchModel", Category = "model/patch/unet", DisplayName = "TomePatchModel")]
public sealed class TomePatchModel
{
    /// <summary>Token merging in the self-attention: <paramref name="ratio"/> of the tokens merged (random destinations, drawn per run).</summary>
    public ModelHandle Execute(ModelHandle model, [Input(Min = 0.0, Max = 1.0, Step = 0.01)] double ratio = 0.3) =>
        UNetPatch.With(model, o => o with { TokenMerge = new SdTokenMerge((float)ratio) });
}

[Node("PerturbedAttentionGuidance", Category = "model/patch/unet", DisplayName = "PerturbedAttentionGuidance")]
public sealed class PerturbedAttentionGuidance
{
    /// <summary>
    /// After guidance, the prompt is evaluated once more with the middle block's self-attention replaced by the identity, and the result moves by
    /// (prompt's estimate - perturbed estimate) * scale.
    /// </summary>
    public ModelHandle Execute(ModelHandle model, [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double scale = 3.0) =>
        model.WithPatches(p => p.WithPostCfgAsync(ModelPatches.Stateless<Func<PostCfgArgs, Task<float[]>>>(async a =>
        {
            if (scale == 0 || a.Cond is null || a.CondDenoised is null) return a.Denoised;
            var perturbed = await a.Context.EvaluateWith(a.Cond, (a.Context.UNet ?? UNetOptions.None) with { PerturbMiddleSelfAttention = true });
            var r = new float[a.Denoised.Length];
            for (int i = 0; i < r.Length; i++) r[i] = a.Denoised[i] + (a.CondDenoised[i] - perturbed[i]) * (float)scale;
            return r;
        }), disableCfg1: true));
}

[Node("SelfAttentionGuidance", Category = "experimental", DisplayName = "Self-Attention Guidance")]
public sealed class SelfAttentionGuidance
{
    /// <summary>
    /// Self-Attention Guidance: the negative's estimate is blurred where the middle block's self-attention concentrates (SdGuidance.SagBlur), noised
    /// back to the current sigma with the negative's own noise, and evaluated again; the result moves by (blurred estimate - that) * scale.
    /// </summary>
    public ModelHandle Execute(ModelHandle model, [Input(Min = -2.0, Max = 5.0, Step = 0.01)] double scale = 0.5,
        [Input(Min = 0.0, Max = 10.0, Step = 0.1)] double blur_sigma = 2.0) =>
        model.WithPatches(p => p.WithPostCfgAsync(ModelPatches.Stateless<Func<PostCfgArgs, Task<float[]>>>(async a =>
        {
            if (!a.HasUncond || a.Uncond is null || a.UncondDenoised is null || a.Context.ReadMiddleAttention is null) return a.Denoised;
            var mass = await a.Context.ReadMiddleAttention();
            if (mass is null) return a.Denoised;
            var ctx = a.Context;
            int frame = ctx.Channels * ctx.Height * ctx.Width, pictures = a.UncondDenoised.Length / frame, massPer = mass.Length / Math.Max(pictures, 1);
            var degraded = new float[a.UncondDenoised.Length];
            var blurredAll = new float[a.UncondDenoised.Length];
            for (int b = 0; b < pictures; b++)
            {
                var x0 = a.UncondDenoised.AsSpan(b * frame, frame).ToArray();
                var blurred = SdGuidance.SagBlur(x0, ctx.Channels, ctx.Height, ctx.Width, mass.AsSpan(b * massPer, massPer).ToArray(), (float)blur_sigma);
                for (int i = 0; i < frame; i++) { blurredAll[b * frame + i] = blurred[i]; degraded[b * frame + i] = blurred[i] + a.Input[b * frame + i] - x0[i]; }
            }
            var sag = await ctx.EvaluateAt(a.Uncond, degraded);
            var r = new float[a.Denoised.Length];
            for (int i = 0; i < r.Length; i++) r[i] = a.Denoised[i] + (blurredAll[i] - sag[i]) * (float)scale;
            return r;
        }), disableCfg1: true) with { RecordUncondMiddleAttention = true });
}
