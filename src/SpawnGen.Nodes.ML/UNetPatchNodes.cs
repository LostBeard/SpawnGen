using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.ML;

[Node("FreeU", Category = "model/patch/unet", DisplayName = "FreeU")]
public sealed class FreeU
{
    /// <summary>
    /// FreeU (arXiv 2309.11497) as ComfyUI does it: the decoder's backbone features at 4x and 2x the model's channel count are scaled by b1 / b2 (half of their
    /// channels) and the skip features by s1 / s2 (their low frequencies, via an FFT).
    /// </summary>
    public ModelHandle Execute(ModelHandle model,
        [Input(Min = 0, Max = 10, Step = 0.01)] double b1 = 1.1, [Input(Min = 0, Max = 10, Step = 0.01)] double b2 = 1.2,
        [Input(Min = 0, Max = 10, Step = 0.01)] double s1 = 0.9, [Input(Min = 0, Max = 10, Step = 0.01)] double s2 = 0.2) =>
        model.WithPatches(p => p with { UNet = p.UNet with { FreeU = new SdFreeU((float)b1, (float)b2, (float)s1, (float)s2, false) } });
}

[Node("FreeU_V2", Category = "model/patch/unet", DisplayName = "FreeU_V2")]
public sealed class FreeUV2
{
    /// <summary>FreeU v2: the backbone scale follows each picture's mean feature map (scaled to [0, 1]) instead of applying evenly.</summary>
    public ModelHandle Execute(ModelHandle model,
        [Input(Min = 0, Max = 10, Step = 0.01)] double b1 = 1.3, [Input(Min = 0, Max = 10, Step = 0.01)] double b2 = 1.4,
        [Input(Min = 0, Max = 10, Step = 0.01)] double s1 = 0.9, [Input(Min = 0, Max = 10, Step = 0.01)] double s2 = 0.2) =>
        model.WithPatches(p => p with { UNet = p.UNet with { FreeU = new SdFreeU((float)b1, (float)b2, (float)s1, (float)s2, true) } });
}

[Node("PerturbedAttentionGuidance", Category = "model/patch/unet", DisplayName = "PerturbedAttentionGuidance")]
public sealed class PerturbedAttentionGuidance
{
    /// <summary>
    /// Perturbed-Attention Guidance (arXiv 2403.17377), ComfyUI's version: after cfg the prompt is evaluated again with the middle block's self-attention returning its
    /// values, and the guided estimate moves away from that perturbed estimate by <c>scale</c>.
    /// </summary>
    public ModelHandle Execute(ModelHandle model, [Input(Min = 0, Max = 100, Step = 0.01)] double scale = 3.0) =>
        model.WithPatches(p => p.WithPostCfgAsync(ModelPatches.Stateless<Func<PostCfgArgs, Task<float[]>>>(async a =>
        {
            if (scale == 0 || a.Cond is not { } cond) return a.Denoised;
            var perturbed = await a.Context.EvaluateWith(cond, a.Context.UNet with { PerturbMiddleSelfAttention = true });
            var r = new float[a.Denoised.Length];
            for (int i = 0; i < r.Length; i++) r[i] = (float)(a.Denoised[i] + (a.CondDenoised[i] - perturbed[i]) * scale);
            return r;
        })));
}

[Node("PatchModelAddDownscale", Category = "model/patch/unet", DisplayName = "PatchModelAddDownscale (Kohya Deep Shrink)")]
public sealed class PatchModelAddDownscale
{
    public static string[] Methods() => ["bicubic", "nearest-exact", "bilinear", "area", "bislerp"];

    /// <summary>
    /// Kohya's deep shrink: while the step's sigma is inside [sigma(end_percent), sigma(start_percent)], the features after input block <c>block_number</c> are
    /// shrunk by <c>downscale_factor</c> (after its skip connection is kept, unless <c>downscale_after_skip</c> is off), and an output block whose features no
    /// longer meet their skip connection's size grows them back. It lets a model trained at 512 compose a larger picture.
    /// </summary>
    public ModelHandle Execute(ModelHandle model,
        [Input(Min = 1, Max = 32, Step = 1)] int block_number = 3,
        [Input(Min = 0.1, Max = 9.0, Step = 0.001)] double downscale_factor = 2.0,
        [Input(Min = 0, Max = 1, Step = 0.001)] double start_percent = 0.0,
        [Input(Min = 0, Max = 1, Step = 0.001)] double end_percent = 0.35,
        bool downscale_after_skip = true,
        [Input(OptionsProvider = nameof(Methods))] string downscale_method = "bicubic",
        [Input(OptionsProvider = nameof(Methods))] string upscale_method = "bicubic")
    {
        var shrink = new SdDeepShrink(block_number, downscale_factor, downscale_after_skip, Resize(downscale_method), Resize(upscale_method));
        var window = new DeepShrinkWindow(shrink, model.Sampling.PercentToSigma(start_percent), model.Sampling.PercentToSigma(end_percent));
        return model.WithPatches(p => p with { UNet = p.UNet with { DeepShrink = window } });
    }

    static SdResize Resize(string method) => method switch
    {
        "bicubic" => SdResize.Bicubic,
        "bilinear" => SdResize.Bilinear,
        "nearest-exact" => SdResize.NearestExact,
        _ => throw new NotSupportedException($"Deep shrink resizes the UNet's features on the GPU with bicubic, bilinear or nearest-exact; \"{method}\" is not available there yet."),
    };
}

[Node("HyperTile", Category = "model/patch/unet", DisplayName = "HyperTile")]
public sealed class HyperTile
{
    /// <summary>
    /// HyperTile: the self-attention of the first <c>max_depth</c> + 1 resolutions runs inside tiles of about <c>tile_size</c> pixels (doubled per depth with
    /// <c>scale_depth</c>), which speeds up large pictures. The tile counts are picked at random among the first <c>swap_size</c> - 1 candidates, as ComfyUI does,
    /// so 1 or 2 is deterministic.
    /// </summary>
    public ModelHandle Execute(ModelHandle model,
        [Input(Min = 1, Max = 2048)] int tile_size = 256,
        [Input(Min = 1, Max = 128)] int swap_size = 2,
        [Input(Min = 0, Max = 10)] int max_depth = 0,
        bool scale_depth = false) =>
        model.WithPatches(p => p with { UNet = p.UNet with { HyperTile = new SdHyperTile(tile_size, swap_size, max_depth, scale_depth) } });
}

[Node("SelfAttentionGuidance", Category = "experimental", DisplayName = "Self-Attention Guidance")]
public sealed class SelfAttentionGuidance
{
    /// <summary>
    /// Self-Attention Guidance (arXiv 2210.00939), ComfyUI's version: the unconditional estimate is blurred where the middle block's self-attention is strongest,
    /// the model is asked about that degraded picture without the prompt, and the guided estimate moves away from its answer by <c>scale</c>.
    /// </summary>
    public ModelHandle Execute(ModelHandle model,
        [Input(Min = -2.0, Max = 5.0, Step = 0.01)] double scale = 0.5,
        [Input(Min = 0.0, Max = 10.0, Step = 0.1)] double blur_sigma = 2.0) =>
        model.WithPatches(p => p.WithPostCfgAsync(ModelPatches.Stateless<Func<PostCfgArgs, Task<float[]>>>(async a =>
        {
            var ctx = a.Context;
            if (Math.Min(ctx.Height, ctx.Width) <= 4 || a.Uncond is not { } uncond) return a.Denoised; // too small to pad (ComfyUI skips it too)
            var mass = await ctx.ReadMiddleAttention()
                ?? throw new InvalidOperationException("Self-Attention Guidance found no attention from the unconditional pass; it needs a negative prompt and a single-file checkpoint.");
            var degraded = SdGuidance.SagBlur(a.UncondDenoised, ctx.Channels, ctx.Height, ctx.Width, mass, (float)blur_sigma, 1f);
            var noised = new float[degraded.Length];
            for (int i = 0; i < noised.Length; i++) noised[i] = degraded[i] + a.Input[i] - a.UncondDenoised[i];
            var sag = await ctx.EvaluateAt(uncond, noised);
            var r = new float[a.Denoised.Length];
            for (int i = 0; i < r.Length; i++) r[i] = (float)(a.Denoised[i] + (degraded[i] - sag[i]) * scale);
            return r;
        }), disableCfg1: true) with { RecordUncondMiddleAttention = true });
}
