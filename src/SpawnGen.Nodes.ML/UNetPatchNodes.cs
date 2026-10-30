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
