using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries, LatentOperationTests (the pre-cfg rule and its sigma window) and
// RescaleCfgTests; the rescale maths is GuidancePatches.RescaleCfg.

[Node("LatentApplyOperationCFG", Category = "model/latent/advanced/operations", DisplayName = "LatentApplyOperationCFG")]
public sealed class LatentApplyOperationCFG
{
    /// <summary>A copy of the model whose guidance (cond - uncond) goes through the operation between the two fractions of the schedule.</summary>
    public ModelHandle Execute(ModelHandle model, LatentOperation operation,
        [Input(Min = 0.0, Max = 1.0, Step = 0.001, Optional = true, Tooltip = "Start applying the operation at this fraction of the model's denoising schedule. 0 is the beginning.")] double startPercent = 0.0,
        [Input(Min = 0.0, Max = 1.0, Step = 0.001, Optional = true, Tooltip = "Stop applying the operation at this fraction of the model's denoising schedule. 1 is the end.")] double endPercent = 1.0)
    {
        double sigmaStart = model.Sampling.PercentToSigma(startPercent), sigmaEnd = model.Sampling.PercentToSigma(endPercent);
        return model.WithPatches(p => p.WithPreCfg(ModelPatches.Stateless<Func<PreCfgArgs, float[][]>>(a => PreCfg(a, operation, sigmaStart, sigmaEnd))));
    }

    /// <summary>
    /// With both the prompt's and the negative's outputs and sigma_end &lt;= sigma &lt;= sigma_start: the prompt's output becomes
    /// operation(cond - uncond) + uncond, the operation told the batch position of the picture; otherwise the outputs pass unchanged.
    /// </summary>
    public static float[][] PreCfg(PreCfgArgs args, LatentOperation operation, double sigmaStart, double sigmaEnd)
    {
        var outs = args.CondsOut;
        if (args.Sigma > sigmaStart || args.Sigma < sigmaEnd) return outs;
        if (outs.Length < 2 || args.CondsPresent.Length < 2 || !args.CondsPresent[0] || !args.CondsPresent[1]) return outs;
        var ctx = args.Context;
        float[] cond = outs[0], uncond = outs[1];
        int plane = ctx.Channels * ctx.Height * ctx.Width;
        var guidance = new float[cond.Length];
        for (int i = 0; i < guidance.Length; i++) guidance[i] = cond[i] - uncond[i];
        var changed = operation.Apply(new LatentBatch(guidance.Length / plane, ctx.Channels, ctx.Height, ctx.Width, guidance), ctx.BatchIndex).Data;
        var result = new float[cond.Length];
        for (int i = 0; i < result.Length; i++) result[i] = changed[i] + uncond[i];
        return [result, .. outs[1..]];
    }
}

[Node("RescaleCFG", Category = "model/patch", DisplayName = "RescaleCFG")]
public sealed class RescaleCFG
{
    /// <summary>A copy of the model whose classifier-free guidance is rescaled (Lin et al. 2023), by <paramref name="multiplier"/>.</summary>
    public ModelHandle Execute(ModelHandle model, [Input(Min = 0.0, Max = 1.0, Step = 0.01)] double multiplier = 0.7)
    {
        bool flow = model.Sampling.IsConst;
        return model.WithPatches(p => p.WithCfgFunction(ModelPatches.Stateless<Func<CfgFunctionArgs, Task<float[]>>>(
            a => Task.FromResult(GuidancePatches.RescaleCfg(a, multiplier, flow)))));
    }
}
