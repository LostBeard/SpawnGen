using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

[Node(Category = "model/latent/advanced/operations", DisplayName = "LatentApplyOperationCFG")]
public sealed class LatentApplyOperationCFG
{
    /// <summary>
    /// Applies a latent operation inside sampling: at each step, while the noise level is inside [start_percent, end_percent] of the schedule, the operation is applied
    /// to the guidance (prompt prediction minus negative prediction) before cfg scales it. Connect it between the model loader and the sampler.
    /// </summary>
    public ModelHandle Execute(ModelHandle model, LatentOperation operation,
        [Input(Optional = true, Min = 0, Max = 1, Step = 0.001, Tooltip = "Start applying the operation at this fraction of the model's denoising schedule. 0 is the beginning.")] double startPercent = 0.0,
        [Input(Optional = true, Min = 0, Max = 1, Step = 0.001, Tooltip = "Stop applying the operation at this fraction of the model's denoising schedule. 1 is the end.")] double endPercent = 1.0)
    {
        double sigmaStart = model.Sampling.PercentToSigma(startPercent), sigmaEnd = model.Sampling.PercentToSigma(endPercent);
        return model.WithPatches(p => p.WithPreCfg(ModelPatches.Stateless<Func<PreCfgArgs, float[][]>>(args => PreCfg(args, operation, sigmaStart, sigmaEnd))));
    }

    /// <summary>
    /// ComfyUI's pre_cfg_function for this node: inside [sigma_end, sigma_start] the operation is applied to the guidance (prompt minus negative estimate),
    /// or to the prompt's estimate alone when there is only one.
    /// </summary>
    internal static float[][] PreCfg(PreCfgArgs args, LatentOperation operation, double sigmaStart, double sigmaEnd)
    {
        if (args.Sigma > sigmaStart || args.Sigma < sigmaEnd) return args.CondsOut;
        var c = args.Context;
        var outs = args.CondsOut.ToArray();
        LatentBatch Wrap(float[] data) => new(1, c.Channels, c.Height, c.Width, data);
        if (outs.Length == 2)
        {
            var guidance = new float[outs[0].Length];
            for (int i = 0; i < guidance.Length; i++) guidance[i] = outs[0][i] - outs[1][i];
            var changed = operation.Apply(Wrap(guidance), c.BatchIndex).Data;
            var cond = new float[guidance.Length];
            for (int i = 0; i < cond.Length; i++) cond[i] = changed[i] + outs[1][i];
            outs[0] = cond;
        }
        else outs[0] = operation.Apply(Wrap(outs[0]), c.BatchIndex).Data;
        return outs;
    }
}
