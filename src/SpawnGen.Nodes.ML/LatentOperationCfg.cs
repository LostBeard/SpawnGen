using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// What a pre-CFG function sees and may replace (ComfyUI's <c>sampler_pre_cfg_function</c> arguments): the model's denoised predictions for the prompt and for the
/// negative prompt of one picture, before cfg mixes them. When cfg is 1 the negative prompt is not evaluated and <see cref="Uncond"/> is zeros, as in ComfyUI.
/// </summary>
public sealed class PreCfgArgs
{
    public required float[] Cond { get; set; }
    public required float[] Uncond { get; set; }
    public required float Sigma { get; init; }
    public required float CondScale { get; init; }
    /// <summary>Where this picture sits in the latent batch (the sampler goes picture by picture).</summary>
    public required int BatchIndex { get; init; }
    public required int Channels { get; init; }
    public required int Height { get; init; }
    public required int Width { get; init; }
}

/// <summary>A model patch run at every model evaluation before cfg (ComfyUI's <c>set_model_sampler_pre_cfg_function</c>).</summary>
public delegate void PreCfgFunction(PreCfgArgs args);

internal static class PreCfg
{
    /// <summary>
    /// Runs the model's pre-CFG functions on one picture. The sampler has the UNet's noise predictions; ComfyUI's functions work on denoised predictions
    /// (<c>x - sigma * eps</c>), so they are converted there and back. <paramref name="epsUncond"/> is null when cfg is 1 (not evaluated).
    /// </summary>
    public static (float[] Cond, float[]? Uncond) Apply(IReadOnlyList<PreCfgFunction> functions, float[] sample, float[] epsCond, float[]? epsUncond,
        float sigma, float cfg, int batchIndex, int channels, int height, int width)
    {
        int n = sample.Length;
        var cond = new float[n];
        var uncond = new float[n];
        for (int i = 0; i < n; i++)
        {
            cond[i] = sample[i] - sigma * epsCond[i];
            if (epsUncond is not null) uncond[i] = sample[i] - sigma * epsUncond[i];
        }
        var args = new PreCfgArgs
        {
            Cond = cond, Uncond = uncond, Sigma = sigma, CondScale = cfg, BatchIndex = batchIndex, Channels = channels, Height = height, Width = width,
        };
        foreach (var f in functions) f(args);
        var outCond = new float[n];
        for (int i = 0; i < n; i++) outCond[i] = (sample[i] - args.Cond[i]) / sigma;
        if (epsUncond is null) return (outCond, null);
        var outUncond = new float[n];
        for (int i = 0; i < n; i++) outUncond[i] = (sample[i] - args.Uncond[i]) / sigma;
        return (outCond, outUncond);
    }
}

[Node(Category = "model/latent/advanced/operations", DisplayName = "LatentApplyOperationCFG")]
public sealed class LatentApplyOperationCFG
{
    /// <summary>
    /// Applies a latent operation inside sampling: at each step, while the noise level is inside [start_percent, end_percent] of the schedule, the operation is applied
    /// to the guidance (prompt prediction minus negative prediction) before cfg scales it. Connect it between the model loader and the sampler.
    /// </summary>
    public ModelHandle Execute(ModelHandle model, LatentOperation operation,
        [Input(Min = 0, Max = 1, Step = 0.001, Tooltip = "Start applying the operation at this fraction of the model's denoising schedule. 0 is the beginning.")] double startPercent = 0.0,
        [Input(Min = 0, Max = 1, Step = 0.001, Tooltip = "Stop applying the operation at this fraction of the model's denoising schedule. 1 is the end.")] double endPercent = 1.0) =>
        new(model.Checkpoint) { RescaleCfg = model.RescaleCfg, PreCfg = [.. model.PreCfg, Function(model.Checkpoint.Sigmas, operation, startPercent, endPercent)] };

    /// <summary>ComfyUI's <c>pre_cfg_function</c> for this node, with the percents turned into sigmas the way the model's sampling does.</summary>
    internal static PreCfgFunction Function(ModelSigmas sigmas, LatentOperation operation, double startPercent, double endPercent)
    {
        float sigmaStart = SamplingCore.PercentToSigma((float)startPercent, sigmas), sigmaEnd = SamplingCore.PercentToSigma((float)endPercent, sigmas);
        return args =>
        {
            if (args.Sigma > sigmaStart || args.Sigma < sigmaEnd) return;
            int n = args.Cond.Length;
            var guidance = new float[n];
            for (int i = 0; i < n; i++) guidance[i] = args.Cond[i] - args.Uncond[i];
            var changed = operation.Apply(new LatentBatch(1, args.Channels, args.Height, args.Width, guidance), args.BatchIndex);
            var cond = new float[n];
            for (int i = 0; i < n; i++) cond[i] = changed.Data[i] + args.Uncond[i];
            args.Cond = cond;
        };
    }
}
