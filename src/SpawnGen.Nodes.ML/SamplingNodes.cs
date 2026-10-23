using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

[Node(Category = "model/sampling", DisplayName = "KSampler")]
public sealed class KSampler
{
    public static string[] SamplerNames() => Samplers.Names;
    public static string[] SchedulerNames() => ModelSigmas.SchedulerNames;

    // Socket inputs first, then the widgets in ComfyUI's order with its defaults (so a new node is runnable as dropped on the canvas).
    public async Task<LatentBatch> ExecuteAsync(
        ModelHandle model, Conditioning positive, Conditioning negative, LatentBatch latentImage,
        NodeContext context, IServiceProvider services,
        [Input(Min = 0, Max = 18446744073709551615d)] long seed = 0,
        [Input(Min = 1, Max = 10000)] int steps = 20,
        [Input(Min = 0, Max = 100, Step = 0.1)] double cfg = 8.0,
        [Input(OptionsProvider = nameof(SamplerNames))] string samplerName = "euler",
        [Input(OptionsProvider = nameof(SchedulerNames))] string scheduler = "simple",
        [Input(Min = 0, Max = 1, Step = 0.01)] double denoise = 1.0)
    {
        var sigmas = Sigmas(model.Sampling.Sigmas, samplerName, scheduler, steps, denoise);
        if (sigmas.Length == 0) return latentImage; // nothing to denoise: the input passes through
        var acc = (await GpuHelpers.RuntimeAsync(services)).Accelerator;
        return await SamplingCore.RunAsync(acc, new CfgGuider(model, positive, negative, cfg), latentImage, sigmas, new SamplingSettings(seed, samplerName), context);
    }

    /// <summary>
    /// ComfyUI's KSampler.calculate_sigmas: the scheduler's sigmas, computed one step longer with the second to last dropped for dpm_2, dpm_2_ancestral and uni_pc.
    /// </summary>
    internal static float[] Schedule(ModelSigmas model, string sampler, string scheduler, int steps)
    {
        if (!Samplers.DiscardsPenultimateSigma(sampler)) return model.Schedule(scheduler, steps);
        var s = model.Schedule(scheduler, steps + 1);
        return s.Length < 2 ? s : [.. s[..^2], s[^1]];
    }

    /// <summary>KSampler.set_steps: the full schedule above denoise 0.9999, none at 0, else the last steps + 1 sigmas of an int(steps / denoise) step schedule.</summary>
    internal static float[] Sigmas(ModelSigmas model, string sampler, string scheduler, int steps, double denoise)
    {
        if (denoise > 0.9999) return Schedule(model, sampler, scheduler, steps);
        if (denoise <= 0.0) return [];
        var full = Schedule(model, sampler, scheduler, (int)(steps / denoise));
        return full[Math.Max(0, full.Length - (steps + 1))..];
    }
}

[Node(Category = "model/patch", DisplayName = "RescaleCFG")]
public sealed class RescaleCFG
{
    /// <summary>
    /// Lets a high cfg keep colours and contrast: the guided prediction is rescaled towards the spread of the prompt's own prediction. <c>multiplier</c> 0 does nothing,
    /// 1 rescales fully; ComfyUI's default is 0.7. Connect it between the checkpoint and the sampler.
    /// </summary>
    public ModelHandle Execute(ModelHandle model, [Input(Min = 0, Max = 1, Step = 0.01)] double multiplier = 0.7)
    {
        bool isFlow = model.Sampling.IsConst;
        return model.WithPatches(p => p.WithCfgFunction(ModelPatches.Stateless<Func<CfgFunctionArgs, Task<float[]>>>(
            a => Task.FromResult(GuidancePatches.RescaleCfg(a, multiplier, isFlow)))));
    }
}

[Node(Category = "model/sampling", DisplayName = "KSampler (Advanced)")]
public sealed class KSamplerAdvanced
{
    public static string[] Toggle() => ["enable", "disable"];
    public static string[] Leftover() => ["disable", "enable"];
    public static string[] SamplerNames() => Samplers.Names;
    public static string[] SchedulerNames() => ModelSigmas.SchedulerNames;

    /// <summary>
    /// Like KSampler, but you choose whether noise is added, which steps of the schedule to run (<c>start_at_step</c> to <c>end_at_step</c>), and whether the
    /// last step cleans the picture completely or leaves its remaining noise for the next sampler (<c>return_with_leftover_noise</c>).
    /// </summary>
    public async Task<LatentBatch> ExecuteAsync(
        ModelHandle model, Conditioning positive, Conditioning negative, LatentBatch latentImage,
        NodeContext context, IServiceProvider services,
        [Input(OptionsProvider = nameof(Toggle))] string addNoise = "enable",
        [Input(Min = 0, Max = 18446744073709551615d)] long noiseSeed = 0,
        [Input(Min = 1, Max = 10000)] int steps = 20,
        [Input(Min = 0, Max = 100, Step = 0.1)] double cfg = 8.0,
        [Input(OptionsProvider = nameof(SamplerNames))] string samplerName = "euler",
        [Input(OptionsProvider = nameof(SchedulerNames))] string scheduler = "simple",
        [Input(Min = 0, Max = 10000)] int startAtStep = 0,
        [Input(Min = 0, Max = 10000)] int endAtStep = 10000,
        [Input(OptionsProvider = nameof(Leftover))] string returnWithLeftoverNoise = "disable")
    {
        var sigmas = Slice(KSampler.Schedule(model.Sampling.Sigmas, samplerName, scheduler, steps), startAtStep, endAtStep, returnWithLeftoverNoise == "disable");
        if (sigmas.Length == 0) return latentImage; // the start step is past the end of the schedule
        var acc = (await GpuHelpers.RuntimeAsync(services)).Accelerator;
        return await SamplingCore.RunAsync(acc, new CfgGuider(model, positive, negative, cfg), latentImage, sigmas,
            new SamplingSettings(noiseSeed, samplerName, AddNoise: addNoise == "enable"), context);
    }

    /// <summary>
    /// ComfyUI's KSampler.sample slicing: an end step before the last cuts the schedule at sigma[end] (set to 0 unless leftover noise is wanted), then the run starts at
    /// sigma[start]; a start at or past the end of what remains means nothing runs.
    /// </summary>
    internal static float[] Slice(float[] sigmas, int start, int end, bool forceFullDenoise)
    {
        var s = (float[])sigmas.Clone();
        if (end < s.Length - 1)
        {
            s = s[..(end + 1)];
            if (forceFullDenoise) s[^1] = 0f;
        }
        return start < s.Length - 1 ? s[start..] : [];
    }
}
