using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

[Node(Category = "sampling", DisplayName = "KSampler")]
public sealed class KSampler
{
    public static string[] SamplerNames() => Samplers.Names;
    public static string[] SchedulerNames() => ModelSigmas.SchedulerNames;

    // Socket inputs first, then the widgets in ComfyUI's order with its defaults (so a new node is runnable as dropped on the canvas).
    public async Task<LatentBatch> ExecuteAsync(
        ModelHandle model, Conditioning positive, Conditioning negative, LatentBatch latentImage,
        NodeContext context, IServiceProvider services,
        long seed = 0,
        [Input(Min = 1, Max = 10000)] int steps = 20,
        [Input(Min = 0, Max = 100, Step = 0.1)] float cfg = 8f,
        [Input(OptionsProvider = nameof(SamplerNames))] string samplerName = "euler",
        [Input(OptionsProvider = nameof(SchedulerNames))] string scheduler = "normal",
        [Input(Min = 0, Max = 1, Step = 0.01)] float denoise = 1f)
    {
        if (denoise <= 0f) return latentImage; // nothing to denoise: the input passes through
        var acc = (await GpuHelpers.RuntimeAsync(services)).Accelerator;
        var sigmas = model.Checkpoint.Sigmas.ScheduleForDenoise(scheduler, steps, denoise);
        return await SamplingCore.RunAsync(acc, model.Checkpoint, positive, negative, latentImage, sigmas, new SamplingSettings(seed, cfg, samplerName, RescaleCfg: model.RescaleCfg), context);
    }
}

[Node(Category = "advanced/model", DisplayName = "RescaleCFG")]
public sealed class RescaleCFG
{
    /// <summary>
    /// Lets a high cfg keep colours and contrast: the guided prediction is rescaled towards the spread of the prompt's own prediction. <c>multiplier</c> 0 does nothing,
    /// 1 rescales fully; ComfyUI's default is 0.7. Connect it between the checkpoint and the sampler.
    /// </summary>
    public ModelHandle Execute(ModelHandle model, [Input(Min = 0, Max = 1, Step = 0.01)] float multiplier = 0.7f) =>
        new(model.Checkpoint) { RescaleCfg = multiplier };
}

[Node(Category = "sampling", DisplayName = "KSampler (Advanced)")]
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
        long noiseSeed = 0,
        [Input(Min = 1, Max = 10000)] int steps = 20,
        [Input(Min = 0, Max = 100, Step = 0.1)] float cfg = 8f,
        [Input(OptionsProvider = nameof(SamplerNames))] string samplerName = "euler",
        [Input(OptionsProvider = nameof(SchedulerNames))] string scheduler = "normal",
        [Input(Min = 0, Max = 10000)] int startAtStep = 0,
        [Input(Min = 0, Max = 10000)] int endAtStep = 10000,
        [Input(OptionsProvider = nameof(Leftover))] string returnWithLeftoverNoise = "disable")
    {
        var acc = (await GpuHelpers.RuntimeAsync(services)).Accelerator;
        var sigmas = Slice(model.Checkpoint.Sigmas.Schedule(scheduler, steps), startAtStep, endAtStep, returnWithLeftoverNoise == "disable");
        if (sigmas.Length < 2) return latentImage; // the start step is past the end of the schedule
        return await SamplingCore.RunAsync(acc, model.Checkpoint, positive, negative, latentImage, sigmas,
            new SamplingSettings(noiseSeed, cfg, samplerName, AddNoise: addNoise == "enable", RescaleCfg: model.RescaleCfg), context);
    }

    /// <summary>The part of a schedule a run covers: sigmas[start .. end]; unless leftover noise is wanted, the last one is set to 0 so the run ends clean.</summary>
    internal static float[] Slice(float[] sigmas, int start, int end, bool forceFullDenoise)
    {
        int steps = sigmas.Length - 1;
        if (start >= steps) return [];
        int last = Math.Min(end, steps);          // end_at_step counts steps, so the slice ends at sigma[last]
        var part = sigmas[start..(last + 1)];
        if (forceFullDenoise && last < steps) part[^1] = 0f; // stop early but finish the picture, like ComfyUI
        return part;
    }
}
