using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries, Fixtures/sampling/reference.json (the KSampler sigmas: denoise and the
// samplers that take one step more and drop the penultimate sigma, whose list was read from a running ComfyUI) and SamplingSetupTests (the
// advanced sampler's slicing).

[Node("KSampler", Category = "model/sampling", DisplayName = "KSampler",
    Description = "Uses the provided model, positive and negative conditioning to denoise the latent image.")]
public sealed class KSampler
{
    /// <summary>Samplers that run one step more and drop the penultimate sigma (measured).</summary>
    static readonly HashSet<string> DropPenultimate = ["dpm_2", "dpm_2_ancestral", "uni_pc", "uni_pc_bh2"];

    public static string[] SamplerNames() => [.. Samplers.Names];

    public static string[] SchedulerNames() => ModelSigmas.SchedulerNames;

    public async Task<LatentBatch> ExecuteAsync(
        [Input(Tooltip = "The model used for denoising the input latent.")] ModelHandle model,
        [Input(Tooltip = "The conditioning describing the attributes you want to include in the image.")] Conditioning positive,
        [Input(Tooltip = "The conditioning describing the attributes you want to exclude from the image.")] Conditioning negative,
        [Input(Tooltip = "The latent image to denoise.")] LatentBatch latentImage,
        NodeContext context, IServiceProvider services,
        [Input(Min = 0, Max = 18446744073709551615d, Tooltip = "The random seed used for creating the noise.")] long seed = 0,
        [Input(Min = 1, Max = 10000, Tooltip = "The number of steps used in the denoising process.")] int steps = 20,
        [Input(Min = 0.0, Max = 100.0, Step = 0.1, Tooltip = "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality.")] double cfg = 8.0,
        [Input(OptionsProvider = nameof(SamplerNames), Tooltip = "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output.")] string samplerName = "euler",
        [Input(OptionsProvider = nameof(SchedulerNames), Tooltip = "The scheduler controls how noise is gradually removed to form the image.")] string scheduler = "simple",
        [Input(Min = 0.0, Max = 1.0, Step = 0.01, Tooltip = "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling.")] double denoise = 1.0)
    {
        var sigmas = Sigmas(model.Sampling.Sigmas, samplerName, scheduler, steps, denoise);
        return await Run(model, positive, negative, latentImage, context, services, cfg, sigmas, new SamplingSettings(seed, samplerName));
    }

    internal static async Task<LatentBatch> Run(ModelHandle model, Conditioning positive, Conditioning negative, LatentBatch latent, NodeContext context,
        IServiceProvider services, double cfg, float[] sigmas, SamplingSettings settings)
    {
        if (model.Model is { } m) await m.UseWeightsAsync(model.Weights, context.CancellationToken);
        var runtime = await GpuHelpers.RuntimeAsync(services);
        return await SamplingCore.RunAsync(runtime.Accelerator, new CfgGuider(model, positive, negative, cfg), latent, sigmas, settings, context);
    }

    /// <summary>The scheduler's sigmas for the sampler: one step more with the penultimate sigma dropped for the samplers that need it.</summary>
    public static float[] Schedule(ModelSigmas model, string sampler, string scheduler, int steps)
    {
        if (!DropPenultimate.Contains(sampler)) return model.Schedule(scheduler, steps);
        var s = model.Schedule(scheduler, steps + 1);
        return [.. s[..^2], s[^1]];
    }

    /// <summary>
    /// The sigmas a run uses: the whole schedule when denoise is above 0.9999, none at 0 or below, otherwise the last steps + 1 sigmas of the
    /// schedule for int(steps / denoise) steps.
    /// </summary>
    public static float[] Sigmas(ModelSigmas model, string sampler, string scheduler, int steps, double denoise)
    {
        if (denoise > 0.9999) return Schedule(model, sampler, scheduler, steps);
        if (denoise <= 0) return [];
        var full = Schedule(model, sampler, scheduler, (int)(steps / denoise));
        return full[Math.Max(0, full.Length - (steps + 1))..];
    }
}

[Node("KSamplerAdvanced", Category = "model/sampling", DisplayName = "KSampler (Advanced)")]
public sealed class KSamplerAdvanced
{
    public static string[] Toggle() => ["enable", "disable"];

    public static string[] Leftover() => ["disable", "enable"];

    public static string[] SamplerNames() => KSampler.SamplerNames();

    public static string[] SchedulerNames() => KSampler.SchedulerNames();

    public async Task<LatentBatch> ExecuteAsync(ModelHandle model, Conditioning positive, Conditioning negative, LatentBatch latentImage,
        NodeContext context, IServiceProvider services,
        [Input(OptionsProvider = nameof(Toggle))] string addNoise = "enable",
        [Input(Min = 0, Max = 18446744073709551615d)] long noiseSeed = 0,
        [Input(Min = 1, Max = 10000)] int steps = 20,
        [Input(Min = 0.0, Max = 100.0, Step = 0.1)] double cfg = 8.0,
        [Input(OptionsProvider = nameof(SamplerNames))] string samplerName = "euler",
        [Input(OptionsProvider = nameof(SchedulerNames))] string scheduler = "simple",
        [Input(Min = 0, Max = 10000)] int startAtStep = 0,
        [Input(Min = 0, Max = 10000)] int endAtStep = 10000,
        [Input(OptionsProvider = nameof(Leftover))] string returnWithLeftoverNoise = "disable")
    {
        var sigmas = Slice(KSampler.Schedule(model.Sampling.Sigmas, samplerName, scheduler, steps), startAtStep, endAtStep, returnWithLeftoverNoise != "enable");
        return await KSampler.Run(model, positive, negative, latentImage, context, services, cfg, sigmas,
            new SamplingSettings(noiseSeed, samplerName, AddNoise: addNoise == "enable"));
    }

    /// <summary>
    /// The steps from <paramref name="start"/> to <paramref name="end"/>: cut after sigma[end] (its last sigma set to 0 when the run must end clean),
    /// then from sigma[start]; nothing when the start is at or past the last step. The input is not changed.
    /// </summary>
    public static float[] Slice(float[] sigmas, int start, int end, bool forceFullDenoise)
    {
        var s = (float[])sigmas.Clone();
        if (end < s.Length - 1)
        {
            s = s[..(end + 1)];
            if (forceFullDenoise) s[^1] = 0;
        }
        return start < s.Length - 1 ? s[start..] : [];
    }
}
