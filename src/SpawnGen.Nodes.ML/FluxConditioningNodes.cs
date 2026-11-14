using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

// Written from ComfyUI's /object_info entries and black-box runs of its nodes (2026-11-13, probe nodes SpawnGenDumpConditioning and
// SpawnGenDumpSigmas on the ComfyUI reference). Flux2Scheduler's shift rule is the published FLUX.2 "empirical mu" (Black Forest Labs, as in
// diffusers' FLUX.2 pipelines, Apache-2.0), which reproduces ComfyUI's sigmas for every measured case (Fixtures/flux2_scheduler).

/// <summary>Flux Guidance: every entry gets the distilled guidance scale (ComfyUI's "guidance"; measured: the value as given).</summary>
[Node("FluxGuidance", Category = "model/conditioning/flux", DisplayName = "FluxGuidance")]
public sealed class FluxGuidance
{
    public Conditioning Execute(Conditioning conditioning, [Input(Min = 0.0, Max = 100.0, Step = 0.1)] double guidance = 3.5) =>
        conditioning.Map(e => e with { Guidance = guidance });
}

/// <summary>
/// Set Reference Latent: appends the latent (its raw values) to every entry's reference latents; chained nodes add more, in order. Without a latent
/// the conditioning is passed on as it is (measured).
/// </summary>
[Node("ReferenceLatent", Category = "model/conditioning", DisplayName = "Set Reference Latent",
    Description = "This node sets the guiding latent for an edit model. If the model supports it you can chain multiple to set multiple reference images.")]
public sealed class ReferenceLatent
{
    public Conditioning Execute(Conditioning conditioning, LatentBatch? latent = null) =>
        latent is null ? conditioning : conditioning.Map(e => e with { ReferenceLatents = [.. e.ReferenceLatents ?? [], latent] });
}

/// <summary>Edit Model Reference Method: how an edit model places its reference latents. Measured: "uxo/uno" is stored as "uxo", the rest as written.</summary>
[Node("FluxKontextMultiReferenceLatentMethod", Category = "model/conditioning/flux", DisplayName = "Edit Model Reference Method")]
public sealed class FluxKontextMultiReferenceLatentMethod
{
    public Conditioning Execute(Conditioning conditioning, [Input(Options = ["offset", "index", "uxo/uno", "index_timestep_zero", "index_timestep_zero_isolated",
        "index_timestep_zero_text_start"])] string referenceLatentsMethod = "offset")
    {
        var method = referenceLatentsMethod == "uxo/uno" ? "uxo" : referenceLatentsMethod;
        return conditioning.Map(e => e with { ReferenceLatentsMethod = method });
    }
}

/// <summary>
/// Flux2Scheduler: steps + 1 sigmas, the flow time shift sigma = e^mu / (e^mu + 1 / t - 1) at t = 1, 1 - 1/steps, ..., 1/steps, then 0. mu follows the
/// picture's token count n (width x height / 256 rounded to even, see Tokens) and the step count: above 4300 tokens mu = 0.00016927 n + 0.45666666; below, mu is
/// the line through the 10-step value (8.73809524e-5 n + 1.89833333) and the 200-step one at that n, taken at the step count.
/// </summary>
[Node("Flux2Scheduler", Category = "model/sampling/schedulers", DisplayName = "Flux2Scheduler")]
public sealed class Flux2Scheduler
{
    public Sigmas Execute([Input(Min = 1, Max = 4096)] int steps = 20, [Input(Min = 16, Max = 16384, Step = 1)] int width = 1024,
        [Input(Min = 16, Max = 16384, Step = 1)] int height = 1024) => new(Sigmas(steps, width, height));

    public static double Mu(double tokens, int steps)
    {
        const double a1 = 8.73809524e-05, b1 = 1.89833333, a2 = 0.00016927, b2 = 0.45666666;
        if (tokens > 4300) return a2 * tokens + b2;
        double m200 = a2 * tokens + b2, m10 = a1 * tokens + b1;
        double a = (m200 - m10) / 190.0, b = m200 - 200.0 * a;
        return a * steps + b;
    }

    /// <summary>The picture's token count: width x height / 256 rounded to the nearest whole number, halves to even (measured: 1.5 and 2.5 give 2).</summary>
    public static double Tokens(int width, int height) => Math.Round(width * (double)height / 256.0, MidpointRounding.ToEven);

    public static float[] Sigmas(int steps, int width, int height)
    {
        double shift = Math.Exp(Mu(Tokens(width, height), steps));
        var r = new float[steps + 1];
        for (int i = 0; i < steps; i++)
        {
            double t = 1.0 - (double)i / steps;
            r[i] = (float)(shift / (shift + (1.0 / t - 1.0)));
        }
        return r;
    }
}
