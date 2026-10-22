using SpawnDev.ILGPU.ML.Preprocessing;

namespace SpawnGen.Nodes.ML;

/// <summary>Timesteps and sigmas a sampler walks, computed the way ComfyUI does for <c>denoise &lt; 1</c>.</summary>
public static class SigmaSchedule
{
    /// <summary>
    /// <paramref name="steps"/> sampling steps at the given <paramref name="denoise"/> strength: with denoise below 1 the full schedule has
    /// <c>steps / denoise</c> steps and only its last <paramref name="steps"/> are run, so sampling starts part-way down from a noised input.
    /// Returns <c>timesteps</c> (length steps) and <c>sigmas</c> (length steps + 1, the last is 0).
    /// </summary>
    public static (int[] Timesteps, float[] Sigmas) Build(int steps, float denoise, float[] alphasCumprod)
    {
        if (steps < 1) throw new ArgumentOutOfRangeException(nameof(steps));
        if (denoise is <= 0f or > 1f) throw new ArgumentOutOfRangeException(nameof(denoise), "denoise must be in (0, 1].");

        int total = denoise >= 1f ? steps : (int)(steps / denoise);
        if (total > 1000) total = 1000; // the model was trained on 1000 timesteps
        var timesteps = DiffusionScheduler.GetTimesteps(total);
        var sigmas = DiffusionScheduler.TimestepsToSigmas(timesteps, alphasCumprod);
        steps = Math.Min(steps, total);
        return (timesteps[^steps..], sigmas[^(steps + 1)..]);
    }
}
