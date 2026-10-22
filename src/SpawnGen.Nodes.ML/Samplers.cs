using SpawnDev.ILGPU.ML.Preprocessing;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// Sigma-space samplers for epsilon-prediction models (k-diffusion formulation, same maths as ComfyUI's): the sample <c>x</c> lives at noise
/// level sigma, the denoised estimate is <c>x - sigma * eps</c>, and each step moves <c>x</c> to the next sigma. Pure host maths over small
/// latents; the model call is the only part that touches the GPU.
/// </summary>
public static class Samplers
{
    public static readonly string[] Names = ["euler", "euler_ancestral", "dpmpp_2m"];

    /// <summary>Predicts epsilon for the sample <c>x</c> at <c>step</c> (the callee applies the model-input scaling and guidance).</summary>
    public delegate Task<float[]> EpsilonFn(float[] x, int step);

    /// <summary>Runs <paramref name="sampler"/> over <paramref name="sigmas"/> (length steps + 1, last 0), starting from the already-noised <paramref name="x"/>.</summary>
    public static async Task<float[]> RunAsync(string sampler, float[] x, float[] sigmas, EpsilonFn epsilon, int noiseSeed,
        int channels, int height, int width, Action<int>? onStep = null, CancellationToken ct = default)
    {
        int steps = sigmas.Length - 1;
        float[]? oldDenoised = null;
        for (int i = 0; i < steps; i++)
        {
            ct.ThrowIfCancellationRequested();
            float sigma = sigmas[i], next = sigmas[i + 1];
            var eps = await epsilon(x, i);
            var denoised = new float[x.Length];
            for (int k = 0; k < x.Length; k++) denoised[k] = x[k] - sigma * eps[k];

            switch (sampler)
            {
                case "euler":
                    for (int k = 0; k < x.Length; k++) x[k] += eps[k] * (next - sigma);
                    break;

                case "euler_ancestral":
                {
                    // Step down to sigma_down along the ODE, then add fresh noise worth sigma_up so the total noise level is `next`.
                    float up = next > 0f ? MathF.Min(next, MathF.Sqrt(next * next * (sigma * sigma - next * next) / (sigma * sigma))) : 0f;
                    float down = MathF.Sqrt(next * next - up * up);
                    for (int k = 0; k < x.Length; k++) x[k] += eps[k] * (down - sigma);
                    if (next > 0f)
                    {
                        var noise = DiffusionScheduler.GenerateNoise(channels, height, width, unchecked(noiseSeed + (i + 1) * 1000003));
                        for (int k = 0; k < x.Length; k++) x[k] += noise[k] * up;
                    }
                    break;
                }

                case "dpmpp_2m":
                {
                    if (next == 0f) { Array.Copy(denoised, x, x.Length); break; }
                    float t = -MathF.Log(sigma), tNext = -MathF.Log(next), h = tNext - t;
                    float scale = next / sigma, coeff = -(MathF.Exp(-h) - 1f);
                    if (oldDenoised is null)
                    {
                        for (int k = 0; k < x.Length; k++) x[k] = scale * x[k] + coeff * denoised[k];
                    }
                    else
                    {
                        float hLast = t - -MathF.Log(sigmas[i - 1]), r = hLast / h;
                        for (int k = 0; k < x.Length; k++)
                        {
                            float d = (1f + 1f / (2f * r)) * denoised[k] - 1f / (2f * r) * oldDenoised[k];
                            x[k] = scale * x[k] + coeff * d;
                        }
                    }
                    break;
                }

                default:
                    throw new ArgumentException($"Unknown sampler '{sampler}'. Known: {string.Join(", ", Names)}.", nameof(sampler));
            }
            oldDenoised = denoised;
            onStep?.Invoke(i + 1);
        }
        return x;
    }
}
