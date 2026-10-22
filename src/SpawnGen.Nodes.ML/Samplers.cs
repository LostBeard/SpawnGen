using SpawnDev.ILGPU.ML.Preprocessing;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// Sigma-space samplers (the k-diffusion formulations ComfyUI uses): the sample <c>x</c> lives at noise level sigma, the model returns its denoised
/// estimate, and each step moves <c>x</c> to the next sigma. Pure host maths over small latents; the model call is the only part that touches the GPU.
/// </summary>
public static class Samplers
{
    public static readonly string[] Names =
    [
        "euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms", "dpmpp_2s_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "ddim", "lcm",
    ];

    /// <summary>The model's denoised estimate for the sample <c>x</c> at noise level <c>sigma</c> (the callee applies input scaling, guidance and masks).</summary>
    public delegate Task<float[]> DenoiseFn(float[] x, float sigma);

    /// <summary>Where a stochastic sampler's fresh noise comes from: deterministic per (seed, step) so a seed reproduces a picture.</summary>
    public sealed class NoiseSource(int seed, int channels, int height, int width)
    {
        public float[] At(int step) => DiffusionScheduler.GenerateNoise(channels, height, width, Mix(seed, step));

        // A well-mixed seed per step: nearby seeds give correlated streams in the library generator, and correlated noise would add up coherently.
        static int Mix(int seed, int step)
        {
            unchecked
            {
                uint z = (uint)seed * 0x9E3779B1u + (uint)(step + 1) * 0x85EBCA6Bu;
                z ^= z >> 16; z *= 0x7FEB352Du; z ^= z >> 15; z *= 0x846CA68Bu; z ^= z >> 16;
                return (int)z;
            }
        }
    }

    /// <summary>Runs <paramref name="sampler"/> over <paramref name="sigmas"/> (length steps + 1, the last is 0), starting from the already-noised <paramref name="x"/>.</summary>
    public static async Task<float[]> RunAsync(string sampler, float[] x, float[] sigmas, DenoiseFn model, NoiseSource noise,
        Action<int>? onStep = null, CancellationToken ct = default)
    {
        if (!Names.Contains(sampler)) throw new ArgumentException($"Unknown sampler '{sampler}'. Known: {string.Join(", ", Names)}.", nameof(sampler));
        int steps = sigmas.Length - 1, n = x.Length;
        float[]? oldDenoised = null;
        float hLast = 0;
        var history = new List<float[]>();   // lms: past derivatives

        for (int i = 0; i < steps; i++)
        {
            ct.ThrowIfCancellationRequested();
            float s = sigmas[i], next = sigmas[i + 1];
            var d0 = await model(x, s);

            switch (sampler)
            {
                case "euler" or "ddim":
                {
                    for (int k = 0; k < n; k++) x[k] += (x[k] - d0[k]) / s * (next - s);
                    break;
                }

                case "euler_ancestral":
                {
                    var (down, up) = AncestralStep(s, next);
                    for (int k = 0; k < n; k++) x[k] += (x[k] - d0[k]) / s * (down - s);
                    AddNoise(x, noise, i, up);
                    break;
                }

                case "heun":
                {
                    var d = Derivative(x, d0, s);
                    if (next == 0f) { for (int k = 0; k < n; k++) x[k] += d[k] * (next - s); break; }
                    float dt = next - s;
                    var x2 = new float[n];
                    for (int k = 0; k < n; k++) x2[k] = x[k] + d[k] * dt;
                    var dd = await model(x2, next);
                    for (int k = 0; k < n; k++) x[k] += 0.5f * (d[k] + (x2[k] - dd[k]) / next) * dt;
                    break;
                }

                case "dpm_2":
                {
                    var d = Derivative(x, d0, s);
                    if (next == 0f) { for (int k = 0; k < n; k++) x[k] += d[k] * (next - s); break; }
                    float mid = MathF.Exp((MathF.Log(s) + MathF.Log(next)) / 2), dt1 = mid - s, dt2 = next - s;
                    var x2 = new float[n];
                    for (int k = 0; k < n; k++) x2[k] = x[k] + d[k] * dt1;
                    var dm = await model(x2, mid);
                    for (int k = 0; k < n; k++) x[k] += (x2[k] - dm[k]) / mid * dt2;
                    break;
                }

                case "dpm_2_ancestral":
                {
                    var (down, up) = AncestralStep(s, next);
                    var d = Derivative(x, d0, s);
                    if (down == 0f) { for (int k = 0; k < n; k++) x[k] += d[k] * (down - s); }
                    else
                    {
                        float mid = MathF.Exp((MathF.Log(s) + MathF.Log(down)) / 2), dt1 = mid - s, dt2 = down - s;
                        var x2 = new float[n];
                        for (int k = 0; k < n; k++) x2[k] = x[k] + d[k] * dt1;
                        var dm = await model(x2, mid);
                        for (int k = 0; k < n; k++) x[k] += (x2[k] - dm[k]) / mid * dt2;
                    }
                    AddNoise(x, noise, i, up);
                    break;
                }

                case "lms":
                {
                    const int order = 4;
                    history.Add(Derivative(x, d0, s));
                    if (history.Count > order) history.RemoveAt(0);
                    int cur = Math.Min(i + 1, order);
                    var dsigma = sigmas.Select(v => (double)v).ToArray();
                    for (int j = 0; j < cur; j++)
                    {
                        float coeff = (float)LinearMultistepCoeff(cur, dsigma, i, j);
                        var dj = history[history.Count - 1 - j];
                        for (int k = 0; k < n; k++) x[k] += coeff * dj[k];
                    }
                    break;
                }

                case "dpmpp_2s_ancestral":
                {
                    var (down, up) = AncestralStep(s, next);
                    if (down == 0f)
                    {
                        for (int k = 0; k < n; k++) x[k] += (x[k] - d0[k]) / s * (down - s);
                    }
                    else
                    {
                        // Two-stage step in t = -log(sigma): a midpoint evaluation, then the full step with its result.
                        float t = -MathF.Log(s), tNext = -MathF.Log(down), h = tNext - t, mid = t + h * 0.5f;
                        float sMid = MathF.Exp(-mid), r1 = sMid / s;
                        var x2 = new float[n];
                        for (int k = 0; k < n; k++) x2[k] = r1 * x[k] - Expm1(-h * 0.5f) * d0[k];
                        var dm = await model(x2, sMid);
                        float r2 = MathF.Exp(-tNext) / MathF.Exp(-t);
                        for (int k = 0; k < n; k++) x[k] = r2 * x[k] - Expm1(-h) * dm[k];
                    }
                    AddNoise(x, noise, i, up);
                    break;
                }

                case "dpmpp_2m":
                {
                    if (next == 0f) { Array.Copy(d0, x, n); break; }
                    float t = -MathF.Log(s), tNext = -MathF.Log(next), h = tNext - t, scale = next / s, coeff = -Expm1(-h);
                    if (oldDenoised is null)
                    {
                        for (int k = 0; k < n; k++) x[k] = scale * x[k] + coeff * d0[k];
                    }
                    else
                    {
                        float r = hLast / h;
                        for (int k = 0; k < n; k++)
                        {
                            float d = (1f + 1f / (2f * r)) * d0[k] - 1f / (2f * r) * oldDenoised[k];
                            x[k] = scale * x[k] + coeff * d;
                        }
                    }
                    hLast = h;
                    break;
                }

                case "dpmpp_2m_sde":
                {
                    if (next == 0f) { Array.Copy(d0, x, n); break; }
                    const float eta = 1f;
                    float t = -MathF.Log(s), tNext = -MathF.Log(next), h = tNext - t, etaH = eta * h;
                    float scale = next / s * MathF.Exp(-etaH), coeff = -Expm1(-h - etaH);
                    for (int k = 0; k < n; k++) x[k] = scale * x[k] + coeff * d0[k];
                    if (oldDenoised is not null)
                    {
                        float r = hLast / h;
                        for (int k = 0; k < n; k++) x[k] += 0.5f * coeff * (1f / r) * (d0[k] - oldDenoised[k]); // midpoint solver
                    }
                    AddNoise(x, noise, i, next * MathF.Sqrt(-Expm1(-2f * etaH)));
                    hLast = h;
                    break;
                }

                case "lcm":
                {
                    // Latent consistency sampling: jump to the denoised estimate, then re-noise to the next level.
                    Array.Copy(d0, x, n);
                    if (next > 0f) AddNoise(x, noise, i, next);
                    break;
                }
            }
            oldDenoised = d0;
            onStep?.Invoke(i + 1);
        }
        return x;
    }

    /// <summary>exp(x) - 1 without losing precision for small x (the step sizes of fine schedules).</summary>
    static float Expm1(float x) => MathF.Abs(x) < 1e-4f ? x + 0.5f * x * x + x * x * x / 6f : MathF.Exp(x) - 1f;

    static float[] Derivative(float[] x, float[] denoised, float sigma)
    {
        var d = new float[x.Length];
        for (int k = 0; k < d.Length; k++) d[k] = (x[k] - denoised[k]) / sigma;
        return d;
    }

    /// <summary>How far a step goes down the ODE (<c>down</c>) and how much fresh noise is added (<c>up</c>) so the total noise level is <paramref name="next"/>.</summary>
    static (float Down, float Up) AncestralStep(float sigma, float next)
    {
        if (next == 0f) return (0f, 0f);
        float up = MathF.Min(next, MathF.Sqrt(next * next * (sigma * sigma - next * next) / (sigma * sigma)));
        return (MathF.Sqrt(next * next - up * up), up);
    }

    static void AddNoise(float[] x, NoiseSource noise, int step, float amount)
    {
        if (amount <= 0f) return;
        var z = noise.At(step);
        for (int k = 0; k < x.Length; k++) x[k] += z[k] * amount;
    }

    /// <summary>The weight of the derivative <paramref name="j"/> steps back in the order-<paramref name="order"/> linear multistep step from sigma[i] to sigma[i+1] (integral of its Lagrange basis polynomial).</summary>
    internal static double LinearMultistepCoeff(int order, double[] sigmas, int i, int j)
    {
        if (order - 1 > i) throw new ArgumentException("Not enough previous steps for this order.");
        double Basis(double tau)
        {
            double prod = 1;
            for (int k = 0; k < order; k++)
            {
                if (k == j) continue;
                prod *= (tau - sigmas[i - k]) / (sigmas[i - j] - sigmas[i - k]);
            }
            return prod;
        }
        // Composite Simpson on [sigma_i, sigma_i+1]: the integrand is a polynomial of degree order-1 <= 3, so it is exact.
        double a = sigmas[i], b = sigmas[i + 1], mid = (a + b) / 2;
        return (b - a) / 6 * (Basis(a) + 4 * Basis(mid) + Basis(b));
    }
}
