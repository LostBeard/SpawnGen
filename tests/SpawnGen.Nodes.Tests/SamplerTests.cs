using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>Sampler maths against synthetic models with known answers; no GPU, no weights.</summary>
public class SamplerTests
{
    const int C = 4, H = 32, W = 32, N = C * H * W;

    static float[] Sigmas(int steps)
    {
        var s = new float[steps + 1];
        float max = 14.6f, min = 0.03f;
        for (int i = 0; i < steps; i++) s[i] = MathF.Exp(MathF.Log(max) + (MathF.Log(min) - MathF.Log(max)) * i / (steps - 1));
        return s; // last stays 0
    }

    static float[] StartNoise(float sigma, int seed)
    {
        var z = DiffusionScheduler.GenerateNoise(C, H, W, seed);
        for (int i = 0; i < z.Length; i++) z[i] *= sigma;
        return z;
    }

    /// <summary>Data is the single point x0: the ideal epsilon is (x - x0) / sigma.</summary>
    static Samplers.EpsilonFn PointMass(float x0, float[] sigmas) =>
        (x, step) => Task.FromResult(x.Select(v => (v - x0) / sigmas[step]).ToArray());

    /// <summary>Data is N(mu, s0^2): the ideal denoised estimate is mu + s0^2/(s0^2 + sigma^2) * (x - mu).</summary>
    static Samplers.EpsilonFn Gaussian(float mu, float s0, float[] sigmas) =>
        (x, step) =>
        {
            float sigma = sigmas[step], k = s0 * s0 / (s0 * s0 + sigma * sigma);
            return Task.FromResult(x.Select(v => (v - (mu + k * (v - mu))) / sigma).ToArray());
        };

    [TestCaseSource(typeof(Samplers), nameof(Samplers.Names))]
    public async Task Every_sampler_lands_on_a_point_mass(string sampler)
    {
        var sigmas = Sigmas(8);
        var x = await Samplers.RunAsync(sampler, StartNoise(sigmas[0], 1), sigmas, PointMass(0.7f, sigmas), noiseSeed: 5, C, H, W);
        Assert.That(x.Max(v => Math.Abs(v - 0.7f)), Is.LessThan(1e-3f));
    }

    [Test]
    public async Task Dpmpp_2m_follows_the_probability_flow_ode_closer_than_euler()
    {
        const float mu = 0.5f, s0 = 1f;
        var sigmas = Sigmas(8);
        var start = StartNoise(sigmas[0], 2);
        // Exact ODE solution from sigma_T down to 0 for a Gaussian: x0 = mu + (x_T - mu) * s0 / sqrt(s0^2 + sigma_T^2).
        var exact = start.Select(v => mu + (v - mu) * s0 / MathF.Sqrt(s0 * s0 + sigmas[0] * sigmas[0])).ToArray();

        double Error(float[] x) => Math.Sqrt(x.Zip(exact, (a, b) => (double)(a - b) * (a - b)).Average());
        var euler = await Samplers.RunAsync("euler", (float[])start.Clone(), sigmas, Gaussian(mu, s0, sigmas), 0, C, H, W);
        var dpm = await Samplers.RunAsync("dpmpp_2m", (float[])start.Clone(), sigmas, Gaussian(mu, s0, sigmas), 0, C, H, W);

        TestContext.Out.WriteLine($"rms error vs exact ODE, 8 steps: euler {Error(euler):E2}, dpmpp_2m {Error(dpm):E2}");
        Assert.That(Error(dpm), Is.LessThan(Error(euler)), "A second-order sampler should be more accurate than Euler at the same step count.");
    }

    [Test]
    public async Task Euler_ancestral_converges_to_the_data_variance_as_steps_grow()
    {
        // Ancestral sampling is stochastic: the final sample is a draw from the data distribution, so its spread tends to s0 as the step count
        // grows. It is first order, so with few steps the spread is too small (measured: 0.69 at 8 steps, 0.84 at 16, 0.93 at 32, 1.00 at 128).
        const float mu = 0.5f, s0 = 1f;
        async Task<double> StdAt(int steps)
        {
            var sigmas = Sigmas(steps);
            var x = await Samplers.RunAsync("euler_ancestral", StartNoise(sigmas[0], 3), sigmas, Gaussian(mu, s0, sigmas), noiseSeed: 9, C, H, W);
            Assert.That(x.Average(), Is.EqualTo(mu).Within(0.1));
            return Math.Sqrt(x.Select(v => (v - x.Average()) * (v - x.Average())).Average());
        }
        double few = await StdAt(16), many = await StdAt(128);
        TestContext.Out.WriteLine($"ancestral std: {few:F3} at 16 steps, {many:F3} at 128 steps (expect {s0})");
        Assert.That(many, Is.EqualTo(s0).Within(0.1));
        Assert.That(Math.Abs(many - s0), Is.LessThan(Math.Abs(few - s0)), "More steps must get closer to the data variance.");
    }

    [Test]
    public async Task Ancestral_is_reproducible_for_a_seed_and_differs_between_seeds()
    {
        var sigmas = Sigmas(6);
        Task<float[]> Run(int seed) => Samplers.RunAsync("euler_ancestral", StartNoise(sigmas[0], 4), sigmas, Gaussian(0f, 1f, sigmas), seed, C, H, W);
        var a = await Run(1);
        var b = await Run(1);
        var c = await Run(2);
        Assert.That(b, Is.EqualTo(a));
        Assert.That(c, Is.Not.EqualTo(a));
    }

    [Test]
    public void An_unknown_sampler_name_is_an_error_that_lists_the_known_ones()
    {
        var sigmas = Sigmas(4);
        var ex = Assert.ThrowsAsync<ArgumentException>(async () =>
            await Samplers.RunAsync("nope", StartNoise(sigmas[0], 1), sigmas, PointMass(0, sigmas), 0, C, H, W));
        Assert.That(ex!.Message, Does.Contain("euler_ancestral"));
    }
}
