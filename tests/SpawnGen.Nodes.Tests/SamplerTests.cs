using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>Sampler maths against synthetic models with known answers; no GPU, no weights.</summary>
public class SamplerTests
{
    const int C = 4, H = 16, W = 16, N = C * H * W;

    static float[] Sigmas(int steps)
    {
        var s = new float[steps + 1];
        float max = 14.6f, min = 0.03f;
        for (int i = 0; i < steps; i++) s[i] = MathF.Exp(MathF.Log(max) + (MathF.Log(min) - MathF.Log(max)) * i / (steps - 1));
        return s; // the last stays 0
    }

    static float[] Start(float sigma, int seed)
    {
        var z = DiffusionScheduler.GenerateNoise(C, H, W, seed);
        for (int i = 0; i < z.Length; i++) z[i] *= sigma;
        return z;
    }

    static Samplers.NoiseSource Noise(int seed = 5) => new(seed, C, H, W);

    /// <summary>Data is the single point x0: the ideal denoised estimate is x0 whatever the noise.</summary>
    static Samplers.DenoiseFn PointMass(float x0, Action? onCall = null) => (x, _) => { onCall?.Invoke(); return Task.FromResult(x.Select(_ => x0).ToArray()); };

    /// <summary>Data is N(mu, s0^2): the ideal denoised estimate is mu + s0^2/(s0^2 + sigma^2) * (x - mu).</summary>
    static Samplers.DenoiseFn Gaussian(float mu, float s0) => (x, sigma) =>
    {
        float k = s0 * s0 / (s0 * s0 + sigma * sigma);
        return Task.FromResult(x.Select(v => mu + k * (v - mu)).ToArray());
    };

    [TestCaseSource(typeof(Samplers), nameof(Samplers.Names))]
    public async Task Every_sampler_lands_on_a_point_mass(string sampler)
    {
        var sigmas = Sigmas(8);
        var x = await Samplers.RunAsync(sampler, Start(sigmas[0], 1), sigmas, PointMass(0.7f), Noise());
        // dpm_fast and dpm_adaptive stop at the last sigma above 0 and uni_pc at 0.001 (as in ComfyUI), so some of that noise level is left.
        float left = sampler switch { "dpm_fast" or "dpm_adaptive" => 5 * sigmas[^2], "uni_pc" or "uni_pc_bh2" => 5e-3f, _ => 1e-3f };
        Assert.That(x.Max(v => Math.Abs(v - 0.7f)), Is.LessThan(left));
    }

    [Test]
    public async Task Higher_order_deterministic_samplers_follow_the_probability_flow_ode_closer_than_euler()
    {
        const float mu = 0.5f, s0 = 1f;
        var sigmas = Sigmas(8);
        var start = Start(sigmas[0], 2);
        // Exact ODE solution from sigma_T down to 0 for a Gaussian: x0 = mu + (x_T - mu) * s0 / sqrt(s0^2 + sigma_T^2).
        var exact = start.Select(v => mu + (v - mu) * s0 / MathF.Sqrt(s0 * s0 + sigmas[0] * sigmas[0])).ToArray();
        double Error(float[] x) => Math.Sqrt(x.Zip(exact, (a, b) => (double)(a - b) * (a - b)).Average());

        var errors = new Dictionary<string, double>();
        foreach (var name in new[] { "euler", "heun", "dpm_2", "lms", "dpmpp_2m", "ipndm", "ipndm_v", "res_multistep", "gradient_estimation" })
            errors[name] = Error(await Samplers.RunAsync(name, (float[])start.Clone(), sigmas, Gaussian(mu, s0), Noise()));
        TestContext.Out.WriteLine("rms error vs exact ODE at 8 steps: " + string.Join(", ", errors.Select(e => $"{e.Key} {e.Value:E2}")));

        foreach (var better in new[] { "heun", "dpm_2", "lms", "dpmpp_2m", "ipndm", "ipndm_v", "res_multistep", "gradient_estimation" })
            Assert.That(errors[better], Is.LessThan(errors["euler"]), $"{better} should beat euler at the same step count");
        Assert.That(errors["dpmpp_2m"], Is.LessThan(0.05));
        Assert.That(errors["ipndm"], Is.LessThan(0.05));
        Assert.That(errors["res_multistep"], Is.LessThan(0.1), "second order, but this schedule favours the dpmpp family");
    }

    [TestCase("euler_ancestral")]
    [TestCase("dpm_2_ancestral")]
    [TestCase("dpmpp_2s_ancestral")]
    [TestCase("dpmpp_2m_sde")]
    [TestCase("dpmpp_sde")]
    [TestCase("dpmpp_3m_sde")]
    [TestCase("res_multistep_ancestral")]
    public async Task Stochastic_samplers_converge_to_the_data_distribution_as_steps_grow(string sampler)
    {
        // A stochastic sampler's result is a draw from the data distribution, so its spread tends to s0. First-order ones are too narrow with few steps.
        const float mu = 0.5f, s0 = 1f;
        async Task<(double Mean, double Std)> Stats(int steps)
        {
            var sigmas = Sigmas(steps);
            var x = await Samplers.RunAsync(sampler, Start(sigmas[0], 3), sigmas, Gaussian(mu, s0), Noise(9));
            double mean = x.Average();
            return (mean, Math.Sqrt(x.Select(v => (v - mean) * (v - mean)).Average()));
        }
        var few = await Stats(16);
        var many = await Stats(128);
        TestContext.Out.WriteLine($"{sampler}: std {few.Std:F3} at 16 steps, {many.Std:F3} at 128 (expect {s0})");
        Assert.That(many.Mean, Is.EqualTo(mu).Within(0.12));
        Assert.That(many.Std, Is.EqualTo(s0).Within(0.12));
    }

    [TestCase("euler_ancestral")]
    [TestCase("euler_ancestral_cfg_pp")]
    [TestCase("dpm_2_ancestral")]
    [TestCase("dpmpp_2s_ancestral")]
    [TestCase("dpmpp_2m_sde")]
    [TestCase("dpmpp_sde")]
    [TestCase("dpmpp_3m_sde")]
    [TestCase("res_multistep_ancestral")]
    [TestCase("lcm")]
    public async Task Noise_adding_samplers_are_reproducible_for_a_seed_and_differ_between_seeds(string sampler)
    {
        var sigmas = Sigmas(6);
        Task<float[]> Run(int seed) => Samplers.RunAsync(sampler, Start(sigmas[0], 4), sigmas, Gaussian(0f, 1f), Noise(seed));
        var a = await Run(1);
        Assert.That(await Run(1), Is.EqualTo(a));
        Assert.That(await Run(2), Is.Not.EqualTo(a));
    }

    [TestCase("euler", 6)]
    [TestCase("euler_ancestral", 6)]
    [TestCase("dpmpp_2m", 6)]
    [TestCase("heun", 11)]            // two model calls per step, except the last step (it lands on sigma 0)
    [TestCase("dpm_2", 11)]
    [TestCase("dpm_2_ancestral", 11)]
    [TestCase("dpmpp_2s_ancestral", 11)]
    [TestCase("lms", 6)]
    [TestCase("ipndm", 6)]
    [TestCase("ipndm_v", 6)]
    [TestCase("gradient_estimation", 6)]
    [TestCase("euler_cfg_pp", 6)]
    [TestCase("euler_ancestral_cfg_pp", 6)]
    [TestCase("dpmpp_2m_cfg_pp", 6)]
    [TestCase("gradient_estimation_cfg_pp", 6)]
    [TestCase("res_multistep", 6)]
    [TestCase("res_multistep_ancestral", 6)]
    [TestCase("dpmpp_3m_sde", 6)]
    [TestCase("dpmpp_sde", 11)]
    [TestCase("lcm", 6)]
    public async Task Samplers_ask_the_model_the_expected_number_of_times(string sampler, int expected)
    {
        int calls = 0;
        var sigmas = Sigmas(6);
        await Samplers.RunAsync(sampler, Start(sigmas[0], 1), sigmas, PointMass(0f, () => calls++), Noise());
        Assert.That(calls, Is.EqualTo(expected));
    }

    [TestCase("euler_cfg_pp", "euler")]
    [TestCase("euler_ancestral_cfg_pp", "euler_ancestral")]
    [TestCase("dpmpp_2m_cfg_pp", "dpmpp_2m")]
    [TestCase("gradient_estimation_cfg_pp", "gradient_estimation")]
    public async Task A_cfg_pp_sampler_without_guidance_is_its_plain_counterpart(string sampler, string plain)
    {
        // With no separate unconditional estimate the CFG++ update collapses to the ordinary one.
        var sigmas = Sigmas(8);
        var a = await Samplers.RunAsync(sampler, Start(sigmas[0], 6), sigmas, Gaussian(0.3f, 1f), Noise(4));
        var b = await Samplers.RunAsync(plain, Start(sigmas[0], 6), sigmas, Gaussian(0.3f, 1f), Noise(4));
        Assert.That(a.Zip(b, (p, q) => Math.Abs(p - q)).Max(), Is.LessThan(2e-3f));
    }

    [Test]
    public async Task CFG_plus_plus_samplers_use_the_unconditional_estimate_when_given()
    {
        var sigmas = Sigmas(6);
        GuidanceOutput? last = null;
        Samplers.DenoiseFn model = async (x, sigma) => { var d = await Gaussian(0.3f, 1f)(x, sigma); last = new GuidanceOutput(d, d.Select(v => v + 0.1f).ToArray(), true); return d; };
        var with = await Samplers.RunAsync("euler_cfg_pp", Start(sigmas[0], 6), sigmas, model, Noise(), uncond: () => last);
        var without = await Samplers.RunAsync("euler_cfg_pp", Start(sigmas[0], 6), sigmas, Gaussian(0.3f, 1f), Noise());
        Assert.That(with, Is.Not.EqualTo(without));
        Assert.That(Samplers.UsesUncond("euler_cfg_pp"), Is.True);
        Assert.That(Samplers.UsesUncond("euler"), Is.False);
    }

    [Test]
    public async Task Steps_are_reported_and_cancellation_stops_the_run()
    {
        var sigmas = Sigmas(5);
        var seen = new List<int>();
        await Samplers.RunAsync("euler", Start(sigmas[0], 1), sigmas, PointMass(0f), Noise(), seen.Add);
        Assert.That(seen, Is.EqualTo(new[] { 1, 2, 3, 4, 5 }));

        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.ThrowsAsync<OperationCanceledException>(async () =>
            await Samplers.RunAsync("euler", Start(sigmas[0], 1), sigmas, PointMass(0f), Noise(), null, cts.Token));
    }

    [Test]
    public void An_unknown_sampler_name_is_an_error_that_lists_the_known_ones()
    {
        var sigmas = Sigmas(4);
        var ex = Assert.ThrowsAsync<ArgumentException>(async () => await Samplers.RunAsync("nope", Start(sigmas[0], 1), sigmas, PointMass(0), Noise()));
        Assert.That(ex!.Message, Does.Contain("dpmpp_2m_sde"));
    }

    [Test]
    public void Linear_multistep_coefficients_integrate_a_constant_exactly()
    {
        var sigmas = new double[] { 10, 6, 3.5, 2, 1, 0.4, 0 };
        for (int order = 1; order <= 4; order++)
        {
            int i = 4;
            double sum = Enumerable.Range(0, order).Sum(j => Samplers.LinearMultistepCoeff(order, sigmas, i, j));
            Assert.That(sum, Is.EqualTo(sigmas[i + 1] - sigmas[i]).Within(1e-9), $"order {order}: the weights of a constant derivative add up to the step");
        }
        Assert.That(Samplers.LinearMultistepCoeff(1, sigmas, 2, 0), Is.EqualTo(sigmas[3] - sigmas[2]).Within(1e-12));
    }
}
