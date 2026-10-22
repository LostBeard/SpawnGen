using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

public class BetaSchedulerTests
{
    [Test]
    public void The_beta_CDF_matches_closed_forms_and_the_quantile_inverts_it()
    {
        // Beta(1, 1) is uniform; Beta(2, 2) has CDF 3x^2 - 2x^3; Beta(0.5, 0.5) has CDF (2/pi) asin(sqrt(x)).
        Assert.That(BetaMath.Cdf(0.3, 1, 1), Is.EqualTo(0.3).Within(1e-9));
        Assert.That(BetaMath.Cdf(0.3, 2, 2), Is.EqualTo(3 * 0.09 - 2 * 0.027).Within(1e-9));
        Assert.That(BetaMath.Cdf(0.8, 2, 2), Is.EqualTo(3 * 0.64 - 2 * 0.512).Within(1e-9), "above the mean uses the symmetry");
        Assert.That(BetaMath.Cdf(0.2, 0.5, 0.5), Is.EqualTo(2 / Math.PI * Math.Asin(Math.Sqrt(0.2))).Within(1e-8));
        Assert.That(BetaMath.Cdf(0.6, 0.6, 0.6) + BetaMath.Cdf(0.4, 0.6, 0.6), Is.EqualTo(1).Within(1e-9), "symmetric for alpha = beta");
        foreach (var (a, b) in new[] { (0.6, 0.6), (2.0, 5.0), (1.0, 1.0), (5.0, 1.5) })
            foreach (double q in new[] { 0.01, 0.25, 0.5, 0.9, 0.999 })
                Assert.That(BetaMath.Cdf(BetaMath.Ppf(q, a, b), a, b), Is.EqualTo(q).Within(1e-9), $"Beta({a}, {b}) at {q}");
        Assert.That(BetaMath.Ppf(0, 2, 2), Is.EqualTo(0));
        Assert.That(BetaMath.Ppf(1, 2, 2), Is.EqualTo(1));
        Assert.That(BetaMath.Ppf(0.5, 2, 2), Is.EqualTo(0.5).Within(1e-9));
    }

    [Test]
    public void The_beta_schedule_starts_at_the_noisiest_sigma_falls_and_ends_in_zero_without_repeats()
    {
        var m = SamplingSetupTests.Sd();
        var s = m.Schedule("beta", 20);
        Assert.That(s[0], Is.EqualTo(m.SigmaMax).Within(1e-4f));
        Assert.That(s[^1], Is.EqualTo(0f));
        for (int i = 1; i < s.Length; i++) Assert.That(s[i], Is.LessThan(s[i - 1]), $"strictly falling at {i}");
        Assert.That(s.Length, Is.InRange(15, 21));
        // alpha = beta = 0.6 puts more steps at both ends than the middle: the first gap (in timesteps) is smaller than the middle ones.
        var steps = s.Take(s.Length - 1).Select(v => m.Timestep(v)).ToArray();
        Assert.That(steps[0] - steps[1], Is.LessThan(steps[9] - steps[10]));
        // Other shapes are a node away.
        var skewed = m.Beta(20, 2.0, 5.0);
        Assert.That(skewed[0], Is.EqualTo(m.SigmaMax).Within(1e-4f));
        Assert.That(skewed, Is.Not.EqualTo(s));
    }

    [Test]
    public void Linear_quadratic_follows_ComfyUIs_formula()
    {
        var m = SamplingSetupTests.Sd();
        var s = m.LinearQuadratic(10);
        Assert.That(s, Has.Length.EqualTo(11));
        Assert.That(s[0], Is.EqualTo(m.SigmaMax).Within(1e-4f));
        Assert.That(s[^1], Is.EqualTo(0f).Within(1e-6f));
        for (int i = 1; i < s.Length; i++) Assert.That(s[i], Is.LessThan(s[i - 1]));
        // Hand computed for steps = 10, threshold 0.025, linear_steps = 5: the linear part falls 0.005 of sigma_max per step; at step 5 the quadratic starts at 5 * 0.005 = 0.025
        // (the linear part's next value) and reaches 1 at the end.
        Assert.That(s[1] / m.SigmaMax, Is.EqualTo(1 - 0.005).Within(1e-6));
        Assert.That(s[5] / m.SigmaMax, Is.EqualTo(1 - 0.025).Within(1e-6));
        Assert.That(m.LinearQuadratic(1), Is.EqualTo(new[] { m.SigmaMax, 0f }));
    }

    [Test]
    public void The_scheduler_names_include_the_new_ones_and_denoise_tolerates_merged_steps()
    {
        Assert.That(ModelSigmas.SchedulerNames, Does.Contain("beta").And.Contain("linear_quadratic"));
        var m = SamplingSetupTests.Sd();
        var s = m.ScheduleForDenoise("beta", 6, 0.5f);
        Assert.That(s[^1], Is.EqualTo(0f));
        Assert.That(s.Length, Is.LessThanOrEqualTo(7));
    }
}
