using SpawnGen.Engine.Types;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>Noise schedules, step slicing and conditioning: the parts of sampling that are plain maths and bookkeeping.</summary>
public class SamplingSetupTests
{
    // SD's scaled-linear beta schedule: 1000 steps, beta from 0.00085 to 0.012 in sqrt space.
    internal static ModelSigmas Sd()
    {
        var alphas = new float[1000];
        double cum = 1;
        for (int t = 0; t < 1000; t++)
        {
            double b = Math.Pow(Math.Sqrt(0.00085) + (Math.Sqrt(0.012) - Math.Sqrt(0.00085)) * t / 999, 2);
            cum *= 1 - b;
            alphas[t] = (float)cum;
        }
        return new ModelSigmas(alphas);
    }

    [Test]
    public void Sigma_and_Timestep_are_inverses_and_sigmas_rise_with_the_timestep()
    {
        var m = Sd();
        Assert.That(m.SigmaMin, Is.EqualTo(0.0292f).Within(0.002f));
        Assert.That(m.SigmaMax, Is.EqualTo(14.6146f).Within(0.05f));
        foreach (double t in new[] { 0.0, 10.5, 250.25, 500, 998.5, 999 })
            Assert.That(m.Timestep(m.Sigma(t)), Is.EqualTo(t).Within(1e-2), $"timestep {t}");
        Assert.That(m.Sigma(100), Is.LessThan(m.Sigma(101)));
        Assert.That(m.Timestep(0f), Is.EqualTo(0f));
        Assert.That(m.Timestep(1000f), Is.EqualTo(999f));
    }

    [TestCaseSource(typeof(ModelSigmas), nameof(ModelSigmas.SchedulerNames))]
    public void Every_scheduler_gives_falling_sigmas_ending_in_zero(string scheduler)
    {
        var m = Sd();
        foreach (int steps in new[] { 1, 4, 20 })
        {
            var s = m.Schedule(scheduler, steps);
            Assert.That(s[^1], Is.EqualTo(0f));
            if (scheduler is not ("ddim_uniform" or "beta")) Assert.That(s.Length, Is.EqualTo(steps + 1), $"{scheduler} {steps}");
            for (int i = 1; i < s.Length; i++) Assert.That(s[i], Is.LessThanOrEqualTo(s[i - 1]), $"{scheduler} {steps} at {i}");
            if (!(scheduler == "ddim_uniform" && steps == 1)) Assert.That(s[0], Is.GreaterThan(1f), $"{scheduler} {steps} starts from a high noise level"); // ComfyUI quirk: one ddim_uniform step starts at the lowest level
        }
    }

    [Test]
    public void Normal_karras_and_exponential_span_the_model_range()
    {
        var m = Sd();
        var normal = m.Schedule("normal", 10);
        Assert.That(normal[0], Is.EqualTo(m.SigmaMax).Within(1e-3));
        Assert.That(normal[9], Is.EqualTo(m.SigmaMin).Within(1e-4));
        foreach (var name in new[] { "karras", "exponential" })
        {
            var s = m.Schedule(name, 10);
            Assert.That(s[0], Is.EqualTo(m.SigmaMax).Within(1e-2), name);
            Assert.That(s[9], Is.EqualTo(m.SigmaMin).Within(1e-3), name);
        }
        // Karras (rho = 7) keeps the sigma higher through the middle of the run than the geometric exponential schedule does.
        Assert.That(m.Schedule("karras", 10)[5], Is.GreaterThan(m.Schedule("exponential", 10)[5]));
    }

    [Test]
    public void Unknown_scheduler_and_bad_step_counts_are_errors()
    {
        var m = Sd();
        Assert.That(() => m.Schedule("nope", 5), Throws.ArgumentException.With.Message.Contains("karras"));
        Assert.That(() => m.Schedule("normal", 0), Throws.TypeOf<ArgumentOutOfRangeException>());
        Assert.That(() => m.ScheduleForDenoise("normal", 5, 0f), Throws.TypeOf<ArgumentOutOfRangeException>());
        Assert.That(() => m.ScheduleForDenoise("normal", 5, 1.5f), Throws.TypeOf<ArgumentOutOfRangeException>());
    }

    [Test]
    public void Partial_denoise_runs_the_tail_of_a_longer_schedule()
    {
        var m = Sd();
        var full = m.Schedule("karras", 40);
        var part = m.ScheduleForDenoise("karras", 20, 0.5f);
        Assert.That(part.Length, Is.EqualTo(21));
        Assert.That(part, Is.EqualTo(full[^21..]));
        Assert.That(part[0], Is.LessThan(m.SigmaMax / 2), "starts part-way down, from a noised input");
        Assert.That(m.ScheduleForDenoise("karras", 20, 1f), Is.EqualTo(m.Schedule("karras", 20)));
    }

    [Test]
    public void KSamplerAdvanced_slices_the_schedule_by_step()
    {
        float[] s = [10, 8, 6, 4, 2, 0];
        Assert.That(KSamplerAdvanced.Slice(s, 0, 10000, true), Is.EqualTo(s));
        Assert.That(KSamplerAdvanced.Slice(s, 1, 3, false), Is.EqualTo(new float[] { 8, 6, 4 }), "leftover noise kept: ends at sigma[3]");
        Assert.That(KSamplerAdvanced.Slice(s, 1, 3, true), Is.EqualTo(new float[] { 8, 6, 0 }), "forced clean: the last sigma becomes 0");
        Assert.That(KSamplerAdvanced.Slice(s, 5, 10000, true), Is.Empty, "start past the end: nothing to run");
        Assert.That(s, Is.EqualTo(new float[] { 10, 8, 6, 4, 2, 0 }), "the input schedule is not modified");
    }

    static Conditioning Prompt(params float[] values) => new(values, [1, values.Length / 4, 4]);

    [Test]
    public void Combine_keeps_both_prompts_as_separate_entries()
    {
        var c = new ConditioningCombine().Execute(Prompt(1, 1, 1, 1), Prompt(2, 2, 2, 2));
        Assert.That(c.Entries, Has.Count.EqualTo(2));
        Assert.That(c.Entries[1].Embeddings, Is.EqualTo(new float[] { 2, 2, 2, 2 }));
    }

    [Test]
    public void Concat_appends_tokens_and_checks_the_embedding_size()
    {
        var c = new ConditioningConcat().Execute(Prompt(1, 1, 1, 1), Prompt(2, 2, 2, 2, 3, 3, 3, 3));
        Assert.That(c.Shape, Is.EqualTo(new[] { 1, 3, 4 }));
        Assert.That(c.Embeddings, Is.EqualTo(new float[] { 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3 }));
        Assert.That(() => new ConditioningConcat().Execute(Prompt(1, 1, 1, 1), new Conditioning(new float[6], [1, 2, 3])), Throws.ArgumentException);
    }

    [Test]
    public void Average_blends_and_zero_pads_a_shorter_prompt()
    {
        var c = new ConditioningAverage().Execute(Prompt(4, 4, 4, 4, 8, 8, 8, 8), Prompt(0, 0, 0, 0), 0.25f);
        Assert.That(c.Embeddings, Is.EqualTo(new float[] { 1, 1, 1, 1, 2, 2, 2, 2 }));
    }

    [Test]
    public void ZeroOut_keeps_the_shape_and_clears_the_values()
    {
        var c = new ConditioningZeroOut().Execute(Prompt(5, 5, 5, 5));
        Assert.That(c.Embeddings, Is.All.EqualTo(0f));
        Assert.That(c.Shape, Is.EqualTo(new[] { 1, 1, 4 }));
    }

    [Test]
    public void Area_weight_is_the_strength_inside_the_rectangle_and_zero_outside()
    {
        var c = new ConditioningSetArea().Execute(Prompt(1, 1, 1, 1), width: 16, height: 16, x: 8, y: 0, strength: 2f); // cells: x 1..2, y 0..1
        var w = SamplingCore.Weight(c.Entries[0], 4, 4);
        Assert.That(w.Count(v => v == 2f), Is.EqualTo(4));
        Assert.That(w[0 * 4 + 1], Is.EqualTo(2f));
        Assert.That(w[1 * 4 + 2], Is.EqualTo(2f));
        Assert.That(w[0], Is.EqualTo(0f));
        Assert.That(w[2 * 4 + 1], Is.EqualTo(0f));
    }

    [Test]
    public void Percentage_area_resolves_against_the_latent_size()
    {
        var c = new ConditioningSetAreaPercentage().Execute(Prompt(1, 1, 1, 1), width: 0.5f, height: 1f, x: 0.5f, y: 0f);
        Assert.That(c.Entries[0].Area!.Value.Resolve(64, 48), Is.EqualTo((32, 0, 32, 48)));
        Assert.That(c.Entries[0].Area!.Value.Resolve(8, 8), Is.EqualTo((4, 0, 4, 8)));
    }

    [Test]
    public void Mask_weight_follows_the_mask_resized_to_the_latent()
    {
        var mask = new MaskBatch(1, 2, 2, [1, 0, 1, 0]); // left column on
        var c = new ConditioningSetMask().Execute(Prompt(1, 1, 1, 1), mask, strength: 0.5f);
        var w = SamplingCore.Weight(c.Entries[0], 2, 2);
        Assert.That(w, Is.EqualTo(new float[] { 0.5f, 0, 0.5f, 0 }));
        var big = SamplingCore.Weight(c.Entries[0], 4, 4);
        Assert.That(big[0], Is.EqualTo(0.5f).Within(1e-4));
        Assert.That(big[3], Is.EqualTo(0f).Within(1e-4));
    }

    [Test]
    public void Timestep_range_activates_an_entry_only_inside_its_part_of_the_run()
    {
        var m = Sd();
        var always = Prompt(1, 1, 1, 1).Entries[0];
        var firstHalf = new ConditioningSetTimestepRange().Execute(Prompt(1, 1, 1, 1), 0f, 0.5f).Entries[0];
        var lastHalf = new ConditioningSetTimestepRange().Execute(Prompt(1, 1, 1, 1), 0.5f, 1f).Entries[0];
        float high = m.SigmaMax, low = m.SigmaMin;
        Assert.That(SamplingCore.IsActive(always, high, m) && SamplingCore.IsActive(always, low, m), Is.True);
        Assert.That(SamplingCore.IsActive(firstHalf, high, m), Is.True);
        Assert.That(SamplingCore.IsActive(firstHalf, low, m), Is.False);
        Assert.That(SamplingCore.IsActive(lastHalf, high, m), Is.False);
        Assert.That(SamplingCore.IsActive(lastHalf, low, m), Is.True);
    }

    [Test]
    public void Inpaint_mask_grows_by_a_square_window_and_is_cropped_to_the_encoded_size()
    {
        var mask = new MaskBatch(1, 10, 10);
        mask.Data[5 * 10 + 5] = 1f; // one selected pixel
        var odd = VAEEncodeForInpaint.GrowAndCrop(mask, 1, 10, 10, 8, 8, 3);
        Assert.That(odd.Data.Count(v => v == 1f), Is.EqualTo(9), "a 3x3 window around the pixel");
        Assert.That(odd.Data[5 * 8 + 5], Is.EqualTo(1f));
        Assert.That(odd.Data[4 * 8 + 4], Is.EqualTo(1f));
        Assert.That(odd.Data[3 * 8 + 3], Is.EqualTo(0f));
        var none = VAEEncodeForInpaint.GrowAndCrop(mask, 1, 10, 10, 8, 8, 0);
        Assert.That(none.Data.Count(v => v == 1f), Is.EqualTo(1));
        var soft = new MaskBatch(1, 10, 10);
        soft.Data[5 * 10 + 5] = 0.3f; soft.Data[5 * 10 + 6] = 0.4f;
        Assert.That(VAEEncodeForInpaint.GrowAndCrop(soft, 1, 10, 10, 8, 8, 3).Data[5 * 8 + 5], Is.EqualTo(0.7f).Within(1e-6), "overlapping weights add up, clamped to 1");
    }

    [Test]
    public void Set_area_strength_sets_the_strength_of_every_entry_and_leaves_the_input_alone()
    {
        var a = new Conditioning(new float[] { 1, 2 }, new[] { 1, 1, 2 });
        var both = new ConditioningCombine().Execute(a, a);
        var set = new ConditioningSetAreaStrength().Execute(both, 0.25f);
        Assert.That(set.Entries.Select(e => e.Strength), Is.EqualTo(new[] { 0.25f, 0.25f }));
        Assert.That(both.Entries.Select(e => e.Strength), Is.EqualTo(new[] { 1f, 1f }));
    }
}
