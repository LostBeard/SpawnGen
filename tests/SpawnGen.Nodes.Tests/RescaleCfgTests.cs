using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>RescaleCFG's maths on synthetic predictions: multiplier 0 is plain guidance, 1 gives the prompt's own spread, and the node only sets a flag on the model.</summary>
public class RescaleCfgTests
{
    static float[] Wave(int n, float amplitude, float phase) => Enumerable.Range(0, n).Select(i => amplitude * MathF.Sin(i * 0.37f + phase)).ToArray();

    static (float[] Sample, float[] Cond, float[] Uncond) Inputs()
    {
        const int n = 256;
        return (Wave(n, 3f, 0.2f), Wave(n, 0.9f, 1.1f), Wave(n, 0.6f, 2.0f));
    }

    static double Std(IEnumerable<double> v)
    {
        var a = v.ToArray();
        double mean = a.Average();
        return Math.Sqrt(a.Sum(x => (x - mean) * (x - mean)) / (a.Length - 1));
    }

    [Test]
    public void Multiplier_zero_is_plain_classifier_free_guidance()
    {
        var (x, cond, uncond) = Inputs();
        const float cfg = 7f, sigma = 2.5f;
        var result = new float[x.Length];
        SamplingCore.RescaledDenoised(x, cond, uncond, cfg, sigma, 0f, result);
        for (int i = 0; i < x.Length; i++)
        {
            float eps = uncond[i] + cfg * (cond[i] - uncond[i]);
            Assert.That(result[i], Is.EqualTo(x[i] - sigma * eps).Within(2e-4), $"element {i}");
        }
    }

    [Test]
    public void Multiplier_one_gives_the_guided_prediction_the_spread_of_the_prompts_own()
    {
        var (x, cond, uncond) = Inputs();
        const float cfg = 9f, sigma = 1.7f;
        var result = new float[x.Length];
        SamplingCore.RescaledDenoised(x, cond, uncond, cfg, sigma, 1f, result);
        // Back to v-space: v = (x / (sigma k) ) - k * eps_equivalent, with eps_equivalent = (x - denoised) / sigma.
        double k = Math.Sqrt(sigma * sigma + 1);
        var vResult = Enumerable.Range(0, x.Length).Select(i => x[i] / (sigma * k) - k * (x[i] - result[i]) / sigma).ToArray();
        var vCond = Enumerable.Range(0, x.Length).Select(i => x[i] / (sigma * k) - k * cond[i]).ToArray();
        Assert.That(Std(vResult), Is.EqualTo(Std(vCond)).Within(1e-3));
    }

    [Test]
    public void Without_guidance_there_is_nothing_to_rescale()
    {
        var (x, cond, _) = Inputs();
        var result = new float[x.Length];
        SamplingCore.RescaledDenoised(x, cond, cond, 1f, 3f, 0.7f, result); // cfg 1 and equal predictions: the guided one is the prompt's own
        for (int i = 0; i < x.Length; i++) Assert.That(result[i], Is.EqualTo(x[i] - 3f * cond[i]).Within(2e-4));
    }

    [Test]
    public void The_node_marks_the_model_and_leaves_the_checkpoint_alone()
    {
        var model = new ModelHandle(null!);
        var rescaled = new RescaleCFG().Execute(model, 0.7f);
        Assert.That(rescaled.RescaleCfg, Is.EqualTo(0.7f));
        Assert.That(model.RescaleCfg, Is.EqualTo(0f), "the model that came in is unchanged");
    }
}
