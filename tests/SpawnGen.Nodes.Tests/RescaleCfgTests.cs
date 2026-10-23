using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// RescaleCFG's maths on synthetic predictions: multiplier 0 is plain guidance, 1 gives the guided v-prediction the spread of the prompt's own, and the node only adds
/// a cfg function to a copy of the model. (Its values against ComfyUI's own code are in SamplerParityTests, cases RescaleCFG.)
/// </summary>
public class RescaleCfgTests
{
    static float[] Wave(int n, float amplitude, float phase) => Enumerable.Range(0, n).Select(i => amplitude * MathF.Sin(i * 0.37f + phase)).ToArray();

    static PatchContext Context() => new()
    {
        Channels = 4, Height = 8, Width = 8, Sampling = new ModelSampling(SamplingSetupTests.Sd()), EvaluateUnscheduled = _ => throw new NotSupportedException(),
    };

    /// <summary>The cfg-function arguments for denoised estimates x - sigma * eps of the two noise predictions.</summary>
    static CfgFunctionArgs Args(float[] x, float[] epsCond, float[] epsUncond, float cfg, float sigma)
    {
        var condDen = x.Select((v, i) => v - sigma * epsCond[i]).ToArray();
        var uncondDen = x.Select((v, i) => v - sigma * epsUncond[i]).ToArray();
        return new CfgFunctionArgs(x.Select((v, i) => v - condDen[i]).ToArray(), x.Select((v, i) => v - uncondDen[i]).ToArray(), cfg, sigma, x, condDen, uncondDen, true, true, Context());
    }

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
        var r = GuidancePatches.RescaleCfg(Args(x, cond, uncond, cfg, sigma), 0.0, isFlow: false);
        for (int i = 0; i < x.Length; i++)
        {
            float eps = uncond[i] + cfg * (cond[i] - uncond[i]);
            Assert.That(x[i] - r[i], Is.EqualTo(x[i] - sigma * eps).Within(2e-4), $"element {i}");
        }
    }

    [Test]
    public void Multiplier_one_gives_the_guided_v_prediction_the_spread_of_the_prompts_own()
    {
        var (x, cond, uncond) = Inputs();
        const float cfg = 9f, sigma = 1.7f;
        var r = GuidancePatches.RescaleCfg(Args(x, cond, uncond, cfg, sigma), 1.0, isFlow: false);
        // v = sqrt(1 + sigma^2) * eps - sigma / sqrt(1 + sigma^2) * x, the model's velocity in k-diffusion space; eps of the result is r / sigma.
        double k = Math.Sqrt(sigma * sigma + 1);
        var vResult = Enumerable.Range(0, x.Length).Select(i => k * (r[i] / sigma) - sigma / k * x[i]).ToArray();
        var vCond = Enumerable.Range(0, x.Length).Select(i => k * cond[i] - sigma / k * x[i]).ToArray();
        Assert.That(Std(vResult), Is.EqualTo(Std(vCond)).Within(1e-3));
    }

    [Test]
    public void Without_guidance_there_is_nothing_to_rescale()
    {
        var (x, cond, _) = Inputs();
        var r = GuidancePatches.RescaleCfg(Args(x, cond, cond, 1f, 3f), 0.7, isFlow: false); // cfg 1 and equal predictions: the guided one is the prompt's own
        for (int i = 0; i < x.Length; i++) Assert.That(x[i] - r[i], Is.EqualTo(x[i] - 3f * cond[i]).Within(2e-4));
    }

    [Test]
    public void The_node_adds_a_cfg_function_to_a_copy_of_the_model()
    {
        var model = new ModelHandle(null!, new ModelSampling(SamplingSetupTests.Sd()));
        var rescaled = new RescaleCFG().Execute(model, 0.7);
        Assert.That(rescaled.Patches.CfgFunction, Is.Not.Null);
        Assert.That(model.Patches.CfgFunction, Is.Null, "the model that came in is unchanged");
        Assert.That(rescaled.Sampling, Is.SameAs(model.Sampling));
    }
}
