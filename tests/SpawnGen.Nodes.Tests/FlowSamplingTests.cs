using System.Text.Json;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// ModelSamplingSD3 / ModelSamplingAuraFlow / ModelSamplingFlux against ComfyUI: Fixtures/flow_sigmas.json is what ComfyUI's BasicScheduler gave
/// after each node for every scheduler (20 and 4 steps, and 10 steps at denoise 0.6), measured black-box.
/// </summary>
public class FlowSamplingTests
{
    public static IEnumerable<TestCaseData> Cases()
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "flow_sigmas.json")));
        foreach (var c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            var inputs = c.GetProperty("inputs");
            string node = c.GetProperty("node").GetString()!, scheduler = c.GetProperty("scheduler").GetString()!;
            int steps = c.GetProperty("steps").GetInt32();
            double denoise = c.GetProperty("denoise").GetDouble();
            var expected = c.GetProperty("sigmas").EnumerateArray().Select(e => e.GetDouble()).ToArray();
            yield return new TestCaseData(node, inputs.GetRawText(), scheduler, steps, denoise, expected).SetName($"{node} {inputs.GetRawText()} {scheduler} {steps} {denoise}");
        }
    }

    [TestCaseSource(nameof(Cases))]
    public void Basic_Scheduler_after_a_flow_sampling_node_gives_ComfyUIs_sigmas(string node, string inputsJson, string scheduler, int steps, double denoise, double[] expected)
    {
        var inputs = JsonDocument.Parse(inputsJson).RootElement;
        double D(string name) => inputs.GetProperty(name).GetDouble();
        var plain = new ModelHandle(null!, new ModelSampling(ModelSigmas.LinearSchedule()));
        var model = node switch
        {
            "ModelSamplingSD3" => new ModelSamplingSD3().Execute(plain, D("shift")),
            "ModelSamplingAuraFlow" => new ModelSamplingAuraFlow().Execute(plain, D("shift"), inputs.GetProperty("sampling").GetString()!),
            "ModelSamplingFlux" => new ModelSamplingFlux().Execute(plain, D("max_shift"), D("base_shift"), inputs.GetProperty("width").GetInt32(), inputs.GetProperty("height").GetInt32()),
            _ => throw new ArgumentException(node),
        };
        Assert.That(model.Sampling.IsConst, Is.True, "a flow model");
        var actual = new BasicScheduler().Execute(model, scheduler, steps, denoise).Values;
        Assert.That(actual.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(actual[i], Is.EqualTo(expected[i]).Within(2e-6 * Math.Max(1, Math.Abs(expected[i]))), $"sigma {i}: {string.Join(", ", actual)} vs {string.Join(", ", expected)}");
    }

    [Test]
    public void Flow_models_are_told_sigma_times_their_multiplier()
    {
        var plain = new ModelHandle(null!, new ModelSampling(ModelSigmas.LinearSchedule()));
        var sd3 = new ModelSamplingSD3().Execute(plain, 8);
        Assert.That(sd3.Sampling.Timestep(0.5), Is.EqualTo(500f));
        Assert.That(sd3.Sampling.Sigmas.Sigma(500), Is.EqualTo(8 * 0.5 / (1 + 7 * 0.5)).Within(1e-6), "the SD3 shift");
        Assert.That(new ModelSamplingAuraFlow().Execute(plain, 1.73, "flow").Sampling.Timestep(0.5), Is.EqualTo(0.5f));
        Assert.That(sd3.Sampling.PercentToSigma(0.25), Is.EqualTo(8 * 0.75 / (1 + 7 * 0.75)).Within(1e-12));
    }
}
