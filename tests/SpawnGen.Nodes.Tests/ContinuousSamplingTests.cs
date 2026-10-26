using System.Globalization;
using System.Text.Json;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// ModelSamplingContinuousEDM / ModelSamplingContinuousV against ComfyUI's own model samplings (Fixtures/model_sampling/make_continuous.py -> continuous.json):
/// the sigma table, every scheduler's sigmas, timestep and sigma, percent_to_sigma, and the input / denoised / noise formulas, to float32 precision.
/// </summary>
public class ContinuousSamplingTests
{
    static readonly JsonElement Fixture = JsonDocument.Parse(File.ReadAllText(
        Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "model_sampling", "continuous.json"))).RootElement;

    static ModelSampling Make(string name)
    {
        var d = Fixture.GetProperty(name);
        double max = d.GetProperty("sigma_max").GetDouble(), min = d.GetProperty("sigma_min").GetDouble();
        return name switch
        {
            "edm_v" => new ModelSampling(ModelSigmas.Continuous(min, max, v: false), PredictionType.VPrediction),
            "edm_edm" => new ModelSampling(ModelSigmas.Continuous(min, max, v: false), PredictionType.Edm, 0.5),
            "edm_eps" => new ModelSampling(ModelSigmas.Continuous(min, max, v: false), PredictionType.Eps),
            "cont_v" => new ModelSampling(ModelSigmas.Continuous(min, max, v: true), PredictionType.VPrediction),
            _ => throw new ArgumentException(name),
        };
    }

    static void Close(double ours, double theirs, string what) =>
        Assert.That(ours, Is.EqualTo(theirs).Within(Math.Max(Math.Abs(theirs), 1e-3) * 2e-6), what);

    [TestCase("edm_v")]
    [TestCase("edm_edm")]
    [TestCase("edm_eps")]
    [TestCase("cont_v")]
    public void Matches_ComfyUIs_continuous_model_sampling(string name)
    {
        var d = Fixture.GetProperty(name);
        var s = Make(name);
        int[] at = [0, 1, 2, 499, 500, 501, 997, 998, 999];
        var table = d.GetProperty("table").EnumerateArray().Select(v => v.GetDouble()).ToArray();
        for (int i = 0; i < at.Length; i++) Close(s.Sigmas.TableSigma(at[i]), table[i], $"table[{at[i]}]");

        foreach (var sch in d.GetProperty("schedules").EnumerateObject())
        {
            var theirs = sch.Value.EnumerateArray().Select(v => v.GetDouble()).ToArray();
            var ours = s.Sigmas.Schedule(sch.Name, 10);
            Assert.That(ours, Has.Length.EqualTo(theirs.Length), $"{sch.Name}: length");
            for (int i = 0; i < theirs.Length; i++) Close(ours[i], theirs[i], $"{sch.Name}[{i}]");
        }
        foreach (var t in d.GetProperty("timestep").EnumerateObject())
            Close(s.Timestep(double.Parse(t.Name, CultureInfo.InvariantCulture)), t.Value.GetDouble(), $"timestep({t.Name})");
        foreach (var t in d.GetProperty("sigma").EnumerateObject())
            Close(s.Sigmas.Sigma(double.Parse(t.Name, CultureInfo.InvariantCulture)), t.Value.GetDouble(), $"sigma({t.Name})");
        foreach (var p in d.GetProperty("percent").EnumerateObject())
            Close(s.PercentToSigma(double.Parse(p.Name, CultureInfo.InvariantCulture)), p.Value.GetDouble(), $"percent_to_sigma({p.Name})");

        var x = Enumerable.Range(0, 12).Select(i => (float)(-1.5 + 3.0 * i / 11)).ToArray();
        var output = Enumerable.Range(0, 12).Select(i => (float)(0.7 - 1.1 * i / 11)).ToArray();
        var input = s.CalculateInput(3.5, x);
        var denoised = s.CalculateDenoised(3.5, output, x);
        var noised = s.NoiseScaling(3.5, output, x, maxDenoise: true);
        string[] names = ["input", "denoised", "noise_scaling"];
        float[][] ours2 = [input, denoised, noised];
        for (int k = 0; k < 3; k++)
        {
            var theirs = d.GetProperty(names[k]).EnumerateArray().Select(v => v.GetDouble()).ToArray();
            for (int i = 0; i < 12; i++) Assert.That(ours2[k][i], Is.EqualTo(theirs[i]).Within(1e-5), $"{names[k]}[{i}]");
        }
    }

    [Test]
    public void The_nodes_build_those_samplings_and_refuse_what_SpawnGen_cannot_run()
    {
        var model = new ModelHandle(null!, new ModelSampling(ModelSigmas.LinearSchedule()));
        var edm = new ModelSamplingContinuousEDM().Execute(model, "edm", 120, 0.002);
        Assert.That((edm.Sampling.Prediction, edm.Sampling.SigmaData, edm.Sampling.Sigmas.Kind), Is.EqualTo((PredictionType.Edm, 0.5, ModelSigmas.Continuity.ContinuousEdm)));
        var v = new ModelSamplingContinuousV().Execute(model, "v_prediction", 500, 0.03);
        Assert.That((v.Sampling.Prediction, v.Sampling.Sigmas.Kind), Is.EqualTo((PredictionType.VPrediction, ModelSigmas.Continuity.ContinuousV)));
        Assert.That(() => new ModelSamplingContinuousEDM().Execute(model, "edm_playground_v2.5", 120, 0.002), Throws.TypeOf<NotSupportedException>());
        Assert.That(() => edm.Sampling.Sigmas.ZeroTerminalSnr(), Throws.TypeOf<NotSupportedException>());
    }
}
