using System.Text.Json;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Every sampler, guidance patch, guider and scheduler against ComfyUI's own code: Fixtures/sampling/reference.json is written by make_reference.py, which runs
/// ComfyUI's k-diffusion samplers, cfg functions, patch nodes and schedulers on a stand-in denoiser with recorded noise. Here the same denoiser and noise go
/// through SpawnGen's code and the results must agree to float precision.
/// </summary>
public class SamplerParityTests
{
    static readonly Lazy<JsonElement> Reference = new(() =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "sampling", "reference.json"))).RootElement);

    static double Number(JsonElement v) => v.ValueKind == JsonValueKind.String ? double.Parse(v.GetString()!.Replace("inf", "Infinity"), System.Globalization.CultureInfo.InvariantCulture) : v.GetDouble();
    static float[] Floats(JsonElement e) => [.. e.EnumerateArray().Select(v => (float)Number(v))];
    static double[] Doubles(JsonElement e) => [.. e.EnumerateArray().Select(Number)];

    static readonly Lazy<ModelSigmas> Sd = new(() => ModelSigmas.LinearSchedule());

    static ModelSampling Sampling(string kind) => new(Sd.Value, kind switch
    {
        "img_to_img_flow" => PredictionType.ImgToImgFlow,
        "v_prediction" => PredictionType.VPrediction,
        "x0" => PredictionType.X0,
        "img_to_img" => PredictionType.ImgToImg,
        _ => PredictionType.Eps,
    });

    /// <summary>The stand-in denoiser of make_reference.py: data N(mu, 1) plus a small tanh / sin term, per conditioning.</summary>
    sealed class StandIn
    {
        readonly Dictionary<Conditioning, (float[] Mu, double Amp, bool Tanh)> _conds = new(ReferenceEqualityComparer.Instance);
        public readonly Conditioning C = Make(), U = Make(), M = Make(), E = Make();
        public int Calls;

        public StandIn()
        {
            var r = Reference.Value;
            foreach (var (cond, key) in new[] { (C, "C"), (U, "U"), (M, "M"), (E, "E") })
                _conds[cond] = (Floats(r.GetProperty("mu").GetProperty(key)), r.GetProperty("amp").GetProperty(key).GetDouble(), key is "C" or "M");
        }

        static Conditioning Make() => new(new float[4], [1, 1, 4]);

        public Task<float[]> Evaluate(Conditioning cond, bool scheduled, float[] x, double sigma)
        {
            Calls++;
            var (mu, amp, tanh) = _conds[cond];
            var r = new float[x.Length];
            double s = (float)sigma, k = 1.0 / (1.0 + s * s);
            for (int i = 0; i < r.Length; i++)
            {
                double xd = x[i], f = tanh ? Math.Tanh(xd) : Math.Sin(xd);
                r[i] = (float)(mu[i] + k * (xd - mu[i]) + amp * f * s / (1.0 + s));
            }
            return Task.FromResult(r);
        }
    }

    /// <summary>Replays the recorded draws in order.</summary>
    sealed class RecordedNoise(float[][] draws) : Samplers.NoiseSource(0, 1, 1, 1)
    {
        public override float[] At(int index) => index < draws.Length ? draws[index]
            : throw new InvalidOperationException($"The sampler asked for noise draw {index}; ComfyUI made only {draws.Length}.");
    }

    static SamplerOptions Options(JsonElement o)
    {
        double? D(string n) => o.TryGetProperty(n, out var v) ? v.GetDouble() : null;
        int? I(string n) => o.TryGetProperty(n, out var v) ? v.GetInt32() : null;
        return new SamplerOptions
        {
            Eta = D("eta"), SNoise = D("s_noise"), R = D("r"), SolverType = o.TryGetProperty("solver_type", out var st) ? st.GetString() : null, Order = I("order"),
            Rtol = D("rtol"), Atol = D("atol"), HInit = D("h_init"), PCoeff = D("pcoeff"), ICoeff = D("icoeff"), DCoeff = D("dcoeff"), AcceptSafety = D("accept_safety"),
            SNoiseEnd = D("s_noise_end"), NoiseClipStd = D("noise_clip_std"),
        };
    }

    public static IEnumerable<TestCaseData> Cases() => SamplerCaseNames().Select(n => new TestCaseData(n).SetName("Sampler parity: " + n));

    static IEnumerable<string> SamplerCaseNames()
    {
        // NUnit lists the cases before the fixture's TestDirectory is known in some runners, so read the file next to the assembly.
        var file = Path.Combine(Path.GetDirectoryName(typeof(SamplerParityTests).Assembly.Location)!, "Fixtures", "sampling", "reference.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(file));
        return [.. doc.RootElement.GetProperty("cases").EnumerateArray().Select(c => c.GetProperty("name").GetString()!)];
    }

    static JsonElement Case(string name) => Reference.Value.GetProperty("cases").EnumerateArray().First(c => c.GetProperty("name").GetString() == name);

    [TestCaseSource(nameof(Cases))]
    public async Task Matches_ComfyUI(string name)
    {
        var c = Case(name);
        var sampling = Sampling(c.GetProperty("kind").GetString()!);
        var standIn = new StandIn();
        var handle = new ModelHandle(null!, sampling);
        string sampler = c.GetProperty("sampler").GetString()!;
        var options = Options(c.GetProperty("options"));
        double cfg = c.GetProperty("cfg").GetDouble();

        // The node, patch or guider the case was made with.
        GuiderChoice guider = new CfgGuider(handle, standIn.C, standIn.U, cfg);
        if (c.TryGetProperty("widgets", out var widgets) && widgets.ValueKind == JsonValueKind.Array)
        {
            var w = widgets.EnumerateArray().ToArray();
            var choice = name.Split('/')[0] switch
            {
                "SamplerER_SDE" => new SamplerER_SDE().Execute(w[0].GetString()!, w[1].GetInt32(), w[2].GetDouble(), w[3].GetDouble()),
                "SamplerSASolver" => new SamplerSASolver().Execute(handle, w[0].GetDouble(), w[1].GetDouble(), w[2].GetDouble(), w[3].GetDouble(), w[4].GetInt32(), w[5].GetInt32(), w[6].GetBoolean(), w[7].GetBoolean()),
                "SamplerEulerCFGpp" => new SamplerEulerCFGpp().Execute(w[0].GetString()!),
                var other => throw new InvalidOperationException(other),
            };
            sampler = choice.Name;
            options = choice.Options ?? new SamplerOptions();
        }
        if (c.TryGetProperty("patch", out var patch)) guider = new CfgGuider(Patched(patch.GetString()!, handle, standIn), standIn.C, standIn.U, cfg);
        if (c.TryGetProperty("guider", out var g)) guider = Guider(g.GetString()!, handle, standIn);

        var run = new GuidanceRun(guider, 2, 3, 4, Samplers.DisablesCfg1Optimization(sampler));
        int calls = 0;
        float[]? lastDenoised = null;
        Samplers.DenoiseFn model = (x, sigma) => { calls++; return run.PredictAsync(x, sigma, (cond, scheduled, _) => standIn.Evaluate(cond, scheduled, x, sigma)); };
        var draws = c.GetProperty("draws").EnumerateArray().Select(Floats).ToArray();
        var result = await Samplers.RunAsync(sampler, Floats(c.GetProperty("x")), Floats(c.GetProperty("sigmas")), model, new RecordedNoise(draws),
            uncond: () => run.Last, options: options, sampling: sampling, onDenoised: d => { lastDenoised = d; return Task.CompletedTask; });

        Assert.That(calls, Is.EqualTo(c.GetProperty("calls").GetInt32()), "model calls");
        AssertClose(result, Floats(c.GetProperty("expected")), "sample");
        if (c.GetProperty("last_denoised").ValueKind == JsonValueKind.Array) AssertClose(lastDenoised!, Floats(c.GetProperty("last_denoised")), "last callback estimate");
    }

    static ModelHandle Patched(string patch, ModelHandle m, StandIn standIn) => patch switch
    {
        "CFGZeroStar" or "CFGZeroStar/dpmpp_2m_cfg_pp" => new CFGZeroStar().Execute(m),
        "CFGNorm" => new CFGNorm().Execute(m, 0.8, false),
        "CFGNorm/pre" => new CFGNorm().Execute(m, 0.7, true),
        "APG" => new APG().Execute(m, 0.5, 3.0, 0.0),
        "APG/momentum" => new APG().Execute(m, 1.5, 0.5, -0.5),
        "APG/cfg1" => new APG().Execute(m, 0.5, 3.0, 0.3),
        "Mahiro" => new Mahiro().Execute(m),
        "TCFG" or "TCFG/cfg1" => new TCFG().Execute(m),
        "FreSca" => new FreSca().Execute(m, 0.8, 1.4, 1),
        "FreSca/wide" => new FreSca().Execute(m, 1.2, 0.9, 20),
        "Epsilon Scaling" => new EpsilonScaling().Execute(m, 1.02),
        "TemporalScoreRescaling" => new TemporalScoreRescaling().Execute(m, 0.9, 0.7),
        "RenormCFG" => new RenormCFG().Execute(m, 100.0, 1.0),
        "RenormCFG/trunc" => new RenormCFG().Execute(m, 3.0, 0.5),
        "RescaleCFG" => new RescaleCFG().Execute(m, 0.7),
        "RescaleCFG/full" => new RescaleCFG().Execute(m, 1.0),
        "PerpNeg" => new PerpNeg().Execute(m, standIn.E, 0.8),
        "chain" => new EpsilonScaling().Execute(new CFGZeroStar().Execute(new TCFG().Execute(m)), 1.01),
        _ => throw new InvalidOperationException(patch),
    };

    static GuiderChoice Guider(string guider, ModelHandle m, StandIn s)
    {
        var p = guider.Split('/');
        if (p[0] == "DualCFGGuider") return new DualCFGGuider().Execute(m, s.C, s.M, s.U, double.Parse(p[2]), double.Parse(p[3]), p[1]);
        if (guider == "PerpNegGuider/cfg_pp_zerostar") return new PerpNegGuider().Execute(new CFGZeroStar().Execute(m), s.C, s.U, s.E, 3.0, 0.5);
        return new PerpNegGuider().Execute(m, s.C, s.U, s.E, double.Parse(p[1]), double.Parse(p[2]));
    }

    static void AssertClose(float[] actual, float[] expected, string what, double tolerance = 2e-5)
    {
        Assert.That(actual, Has.Length.EqualTo(expected.Length), what);
        double worst = 0;
        int at = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            double err = Math.Abs(actual[i] - expected[i]) / (1 + Math.Abs(expected[i]));
            if (!(err <= worst)) { worst = err; at = i; }
        }
        TestContext.Out.WriteLine($"{what}: worst relative difference {worst:E2} over {expected.Length} values");
        Assert.That(worst, Is.LessThanOrEqualTo(tolerance), $"{what}: element {at} is {(at >= 0 ? actual[at] : 0)}, ComfyUI {(at >= 0 ? expected[at] : 0)}");
    }

    [Test]
    public async Task Inpainting_re_noises_the_kept_area_and_pins_the_output_like_KSamplerX0Inpaint()
    {
        var sampling = Sampling("eps");
        foreach (var c in Reference.Value.GetProperty("inpaint").EnumerateArray())
        {
            string name = c.GetProperty("name").GetString()!, sampler = c.GetProperty("sampler").GetString()!;
            var sigmas = Floats(c.GetProperty("sigmas"));
            var runSigmas = sigmas.Select(v => (double)v).ToArray();
            var latent = Floats(c.GetProperty("latent"));
            var noise = Floats(c.GetProperty("noise"));
            var mask = Floats(c.GetProperty("mask"));
            var handle = new ModelHandle(null!, sampling);
            if (c.GetProperty("dd_strength").ValueKind == JsonValueKind.Number) handle = new DifferentialDiffusion().Execute(handle, c.GetProperty("dd_strength").GetDouble());
            var standIn = new StandIn();
            var run = new GuidanceRun(new CfgGuider(handle, standIn.C, standIn.U, c.GetProperty("cfg").GetDouble()), 2, 3, 4, false);
            int calls = 0;
            float[]? lastDenoised = null;
            Samplers.DenoiseFn model = (x, sigma) =>
            {
                calls++;
                return SamplingCore.InpaintedCall(x, sigma, mask, handle.Patches.DenoiseMask, runSigmas, sampling, latent, noise, 12,
                    input => run.PredictAsync(input, sigma, (cond, scheduled, _) => standIn.Evaluate(cond, scheduled, input, sigma)));
            };
            var x0 = sampling.NoiseScaling(sigmas[0], noise, latent, sampling.IsMaxDenoise(sigmas[0]));
            var draws = c.GetProperty("draws").EnumerateArray().Select(Floats).ToArray();
            var result = await Samplers.RunAsync(sampler, x0, sigmas, model, new RecordedNoise(draws), sampling: sampling, onDenoised: d => { lastDenoised = d; return Task.CompletedTask; });
            result = sampling.InverseNoiseScaling(sigmas[^1], result);
            Assert.That(calls, Is.EqualTo(c.GetProperty("calls").GetInt32()), name + " model calls");
            AssertClose(result, Floats(c.GetProperty("expected")), name);
            AssertClose(lastDenoised!, Floats(c.GetProperty("last_denoised")), name + " last estimate");
        }
    }

    // --- schedulers and sigma nodes ------------------------------------------------------------------------------------------------------

    [Test]
    public void The_model_sampling_has_ComfyUIs_range_timesteps_and_percent_mapping()
    {
        var f = Reference.Value.GetProperty("model_sampling");
        var sd = Sd.Value;
        Assert.That(sd.SigmaMin, Is.EqualTo(f.GetProperty("sigma_min").GetDouble()).Within(1e-7));
        Assert.That(sd.SigmaMax, Is.EqualTo(f.GetProperty("sigma_max").GetDouble()).Within(1e-5));
        foreach (var pair in f.GetProperty("timesteps").EnumerateArray())
            Assert.That(sd.Timestep(pair[0].GetDouble()), Is.EqualTo(pair[1].GetDouble()), $"timestep of sigma {pair[0]}");
        foreach (var pair in f.GetProperty("percent").EnumerateArray())
            Assert.That(sd.PercentToSigma(pair[0].GetDouble()), Is.EqualTo(pair[1].GetDouble()).Within(1e-5 * (1 + pair[1].GetDouble())), $"percent {pair[0]}");
        var lcm = sd.Distilled();
        foreach (var pair in f.GetProperty("lcm_timesteps").EnumerateArray())
            Assert.That(lcm.Timestep(pair[0].GetDouble()), Is.EqualTo(pair[1].GetDouble()), $"LCM timestep of sigma {pair[0]}");
        foreach (var pair in f.GetProperty("lcm_sigma").EnumerateArray())
            Assert.That(lcm.Sigma(pair[0].GetDouble()), Is.EqualTo(pair[1].GetDouble()).Within(1e-5 * (1 + pair[1].GetDouble())), $"LCM sigma at {pair[0]}");
        Assert.That(sd.ZeroTerminalSnr().SigmaMax, Is.EqualTo(f.GetProperty("zsnr_sigma_max").GetDouble()).Within(0.05), "zero terminal SNR");
    }

    [Test]
    public void Every_scheduler_gives_ComfyUIs_sigmas()
    {
        foreach (var s in Reference.Value.GetProperty("schedulers").EnumerateArray())
        {
            var table = s.GetProperty("lcm").GetBoolean() ? Sd.Value.Distilled() : s.GetProperty("zsnr").GetBoolean() ? Sd.Value.ZeroTerminalSnr() : Sd.Value;
            string name = s.GetProperty("scheduler").GetString()!;
            int steps = s.GetProperty("steps").GetInt32();
            var what = $"{name} {steps} steps (zsnr {s.GetProperty("zsnr")}, lcm {s.GetProperty("lcm")})";
            var expected = Floats(s.GetProperty("sigmas"));
            var actual = table.Schedule(name, steps);
            if (expected.Any(float.IsNaN)) { Assert.That(actual.Select(float.IsNaN), Is.EqualTo(expected.Select(float.IsNaN)), what); continue; }
            AssertClose(actual, expected, what, 1e-5);
        }
    }

    [Test]
    public void KSampler_drops_the_penultimate_sigma_for_dpm_2_and_uni_pc_and_applies_denoise_like_ComfyUI()
    {
        foreach (var k in Reference.Value.GetProperty("ksampler").EnumerateArray())
        {
            var actual = KSampler.Sigmas(Sd.Value, k.GetProperty("sampler").GetString()!, k.GetProperty("scheduler").GetString()!, k.GetProperty("steps").GetInt32(), k.GetProperty("denoise").GetDouble());
            AssertClose(actual, Floats(k.GetProperty("sigmas")), k.ToString(), 1e-5);
        }
    }

    [Test]
    public void The_schedule_and_sigma_nodes_give_ComfyUIs_values()
    {
        var model = new ModelHandle(null!, Sampling("eps"));
        foreach (var t in Reference.Value.GetProperty("tables").EnumerateArray())
        {
            string node = t.GetProperty("node").GetString()!;
            var a = t.GetProperty("args").EnumerateArray().ToArray();
            var what = $"{node}({string.Join(", ", a.Select(v => v.ToString()))})";
            if (node is "SplitSigmas" or "SplitSigmasDenoise")
            {
                var input = new Sigmas(Floats(t.GetProperty("input")));
                var (high, low) = node == "SplitSigmas" ? new SplitSigmas().Execute(input, a[0].GetInt32()) : new SplitSigmasDenoise().Execute(input, a[0].GetDouble());
                Assert.That(high.Values, Is.EqualTo(Floats(t.GetProperty("high"))), what + " high");
                Assert.That(low.Values, Is.EqualTo(Floats(t.GetProperty("low"))), what + " low");
                continue;
            }
            if (t.TryGetProperty("error", out _))
            {
                Assert.That(() => new ManualSigmas().Execute(a[0].GetString()!), Throws.InstanceOf<FormatException>(), what);
                continue;
            }
            float[] actual = node switch
            {
                "AlignYourStepsScheduler" => new AlignYourStepsScheduler().Execute(a[0].GetString()!, a[1].GetInt32(), a[2].GetDouble()).Values,
                "OptimalStepsScheduler" => new OptimalStepsScheduler().Execute(a[0].GetString()!, a[1].GetInt32(), a[2].GetDouble()).Values,
                "GITSScheduler" => new GITSScheduler().Execute(a[0].GetDouble(), a[1].GetInt32(), a[2].GetDouble()).Values,
                "LaplaceScheduler" => new LaplaceScheduler().Execute(a[0].GetInt32(), a[1].GetDouble(), a[2].GetDouble(), a[3].GetDouble(), a[4].GetDouble()).Values,
                "KarrasScheduler" => new KarrasScheduler().Execute(a[0].GetInt32(), a[1].GetDouble(), a[2].GetDouble(), a[3].GetDouble()).Values,
                "ExponentialScheduler" => new ExponentialScheduler().Execute(a[0].GetInt32(), a[1].GetDouble(), a[2].GetDouble()).Values,
                "PolyexponentialScheduler" => new PolyexponentialScheduler().Execute(a[0].GetInt32(), a[1].GetDouble(), a[2].GetDouble(), a[3].GetDouble()).Values,
                "VPScheduler" => new VPScheduler().Execute(a[0].GetInt32(), a[1].GetDouble(), a[2].GetDouble(), a[3].GetDouble()).Values,
                "SDTurboScheduler" => new SDTurboScheduler().Execute(model, a[0].GetInt32(), a[1].GetDouble()).Values,
                "ExtendIntermediateSigmas" => new ExtendIntermediateSigmas().Execute(new Sigmas(Floats(t.GetProperty("input"))), a[0].GetInt32(), a[1].GetDouble(), a[2].GetDouble(), a[3].GetString()!).Values,
                "ManualSigmas" => new ManualSigmas().Execute(a[0].GetString()!).Values,
                "SamplingPercentToSigma" => [(float)new SamplingPercentToSigma().Execute(model, a[0].GetDouble(), a[1].GetBoolean())],
                _ => throw new InvalidOperationException(node),
            };
            AssertClose(actual, Floats(t.GetProperty("sigmas")), what, 1e-5);
        }
    }

    [Test]
    public void The_prediction_types_scale_the_input_and_form_the_estimate_like_ComfyUI()
    {
        foreach (var p in Reference.Value.GetProperty("predictions").EnumerateArray())
        {
            string kind = p.GetProperty("kind").GetString()!;
            var sampling = kind == "lcm" ? new ModelSampling(Sd.Value.Distilled(), PredictionType.Lcm) : Sampling(kind);
            double sigma = p.GetProperty("sigma").GetDouble();
            int n = 24;
            var xs = ModelSigmas.Linspace(-1.5, 1.7, n).Select(v => (float)v).ToArray();
            var outs = Enumerable.Range(0, n).Select(i => MathF.Cos(i)).ToArray();
            var what = $"{kind} at sigma {sigma}";
            AssertClose(sampling.CalculateInput(sigma, xs), Floats(p.GetProperty("input")), what + " input");
            AssertClose(sampling.CalculateDenoised(sigma, outs, xs), Floats(p.GetProperty("denoised")), what + " denoised");
            AssertClose(sampling.NoiseScaling(sigma, outs, xs), Floats(p.GetProperty("noise_scaling")), what + " noise scaling");
            AssertClose(sampling.NoiseScaling(sigma, outs, xs, maxDenoise: true), Floats(p.GetProperty("noise_scaling_max")), what + " noise scaling from the top");
        }
    }

    [Test]
    public void Differential_diffusion_thresholds_the_mask_like_ComfyUI()
    {
        var sampling = Sampling("eps");
        foreach (var d in Reference.Value.GetProperty("differential_diffusion").EnumerateArray())
        {
            var result = GuidancePatches.DifferentialDiffusionMask(d.GetProperty("sigma").GetDouble(), Floats(d.GetProperty("mask")), Doubles(d.GetProperty("sigmas")),
                sampling, d.GetProperty("strength").GetDouble());
            AssertClose(result, Floats(d.GetProperty("result")), d.ToString());
        }
    }

    [Test]
    public void FreScas_Fourier_filter_matches_torch_fft()
    {
        foreach (var f in Reference.Value.GetProperty("fourier").EnumerateArray())
        {
            int h = f.GetProperty("h").GetInt32(), w = f.GetProperty("w").GetInt32();
            var result = GuidancePatches.FourierFilter(Floats(f.GetProperty("input")), 2, h, w, f.GetProperty("low").GetDouble(), f.GetProperty("high").GetDouble(), f.GetProperty("cutoff").GetInt32());
            AssertClose(result, Floats(f.GetProperty("result")), $"{h}x{w} cutoff {f.GetProperty("cutoff")}");
        }
    }
}
