using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// The sampling family's node schemas against ComfyUI's (current master): id, category, display name, socket inputs, widgets in ComfyUI's order (a saved workflow
/// stores widget values by position) and explicitly named outputs. Plus the behaviour of the new nodes that has no numeric fixture.
/// </summary>
public class SamplingNodeSchemaTests
{
    // id, category, display name, socket inputs, widgets in order, output names ComfyUI gives explicitly (lower case; type-named outputs are not compared).
    static readonly string[][] Rows =
    [
        ["KSampler", "model/sampling", "KSampler", "model positive negative latent_image", "seed steps cfg sampler_name scheduler denoise", ""],
        ["KSamplerAdvanced", "model/sampling", "KSampler (Advanced)", "model positive negative latent_image", "add_noise noise_seed steps cfg sampler_name scheduler start_at_step end_at_step return_with_leftover_noise", ""],
        ["SamplerCustom", "model/sampling/custom", "SamplerCustom", "model positive negative sampler sigmas latent_image", "add_noise noise_seed cfg", "output denoised_output"],
        ["SamplerCustomAdvanced", "model/sampling/custom", "SamplerCustomAdvanced", "noise guider sampler sigmas latent_image", "", "output denoised_output"],
        ["BasicGuider", "model/sampling/guiders", "Basic Guider", "model conditioning", "", ""],
        ["CFGGuider", "model/sampling/guiders", "CFG Guider", "model positive negative", "cfg", ""],
        ["DisableNoise", "model/sampling/noise", "DisableNoise", "", "", ""],
        ["RandomNoise", "model/sampling/noise", "RandomNoise", "", "noise_seed", ""],
        ["KSamplerSelect", "model/sampling/samplers", "KSamplerSelect", "", "sampler_name", ""],
        ["BasicScheduler", "model/sampling/schedulers", "BasicScheduler", "model", "scheduler steps denoise", ""],
        ["BetaSamplingScheduler", "model/sampling/schedulers", "BetaSamplingScheduler", "model", "steps alpha beta", ""],
        ["ExponentialScheduler", "model/sampling/schedulers", "ExponentialScheduler", "", "steps sigma_max sigma_min", ""],
        ["KarrasScheduler", "model/sampling/schedulers", "KarrasScheduler", "", "steps sigma_max sigma_min rho", ""],
        ["PolyexponentialScheduler", "model/sampling/schedulers", "PolyexponentialScheduler", "", "steps sigma_max sigma_min rho", ""],
        ["SDTurboScheduler", "model/sampling/schedulers", "SDTurboScheduler", "model", "steps denoise", ""],
        ["VPScheduler", "model/sampling/schedulers", "VPScheduler", "", "steps beta_d beta_min eps_s", ""],
        ["FlipSigmas", "model/sampling/sigmas", "FlipSigmas", "sigmas", "", ""],
        ["SetFirstSigma", "model/sampling/sigmas", "SetFirstSigma", "sigmas", "sigma", ""],
        ["SplitSigmas", "model/sampling/sigmas", "SplitSigmas", "sigmas", "step", "high_sigmas low_sigmas"],
        ["SplitSigmasDenoise", "model/sampling/sigmas", "SplitSigmasDenoise", "sigmas", "denoise", "high_sigmas low_sigmas"],
        ["RescaleCFG", "model/patch", "RescaleCFG", "model", "multiplier", ""],
        ["SamplerEulerAncestral", "model/sampling/samplers", "SamplerEulerAncestral", "", "eta s_noise", ""],
        ["SamplerEulerAncestralCFGPP", "model/sampling/samplers", "SamplerEulerAncestralCFG++", "", "eta s_noise", ""],
        ["SamplerLMS", "model/sampling/samplers", "SamplerLMS", "", "order", ""],
        ["SamplerDPMPP_2M_SDE", "model/sampling/samplers", "SamplerDPMPP_2M_SDE", "", "solver_type eta s_noise noise_device", ""],
        ["SamplerDPMPP_3M_SDE", "model/sampling/samplers", "SamplerDPMPP_3M_SDE", "", "eta s_noise noise_device", ""],
        ["SamplerDPMPP_SDE", "model/sampling/samplers", "SamplerDPMPP_SDE", "", "eta s_noise r noise_device", ""],
        ["SamplerDPMPP_2S_Ancestral", "model/sampling/samplers", "SamplerDPMPP_2S_Ancestral", "", "eta s_noise", ""],
        ["SamplerDPMAdaptative", "model/sampling/samplers", "SamplerDPMAdaptative", "", "order rtol atol h_init pcoeff icoeff dcoeff accept_safety eta s_noise", ""],
        ["SamplerER_SDE", "model/sampling/samplers", "SamplerER_SDE", "", "solver_type max_stage eta s_noise", ""],
        ["SamplerSASolver", "model/sampling/samplers", "SamplerSASolver", "model", "eta sde_start_percent sde_end_percent s_noise predictor_order corrector_order use_pece simple_order_2", ""],
        ["SamplerSEEDS2", "model/sampling/samplers", "SamplerSEEDS2", "", "solver_type eta s_noise r", ""],
        ["AddNoise", "model/sampling/noise", "AddNoise", "model noise sigmas latent_image", "", ""],
        ["CFGOverride", "model/sampling/guiders", "CFG Override", "model", "cfg start_percent end_percent", ""],
        ["DualCFGGuider", "model/sampling/guiders", "Dual CFG Guider", "model cond1 cond2 negative", "cfg_conds cfg_cond2_negative style", ""],
        ["ManualSigmas", "model/sampling/sigmas", "ManualSigmas", "", "sigmas", ""],
        ["LaplaceScheduler", "model/sampling/schedulers", "LaplaceScheduler", "", "steps sigma_max sigma_min mu beta", ""],
        ["ExtendIntermediateSigmas", "model/sampling/sigmas", "ExtendIntermediateSigmas", "sigmas", "steps start_at_sigma end_at_sigma spacing", ""],
        ["SamplingPercentToSigma", "model/sampling/sigmas", "SamplingPercentToSigma", "model", "sampling_percent return_actual_sigma", "sigma_value"],
        ["SamplerLCM", "model/sampling/samplers", "SamplerLCM", "", "s_noise s_noise_end noise_clip_std", ""],
        ["SamplerEulerCFGpp", "experimental", "SamplerEulerCFG++", "", "version", ""],
        ["AlignYourStepsScheduler", "model/sampling/schedulers", "AlignYourStepsScheduler", "", "model_type steps denoise", ""],
        ["GITSScheduler", "model/sampling/schedulers", "GITSScheduler", "", "coeff steps denoise", ""],
        ["OptimalStepsScheduler", "model/sampling/schedulers", "OptimalStepsScheduler", "", "model_type steps denoise", ""],
        ["CFGNorm", "advanced/guidance", "CFGNorm", "model", "strength pre_cfg", "patched_model"],
        ["CFGZeroStar", "advanced/guidance", "CFGZeroStar", "model", "", "patched_model"],
        ["APG", "model/sampling/custom", "Adaptive Projected Guidance", "model", "eta norm_threshold momentum", ""],
        ["Mahiro", "experimental", "Positive-Biased Guidance", "model", "", "patched_model"],
        ["TCFG", "advanced/guidance", "Tangential Damping CFG", "model", "", "patched_model"],
        ["PerpNeg", "experimental", "Perp-Neg (DEPRECATED by Perp-Neg Guider)", "model empty_conditioning", "neg_scale", ""],
        ["PerpNegGuider", "experimental", "Perp-Neg Guider", "model positive negative empty_conditioning", "cfg neg_scale", ""],
        ["FreSca", "experimental", "FreSca", "model", "scale_low scale_high freq_cutoff", ""],
        ["TemporalScoreRescaling", "model/patch/unet", "TSR - Temporal Score Rescaling", "model", "tsr_k tsr_sigma", "patched_model"],
        ["RenormCFG", "model/patch", "RenormCFG", "model", "cfg_trunc renorm_cfg", ""],
        ["DifferentialDiffusion", "experimental", "Differential Diffusion", "model", "strength", ""],
        ["ModelSamplingDiscrete", "model/patch", "ModelSamplingDiscrete", "model", "sampling zsnr", ""],
        ["Epsilon Scaling", "model/patch/unet", "Epsilon Scaling", "model", "scaling_factor", ""],
    ];

    static readonly Lazy<NodeRegistry> Registry = new(() => new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly));

    public static IEnumerable<TestCaseData> Schemas() => Rows.Select(r => new TestCaseData((object)r).SetName("Schema of " + r[0]));

    static string[] Words(string s) => s.Split(' ', StringSplitOptions.RemoveEmptyEntries);

    [TestCaseSource(nameof(Schemas))]
    public void Matches_ComfyUIs_schema(string[] row)
    {
        var schema = Registry.Value.GetSchema(row[0]);
        Assert.That(schema, Is.Not.Null, $"{row[0]} is registered");
        Assert.That(schema!.Category, Is.EqualTo(row[1]), "category");
        Assert.That(schema.DisplayName, Is.EqualTo(row[2]), "display name");
        Assert.That(schema.Inputs.Where(i => i.Widget is null).Select(i => i.Name), Is.EquivalentTo(Words(row[3])), "socket inputs");
        Assert.That(schema.Inputs.Where(i => i.Widget is not null).Select(i => i.Name), Is.EqualTo(Words(row[4])), "widgets in ComfyUI's order");
        if (row[5].Length > 0) Assert.That(schema.Outputs.Select(o => o.Name), Is.EqualTo(Words(row[5])), "output names");
    }

    [Test]
    public void The_sampler_and_scheduler_combos_are_ComfyUIs_lists_in_order()
    {
        // comfy/samplers.py SAMPLER_NAMES (KSAMPLER_NAMES + ddim, uni_pc, uni_pc_bh2) and SCHEDULER_NAMES.
        string[] samplers = ["euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2", "exp_heun_2_x0", "exp_heun_2_x0_sde", "dpm_2", "dpm_2_ancestral",
            "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
            "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",
            "ipndm", "ipndm_v", "deis", "cfgpp_ud10_ab", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp",
            "gradient_estimation", "gradient_estimation_cfg_pp", "er_sde", "seeds_2", "seeds_3", "sa_solver", "sa_solver_pece", "ddim", "uni_pc", "uni_pc_bh2"];
        Assert.That(Samplers.Names, Is.EqualTo(samplers));
        Assert.That(ModelSigmas.SchedulerNames, Is.EqualTo(new[] { "simple", "sgm_uniform", "karras", "exponential", "ddim_uniform", "beta", "normal", "linear_quadratic", "kl_optimal" }));
        var ks = Registry.Value.GetSchema("KSampler")!;
        Assert.That(ks.FindInput("sampler_name")!.Widget!.ResolveOptions(), Is.EqualTo(samplers));
        Assert.That(ks.FindInput("scheduler")!.Widget!.Default, Is.EqualTo("simple"), "a combo without a default starts at its first option in ComfyUI");
    }

    [Test]
    public void Sampler_nodes_pick_ComfyUIs_sampler_and_options()
    {
        Assert.That(new SamplerEulerAncestral().Execute(0.5, 0.8), Is.EqualTo(new SamplerChoice("euler_ancestral", new SamplerOptions { Eta = 0.5, SNoise = 0.8 })));
        Assert.That(new SamplerDPMPP_2M_SDE().Execute("heun", 1, 1, "cpu").Name, Is.EqualTo("dpmpp_2m_sde"));
        Assert.That(new SamplerDPMPP_2M_SDE().Execute("heun", 1, 1, "gpu"), Is.EqualTo(new SamplerChoice("dpmpp_2m_sde_gpu", new SamplerOptions { Eta = 1, SNoise = 1, SolverType = "heun" })));
        Assert.That(new SamplerDPMPP_3M_SDE().Execute().Name, Is.EqualTo("dpmpp_3m_sde_gpu"));
        Assert.That(new SamplerDPMPP_SDE().Execute(r: 0.3).Options!.R, Is.EqualTo(0.3));
        Assert.That(new SamplerLMS().Execute(7).Options!.Order, Is.EqualTo(7));
        Assert.That(new SamplerSEEDS2().Execute("phi_2", 0, 1, 1.0), Is.EqualTo(new SamplerChoice("seeds_2", new SamplerOptions { Eta = 0, SNoise = 1, R = 1.0, SolverType = "phi_2" })));
        Assert.That(new SamplerEulerCFGpp().Execute().Name, Is.EqualTo("euler_cfg_pp"));
        Assert.That(new SamplerEulerCFGpp().Execute("alternative").Name, Is.EqualTo(Samplers.EulerPP));
        var ode = new SamplerER_SDE().Execute("ER-SDE", 3, 0.0, 1.0).Options!;
        Assert.That(ode.SNoise, Is.EqualTo(0.0), "eta 0 is the ODE solver");
        Assert.That(ode.NoiseScaler!(2.0), Is.EqualTo(2.0));
    }

    [Test]
    public void A_patch_node_leaves_the_incoming_model_alone_and_patches_stack_in_order()
    {
        var model = new ModelHandle(null!, new ModelSampling(ModelSigmas.LinearSchedule()));
        var a = new CFGZeroStar().Execute(model);
        var b = new EpsilonScaling().Execute(a, 1.01);
        Assert.That(model.Patches.PostCfg, Is.Empty);
        Assert.That(a.Patches.PostCfg, Has.Count.EqualTo(1));
        Assert.That(b.Patches.PostCfg, Has.Count.EqualTo(2));
        var c = new RenormCFG().Execute(new RescaleCFG().Execute(b), 100, 1);
        Assert.That(c.Patches.CfgFunction, Is.Not.Null, "the newer cfg function replaces the older one, as in ComfyUI");
        Assert.That(c.Patches.PostCfg, Has.Count.EqualTo(2), "post-cfg patches survive a cfg function patch");
        var o = new CFGOverride().Execute(new CFGOverride().Execute(model, 2, 0, 0.5), 5, 0.3, 1);
        Assert.That(o.Patches.CfgOverrides.Select(v => v.Cfg), Is.EqualTo(new[] { 2.0, 5.0 }));
        Assert.That(o.Patches.CfgOverrides[0].SigmaHigh, Is.EqualTo(999999999.9), "start 0 is the very beginning");
        Assert.That(o.Patches.CfgOverrides[1].SigmaLow, Is.EqualTo(0.0), "end 1 is the very end");
    }

    [Test]
    public async Task CFG_override_replaces_cfg_inside_its_range_and_the_override_nearest_the_sampler_wins()
    {
        var sampling = new ModelSampling(ModelSigmas.LinearSchedule());
        var model = new ModelHandle(null!, sampling);
        // Outer override cfg 1 over the whole run, inner (added later, nearer the sampler) cfg 4 for the first half.
        var patched = new CFGOverride().Execute(new CFGOverride().Execute(model, 1, 0, 1), 4, 0, 0.5);
        var pos = new Conditioning(new float[4], [1, 1, 4]);
        var neg = new Conditioning(new float[4], [1, 1, 4]);
        Task<float[]> Eval(Conditioning c, bool scheduled) => Task.FromResult(c == pos ? new[] { 1f } : new[] { 0f });
        var run = new GuidanceRun(new CfgGuider(patched, pos, neg, 7), 1, 1, 1, false);
        double early = sampling.PercentToSigma(0.2), late = sampling.PercentToSigma(0.8);
        Assert.That((await run.PredictAsync([0f], early, Eval))[0], Is.EqualTo(4f), "first half: the inner override");
        Assert.That((await run.PredictAsync([0f], late, Eval))[0], Is.EqualTo(1f), "second half: the outer override (cfg 1: the negative is not even asked)");
        var plain = new GuidanceRun(new CfgGuider(model, pos, neg, 7), 1, 1, 1, false);
        Assert.That((await plain.PredictAsync([0f], late, Eval))[0], Is.EqualTo(7f));
    }

    [Test]
    public async Task Basic_guider_has_no_negative_so_its_unconditional_estimate_is_zero()
    {
        var model = new ModelHandle(null!, new ModelSampling(ModelSigmas.LinearSchedule()));
        var pos = new Conditioning(new float[4], [1, 1, 4]);
        int calls = 0;
        Task<float[]> Eval(Conditioning c, bool scheduled) { calls++; return Task.FromResult(new[] { 0.5f, -0.25f }); }
        var run = new GuidanceRun((GuiderChoice)new BasicGuider().Execute(model, pos), 1, 1, 2, samplerDisablesCfg1: true);
        var d = await run.PredictAsync([0f, 0f], 3.0, Eval);
        Assert.That(d, Is.EqualTo(new[] { 0.5f, -0.25f }));
        Assert.That(calls, Is.EqualTo(1), "only the prompt is evaluated");
        Assert.That(run.Last!.HasUncond, Is.False);
        Assert.That(run.Last.UncondDenoised, Is.EqualTo(new[] { 0f, 0f }), "a CFG++ sampler sees zero, as in ComfyUI");
    }

    [Test]
    public void ModelSamplingDiscrete_switches_the_prediction_type_and_schedule()
    {
        var model = new ModelHandle(null!, new ModelSampling(ModelSigmas.LinearSchedule()));
        var v = new ModelSamplingDiscrete().Execute(model, "v_prediction");
        Assert.That(v.Sampling.Prediction, Is.EqualTo(PredictionType.VPrediction));
        Assert.That(v.Sampling.SigmaMax, Is.EqualTo(model.Sampling.SigmaMax));
        var z = new ModelSamplingDiscrete().Execute(model, "eps", zsnr: true);
        Assert.That(z.Sampling.SigmaMax, Is.GreaterThan(1000), "zero terminal SNR: the last timestep is (almost) pure noise");
        var lcm = new ModelSamplingDiscrete().Execute(model, "lcm");
        Assert.That(lcm.Sampling.Sigmas.TrainingSteps, Is.EqualTo(50));
        Assert.That(lcm.Sampling.Timestep(lcm.Sampling.SigmaMin), Is.EqualTo(19f), "distilled timesteps are told as index * 20 + 19");
        Assert.That(new ModelSamplingDiscrete().Execute(model, "img_to_img_flow").Sampling.IsConst, Is.True);
        Assert.That(() => new ModelSamplingDiscrete().Execute(model, "nope"), Throws.ArgumentException);
    }

    [Test]
    public void AddNoise_noises_the_latent_to_the_span_of_the_sigmas()
    {
        var model = new ModelHandle(null!, new ModelSampling(ModelSigmas.LinearSchedule()));
        var latent = new LatentBatch(2, 4, 8, 8);
        for (int i = 0; i < latent.Data.Length; i++) latent.Data[i] = 0.25f;
        var noise = new RandomNoise().Execute(3);
        var noisy = new AddNoise().Execute(model, noise, new Sigmas([6f, 2f]), latent);
        var expected0 = SamplingCore.InitialNoise(3, 0, latent);
        var expected1 = SamplingCore.InitialNoise(3, 1, latent);
        // ComfyUI: process_latent_out(noise_scaling(|6 - 2|, noise, process_latent_in(latent))), the noise added in the model's space (x 0.18215).
        const float k = 0.18215f;
        for (int i = 0; i < 256; i++)
        {
            Assert.That(noisy.Data[i], Is.EqualTo((0.25f * k + expected0[i] * 4f) / k).Within(1e-4), "|6 - 2| = 4");
            Assert.That(noisy.Data[256 + i], Is.EqualTo((0.25f * k + expected1[i] * 4f) / k).Within(1e-4), "each picture its own noise");
        }
        Assert.That(new AddNoise().Execute(model, noise, new Sigmas([]), latent), Is.SameAs(latent));
        Assert.That(new AddNoise().Execute(model, new DisableNoise().Execute(), new Sigmas([3f]), latent).Data, Is.EqualTo(latent.Data), "no noise: unchanged");
    }

    [Test]
    public void Brownian_noise_has_unit_variance_and_correlates_overlapping_intervals()
    {
        var source = new Samplers.NoiseSource(11, 4, 64, 64);
        var b = new Samplers.BrownianNoise(source, 0.1, 10.0);
        var whole = b.Increment(10.0, 1.0);   // the interval [1, 10]
        var part = b.Increment(10.0, 4.0);    // inside it: [4, 10]
        var other = b.Increment(1.0, 0.5);    // outside it
        static (double Mean, double Var) Stats(float[] v) { double m = v.Average(x => (double)x); return (m, v.Sum(x => (x - m) * (x - m)) / v.Length); }
        static double Corr(float[] a, float[] c) => a.Zip(c, (p, q) => (double)p * q).Average();
        foreach (var v in new[] { whole, part, other })
        {
            var (mean, var) = Stats(v);
            Assert.That(mean, Is.EqualTo(0).Within(0.03));
            Assert.That(var, Is.EqualTo(1).Within(0.04));
        }
        Assert.That(Corr(whole, part), Is.EqualTo(Math.Sqrt(6.0 / 9.0)).Within(0.03), "Brownian increments over nested intervals: sqrt(6 / 9)");
        Assert.That(Corr(whole, other), Is.EqualTo(0).Within(0.03), "disjoint intervals are independent");
    }

    [Test]
    public void KSampler_Advanced_slices_like_ComfyUI_even_when_the_start_is_past_the_end()
    {
        float[] s = [10, 8, 6, 4, 2, 0];
        Assert.That(KSamplerAdvanced.Slice(s, 3, 2, true), Is.Empty, "start past a cut end: nothing runs");
        Assert.That(KSamplerAdvanced.Slice(s, 0, 0, true), Is.Empty);
        Assert.That(KSamplerAdvanced.Slice(s, 2, 4, false), Is.EqualTo(new float[] { 6, 4, 2 }));
    }
}
