using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>SamplerCustom's building blocks: sigma schedules from their formulas, split / flip / set-first, and the graph shape.</summary>
public class CustomSamplingTests
{
    static void AssertClose(float[] actual, float[] expected, double relative = 1e-3)
    {
        Assert.That(actual, Has.Length.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++) Assert.That(actual[i], Is.EqualTo(expected[i]).Within(Math.Abs(expected[i]) * relative + 1e-6), $"at {i}");
    }

    [Test]
    public void Karras_and_exponential_with_the_models_limits_equal_the_models_own_schedulers()
    {
        var m = SamplingSetupTests.Sd();
        AssertClose(SigmaSchedules.Karras(20, m.SigmaMax, m.SigmaMin, 7), m.Schedule("karras", 20));
        AssertClose(SigmaSchedules.Exponential(20, m.SigmaMax, m.SigmaMin), m.Schedule("exponential", 20));
    }

    [Test]
    public void Polyexponential_with_rho_one_is_exponential_and_a_higher_rho_drops_faster()
    {
        AssertClose(SigmaSchedules.Polyexponential(10, 14.6, 0.03, 1), SigmaSchedules.Exponential(10, 14.6, 0.03));
        var steep = SigmaSchedules.Polyexponential(10, 14.6, 0.03, 2);
        var flat = SigmaSchedules.Polyexponential(10, 14.6, 0.03, 1);
        Assert.That(steep[2], Is.LessThan(flat[2]), "rho 2 leaves the high noise levels sooner");
        Assert.That(steep[0], Is.EqualTo(14.6f).Within(1e-3)); Assert.That(steep[^2], Is.EqualTo(0.03f).Within(1e-5)); Assert.That(steep[^1], Is.EqualTo(0f));
    }

    [Test]
    public void VP_schedule_matches_the_formula_at_its_ends()
    {
        var s = SigmaSchedules.VariancePreserving(5, 19.9, 0.1, 0.001);
        Assert.That(s[0], Is.EqualTo((float)Math.Sqrt(Math.Exp(19.9 / 2 + 0.1) - 1)).Within(1e-2)); // t = 1
        Assert.That(s[4], Is.EqualTo((float)Math.Sqrt(Math.Exp(19.9 * 1e-6 / 2 + 0.1 * 0.001) - 1)).Within(1e-5)); // t = eps_s
        Assert.That(s[5], Is.EqualTo(0f));
        for (int i = 1; i < s.Length; i++) Assert.That(s[i], Is.LessThan(s[i - 1]));
    }

    [Test]
    public void SD_Turbo_schedule_takes_the_models_sigmas_at_timesteps_999_899_and_so_on()
    {
        var m = SamplingSetupTests.Sd();
        AssertClose(SigmaSchedules.SdTurbo(m, 1, 1f), [m.Sigma(999), 0f]);
        AssertClose(SigmaSchedules.SdTurbo(m, 3, 1f), [m.Sigma(999), m.Sigma(899), m.Sigma(799), 0f]);
        AssertClose(SigmaSchedules.SdTurbo(m, 2, 0.5f), [m.Sigma(499), m.Sigma(399), 0f]); // denoise 0.5 starts 5 entries in
    }

    [Test]
    public void Split_flip_and_set_first_follow_ComfyUI()
    {
        var s = new Sigmas([4f, 3f, 2f, 1f, 0f]);
        var (high, low) = new SplitSigmas().Execute(s, 2);
        Assert.That(high.Values, Is.EqualTo(new[] { 4f, 3f, 2f })); Assert.That(low.Values, Is.EqualTo(new[] { 2f, 1f, 0f }));
        var (h2, l2) = new SplitSigmasDenoise().Execute(s, 0.5f);
        Assert.That(h2.Values, Is.EqualTo(new[] { 4f, 3f })); Assert.That(l2.Values, Is.EqualTo(new[] { 2f, 1f, 0f }));
        var (h3, l3) = new SplitSigmasDenoise().Execute(s, 1f);
        Assert.That(h3.Values, Is.Empty); Assert.That(l3.Values, Is.EqualTo(s.Values));
        Assert.That(new FlipSigmas().Execute(s).Values, Is.EqualTo(new[] { 1e-4f, 1f, 2f, 3f, 4f }), "a leading 0 becomes tiny");
        Assert.That(new FlipSigmas().Execute(new Sigmas([3f, 2f, 1f])).Values, Is.EqualTo(new[] { 1f, 2f, 3f }));
        Assert.That(new SetFirstSigma().Execute(s, 9f).Values, Is.EqualTo(new[] { 9f, 3f, 2f, 1f, 0f }));
        Assert.That(s.Values[0], Is.EqualTo(4f), "the input is not changed");
    }

    [Test]
    public void The_sampler_select_node_rejects_unknown_names()
    {
        Assert.That(new KSamplerSelect().Execute("dpmpp_2m").Name, Is.EqualTo("dpmpp_2m"));
        Assert.That(() => new KSamplerSelect().Execute("nope"), Throws.ArgumentException);
    }

    [Test]
    public void A_custom_sampling_graph_validates_and_SamplerCustom_has_both_latent_outputs()
    {
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly);
        var outputs = registry.GetSchema("SamplerCustom")!.Outputs.Select(o => o.Name).ToArray();
        Assert.That(outputs, Is.EqualTo(new[] { "output", "denoised_output" }));
        Assert.That(registry.GetSchema("SplitSigmas")!.Outputs.Select(o => o.Name), Is.EqualTo(new[] { "high", "low" }));

        var g = new PromptGraph();
        g.Add("1", "CheckpointLoaderSimple", ("ckpt_name", "sd_turbo"));
        g.Add("2", "CLIPTextEncode", ("text", "a"), ("clip", PromptInput.Link("1", 1)));
        g.Add("3", "CLIPTextEncode", ("text", ""), ("clip", PromptInput.Link("1", 1)));
        g.Add("4", "EmptyLatentImage", ("width", 64), ("height", 64), ("batch_size", 1));
        g.Add("5", "KSamplerSelect", ("sampler_name", "euler"));
        g.Add("6", "BasicScheduler", ("model", PromptInput.Link("1")), ("scheduler", "simple"), ("steps", 4), ("denoise", 1.0));
        g.Add("7", "SamplerCustom", ("model", PromptInput.Link("1")), ("positive", PromptInput.Link("2")), ("negative", PromptInput.Link("3")),
            ("sampler", PromptInput.Link("5")), ("sigmas", PromptInput.Link("6")), ("latent_image", PromptInput.Link("4")), ("add_noise", true), ("noise_seed", 1), ("cfg", 1.0));
        g.Add("8", "VAEDecode", ("samples", PromptInput.Link("7", 1)), ("vae", PromptInput.Link("1", 2)));
        g.Add("9", "SaveImage", ("images", PromptInput.Link("8")), ("filename_prefix", "custom"));
        var result = new PromptExecutor(registry).Validate(g);
        Assert.That(result.Errors, Is.Empty, string.Join("\n", result.Errors.Select(e => e.Message)));
    }

    /// <summary>The same one-step SD-Turbo picture through KSampler and through SDTurboScheduler + KSamplerSelect + SamplerCustom must be byte-identical (same sigmas, same noise).</summary>
    [Test, Explicit("Needs SPAWNGEN_MODELS and a GPU.")]
    public async Task SamplerCustom_with_the_turbo_schedule_draws_the_same_picture_as_KSampler()
    {
        var models = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        if (string.IsNullOrEmpty(models)) Assert.Ignore("SPAWNGEN_MODELS is not set.");
        var outDir = Path.Combine(Path.GetTempPath(), "spawngen-custom-" + Guid.NewGuid().ToString("N"));
        using var runtime = new MlRuntime(new FolderModelSource(models!));
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly);
        var executor = new PromptExecutor(registry, new Services(runtime, new OutputOptions { Directory = outDir }));

        PromptGraph Common(string prefix, bool custom)
        {
            var g = new PromptGraph();
            g.Add("1", "CheckpointLoaderSimple", ("ckpt_name", "sd_turbo"));
            g.Add("2", "CLIPTextEncode", ("text", "a photo of a red apple on a wooden table"), ("clip", PromptInput.Link("1", 1)));
            g.Add("3", "CLIPTextEncode", ("text", ""), ("clip", PromptInput.Link("1", 1)));
            g.Add("4", "EmptyLatentImage", ("width", 512), ("height", 512), ("batch_size", 1));
            if (custom)
            {
                g.Add("5", "KSamplerSelect", ("sampler_name", "euler"));
                g.Add("6", "SDTurboScheduler", ("model", PromptInput.Link("1")), ("steps", 1), ("denoise", 1.0));
                g.Add("7", "SamplerCustom", ("model", PromptInput.Link("1")), ("positive", PromptInput.Link("2")), ("negative", PromptInput.Link("3")),
                    ("sampler", PromptInput.Link("5")), ("sigmas", PromptInput.Link("6")), ("latent_image", PromptInput.Link("4")), ("add_noise", true), ("noise_seed", 7), ("cfg", 1.0));
            }
            else
                g.Add("7", "KSampler", ("model", PromptInput.Link("1")), ("positive", PromptInput.Link("2")), ("negative", PromptInput.Link("3")), ("latent_image", PromptInput.Link("4")),
                    ("seed", 7), ("steps", 1), ("cfg", 1.0), ("sampler_name", "euler"), ("scheduler", "normal"), ("denoise", 1.0));
            g.Add("8", "VAEDecode", ("samples", PromptInput.Link("7")), ("vae", PromptInput.Link("1", 2)));
            g.Add("9", "SaveImage", ("images", PromptInput.Link("8")), ("filename_prefix", prefix));
            return g;
        }

        await executor.ExecuteAsync(Common("plain", custom: false));
        await executor.ExecuteAsync(Common("custom", custom: true));
        var advanced = Common("advanced", custom: true);
        advanced.Add("20", "RandomNoise", ("noise_seed", 7));
        advanced.Add("21", "BasicGuider", ("model", PromptInput.Link("1")), ("conditioning", PromptInput.Link("2")));
        advanced.Add("7", "SamplerCustomAdvanced", ("noise", PromptInput.Link("20")), ("guider", PromptInput.Link("21")), ("sampler", PromptInput.Link("5")), ("sigmas", PromptInput.Link("6")), ("latent_image", PromptInput.Link("4")));
        await executor.ExecuteAsync(advanced);
        var plain = Directory.GetFiles(outDir, "plain_*.png").Single();
        var custom = Directory.GetFiles(outDir, "custom_*.png").Single();
        TestContext.Out.WriteLine(plain + " and " + custom);
        Assert.That(PngReader.Decode(File.ReadAllBytes(custom)).Data, Is.EqualTo(PngReader.Decode(File.ReadAllBytes(plain)).Data));
        var adv = Directory.GetFiles(outDir, "advanced_*.png").Single();
        Assert.That(PngReader.Decode(File.ReadAllBytes(adv)).Data, Is.EqualTo(PngReader.Decode(File.ReadAllBytes(plain)).Data), "SamplerCustomAdvanced with RandomNoise and BasicGuider gives the same picture");
    }

    [Test]
    public void The_advanced_sampler_and_its_pieces_validate_in_a_graph()
    {
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly);
        var g = new PromptGraph();
        g.Add("1", "CheckpointLoaderSimple", ("ckpt_name", "sd_turbo"));
        g.Add("2", "CLIPTextEncode", ("text", "a"), ("clip", PromptInput.Link("1", 1)));
        g.Add("3", "CLIPTextEncode", ("text", ""), ("clip", PromptInput.Link("1", 1)));
        g.Add("4", "EmptyLatentImage", ("width", 64), ("height", 64), ("batch_size", 1));
        g.Add("5", "KSamplerSelect", ("sampler_name", "euler"));
        g.Add("6", "SDTurboScheduler", ("model", PromptInput.Link("1")), ("steps", 1), ("denoise", 1.0));
        g.Add("7", "RandomNoise", ("noise_seed", 7));
        g.Add("8", "BasicGuider", ("model", PromptInput.Link("1")), ("conditioning", PromptInput.Link("2")));
        g.Add("9", "CFGGuider", ("model", PromptInput.Link("1")), ("positive", PromptInput.Link("2")), ("negative", PromptInput.Link("3")), ("cfg", 7.0));
        g.Add("10", "SamplerCustomAdvanced", ("noise", PromptInput.Link("7")), ("guider", PromptInput.Link("8")), ("sampler", PromptInput.Link("5")), ("sigmas", PromptInput.Link("6")), ("latent_image", PromptInput.Link("4")));
        g.Add("11", "SamplerCustomAdvanced", ("noise", PromptInput.Link("7")), ("guider", PromptInput.Link("9")), ("sampler", PromptInput.Link("5")), ("sigmas", PromptInput.Link("6")), ("latent_image", PromptInput.Link("4")));
        g.Add("12", "VAEDecode", ("samples", PromptInput.Link("10")), ("vae", PromptInput.Link("1", 2)));
        g.Add("13", "SaveImage", ("images", PromptInput.Link("12")), ("filename_prefix", "adv"));
        g.Add("14", "VAEDecode", ("samples", PromptInput.Link("11", 1)), ("vae", PromptInput.Link("1", 2)));
        g.Add("15", "PreviewImage", ("images", PromptInput.Link("14")));
        var result = new PromptExecutor(registry).Validate(g);
        Assert.That(result.Errors, Is.Empty, string.Join("; ", result.Errors.Select(e => e.Message)));
        Assert.That(new DisableNoise().Execute().Enabled, Is.False);
        Assert.That(new BasicGuider().Execute(null!, null!).Cfg, Is.EqualTo(1f));
    }

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }
}
