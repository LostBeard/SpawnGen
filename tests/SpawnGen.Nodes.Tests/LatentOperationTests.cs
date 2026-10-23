using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// The LATENT_OPERATION nodes (comfy_extras/nodes_latent.py) on small latents whose results are worked out from ComfyUI's formulas, and Latent Apply Operation CFG's
/// pre-CFG patch: what it does to the two predictions, when in the schedule, and how the sampler hands them over.
/// </summary>
public class LatentOperationTests
{
    static LatentBatch Lat(int b, int c, int h, int w, params float[] v) => new(b, c, h, w, v);

    /// <summary>An operation for the tests: doubles the latent and records which batch position it was given.</summary>
    static LatentOperation Doubler(List<int> seen) => new((l, first) =>
    {
        seen.Add(first);
        return new LatentBatch(l.Batch, l.Channels, l.Height, l.Width, l.Data.Select(v => v * 2).ToArray());
    });

    [Test]
    public void Apply_operation_runs_it_and_keeps_the_latents_mask_and_batch_index()
    {
        var mask = MaskBatch.Filled(1, 1, 2, 1f);
        var l = new LatentBatch(1, 1, 1, 2, [1f, 2f]) { NoiseMask = mask, BatchIndex = [4] };
        var seen = new List<int>();
        var r = new LatentApplyOperation().Execute(l, Doubler(seen));
        Assert.That(r.Data, Is.EqualTo(new[] { 2f, 4f }));
        Assert.That((r.NoiseMask, r.BatchIndex), Is.EqualTo((mask, l.BatchIndex)));
        Assert.That(seen, Is.EqualTo(new[] { 0 }));
    }

    [Test]
    public void Blend_lerps_toward_the_nearest_resized_reference()
    {
        var reference = Lat(1, 1, 1, 1, 10f);
        var latent = Lat(1, 1, 2, 2, 0f, 1f, 2f, 3f);
        // torch.lerp(a, 10, w): a + w (10 - a) below 0.5, 10 - (10 - a)(1 - w) from 0.5.
        Assert.That(new LatentOperationBlend().Execute(reference, 0.25).Apply(latent).Data, Is.EqualTo(new[] { 2.5f, 3.25f, 4f, 4.75f }));
        Assert.That(new LatentOperationBlend().Execute(reference, 0.75).Apply(latent).Data, Is.EqualTo(new[] { 7.5f, 7.75f, 8f, 8.25f }));
        Assert.That(new LatentOperationBlend().Execute(reference, 0).Apply(latent), Is.SameAs(latent));
        // The reference batch is repeated over the whole batch: the sampler's picture 1 gets reference picture 1.
        var two = Lat(2, 1, 1, 1, 10f, 20f);
        Assert.That(new LatentOperationBlend().Execute(two, 1).Apply(Lat(1, 1, 1, 1, 0f), first: 1).Data, Is.EqualTo(new[] { 20f }));
        Assert.That(new LatentOperationBlend().Execute(two, 1).Apply(Lat(1, 1, 1, 1, 0f), first: 2).Data, Is.EqualTo(new[] { 10f }));
    }

    [Test]
    public void Tonemap_compresses_lengths_toward_the_ceiling_and_keeps_directions()
    {
        // One channel, lengths 1 and 3: mean 2, unbiased std sqrt(2), ceiling top = 5 sqrt(2) + 2. Reinhard: x / (x + 1) * top with x = length / top,
        // which is length * top / (length + top).
        double top = 5 * Math.Sqrt(2) + 2;
        var r = new LatentOperationTonemapReinhard().Execute(1).Apply(Lat(1, 1, 1, 2, 1f, -3f)).Data;
        Assert.That(r, Is.EqualTo(new[] { (float)(top / (1 + top)), (float)(-3 * top / (3 + top)) }).Within(1e-5));
        // Two channels at a position keep their ratio.
        var two = new LatentOperationTonemapReinhard().Execute(0.5).Apply(Lat(1, 2, 1, 2, 3f, 0f, 4f, 1f)).Data;
        Assert.That(two[0] / two[2], Is.EqualTo(0.75).Within(1e-6));
        Assert.That(new LatentOperationTonemapReinhard().Execute(0).Apply(Lat(1, 1, 1, 2, 1f, 3f)).Data, Is.All.NaN, "multiplier 0 divides by zero in ComfyUI too");
    }

    [Test]
    public void Sharpen_kernel_is_ComfyUIs_gaussian_turned_into_an_unsharp_mask()
    {
        // gaussian_kernel(3, 1): linspace(-1, 1, 3) squared distances 0, 1, 2 -> exp(-d^2 / 2), normalised by S; times -10 alpha = -1; centre raised to sum to 1.
        double s = 1 + 4 * Math.Exp(-0.5) + 4 * Math.Exp(-1);
        var k = LatentOperationSharpen.Kernel(1, 1, 0.1);
        Assert.That(k, Is.EqualTo(new[]
        {
            -Math.Exp(-1) / s, -Math.Exp(-0.5) / s, -Math.Exp(-1) / s,
            -Math.Exp(-0.5) / s, 2 - 1 / s, -Math.Exp(-0.5) / s,
            -Math.Exp(-1) / s, -Math.Exp(-0.5) / s, -Math.Exp(-1) / s,
        }.Select(v => (float)v).ToArray()).Within(1e-6));
        Assert.That(k.Sum(), Is.EqualTo(1f).Within(1e-6));
    }

    [Test]
    public void Sharpen_works_on_directions_with_reflected_edges()
    {
        // Two channels on 3x3: channel 0 all 1, channel 1 a single 1 in the middle (so the middle direction tilts). Radius 1, sigma 1, alpha 0.1.
        double s = 1 + 4 * Math.Exp(-0.5) + 4 * Math.Exp(-1), centre = 2 - 1 / s, edge = -Math.Exp(-0.5) / s, corner = -Math.Exp(-1) / s;
        var data = new float[18];
        for (int i = 0; i < 9; i++) data[i] = 1f;
        data[9 + 4] = 1f;
        var r = new LatentOperationSharpen().Execute(1, 1, 0.1).Apply(Lat(1, 2, 3, 3, data)).Data;
        double lm = Math.Sqrt(2) + 1e-6, lo = 1 + 1e-6; // the lengths in the middle and elsewhere, as ComfyUI adds 1e-6
        // Middle: channel 1 only has its own sample; channel 0 sees 1/lo all around.
        Assert.That(r[9 + 4], Is.EqualTo(centre).Within(1e-5));
        Assert.That(r[4], Is.EqualTo(centre + lm * 4 * (edge + corner) / lo).Within(1e-5));
        // Corner (0, 0): reflect padding maps the four diagonal taps onto the middle sample.
        Assert.That(r[9], Is.EqualTo(lo * 4 * corner / lm).Within(1e-5));
        Assert.That(r[0], Is.EqualTo(centre + 4 * edge + lo * 4 * corner / lm).Within(1e-5));
        Assert.That(() => new LatentOperationSharpen().Execute(9, 1, 0.1).Apply(Lat(1, 1, 3, 3, data[..9])), Throws.ArgumentException, "reflect padding needs radius < size, as in PyTorch");
    }

    // ---- Latent Apply Operation CFG ----

    static readonly ModelSampling Sd = new(SamplingSetupTests.Sd());

    /// <summary>ComfyUI's pre-cfg arguments for one picture of <paramref name="width"/> one-channel cells: [cond, uncond] denoised estimates at <paramref name="sigma"/>.</summary>
    static PreCfgArgs Args(double sigma, float[] cond, float[] uncond, int index = 0) =>
        new([true, true], [cond, uncond], 5.0, sigma, new float[cond.Length], new PatchContext
        {
            Channels = 1, Height = 1, Width = cond.Length, Sampling = Sd, BatchIndex = index, EvaluateUnscheduled = _ => Task.FromResult(Array.Empty<float>()),
        });

    [Test]
    public void The_patch_changes_the_guidance_cond_minus_uncond_and_puts_the_uncond_back()
    {
        var seen = new List<int>();
        var outs = LatentApplyOperationCFG.PreCfg(Args(3.0, [5f, 7f], [1f, 1f], index: 2), Doubler(seen), double.MaxValue, 0);
        Assert.That(outs[0], Is.EqualTo(new[] { 9f, 13f }), "2 (cond - uncond) + uncond");
        Assert.That(outs[1], Is.EqualTo(new[] { 1f, 1f }));
        Assert.That(seen, Is.EqualTo(new[] { 2 }), "the operation is told which picture of the batch it has");
    }

    [Test]
    public void The_patch_only_acts_inside_its_part_of_the_schedule()
    {
        double start = Sd.PercentToSigma(0.25), end = Sd.PercentToSigma(0.75);
        Assert.That(start, Is.GreaterThan(end));
        float[] Run(double sigma) => LatentApplyOperationCFG.PreCfg(Args(sigma, [3f], [1f]), Doubler([]), start, end)[0];
        Assert.That(Run(start * 1.01), Is.EqualTo(new[] { 3f }), "before start_percent (sigma above sigma_start)");
        Assert.That(Run(start), Is.EqualTo(new[] { 5f }), "sigma_start itself is inside");
        Assert.That(Run((start + end) / 2), Is.EqualTo(new[] { 5f }));
        Assert.That(Run(end), Is.EqualTo(new[] { 5f }), "sigma_end itself is inside");
        Assert.That(Run(end * 0.99), Is.EqualTo(new[] { 3f }), "after end_percent");
    }

    /// <summary>
    /// End to end on SD-Turbo: a model patched with Latent Apply Operation CFG (Reinhard tone mapping of the guidance) samples a different, calmer picture than the
    /// plain model; and a picture taken from a batch with Get Latent From Batch is sampled from the same noise it had in the batch (ComfyUI's batch_index).
    /// </summary>
    [Test, Explicit("Needs SPAWNGEN_MODELS and a GPU.")]
    public async Task Operation_CFG_changes_sampling_and_a_picture_from_a_batch_keeps_its_noise()
    {
        var models = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        if (string.IsNullOrEmpty(models)) Assert.Ignore("SPAWNGEN_MODELS is not set.");
        var outDir = Path.Combine(Path.GetTempPath(), "spawngen-latentop-" + Guid.NewGuid().ToString("N"));
        using var runtime = new MlRuntime(new FolderModelSource(models!));
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly);
        var executor = new PromptExecutor(registry, new Services(runtime, new OutputOptions { Directory = outDir }));

        PromptGraph Graph(string prefix, bool patched, int batch, int? pick)
        {
            static PromptInput L(string id, int slot = 0) => PromptInput.Link(id, slot);
            var g = new PromptGraph();
            g.Add("1", "CheckpointLoaderSimple", ("ckpt_name", "sd_turbo"));
            g.Add("2", "CLIPTextEncode", ("text", "a photo of a red apple on a wooden table"), ("clip", L("1", 1)));
            g.Add("3", "CLIPTextEncode", ("text", "blurry"), ("clip", L("1", 1)));
            g.Add("4", "EmptyLatentImage", ("width", 256), ("height", 256), ("batch_size", batch));
            string latent = "4", model = "1";
            if (pick is { } p) { g.Add("5", "LatentFromBatch", ("samples", L("4")), ("batch_index", p), ("length", 1)); latent = "5"; }
            if (patched)
            {
                g.Add("10", "LatentOperationTonemapReinhard", ("multiplier", 0.3));
                g.Add("11", "LatentApplyOperationCFG", ("model", L("1")), ("operation", L("10")), ("start_percent", 0.0), ("end_percent", 1.0));
                model = "11";
            }
            g.Add("7", "KSampler", ("model", L(model)), ("positive", L("2")), ("negative", L("3")), ("latent_image", L(latent)),
                ("seed", 11), ("steps", 2), ("cfg", 3.0), ("sampler_name", "euler"), ("scheduler", "normal"), ("denoise", 1.0));
            g.Add("8", "VAEDecode", ("samples", L("7")), ("vae", L("1", 2)));
            g.Add("9", "SaveImage", ("images", L("8")), ("filename_prefix", prefix));
            return g;
        }

        await executor.ExecuteAsync(Graph("plain", false, 2, null));
        await executor.ExecuteAsync(Graph("patched", true, 1, null));
        await executor.ExecuteAsync(Graph("picked", false, 2, 1));
        ImageBatch Read(string name) => PngReader.Decode(File.ReadAllBytes(Directory.GetFiles(outDir, name).Single()));
        var plain0 = Read("plain_00001_.png");
        var plain1 = Read("plain_00002_.png");
        var patched = Read("patched_00001_.png");
        var picked = Read("picked_00001_.png");
        TestContext.Out.WriteLine(outDir);
        Assert.That(picked.Data, Is.EqualTo(plain1.Data), "Get Latent From Batch index 1 samples from the batch's second noise");
        Assert.That(plain0.Data, Is.Not.EqualTo(plain1.Data));
        double diff = plain0.Data.Zip(patched.Data, (a, b) => Math.Abs(a - b)).Average();
        Assert.That(diff, Is.GreaterThan(0.005), "the patch changes the picture");
    }

    [Test]
    public void RescaleCFG_keeps_the_patches_already_on_the_model()
    {
        var model = new LatentApplyOperationCFG().Execute(new ModelHandle(null!, Sd), Doubler([]));
        var rescaled = new RescaleCFG().Execute(model, 0.7);
        Assert.That(rescaled.Patches.PreCfg, Is.EqualTo(model.Patches.PreCfg), "the latent operation stays");
        Assert.That(rescaled.Patches.CfgFunction, Is.Not.Null);
        Assert.That(model.Patches.CfgFunction, Is.Null, "patching never changes the model that came in");
    }

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }
}
