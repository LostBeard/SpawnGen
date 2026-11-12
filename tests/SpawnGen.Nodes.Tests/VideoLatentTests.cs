using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Video latents (5-D LATENT) against ComfyUI's Wan 2.2 latent nodes, measured black-box with its Wan 2.2 VAE (Fixtures/wan22): shapes, the start
/// picture's frames, the latent-mean fill and the noise mask; and the sampler's handling of them with a stand-in model.
/// </summary>
public class VideoLatentTests
{
    static readonly string Fixtures = Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "wan22");

    static float[] Raw(string file) =>
        MemoryMarshalFloats(File.ReadAllBytes(Path.Combine(Fixtures, file)));

    static float[] MemoryMarshalFloats(byte[] bytes)
    {
        var r = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, r, 0, bytes.Length);
        return r;
    }

    static NodeContext Context(IServiceProvider? services = null) =>
        (NodeContext)Activator.CreateInstance(typeof(NodeContext), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, null,
            ["1", "X", services ?? new Services(), CancellationToken.None, null, new Lazy<string>(() => "{}"), new Dictionary<string, string>(), null, null, 0], null)!;

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }

    /// <summary>A video VAE whose encode is ComfyUI's measured encode of the 256x160 probe picture (it checks it was given that picture, fitted).</summary>
    sealed class MeasuredVae : IVideoVaeModel
    {
        public List<ImageBatch> Encoded { get; } = [];
        public string Name => "wan2.2_vae (measured)";
        public SdVae? SdVae => null;
        public int SpatialScale => 16;
        public int TemporalScale => 4;
        public int LatentChannels => 48;
        public Task<float[]> DecodeAsync(float[] latent, int h, int w, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public Task<float[]> EncodeImageAsync(float[] image, int H, int W, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public Task<ImageBatch> DecodeVideoAsync(LatentBatch latent, CancellationToken cancellationToken = default) => throw new NotSupportedException();

        public Task<LatentBatch> EncodeVideoAsync(ImageBatch frames, CancellationToken cancellationToken = default)
        {
            Encoded.Add(frames);
            int f = (frames.Batch - 1) / 4 + 1, h = frames.Height / 16, w = frames.Width / 16;
            return Task.FromResult(f == 1 && h == 10 && w == 16
                ? new LatentBatch(1, 48, 1, 10, 16, Raw("wan22_vaeencode_256x160_samples_f32_1x48x1x10x16.raw"))
                : new LatentBatch(1, 48, f, h, w));
        }
    }

    static ImageBatch Probe(string file) => ImageDecoder.Decode(File.ReadAllBytes(Path.Combine(Fixtures, file)));

    [TestCase(256, 160, 9, 1, new[] { 1, 48, 3, 10, 16 })]
    [TestCase(256, 160, 10, 1, new[] { 1, 48, 3, 10, 16 })]
    [TestCase(1280, 704, 121, 1, new[] { 1, 48, 31, 44, 80 })]
    [TestCase(288, 160, 1, 2, new[] { 1, 48, 1, 10, 18 })] // batch_size is not applied without a start picture (measured)
    public async Task Wan22ImageToVideoLatent_without_a_start_picture_is_zeros_of_ComfyUIs_shape(int w, int h, int length, int batch, int[] shape)
    {
        var latent = await new Wan22ImageToVideoLatent().ExecuteAsync(new VaeHandle(new MeasuredVae()), Context(), w, h, length, batch);
        Assert.That(latent.Shape, Is.EqualTo(shape));
        Assert.That(latent.IsVideo && latent.Data.All(v => v == 0f) && latent.NoiseMask is null);
    }

    [TestCase(256, 160, 9, 1, new[] { 1, 16, 3, 20, 32 })]
    [TestCase(1280, 704, 121, 1, new[] { 1, 16, 31, 88, 160 })]
    [TestCase(288, 160, 1, 2, new[] { 2, 16, 1, 20, 36 })]
    public void EmptyHunyuanLatentVideo_has_ComfyUIs_shape(int w, int h, int length, int batch, int[] shape) =>
        Assert.That(new EmptyHunyuanLatentVideo().Execute(w, h, length, batch).Shape, Is.EqualTo(shape));

    [TestCase("spawngen_wan_probe_256x160.png", "wan22_start_256x160_256x160_9")]
    [TestCase("spawngen_wan_probe_300x200.png", "wan22_start_300x200_256x160_9")]
    public async Task Wan22ImageToVideoLatent_with_a_start_picture_matches_ComfyUI(string picture, string measured)
    {
        var vae = new MeasuredVae();
        var latent = await new Wan22ImageToVideoLatent().ExecuteAsync(new VaeHandle(vae), Context(), 256, 160, 9, 1, Probe(picture));
        Assert.That(latent.Shape, Is.EqualTo(new[] { 1, 48, 3, 10, 16 }));
        // The VAE got the picture fitted to 256x160 by a bilinear resize with a centre crop (ComfyUI's encode of exactly that equals its latent's
        // first frame; measured against ImageScale + VAEEncode with every method and crop).
        var fitted = CommonUpscale.Resize(Probe(picture), 256, 160, "bilinear", "center");
        Assert.That(vae.Encoded.Single().Data, Is.EqualTo(fitted.Data));

        var mask = Raw($"{measured}_noise_mask_f32_1x1x3x10x16.raw");
        Assert.That(latent.NoiseMask!.Shape, Is.EqualTo(new[] { 1, 3 * 10, 16 }));
        Assert.That(latent.NoiseMask.Data, Is.EqualTo(mask));
        if (picture.Contains("256x160"))
            Assert.That(latent.Data, Is.EqualTo(Raw($"{measured}_samples_f32_1x48x3x10x16.raw")).Within(1e-6f), "the encoded frame, then the latent mean");
        else
        {
            // The measured VAE stands in for the 256x160 picture's encode, so only the frames after the first can be compared here.
            var comfy = Raw($"{measured}_samples_f32_1x48x3x10x16.raw");
            for (int c = 0; c < 48; c++)
                for (int i = 160; i < 480; i++) Assert.That(latent.Data[c * 480 + i], Is.EqualTo(comfy[c * 480 + i]).Within(1e-6f));
        }
    }

    [Test]
    public async Task Wan22ImageToVideoLatent_repeats_a_start_picture_latent_batch_size_times()
    {
        var latent = await new Wan22ImageToVideoLatent().ExecuteAsync(new VaeHandle(new MeasuredVae()), Context(), 256, 160, 9, 2, Probe("spawngen_wan_probe_256x160.png"));
        Assert.That(latent.Shape, Is.EqualTo(new[] { 2, 48, 3, 10, 16 }));
        Assert.That(latent.Data.AsSpan(0, latent.ItemSize).ToArray(), Is.EqualTo(latent.Data.AsSpan(latent.ItemSize).ToArray()));
        Assert.That(latent.NoiseMask!.Batch, Is.EqualTo(2));
    }

    [Test]
    public async Task Wan22ImageToVideoLatent_encodes_at_most_length_pictures()
    {
        var vae = new MeasuredVae();
        var thirteen = new ImageBatch(13, 160, 256, 3);
        await new Wan22ImageToVideoLatent().ExecuteAsync(new VaeHandle(vae), Context(), 256, 160, 9, 1, thirteen);
        Assert.That(vae.Encoded.Single().Batch, Is.EqualTo(9));
    }

    [Test]
    public void Wan22_latent_format_normalises_per_channel_and_its_zero_is_the_latent_mean()
    {
        var mean = LatentFormat.Wan22.ProcessOut(new float[48 * 6]);
        var comfyFill = Raw("wan22_start_256x160_256x160_9_samples_f32_1x48x3x10x16.raw");
        for (int c = 0; c < 48; c++)
        {
            Assert.That(mean[c * 6], Is.EqualTo(LatentFormat.Wan22.LatentsMean![c]));
            Assert.That(comfyFill[c * 480 + 300], Is.EqualTo(LatentFormat.Wan22.LatentsMean[c]), "ComfyUI fills the frames after the start with the mean");
        }
        var x = Enumerable.Range(0, 48 * 6).Select(i => (float)Math.Sin(i)).ToArray();
        var back = LatentFormat.Wan22.ProcessOut(LatentFormat.Wan22.ProcessIn(x));
        Assert.That(back, Is.EqualTo(x).Within(1e-5f));
        Assert.That(LatentFormat.Wan22.ProcessIn(new float[48]), Is.All.EqualTo(0f), "an empty latent stays empty");
    }

    [Test]
    public void VAE_Encode_with_a_video_VAE_cuts_to_multiples_of_16_centred()
    {
        var vae = new MeasuredVae();
        var picture = Probe("spawngen_wan_probe_300x200.png");
        _ = new VAEEncode().ExecuteAsync(picture, new VaeHandle(vae), Context(), new Services()).Result;
        var given = vae.Encoded.Single();
        Assert.That((given.Width, given.Height), Is.EqualTo((288, 192)));
        // measured: the 288x192 cut at (6, 4) encodes to the same latent as the whole picture
        Assert.That(given.Data.AsSpan(0, 3).ToArray(), Is.EqualTo(picture.Data.AsSpan((4 * 300 + 6) * 3, 3).ToArray()));
    }

    [TestCase(null, false)]
    [TestCase("ones", false)]
    [TestCase("first", true)]
    [TestCase("zeros", true)]
    public void A_video_noise_mask_runs_the_first_frame_clean_only_when_it_keeps_just_that_frame(string? kind, bool clean)
    {
        const int frames = 3, plane = 4;
        float[]? mask = kind switch
        {
            null => null,
            "ones" => Enumerable.Repeat(1f, frames * plane).ToArray(),
            "zeros" => new float[frames * plane],
            _ => Enumerable.Range(0, frames * plane).Select(i => i < plane ? 0f : 1f).ToArray(),
        };
        Assert.That(SamplingCoreAccess.FirstFrameOnlyKept(mask, frames, plane), Is.EqualTo(clean));
    }

    [Test]
    public void A_video_noise_mask_keeping_other_regions_is_refused()
    {
        var mask = Enumerable.Repeat(1f, 12).ToArray();
        mask[5] = 0f;
        Assert.That(() => SamplingCoreAccess.FirstFrameOnlyKept(mask, 3, 4), Throws.InstanceOf<NotSupportedException>());
    }

    static class SamplingCoreAccess
    {
        public static bool FirstFrameOnlyKept(float[]? mask, int frames, int plane)
        {
            try
            {
                return (bool)typeof(KSampler).Assembly.GetType("SpawnGen.Nodes.ML.SamplingCore")!
                    .GetMethod("FirstFrameOnlyKept", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!
                    .Invoke(null, [mask, frames, plane])!;
            }
            catch (System.Reflection.TargetInvocationException e) when (e.InnerException is not null) { throw e.InnerException; }
        }
    }

    /// <summary>A video model that predicts zero velocity and records what the sampler tells it.</summary>
    sealed class StillVideoModel : IDiffusionModel
    {
        public readonly List<(int Frames, bool FirstClean, int H, int W, float T)> Calls = [];
        public string Name => "still";
        public SdFamily Family => SdFamily.SD1;
        public LatentFormat LatentFormat => LatentFormat.Wan22;
        public ModelSampling Sampling { get; } = new(ModelSigmas.Flow(ModelSigmas.Sd3Shift(8), 1000), PredictionType.Const);
        public SdUNet? UNet => null;
        public Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) => Task.CompletedTask;
        public UNetRun BeginRun(Accelerator accelerator) => new Run(this);

        sealed class Run(StillVideoModel model) : UNetRun
        {
            public override Task<float[]> PredictAsync(float[] input, int h, int w, float timestep, CondEntry entry, IReadOnlyList<ControlCall>? controls,
                UNetOptions options, float[]? concat, CancellationToken cancellationToken)
            {
                model.Calls.Add((Frames, FirstFrameClean, h, w, timestep));
                return Task.FromResult(new float[input.Length]);
            }

            public override void Dispose() { }
        }
    }

    [Test]
    public async Task KSampler_samples_a_video_latent_keeping_its_shape_and_start_frame()
    {
        var start = await new Wan22ImageToVideoLatent().ExecuteAsync(new VaeHandle(new MeasuredVae()), Context(), 256, 160, 9, 1, Probe("spawngen_wan_probe_256x160.png"));
        var model = new StillVideoModel();
        var cond = new Conditioning(new float[8], [1, 2, 4]);
        var sigmas = model.Sampling.Sigmas.Schedule("simple", 3);
        var result = await SamplingAccess.RunAsync(new CfgGuider(new ModelHandle(model), cond, cond, 5.0), start, sigmas, new SamplingSettingsAccess(42, "euler"), Context());

        Assert.That(result.Shape, Is.EqualTo(start.Shape));
        Assert.That(model.Calls, Is.Not.Empty);
        Assert.That(model.Calls.All(c => c.Frames == 3 && c.FirstClean && c.H == 30 && c.W == 16), Is.True, "frames stacked, first frame clean");
        Assert.That(model.Calls[0].T, Is.EqualTo(1000f).Within(1e-3f), "flow timestep = sigma x 1000");
        // The start frame is pinned to the original; with zero velocity the rest stays the seed's noise (in the model's units) carried back out.
        for (int c = 0; c < 48; c++)
            for (int i = 0; i < 160; i++) Assert.That(result.Data[c * 480 + i], Is.EqualTo(start.Data[c * 480 + i]).Within(1e-5f));
        var noise = SamplingAccess.InitialNoise(42, 0, start);
        var expected = LatentFormat.Wan22.ProcessOut(noise);
        Assert.That(result.Data[5 * 480 + 200], Is.EqualTo(expected[5 * 480 + 200]).Within(1e-4f));
    }

    [Test]
    public async Task A_picture_only_node_refuses_a_video_latent_before_it_runs()
    {
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly);
        var g = new PromptGraph();
        g.Add("1", "EmptyHunyuanLatentVideo", ("width", 64L), ("height", 64L), ("length", 5L), ("batch_size", 1L));
        g.Add("2", "LatentUpscaleBy", ("samples", PromptInput.Link("1")), ("upscale_method", "nearest-exact"), ("scale_by", 2.0));
        g.Add("3", "PreviewAny", ("source", PromptInput.Link("2")));
        var error = Assert.ThrowsAsync<NodeExecutionException>(async () => await new PromptExecutor(registry, new Services()).ExecuteAsync(g));
        Assert.That(error!.ToString(), Does.Contain("does not take video latents"));
        await Task.CompletedTask;
    }

    static class SamplingAccess
    {
        static readonly Type Core = typeof(KSampler).Assembly.GetType("SpawnGen.Nodes.ML.SamplingCore")!;

        public static async Task<LatentBatch> RunAsync(GuiderChoice guider, LatentBatch latent, float[] sigmas, SamplingSettingsAccess settings, NodeContext context)
        {
            var settingsType = typeof(KSampler).Assembly.GetType("SpawnGen.Nodes.ML.SamplingSettings")!;
            var s = Activator.CreateInstance(settingsType, settings.Seed, settings.Sampler, true, null)!;
            var task = (Task<LatentBatch>)Core.GetMethod("RunAsync")!.Invoke(null, [null, guider, latent, sigmas, s, context])!;
            return await task;
        }

        public static float[] InitialNoise(long seed, int index, LatentBatch latent) =>
            (float[])Core.GetMethod("InitialNoise", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!.Invoke(null, [seed, index, latent])!;
    }

    sealed record SamplingSettingsAccess(long Seed, string Sampler);
}
