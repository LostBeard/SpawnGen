using SpawnDev.Codecs.Images;
using SpawnDev.Codecs.Images.WebP;
using SpawnDev.ILGPU.ML.Wan;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// Save Animated WEBP (ComfyUI's schema): the batch as one animated WebP in the output folder, each frame shown for <c>int(1000 / fps)</c> ms
/// (as Save Animated PNG). Encoded on the GPU by SpawnDev.Codecs. Its released build has the lossless animation encoder only, so a lossy request
/// is written lossless for now (the lossy one is in the next Codecs release); the prompt / workflow metadata comes with that release too.
/// </summary>
[Node("SaveAnimatedWEBP", Category = "image", DisplayName = "Save Animated WEBP", IsOutput = true)]
public sealed class SaveAnimatedWEBP
{
    public async Task<(ImageBatch Images, UiOutput Ui)> ExecuteAsync(ImageBatch images, NodeContext context, IServiceProvider services,
        string filenamePrefix = "ComfyUI",
        [Input(Min = 0.01, Max = 1000.0, Step = 0.01)] double fps = 6.0,
        bool lossless = true,
        [Input(Min = 0, Max = 100)] long quality = 80,
        [Input(Options = ["default", "fastest", "slowest"])] string method = "default")
    {
        var webp = await EncodeAsync(images, fps, (int)quality, services);
        var prefix = SavePrefix.ComputeVars(filenamePrefix, images.Width, images.Height, DateTime.Now);
        var file = await Results.For(services).SaveAsync("output", prefix, webp, "webp");
        var ui = new UiOutput("images", new object[] { file });
        ui["animated"] = new object[] { images.Batch > 1 };
        return (images, ui);
    }

    /// <summary>The frames as an animated WebP (lossless), each shown for int(1000 / fps) ms, looping forever.</summary>
    public static async Task<byte[]> EncodeAsync(ImageBatch images, double fps, int quality, IServiceProvider services) =>
        await EncodeAsync(images, fps, quality, (await GpuHelpers.RuntimeAsync(services)).Accelerator);

    public static async Task<byte[]> EncodeAsync(ImageBatch images, double fps, int quality, ILGPU.Runtime.Accelerator accelerator)
    {
        if (fps <= 0) throw new ArgumentOutOfRangeException(nameof(fps), "The frame rate must be above 0.");
        int duration = (int)(1000.0 / fps);
        var frames = new List<ReadOnlyMemory<byte>>();
        var infos = new List<ImageFrameInfo>();
        for (int f = 0; f < images.Batch; f++)
        {
            var packed = Rgba.Pack(images, f);
            var bytes = new byte[packed.Length * 4];
            Buffer.BlockCopy(packed, 0, bytes, 0, bytes.Length);
            frames.Add(bytes);
            infos.Add(new ImageFrameInfo(f, 0, 0, images.Width, images.Height, duration, FrameDispose.None, FrameBlend.Source));
        }
        using var encoder = new WebPLosslessEncoderGpu(accelerator);
        return await encoder.EncodeAnimationAsync(frames, infos, images.Width, images.Height,
            new WebPAnimationEncodeOptions { Quality = Math.Clamp(quality, 0, 100), LoopCount = 0 });
    }
}

/// <summary>
/// Wan 2.2 TI2V 5B (Alibaba, Apache-2.0) from the library's 8-bit export: a video from a prompt, or from a start picture and a prompt. One node for
/// the whole model, like FLUX.2 Klein: the library runs its text encoder (UMT5-XXL), flow transformer and video VAE in turn. Gives the frames (an
/// image batch, for Create Video / Save Animated WEBP) and the model's frame rate (24). The defaults are ComfyUI's Wan 2.2 5B template's
/// (1280x704, 121 frames, 20 steps, cfg 5, shift 8, UniPC); sides are multiples of 32 and the length is 4n+1 frames.
/// </summary>
[Node(Category = "video", DisplayName = "Wan 2.2 Video (TI2V 5B)", OutputNames = ["images", "fps"])]
public sealed class Wan22Video
{
    /// <summary>Wan 2.2 5B makes 24 frames a second.</summary>
    public const double FrameRate = 24.0;

    public static string[] Samplers() => ["uni_pc", "few_step"];

    sealed class Loaded(Wan22Pipeline pipeline) : IDisposable
    {
        public Wan22Pipeline Pipeline { get; } = pipeline;
        public void Dispose() { }
    }

    public async Task<(ImageBatch Images, double Fps)> ExecuteAsync(NodeContext context, IServiceProvider services,
        [Input(Multiline = true)] string prompt = "",
        [Input(Multiline = true)] string negativePrompt = Wan22Pipeline.DefaultNegativePrompt,
        [Input(Min = 32, Max = 2048, Step = 32)] int width = 1280,
        [Input(Min = 32, Max = 2048, Step = 32)] int height = 704,
        [Input(Min = 1, Max = 481, Step = 4)] int length = 121,
        [Input(Min = 1, Max = 100)] int steps = 20,
        [Input(Min = 0.0, Max = 30.0, Step = 0.1)] double cfg = 5.0,
        [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double shift = 8.0,
        long seed = 0,
        [Input(OptionsProvider = nameof(Samplers))] string sampler = "uni_pc",
        ImageBatch? startImage = null)
    {
        if (string.IsNullOrWhiteSpace(prompt)) throw new ArgumentException("The prompt is empty.", nameof(prompt));
        if (width % 32 != 0 || height % 32 != 0) throw new ArgumentException($"Wan 2.2 needs sides that are multiples of 32, got {width}x{height}.");
        if ((length - 1) % 4 != 0) throw new ArgumentException($"Wan 2.2 makes 4n+1 frames (1, 5, 9 ... 121), got {length}.");
        var runtime = await GpuHelpers.RuntimeAsync(services);
        // The library's model host has the 8-bit text encoder (its default file, bf16, is not hosted yet).
        var loaded = await runtime.GetOrLoadAsync($"wan22:{Wan22Pipeline.DefaultRepo}", () => Task.FromResult(new Loaded(
            new Wan22Pipeline(runtime.Accelerator, runtime.ModelSource, Wan22Pipeline.DefaultRepo, Wan22Paths.InRepo(textEncoder: Wan22Paths.TextEncoderFile("q8"))))));

        byte[]? rgb = null;
        if (startImage is not null)
        {
            var packed = Rgba.Pack(startImage, 0);
            rgb = new byte[packed.Length * 3];
            for (int i = 0; i < packed.Length; i++) { rgb[i * 3] = (byte)packed[i]; rgb[i * 3 + 1] = (byte)(packed[i] >> 8); rgb[i * 3 + 2] = (byte)(packed[i] >> 16); }
        }
        var request = new WanRequest
        {
            Prompt = prompt, NegativePrompt = negativePrompt, Width = width, Height = height, Frames = length, Steps = steps,
            GuidanceScale = cfg, Shift = shift, Seed = seed, Sampler = sampler == "few_step" ? WanSampler.FewStep : WanSampler.UniPC, RgbaFrames = true,
            ImageRgb = rgb, ImageWidth = startImage?.Width ?? 0, ImageHeight = startImage?.Height ?? 0,
        };
        var progress = new Progress<WanProgress>(p =>
        {
            context.Progress(p.Fraction, 1, $"Wan 2.2: {p.Stage}{(string.IsNullOrEmpty(p.Message) ? "" : " " + p.Message)}");
        });
        var video = await loaded.Pipeline.GenerateAsync(request, progress, context.CancellationToken);

        var frames = new ImageBatch(video.Frames, video.Height, video.Width, 3);
        int pixels = video.Width * video.Height;
        for (int f = 0; f < video.Frames; f++)
        {
            var rgba = video.GetFrameRgba(f);
            int o = f * pixels * 3;
            for (int i = 0; i < pixels; i++)
            {
                frames.Data[o + i * 3] = rgba[i * 4] / 255f;
                frames.Data[o + i * 3 + 1] = rgba[i * 4 + 1] / 255f;
                frames.Data[o + i * 3 + 2] = rgba[i * 4 + 2] / 255f;
            }
        }
        return (frames, FrameRate);
    }
}
