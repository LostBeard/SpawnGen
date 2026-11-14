using SpawnDev.ILGPU.ML.Ltx;
using SpawnDev.ILGPU.ML.Wan;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// Wan 2.2 TI2V 5B (Alibaba, Apache-2.0) from the library's 8-bit export: a video from a prompt, or from a start picture and a prompt. One node for
/// the whole model, like FLUX.2 Klein: the library runs its text encoder (UMT5-XXL), flow transformer and video VAE in turn. Gives the frames (an
/// image batch, for Create Video / Save Animated WEBP) and the model's frame rate (24). The defaults are ComfyUI's Wan 2.2 5B template's
/// (1280x704, 121 frames, 20 steps, cfg 5, shift 8, UniPC); sides are multiples of 32 and the length is 4n+1 frames. Below its 1280x704 training
/// size the model degrades (checked against the official implementation at 640x352). "5B Turbo" is the distilled transformer: 4 steps, cfg 1,
/// few_step. Tiled decode keeps 720p within an 8 GB card.
/// </summary>
[Node(Category = "video", DisplayName = "Wan 2.2 Video (TI2V 5B)", OutputNames = ["images", "fps"])]
public sealed class Wan22Video
{
    /// <summary>Wan 2.2 5B makes 24 frames a second.</summary>
    public const double FrameRate = 24.0;

    public static string[] Samplers() => ["uni_pc", "few_step"];
    public static string[] Models() => ["5B", "5B Turbo"];

    sealed class Loaded(Wan22Pipeline pipeline) : IDisposable
    {
        public Wan22Pipeline Pipeline { get; } = pipeline;
        public void Dispose() { }
    }

    public async Task<(ImageBatch Images, double Fps)> ExecuteAsync(NodeContext context, IServiceProvider services,
        [Input(Multiline = true)] string prompt = "",
        [Input(Multiline = true)] string negativePrompt = Wan22Pipeline.DefaultNegativePrompt,
        [Input(Min = 32, Max = 2048, Step = 32, Tooltip = "Wan 2.2 5B was trained at 1280x704 and 704x1280; well below that it degrades (posterised, over-saturated: the reference does too).")] int width = 1280,
        [Input(Min = 32, Max = 2048, Step = 32, Tooltip = "Wan 2.2 5B was trained at 1280x704 and 704x1280; well below that it degrades (posterised, over-saturated: the reference does too).")] int height = 704,
        [Input(Min = 1, Max = 481, Step = 4)] int length = 121,
        [Input(Min = 1, Max = 100)] int steps = 20,
        [Input(Min = 0.0, Max = 30.0, Step = 0.1)] double cfg = 5.0,
        [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double shift = 8.0,
        long seed = 0,
        [Input(OptionsProvider = nameof(Samplers))] string sampler = "uni_pc",
        ImageBatch? startImage = null,
        [Input(OptionsProvider = nameof(Models), Tooltip = "5B: the model as released (20-50 steps, cfg 5). 5B Turbo: the distilled transformer, 4 steps with cfg 1 and the few_step sampler, far faster.")] string model = "5B",
        [Input(Tooltip = "Decode the video in tiles: much less GPU memory, nearly the same pictures, but slower (measured about 5x on an RX 590). Leave it off unless the decode runs out of memory.")] bool tiledDecode = false)
    {
        if (string.IsNullOrWhiteSpace(prompt)) throw new ArgumentException("The prompt is empty.", nameof(prompt));
        if (width % 32 != 0 || height % 32 != 0) throw new ArgumentException($"Wan 2.2 needs sides that are multiples of 32, got {width}x{height}.");
        if ((length - 1) % 4 != 0) throw new ArgumentException($"Wan 2.2 makes 4n+1 frames (1, 5, 9 ... 121), got {length}.");
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var dit = Wan22Paths.DitFile(model == "5B Turbo" ? "turbo_q8" : "q8");
        var loaded = await runtime.GetOrLoadAsync($"wan22:{Wan22Pipeline.DefaultRepo}:{dit}", () => Task.FromResult(new Loaded(
            new Wan22Pipeline(runtime.Accelerator, runtime.ModelSource, Wan22Pipeline.DefaultRepo, Wan22Paths.InRepo(dit)))));

        var rgb = startImage is null ? null : VideoFrames.Rgb(startImage);
        var request = new WanRequest
        {
            Prompt = prompt, NegativePrompt = negativePrompt, Width = width, Height = height, Frames = length, Steps = steps,
            GuidanceScale = cfg, Shift = shift, Seed = seed, Sampler = sampler == "few_step" ? WanSampler.FewStep : WanSampler.UniPC, RgbaFrames = true, TiledDecode = tiledDecode,
            ImageRgb = rgb, ImageWidth = startImage?.Width ?? 0, ImageHeight = startImage?.Height ?? 0,
        };
        var progress = new Progress<WanProgress>(p =>
        {
            context.Progress(p.Fraction, 1, $"Wan 2.2: {p.Stage}{(string.IsNullOrEmpty(p.Message) ? "" : " " + p.Message)}");
        });
        var video = await loaded.Pipeline.GenerateAsync(request, progress, context.CancellationToken);
        return (VideoFrames.Images(video), FrameRate);
    }
}

/// <summary>Between the library's video results and SpawnGen's IMAGE batches.</summary>
static class VideoFrames
{
    /// <summary>A picture of a batch as packed 8-bit RGB (how the library's pipelines take a start picture).</summary>
    public static byte[] Rgb(ImageBatch image, int index = 0)
    {
        var packed = Rgba.Pack(image, index);
        var rgb = new byte[packed.Length * 3];
        for (int i = 0; i < packed.Length; i++) { rgb[i * 3] = (byte)packed[i]; rgb[i * 3 + 1] = (byte)(packed[i] >> 8); rgb[i * 3 + 2] = (byte)(packed[i] >> 16); }
        return rgb;
    }

    /// <summary>A generated clip's frames as an IMAGE batch (0..1).</summary>
    public static ImageBatch Images(WanVideo video)
    {
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
        return frames;
    }
}

/// <summary>
/// LTX-Video 2B 0.9.8 distilled (Lightricks, LTXV Open Weights License) from the library's export: a video from a prompt, from a start picture and a
/// prompt, or between a start and an end picture (keyframes). One node for the whole model, like Wan 2.2: the library runs its T5-XXL text encoder,
/// the 7-step distilled transformer (guidance 1, so no negative prompt) and the video VAE in turn. Sides are multiples of 32 and the length is 8n+1
/// frames; 704x480 is the model's native size. "upscale" runs the 0.9.8 two-pass recipe: a first pass at 2/3 of the size, the x2 latent upscaler
/// and a short second pass, so the clip comes out at twice the first pass's size (704x480 asked gives 896x576). Gives the frames and the frame rate.
/// </summary>
[Node(Category = "video", DisplayName = "LTX-Video 2B (0.9.8 distilled)", OutputNames = ["images", "fps"])]
public sealed class LtxVideo
{
    public static string[] Precisions() => ["q8", "bf16"];

    sealed class Loaded(LtxPipeline pipeline) : IDisposable
    {
        public LtxPipeline Pipeline { get; } = pipeline;
        public void Dispose() => Pipeline.Dispose();
    }

    public async Task<(ImageBatch Images, double Fps)> ExecuteAsync(NodeContext context, IServiceProvider services,
        [Input(Multiline = true)] string prompt = "",
        [Input(Min = 32, Max = 2048, Step = 32, Tooltip = "704x480 is the model's native size.")] int width = 704,
        [Input(Min = 32, Max = 2048, Step = 32, Tooltip = "704x480 is the model's native size.")] int height = 480,
        [Input(Min = 1, Max = 257, Step = 8, Tooltip = "8n+1 frames: 121 is five seconds at 24 fps.")] int length = 121,
        [Input(Min = 1.0, Max = 60.0, Step = 1.0, Tooltip = "The rate the clip plays at; the model was trained on 24 to 30.")] double frameRate = 24.0,
        long seed = 42,
        [Input(Tooltip = "Optional: the first frame (image to video).")] ImageBatch? startImage = null,
        [Input(Min = 0.0, Max = 1.0, Step = 0.01, Tooltip = "How firmly the video keeps the start picture (1: exactly).")] double startStrength = 1.0,
        [Input(Tooltip = "Optional: the last frame (a keyframe the video moves towards).")] ImageBatch? endImage = null,
        [Input(Min = 0.0, Max = 1.0, Step = 0.01)] double endStrength = 1.0,
        [Input(Tooltip = "The 0.9.8 two-pass recipe: a first pass at 2/3 of the size, the x2 latent upscaler, a short second pass. Sharper; the clip comes out at twice the first pass's size.")] bool upscale = false,
        [Input(OptionsProvider = nameof(Precisions), Tooltip = "The transformer's weights: q8 (2.0 GB, fits an 8 GB card) or bf16 (3.6 GB).")] string precision = "q8")
    {
        if (string.IsNullOrWhiteSpace(prompt)) throw new ArgumentException("The prompt is empty.", nameof(prompt));
        if (width % 32 != 0 || height % 32 != 0) throw new ArgumentException($"LTX-Video needs sides that are multiples of 32, got {width}x{height}.");
        if ((length - 1) % 8 != 0) throw new ArgumentException($"LTX-Video makes 8n+1 frames (1, 9, 17 ... 121), got {length}.");
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var paths = LtxPaths.InRepo(LtxPaths.DitFile(precision == "bf16" ? "bf16" : "q8"));
        var loaded = await runtime.GetOrLoadAsync($"ltx:{LtxPipeline.DefaultRepo}:{paths.Dit}", () => Task.FromResult(new Loaded(
            new LtxPipeline(runtime.Accelerator, runtime.ModelSource, LtxPipeline.DefaultRepo, paths))));

        var conditions = new List<LtxCondition>();
        if (startImage is not null)
            conditions.Add(LtxCondition.Image(VideoFrames.Rgb(startImage), startImage.Width, startImage.Height, 0, (float)startStrength));
        if (endImage is not null)
            conditions.Add(LtxCondition.Image(VideoFrames.Rgb(endImage), endImage.Width, endImage.Height, length - 1, (float)endStrength));
        var request = new LtxRequest
        {
            Prompt = prompt, Width = width, Height = height, Frames = length, FrameRate = (float)frameRate, Seed = seed, Upscale = upscale, RgbaFrames = true,
            Conditions = conditions.Count > 0 ? conditions : null,
        };
        var progress = new Progress<WanProgress>(p =>
            context.Progress(p.Fraction, 1, $"LTX-Video: {p.Stage}{(string.IsNullOrEmpty(p.Message) ? "" : " " + p.Message)}"));
        var video = await loaded.Pipeline.GenerateAsync(request, progress, context.CancellationToken);
        return (VideoFrames.Images(video), frameRate);
    }
}
