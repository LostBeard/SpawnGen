using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

/// <summary>A model file on the model source, with what a node needs to know to use it.</summary>
[Socket("UPSCALE_MODEL")]
public sealed record UpscaleModel(string Name, string Repo, string File, int Scale);

/// <summary>Host pictures to and from the packed RGBA ints the library's vision pipelines take (byte order R, G, B, A from the low byte up).</summary>
internal static class Rgba
{
    public static int[] Pack(ImageBatch images, int frame)
    {
        int w = images.Width, h = images.Height, c = images.Channels;
        var packed = new int[w * h];
        int offset = frame * h * w * c;
        for (int i = 0; i < packed.Length; i++)
        {
            int p = offset + i * c;
            int r = Byte(images.Data[p]), g = Byte(images.Data[p + Math.Min(1, c - 1)]), b = Byte(images.Data[p + Math.Min(2, c - 1)]);
            int a = c >= 4 ? Byte(images.Data[p + 3]) : 255;
            packed[i] = r | (g << 8) | (b << 16) | (a << 24);
        }
        return packed;
    }

    /// <summary>Unpacks into RGB; the alpha is returned separately when <paramref name="alpha"/> is given.</summary>
    public static ImageBatch Unpack(IReadOnlyList<int[]> frames, int width, int height, float[]? alpha = null)
    {
        var images = new ImageBatch(frames.Count, height, width, 3);
        for (int f = 0; f < frames.Count; f++)
            for (int i = 0; i < width * height; i++)
            {
                int v = frames[f][i], o = (f * width * height + i) * 3;
                images.Data[o] = (v & 255) / 255f;
                images.Data[o + 1] = ((v >> 8) & 255) / 255f;
                images.Data[o + 2] = ((v >> 16) & 255) / 255f;
                if (alpha is not null) alpha[f * width * height + i] = ((v >>> 24) & 255) / 255f;
            }
        return images;
    }

    static int Byte(float v) => Math.Clamp((int)MathF.Round(v * 255f), 0, 255);
}

[Node(Category = "loaders", DisplayName = "Load Upscale Model")]
public sealed class UpscaleModelLoader
{
    /// <summary>The upscalers the library can run, by the name the node shows.</summary>
    public static readonly IReadOnlyDictionary<string, UpscaleModel> Known = new Dictionary<string, UpscaleModel>
    {
        ["espcn_x3.onnx"] = new("espcn_x3.onnx", "onnxmodelzoo/super-resolution-10", "super-resolution-10.onnx", 3),
    };

    public static string[] Models() => [.. Known.Keys];

    public UpscaleModel Execute([Input(OptionsProvider = nameof(Models))] string modelName = "espcn_x3.onnx") =>
        Known.TryGetValue(modelName, out var model) ? model
            : throw new ArgumentException($"Unknown upscale model '{modelName}'. Known: {string.Join(", ", Known.Keys)}.", nameof(modelName));
}

[Node(Category = "image/upscaling", DisplayName = "Upscale Image (using Model)")]
public sealed class ImageUpscaleWithModel
{
    /// <summary>Upscales each picture with the model (the original scale factor of the model: x3 for ESPCN). Colour comes from the source, detail from the network.</summary>
    public async Task<ImageBatch> ExecuteAsync(UpscaleModel upscaleModel, ImageBatch image, NodeContext context, IServiceProvider services)
    {
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var pipeline = await runtime.GetOrLoadAsync($"upscale:{upscaleModel.Repo}/{upscaleModel.File}", async () =>
        {
            await using var stream = await runtime.ModelSource.OpenAsync(upscaleModel.Repo, upscaleModel.File);
            var session = await InferenceSession.CreateFromOnnxStreamAsync(runtime.Accelerator, stream);
            return new SuperResolutionPipeline(session, runtime.Accelerator, upscaleModel.Scale);
        });

        var frames = new List<int[]>();
        int outW = 0, outH = 0;
        for (int b = 0; b < image.Batch; b++)
        {
            context.CancellationToken.ThrowIfCancellationRequested();
            var result = await pipeline.UpscaleAsync(Rgba.Pack(image, b), image.Width, image.Height);
            frames.Add(result.RgbaPixels);
            (outW, outH) = (result.Width, result.Height);
            context.Progress(b + 1, image.Batch);
        }
        return Rgba.Unpack(frames, outW, outH);
    }
}

[Node(Category = "image/preprocessors", DisplayName = "Remove Background")]
public sealed class RemoveBackground
{
    const string Repo = "briaai/RMBG-1.4", File = "onnx/model.onnx";

    /// <summary>
    /// Finds the foreground with RMBG 1.4: the picture comes back unchanged and the mask says how much of each pixel is foreground (1) or background (0).
    /// Combine them with Composite Image (masked), or invert the mask to select the background.
    /// </summary>
    public async Task<(ImageBatch Image, MaskBatch Mask)> ExecuteAsync(ImageBatch image, NodeContext context, IServiceProvider services,
        [Input(Min = 0, Max = 1, Step = 0.01)] float threshold = 0.5f)
    {
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var pipeline = await runtime.GetOrLoadAsync($"rmbg:{Repo}", async () =>
        {
            await using var stream = await runtime.ModelSource.OpenAsync(Repo, File);
            var session = await InferenceSession.CreateFromOnnxStreamAsync(runtime.Accelerator, stream,
                inputShapes: new Dictionary<string, int[]> { ["input"] = [1, 3, 1024, 1024] });
            return new BackgroundRemovalPipeline(session, runtime.Accelerator, 1024);
        });

        var mask = new MaskBatch(image.Batch, image.Height, image.Width);
        for (int b = 0; b < image.Batch; b++)
        {
            context.CancellationToken.ThrowIfCancellationRequested();
            var result = await pipeline.RemoveBackgroundAsync(Rgba.Pack(image, b), image.Width, image.Height, threshold);
            for (int i = 0; i < image.Width * image.Height; i++) mask.Data[b * image.Width * image.Height + i] = ((result.ResultPixels[i] >>> 24) & 255) / 255f;
            context.Progress(b + 1, image.Batch);
        }
        return (image, mask);
    }
}

[Node(Category = "image/preprocessors", DisplayName = "Depth Anything V2")]
public sealed class DepthAnythingV2
{
    const string Repo = "onnx-community/depth-anything-v2-small";

    /// <summary>A depth map of each picture as a grey picture of the same size: white is near, black is far (what depth ControlNets expect).</summary>
    public async Task<ImageBatch> ExecuteAsync(ImageBatch image, NodeContext context, IServiceProvider services)
    {
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var pipeline = await runtime.GetOrLoadAsync($"depth:{Repo}", () =>
            DepthEstimationPipeline.CreateFromHubAsync(runtime.Accelerator, runtime.ModelSource, Repo, "onnx/model.onnx", externalDataFile: "",
                inputShapes: new Dictionary<string, int[]> { ["pixel_values"] = [1, 3, 518, 518] }));

        var result = new ImageBatch(image.Batch, image.Height, image.Width, 3);
        for (int b = 0; b < image.Batch; b++)
        {
            context.CancellationToken.ThrowIfCancellationRequested();
            var depth = await pipeline.EstimateAsync(Rgba.Pack(image, b), image.Width, image.Height);
            if (depth.Width != image.Width || depth.Height != image.Height)
                throw new InvalidOperationException($"The depth model returned {depth.Width}x{depth.Height} for a {image.Width}x{image.Height} picture.");
            for (int i = 0; i < image.Width * image.Height; i++)
            {
                float v = Math.Clamp(depth.DepthMap[i], 0f, 1f);
                int o = (b * image.Width * image.Height + i) * 3;
                result.Data[o] = result.Data[o + 1] = result.Data[o + 2] = v;
            }
            context.Progress(b + 1, image.Batch);
        }
        return result;
    }
}

[Node(Category = "image/style", DisplayName = "Style Transfer")]
public sealed class StyleTransfer
{
    /// <summary>The fast neural styles of the ONNX model zoo, by the name the node shows.</summary>
    public static readonly IReadOnlyDictionary<string, (string Repo, string File)> Styles = new Dictionary<string, (string, string)>
    {
        ["mosaic"] = ("onnxmodelzoo/mosaic-9", "mosaic-9.onnx"),
        ["candy"] = ("onnxmodelzoo/candy-9", "candy-9.onnx"),
        ["rain_princess"] = ("onnxmodelzoo/rain-princess-9", "rain-princess-9.onnx"),
        ["udnie"] = ("onnxmodelzoo/udnie-9", "udnie-9.onnx"),
        ["pointilism"] = ("onnxmodelzoo/pointilism-9", "pointilism-9.onnx"),
    };

    public static string[] StyleNames() => [.. Styles.Keys];

    public async Task<ImageBatch> ExecuteAsync(ImageBatch image, NodeContext context, IServiceProvider services,
        [Input(OptionsProvider = nameof(StyleNames))] string style = "mosaic")
    {
        if (!Styles.TryGetValue(style, out var model)) throw new ArgumentException($"Unknown style '{style}'. Known: {string.Join(", ", Styles.Keys)}.", nameof(style));
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var pipeline = await runtime.GetOrLoadAsync($"style:{model.Repo}", async () =>
        {
            await using var stream = await runtime.ModelSource.OpenAsync(model.Repo, model.File);
            var session = await InferenceSession.CreateFromOnnxStreamAsync(runtime.Accelerator, stream);
            return new StyleTransferPipeline(session, runtime.Accelerator);
        });

        var frames = new List<int[]>();
        int outW = image.Width, outH = image.Height;
        for (int b = 0; b < image.Batch; b++)
        {
            context.CancellationToken.ThrowIfCancellationRequested();
            var result = await pipeline.TransferAsync(Rgba.Pack(image, b), image.Width, image.Height);
            frames.Add(result.RgbaPixels);
            (outW, outH) = (result.Width, result.Height);
            context.Progress(b + 1, image.Batch);
        }
        return Rgba.Unpack(frames, outW, outH);
    }
}

[Node(Category = "image/preprocessors", DisplayName = "Pose Estimation (MoveNet)")]
public sealed class PoseEstimation
{
    const string Repo = "Xenova/movenet-singlepose-lightning", File = "onnx/model.onnx";

    /// <summary>
    /// Finds the 17 body keypoints of one person (MoveNet Lightning) and draws the skeleton: on black, which is what pose ControlNets take, or over the picture.
    /// <c>keypoints</c> lists them as <c>name x y confidence</c>, one per line, in picture pixels.
    /// </summary>
    public async Task<(ImageBatch Image, string Keypoints)> ExecuteAsync(ImageBatch image, NodeContext context, IServiceProvider services,
        [Input(Min = 0, Max = 1, Step = 0.01)] float confidence = 0.3f, bool overlay = false)
    {
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var pipeline = await runtime.GetOrLoadAsync($"pose:{Repo}", async () =>
        {
            await using var stream = await runtime.ModelSource.OpenAsync(Repo, File);
            var session = await InferenceSession.CreateFromOnnxStreamAsync(runtime.Accelerator, stream);
            return new PoseEstimationPipeline(session, runtime.Accelerator);
        });

        var frames = new List<int[]>();
        var text = new System.Text.StringBuilder();
        for (int b = 0; b < image.Batch; b++)
        {
            context.CancellationToken.ThrowIfCancellationRequested();
            var packed = Rgba.Pack(image, b);
            var result = await pipeline.EstimateAsync(packed, image.Width, image.Height, confidence);
            var canvas = new byte[image.Width * image.Height * 4];
            if (overlay) Buffer.BlockCopy(packed, 0, canvas, 0, canvas.Length);
            else for (int i = 3; i < canvas.Length; i += 4) canvas[i] = 255;
            DrawingUtils.DrawSkeleton(canvas, image.Width, image.Height, result.Keypoints, confidence, lineThickness: Math.Max(2, image.Width / 128), pointRadius: Math.Max(3, image.Width / 96));
            var outPacked = new int[packed.Length];
            Buffer.BlockCopy(canvas, 0, outPacked, 0, canvas.Length);
            frames.Add(outPacked);
            foreach (var k in result.Keypoints)
                text.AppendLine(FormattableString.Invariant($"{k.Name} {k.X:0.#} {k.Y:0.#} {k.Confidence:0.00}"));
            context.Progress(b + 1, image.Batch);
        }
        return (Rgba.Unpack(frames, image.Width, image.Height), text.ToString().TrimEnd());
    }
}

[Node(Category = "image/preprocessors", DisplayName = "Object Detection (YOLOv8)")]
public sealed class ObjectDetection
{
    const string Repo = "salim4n/yolov8n-detect-onnx", File = "yolov8n-onnx-web/yolov8n.onnx";

    /// <summary>
    /// Finds the 80 COCO kinds of object (YOLOv8 nano) and draws a box around each. <c>detections</c> lists them as <c>label confidence x y width height</c>, one per line,
    /// in picture pixels (the box is each object's colour; the label is in the list).
    /// </summary>
    public async Task<(ImageBatch Image, string Detections)> ExecuteAsync(ImageBatch image, NodeContext context, IServiceProvider services,
        [Input(Min = 0, Max = 1, Step = 0.01)] float confidence = 0.25f)
    {
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var pipeline = await runtime.GetOrLoadAsync($"yolo:{Repo}", async () =>
        {
            await using var stream = await runtime.ModelSource.OpenAsync(Repo, File);
            var session = await InferenceSession.CreateFromOnnxStreamAsync(runtime.Accelerator, stream);
            return new ObjectDetectionPipeline(session, runtime.Accelerator);
        });

        var frames = new List<int[]>();
        var text = new System.Text.StringBuilder();
        for (int b = 0; b < image.Batch; b++)
        {
            context.CancellationToken.ThrowIfCancellationRequested();
            var packed = Rgba.Pack(image, b);
            var result = await pipeline.DetectAsync(packed, image.Width, image.Height, confidence);
            var canvas = new byte[image.Width * image.Height * 4];
            Buffer.BlockCopy(packed, 0, canvas, 0, canvas.Length);
            foreach (var o in result.Objects)
            {
                var (r, g, bl) = DrawingUtils.GetClassColor(o.ClassId);
                DrawingUtils.DrawRect(canvas, image.Width, image.Height, (int)o.X, (int)o.Y, (int)(o.X + o.Width), (int)(o.Y + o.Height), r, g, bl, Math.Max(2, image.Width / 200));
                text.AppendLine(FormattableString.Invariant($"{o.Label} {o.Confidence:0.00} {o.X:0} {o.Y:0} {o.Width:0} {o.Height:0}"));
            }
            var outPacked = new int[packed.Length];
            Buffer.BlockCopy(canvas, 0, outPacked, 0, canvas.Length);
            frames.Add(outPacked);
            context.Progress(b + 1, image.Batch);
        }
        return (Rgba.Unpack(frames, image.Width, image.Height), text.ToString().TrimEnd());
    }
}
