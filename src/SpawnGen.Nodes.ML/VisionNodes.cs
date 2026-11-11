using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries for Load Upscale Model, Upscale Image (using Model), Load Background
// Removal Model and Remove Background; the models run through SpawnDev.ILGPU.ML's SuperResolutionPipeline (ESPCN) and BackgroundRemovalPipeline
// (RMBG 1.4), as their docs describe them.

/// <summary>UPSCALE_MODEL: an upscaling model on the model source and its scale.</summary>
[Socket("UPSCALE_MODEL")]
public sealed record UpscaleModel(string Name, string Repo, string File, int Scale);

[Node("UpscaleModelLoader", Category = "model/loaders", DisplayName = "Load Upscale Model")]
public sealed class UpscaleModelLoader
{
    /// <summary>The upscalers SpawnGen runs (ONNX exports it downloads on first use).</summary>
    public static readonly IReadOnlyDictionary<string, UpscaleModel> Known = new Dictionary<string, UpscaleModel>
    {
        ["espcn_x3.onnx"] = new("espcn_x3.onnx", "onnxmodelzoo/super-resolution-10", "super-resolution-10.onnx", 3),
    };

    public static string[] Models() => [.. Known.Keys];

    public UpscaleModel Execute([Input(OptionsProvider = nameof(Models))] string modelName = "espcn_x3.onnx") =>
        Known.TryGetValue(modelName, out var m) ? m : throw new ArgumentException($"Unknown upscale model '{modelName}'. The choices are: {string.Join(", ", Known.Keys)}.", nameof(modelName));
}

[Node("ImageUpscaleWithModel", Category = "image/upscaling", DisplayName = "Upscale Image (using Model)")]
public sealed class ImageUpscaleWithModel
{
    /// <summary>Every picture upscaled by the model (tiled over the model's patch size by the pipeline); alpha is dropped.</summary>
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
        int width = 0, height = 0;
        for (int b = 0; b < image.Batch; b++)
        {
            context.Progress(b, image.Batch);
            var r = await pipeline.UpscaleAsync(Rgba.Pack(image, b), image.Width, image.Height);
            frames.Add(r.RgbaPixels);
            (width, height) = (r.Width, r.Height);
        }
        return Rgba.Unpack(frames, width, height);
    }
}

/// <summary>BACKGROUND_REMOVAL: the chosen background removal model.</summary>
[Socket("BACKGROUND_REMOVAL")]
public sealed class BackgroundRemovalModel(string name)
{
    public string Name { get; } = name;
}

[Node("LoadBackgroundRemovalModel", Category = "model/loaders", DisplayName = "Load Background Removal Model", OutputNames = ["bg_model"])]
public sealed class LoadBackgroundRemovalModel
{
    public static string[] Models() => [RemoveBackground.Rmbg14];

    public BackgroundRemovalModel Execute(
        [Input(OptionsProvider = nameof(Models), Tooltip = "The model used to remove backgrounds from images")] string bgRemovalName = RemoveBackground.Rmbg14) =>
        bgRemovalName == RemoveBackground.Rmbg14 ? new BackgroundRemovalModel(bgRemovalName)
            : throw new ArgumentException($"Unknown background removal model '{bgRemovalName}'.", nameof(bgRemovalName));
}

[Node("RemoveBackground", Category = "image/background removal", DisplayName = "Remove Background", OutputNames = ["mask"],
    Description = "Generates a foreground mask to remove the background from an image using a background removal model.")]
public sealed class RemoveBackground
{
    public const string Rmbg14 = "RMBG-1.4";

    /// <summary>The soft mask: the model's foreground probability, not thresholded.</summary>
    public const float MaskThreshold = 0f;

    const string Repo = "briaai/RMBG-1.4", File = "onnx/model.onnx";

    public async Task<MaskBatch> ExecuteAsync(
        [Input(Tooltip = "Background removal model used to generate the mask")] BackgroundRemovalModel bgRemovalModel,
        [Input(Tooltip = "Input image to remove the background from")] ImageBatch image,
        NodeContext context, IServiceProvider services) =>
        await MasksAsync(image, context, services, MaskThreshold);

    /// <summary>One foreground mask (1 = foreground) per picture, at the picture's size.</summary>
    public static async Task<MaskBatch> MasksAsync(ImageBatch image, NodeContext context, IServiceProvider services, float threshold)
    {
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var pipeline = await runtime.GetOrLoadAsync($"rmbg:{Repo}/{File}", async () =>
        {
            await using var stream = await runtime.ModelSource.OpenAsync(Repo, File);
            var session = await InferenceSession.CreateFromOnnxStreamAsync(runtime.Accelerator, stream);
            return new BackgroundRemovalPipeline(session, runtime.Accelerator);
        });
        int plane = image.Height * image.Width;
        var masks = new MaskBatch(image.Batch, image.Height, image.Width);
        for (int b = 0; b < image.Batch; b++)
        {
            context.Progress(b, image.Batch);
            var r = await pipeline.RemoveBackgroundAsync(Rgba.Pack(image, b), image.Width, image.Height, threshold);
            Array.Copy(r.Mask, 0, masks.Data, b * plane, plane);
        }
        return masks;
    }
}
