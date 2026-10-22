using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Pipelines;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

[Node(Category = "image/analysis", DisplayName = "Image Classification (MobileNetV2)")]
public sealed class ImageClassification
{
    const string Repo = "onnxmodelzoo/mobilenetv2-12", File = "mobilenetv2-12.onnx";

    /// <summary>
    /// Names what is in each picture from the 1000 ImageNet classes (MobileNetV2). <c>labels</c> lists the best guesses, one per line as <c>label confidence</c>
    /// (several pictures: a heading per picture); <c>top</c> is just the best label of the first picture, to feed a prompt or a text switch.
    /// </summary>
    public async Task<(string Labels, string Top)> ExecuteAsync(ImageBatch image, NodeContext context, IServiceProvider services,
        [Input(Min = 1, Max = 20)] int topK = 5)
    {
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var pipeline = await runtime.GetOrLoadAsync($"classify:{Repo}", async () =>
        {
            await using var stream = await runtime.ModelSource.OpenAsync(Repo, File);
            var session = await InferenceSession.CreateFromOnnxStreamAsync(runtime.Accelerator, stream);
            return new ClassificationPipeline(session, runtime.Accelerator);
        });

        var text = new System.Text.StringBuilder();
        string top = "";
        for (int b = 0; b < image.Batch; b++)
        {
            context.CancellationToken.ThrowIfCancellationRequested();
            var results = await pipeline.ClassifyAsync(Rgba.Pack(image, b), image.Width, image.Height, topK);
            if (image.Batch > 1) text.AppendLine($"image {b + 1}:");
            foreach (var r in results) text.AppendLine(FormattableString.Invariant($"{Clean(r.Label)} {r.Confidence:0.000}"));
            if (b == 0 && results.Length > 0) top = Clean(results[0].Label);
            context.Progress(b + 1, image.Batch);
        }
        return (text.ToString().TrimEnd(), top);
    }

    /// <summary>ImageNet labels come as "n01440764 tench, Tinca tinca" in some lists: keep the words.</summary>
    static string Clean(string label) =>
        label.Length > 10 && label[0] == 'n' && char.IsDigit(label[1]) && label[9] == ' ' ? label[10..].Trim() : label;
}

[Node(Category = "image/analysis", DisplayName = "Match Labels to Image (CLIP)")]
public sealed class ZeroShotClassification
{
    const string Repo = "Xenova/clip-vit-base-patch32";

    /// <summary>
    /// Scores how well each of your own descriptions fits the picture (CLIP ViT-B/32; no training on your labels). Give the descriptions one per line
    /// or separated by commas, such as <c>a photo of a cat</c>. <c>ranked</c> lists them best first as <c>label share</c> (the shares add up to 1);
    /// <c>best</c> is the top description of the first picture.
    /// </summary>
    public async Task<(string Ranked, string Best)> ExecuteAsync(ImageBatch image, NodeContext context, IServiceProvider services,
        [Input(Multiline = true)] string labels = "a photo of a cat\na photo of a dog\na photo of a person")
    {
        var options = labels.Split(['\n', ','], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (options.Length < 2) throw new ArgumentException("Give at least two descriptions to choose between, one per line.", nameof(labels));

        var runtime = await GpuHelpers.RuntimeAsync(services);
        var pipeline = await runtime.GetOrLoadAsync($"clip:{Repo}", async () =>
        {
            await using var vision = await runtime.ModelSource.OpenAsync(Repo, "onnx/vision_model.onnx");
            var visionSession = await InferenceSession.CreateFromOnnxStreamAsync(runtime.Accelerator, vision);
            await using var textModel = await runtime.ModelSource.OpenAsync(Repo, "onnx/text_model.onnx");
            var textSession = await InferenceSession.CreateFromOnnxStreamAsync(runtime.Accelerator, textModel);
            var p = new ZeroShotClassificationPipeline(visionSession, textSession, runtime.Accelerator);
            p.LoadTokenizer(System.Text.Encoding.UTF8.GetString(await runtime.ModelSource.FetchBytesAsync(Repo, "tokenizer.json")));
            return p;
        });

        var text = new System.Text.StringBuilder();
        string best = "";
        for (int b = 0; b < image.Batch; b++)
        {
            context.CancellationToken.ThrowIfCancellationRequested();
            var result = await pipeline.ClassifyAsync(Rgba.Pack(image, b), image.Width, image.Height, options);
            if (image.Batch > 1) text.AppendLine($"image {b + 1}:");
            foreach (var p in result.Predictions) text.AppendLine(FormattableString.Invariant($"{p.Label} {p.Confidence:0.000}"));
            if (b == 0) best = result.TopLabel;
            context.Progress(b + 1, image.Batch);
        }
        return (text.ToString().TrimEnd(), best);
    }
}
