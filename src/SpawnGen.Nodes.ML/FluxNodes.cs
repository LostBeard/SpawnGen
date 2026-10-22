using SpawnDev.ILGPU.ML.Flux2;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// FLUX.2 klein 4B (Black Forest Labs, Apache-2.0) from the library's 4-bit / 8-bit export: text to image, and picture editing or composing when reference pictures are given.
/// One node for the whole model, because the library runs its three stages (Qwen3 text encoder, flow transformer, VAE) one after the other on the GPU and streams the big two in per run.
/// </summary>
[Node(Category = "sampling", DisplayName = "FLUX.2 Klein")]
public sealed class Flux2Klein
{
    const string Repo = "LostBeard/FLUX.2-klein-4B-q4-ILGPU";

    public static string[] Weights() => ["4-bit", "8-bit"];

    /// <summary>
    /// Draws <paramref name="batchSize"/> pictures of the prompt (seeds <paramref name="seed"/>, +1, ...). With <paramref name="referenceImages"/> (up to 10) the prompt says what to do with them
    /// ("give the knight a red plume") and the picture takes the first one's proportions; width and height are then ignored. The klein model is distilled for 4 steps.
    /// </summary>
    public async Task<ImageBatch> ExecuteAsync(NodeContext context, IServiceProvider services,
        [Input(Multiline = true)] string prompt = "",
        [Input(Min = 64, Max = 2048, Step = 16)] int width = 512,
        [Input(Min = 64, Max = 2048, Step = 16)] int height = 512,
        [Input(Min = 1, Max = 16)] int steps = 4,
        long seed = 0,
        [Input(Min = 1, Max = 16)] int batchSize = 1,
        [Input(OptionsProvider = nameof(Weights))] string weights = "4-bit",
        ImageBatch? referenceImages = null)
    {
        if (string.IsNullOrWhiteSpace(prompt)) throw new ArgumentException("The prompt is empty.", nameof(prompt));
        if (width % 16 != 0 || height % 16 != 0) throw new ArgumentException($"FLUX.2 needs sides that are multiples of 16, got {width}x{height}.");
        int bits = weights == "8-bit" ? 8 : 4;
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var pipeline = await runtime.GetOrLoadAsync($"flux2:{Repo}:{bits}", async () =>
        {
            string? last = null;
            var p = await Flux2Pipeline.CreateAsync(runtime.Accelerator, runtime.ModelSource, Repo, 512, (stage, pct) =>
            {
                if (stage != last) { last = stage; context.Status($"Loading FLUX.2 Klein: {stage}"); }
            }, context.CancellationToken, bits);
            p.KeepTransformerResident = true;
            return p;
        });

        IReadOnlyList<Flux2ReferenceImage>? references = null;
        if (referenceImages is not null)
        {
            if (referenceImages.Batch > Flux2Pipeline.MaxReferences) throw new ArgumentException($"At most {Flux2Pipeline.MaxReferences} reference pictures, got {referenceImages.Batch}.");
            var list = new List<Flux2ReferenceImage>();
            for (int b = 0; b < referenceImages.Batch; b++)
            {
                var packed = Rgba.Pack(referenceImages, b);
                var bytes = new byte[packed.Length * 4];
                Buffer.BlockCopy(packed, 0, bytes, 0, bytes.Length);
                list.Add(new Flux2ReferenceImage(referenceImages.Width, referenceImages.Height, bytes));
            }
            references = list;
        }

        var request = new Flux2Request
        {
            Prompt = prompt,
            Width = references is null ? width : 0,
            Height = references is null ? height : 0,
            Steps = steps,
            References = references,
            Progress = (stage, step, total) =>
            {
                context.CancellationToken.ThrowIfCancellationRequested();
                context.Status($"FLUX.2 Klein: {stage} {step}/{total}");
            },
        };
        var seeds = Enumerable.Range(0, batchSize).Select(i => (int)((seed + i) & 0x7fffffff)).ToArray();
        int done = 0;
        context.Progress(0, batchSize);
        var images = await pipeline.RunManyAsync(request, seeds, (_, _) => { context.Progress(++done, batchSize); return Task.CompletedTask; });

        var first = images[0];
        var result = new ImageBatch(images.Count, first.Height, first.Width, 3);
        for (int f = 0; f < images.Count; f++)
            for (int i = 0; i < first.Width * first.Height; i++)
            {
                int o = (f * first.Width * first.Height + i) * 3;
                result.Data[o] = images[f].Rgba[i * 4] / 255f;
                result.Data[o + 1] = images[f].Rgba[i * 4 + 1] / 255f;
                result.Data[o + 2] = images[f].Rgba[i * 4 + 2] / 255f;
            }
        return result;
    }
}

/// <summary>
/// FLUX 3 Image Klein 5.6B (Black Forest Labs, FLUX Kommunity License: non-commercial) from the library's 4-bit / 8-bit export: 16 steps, trained on full scene captions
/// (a few sentences describing subject, setting, light and style do far better than a short phrase), up to 10 reference pictures to edit or combine.
/// </summary>
[Node(Category = "sampling", DisplayName = "FLUX 3 Klein")]
public sealed class Flux3Klein
{
    const string Repo = "LostBeard/flux-3-image-klein-q4-ILGPU";

    public static string[] Weights() => ["4-bit", "8-bit"];

    public async Task<ImageBatch> ExecuteAsync(NodeContext context, IServiceProvider services,
        [Input(Multiline = true)] string caption = "",
        [Input(Min = 64, Max = 2048, Step = 16)] int width = 512,
        [Input(Min = 64, Max = 2048, Step = 16)] int height = 512,
        [Input(Min = 1, Max = 64)] int steps = 16,
        long seed = 0,
        [Input(Min = 1, Max = 16)] int batchSize = 1,
        [Input(OptionsProvider = nameof(Weights))] string weights = "4-bit",
        ImageBatch? referenceImages = null)
    {
        if (string.IsNullOrWhiteSpace(caption)) throw new ArgumentException("The caption is empty.", nameof(caption));
        if (width % 16 != 0 || height % 16 != 0) throw new ArgumentException($"FLUX 3 needs sides that are multiples of 16, got {width}x{height}.");
        int bits = weights == "8-bit" ? 8 : 4;
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var pipeline = await runtime.GetOrLoadAsync($"flux3:{Repo}:{bits}", async () =>
        {
            string? last = null;
            var p = await SpawnDev.ILGPU.ML.Flux3.Flux3Pipeline.CreateAsync(runtime.Accelerator, runtime.ModelSource, Repo, (stage, pct) =>
            {
                if (stage != last) { last = stage; context.Status($"Loading FLUX 3 Klein: {stage}"); }
            }, context.CancellationToken, bits);
            p.KeepTransformerResident = true;
            return p;
        });

        IReadOnlyList<Flux2ReferenceImage>? references = null;
        if (referenceImages is not null)
        {
            if (referenceImages.Batch > SpawnDev.ILGPU.ML.Flux3.Flux3Pipeline.MaxReferences) throw new ArgumentException($"At most {SpawnDev.ILGPU.ML.Flux3.Flux3Pipeline.MaxReferences} reference pictures, got {referenceImages.Batch}.");
            var list = new List<Flux2ReferenceImage>();
            for (int b = 0; b < referenceImages.Batch; b++)
            {
                var packed = Rgba.Pack(referenceImages, b);
                var bytes = new byte[packed.Length * 4];
                Buffer.BlockCopy(packed, 0, bytes, 0, bytes.Length);
                list.Add(new Flux2ReferenceImage(referenceImages.Width, referenceImages.Height, bytes));
            }
            references = list;
        }

        var images = new List<Flux2Image>();
        context.Progress(0, batchSize);
        for (int i = 0; i < batchSize; i++)
        {
            context.CancellationToken.ThrowIfCancellationRequested();
            int run = i;
            images.Add(await pipeline.RunAsync(new SpawnDev.ILGPU.ML.Flux3.Flux3Request
            {
                Caption = caption, Seed = (int)((seed + i) & 0x7fffffff), Width = references is null ? width : 0, Height = references is null ? height : 0,
                Steps = steps, References = references,
                Progress = (stage, step, total) =>
                {
                    context.CancellationToken.ThrowIfCancellationRequested();
                    context.Status($"FLUX 3 Klein {run + 1}/{batchSize}: {stage} {step}/{total}");
                },
            }));
            context.Progress(i + 1, batchSize);
        }

        var first = images[0];
        var result = new ImageBatch(images.Count, first.Height, first.Width, 3);
        for (int f = 0; f < images.Count; f++)
            for (int p = 0; p < first.Width * first.Height; p++)
            {
                int o = (f * first.Width * first.Height + p) * 3;
                result.Data[o] = images[f].Rgba[p * 4] / 255f;
                result.Data[o + 1] = images[f].Rgba[p * 4 + 1] / 255f;
                result.Data[o + 2] = images[f].Rgba[p * 4 + 2] / 255f;
            }
        return result;
    }
}
