using System.Text;
using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.StableDiffusion;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md): SpawnGen's own design. A checkpoint is the MODEL, CLIP and VAE that Load Checkpoint puts out, over a
// single-file checkpoint (SafetensorsCheckpointModels, SpawnDev.ILGPU.ML's SdCheckpoint) or an ONNX export (OnnxCheckpointModels). The CLIP
// tokenizer is OpenAI's CLIP ViT-L/14 vocabulary (vocab.json, merges.txt), which SD 1.x, SD 2.x and SDXL share.

/// <summary>A loaded Stable Diffusion checkpoint: diffusion model, text encoder(s) and VAE.</summary>
public sealed class Checkpoint : IDiffusionModel, ITextEncoderModel, IVaeModel, IDisposable
{
    public const int LatentChannels = 4;

    /// <summary>The token count of one CLIP prompt chunk.</summary>
    public const int TokenCount = 77;

    /// <summary>Where the CLIP tokenizer of a single-file checkpoint comes from.</summary>
    public const string ClipTokenizerRepo = "openai/clip-vit-large-patch14";

    readonly ICheckpointModels _models;
    readonly EmptyPromptCache _empty = new();
    readonly Lazy<ComponentWeights?> _unetWeights, _textWeights;

    internal Checkpoint(string name, ICheckpointModels models, BPETokenizer tokenizer)
    {
        Name = name;
        _models = models;
        Tokenizer = tokenizer;
        Family = models.Family;
        LatentFormat = SdFamilies.LatentFormat(Family);
        TextWidth = models.TextWidth;
        FirstTowerWidth = SdFamilies.FirstTowerWidth(Family, TextWidth);
        Sigmas = ModelSigmas.LinearSchedule();
        Sampling = new ModelSampling(Sigmas, models.Prediction);
        DefaultClipLayer = SdFamilies.DefaultClipLayer(Family);
        _unetWeights = new(() => models.UNet is { } u ? ComponentWeights.For(u, models.Source) : null);
        _textWeights = new(() => models.TextEncoder is { } t ? ComponentWeights.For(t, models.Source) : null);
    }

    public string Name { get; }
    public SdFamily Family { get; }
    public LatentFormat LatentFormat { get; }
    public int TextWidth { get; }
    public int FirstTowerWidth { get; }
    public BPETokenizer Tokenizer { get; }
    public ModelSigmas Sigmas { get; }
    public ModelSampling Sampling { get; }
    public int DefaultClipLayer { get; }
    public SdUNet? UNet => _models.UNet;
    public SdTextEncoder? TextEncoder => _models.TextEncoder;
    public SdVae? SdVae => _models.Vae;

    /// <summary>The single-file checkpoint (null for an ONNX export): Save Checkpoint writes it back with its current weights.</summary>
    public SdCheckpoint? File => _models.File;

    public Task<TextEncoding> EncodeAsync(int[] ids, int[]? ids2 = null, int? layer = null, CancellationToken cancellationToken = default) =>
        _models.EncodeAsync(ids, ids2, layer, cancellationToken);

    public Task<TextEncoding> EmptyPromptAsync(int? layer = null, IReadOnlyList<WeightStep>? clipWeights = null) =>
        _empty.GetAsync(this, layer ?? DefaultClipLayer, clipWeights ?? []);

    Task IDiffusionModel.UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) =>
        UseAsync(_unetWeights.Value, weights, cancellationToken);

    Task ITextEncoderModel.UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) =>
        UseAsync(_textWeights.Value, weights, cancellationToken);

    Task UseAsync(ComponentWeights? component, IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken)
    {
        if (component is not null) return component.UseAsync(weights, cancellationToken);
        return weights.Count == 0 ? Task.CompletedTask
            : throw new NotSupportedException($"{Name} is an ONNX export: LoRAs and merges need a single-file (.safetensors) checkpoint.");
    }

    public UNetRun BeginRun(Accelerator accelerator) => _models.BeginRun(accelerator);

    public Task<float[]> DecodeAsync(float[] latent, int h, int w, CancellationToken cancellationToken = default) => _models.DecodeAsync(latent, h, w, cancellationToken);

    public Task<float[]> EncodeImageAsync(float[] image, int H, int W, CancellationToken cancellationToken = default) => _models.EncodeImageAsync(image, H, W, cancellationToken);

    /// <summary>
    /// Loads a checkpoint: a single-file one when <paramref name="info"/> names its file (tokenizer from <see cref="ClipTokenizerRepo"/>), otherwise an
    /// ONNX export with its own tokenizer folder. <paramref name="onProgress"/> gets a stage and a percentage.
    /// </summary>
    public static async Task<Checkpoint> LoadAsync(MlRuntime runtime, string name, string repoId, MlRuntime.CheckpointInfo info, Action<string, int>? onProgress)
    {
        ICheckpointModels models = info.File is { } file
            ? await SafetensorsCheckpointModels.LoadAsync(runtime, repoId, file, onProgress)
            : await OnnxCheckpointModels.LoadAsync(runtime, repoId, info, onProgress);
        try
        {
            var tokenizer = info.File is not null
                ? await LoadClipTokenizerAsync(runtime.ModelSource, ClipTokenizerRepo, "", onProgress)
                : await LoadClipTokenizerAsync(runtime.ModelSource, repoId, "tokenizer", onProgress);
            return new Checkpoint(name, models, tokenizer);
        }
        catch
        {
            models.Dispose();
            throw;
        }
    }

    /// <summary>The CLIP BPE tokenizer from <c>vocab.json</c> and <c>merges.txt</c> in <paramref name="folder"/> of the repo ("" for its top).</summary>
    public static async Task<BPETokenizer> LoadClipTokenizerAsync(IModelSource source, string repoId, string folder, Action<string, int>? onProgress)
    {
        var prefix = string.IsNullOrEmpty(folder) ? "" : folder.TrimEnd('/') + "/";
        onProgress?.Invoke("tokenizer", 0);
        var vocab = Encoding.UTF8.GetString(await source.FetchBytesAsync(repoId, prefix + "vocab.json"));
        var merges = Encoding.UTF8.GetString(await source.FetchBytesAsync(repoId, prefix + "merges.txt"));
        onProgress?.Invoke("tokenizer", 100);
        return BPETokenizer.Load(vocab, merges);
    }

    public void Dispose()
    {
        if (_unetWeights.IsValueCreated) _unetWeights.Value?.Dispose();
        if (_textWeights.IsValueCreated) _textWeights.Value?.Dispose();
        _models.Dispose();
    }
}
