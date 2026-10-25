using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.ML;

/// <summary>What a text encoder made of one prompt: the context rows the UNet cross-attends to (<c>[1, rows, width]</c>) and, for SDXL, the pooled embedding.</summary>
public sealed record TextEncoding(float[] Context, int[] Shape, float[]? Pooled);

/// <summary>
/// A loaded Stable Diffusion checkpoint (text encoder(s), UNet, VAE and the CLIP tokenizer), whatever it was loaded from: an Optimum-style ONNX export
/// (<see cref="OnnxCheckpointModels"/>) or a single-file <c>.safetensors</c> checkpoint as ComfyUI loads them (<see cref="SafetensorsCheckpointModels"/>).
/// Nodes talk to it in host arrays; the sampler drives the UNet through a <see cref="UNetRun"/> that keeps each run's conditioning on the GPU.
/// </summary>
public sealed class Checkpoint : IDisposable
{
    public const int LatentChannels = 4;
    public const int TokenCount = 77;

    readonly ICheckpointModels _models;
    readonly object _gate = new();
    readonly Dictionary<int, Task<TextEncoding>> _emptyPrompt = new();

    internal Checkpoint(string name, ICheckpointModels models, BPETokenizer tokenizer)
    {
        Name = name;
        _models = models;
        Tokenizer = tokenizer;
        Sigmas = ModelSigmas.LinearSchedule(); // ComfyUI's float64 table (the library's alphas are float32 products, about 1e-5 off)
        Sampling = new ModelSampling(Sigmas, models.Prediction);
    }

    public string Name { get; }
    /// <summary>SD 1.x, SD 2.x or SDXL.</summary>
    public SdFamily Family => _models.Family;
    /// <summary>How the model's latents relate to its VAE's (scale, preview colours).</summary>
    public LatentFormat LatentFormat => Family == SdFamily.SDXL ? LatentFormat.SDXL : LatentFormat.SD15;
    /// <summary>The width of a context row (768 SD 1.x, 1024 SD 2.x, 2048 SDXL).</summary>
    public int TextWidth => _models.TextWidth;
    /// <summary>The width of the first tower's part of a row (SDXL: CLIP-L's 768 of 2048); the whole row for one-tower models.</summary>
    public int FirstTowerWidth => Family == SdFamily.SDXL ? 768 : TextWidth;
    public BPETokenizer Tokenizer { get; }
    /// <summary>The noise levels of the model's training steps, for schedulers and for telling the model its timestep.</summary>
    public ModelSigmas Sigmas { get; }
    /// <summary>The model's own sampling: eps or v prediction over <see cref="Sigmas"/>.</summary>
    public ModelSampling Sampling { get; }
    /// <summary>The layer a prompt is read from when CLIP Set Last Layer was not used (-1 the last, -2 the penultimate), ComfyUI's per-family default.</summary>
    public int DefaultClipLayer => Family == SdFamily.SD1 ? -1 : -2;

    /// <summary>
    /// Encodes token ids as the tokenizer makes them (<see cref="BPETokenizer.EncodeCLIP"/>: start, tokens, end, padded with the end token).
    /// Each tower gets ComfyUI's padding: CLIP-L pads with the end token, OpenCLIP (SD 2.x, SDXL's second tower) with 0. <paramref name="ids2"/> is SDXL's
    /// second tower's prompt when it differs (CLIP Text Encode SDXL's text_g). <paramref name="layer"/> null = <see cref="DefaultClipLayer"/>.
    /// </summary>
    public Task<TextEncoding> EncodeAsync(int[] ids, int[]? ids2 = null, int? layer = null, CancellationToken cancellationToken = default) =>
        _models.EncodeAsync(ids, ids2, layer, cancellationToken);

    /// <summary>The empty prompt's encoding at a layer, computed once: the baseline weighted prompts are measured against.</summary>
    public Task<TextEncoding> EmptyPromptAsync(int? layer = null)
    {
        int key = layer ?? DefaultClipLayer;
        lock (_gate)
        {
            if (!_emptyPrompt.TryGetValue(key, out var task) || task.IsFaulted || task.IsCanceled)
                _emptyPrompt[key] = task = EncodeAsync(Tokenizer.EncodeCLIP("", TokenCount), layer: key);
            return task;
        }
    }

    /// <summary>Starts a sampling run on the UNet; dispose it when the run ends (it frees the run's GPU buffers).</summary>
    internal UNetRun BeginRun(Accelerator accelerator) => _models.BeginRun(accelerator);

    /// <summary>Decodes one latent in the VAE's values, [4, h, w], to a picture [3, 8h, 8w] in [-1, 1] (not clamped).</summary>
    public Task<float[]> DecodeAsync(float[] latent, int h, int w, CancellationToken cancellationToken = default) =>
        _models.DecodeAsync(latent, h, w, cancellationToken);

    /// <summary>Encodes a picture [3, H, W] in [-1, 1] (H and W multiples of 8) to the VAE's latent [4, H/8, W/8] (the posterior mean).</summary>
    public Task<float[]> EncodeImageAsync(float[] image, int H, int W, CancellationToken cancellationToken = default) =>
        _models.EncodeImageAsync(image, H, W, cancellationToken);

    internal static async Task<Checkpoint> LoadAsync(MlRuntime runtime, string name, string repoId, MlRuntime.CheckpointInfo info, Action<string, int>? onProgress)
    {
        if (info.File is { } file)
        {
            var tokenizer = await LoadClipTokenizerAsync(runtime.ModelSource, ClipTokenizerRepo, "", onProgress);
            var models = await SafetensorsCheckpointModels.LoadAsync(runtime, repoId, file, onProgress);
            return new Checkpoint(name, models, tokenizer);
        }
        var onnxTokenizer = await LoadClipTokenizerAsync(runtime.ModelSource, repoId, "tokenizer/", onProgress);
        var onnx = await OnnxCheckpointModels.LoadAsync(runtime, repoId, info, onProgress);
        return new Checkpoint(name, onnx, onnxTokenizer);
    }

    /// <summary>
    /// Where a single-file checkpoint's tokenizer comes from (the file holds none): CLIP ViT-L's vocabulary, which OpenCLIP ViT-H and ViT-bigG share
    /// (ComfyUI ships the same files as its sd1_tokenizer).
    /// </summary>
    public const string ClipTokenizerRepo = "openai/clip-vit-large-patch14";

    static async Task<BPETokenizer> LoadClipTokenizerAsync(SpawnDev.ILGPU.ML.Hub.IModelSource source, string repoId, string folder, Action<string, int>? onProgress)
    {
        onProgress?.Invoke("tokenizer", 0);
        string vocabJson, mergesText;
        await using (var s = await source.OpenAsync(repoId, folder + "vocab.json"))
        using (var r = new StreamReader(s))
            vocabJson = await r.ReadToEndAsync();
        await using (var s = await source.OpenAsync(repoId, folder + "merges.txt"))
        using (var r = new StreamReader(s))
            mergesText = await r.ReadToEndAsync();
        onProgress?.Invoke("tokenizer", 100);
        return BPETokenizer.Load(vocabJson, mergesText);
    }

    public void Dispose() => _models.Dispose();
}




/// <summary>One sampling run's UNet: each conditioning entry goes to the GPU once, every step uploads only the latent.</summary>
internal abstract class UNetRun : IDisposable
{
    /// <summary>One prediction (eps or v, as the model was trained) for <paramref name="input"/> [4, h, w] in the model's space at <paramref name="timestep"/>.</summary>
    public abstract Task<float[]> PredictAsync(float[] input, int h, int w, float timestep, CondEntry entry, CancellationToken cancellationToken);
    public abstract void Dispose();
}

/// <summary>What a checkpoint is loaded as.</summary>
internal interface ICheckpointModels : IDisposable
{
    SdFamily Family { get; }
    PredictionType Prediction { get; }
    int TextWidth { get; }
    Task<TextEncoding> EncodeAsync(int[] ids, int[]? ids2, int? layer, CancellationToken cancellationToken);
    UNetRun BeginRun(Accelerator accelerator);
    Task<float[]> DecodeAsync(float[] latent, int h, int w, CancellationToken cancellationToken);
    Task<float[]> EncodeImageAsync(float[] image, int H, int W, CancellationToken cancellationToken);
}

/// <summary>The denoiser (UNet) of a loaded checkpoint, with the sampling math constants.</summary>
[Socket("MODEL")]
public sealed class ModelHandle(Checkpoint checkpoint, ModelSampling sampling)
{
    public ModelHandle(Checkpoint checkpoint) : this(checkpoint, checkpoint?.Sampling!) { }

    public Checkpoint Checkpoint { get; } = checkpoint;
    /// <summary>How the denoiser is driven: its noise levels, timesteps and prediction type (ModelSamplingDiscrete swaps it).</summary>
    public ModelSampling Sampling { get; init; } = sampling;
    /// <summary>The model family's latent format (SD 1.5's when there is no checkpoint, as in tests of the sampling maths).</summary>
    public LatentFormat LatentFormat => Checkpoint?.LatentFormat ?? LatentFormat.SD15;
    /// <summary>The guidance patches (RescaleCFG, CFGNorm, APG, ...) sampling applies, ComfyUI's model_options.</summary>
    public ModelPatches Patches { get; init; } = ModelPatches.None;

    /// <summary>A copy with other patches: patch nodes never change the model that came in, like ComfyUI's model.clone().</summary>
    public ModelHandle WithPatches(Func<ModelPatches, ModelPatches> change) => new(Checkpoint, Sampling) { Patches = change(Patches) };
}

/// <summary>The CLIP text encoder(s) and tokenizer of a loaded checkpoint, and the layer prompts are read from.</summary>
[Socket("CLIP")]
public sealed class ClipHandle(Checkpoint checkpoint)
{
    public Checkpoint Checkpoint { get; } = checkpoint;
    /// <summary>CLIP Set Last Layer's choice (-1 the last layer, -2 the penultimate, ...); null = the model family's default.</summary>
    public int? Layer { get; init; }
}

/// <summary>The VAE of a loaded checkpoint.</summary>
[Socket("VAE")]
public sealed class VaeHandle(Checkpoint checkpoint) { public Checkpoint Checkpoint { get; } = checkpoint; }
