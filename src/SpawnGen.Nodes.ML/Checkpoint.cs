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
public sealed class Checkpoint : IDiffusionModel, ITextEncoderModel, IVaeModel, IDisposable
{
    public const int LatentChannels = 4;
    public const int TokenCount = 77;

    readonly ICheckpointModels _models;
    readonly object _gate = new();
    readonly EmptyPromptCache _emptyPrompt = new();
    readonly ComponentWeights? _unetWeights, _clipWeights;

    internal Checkpoint(string name, ICheckpointModels models, BPETokenizer tokenizer)
    {
        Name = name;
        _models = models;
        Tokenizer = tokenizer;
        Sigmas = ModelSigmas.LinearSchedule(); // ComfyUI's float64 table (the library's alphas are float32 products, about 1e-5 off)
        Sampling = new ModelSampling(Sigmas, models.Prediction);
        if (models.UNet is { } unet) _unetWeights = ComponentWeights.For(unet, models.Source);
        if (models.TextEncoder is { } te) _clipWeights = ComponentWeights.For(te, models.Source);
    }

    public string Name { get; }
    /// <summary>SD 1.x, SD 2.x or SDXL.</summary>
    public SdFamily Family => _models.Family;
    /// <summary>How the model's latents relate to its VAE's (scale, preview colours).</summary>
    public LatentFormat LatentFormat => SdFamilies.LatentFormat(Family);
    /// <summary>The width of a context row (768 SD 1.x, 1024 SD 2.x, 2048 SDXL).</summary>
    public int TextWidth => _models.TextWidth;
    /// <summary>The width of the first tower's part of a row (SDXL: CLIP-L's 768 of 2048); the whole row for one-tower models.</summary>
    public int FirstTowerWidth => SdFamilies.FirstTowerWidth(Family, TextWidth);
    public BPETokenizer Tokenizer { get; }
    /// <summary>The noise levels of the model's training steps, for schedulers and for telling the model its timestep.</summary>
    public ModelSigmas Sigmas { get; }
    /// <summary>The model's own sampling: eps or v prediction over <see cref="Sigmas"/>.</summary>
    public ModelSampling Sampling { get; }
    /// <summary>The layer a prompt is read from when CLIP Set Last Layer was not used (-1 the last, -2 the penultimate), ComfyUI's per-family default.</summary>
    public int DefaultClipLayer => SdFamilies.DefaultClipLayer(Family);

    /// <summary>
    /// Encodes token ids as the tokenizer makes them (<see cref="BPETokenizer.EncodeCLIP"/>: start, tokens, end, padded with the end token).
    /// Each tower gets ComfyUI's padding: CLIP-L pads with the end token, OpenCLIP (SD 2.x, SDXL's second tower) with 0. <paramref name="ids2"/> is SDXL's
    /// second tower's prompt when it differs (CLIP Text Encode SDXL's text_g). <paramref name="layer"/> null = <see cref="DefaultClipLayer"/>.
    /// </summary>
    public Task<TextEncoding> EncodeAsync(int[] ids, int[]? ids2 = null, int? layer = null, CancellationToken cancellationToken = default) =>
        _models.EncodeAsync(ids, ids2, layer, cancellationToken);

    /// <summary>
    /// The empty prompt's encoding at a layer, computed once per layer and CLIP weight steps: the baseline weighted prompts are measured against. Call it while the
    /// CLIP's steps are in (<see cref="ITextEncoderModel.UseWeightsAsync"/>).
    /// </summary>
    public Task<TextEncoding> EmptyPromptAsync(int? layer = null, IReadOnlyList<WeightStep>? clipWeights = null) =>
        _emptyPrompt.GetAsync(this, layer ?? DefaultClipLayer, clipWeights ?? []);

    Task<TextEncoding> ITextEncoderModel.EmptyPromptAsync(int? layer, IReadOnlyList<WeightStep> weights) => EmptyPromptAsync(layer, weights);

    /// <summary>The UNet as SpawnDev.ILGPU.ML's Stable Diffusion code loaded it (null for an ONNX export).</summary>
    public SdUNet? UNet => _models.UNet;
    /// <summary>The text encoder(s) as SpawnDev.ILGPU.ML's Stable Diffusion code loaded them (null for an ONNX export).</summary>
    public SdTextEncoder? TextEncoder => _models.TextEncoder;
    /// <summary>The VAE as SpawnDev.ILGPU.ML's Stable Diffusion code loaded it (null for an ONNX export).</summary>
    public SdVae? SdVae => _models.Vae;
    /// <summary>The single file the checkpoint was loaded from (null for an ONNX export): Save Checkpoint writes it with its current weights.</summary>
    internal SdCheckpoint? File => _models.File;

    // The UNet's and the text encoder's steps are kept apart, as ComfyUI keeps a MODEL's and a CLIP's patches apart.
    Task IDiffusionModel.UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) =>
        UseAsync(_unetWeights, weights, cancellationToken);
    Task ITextEncoderModel.UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) =>
        UseAsync(_clipWeights, weights, cancellationToken);

    static Task UseAsync(ComponentWeights? component, IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) =>
        component is not null ? component.UseAsync(weights, cancellationToken)
        : weights.Count == 0 ? Task.CompletedTask
        : throw new NotSupportedException("LoRAs and merges change single-file checkpoints; this is an ONNX export. Load the model as a .safetensors checkpoint.");

    /// <summary>Starts a sampling run on the UNet; dispose it when the run ends (it frees the run's GPU buffers).</summary>
    public UNetRun BeginRun(Accelerator accelerator) => _models.BeginRun(accelerator);

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

    internal static async Task<BPETokenizer> LoadClipTokenizerAsync(SpawnDev.ILGPU.ML.Hub.IModelSource source, string repoId, string folder, Action<string, int>? onProgress)
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

    public void Dispose()
    {
        _models.Dispose();
        _unetWeights?.Dispose();
        _clipWeights?.Dispose();
    }
}




/// <summary>One sampling run's UNet: each conditioning entry goes to the GPU once, every step uploads only the latent.</summary>
public abstract class UNetRun : IDisposable
{
    /// <summary>
    /// One prediction (eps or v, as the model was trained) for <paramref name="input"/> [4, h, w] in the model's space at <paramref name="timestep"/>, with the
    /// ControlNets active for this call (null or empty = none), the UNet options of the call (FreeU, PAG's perturbed pass) and, for an inpainting model,
    /// the extra input channels [<see cref="ConcatChannels"/>, h, w].
    /// </summary>
    public abstract Task<float[]> PredictAsync(float[] input, int h, int w, float timestep, CondEntry entry, IReadOnlyList<ControlCall>? controls,
        UNetOptions options, float[]? concat, CancellationToken cancellationToken);
    /// <summary>Extra input channels the UNet takes besides the latent (5 for an inpainting model: the mask and the masked picture's latent); 0 for most.</summary>
    public virtual int ConcatChannels => 0;
    /// <summary>The middle-attention mass of the last call made with <see cref="UNetOptions.RecordMiddleSelfAttention"/> (SAG), or null.</summary>
    public virtual Task<float[]?> ReadMiddleAttentionAsync() => Task.FromResult<float[]?>(null);
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
    /// <summary>The UNet and text encoder(s) as SpawnDev.ILGPU.ML's Stable Diffusion code loaded them, for LoRAs and merges (null for an ONNX export).</summary>
    SdUNet? UNet { get; }
    SdTextEncoder? TextEncoder { get; }
    SdVae? Vae { get; }
    /// <summary>The single file it was loaded from, for Save Checkpoint (null for an ONNX export).</summary>
    SdCheckpoint? File { get; }
    /// <summary>Where LoRA files are read from.</summary>
    SpawnDev.ILGPU.ML.Hub.IModelSource Source { get; }
}

/// <summary>The denoiser (UNet) of a loaded checkpoint, with the sampling math constants.</summary>
[Socket("MODEL")]
public sealed class ModelHandle(IDiffusionModel model, ModelSampling sampling)
{
    public ModelHandle(IDiffusionModel model) : this(model, model?.Sampling!) { }

    /// <summary>The diffusion model (a checkpoint's UNet or one loaded alone).</summary>
    public IDiffusionModel Model { get; } = model;
    /// <summary>How the denoiser is driven: its noise levels, timesteps and prediction type (ModelSamplingDiscrete swaps it).</summary>
    public ModelSampling Sampling { get; init; } = sampling;
    /// <summary>The model family's latent format (SD 1.5's when there is no checkpoint, as in tests of the sampling maths).</summary>
    public LatentFormat LatentFormat => Model?.LatentFormat ?? LatentFormat.SD15;
    /// <summary>The guidance patches (RescaleCFG, CFGNorm, APG, ...) sampling applies, ComfyUI's model_options.</summary>
    public ModelPatches Patches { get; init; } = ModelPatches.None;
    /// <summary>The LoRAs and merges put into the UNet while this model samples, in order (Load LoRA and the merge nodes add to a copy).</summary>
    public IReadOnlyList<WeightStep> Weights { get; init; } = [];

    /// <summary>A copy with other patches: patch nodes never change the model that came in, like ComfyUI's model.clone().</summary>
    public ModelHandle WithPatches(Func<ModelPatches, ModelPatches> change) => new(Model, Sampling) { Patches = change(Patches), Weights = Weights };
}

/// <summary>The CLIP text encoder(s) and tokenizer of a loaded checkpoint, and the layer prompts are read from.</summary>
[Socket("CLIP")]
public sealed class ClipHandle(ITextEncoderModel encoder)
{
    /// <summary>The text encoder(s) (a checkpoint's or loaded alone).</summary>
    public ITextEncoderModel Encoder { get; } = encoder;
    /// <summary>CLIP Set Last Layer's choice (-1 the last layer, -2 the penultimate, ...); null = the model family's default.</summary>
    public int? Layer { get; init; }
    /// <summary>The LoRAs and merges put into the text encoder(s) while this CLIP encodes, in order (Load LoRA and the merge nodes add to a copy).</summary>
    public IReadOnlyList<WeightStep> Weights { get; init; } = [];
}

/// <summary>The VAE of a loaded checkpoint.</summary>
[Socket("VAE")]
public sealed class VaeHandle(IVaeModel vae) { public IVaeModel Vae { get; } = vae; }
