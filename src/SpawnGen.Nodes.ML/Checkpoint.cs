using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.ML;

/// <summary>One LoRA a model or CLIP carries: its file on the model source and how strongly it is merged.</summary>
public sealed record LoraUse(string Repo, string File, float Strength);

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
    readonly Dictionary<string, Task<TextEncoding>> _emptyPrompt = new();
    readonly SemaphoreSlim _loraGate = new(1, 1);
    IReadOnlyList<LoraUse> _unetLoras = [], _clipLoras = [];
    string _appliedLoras = "";

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

    /// <summary>
    /// The empty prompt's encoding at a layer, computed once per layer and CLIP LoRA set: the baseline weighted prompts are measured against. Call it while the
    /// CLIP's LoRAs are merged (<see cref="UseLorasAsync"/>).
    /// </summary>
    public Task<TextEncoding> EmptyPromptAsync(int? layer = null, IReadOnlyList<LoraUse>? clipLoras = null)
    {
        int l = layer ?? DefaultClipLayer;
        string key = l + "|" + LoraKey(clipLoras ?? [], []);
        lock (_gate)
        {
            if (!_emptyPrompt.TryGetValue(key, out var task) || task.IsFaulted || task.IsCanceled)
                _emptyPrompt[key] = task = EncodeAsync(Tokenizer.EncodeCLIP("", TokenCount), layer: l);
            return task;
        }
    }

    /// <summary>
    /// Merges the LoRAs a model (<paramref name="unet"/>) or a CLIP (<paramref name="clip"/>) carries into the weights, keeping the other part's set (null = keep).
    /// The merged set is replaced only when it changes; the library restores the original weights first, so the handles that come into a node never change
    /// (ComfyUI's model.clone()). Nodes run one at a time, so a sampling run or an encode always sees the set it asked for.
    /// </summary>
    internal async Task UseLorasAsync(IReadOnlyList<LoraUse>? unet, IReadOnlyList<LoraUse>? clip, CancellationToken cancellationToken = default)
    {
        await _loraGate.WaitAsync(cancellationToken);
        try
        {
            var u = unet ?? _unetLoras;
            var c = clip ?? _clipLoras;
            string key = LoraKey(u, c);
            if (key != _appliedLoras)
            {
                _appliedLoras = "?"; // unknown until the merge finishes: a failed merge is redone next time
                await _models.SetLorasAsync(CombineLoras(u, c), cancellationToken);
                _appliedLoras = key;
            }
            _unetLoras = u;
            _clipLoras = c;
        }
        finally
        {
            _loraGate.Release();
        }
    }

    /// <summary>One entry per LoRA file in order of first use: its UNet strength (the model's uses summed) and its text-encoder strength (the CLIP's uses summed).</summary>
    internal static List<(string Repo, string File, float Strength, float ClipStrength)> CombineLoras(IReadOnlyList<LoraUse> unet, IReadOnlyList<LoraUse> clip)
    {
        var combined = new List<(string Repo, string File, float Strength, float ClipStrength)>();
        void Add(LoraUse l, bool isClip)
        {
            int i = combined.FindIndex(e => e.Repo == l.Repo && e.File == l.File);
            if (i < 0) { combined.Add((l.Repo, l.File, 0f, 0f)); i = combined.Count - 1; }
            var e = combined[i];
            combined[i] = isClip ? e with { ClipStrength = e.ClipStrength + l.Strength } : e with { Strength = e.Strength + l.Strength };
        }
        foreach (var l in unet) Add(l, false);
        foreach (var l in clip) Add(l, true);
        return combined;
    }

    static string LoraKey(IReadOnlyList<LoraUse> unet, IReadOnlyList<LoraUse> clip) =>
        string.Join(";", CombineLoras(unet, clip).Select(e => $"{e.Repo}/{e.File}:{e.Strength:R}:{e.ClipStrength:R}"));

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

    public void Dispose()
    {
        _models.Dispose();
        _loraGate.Dispose();
    }
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
    /// <summary>Merges exactly these LoRAs (an empty list removes every LoRA), replacing the set merged before.</summary>
    Task SetLorasAsync(IReadOnlyList<(string Repo, string File, float Strength, float ClipStrength)> loras, CancellationToken cancellationToken);
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
    /// <summary>The LoRAs merged into the UNet while this model samples (Load LoRA adds to a copy).</summary>
    public IReadOnlyList<LoraUse> Loras { get; init; } = [];

    /// <summary>A copy with other patches: patch nodes never change the model that came in, like ComfyUI's model.clone().</summary>
    public ModelHandle WithPatches(Func<ModelPatches, ModelPatches> change) => new(Checkpoint, Sampling) { Patches = change(Patches), Loras = Loras };
}

/// <summary>The CLIP text encoder(s) and tokenizer of a loaded checkpoint, and the layer prompts are read from.</summary>
[Socket("CLIP")]
public sealed class ClipHandle(Checkpoint checkpoint)
{
    public Checkpoint Checkpoint { get; } = checkpoint;
    /// <summary>CLIP Set Last Layer's choice (-1 the last layer, -2 the penultimate, ...); null = the model family's default.</summary>
    public int? Layer { get; init; }
    /// <summary>The LoRAs merged into the text encoder(s) while this CLIP encodes (Load LoRA adds to a copy).</summary>
    public IReadOnlyList<LoraUse> Loras { get; init; } = [];
}

/// <summary>The VAE of a loaded checkpoint.</summary>
[Socket("VAE")]
public sealed class VaeHandle(Checkpoint checkpoint) { public Checkpoint Checkpoint { get; } = checkpoint; }
