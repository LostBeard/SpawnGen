using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.StableDiffusion;

namespace SpawnGen.Nodes.ML;

/// <summary>The diffusion model (UNet) a MODEL carries: a checkpoint's, or one loaded alone (Load Diffusion Model).</summary>
public interface IDiffusionModel
{
    string Name { get; }
    SdFamily Family { get; }
    /// <summary>How the model's latents relate to its VAE's (scale, preview colours).</summary>
    LatentFormat LatentFormat { get; }
    /// <summary>The model's own sampling: eps or v prediction over SD's noise levels.</summary>
    ModelSampling Sampling { get; }
    /// <summary>Merges exactly these LoRAs into the UNet (an empty list removes them).</summary>
    Task UseLorasAsync(IReadOnlyList<LoraUse> loras, CancellationToken cancellationToken);
    /// <summary>Starts a sampling run; dispose it when the run ends (it frees the run's GPU buffers).</summary>
    UNetRun BeginRun(Accelerator accelerator);
}

/// <summary>The text encoder(s) a CLIP carries, with the tokenizer: a checkpoint's, or loaded alone (Load CLIP, DualCLIPLoader).</summary>
public interface ITextEncoderModel
{
    string Name { get; }
    SdFamily Family { get; }
    BPETokenizer Tokenizer { get; }
    /// <summary>The width of the first tower's part of a context row (SDXL: CLIP-L's 768 of 2048); the whole row for one-tower models.</summary>
    int FirstTowerWidth { get; }
    /// <summary>
    /// Encodes token ids as the tokenizer makes them (start, tokens, end, padded with the end token); each tower gets ComfyUI's padding. <paramref name="ids2"/> is
    /// SDXL's second tower's prompt when it differs; <paramref name="layer"/> null = the family's default layer.
    /// </summary>
    Task<TextEncoding> EncodeAsync(int[] ids, int[]? ids2 = null, int? layer = null, CancellationToken cancellationToken = default);
    /// <summary>The empty prompt's encoding at a layer, computed once per layer and LoRA set (call it while those LoRAs are merged).</summary>
    Task<TextEncoding> EmptyPromptAsync(int? layer, IReadOnlyList<LoraUse> loras);
    /// <summary>Merges exactly these LoRAs into the text encoder(s) (an empty list removes them).</summary>
    Task UseLorasAsync(IReadOnlyList<LoraUse> loras, CancellationToken cancellationToken);
}

/// <summary>The VAE a VAE socket carries: a checkpoint's, or one loaded alone (Load VAE).</summary>
public interface IVaeModel
{
    string Name { get; }
    /// <summary>Decodes one latent in the VAE's values, [4, h, w], to a picture [3, 8h, 8w] in [-1, 1] (not clamped).</summary>
    Task<float[]> DecodeAsync(float[] latent, int h, int w, CancellationToken cancellationToken = default);
    /// <summary>Encodes a picture [3, H, W] in [-1, 1] (H and W multiples of 8) to the VAE's latent [4, H/8, W/8] (the posterior mean).</summary>
    Task<float[]> EncodeImageAsync(float[] image, int H, int W, CancellationToken cancellationToken = default);
}

/// <summary>Per-family facts shared by checkpoints and components loaded alone.</summary>
internal static class SdFamilies
{
    public static LatentFormat LatentFormat(SdFamily family) => family == SdFamily.SDXL ? ML.LatentFormat.SDXL : ML.LatentFormat.SD15;
    public static int DefaultClipLayer(SdFamily family) => family == SdFamily.SD1 ? -1 : -2;
    public static int FirstTowerWidth(SdFamily family, int width) => family == SdFamily.SDXL ? 768 : width;

    /// <summary>ComfyUI's per-tower padding: CLIP-L (SD 1.x, SDXL's first) with the end token, OpenCLIP (SD 2.x, SDXL's second) with 0.</summary>
    public static async Task<TextEncoding> EncodeAsync(SdTextEncoder encoder, SdFamily family, int[] ids, int[]? ids2, int? layer, CancellationToken cancellationToken)
    {
        var first = ClipPadding.Pad(ids, family == SdFamily.SD2 ? 0 : ClipPadding.End);
        var second = family == SdFamily.SDXL ? ClipPadding.Pad(ids2 ?? ids, 0) : null;
        var te = await encoder.EncodeAsync(first, second, layer, cancellationToken);
        return new TextEncoding(te.Context, [1, te.Rows, te.Width], te.Pooled);
    }

    /// <summary>The family a text encoder's width says (768 CLIP-L = SD 1.x, 1024 OpenCLIP-H = SD 2.x, 2048 = SDXL's two towers).</summary>
    public static SdFamily FromTextWidth(int width, bool xl) => xl ? SdFamily.SDXL : width == 1024 ? SdFamily.SD2 : SdFamily.SD1;

    /// <summary>The family a UNet's cross-attention width says.</summary>
    public static SdFamily FromUNet(SdUNet unet) => unet.IsXL ? SdFamily.SDXL : unet.ContextDim == 1024 ? SdFamily.SD2 : SdFamily.SD1;
}

/// <summary>Empty-prompt encodings cached per layer and LoRA set.</summary>
internal sealed class EmptyPromptCache
{
    readonly Dictionary<string, Task<TextEncoding>> _cache = new();

    public Task<TextEncoding> GetAsync(ITextEncoderModel encoder, int layer, IReadOnlyList<LoraUse> loras)
    {
        string key = layer + "|" + string.Join(";", loras.Select(l => $"{l.Repo}/{l.File}:{l.Strength:R}"));
        lock (_cache)
        {
            if (!_cache.TryGetValue(key, out var task) || task.IsFaulted || task.IsCanceled)
                _cache[key] = task = encoder.EncodeAsync(encoder.Tokenizer.EncodeCLIP("", Checkpoint.TokenCount), layer: layer);
            return task;
        }
    }
}

/// <summary>
/// The LoRA set merged into one component loaded alone: merged again only when it changes (the library restores the original weights first). Nodes run one at a
/// time, so a sampling run or an encode always sees the set it asked for.
/// </summary>
internal sealed class ComponentLoras(SpawnDev.ILGPU.ML.Hub.IModelSource source, Func<IReadOnlyList<SdLoraSpec>, CancellationToken, Task<SdLoraReport>> apply, bool clip)
    : IDisposable
{
    readonly SemaphoreSlim _gate = new(1, 1);
    string _applied = "";

    public async Task UseAsync(IReadOnlyList<LoraUse> loras, CancellationToken cancellationToken)
    {
        await _gate.WaitAsync(cancellationToken);
        try
        {
            string key = string.Join(";", loras.Select(l => $"{l.Repo}/{l.File}:{l.Strength:R}"));
            if (key == _applied) return;
            _applied = "?"; // unknown until the merge finishes: a failed merge is redone next time
            // One entry per file, its uses summed; this component reads Strength (a UNet) or ClipStrength (a text encoder).
            var specs = loras.GroupBy(l => (l.Repo, l.File)).Select(g =>
            {
                float s = g.Sum(l => l.Strength);
                return SdLoraSpec.From(source, g.Key.Repo, g.Key.File, clip ? 0f : s, clip ? s : 0f);
            }).ToList();
            await apply(specs, cancellationToken);
            _applied = key;
        }
        finally
        {
            _gate.Release();
        }
    }

    public void Dispose() => _gate.Dispose();
}

/// <summary>A UNet loaded alone (Load Diffusion Model): LDM names with or without <c>model.diffusion_model.</c>, or diffusers names.</summary>
internal sealed class StandaloneDiffusionModel(string name, SdUNet unet, PredictionType prediction, SpawnDev.ILGPU.ML.Hub.IModelSource source)
    : IDiffusionModel, IDisposable
{
    readonly ComponentLoras _loras = new(source, unet.ApplyLorasAsync, clip: false);

    public string Name { get; } = name;
    public SdFamily Family { get; } = SdFamilies.FromUNet(unet);
    public LatentFormat LatentFormat => SdFamilies.LatentFormat(Family);
    public ModelSampling Sampling { get; } = new(ModelSigmas.LinearSchedule(), prediction);

    public Task UseLorasAsync(IReadOnlyList<LoraUse> loras, CancellationToken cancellationToken) => _loras.UseAsync(loras, cancellationToken);

    public UNetRun BeginRun(Accelerator accelerator) => new SdUNetRun(unet, accelerator);

    /// <summary>Loads a UNet-only file; its prediction type (eps or v) is read from the weights by ComfyUI's rules.</summary>
    public static async Task<StandaloneDiffusionModel> LoadAsync(MlRuntime runtime, string name, string repo, string file)
    {
        var unet = await SdUNet.LoadAsync(runtime.Accelerator, runtime.ModelSource, repo, file);
        try
        {
            var prediction = await unet.InferPredictionAsync() == SdPrediction.V ? PredictionType.VPrediction : PredictionType.Eps;
            return new StandaloneDiffusionModel(name, unet, prediction, runtime.ModelSource);
        }
        catch { unet.Dispose(); throw; }
    }

    public void Dispose()
    {
        _loras.Dispose();
        unet.Dispose();
    }
}

/// <summary>Text encoder(s) loaded alone (Load CLIP: one tower; DualCLIPLoader: SDXL's two), with the CLIP tokenizer.</summary>
internal sealed class StandaloneTextEncoder(string name, SdTextEncoder encoder, BPETokenizer tokenizer, SpawnDev.ILGPU.ML.Hub.IModelSource source)
    : ITextEncoderModel, IDisposable
{
    readonly EmptyPromptCache _empty = new();
    readonly ComponentLoras _loras = new(source, encoder.ApplyLorasAsync, clip: true);

    public string Name { get; } = name;
    public SdFamily Family { get; } = SdFamilies.FromTextWidth(encoder.Width, encoder.IsXL);
    public BPETokenizer Tokenizer { get; } = tokenizer;
    public int FirstTowerWidth => SdFamilies.FirstTowerWidth(Family, encoder.Width);

    public Task<TextEncoding> EncodeAsync(int[] ids, int[]? ids2 = null, int? layer = null, CancellationToken cancellationToken = default) =>
        SdFamilies.EncodeAsync(encoder, Family, ids, ids2, layer ?? SdFamilies.DefaultClipLayer(Family), cancellationToken);

    public Task<TextEncoding> EmptyPromptAsync(int? layer, IReadOnlyList<LoraUse> loras) =>
        _empty.GetAsync(this, layer ?? SdFamilies.DefaultClipLayer(Family), loras);

    public Task UseLorasAsync(IReadOnlyList<LoraUse> loras, CancellationToken cancellationToken) => _loras.UseAsync(loras, cancellationToken);

    public void Dispose()
    {
        _loras.Dispose();
        encoder.Dispose();
    }
}

/// <summary>A VAE loaded alone (Load VAE): LDM names with or without <c>first_stage_model.</c>, or diffusers names.</summary>
internal sealed class StandaloneVae(string name, SdVae vae) : IVaeModel, IDisposable
{
    public string Name { get; } = name;
    public Task<float[]> DecodeAsync(float[] latent, int h, int w, CancellationToken cancellationToken = default) => vae.DecodeAsync(latent, h, w, cancellationToken);
    public Task<float[]> EncodeImageAsync(float[] image, int H, int W, CancellationToken cancellationToken = default) => vae.EncodeAsync(image, H, W, cancellationToken);
    public void Dispose() => vae.Dispose();
}
