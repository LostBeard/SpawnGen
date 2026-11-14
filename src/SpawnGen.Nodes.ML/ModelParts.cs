using ILGPU.Runtime;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md): SpawnGen's own model plumbing over SpawnDev.ILGPU.ML (SdUNet, SdTextEncoder, SdVae, SdTaesd, as their docs
// describe them). The default CLIP layers are the models' published ones (SD 1.x the last layer; SD 2.x and SDXL the penultimate); the handle
// semantics are LoraNodeTests'.

/// <summary>A prompt's text conditioning: the context rows ([1, rows, width]) and, for SDXL, the pooled embedding.</summary>
public sealed record TextEncoding(float[] Context, int[] Shape, float[]? Pooled);

/// <summary>One sampling run on a model's UNet: predictions for a conditioning entry at a timestep. Holds per-run GPU buffers until disposed.</summary>
public abstract class UNetRun : IDisposable
{
    /// <summary>A video model's latent frames: the inputs' rows are that many frames stacked (set by the sampler before the first call; 1 for pictures).</summary>
    public int Frames { get; set; } = 1;

    /// <summary>Image to video: the first latent frame is the start picture, kept (a video model runs its tokens at timestep 0). Set per batch item.</summary>
    public bool FirstFrameClean { get; set; }

    /// <summary>The extra input channels an inpainting UNet takes besides the latent (0 for an ordinary one).</summary>
    public virtual int ConcatChannels => 0;

    /// <summary>The model's output for <paramref name="input"/> (channels x h x w, already scaled for the model) at <paramref name="timestep"/>.</summary>
    public abstract Task<float[]> PredictAsync(float[] input, int h, int w, float timestep, CondEntry entry, IReadOnlyList<ControlCall>? controls,
        UNetOptions options, float[]? concat, CancellationToken cancellationToken);

    /// <summary>The middle self-attention mass of the last prediction made while recording it (Self-Attention Guidance), or null when not supported.</summary>
    public virtual Task<float[]?> ReadMiddleAttentionAsync() => Task.FromResult<float[]?>(null);

    public abstract void Dispose();
}

/// <summary>What a loaded checkpoint provides: its family, prediction and parts.</summary>
internal interface ICheckpointModels : IDisposable
{
    SdFamily Family { get; }
    PredictionType Prediction { get; }
    int TextWidth { get; }
    SdUNet? UNet { get; }
    SdTextEncoder? TextEncoder { get; }
    SdVae? Vae { get; }
    SdCheckpoint? File { get; }
    IModelSource Source { get; }
    Task<TextEncoding> EncodeAsync(int[] ids, int[]? ids2, int? layer, CancellationToken cancellationToken);
    UNetRun BeginRun(Accelerator accelerator);
    Task<float[]> DecodeAsync(float[] latent, int h, int w, CancellationToken cancellationToken);
    Task<float[]> EncodeImageAsync(float[] image, int H, int W, CancellationToken cancellationToken);
}

/// <summary>A diffusion model (MODEL): a checkpoint's UNet or one loaded alone.</summary>
public interface IDiffusionModel
{
    string Name { get; }
    SdFamily Family { get; }
    LatentFormat LatentFormat { get; }
    ModelSampling Sampling { get; }
    SdUNet? UNet { get; }

    /// <summary>Brings the UNet's weights to these steps (LoRAs, merges) before a run.</summary>
    Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken);

    UNetRun BeginRun(Accelerator accelerator);
}

/// <summary>A text encoder (CLIP): a checkpoint's or one loaded alone.</summary>
public interface ITextEncoderModel
{
    string Name { get; }
    SdFamily Family { get; }
    BPETokenizer Tokenizer { get; }

    /// <summary>The width of the first tower's part of the context (768 for SDXL's CLIP-L; the whole width otherwise).</summary>
    int FirstTowerWidth { get; }

    SdTextEncoder? TextEncoder { get; }

    /// <summary>One 77-token chunk's context (and pooled output); <paramref name="layer"/> as CLIP Set Last Layer counts (-1 the last), null the family's default.</summary>
    Task<TextEncoding> EncodeAsync(int[] ids, int[]? ids2 = null, int? layer = null, CancellationToken cancellationToken = default);

    /// <summary>The empty prompt's encoding at a layer and with these weights (what prompt weights are measured from).</summary>
    Task<TextEncoding> EmptyPromptAsync(int? layer, IReadOnlyList<WeightStep> weights);

    Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken);
}

/// <summary>A VAE: latents (UNSCALED, the VAE's own units) to pictures in -1..1 and back.</summary>
public interface IVaeModel
{
    string Name { get; }
    SdVae? SdVae { get; }
    Task<float[]> DecodeAsync(float[] latent, int h, int w, CancellationToken cancellationToken = default);
    Task<float[]> EncodeImageAsync(float[] image, int H, int W, CancellationToken cancellationToken = default);
    /// <summary>Pixels per latent cell along each side: 8 for the Stable Diffusion family's VAEs, 16 for FLUX.2's (a 2 x 2 patch of its 8x VAE).</summary>
    int Downscale => 8;
}

/// <summary>
/// MODEL: a diffusion model with its sampling, the guidance patches put on it, and the weight steps (LoRAs, merges) it runs with. Nodes never change
/// a handle; they make a new one.
/// </summary>
[Socket("MODEL")]
public sealed class ModelHandle(IDiffusionModel model, ModelSampling sampling)
{
    public ModelHandle(IDiffusionModel model) : this(model, model.Sampling) { }

    public IDiffusionModel Model { get; } = model;
    public ModelSampling Sampling { get; init; } = sampling;

    /// <summary>The model's latent format (SD 1.x's when there is no model, as in tests).</summary>
    public LatentFormat LatentFormat => Model?.LatentFormat ?? LatentFormat.SD15;

    public ModelPatches Patches { get; init; } = ModelPatches.None;
    public IReadOnlyList<WeightStep> Weights { get; init; } = [];

    /// <summary>A copy with the patches changed; sampling and weights are kept.</summary>
    public ModelHandle WithPatches(Func<ModelPatches, ModelPatches> change) => new(Model, Sampling) { Patches = change(Patches), Weights = Weights };
}

/// <summary>CLIP: a text encoder, the layer CLIP Set Last Layer chose (null: the family's default) and the weight steps it encodes with.</summary>
[Socket("CLIP")]
public sealed class ClipHandle(ITextEncoderModel encoder)
{
    public ITextEncoderModel Encoder { get; } = encoder;
    public int? Layer { get; init; }
    public IReadOnlyList<WeightStep> Weights { get; init; } = [];
}

/// <summary>VAE.</summary>
[Socket("VAE")]
public sealed class VaeHandle(IVaeModel vae)
{
    public IVaeModel Vae { get; } = vae;
}

/// <summary>What differs between the Stable Diffusion families.</summary>
internal static class SdFamilies
{
    public static LatentFormat LatentFormat(SdFamily family) => family == SdFamily.SDXL ? ML.LatentFormat.SDXL : ML.LatentFormat.SD15;

    /// <summary>The CLIP layer the family's models were trained with: the last for SD 1.x, the penultimate for SD 2.x and SDXL.</summary>
    public static int DefaultClipLayer(SdFamily family) => family == SdFamily.SD1 ? -1 : -2;

    /// <summary>The first tower's width: SDXL's context is CLIP-L (768) then OpenCLIP-bigG side by side.</summary>
    public static int FirstTowerWidth(SdFamily family, int width) => family == SdFamily.SDXL ? 768 : width;

    public static async Task<TextEncoding> EncodeAsync(SdTextEncoder encoder, SdFamily family, int[] ids, int[]? ids2, int? layer, CancellationToken cancellationToken)
    {
        var e = await encoder.EncodeAsync(ids, ids2, layer ?? DefaultClipLayer(family), cancellationToken);
        return new TextEncoding(e.Context, [1, e.Rows, e.Width], e.Pooled);
    }

    public static SdFamily FromTextWidth(int width, bool xl) => xl || width == 2048 ? SdFamily.SDXL : width == 1024 ? SdFamily.SD2 : SdFamily.SD1;

    public static SdFamily FromUNet(SdUNet unet) => unet.Family;

    public static PredictionType Prediction(SdPrediction prediction) => prediction == SdPrediction.V ? PredictionType.VPrediction : PredictionType.Eps;
}

/// <summary>The empty prompt's encoding per encoder, layer and weights, made once.</summary>
internal sealed class EmptyPromptCache
{
    readonly Dictionary<(int Layer, string Weights), Task<TextEncoding>> _known = [];
    readonly Lock _gate = new();

    public Task<TextEncoding> GetAsync(ITextEncoderModel encoder, int layer, IReadOnlyList<WeightStep> weights)
    {
        var key = (layer, WeightStep.KeyOf(weights));
        lock (_gate)
        {
            if (!_known.TryGetValue(key, out var task) || task.IsFaulted || task.IsCanceled) _known[key] = task = MakeAsync();
            return task;
        }

        async Task<TextEncoding> MakeAsync()
        {
            await encoder.UseWeightsAsync(weights, CancellationToken.None);
            var ids = ClipPadding.Pad(PromptWeights.Chunks("", _ => [], Checkpoint.TokenCount).Single().Ids, ClipPadding.ForTextWidth(encoder.FirstTowerWidth));
            // SDXL's second tower pads with 0 instead of the end token.
            var ids2 = encoder.Family == SdFamily.SDXL ? PromptWeights.Chunks("", _ => [], Checkpoint.TokenCount, padToken: 0).Single().Ids : null;
            return await encoder.EncodeAsync(ids, ids2, layer);
        }
    }
}

/// <summary>A UNet loaded on its own (Load Diffusion Model).</summary>
internal sealed class StandaloneDiffusionModel : IDiffusionModel, IDisposable
{
    readonly ComponentWeights _weights;

    public StandaloneDiffusionModel(string name, SdUNet unet, PredictionType prediction, IModelSource source)
    {
        Name = name;
        UNet = unet;
        Family = unet.Family;
        LatentFormat = SdFamilies.LatentFormat(Family);
        Sampling = new ModelSampling(ModelSigmas.LinearSchedule(), prediction);
        _weights = ComponentWeights.For(unet, source);
    }

    public string Name { get; }
    public SdFamily Family { get; }
    public LatentFormat LatentFormat { get; }
    public ModelSampling Sampling { get; }
    public SdUNet UNet { get; }

    public Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) => _weights.UseAsync(weights, cancellationToken);

    public UNetRun BeginRun(Accelerator accelerator) => new SdUNetRun(UNet, accelerator);

    /// <summary>Loads a UNet-only file and tells its prediction type from its tensors.</summary>
    public static async Task<StandaloneDiffusionModel> LoadAsync(MlRuntime runtime, string name, string repo, string file)
    {
        var unet = await SdUNet.LoadAsync(runtime.Accelerator, runtime.ModelSource, repo, file);
        try { return new StandaloneDiffusionModel(name, unet, SdFamilies.Prediction(await unet.InferPredictionAsync()), runtime.ModelSource); }
        catch { unet.Dispose(); throw; }
    }

    public void Dispose()
    {
        _weights.Dispose();
        UNet.Dispose();
    }
}

/// <summary>A text encoder loaded on its own (Load CLIP, Dual CLIP Loader).</summary>
internal sealed class StandaloneTextEncoder : ITextEncoderModel, IDisposable
{
    readonly ComponentWeights _weights;
    readonly EmptyPromptCache _empty = new();

    public StandaloneTextEncoder(string name, SdTextEncoder encoder, BPETokenizer tokenizer, IModelSource source)
    {
        Name = name;
        TextEncoder = encoder;
        Tokenizer = tokenizer;
        Family = encoder.Family;
        FirstTowerWidth = SdFamilies.FirstTowerWidth(Family, encoder.Width);
        _weights = ComponentWeights.For(encoder, source);
    }

    public string Name { get; }
    public SdFamily Family { get; }
    public BPETokenizer Tokenizer { get; }
    public int FirstTowerWidth { get; }
    public SdTextEncoder TextEncoder { get; }

    public Task<TextEncoding> EncodeAsync(int[] ids, int[]? ids2 = null, int? layer = null, CancellationToken cancellationToken = default) =>
        SdFamilies.EncodeAsync(TextEncoder, Family, ids, ids2, layer, cancellationToken);

    public Task<TextEncoding> EmptyPromptAsync(int? layer, IReadOnlyList<WeightStep> weights) =>
        _empty.GetAsync(this, layer ?? SdFamilies.DefaultClipLayer(Family), weights ?? []);

    public Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) => _weights.UseAsync(weights, cancellationToken);

    public void Dispose()
    {
        _weights.Dispose();
        TextEncoder.Dispose();
    }
}

/// <summary>A VAE loaded on its own (Load VAE).</summary>
internal sealed class StandaloneVae(string name, SdVae vae) : IVaeModel, IDisposable
{
    public string Name { get; } = name;
    public SdVae SdVae { get; } = vae;

    public Task<float[]> DecodeAsync(float[] latent, int h, int w, CancellationToken cancellationToken = default) => SdVae.DecodeAsync(latent, h, w, cancellationToken);

    public Task<float[]> EncodeImageAsync(float[] image, int H, int W, CancellationToken cancellationToken = default) => SdVae.EncodeAsync(image, H, W, cancellationToken);

    public void Dispose() => SdVae.Dispose();
}

/// <summary>The tiny VAE (TAESD / TAESDXL) loaded as a VAE (Load VAE's "taesd" entries).</summary>
internal sealed class StandaloneTaesd(string name, SdTaesd tae) : IVaeModel, IDisposable
{
    public string Name { get; } = name;

    /// <summary>None: the tiny VAE is not an SdVae (Save VAE cannot write it).</summary>
    public SdVae? SdVae => null;

    public Task<float[]> DecodeAsync(float[] latent, int h, int w, CancellationToken cancellationToken = default) => tae.DecodeAsync(latent, h, w, cancellationToken);

    public Task<float[]> EncodeImageAsync(float[] image, int H, int W, CancellationToken cancellationToken = default) =>
        tae.HasEncoder ? tae.EncodeAsync(image, H, W, cancellationToken) : throw new NotSupportedException($"{Name} has no encoder (only the decoder was loaded).");

    public void Dispose() => tae.Dispose();
}
