using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.ILGPU.ML.Wan;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

// SpawnGen's own plumbing (docs/CLEANROOM.md) over SpawnDev.ILGPU.ML's Wan 2.2 building blocks as their docs describe them (Wan22Pipeline's
// LoadWeightsAsync / LoadTokenizerAsync / TokenizePrompt, WanDiT, UmT5Encoder, WanVae and its latent normalisation), so ComfyUI's split Wan graph
// (Load Diffusion Model, Load CLIP type "wan", Load VAE, Wan22ImageToVideoLatent, ModelSamplingSD3, KSampler, VAE Decode) runs on them.

/// <summary>The Wan 2.2 TI2V 5B files the library's model host serves (downloaded on first use), as the loaders list them.</summary>
public static class WanFiles
{
    public static string Repo => Wan22Pipeline.DefaultRepo;
    static readonly Wan22Paths Paths = Wan22Paths.InRepo();

    /// <summary>The 8-bit transformer and the distilled (Turbo: 4 steps, cfg 1) one.</summary>
    public static string[] DiffusionModels { get; } = [Paths.Dit, Wan22Paths.DitFile("turbo_q8")];
    public static string TextEncoder => Paths.TextEncoder;
    public static string Tokenizer => Paths.Tokenizer;
    public static string Vae => Paths.Vae;

    /// <summary>Whether tensor names are a Wan transformer in the original layout (ComfyUI's repackaged files: q, k and v apart), which the library does not load yet.</summary>
    public static bool IsOriginalLayoutDit(IReadOnlyCollection<string> names) =>
        names.Contains("patch_embedding.weight") && names.Contains("blocks.0.cross_attn.k.weight");
}

/// <summary>
/// The big Wan 2.2 parts take turns on the device, as the library's own pipeline does (the 5.6 GB transformer and the 1.4 GB VAE do not fit an 8 GB
/// card together with a 720p run's activations): loading one unloads the other; it loads again when next needed.
/// </summary>
internal static class WanResidency
{
    static IWanResident? _current;

    public static void Use(IWanResident part)
    {
        if (_current is { } other && !ReferenceEquals(other, part)) other.Unload();
        _current = part;
    }

    public static void Forget(IWanResident part)
    {
        if (ReferenceEquals(_current, part)) _current = null;
    }
}

internal interface IWanResident
{
    void Unload();
}

/// <summary>Wan 2.2's transformer as a MODEL (Load Diffusion Model): flow matching, the Wan 2.2 latent format, loaded when a sampler first needs it.</summary>
internal sealed class WanDiffusionModel(MlRuntime runtime, string file) : IDiffusionModel, IDisposable, IWanResident
{
    GpuWeights? _weights;
    DiTKernels? _kernels;
    WanDiT? _dit;

    public string Name => file;

    /// <summary>Not a Stable Diffusion family; only checkpoint saving reads it, and that refuses a Wan model.</summary>
    public SdFamily Family => SdFamily.SD1;
    public LatentFormat LatentFormat => LatentFormat.Wan22;

    /// <summary>Flow matching with Wan 2.2 TI2V's published shift (5); ComfyUI's template sets its own with ModelSamplingSD3 (8).</summary>
    public ModelSampling Sampling { get; } = new(ModelSigmas.Flow(ModelSigmas.Sd3Shift(5.0), 1000), PredictionType.Const);
    public SdUNet? UNet => null;

    public Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) =>
        weights.Count == 0 ? Task.CompletedTask : throw new NotSupportedException("LoRAs and model merges on Wan 2.2 are not available yet.");

    public UNetRun BeginRun(Accelerator accelerator) => new WanRun(this, accelerator);

    public async Task<WanDiT> DitAsync(CancellationToken ct)
    {
        WanResidency.Use(this);
        if (_dit is not null) return _dit;
        _weights = await Wan22Pipeline.LoadWeightsAsync(runtime.Accelerator, runtime.ModelSource, WanFiles.Repo, file, ct);
        _kernels = new DiTKernels(runtime.Accelerator);
        return _dit = new WanDiT(runtime.Accelerator, _kernels, _weights);
    }

    public void Unload()
    {
        _dit?.Dispose();
        _weights?.Dispose();
        (_dit, _kernels, _weights) = (null, null, null);
    }

    public void Dispose()
    {
        WanResidency.Forget(this);
        Unload();
    }

    /// <summary>
    /// One sampling run: the latent's frames come stacked as rows (the sampler sets <see cref="UNetRun.Frames"/>); the prompt and the negative take
    /// the transformer's two prompt slots; a start picture's frame runs at timestep 0.
    /// </summary>
    sealed class WanRun(WanDiffusionModel model, Accelerator acc) : UNetRun
    {
        MemoryBuffer1D<float, Stride1D.Dense>? _in, _out;
        (int F, int H, int W) _grid;
        readonly CondEntry?[] _slots = new CondEntry?[2];
        int _nextSlot;

        public override async Task<float[]> PredictAsync(float[] input, int h, int w, float timestep, CondEntry entry, IReadOnlyList<ControlCall>? controls,
            UNetOptions options, float[]? concat, CancellationToken cancellationToken)
        {
            if (controls is { Count: > 0 }) throw new NotSupportedException("ControlNets do not apply to Wan 2.2.");
            if (entry.Shape[^1] != UmT5Encoder.Dim)
                throw new ArgumentException($"Wan 2.2 takes umT5 text ({UmT5Encoder.Dim} wide); this conditioning is {entry.Shape[^1]} wide. Encode the prompt with Load CLIP type \"wan\".");
            int frames = Frames, rows = h / Math.Max(frames, 1);
            if (frames < 1 || rows * frames != h) throw new InvalidOperationException($"The latent's {h} rows are not {frames} frames.");
            var dit = await model.DitAsync(cancellationToken);
            if (_grid != (frames, rows, w))
            {
                dit.SetGrid(frames, rows, w);
                _grid = (frames, rows, w);
            }
            int slot = Array.FindIndex(_slots, e => ReferenceEquals(e, entry));
            if (slot < 0)
            {
                slot = _nextSlot;
                _nextSlot = 1 - _nextSlot;
                _slots[slot] = entry;
                dit.SetContext(slot, entry.Embeddings, entry.Shape[^2]);
            }
            int n = input.Length;
            if (_in is null || _in.Length < n) { _in?.Dispose(); _in = acc.Allocate1D<float>(n); }
            if (_out is null || _out.Length < n) { _out?.Dispose(); _out = acc.Allocate1D<float>(n); }
            var inView = _in.View.SubView(0, n);
            inView.CopyFromCPU(input);
            dit.Forward(inView, timestep, FirstFrameClean, slot, _out.View.SubView(0, n));
            await acc.SynchronizeAsync();
            return await _out.CopyToHostAsync(0, n);
        }

        public override void Dispose()
        {
            _in?.Dispose();
            _out?.Dispose();
        }
    }
}

/// <summary>A text encoder that takes the prompt as text (its own tokenizer, no 77-token chunks): umT5 for Wan.</summary>
public interface IPromptTextEncoder
{
    Task<TextEncoding> EncodePromptAsync(string text, CancellationToken cancellationToken);
}

/// <summary>
/// Wan 2.2's umT5-XXL as a CLIP (Load CLIP type "wan"): the prompt tokenized as the library's pipeline does (whitespace cleaned, end token,
/// at most 512 tokens, no padding), encoded to [1, tokens, 4096]. The encoder streams its layers, so it holds little device memory.
/// </summary>
internal sealed class WanTextEncoderModel(MlRuntime runtime, string file) : ITextEncoderModel, IPromptTextEncoder, IDisposable
{
    UnigramTokenizer? _tokenizer;
    UmT5Encoder? _encoder;

    public string Name => file;
    public SdFamily Family => SdFamily.SD1;
    public BPETokenizer Tokenizer => throw new NotSupportedException("umT5 has no CLIP tokenizer.");
    public int FirstTowerWidth => UmT5Encoder.Dim;
    public SdTextEncoder? TextEncoder => null;

    public async Task<TextEncoding> EncodePromptAsync(string text, CancellationToken cancellationToken)
    {
        _tokenizer ??= await Wan22Pipeline.LoadTokenizerAsync(runtime.ModelSource, WanFiles.Repo, WanFiles.Tokenizer, cancellationToken);
        _encoder ??= await UmT5Encoder.CreateAsync(runtime.Accelerator, runtime.ModelSource, WanFiles.Repo, file, cancellationToken);
        var ids = Wan22Pipeline.TokenizePrompt(_tokenizer, text);
        var context = await _encoder.EncodeAsync(ids, cancellationToken);
        return new TextEncoding(context, [1, context.Length / UmT5Encoder.Dim, UmT5Encoder.Dim], null);
    }

    public Task<TextEncoding> EncodeAsync(int[] ids, int[]? ids2 = null, int? layer = null, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException("umT5 encodes prompts as text (CLIP Text Encode); CLIP token chunks do not apply.");

    public Task<TextEncoding> EmptyPromptAsync(int? layer, IReadOnlyList<WeightStep> weights) => EncodePromptAsync("", CancellationToken.None);

    public Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) =>
        weights.Count == 0 ? Task.CompletedTask : throw new NotSupportedException("LoRAs on umT5 are not available yet.");

    public void Dispose() => _encoder?.Dispose();
}

/// <summary>
/// Wan 2.2's VAE (Load VAE): pictures to a video latent and back in ComfyUI's units (the raw VAE latent; the library works normalised, raw = z std + mean).
/// Encodes one picture (the first latent frame) for now: a longer video needs the library's video encoder.
/// </summary>
internal sealed class WanVaeModel(MlRuntime runtime, string file) : IVideoVaeModel, IDisposable, IWanResident
{
    GpuWeights? _weights;
    WanVae? _vae;
    float[]? _mean, _std;

    public string Name => file;
    public SdVae? SdVae => null;
    public int SpatialScale => 16;
    public int TemporalScale => 4;
    public int LatentChannels => 48;

    async Task<WanVae> VaeAsync(CancellationToken ct)
    {
        WanResidency.Use(this);
        if (_vae is not null) return _vae;
        _weights = await Wan22Pipeline.LoadWeightsAsync(runtime.Accelerator, runtime.ModelSource, WanFiles.Repo, file, ct);
        _vae = new WanVae(runtime.Accelerator, _weights);
        (_mean, _std) = await _vae.GetLatentNormalizationAsync();
        return _vae;
    }

    public async Task<LatentBatch> EncodeVideoAsync(ImageBatch frames, CancellationToken cancellationToken = default)
    {
        if (VideoVae.LatentFrames(frames.Batch, TemporalScale) > 1)
            throw new NotSupportedException($"Encoding {frames.Batch} pictures as a video (more than one latent frame) is not available yet; give one start picture.");
        if (frames.Height % SpatialScale != 0 || frames.Width % SpatialScale != 0)
            throw new ArgumentException($"Wan 2.2's VAE takes sides that are multiples of 16, got {frames.Width}x{frames.Height}.");
        var vae = await VaeAsync(cancellationToken);
        var z = await vae.EncodeImageAsync(VaePictures.ToVae(frames, 0), frames.Height, frames.Width); // [48, 1, h, w], normalised
        int plane = z.Length / LatentChannels;
        for (int i = 0; i < z.Length; i++) z[i] = z[i] * _std![i / plane] + _mean![i / plane];
        return new LatentBatch(1, LatentChannels, 1, frames.Height / SpatialScale, frames.Width / SpatialScale, z);
    }

    public async Task<ImageBatch> DecodeVideoAsync(LatentBatch latent, CancellationToken cancellationToken = default)
    {
        if (!latent.IsVideo || latent.Channels != LatentChannels)
            throw new ArgumentException($"Wan 2.2's VAE decodes 48-channel video latents; this latent is [{string.Join(", ", latent.Shape)}].");
        var vae = await VaeAsync(cancellationToken);
        int f = latent.Frames, h = latent.Height, w = latent.Width, per = latent.ItemSize, plane = f * h * w;
        int frames = (f - 1) * TemporalScale + 1, H = h * SpatialScale, W = w * SpatialScale, pixels = H * W;
        var images = new ImageBatch(latent.Batch * frames, H, W, 3);
        for (int b = 0; b < latent.Batch; b++)
        {
            var z = latent.Data.AsSpan(b * per, per).ToArray();
            for (int i = 0; i < z.Length; i++) z[i] = (z[i] - _mean![i / plane]) / _std![i / plane];
            var video = await vae.DecodeAsync(z, f, h, w); // [3, frames, H, W] in -1..1
            for (int t = 0; t < frames; t++)
            {
                int o = (b * frames + t) * pixels * 3;
                for (int c = 0; c < 3; c++)
                {
                    int src = (c * frames + t) * pixels;
                    for (int p = 0; p < pixels; p++) images.Data[o + p * 3 + c] = Math.Clamp((video[src + p] + 1f) / 2f, 0f, 1f);
                }
            }
        }
        return images;
    }

    public Task<float[]> DecodeAsync(float[] latent, int h, int w, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException("Wan 2.2's VAE decodes video latents.");

    public Task<float[]> EncodeImageAsync(float[] image, int H, int W, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException("Wan 2.2's VAE encodes into video latents.");

    public void Unload()
    {
        _vae?.Dispose();
        _weights?.Dispose();
        (_vae, _weights) = (null, null);
    }

    public void Dispose()
    {
        WanResidency.Forget(this);
        Unload();
    }
}
