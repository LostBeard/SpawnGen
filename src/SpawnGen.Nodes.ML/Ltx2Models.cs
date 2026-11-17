using System.Text;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Ltx2;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

// SpawnGen's plumbing over SpawnDev.ILGPU.ML's LTX-2 (Ltx2: Gemma3TextEncoder, Ltx2TextConnectors, Ltx2AvModel, Ltx2VideoDecoder,
// Ltx2AudioDecoder), split where ComfyUI's video_ltx2_* graphs split. Behaviour measured black box against ComfyUI (bef11fc; _DevComms
// 2026-11-17-1700 / 1900 / 2100; Fixtures/ltx2).

/// <summary>Which files are LTX-2's: the checkpoint (transformer, video VAE, audio VAE, vocoder, text connectors in one file).</summary>
internal static class Ltx2Checkpoint
{
    public static bool Is(IReadOnlyCollection<string> names) =>
        names.Contains("model.diffusion_model.audio_patchify_proj.weight") && names.Any(n => n.StartsWith("vocoder.", StringComparison.Ordinal));

    /// <summary>
    /// The checkpoint's parts as ComfyUI's Load Checkpoint gives them: the model, no text encoder (ComfyUI warns "no CLIP/text encoder weights in
    /// checkpoint"; LTX-2's comes from Load LTXV Audio Text Encoder) and the video VAE.
    /// </summary>
    public static (ModelHandle, ClipHandle, VaeHandle) Load(MlRuntime runtime, string name, string repo, string file) =>
        (new ModelHandle(new Ltx2DiffusionModel(runtime, name, repo, file)), new ClipHandle(new NoTextEncoder(name)), new VaeHandle(new Ltx2VideoVaeModel(runtime, name, repo, file)));

    sealed class NoTextEncoder(string name) : ITextEncoderModel
    {
        public string Name => name;
        public SdFamily Family => SdFamily.SD1;
        public BPETokenizer Tokenizer => throw Missing();
        public int FirstTowerWidth => 0;
        public SdTextEncoder? TextEncoder => null;
        public Task<TextEncoding> EncodeAsync(int[] ids, int[]? ids2 = null, int? layer = null, CancellationToken cancellationToken = default) => throw Missing();
        public Task<TextEncoding> EmptyPromptAsync(int? layer, IReadOnlyList<WeightStep> weights) => throw Missing();
        public Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) => throw Missing();
        Exception Missing() => new NotSupportedException($"{name} holds no text encoder (as in ComfyUI): use Load LTXV Audio Text Encoder for LTX-2's Gemma.");
    }
}

/// <summary>
/// LTX-2's text side (Load LTXV Audio Text Encoder: gemma_3_12B_it(_fp4_mixed) + the checkpoint's text connectors). Measured with ComfyUI: the prompt
/// is BOS (2) then Gemma's SentencePiece BPE ids (no dummy prefix); Gemma's 49 hidden states are normalised and projected, and the video and audio connectors give the
/// conditioning [1, 1024, 7680] (video | audio). The two models load in turn and are released after the encode (a 12 GB card holds one at a time).
/// </summary>
internal sealed class Ltx2TextModel(MlRuntime runtime, string name, (string Repo, string File) gemma, (string Repo, string File) checkpoint)
    : ITextEncoderModel, IPromptTextEncoder, IDisposable
{
    public const int Width = 7680;
    SentencePieceTokenizer? _tokenizer;

    public string Name => name;
    public SdFamily Family => SdFamily.SD1;
    public BPETokenizer Tokenizer => throw new NotSupportedException("LTX-2's Gemma uses SentencePiece, not CLIP's tokenizer.");
    public int FirstTowerWidth => Width;
    public SdTextEncoder? TextEncoder => null;

    long ResidentBytes => Math.Max(1L << 30, runtime.Accelerator.MemorySize - (3L << 30));

    /// <summary>The prompt's ids as ComfyUI feeds Gemma: BOS, then the SentencePiece ids (the left padding to 1024 is the encoder's business).</summary>
    public async Task<int[]> TokenizeAsync(string text, CancellationToken ct)
    {
        if (_tokenizer is null)
        {
            var proto = await SafetensorsHeader.TensorBytesAsync(runtime.ModelSource, gemma.Repo, gemma.File, "spiece_model")
                ?? throw new InvalidDataException($"{gemma.File} has no spiece_model tensor: Load LTXV Audio Text Encoder takes ComfyUI's Gemma 3 12B file.");
            // SentencePiece BPE from the file's own model (SpawnDev.ILGPU.ML 3b59a3fb; measured: all 116 ids equal ComfyUI's)
            _tokenizer = SentencePieceTokenizer.FromSentencePieceModel(proto);
        }
        int[] ids = [_tokenizer.BosId, .. _tokenizer.Encode(text)];
        if (ids.Length > Gemma3TextEncoder.MaxPositions)
            throw new NotSupportedException($"The prompt is {ids.Length} tokens; LTX-2's text encoder takes {Gemma3TextEncoder.MaxPositions}.");
        return ids;
    }

    public async Task<TextEncoding> EncodePromptAsync(string text, CancellationToken cancellationToken)
    {
        var ids = await TokenizeAsync(text, cancellationToken);
        var acc = runtime.Accelerator;
        ModelResidency.ReleaseTransformers();
        float[] features;
        using (var encoder = await Gemma3TextEncoder.OpenAsync(acc, runtime.ModelSource, gemma.Repo, gemma.File, ResidentBytes, cancellationToken))
        {
            var f = await encoder.EncodeAsync(ids, cancellationToken);
            await acc.SynchronizeAsync();
            features = await f.SubView(0, (long)ids.Length * Gemma3TextEncoder.FeatureWidth).CopyToHostAsync();
        }
        using var connectors = await Ltx2TextConnectors.OpenAsync(acc, runtime.ModelSource, checkpoint.Repo, checkpoint.File, cancellationToken);
        using var upload = acc.Allocate1D(features);
        var conditioning = await connectors.RunAsync(upload.View, ids.Length, cancellationToken);
        await acc.SynchronizeAsync();
        var context = await conditioning.SubView(0, (long)Ltx2TextConnectors.Tokens * Width).CopyToHostAsync();
        return new TextEncoding(context, [1, Ltx2TextConnectors.Tokens, Width], null);
    }

    public Task<TextEncoding> EncodeAsync(int[] ids, int[]? ids2 = null, int? layer = null, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException("LTX-2's Gemma encodes prompts as text (CLIP Text Encode); CLIP token chunks do not apply.");

    public Task<TextEncoding> EmptyPromptAsync(int? layer, IReadOnlyList<WeightStep> weights) => EncodePromptAsync("", CancellationToken.None);

    public Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) =>
        weights.Count == 0 ? Task.CompletedTask : throw new NotSupportedException("LoRAs on LTX-2's text encoder are not available yet.");

    public void Dispose() { }
}

/// <summary>
/// LTX-2's audio + video transformer as a MODEL (from Load Checkpoint). It runs on the packed audio + video latent ([1, 1, video then audio], the
/// layout ComfyUI's model receives): one model call per sampler step, ComfyUI's denoised output turned into the velocity SpawnGen's flow sampling
/// expects (v = (x - denoised) / sigma), the timestep sigma itself. The frame rate comes from LTXVConditioning.
/// </summary>
internal sealed class Ltx2DiffusionModel(MlRuntime runtime, string name, string repo, string file) : IDiffusionModel, IDisposable
{
    Ltx2AvModel? _model;
    (int F, int H, int W, int Ta) _size;

    public string Name => name;
    public SdFamily Family => SdFamily.SD1;
    public LatentFormat LatentFormat => LatentFormat.Ltx2Av;
    /// <summary>Flow with the timestep = sigma (LTX-2's schedules come from LTXVScheduler; this one serves BasicScheduler).</summary>
    public ModelSampling Sampling { get; } = new(ModelSigmas.Flow(ModelSigmas.Sd3Shift(1.0), 1), PredictionType.Const);
    public SdUNet? UNet => null;

    long ResidentBytes => Math.Max(1L << 30, runtime.Accelerator.MemorySize - (3L << 30));

    public Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) =>
        weights.Count == 0 ? Task.CompletedTask : throw new NotSupportedException("LoRAs on LTX-2 are not available yet (the template's distilled and camera LoRAs need them).");

    public UNetRun BeginRun(Accelerator accelerator) => new Ltx2Run(this);

    async Task<Ltx2AvModel> ModelAsync(int F, int H, int W, int Ta, CancellationToken ct)
    {
        if (_model is not null && _size == (F, H, W, Ta)) return _model;
        _model?.Dispose();
        _model = null;
        ModelResidency.ReleaseTransformers();
        _model = await Ltx2AvModel.OpenAsync(runtime.Accelerator, runtime.ModelSource, repo, file, 1, F, H, W, Ta, Ltx2TextConnectors.Tokens, ResidentBytes, ct);
        _size = (F, H, W, Ta);
        Ltx2Residency.Holding(this);
        return _model;
    }

    public void Unload() { _model?.Dispose(); _model = null; }

    public void Dispose() { Ltx2Residency.Forget(this); Unload(); }

    sealed class Ltx2Run(Ltx2DiffusionModel model) : UNetRun
    {
        readonly Dictionary<CondEntry, MemoryBuffer1D<float, Stride1D.Dense>> _contexts = new(ReferenceEqualityComparer.Instance);

        public override async Task<float[]> PredictAsync(float[] input, int h, int w, float timestep, CondEntry entry, IReadOnlyList<ControlCall>? controls,
            UNetOptions options, float[]? concat, CancellationToken cancellationToken)
        {
            if (controls is { Count: > 0 }) throw new NotSupportedException("ControlNets do not apply to LTX-2.");
            if (Parts is not [var video, var audio] || video.Length != 5 || audio.Length != 4)
                throw new ArgumentException("LTX-2 samples an audio + video latent (Concat AV Latent of Empty LTXV Latent Video and Empty LTXV Latent Audio).");
            if (entry.Shape[^1] != Ltx2TextModel.Width || entry.Shape[^2] != Ltx2TextConnectors.Tokens)
                throw new ArgumentException($"LTX-2 takes Load LTXV Audio Text Encoder's conditioning [1, 1024, 7680]; this one is [{string.Join(", ", entry.Shape)}].");
            if (entry.FrameRate is not { } fps)
                throw new ArgumentException("LTX-2's conditioning needs a frame rate: put LTXVConditioning after the text encode, as ComfyUI's templates do.");
            var ltx = await model.ModelAsync(video[2], video[3], video[4], audio[2], cancellationToken);
            if (input.Length != ltx.PackedLength) throw new ArgumentException($"The packed latent has {input.Length} values; the model expects {ltx.PackedLength}.");
            if (!_contexts.TryGetValue(entry, out var context))
                _contexts[entry] = context = model.Accelerator.Allocate1D(entry.Embeddings);
            var denoised = await ltx.DenoiseAsync(input, [timestep], context.View, (float)fps, cancellationToken);
            var v = new float[input.Length];
            for (int i = 0; i < v.Length; i++) v[i] = (input[i] - denoised[i]) / timestep;
            return v;
        }

        public override void Dispose()
        {
            foreach (var c in _contexts.Values) c.Dispose();
            _contexts.Clear();
        }
    }

    internal Accelerator Accelerator => runtime.Accelerator;
}

/// <summary>The loaded LTX-2 transformer gives way to the other big models.</summary>
internal static class Ltx2Residency
{
    static Ltx2DiffusionModel? _resident;
    public static void Holding(Ltx2DiffusionModel model) => _resident = model;
    public static void Release() { _resident?.Unload(); _resident = null; }
    public static void Forget(Ltx2DiffusionModel model) { if (ReferenceEquals(_resident, model)) _resident = null; }
}

/// <summary>
/// LTX-2's video VAE (Load Checkpoint's VAE): decodes a video latent [b, 128, F, h, w] to 8 (F - 1) + 1 frames of 32 h x 32 w pixels. Measured
/// with ComfyUI: images = clamp((decoder output + 1) / 2) (9.4e-6). Encoding is not in SpawnDev.ILGPU.ML yet.
/// </summary>
internal sealed class Ltx2VideoVaeModel(MlRuntime runtime, string name, string repo, string file) : IVideoVaeModel
{
    public string Name => name;
    public SdVae? SdVae => null;
    public int SpatialScale => 32;
    public int TemporalScale => 8;
    public int LatentChannels => Ltx2VideoDecoder.LatentChannels;
    public int Downscale => 32;

    public Task<float[]> DecodeAsync(float[] latent, int h, int w, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException("LTX-2's VAE decodes video latents.");

    public Task<float[]> EncodeImageAsync(float[] image, int H, int W, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException("LTX-2's video VAE encoder is not in SpawnDev.ILGPU.ML yet.");

    public Task<LatentBatch> EncodeVideoAsync(ImageBatch frames, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException("LTX-2's video VAE encoder is not in SpawnDev.ILGPU.ML yet.");

    public async Task<ImageBatch> DecodeVideoAsync(LatentBatch latent, CancellationToken cancellationToken = default)
    {
        if (!latent.IsVideo || latent.Channels != LatentChannels)
            throw new ArgumentException($"LTX-2's VAE decodes video latents [b, 128, F, h, w]; this one is [{string.Join(", ", latent.Shape)}].");
        ModelResidency.ReleaseTransformers();
        using var decoder = await Ltx2VideoDecoder.OpenAsync(runtime.Accelerator, runtime.ModelSource, repo, file, cancellationToken);
        int Fz = latent.Frames, h = latent.Height, w = latent.Width, frames = 8 * (Fz - 1) + 1, H = 32 * h, W = 32 * w, plane = H * W;
        var all = new float[latent.Batch * frames * plane * 3];
        for (int b = 0; b < latent.Batch; b++)
        {
            var pixels = await decoder.DecodeAsync(latent.Data.AsSpan(b * latent.ItemSize, latent.ItemSize).ToArray(), Fz, h, w);   // [3, frames, H, W], -1..1
            int o = b * frames * plane * 3;
            for (int f = 0; f < frames; f++)
                for (int i = 0; i < plane; i++)
                    for (int c = 0; c < 3; c++)
                        all[o + (f * plane + i) * 3 + c] = Math.Clamp((pixels[(c * frames + f) * plane + i] + 1f) / 2f, 0f, 1f);
        }
        return new ImageBatch(latent.Batch * frames, H, W, 3, all);
    }
}
