using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// The three SD-Turbo sub-models (CLIP text encoder, UNet, VAE decoder) plus the CLIP tokenizer, loaded once.
/// Mirrors <c>ImageGenerationPipeline.CreateAsync</c> in SpawnDev.ILGPU.ML, minus the one-call generation loop.
/// </summary>
public sealed class Checkpoint : IDisposable
{
    /// <summary>SD 1.x / 2.x / Turbo VAE latent scale: the sampler output times 1/0.18215 is the VAE input.</summary>
    public const float VaeScale = 0.18215f;
    public const int LatentChannels = 4;
    public const int TokenCount = 77;

    readonly MlRuntime _runtime;
    readonly string _repoId;
    readonly object _encoderGate = new();
    Task<SpawnDev.ILGPU.ML.InferenceSession>? _vaeEncoder;

    Checkpoint(MlRuntime runtime, string repoId, string name, SpawnDev.ILGPU.ML.InferenceSession textEncoder,
        SpawnDev.ILGPU.ML.InferenceSession unet, SpawnDev.ILGPU.ML.InferenceSession vae, BPETokenizer tokenizer)
    {
        _runtime = runtime;
        _repoId = repoId;
        Name = name;
        TextEncoder = textEncoder;
        Unet = unet;
        Vae = vae;
        Tokenizer = tokenizer;
        AlphasCumprod = DiffusionScheduler.ComputeAlphasCumprod();
        Sigmas = new ModelSigmas(AlphasCumprod);
    }

    public string Name { get; }

    Task<float[]>? _emptyPrompt;

    /// <summary>The text encoder's output for the empty prompt, computed once by <paramref name="compute"/>: the baseline weighted prompts are measured against.</summary>
    public Task<float[]> EmptyPromptEmbeddingsAsync(Func<Task<float[]>> compute)
    {
        lock (_encoderGate) return _emptyPrompt ??= compute();
    }
    public SpawnDev.ILGPU.ML.InferenceSession TextEncoder { get; }
    public SpawnDev.ILGPU.ML.InferenceSession Unet { get; }
    public SpawnDev.ILGPU.ML.InferenceSession Vae { get; }
    public BPETokenizer Tokenizer { get; }
    public float[] AlphasCumprod { get; }
    /// <summary>The noise levels of the model's training steps, for schedulers and for telling the model its timestep.</summary>
    public ModelSigmas Sigmas { get; }

    /// <summary>The VAE encoder (pixels to latents), loaded on first use: text-to-image never needs it.</summary>
    public Task<SpawnDev.ILGPU.ML.InferenceSession> GetVaeEncoderAsync()
    {
        lock (_encoderGate)
            return _vaeEncoder ??= LoadVaeEncoderAsync();
    }

    async Task<SpawnDev.ILGPU.ML.InferenceSession> LoadVaeEncoderAsync()
    {
        await using var m = await _runtime.ModelSource.OpenAsync(_repoId, "vae_encoder/model.onnx");
        var session = await SpawnDev.ILGPU.ML.InferenceSession.CreateFromOnnxStreamAsync(_runtime.Accelerator, m,
            inputShapes: new Dictionary<string, int[]> { ["sample"] = [1, 3, 512, 512] });
        session.CacheShapeReadbacks = false; // the encoder is run at whatever size the picture has
        return session;
    }

    internal static async Task<Checkpoint> LoadAsync(MlRuntime runtime, string name, string repoId, Action<string, int>? onProgress)
    {
        var source = runtime.ModelSource;
        var acc = runtime.Accelerator;

        onProgress?.Invoke("tokenizer", 0);
        string vocabJson, mergesText;
        await using (var s = await source.OpenAsync(repoId, "tokenizer/vocab.json"))
        using (var r = new StreamReader(s))
            vocabJson = await r.ReadToEndAsync();
        await using (var s = await source.OpenAsync(repoId, "tokenizer/merges.txt"))
        using (var r = new StreamReader(s))
            mergesText = await r.ReadToEndAsync();
        var tokenizer = BPETokenizer.Load(vocabJson, mergesText);
        onProgress?.Invoke("tokenizer", 100);

        // The sub-models have fixed shapes (batch 1, 77 tokens, 64x64 latent); the UNet and VAE are recompiled
        // for other sizes by the session on first run.
        onProgress?.Invoke("text_encoder", 0);
        SpawnDev.ILGPU.ML.InferenceSession te, unet, vae;
        await using (var m = await source.OpenAsync(repoId, "text_encoder/model.onnx"))
            te = await SpawnDev.ILGPU.ML.InferenceSession.CreateFromOnnxStreamAsync(acc, m,
                onProgress: (s, p) => onProgress?.Invoke($"text_encoder:{s}", p),
                inputShapes: new Dictionary<string, int[]> { ["input_ids"] = [1, TokenCount] });
        onProgress?.Invoke("unet", 0);
        var info = MlRuntime.CheckpointDetails.GetValueOrDefault(name) ?? new MlRuntime.CheckpointInfo(1024);
        await using var unetData = info.UnetExternalData is null ? null : await source.OpenAsync(repoId, info.UnetExternalData);
        await using (var m = await source.OpenAsync(repoId, "unet/model.onnx"))
            unet = await SpawnDev.ILGPU.ML.InferenceSession.CreateFromOnnxStreamAsync(acc, m, externalDataStream: unetData,
                onProgress: (s, p) => onProgress?.Invoke($"unet:{s}", p),
                inputShapes: new Dictionary<string, int[]>
                {
                    ["sample"] = [1, LatentChannels, 64, 64],
                    ["timestep"] = [1],
                    ["encoder_hidden_states"] = [1, TokenCount, info.TextWidth],
                });
        onProgress?.Invoke("vae_decoder", 0);
        await using (var m = await source.OpenAsync(repoId, "vae_decoder/model.onnx"))
            vae = await SpawnDev.ILGPU.ML.InferenceSession.CreateFromOnnxStreamAsync(acc, m,
                onProgress: (s, p) => onProgress?.Invoke($"vae_decoder:{s}", p),
                inputShapes: new Dictionary<string, int[]> { ["latent_sample"] = [1, LatentChannels, 64, 64] });

        // Shapes are fixed across runs, so later runs can serve shape-constant readbacks from a CPU cache.
        te.CacheShapeReadbacks = true;
        unet.CacheShapeReadbacks = true;
        vae.CacheShapeReadbacks = true;
        onProgress?.Invoke("done", 100);
        return new Checkpoint(runtime, repoId, name, te, unet, vae, tokenizer);
    }

    public void Dispose()
    {
        TextEncoder.Dispose();
        Unet.Dispose();
        Vae.Dispose();
        if (_vaeEncoder is { IsCompletedSuccessfully: true }) _vaeEncoder.Result.Dispose();
    }
}

/// <summary>The denoiser (UNet) of a loaded checkpoint, with the sampling math constants.</summary>
[Socket("MODEL")]
public sealed class ModelHandle(Checkpoint checkpoint)
{
    public Checkpoint Checkpoint { get; } = checkpoint;
    /// <summary>RescaleCFG's multiplier (0 = off): sampling rescales the guided prediction towards the spread of the prompt's own (see <see cref="SamplingCore.RescaledDenoised"/>).</summary>
    public float RescaleCfg { get; init; }
    /// <summary>Functions run on the predictions before cfg at every step, in order (Latent Apply Operation CFG adds one; see <see cref="PreCfgArgs"/>).</summary>
    public IReadOnlyList<PreCfgFunction> PreCfg { get; init; } = [];
}

/// <summary>The CLIP text encoder and tokenizer of a loaded checkpoint.</summary>
[Socket("CLIP")]
public sealed class ClipHandle(Checkpoint checkpoint) { public Checkpoint Checkpoint { get; } = checkpoint; }

/// <summary>The VAE decoder of a loaded checkpoint.</summary>
[Socket("VAE")]
public sealed class VaeHandle(Checkpoint checkpoint) { public Checkpoint Checkpoint { get; } = checkpoint; }

