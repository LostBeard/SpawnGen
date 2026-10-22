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

    Checkpoint(string name, SpawnDev.ILGPU.ML.InferenceSession textEncoder, SpawnDev.ILGPU.ML.InferenceSession unet,
        SpawnDev.ILGPU.ML.InferenceSession vae, BPETokenizer tokenizer)
    {
        Name = name;
        TextEncoder = textEncoder;
        Unet = unet;
        Vae = vae;
        Tokenizer = tokenizer;
        AlphasCumprod = DiffusionScheduler.ComputeAlphasCumprod();
    }

    public string Name { get; }
    public SpawnDev.ILGPU.ML.InferenceSession TextEncoder { get; }
    public SpawnDev.ILGPU.ML.InferenceSession Unet { get; }
    public SpawnDev.ILGPU.ML.InferenceSession Vae { get; }
    public BPETokenizer Tokenizer { get; }
    public float[] AlphasCumprod { get; }

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
        await using (var m = await source.OpenAsync(repoId, "unet/model.onnx"))
            unet = await SpawnDev.ILGPU.ML.InferenceSession.CreateFromOnnxStreamAsync(acc, m,
                onProgress: (s, p) => onProgress?.Invoke($"unet:{s}", p),
                inputShapes: new Dictionary<string, int[]>
                {
                    ["sample"] = [1, LatentChannels, 64, 64],
                    ["timestep"] = [1],
                    ["encoder_hidden_states"] = [1, TokenCount, 1024],
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
        return new Checkpoint(name, te, unet, vae, tokenizer);
    }

    public void Dispose()
    {
        TextEncoder.Dispose();
        Unet.Dispose();
        Vae.Dispose();
    }
}

/// <summary>The denoiser (UNet) of a loaded checkpoint, with the sampling math constants.</summary>
[Socket("MODEL")]
public sealed class ModelHandle(Checkpoint checkpoint) { public Checkpoint Checkpoint { get; } = checkpoint; }

/// <summary>The CLIP text encoder and tokenizer of a loaded checkpoint.</summary>
[Socket("CLIP")]
public sealed class ClipHandle(Checkpoint checkpoint) { public Checkpoint Checkpoint { get; } = checkpoint; }

/// <summary>The VAE decoder of a loaded checkpoint.</summary>
[Socket("VAE")]
public sealed class VaeHandle(Checkpoint checkpoint) { public Checkpoint Checkpoint { get; } = checkpoint; }

/// <summary>Text embeddings from CLIP, <c>[1, 77, 1024]</c>, host side (315 KB).</summary>
[Socket("CONDITIONING")]
public sealed class Conditioning(float[] embeddings, int[] shape)
{
    public float[] Embeddings { get; } = embeddings;
    public int[] Shape { get; } = shape;
}
