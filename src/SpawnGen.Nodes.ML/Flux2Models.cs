using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.StableDiffusion;

namespace SpawnGen.Nodes.ML;

// SpawnGen's plumbing over SpawnDev.ILGPU.ML's FLUX.2 building blocks as its docs describe them (SdVae for ComfyUI's flux2-vae.safetensors,
// Flux2VaeNorm, Flux2Kernels.VaeToLatent / LatentToVae), so ComfyUI's split FLUX.2 graph keeps ComfyUI's LATENT. Measured with ComfyUI
// (2026-11-13, _DevComms 2026-11-13-2230 dumps, gated by the library against them): FLUX.2's LATENT is [b, 128, H/16, W/16], the VAE encoder's
// mean patchified 2 x 2 (channel c * 4 + dy * 2 + dx) and batch-normalised with the statistics the VAE file carries; a picture is cut to multiples
// of 16 around its middle first.

/// <summary>FLUX.2's VAE (ComfyUI's flux2-vae.safetensors, Load VAE) giving and taking ComfyUI's FLUX.2 LATENT.</summary>
internal sealed class Flux2VaeModel(string name, SdVae vae, Flux2VaeNorm norm, Accelerator accelerator) : IVaeModel, IDisposable
{
    /// <summary>Latent channels of FLUX.2's LATENT (the VAE's 32 times the 2 x 2 patch).</summary>
    public const int LatentChannels = 128;

    readonly Flux2Kernels _kernels = new(accelerator);
    readonly MemoryBuffer1D<float, Stride1D.Dense> _mean = accelerator.Allocate1D(norm.Mean);
    readonly MemoryBuffer1D<float, Stride1D.Dense> _std = accelerator.Allocate1D(norm.Std);

    public string Name { get; } = name;
    public SdVae SdVae { get; } = vae;

    /// <summary>One LATENT cell is 16 x 16 pixels.</summary>
    public int Downscale => 16;

    public static async Task<Flux2VaeModel> CreateAsync(string name, SdVae vae, Accelerator accelerator, IModelSource source, string repo, string file,
        CancellationToken ct = default) => new(name, vae, await Flux2VaeNorm.FromSafeTensorsAsync(source, repo, file, ct), accelerator);

    /// <summary>[128, h, w] LATENT to the picture [3, 16h, 16w] in [-1, 1].</summary>
    public async Task<float[]> DecodeAsync(float[] latent, int h, int w, CancellationToken cancellationToken = default)
    {
        if (latent.Length != LatentChannels * h * w) throw new ArgumentException($"A FLUX.2 latent is {LatentChannels} channels; this one has {latent.Length / (h * w)}.");
        using var lat = accelerator.Allocate1D(latent);
        using var z = accelerator.Allocate1D<float>(32 * 4 * h * w);
        using var picture = accelerator.Allocate1D<float>(3 * 256 * h * w);
        _kernels.LatentToVae(lat.View, _mean.View, _std.View, z.View, h, w);
        await SdVae.DecodeAsync(z.View, 2 * h, 2 * w, picture.View, cancellationToken);
        await accelerator.SynchronizeAsync();
        return await picture.CopyToHostAsync(0, picture.Length);
    }

    /// <summary>The picture [3, H, W] in [-1, 1] (H, W multiples of 16) to the LATENT [128, H/16, W/16].</summary>
    public async Task<float[]> EncodeImageAsync(float[] image, int H, int W, CancellationToken cancellationToken = default)
    {
        if (H % 16 != 0 || W % 16 != 0) throw new ArgumentException($"FLUX.2's VAE encodes sides that are multiples of 16, got {W}x{H}.");
        using var pixels = accelerator.Allocate1D(image);
        using var z = accelerator.Allocate1D<float>(32 * (H / 8) * (W / 8));
        using var latent = accelerator.Allocate1D<float>(LatentChannels * (H / 16) * (W / 16));
        await SdVae.EncodeAsync(pixels.View, H, W, z.View, cancellationToken);
        _kernels.VaeToLatent(z.View, _mean.View, _std.View, latent.View, H / 8, W / 8);
        await accelerator.SynchronizeAsync();
        return await latent.CopyToHostAsync(0, latent.Length);
    }

    public void Dispose()
    {
        _mean.Dispose();
        _std.Dispose();
        SdVae.Dispose();
    }
}

/// <summary>
/// FLUX.2 Klein's text encoder (Load CLIP type "flux2": ComfyUI's qwen_3_4b.safetensors): the prompt in Qwen3's chat template (thinking off),
/// right-padded to 512 tokens, encoded to [1, 512, 7680] (hidden states 9, 18 and 27 side by side). Measured with ComfyUI (2026-11-13): its
/// CONDITIONING is [1, 512, 7680] with an attention mask over the real tokens (an empty prompt is 12 template tokens, "a red apple on a wooden table"
/// 19) and no pooled output; the padding rows are not zero. The weights (6.2 GB of the file's 8 GB) are loaded for each encode and freed after it,
/// so the transformer has the device afterwards.
/// </summary>
internal sealed class Flux2TextEncoderModel(MlRuntime runtime, string name, string repo, string file) : ITextEncoderModel, IPromptTextEncoder, IDisposable
{
    /// <summary>Where the Qwen tokenizer comes from (ComfyUI's file carries none): the library's hosted FLUX.2 klein export.</summary>
    public const string TokenizerRepo = "LostBeard/FLUX.2-klein-4B-q4-ILGPU";
    public const int Length = 512;

    Qwen2Tokenizer? _tokenizer;

    public string Name => name;
    public SdFamily Family => SdFamily.SD1;
    public SpawnDev.ILGPU.ML.Preprocessing.BPETokenizer Tokenizer => throw new NotSupportedException("FLUX.2's Qwen3 has no CLIP tokenizer.");
    public int FirstTowerWidth => Flux2TextEncoder.EmbedWidth;
    public SdTextEncoder? TextEncoder => null;

    /// <summary>The prompt's ids as ComfyUI's flux2 CLIP feeds them: chat-templated, padded to 512; and the real token count.</summary>
    public async Task<(int[] Ids, int Real)> TokenizeAsync(string text, CancellationToken cancellationToken)
    {
        _tokenizer ??= new Qwen2Tokenizer(System.Text.Encoding.UTF8.GetString(await runtime.ModelSource.FetchBytesAsync(TokenizerRepo, Flux2Pipeline.TokenizerFile, cancellationToken)));
        return _tokenizer.EncodePadded(text, Length);
    }

    public async Task<TextEncoding> EncodePromptAsync(string text, CancellationToken cancellationToken)
    {
        var (ids, real) = await TokenizeAsync(text, cancellationToken);
        var acc = runtime.Accelerator;
        using var weights = await Flux2Weights.LoadAsync(acc, runtime.ModelSource, repo, file, cancellationToken);
        using var encoder = new Flux2TextEncoder(acc, weights);
        var rows = encoder.Encode(ids, real);
        await acc.SynchronizeAsync();
        var context = new float[Length * Flux2TextEncoder.EmbedWidth];
        rows.SubView(0, context.Length).CopyToCPU(context);
        return new TextEncoding(context, [1, Length, Flux2TextEncoder.EmbedWidth], null);
    }

    public Task<TextEncoding> EncodeAsync(int[] ids, int[]? ids2 = null, int? layer = null, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException("FLUX.2's Qwen3 encodes prompts as text (CLIP Text Encode); CLIP token chunks do not apply.");

    public Task<TextEncoding> EmptyPromptAsync(int? layer, IReadOnlyList<WeightStep> weights) => EncodePromptAsync("", CancellationToken.None);

    public Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) =>
        weights.Count == 0 ? Task.CompletedTask : throw new NotSupportedException("LoRAs on FLUX.2's text encoder are not available yet.");

    public void Dispose() { }
}
