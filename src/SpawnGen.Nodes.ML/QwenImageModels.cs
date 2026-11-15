using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML.Flux1;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.QwenImage;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnDev.ILGPU.ML.Wan;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

// SpawnGen's plumbing over SpawnDev.ILGPU.ML's Qwen-Image blocks as its Docs/flux1.md describes them (QwenImageTransformer, Flux2Weights /
// Flux2TextEncoder with the final norm, WanComfyFormat + WanVae in Wan 2.1 mode, Flux1Kernels' patchify), so ComfyUI's split Qwen-Image graph
// (Load Diffusion Model, Load CLIP type qwen_image, Load VAE qwen_image_vae, ModelSamplingAuraFlow, KSampler, VAE Decode) runs on them.
// Behaviour measured black box against ComfyUI (bef11fc; Fixtures/qwenimage, _DevComms 2026-11-14-0800).

/// <summary>
/// Qwen-Image's text encoder (Load CLIP type qwen_image with ComfyUI's qwen_2.5_vl_7b_fp8_scaled.safetensors): Qwen2.5-VL-7B's language model, the
/// last hidden state after the final norm, of the prompt in ComfyUI's Qwen-Image template, with the template's first 34 rows (the system turn and
/// "&lt;|im_start|&gt;user\n") dropped: [1, tokens, 3584], no mask, no pooled output. Measured with ComfyUI (2026-11-14): 46 ids for "a red apple on a
/// wooden table" -> 12 rows, 39 for the empty prompt -> 5; the ids are Qwen's tokenizer's (identical, checked by the library's gate).
/// </summary>
internal sealed class QwenImageTextEncoderModel(MlRuntime runtime, string name, string repo, string file) : ITextEncoderModel, IPromptTextEncoder, IDisposable
{
    public const int Width = 3584, Layers = 28, Dropped = 34;

    /// <summary>ComfyUI's Qwen-Image template around the prompt (measured: the ids it fed the embedding table, decoded).</summary>
    public const string Prefix = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";
    public const string Suffix = "<|im_end|>\n<|im_start|>assistant\n";

    Qwen2Tokenizer? _tokenizer;

    public string Name => name;
    public SdFamily Family => SdFamily.SD1;
    public SpawnDev.ILGPU.ML.Preprocessing.BPETokenizer Tokenizer => throw new NotSupportedException("Qwen-Image's Qwen2.5-VL has no CLIP tokenizer.");
    public int FirstTowerWidth => Width;
    public SdTextEncoder? TextEncoder => null;

    /// <summary>The ids of a prompt in the template (all of them: ComfyUI does not cut long prompts).</summary>
    public async Task<int[]> TokenizeAsync(string text, CancellationToken cancellationToken)
    {
        _tokenizer ??= new Qwen2Tokenizer(System.Text.Encoding.UTF8.GetString(await runtime.ModelSource.FetchBytesAsync(Flux2TextEncoderModel.TokenizerRepo, Flux2Pipeline.TokenizerFile, cancellationToken)));
        return _tokenizer.Encode(Prefix + text + Suffix);
    }

    public async Task<TextEncoding> EncodePromptAsync(string text, CancellationToken cancellationToken)
    {
        var ids = await TokenizeAsync(text, cancellationToken);
        int rows = ids.Length - Dropped;
        var acc = runtime.Accelerator;
        // The 7B encoder and a loaded transformer do not fit one card together.
        ModelResidency.ReleaseTransformers();
        using var weights = await Flux2Weights.LoadAsync(acc, runtime.ModelSource, repo, file, Layers, finalNorm: true, cancellationToken);
        // Sized for this prompt (any length, as ComfyUI); precise activations: the reference's fp32 (CUDA's tensor cores would round to bf16).
        using var encoder = new Flux2TextEncoder(acc, weights, [Layers], finalNorm: true, maxTokens: ids.Length) { PreciseActivations = true };
        var all = encoder.Encode(ids, ids.Length);
        await acc.SynchronizeAsync();
        var context = new float[rows * Width];
        all.SubView((long)Dropped * Width, context.Length).CopyToCPU(context);
        return new TextEncoding(context, [1, rows, Width], null);
    }

    public Task<TextEncoding> EncodeAsync(int[] ids, int[]? ids2 = null, int? layer = null, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException("Qwen-Image's Qwen2.5-VL encodes prompts as text (CLIP Text Encode); CLIP token chunks do not apply.");

    public Task<TextEncoding> EmptyPromptAsync(int? layer, IReadOnlyList<WeightStep> weights) => EncodePromptAsync("", CancellationToken.None);

    public Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) =>
        weights.Count == 0 ? Task.CompletedTask : throw new NotSupportedException("LoRAs on Qwen-Image's text encoder are not available yet.");

    public void Dispose() { }
}

/// <summary>The loaded image transformers give way to a text encoder or another transformer (an 8 GB card holds one big model at a time).</summary>
internal static class ModelResidency
{
    public static void ReleaseTransformers()
    {
        Flux2Residency.Release();
        ZImageResidency.Release();
        Flux1Residency.Release();
        QwenImageResidency.Release();
    }
}

/// <summary>The loaded Qwen-Image transformer gives way to the text encoder (see <see cref="Flux2Residency"/>).</summary>
internal static class QwenImageResidency
{
    static QwenImageDiffusionModel? _resident;

    public static void Holding(QwenImageDiffusionModel model) => _resident = model;

    public static void Release()
    {
        _resident?.Unload();
        _resident = null;
    }

    public static void Forget(QwenImageDiffusionModel model)
    {
        if (ReferenceEquals(_resident, model)) _resident = null;
    }
}

/// <summary>
/// Qwen-Image as a MODEL (Load Diffusion Model: ComfyUI's qwen_image_fp8_e4m3fn.safetensors). Measured with ComfyUI (2026-11-14 / 15): the model sees
/// the Wan 2.1 VAE's normalised latent ((LATENT - mean) / std, one frame) and the text rows unpadded, with sigma itself as the timestep; ComfyUI's
/// denoised estimate is x - sigma x output (the output is the velocity). Its own schedule is a flow shift of e^1.15 (simple, 2 steps: 1, 0.75951, 0);
/// the template sets ModelSamplingAuraFlow 3.1. The prompt and the negative run as separate calls at their own lengths. The 20B transformer's blocks
/// past the device's budget stream from the file on every call.
/// </summary>
internal sealed class QwenImageDiffusionModel(MlRuntime runtime, string name, string repo, string file) : IDiffusionModel, IDisposable
{
    /// <summary>Whether tensor names are a Qwen-Image transformer (bare, or under ComfyUI's model.diffusion_model. prefix): its text norm and image input.</summary>
    public static bool Is(IReadOnlyCollection<string> names) =>
        names.Any(n => n.EndsWith("txt_norm.weight", StringComparison.Ordinal)) && names.Any(n => n.EndsWith("img_in.weight", StringComparison.Ordinal))
        && names.Any(n => n.Contains("transformer_blocks.0.attn.to_q.weight", StringComparison.Ordinal));

    /// <summary>The longest text a run can pass (rows after the template's 34): one open serves the prompt and its negative at their own lengths.</summary>
    public const int MaxTextTokens = 1024;

    QwenImageTransformer? _transformer;
    (int Rows, int Cols) _layout;

    public string Name => name;
    public SdFamily Family => SdFamily.SD1;
    public LatentFormat LatentFormat => LatentFormat.Wan21;
    public ModelSampling Sampling { get; } = new(ModelSigmas.Flow(ModelSigmas.Sd3Shift(Math.Exp(1.15)), 1), PredictionType.Const);
    public SdUNet? UNet => null;

    public Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) =>
        weights.Count == 0 ? Task.CompletedTask : throw new NotSupportedException("LoRAs and model merges on Qwen-Image are not available yet.");

    public UNetRun BeginRun(Accelerator accelerator) => new QwenImageRun(this, accelerator);

    /// <summary>The device memory the transformer keeps resident: all but 3 GB of the device (the rest of its blocks stream from the file).</summary>
    long ResidentBytes => Math.Max(1L << 30, runtime.Accelerator.MemorySize - (3L << 30));

    /// <summary>The transformer for this picture size (token grid = latent / 2); each call passes its text's length.</summary>
    public async Task<QwenImageTransformer> TransformerAsync(int rows, int cols, CancellationToken ct)
    {
        if (_transformer is not null && _layout == (rows, cols)) return _transformer;
        ModelResidency.ReleaseTransformers();
        _transformer = await QwenImageTransformer.OpenAsync(runtime.Accelerator, runtime.ModelSource, repo, file, MaxTextTokens, rows, cols, ResidentBytes, ct);
        _layout = (rows, cols);
        QwenImageResidency.Holding(this);
        return _transformer;
    }

    public void Unload()
    {
        _transformer?.Dispose();
        _transformer = null;
    }

    public void Dispose()
    {
        QwenImageResidency.Forget(this);
        Unload();
    }

    sealed class QwenImageRun(QwenImageDiffusionModel model, Accelerator acc) : UNetRun
    {
        readonly Dictionary<CondEntry, MemoryBuffer1D<float, Stride1D.Dense>> _text = new(ReferenceEqualityComparer.Instance);
        MemoryBuffer1D<float, Stride1D.Dense>? _latent, _tokens, _zeros, _velocity;
        Flux1Kernels? _kernels;

        /// <summary>The velocity the sampler expects (PredictionType.Const: denoised = x - sigma v), which is the transformer's output, unpatchified.</summary>
        public override async Task<float[]> PredictAsync(float[] input, int h, int w, float timestep, CondEntry entry, IReadOnlyList<ControlCall>? controls,
            UNetOptions options, float[]? concat, CancellationToken cancellationToken)
        {
            if (controls is { Count: > 0 }) throw new NotSupportedException("ControlNets do not apply to Qwen-Image yet.");
            if (entry.ReferenceLatents is { Count: > 0 }) throw new NotSupportedException("Reference latents (Qwen-Image Edit) are not available yet.");
            if (entry.Shape[^1] != QwenImageTextEncoderModel.Width)
                throw new ArgumentException($"Qwen-Image takes its Qwen2.5-VL text ([1, tokens, {QwenImageTextEncoderModel.Width}]); this conditioning is [{string.Join(", ", entry.Shape)}]. Encode the prompt with Load CLIP type qwen_image.");
            const int c = 16;
            int n = c * h * w;
            if (input.Length != n) throw new ArgumentException($"A Qwen-Image latent is {c} channels of one frame; this one has {input.Length / (h * w)} planes.");
            if (h % 2 != 0 || w % 2 != 0) throw new ArgumentException($"Qwen-Image needs an even latent size (pictures in multiples of 16); this latent is {w} x {h}.");
            int rows = h / 2, cols = w / 2, text = entry.Shape[^2];
            if (text > MaxTextTokens) throw new ArgumentException($"Qwen-Image takes at most {MaxTextTokens} text rows; this conditioning has {text}.");
            var transformer = await model.TransformerAsync(rows, cols, cancellationToken);
            _kernels ??= new Flux1Kernels(acc);
            if (_latent is null || _latent.Length != n)
            {
                _latent?.Dispose(); _tokens?.Dispose(); _zeros?.Dispose(); _velocity?.Dispose();
                _latent = acc.Allocate1D<float>(n);
                _tokens = acc.Allocate1D<float>(n * 4L);
                _zeros = acc.Allocate1D<float>(n);
                _zeros.MemSetToZero();
                _velocity = acc.Allocate1D<float>(n);
            }
            if (!_text.TryGetValue(entry, out var buffer)) _text[entry] = buffer = acc.Allocate1D(entry.Embeddings);
            _latent.View.CopyFromCPU(input);
            _kernels.Patchify(_latent.View, _tokens!.View, rows, cols);
            await transformer.ForwardAsync(_tokens.View, buffer.View, timestep, text, cancellationToken);
            // 0 - (-1) x output, unpatchified: the velocity.
            _kernels.Denoised(_zeros!.View, transformer.Output, _velocity!.View, -1f, rows, cols);
            await acc.SynchronizeAsync();
            return await _velocity.CopyToHostAsync(0, n);
        }

        public override void Dispose()
        {
            _latent?.Dispose(); _tokens?.Dispose(); _zeros?.Dispose(); _velocity?.Dispose();
            foreach (var t in _text.Values) t.Dispose();
            _text.Clear();
        }
    }
}

/// <summary>
/// The Wan 2.1 VAE in ComfyUI's layout (Load VAE: qwen_image_vae.safetensors; wan_2.1_vae.safetensors is the same network): pictures to a one-frame
/// video latent [1, 16, 1, H/8, W/8] in ComfyUI's units (the raw VAE latent; the library works normalised, raw = z std + mean) and back. Measured with
/// ComfyUI (2026-11-14): VAE Encode of a 256 x 192 picture gives [1, 16, 1, 24, 32]; the library's encode is 1.75e-4 from it, its decode 5.0e-5.
/// </summary>
internal sealed class WanComfyVaeModel(MlRuntime runtime, string name, string repo, string file) : IVideoVaeModel, IDisposable
{
    /// <summary>Whether tensor names are a Wan VAE in ComfyUI's layout (conv1 / conv2 around encoder and decoder, no latents_mean).</summary>
    public static bool Is(IReadOnlyCollection<string> names) =>
        names.Contains("conv1.weight") && names.Contains("encoder.conv1.weight") && names.Contains("decoder.conv1.weight") && !names.Contains("latents_mean");

    GpuWeights? _weights;
    WanVae? _vae;
    float[]? _mean, _std;

    public string Name => name;
    public SdVae? SdVae => null;
    public int SpatialScale => 8;
    public int TemporalScale => 4;
    public int LatentChannels => 16;

    async Task<WanVae> VaeAsync(CancellationToken ct)
    {
        if (_vae is not null) return _vae;
        ModelResidency.ReleaseTransformers();
        // ComfyUI's layout is rewritten into the library's on the device as it loads (WanComfyFormat).
        _weights = await Wan22Pipeline.LoadWeightsAsync(runtime.Accelerator, runtime.ModelSource, repo, file, ct);
        _vae = new WanVae(runtime.Accelerator, _weights);
        if (!_vae.IsWan21 || _vae.LatentChannels != 16 || _vae.SpatialRatio != 8)
            throw new NotSupportedException($"'{name}' is not the Wan 2.1 VAE (16 channels, 8x); SpawnGen reads ComfyUI's Wan 2.1 / Qwen-Image VAE in this layout.");
        (_mean, _std) = await _vae.GetLatentNormalizationAsync();
        return _vae;
    }

    public async Task<LatentBatch> EncodeVideoAsync(ImageBatch frames, CancellationToken cancellationToken = default)
    {
        if (frames.Batch != 1)
            throw new NotSupportedException($"Encoding {frames.Batch} pictures as a video (more than one frame) with the Wan 2.1 VAE is not available yet; give one picture.");
        var vae = await VaeAsync(cancellationToken);
        var z = await vae.EncodeImageAsync(VaePictures.ToVae(frames, 0), frames.Height, frames.Width); // [16, h, w], normalised
        int plane = z.Length / LatentChannels;
        for (int i = 0; i < z.Length; i++) z[i] = z[i] * _std![i / plane] + _mean![i / plane];
        return new LatentBatch(1, LatentChannels, 1, frames.Height / SpatialScale, frames.Width / SpatialScale, z);
    }

    public async Task<ImageBatch> DecodeVideoAsync(LatentBatch latent, CancellationToken cancellationToken = default)
    {
        if (!latent.IsVideo || latent.Channels != LatentChannels)
            throw new ArgumentException($"The Wan 2.1 VAE decodes 16-channel video latents; this latent is [{string.Join(", ", latent.Shape)}].");
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
                for (int ch = 0; ch < 3; ch++)
                {
                    int src = (ch * frames + t) * pixels;
                    for (int p = 0; p < pixels; p++) images.Data[o + p * 3 + ch] = Math.Clamp((video[src + p] + 1f) / 2f, 0f, 1f);
                }
            }
        }
        return images;
    }

    public Task<float[]> DecodeAsync(float[] latent, int h, int w, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException("The Wan 2.1 VAE decodes video latents ([b, 16, frames, h, w]).");

    public Task<float[]> EncodeImageAsync(float[] image, int H, int W, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException("The Wan 2.1 VAE encodes into video latents.");

    public void Dispose()
    {
        _vae?.Dispose();
        _weights?.Dispose();
        (_vae, _weights) = (null, null);
    }
}
