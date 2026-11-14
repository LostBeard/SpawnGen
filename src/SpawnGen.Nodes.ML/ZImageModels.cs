using ILGPU.Runtime;
using ILGPU;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnDev.ILGPU.ML.ZImage;
using SpawnDev.ILGPU;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// Z-Image's text encoder (Load CLIP type lumina2 with ComfyUI's qwen_3_4b.safetensors): Qwen3-4B's hidden state after layer 35 of 36 (no final
/// norm), the prompt in Qwen3's chat template without the thinking block, not padded: [1, tokens, 2560]. Measured with ComfyUI (2026-11-14,
/// SpawnGenDumpConditioning on the Z-Image Turbo template graph): 15 rows for a 7-word prompt, 8 for the empty one; the library's encoder gives
/// both to 3.9e-3 (Tuvok's gate on these dumps).
/// </summary>
internal sealed class ZImageTextEncoderModel(MlRuntime runtime, string name, string repo, string file) : ITextEncoderModel, IPromptTextEncoder, IDisposable
{
    /// <summary>The Qwen3 layer whose output is the conditioning (1-based): ComfyUI's lumina2 reads hidden_states[-2].</summary>
    public const int Layer = 35;
    public const int Width = 2560;

    Qwen2Tokenizer? _tokenizer;

    public string Name => name;
    public SdFamily Family => SdFamily.SD1;
    public SpawnDev.ILGPU.ML.Preprocessing.BPETokenizer Tokenizer => throw new NotSupportedException("Z-Image's Qwen3 has no CLIP tokenizer.");
    public int FirstTowerWidth => Width;
    public SdTextEncoder? TextEncoder => null;

    public async Task<TextEncoding> EncodePromptAsync(string text, CancellationToken cancellationToken)
    {
        _tokenizer ??= new Qwen2Tokenizer(System.Text.Encoding.UTF8.GetString(await runtime.ModelSource.FetchBytesAsync(Flux2TextEncoderModel.TokenizerRepo, Flux2Pipeline.TokenizerFile, cancellationToken)));
        var (ids, real) = _tokenizer.EncodePadded(text, Qwen2Tokenizer.ChatPromptNoThinking, Flux2TextEncoderModel.Length);
        var acc = runtime.Accelerator;
        // The encoder (35 layers, bf16) and a loaded transformer do not fit an 8 GB card together.
        Flux2Residency.Release();
        ZImageResidency.Release();
        using var weights = await Flux2Weights.LoadAsync(acc, runtime.ModelSource, repo, file, Layer, cancellationToken);
        // Precise activations: the reference's fp32 (CUDA's tensor-core path would round them to bf16; OpenCL is fp32 already).
        using var encoder = new Flux2TextEncoder(acc, weights, [Layer]) { PreciseActivations = true };
        var rows = encoder.Encode(ids, real);
        await acc.SynchronizeAsync();
        var context = new float[real * Width];
        rows.SubView(0, context.Length).CopyToCPU(context);
        return new TextEncoding(context, [1, real, Width], null);
    }

    public Task<TextEncoding> EncodeAsync(int[] ids, int[]? ids2 = null, int? layer = null, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException("Z-Image's Qwen3 encodes prompts as text (CLIP Text Encode); CLIP token chunks do not apply.");

    public Task<TextEncoding> EmptyPromptAsync(int? layer, IReadOnlyList<WeightStep> weights) => EncodePromptAsync("", CancellationToken.None);

    public Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) =>
        weights.Count == 0 ? Task.CompletedTask : throw new NotSupportedException("LoRAs on Z-Image's text encoder are not available yet.");

    public void Dispose() { }
}

/// <summary>The loaded Z-Image transformer gives way to the text encoder (see <see cref="Flux2Residency"/>).</summary>
internal static class ZImageResidency
{
    static ZImageDiffusionModel? _resident;

    public static void Holding(ZImageDiffusionModel model) => _resident = model;

    public static void Release()
    {
        _resident?.Unload();
        _resident = null;
    }

    public static void Forget(ZImageDiffusionModel model)
    {
        if (ReferenceEquals(_resident, model)) _resident = null;
    }
}

/// <summary>
/// Z-Image (Turbo) as a MODEL (Load Diffusion Model: ComfyUI's z_image_turbo_bf16.safetensors). Measured with ComfyUI (2026-11-14,
/// SpawnGenDumpModelCall on its Z-Image Turbo template graph, gated in the library): the model sees the model-space latent (FLUX.1's ae space,
/// (LATENT - 0.1159) x 0.3611) and the caption rows unpadded, at the reference's model time 1 - sigma; ComfyUI's denoised estimate is x + sigma x
/// output (the model predicts minus the velocity). Its own schedule is the shift-3 flow ComfyUI's ModelSamplingAuraFlow(3) also gives (simple, 8
/// steps: 1, 0.95455, 0.9, 0.83333, 0.75, 0.64286, 0.5, 0.3, 0), with sigma itself as the timestep. The 6B transformer's blocks past the device's
/// budget stream from the file on every call.
/// </summary>
internal sealed class ZImageDiffusionModel(MlRuntime runtime, string name, string repo, string file) : IDiffusionModel, IDisposable
{
    /// <summary>A tensor Z-Image's file carries and FLUX / SD files do not: how Load Diffusion Model recognises it.</summary>
    public const string MarkerTensor = "cap_pad_token";

    ZImageTransformer? _transformer;
    (int Rows, int Cols) _layout;

    /// <summary>The longest caption a run can pass (the text encoder's 512 rows): one open serves the prompt and its negative at their own lengths.</summary>
    public const int MaxCaptionTokens = 512;

    public string Name => name;
    public SdFamily Family => SdFamily.SD1;
    public LatentFormat LatentFormat => LatentFormat.Flux1;
    public ModelSampling Sampling { get; } = new(ModelSigmas.Flow(ModelSigmas.Sd3Shift(3.0), 1), PredictionType.Const);
    public SdUNet? UNet => null;

    public Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) =>
        weights.Count == 0 ? Task.CompletedTask : throw new NotSupportedException("LoRAs and model merges on Z-Image are not available yet.");

    public UNetRun BeginRun(Accelerator accelerator) => new ZImageRun(this, accelerator);

    /// <summary>The device memory the transformer keeps resident: all but 3 GB of the device (the rest of its blocks stream from the file).</summary>
    long ResidentBytes => Math.Max(1L << 30, runtime.Accelerator.MemorySize - (3L << 30));

    /// <summary>The transformer for this picture size; each call passes its caption's length (the prompt and the negative differ).</summary>
    public async Task<ZImageTransformer> TransformerAsync(int rows, int cols, CancellationToken ct)
    {
        if (_transformer is not null && _layout == (rows, cols)) return _transformer;
        Unload();
        _transformer = await ZImageTransformer.OpenAsync(runtime.Accelerator, runtime.ModelSource, repo, file, rows, cols, MaxCaptionTokens, ResidentBytes, ct);
        _layout = (rows, cols);
        ZImageResidency.Holding(this);
        return _transformer;
    }

    public void Unload()
    {
        _transformer?.Dispose();
        _transformer = null;
    }

    public void Dispose()
    {
        ZImageResidency.Forget(this);
        Unload();
    }

    sealed class ZImageRun(ZImageDiffusionModel model, Accelerator acc) : UNetRun
    {
        readonly Dictionary<CondEntry, MemoryBuffer1D<float, Stride1D.Dense>> _text = new(ReferenceEqualityComparer.Instance);
        MemoryBuffer1D<float, Stride1D.Dense>? _latent, _zeros, _velocity;
        ZImageKernels? _kernels;

        /// <summary>The velocity the sampler expects (PredictionType.Const: denoised = x - sigma v): ComfyUI's x + sigma out is x - sigma (-out).</summary>
        public override async Task<float[]> PredictAsync(float[] input, int h, int w, float timestep, CondEntry entry, IReadOnlyList<ControlCall>? controls,
            UNetOptions options, float[]? concat, CancellationToken cancellationToken)
        {
            if (controls is { Count: > 0 }) throw new NotSupportedException("ControlNets do not apply to Z-Image yet.");
            if (entry.Shape[^1] != ZImageTextEncoderModel.Width)
                throw new ArgumentException($"Z-Image takes its Qwen3 text ([1, tokens, {ZImageTextEncoderModel.Width}]); this conditioning is [{string.Join(", ", entry.Shape)}]. Encode the prompt with Load CLIP type lumina2.");
            const int c = 16;
            int n = c * h * w;
            if (input.Length != n) throw new ArgumentException($"A Z-Image latent is {c} channels; this one has {input.Length / (h * w)}.");
            if (h % 2 != 0 || w % 2 != 0) throw new ArgumentException($"Z-Image needs an even latent size (pictures in multiples of 16); this latent is {w} x {h}.");
            int rows = h / 2, cols = w / 2, caption = entry.Shape[^2];
            if (caption > MaxCaptionTokens) throw new ArgumentException($"Z-Image takes at most {MaxCaptionTokens} caption rows; this conditioning has {caption}.");
            var transformer = await model.TransformerAsync(rows, cols, cancellationToken);
            _kernels ??= new ZImageKernels(acc);
            if (_latent is null || _latent.Length != n)
            {
                _latent?.Dispose(); _zeros?.Dispose(); _velocity?.Dispose();
                _latent = acc.Allocate1D<float>(n);
                _zeros = acc.Allocate1D<float>(n);
                _zeros.MemSetToZero();
                _velocity = acc.Allocate1D<float>(n);
            }
            if (!_text.TryGetValue(entry, out var text)) _text[entry] = text = acc.Allocate1D(entry.Embeddings);
            _latent.View.CopyFromCPU(input);
            await transformer.ForwardAsync(_latent.View, text.View, 1f - timestep, caption, cancellationToken);
            // 0 - 1 x output, unpatchified: the velocity.
            _kernels.Denoised(_zeros!.View, transformer.Output, _velocity!.View, -1f, rows, cols);
            await acc.SynchronizeAsync();
            return await _velocity.CopyToHostAsync(0, n);
        }

        public override void Dispose()
        {
            _latent?.Dispose(); _zeros?.Dispose(); _velocity?.Dispose();
            foreach (var t in _text.Values) t.Dispose();
            _text.Clear();
        }
    }
}
