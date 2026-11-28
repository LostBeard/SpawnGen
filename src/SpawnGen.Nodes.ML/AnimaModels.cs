using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML.Anima;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnDev.ILGPU.ML.Trellis2;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md): Anima as ComfyUI's black box shows it (docs/PLANS.md "Next: the Anima family": runs 1 to 5, the dumps
// in _scratch/probe/anima) over SpawnDev.ILGPU.ML's AnimaLlmAdapter / AnimaDiT (the library's tools/anima-ref/ANIMA-SPEC.md). No ComfyUI source read.

/// <summary>Anima's files besides ComfyUI's (Apache-2.0 tokenizers on Hugging Face, downloaded on first use) and how its files are recognised.</summary>
public static class AnimaFiles
{
    /// <summary>Qwen3-0.6B-Base's tokenizer: ComfyUI's ids for the Qwen3 side are exactly its plain encoding (no special tokens; measured 2026-11-27).</summary>
    public const string QwenTokenizerRepo = "Qwen/Qwen3-0.6B-Base", QwenTokenizerFile = "tokenizer.json";
    /// <summary>
    /// T5's SentencePiece vocabulary: T5 v1.1 XXL's tokenizer.json as LTX-Video already uses it (LostBeard's LTX repo; the same file the library reads
    /// for LTX): ComfyUI's t5xxl_ids are exactly its encoding with EOS (1) last (both run 1 prompts, checked 2026-11-28). t5-base's tokenizer.json
    /// gives the same ids but is the older format without model.type, which the library's reader does not take (asked of Tuvok).
    /// </summary>
    public const string T5TokenizerRepo = "LostBeard/LTX-Video-2B-0.9.8-distilled-ILGPU", T5TokenizerFile = "google/t5-v1_1-xxl/tokenizer.json";
    /// <summary>The LLM adapter's T5 embedding: only Anima's DiT files carry it.</summary>
    public const string AdapterMarker = "net.llm_adapter.embed.weight";

    public static bool IsDiT(IReadOnlyCollection<string> names) => names.Contains(AdapterMarker);
}

/// <summary>
/// Anima's text encoder (Load CLIP type stable_diffusion with ComfyUI's qwen_3_06b_base.safetensors). Measured with ComfyUI (2026-11-27, run 1):
/// the conditioning is Qwen3-0.6B's last hidden state after its final norm, [1, N, 1024], N = the prompt's plain Qwen3 ids (no BOS / EOS); HF
/// transformers' Qwen3Model on the same file matches it to 6.9e-7. Beside it the conditioning carries the prompt's T5 ids (EOS last) for the
/// model's LLM adapter; the adapter's weights live in the DiT file, so the MODEL runs it (<see cref="AnimaDiffusionModel"/>).
/// </summary>
internal sealed class AnimaTextEncoderModel(MlRuntime runtime, string name, string repo, string file) : ITextEncoderModel, IPromptTextEncoder, IDisposable
{
    public const int Width = 1024, Layers = 28;

    /// <summary>Qwen3-0.6B: a Qwen3 language model whose embedding is 1024 wide with 28 layers (ComfyUI's Anima encoder file).</summary>
    public static async Task<bool> IsAsync(SpawnDev.ILGPU.ML.Hub.IModelSource source, string repo, string file, IReadOnlyCollection<string> names)
    {
        if (!Qwen3TextModel.Is(names) || !names.Any(n => n.EndsWith($"layers.{Layers - 1}.input_layernorm.weight", StringComparison.Ordinal))
            || names.Any(n => n.EndsWith($"layers.{Layers}.input_layernorm.weight", StringComparison.Ordinal))) return false;
        var embed = names.First(n => n.EndsWith("model.embed_tokens.weight", StringComparison.Ordinal));
        return await SafetensorsHeader.ShapeAsync(source, repo, file, embed) is { Length: 2 } shape && shape[1] == Width;
    }

    Qwen2Tokenizer? _qwen;
    UnigramTokenizer? _t5;

    public string Name => name;
    public SdFamily Family => SdFamily.SD1;
    public BPETokenizer Tokenizer => throw new NotSupportedException("Anima's Qwen3 has no CLIP tokenizer.");
    public int FirstTowerWidth => Width;
    public SdTextEncoder? TextEncoder => null;

    /// <summary>The prompt's Qwen3 ids (plain) and T5 ids (EOS last), as ComfyUI's encoder feeds them.</summary>
    public async Task<(int[] Qwen, int[] T5)> TokenizeAsync(string text, CancellationToken ct)
    {
        _qwen ??= new Qwen2Tokenizer(System.Text.Encoding.UTF8.GetString(await runtime.ModelSource.FetchBytesAsync(AnimaFiles.QwenTokenizerRepo, AnimaFiles.QwenTokenizerFile, ct)));
        if (_t5 is null)
        {
            using var json = new MemoryStream(await runtime.ModelSource.FetchBytesAsync(AnimaFiles.T5TokenizerRepo, AnimaFiles.T5TokenizerFile, ct));
            _t5 = UnigramTokenizer.FromTokenizerJson(json);
        }
        return (_qwen.Encode(text), _t5.Encode(text, addEos: true));
    }

    public async Task<TextEncoding> EncodePromptAsync(string text, CancellationToken cancellationToken)
    {
        var (qwen, t5) = await TokenizeAsync(text, cancellationToken);
        if (qwen.Length == 0)
            throw new NotSupportedException("An empty prompt for Anima: what ComfyUI's encoder gives for no Qwen3 tokens is not measured yet; write a prompt (the templates' negatives are not empty).");
        var acc = runtime.Accelerator;
        await runtime.MakeRoomAsync(this);
        using var weights = await Flux2Weights.LoadAsync(acc, runtime.ModelSource, repo, file, Layers, finalNorm: true, cancellationToken);
        // Precise activations: the reference's fp32 (CUDA's tensor-core path would round them to bf16; OpenCL is fp32 already).
        using var encoder = new Flux2TextEncoder(acc, weights, [Layers], finalNorm: true, maxTokens: qwen.Length) { PreciseActivations = true };
        var states = encoder.Encode(qwen, qwen.Length);
        await acc.SynchronizeAsync();
        var rows = await states.SubView(0, (long)qwen.Length * Width).CopyToHostAsync();
        return new TextEncoding(rows, [1, qwen.Length, Width], null) { T5Ids = t5 };
    }

    public Task<TextEncoding> EncodeAsync(int[] ids, int[]? ids2 = null, int? layer = null, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException("Anima's encoder encodes prompts as text (CLIP Text Encode); CLIP token chunks do not apply.");

    public Task<TextEncoding> EmptyPromptAsync(int? layer, IReadOnlyList<WeightStep> weights) => EncodePromptAsync("", CancellationToken.None);

    public Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) =>
        weights.Count == 0 ? Task.CompletedTask : throw new NotSupportedException("LoRAs on Anima's text encoder are not available yet.");

    public void Dispose() { }
}

/// <summary>The loaded Anima DiT gives way to the other big models (an 8 GB card holds one at a time).</summary>
internal static class AnimaResidency
{
    static AnimaDiffusionModel? _resident;
    public static void Holding(AnimaDiffusionModel model) => _resident = model;
    public static bool Release() { bool had = _resident is not null; _resident?.Unload(); _resident = null; return had; }
    public static void Forget(AnimaDiffusionModel model) { if (ReferenceEquals(_resident, model)) _resident = null; }
}

/// <summary>
/// Anima as a MODEL (Load Diffusion Model: ComfyUI's anima-preview3-base / anima-base-v1.0.safetensors: the Cosmos-Predict2-2B DiT with its LLM
/// adapter). Measured with ComfyUI (2026-11-27, runs 1 to 3): the DiT sees the model-space latent [16, frames, h, w] unscaled at timestep = sigma
/// and returns the velocity (ComfyUI's denoised = x - sigma v); its context is the adapter's output for the conditioning's Qwen3 rows and T5 ids,
/// zero-padded to 512 rows (unmasked, inside the library); the schedule is the shift-3 flow (simple, 4 steps: 1, 0.9, 0.75, 0.5, 0). The latent is
/// the Qwen-Image VAE's (Wan 2.1 format, one frame for a picture).
/// </summary>
internal sealed class AnimaDiffusionModel(MlRuntime runtime, string name, string repo, string file) : IDiffusionModel, IDisposable
{
    public const int Channels = 16;

    GpuWeights? _weights;
    AnimaDiT? _dit;
    AnimaLlmAdapter? _adapter;
    DiTKernels? _kernels;
    // LLLite patches by file (weights stay on the device with the DiT); the picture each one was last given
    readonly Dictionary<string, (GpuWeights Weights, AnimaLLLite Net, int Channels)> _lllite = new(StringComparer.Ordinal);
    readonly Dictionary<AnimaLLLite, object> _pictureKey = new(ReferenceEqualityComparer.Instance);

    public static bool Is(IReadOnlyCollection<string> names) => AnimaFiles.IsDiT(names);

    public string Name => name;
    public SdFamily Family => SdFamily.SD1;
    public LatentFormat LatentFormat => LatentFormat.Wan21;
    public ModelSampling Sampling { get; } = new(ModelSigmas.Flow(ModelSigmas.Sd3Shift(3.0), 1), PredictionType.Const);
    public SdUNet? UNet => null;

    public Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) =>
        weights.Count == 0 ? Task.CompletedTask : throw new NotSupportedException("LoRAs and model merges on Anima are not available yet.");

    public UNetRun BeginRun(Accelerator accelerator) => new AnimaRun(this, accelerator);

    /// <summary>The DiT and the adapter on one read of the file (bf16 on the device, about 4.2 GB).</summary>
    public async Task<(AnimaDiT Dit, AnimaLlmAdapter Adapter)> LoadAsync(CancellationToken ct)
    {
        if (_dit is not null && _adapter is not null) return (_dit, _adapter);
        await runtime.MakeRoomAsync(this);
        var acc = runtime.Accelerator;
        await using (var stream = await runtime.ModelSource.OpenAsync(repo, file, ct))
            _weights = await GpuWeights.LoadAsync(acc, stream, JsonDocument.Parse("{}").RootElement, name, cancellationToken: ct);
        // Precise activations: the reference's fp32 (the library's gate; CUDA's tensor-core path would round them to bf16).
        _kernels = new DiTKernels(acc);
        _dit = new AnimaDiT(acc, _kernels, _weights) { PreciseActivations = true };
        _adapter = new AnimaLlmAdapter(acc, new DiTKernels(acc), _weights) { PreciseActivations = true };
        AnimaResidency.Holding(this);
        return (_dit, _adapter);
    }

    /// <summary>
    /// The LLLite of <paramref name="use"/> with its picture set for a latent of <paramref name="rows"/> x <paramref name="cols"/>: the patch file
    /// loaded once, the picture (and mask) given to it again only when they change. The picture goes to the device as it is (ComfyUI's
    /// [H, W, C] IMAGE) and is reordered there; no pixel passes through .NET.
    /// </summary>
    public async Task<AnimaLLLite> LLLiteAsync(AnimaLLLiteUse use, int rows, int cols, CancellationToken ct)
    {
        var acc = runtime.Accelerator;
        if (!_lllite.TryGetValue(use.Patch.File, out var patch))
        {
            var shape = await SafetensorsHeader.ShapeAsync(runtime.ModelSource, use.Patch.Repo, use.Patch.File, "lllite_conditioning1.conv1.weight")
                ?? throw new NotSupportedException($"'{use.Patch.Name}' is not an Anima LLLite patch (no lllite_conditioning1.conv1.weight).");
            GpuWeights weights;
            await using (var stream = await runtime.ModelSource.OpenAsync(use.Patch.Repo, use.Patch.File, ct))
                weights = await GpuWeights.LoadAsync(acc, stream, JsonDocument.Parse("{}").RootElement, use.Patch.Name, cancellationToken: ct);
            _lllite[use.Patch.File] = patch = (weights, new AnimaLLLite(acc, _kernels!, weights), (int)shape[1]);
        }
        var net = patch.Net;
        net.Strength = use.Strength;
        int H = rows * 8, W = cols * 8;
        var key = (use.Image, use.Mask, H, W);
        if (_pictureKey.TryGetValue(net, out var had) && had.Equals(key)) return net;
        var image = use.Image;
        if (image.Height != H || image.Width != W)
            throw new NotSupportedException($"Apply Anima LLLite: the picture is {image.Width} x {image.Height} and the latent wants {W} x {H}; how ComfyUI fits another size is not measured yet, so give a picture of the latent's size (Resize Image / Mask).");
        if (image.Channels < 3) throw new ArgumentException($"Apply Anima LLLite takes an RGB picture; this one has {image.Channels} channel(s).");
        int plane = H * W;
        var pixels = await image.ToDeviceAsync(acc);
        using var chw = acc.Allocate1D<float>((long)image.Channels * plane);
        _kernels!.Transpose(pixels.SubView(0, (long)plane * image.Channels), chw.View, plane, image.Channels);   // [H*W, C] -> [C, H*W]; the first picture of a batch
        if (patch.Channels == 4)
        {
            var mask = use.Mask ?? throw new NotSupportedException($"'{use.Patch.Name}' is an inpainting LLLite: connect a mask (1 = repaint), as the Painter template does.");
            if (mask.Height != H || mask.Width != W) throw new NotSupportedException($"Apply Anima LLLite: the mask is {mask.Width} x {mask.Height}; the picture is {W} x {H}. Give both the same size.");
            var maskView = await mask.ToDeviceAsync(acc);
            using var picture = acc.Allocate1D<float>(4L * plane);
            new AnimaKernels(acc).InpaintPicture(chw.View.SubView(0, 3L * plane), maskView.SubView(0, plane), picture.View, H, W);
            await net.SetPictureAsync(picture.View, H, W);
        }
        else throw new NotSupportedException($"'{use.Patch.Name}' takes a {patch.Channels}-channel picture: SpawnGen runs the 4-channel inpainting LLLite so far (the control-picture ones are next, with their ComfyUI runs).");
        await acc.SynchronizeAsync();
        _pictureKey[net] = key;
        return net;
    }

    public void Unload()
    {
        // the classes do not own the weights: models first, then the weights
        foreach (var (w, net, _) in _lllite.Values) { net.Dispose(); w.Dispose(); }
        _lllite.Clear();
        _pictureKey.Clear();
        _dit?.Dispose(); _dit = null;
        _adapter?.Dispose(); _adapter = null;
        _weights?.Dispose(); _weights = null;
        _kernels = null;
    }

    public void Dispose()
    {
        AnimaResidency.Forget(this);
        Unload();
    }

    sealed class AnimaRun(AnimaDiffusionModel model, Accelerator acc) : UNetRun
    {
        // the adapter's output per conditioning entry: computed once (it does not depend on the step), kept on the device for the run
        readonly Dictionary<CondEntry, (MemoryBuffer1D<float, Stride1D.Dense> Rows, int Count)> _context = new(ReferenceEqualityComparer.Instance);
        MemoryBuffer1D<float, Stride1D.Dense>? _latent, _velocity;

        /// <summary>The velocity the sampler expects (PredictionType.Const: denoised = x - sigma v): the DiT's own output.</summary>
        public override async Task<float[]> PredictAsync(float[] input, int h, int w, float timestep, CondEntry entry, IReadOnlyList<ControlCall>? controls,
            UNetOptions options, float[]? concat, CancellationToken cancellationToken)
        {
            if (controls is { Count: > 0 }) throw new NotSupportedException("ControlNets do not apply to Anima (its control is LLLite, not available yet).");
            if (entry.T5Ids is null || entry.Shape[^1] != AnimaTextEncoderModel.Width)
                throw new ArgumentException($"Anima takes its own text encoding (Qwen3-0.6B rows [1, tokens, {AnimaTextEncoderModel.Width}] with T5 ids); this conditioning is [{string.Join(", ", entry.Shape)}]{(entry.T5Ids is null ? " without T5 ids" : "")}. Encode the prompt with Load CLIP (qwen_3_06b_base, type stable_diffusion).");
            int frames = Frames, rows = h / frames, n = Channels * h * w;
            if (input.Length != n) throw new ArgumentException($"An Anima latent is {Channels} channels; this one has {input.Length / (h * w)}.");
            if (rows % 2 != 0 || w % 2 != 0) throw new ArgumentException($"Anima needs an even latent size (pictures in multiples of 16); this latent is {w} x {rows}.");
            var (dit, adapter) = await model.LoadAsync(cancellationToken);
            if (!_context.TryGetValue(entry, out var context))
            {
                int qwenRows = entry.Shape[^2];
                using var qwen = acc.Allocate1D(entry.Embeddings);
                var output = await adapter.ForwardAsync(qwen.View, qwenRows, entry.T5Ids);
                var keep = acc.Allocate1D<float>((long)entry.T5Ids.Length * AnimaTextEncoderModel.Width);
                keep.View.CopyFrom(output.SubView(0, keep.Length));
                await acc.SynchronizeAsync();
                _context[entry] = context = (keep, entry.T5Ids.Length);
            }
            if (_latent is null || _velocity is null || _latent.Length != n)
            {
                _latent?.Dispose(); _velocity?.Dispose();
                _latent = acc.Allocate1D<float>(n);
                _velocity = acc.Allocate1D<float>(n);
            }
            // Apply Anima LLLite: in force inside its window (options.At has dropped it outside), for both CFG rows
            dit.LLLite = options.AnimaLLLite is { } lllite ? await model.LLLiteAsync(lllite, rows, w, cancellationToken) : null;
            _latent.View.CopyFromCPU(input);
            dit.SetGrid(frames, rows, w);
            dit.SetContext(0, context.Rows.View, context.Count);
            dit.Forward(_latent.View, timestep, 0, _velocity.View);
            await acc.SynchronizeAsync();
            return await _velocity.View.CopyToHostAsync();
        }

        public override void Dispose()
        {
            _latent?.Dispose(); _velocity?.Dispose();
            foreach (var c in _context.Values) c.Rows.Dispose();
            _context.Clear();
        }
    }
}
