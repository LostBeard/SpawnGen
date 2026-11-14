using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML.Flux1;
using SpawnDev.ILGPU.ML.Ltx;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnDev.ILGPU.ML.Wan;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// ComfyUI's all-in-one FLUX.1 checkpoints (flux1-dev-fp8.safetensors: the transformer under model.diffusion_model., T5-XXL and CLIP-L under
/// text_encoders., the ae under vae.) as Load Checkpoint's MODEL, CLIP and VAE.
/// </summary>
internal static class Flux1Checkpoint
{
    /// <summary>Tensors only a FLUX.1 all-in-one file carries together: the transformer's first double block and the T5 text encoder.</summary>
    public const string TransformerMarker = "model.diffusion_model.double_blocks.0.img_attn.qkv.weight", T5Prefix = "text_encoders.t5xxl.";

    public static bool Is(IReadOnlyCollection<string> names) => names.Contains(TransformerMarker) && names.Any(n => n.StartsWith(T5Prefix, StringComparison.Ordinal));

    public static async Task<(ModelHandle, ClipHandle, VaeHandle)> LoadAsync(MlRuntime runtime, string name, string repo, string file)
    {
        var model = await runtime.GetOrLoadAsync($"flux1:{repo}/{file}", () => Task.FromResult(new Flux1DiffusionModel(runtime, name, repo, file)));
        var clip = await runtime.GetOrLoadAsync($"flux1-text:{repo}/{file}", () => Task.FromResult(new Flux1TextEncoderModel(runtime, name, repo, file)));
        var vae = await runtime.GetOrLoadAsync<IDisposable>($"vae:{repo}/{file}", async () => new StandaloneVae(name, await SdVae.LoadAsync(runtime.Accelerator, runtime.ModelSource, repo, file)));
        return (new ModelHandle(model), new ClipHandle(clip), new VaeHandle((IVaeModel)vae));
    }
}

/// <summary>
/// FLUX.1's text conditioning from the checkpoint's own encoders: T5-XXL's rows for the prompt, padded to 256 with id 0 and all attended (no mask), and
/// CLIP-L's pooled vector (the final-norm end-token row, not projected). Measured with ComfyUI (2026-11-14, flux1-dev-fp8.safetensors, gated in the
/// library): T5 [1, 256, 4096], pooled [1, 768]; an empty prompt is the end token alone.
/// </summary>
internal sealed class Flux1TextEncoderModel(MlRuntime runtime, string name, string repo, string file) : ITextEncoderModel, IPromptTextEncoder, IDisposable
{
    public const int Rows = 256, Width = 4096, PooledWidth = 768;

    UnigramTokenizer? _t5Tokenizer;
    BPETokenizer? _clipTokenizer;

    public string Name => name;
    public SdFamily Family => SdFamily.SD1;
    public BPETokenizer Tokenizer => throw new NotSupportedException("FLUX.1 encodes prompts as text (CLIP Text Encode).");
    public int FirstTowerWidth => Width;
    public SdTextEncoder? TextEncoder => null;

    public async Task<TextEncoding> EncodePromptAsync(string text, CancellationToken cancellationToken)
    {
        var acc = runtime.Accelerator;
        Flux1Residency.Release(); // the encoders and a loaded transformer take turns on an 8 GB card
        if (_t5Tokenizer is null)
        {
            await using var json = await runtime.ModelSource.OpenAsync(LtxPipeline.DefaultRepo, LtxPaths.InRepo(LtxPaths.DitFile("q8")).Tokenizer, cancellationToken);
            _t5Tokenizer = UnigramTokenizer.FromTokenizerJson(json);
        }
        var t5 = _t5Tokenizer.Encode(text, true);
        if (t5.Length > Rows) t5 = t5[..Rows];
        var ids = new int[Rows];
        Array.Copy(t5, ids, t5.Length);
        float[] rows;
        using (var encoder = await UmT5Encoder.CreateAsync(acc, runtime.ModelSource, repo, file, cancellationToken))
            rows = await encoder.EncodeAsync(ids, cancellationToken);

        _clipTokenizer ??= await Checkpoint.LoadClipTokenizerAsync(runtime.ModelSource, Checkpoint.ClipTokenizerRepo, "", null);
        float[] pooled;
        using (var clip = await SdTextEncoder.LoadAsync(acc, runtime.ModelSource, repo, file, null, cancellationToken))
            pooled = await clip.PooledAsync(ClipIds(_clipTokenizer, text), false, cancellationToken);
        return new TextEncoding(rows, [1, Rows, Width], pooled);
    }

    /// <summary>CLIP-L's 77 ids: start, the prompt's CLIP BPE tokens (CLIP's own text cleaning; at most 75), end, then the end token as padding.</summary>
    internal static int[] ClipIds(BPETokenizer tokenizer, string text)
    {
        var tokens = tokenizer.EncodeCLIP(text, text.Length * 4 + 8, ClipIdsMarkers.End);
        int start = Array.IndexOf(tokens, ClipIdsMarkers.Start) + 1, end = Array.IndexOf(tokens, ClipIdsMarkers.End, start);
        var body = tokens[start..(end < 0 ? tokens.Length : end)];
        if (body.Length > 75) body = body[..75];
        var ids = new int[77];
        ids[0] = ClipIdsMarkers.Start;
        Array.Copy(body, 0, ids, 1, body.Length);
        for (int i = body.Length + 1; i < 77; i++) ids[i] = ClipIdsMarkers.End;
        return ids;
    }

    public Task<TextEncoding> EncodeAsync(int[] ids, int[]? ids2 = null, int? layer = null, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException("FLUX.1 encodes prompts as text (CLIP Text Encode); CLIP token chunks do not apply.");

    public Task<TextEncoding> EmptyPromptAsync(int? layer, IReadOnlyList<WeightStep> weights) => EncodePromptAsync("", CancellationToken.None);

    public Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) =>
        weights.Count == 0 ? Task.CompletedTask : throw new NotSupportedException("LoRAs on FLUX.1's text encoders are not available yet.");

    public void Dispose() { }
}

/// <summary>CLIP's start and end token ids.</summary>
internal static class ClipIdsMarkers
{
    public const int Start = 49406, End = 49407;
}

/// <summary>The loaded FLUX.1 transformer gives way to the text encoders (see <see cref="Flux2Residency"/>).</summary>
internal static class Flux1Residency
{
    static Flux1DiffusionModel? _resident;

    public static void Holding(Flux1DiffusionModel model) => _resident = model;

    public static void Release()
    {
        _resident?.Unload();
        _resident = null;
    }

    public static void Forget(Flux1DiffusionModel model)
    {
        if (ReferenceEquals(_resident, model)) _resident = null;
    }
}

/// <summary>
/// FLUX.1 (dev / schnell) as a MODEL. Measured with ComfyUI (2026-11-14, SpawnGenDumpModelCall on flux1-dev-fp8.safetensors, gated in the library):
/// the model sees the model-space latent ((LATENT - 0.1159) x 0.3611) at timestep sigma, the T5 rows, the pooled CLIP-L vector and FluxGuidance's
/// value (3.5 when none is set); its output is the velocity (ComfyUI's denoised estimate x - sigma v). Its own schedule is the flow with mu 1.15
/// (simple, 20 steps: 1, 0.98361, 0.96601, ...), sigma itself as the timestep. The 12B transformer keeps all but 3 GB of the device resident and
/// streams the other blocks from the file on every call.
/// </summary>
internal sealed class Flux1DiffusionModel(MlRuntime runtime, string name, string repo, string file) : IDiffusionModel, IDisposable
{
    public const double Mu = 1.15;

    Flux1Weights? _weights;

    public string Name => name;
    public SdFamily Family => SdFamily.SD1;
    public LatentFormat LatentFormat => LatentFormat.Flux1;
    public ModelSampling Sampling { get; } = new(ModelSigmas.Flow(ModelSigmas.Sd3Shift(Math.Exp(Mu)), 1, entries: 10000), PredictionType.Const);
    public SdUNet? UNet => null;

    public Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) =>
        weights.Count == 0 ? Task.CompletedTask : throw new NotSupportedException("LoRAs and model merges on FLUX.1 are not available yet.");

    public UNetRun BeginRun(Accelerator accelerator) => new Flux1Run(this, accelerator);

    long ResidentBytes => Math.Max(1L << 30, runtime.Accelerator.MemorySize - (3L << 30));

    public async Task<Flux1Weights> WeightsAsync(CancellationToken ct)
    {
        if (_weights is not null) return _weights;
        _weights = await Flux1Weights.OpenAsync(runtime.Accelerator, runtime.ModelSource, repo, file, ResidentBytes, ct);
        Flux1Residency.Holding(this);
        return _weights;
    }

    public void Unload()
    {
        _weights?.Dispose();
        _weights = null;
    }

    public void Dispose()
    {
        Flux1Residency.Forget(this);
        Unload();
    }

    sealed class Flux1Run(Flux1DiffusionModel model, Accelerator acc) : UNetRun
    {
        Flux1Transformer? _transformer;
        (int Rows, int Cols) _layout;
        readonly Dictionary<CondEntry, (MemoryBuffer1D<float, Stride1D.Dense> Text, MemoryBuffer1D<float, Stride1D.Dense> Pooled)> _text = new(ReferenceEqualityComparer.Instance);
        MemoryBuffer1D<float, Stride1D.Dense>? _latent, _tokens, _zeros, _velocity;
        Flux1Kernels? _kernels;

        public override async Task<float[]> PredictAsync(float[] input, int h, int w, float timestep, CondEntry entry, IReadOnlyList<ControlCall>? controls,
            UNetOptions options, float[]? concat, CancellationToken cancellationToken)
        {
            if (controls is { Count: > 0 }) throw new NotSupportedException("ControlNets do not apply to FLUX.1 yet.");
            if (entry.Shape[^1] != Flux1TextEncoderModel.Width || entry.Shape[^2] != Flux1TextEncoderModel.Rows || entry.Pooled is not { Length: Flux1TextEncoderModel.PooledWidth })
                throw new ArgumentException($"FLUX.1 takes T5 rows [1, 256, 4096] and a pooled CLIP-L vector [768]; this conditioning is [{string.Join(", ", entry.Shape)}]. Encode the prompt with FLUX.1's CLIP.");
            const int c = 16;
            int n = c * h * w;
            if (input.Length != n) throw new ArgumentException($"A FLUX.1 latent is {c} channels; this one has {input.Length / (h * w)}.");
            if (h % 2 != 0 || w % 2 != 0) throw new ArgumentException($"FLUX.1 needs an even latent size (pictures in multiples of 16); this latent is {w} x {h}.");
            int rows = h / 2, cols = w / 2;
            var weights = await model.WeightsAsync(cancellationToken);
            _kernels ??= new Flux1Kernels(acc);
            if (_transformer is null || _layout != (rows, cols))
            {
                _transformer?.Dispose();
                _transformer = new Flux1Transformer(acc, weights, Flux1TextEncoderModel.Rows, rows * cols);
                _transformer.SetPositions(rows, cols);
                _layout = (rows, cols);
                _latent?.Dispose(); _tokens?.Dispose(); _zeros?.Dispose(); _velocity?.Dispose();
                _latent = acc.Allocate1D<float>(n);
                _tokens = acc.Allocate1D<float>(n);
                _zeros = acc.Allocate1D<float>(n);
                _zeros.MemSetToZero();
                _velocity = acc.Allocate1D<float>(n);
            }
            if (!_text.TryGetValue(entry, out var text)) _text[entry] = text = (acc.Allocate1D(entry.Embeddings), acc.Allocate1D(entry.Pooled));
            _latent!.View.CopyFromCPU(input);
            _kernels.Patchify(_latent.View, _tokens!.View, rows, cols);
            await _transformer.ForwardAsync(_tokens.View, text.Text.View, text.Pooled.View, timestep, (float)(entry.Guidance ?? 3.5), cancellationToken);
            // 0 - (-1) x velocity, unpatchified: the velocity itself.
            _kernels.Denoised(_zeros!.View, _transformer.Output, _velocity!.View, -1f, rows, cols);
            await acc.SynchronizeAsync();
            return await _velocity.CopyToHostAsync(0, n);
        }

        public override void Dispose()
        {
            _transformer?.Dispose();
            _latent?.Dispose(); _tokens?.Dispose(); _zeros?.Dispose(); _velocity?.Dispose();
            foreach (var t in _text.Values) { t.Text.Dispose(); t.Pooled.Dispose(); }
            _text.Clear();
        }
    }
}
