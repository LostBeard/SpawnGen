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
internal sealed class Flux2TextEncoderModel(MlRuntime runtime, string name, string repo, string file) : ITextEncoderModel, IPromptTextEncoder, IQwen3File, IDisposable
{
    public (string Repo, string File) Qwen3File => (repo, file);

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
        Flux2Residency.Release(); // the encoder's 6.2 GB and a loaded transformer do not fit an 8 GB card together
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

/// <summary>
/// FLUX.2's big parts take turns on the device: the Qwen3 text encoder (6.2 GB) and the transformer (4.2 GB with the fp8 file, 7.8 GB bf16) do not
/// fit an 8 GB card together. Loading the encoder unloads a loaded transformer; the transformer loads again when a sampler next needs it.
/// </summary>
internal static class Flux2Residency
{
    static Flux2DiffusionModel? _resident;

    public static void Holding(Flux2DiffusionModel model) => _resident = model;

    /// <summary>Frees the transformer if one is loaded (the text encoder is about to load).</summary>
    public static void Release()
    {
        _resident?.Unload();
        _resident = null;
    }

    public static void Forget(Flux2DiffusionModel model)
    {
        if (ReferenceEquals(_resident, model)) _resident = null;
    }
}

/// <summary>
/// FLUX.2 Klein's transformer as a MODEL (Load Diffusion Model: ComfyUI's flux-2-klein-*.safetensors, bf16 or fp8). Measured with ComfyUI
/// (2026-11-14, SpawnGenDumpModelCall on its distilled Klein 4B graph): one model call per step with cfg 1; the model sees the LATENT as it is
/// ([b, 128, h, w]) and all 512 rows of the text conditioning (no mask), at timestep sigma; its result gives the denoised latent x - sigma v
/// (an Euler step from it reproduces ComfyUI's next input exactly). Its own schedule (BasicScheduler) is a flow with the shift e^2.02 at every
/// size. The library's transformer takes timestep sigma x 1000 and gives v.
/// </summary>
internal sealed class Flux2DiffusionModel(MlRuntime runtime, string name, string repo, string file) : IDiffusionModel, IDisposable
{
    /// <summary>The tensor only FLUX.2's transformers carry (the double blocks share one modulation): how Load Diffusion Model tells FLUX.2 from FLUX.1.</summary>
    public const string MarkerTensor = "double_stream_modulation_img.lin.weight";

    /// <summary>Measured: ComfyUI's Klein schedule is e^mu / (e^mu + 1/t - 1) with mu = 2.02 at every resolution (simple, 20 steps: each sigma solves to mu 2.0200).</summary>
    public const double Mu = 2.02;

    Flux2Weights? _weights;

    public string Name => name;
    public SdFamily Family => SdFamily.SD1;
    public LatentFormat LatentFormat => LatentFormat.Flux2;
    public ModelSampling Sampling { get; } = new(ModelSigmas.Flow(ModelSigmas.Sd3Shift(Math.Exp(Mu)), 1000, entries: 10000), PredictionType.Const);
    public SdUNet? UNet => null;

    public Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) =>
        weights.Count == 0 ? Task.CompletedTask : throw new NotSupportedException("LoRAs and model merges on FLUX.2 are not available yet.");

    public UNetRun BeginRun(Accelerator accelerator) => new Flux2Run(this, accelerator);

    public async Task<Flux2Weights> WeightsAsync(CancellationToken ct)
    {
        if (_weights is not null) return _weights;
        _weights = await Flux2Weights.LoadAsync(runtime.Accelerator, runtime.ModelSource, repo, file, ct);
        Flux2Residency.Holding(this);
        return _weights;
    }

    public void Unload()
    {
        _weights?.Dispose();
        _weights = null;
    }

    public void Dispose()
    {
        Flux2Residency.Forget(this);
        Unload();
    }

    /// <summary>One sampling run: the transformer for this latent's size, each conditioning entry's text kept on the device for the run.</summary>
    sealed class Flux2Run(Flux2DiffusionModel model, Accelerator acc) : UNetRun
    {
        Flux2Transformer? _transformer;
        string _layout = "";
        readonly Dictionary<CondEntry, MemoryBuffer1D<float, Stride1D.Dense>> _text = new(ReferenceEqualityComparer.Instance);
        MemoryBuffer1D<float, Stride1D.Dense>? _latent;
        Flux2Kernels? _kernels;

        /// <summary>
        /// The picture's tokens, then each reference latent's (Set Reference Latent: ComfyUI hands the model its LATENTs as ref_latents, measured),
        /// placed as the library's pipeline places them: reference i at time position 10 + 10 i on its own row / column grid.
        /// </summary>
        public override async Task<float[]> PredictAsync(float[] input, int h, int w, float timestep, CondEntry entry, IReadOnlyList<ControlCall>? controls,
            UNetOptions options, float[]? concat, CancellationToken cancellationToken)
        {
            if (controls is { Count: > 0 }) throw new NotSupportedException("ControlNets do not apply to FLUX.2.");
            if (entry.Shape[^1] != Flux2TextEncoder.EmbedWidth || entry.Shape[^2] != Flux2TextEncoderModel.Length)
                throw new ArgumentException($"FLUX.2 takes its Qwen3 text ([1, 512, {Flux2TextEncoder.EmbedWidth}]); this conditioning is [{string.Join(", ", entry.Shape)}]. Encode the prompt with Load CLIP type flux2.");
            int c = Flux2VaeModel.LatentChannels, n = c * h * w;
            if (input.Length != n) throw new ArgumentException($"A FLUX.2 latent is {c} channels; this one has {input.Length / (h * w)}.");
            var refs = entry.ReferenceLatents ?? [];
            if (refs.Count > Flux2Pipeline.MaxReferences) throw new NotSupportedException($"FLUX.2 takes at most {Flux2Pipeline.MaxReferences} reference latents; this conditioning has {refs.Count}.");
            foreach (var r in refs)
                if (r.Channels != c || r.IsVideo) throw new ArgumentException($"A reference latent for FLUX.2 must be a FLUX.2 LATENT ({c} channels); this one is [{string.Join(", ", r.Shape)}]. Encode it with FLUX.2's VAE.");
            var weights = await model.WeightsAsync(cancellationToken);
            _kernels ??= new Flux2Kernels(acc);
            var layout = $"{h}x{w}|" + string.Join(",", refs.Select(r => $"{r.Height}x{r.Width}:{System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(r)}"));
            if (_transformer is null || _layout != layout)
            {
                _transformer?.Dispose();
                int nRef = refs.Sum(r => r.Height * r.Width);
                _transformer = new Flux2Transformer(acc, weights, Flux2TextEncoderModel.Length, h * w + nRef);
                var ids = new List<int>(Flux2Pipeline.ImageIds(h, w));
                for (int i = 0; i < refs.Count; i++) ids.AddRange(Flux2Pipeline.ImageIds(refs[i].Height, refs[i].Width, 10 + 10 * i));
                _transformer.SetPositions([.. ids], Flux2Pipeline.TextIds(Flux2TextEncoderModel.Length));
                // The references' tokens go in once per layout: they do not change between steps.
                long offset = (long)h * w * c;
                foreach (var r in refs)
                {
                    using var refLatent = acc.Allocate1D(r.Data.AsSpan(0, c * r.Height * r.Width).ToArray());
                    _kernels.LatentToTokens(refLatent.View, _transformer.InputLatents.SubView(offset, (long)r.Height * r.Width * c), r.Height, r.Width);
                    await acc.SynchronizeAsync();
                    offset += (long)r.Height * r.Width * c;
                }
                _layout = layout;
                _latent?.Dispose();
                _latent = acc.Allocate1D<float>(n);
            }
            if (!_text.TryGetValue(entry, out var text)) _text[entry] = text = acc.Allocate1D(entry.Embeddings);
            _latent!.View.CopyFromCPU(input);
            _kernels.LatentToTokens(_latent.View, _transformer.InputLatents.SubView(0, n), h, w);
            _transformer.Forward(_transformer.InputLatents, text.View, timestep);
            _kernels.TokensToLatent(_transformer.Output.SubView(0, n), _latent.View, h, w);
            await acc.SynchronizeAsync();
            return await _latent.CopyToHostAsync(0, n);
        }

        public override void Dispose()
        {
            _transformer?.Dispose();
            _latent?.Dispose();
            foreach (var t in _text.Values) t.Dispose();
            _text.Clear();
        }
    }
}
