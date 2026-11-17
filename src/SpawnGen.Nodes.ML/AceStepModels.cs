using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML.AceStep;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.SkinTokens;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnDev.ILGPU.ML.Trellis2;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

// SpawnGen's plumbing over SpawnDev.ILGPU.ML's ACE-Step 1.5 blocks (AceStepPrompts, AceStepCodeLm, AceStepConditionEncoder, AceStepDiT,
// AceStepCodeDecoder, OobleckDecoder, AceStepSilenceLatent), split where ComfyUI's graph splits: the text encode makes the conditioning, the model
// runs one call per sampler step, the VAE decodes. Behaviour measured black box against ComfyUI (bef11fc; _DevComms 2026-11-17-0010 / 1100 / 1230).

/// <summary>ACE-Step 1.5's MIT files the library reads besides ComfyUI's (ACE-Step/Ace-Step1.5 on Hugging Face, downloaded on first use).</summary>
public static class AceStepFiles
{
    public const string Repo = "ACE-Step/Ace-Step1.5";
    /// <summary>ACE-Step's own silence latent (MIT; every 1.5 checkpoint ships the same file). ComfyUI's stored table differs (it is ComfyUI's data, not shipped).</summary>
    public const string SilenceLatent = "acestep-v15-turbo/silence_latent.pt";
    public const string TextTokenizer = "Qwen3-Embedding-0.6B/tokenizer.json";
    public const string LmTokenizer = "acestep-5Hz-lm-1.7B/tokenizer.json";
}

/// <summary>The loaded ACE-Step parts give way to the other big models (an 8 GB card holds one at a time).</summary>
internal static class AceStepResidency
{
    static AceStepDiffusionModel? _resident;
    public static void Holding(AceStepDiffusionModel model) => _resident = model;
    public static void Release() { _resident?.Unload(); _resident = null; }
    public static void Forget(AceStepDiffusionModel model) { if (ReferenceEquals(_resident, model)) _resident = null; }
}

/// <summary>What TextEncodeAceStepAudio1.5 asks of the text model (ComfyUI's inputs).</summary>
public sealed record AceStepEncodeSettings(string Tags, string Lyrics, long Seed, int Bpm, double Duration, string TimeSignature, string Language,
    string KeyScale, bool GenerateAudioCodes, double CfgScale, double Temperature, double TopP, long TopK, double MinP);

/// <summary>
/// ACE-Step 1.5's text side (Load CLIP (Dual) type ace: qwen_0.6b_ace15 for the caption and lyrics, qwen_1.7b_ace15 for the audio codes). Measured
/// with ComfyUI: the conditioning is the caption's hidden states [1, rows, 1024] (Qwen3-Embedding-0.6B, final norm), with the lyric prompt's token
/// embeddings [1, rows, 1024] and, when generate_audio_codes is on, the code LM's 5 Hz audio codes; the prompts are ACE-Step's templates
/// (AceStepPrompts: the ids equal ComfyUI's on six measured runs).
/// </summary>
internal sealed class AceStepTextModel(MlRuntime runtime, string name, (string Repo, string File) caption, (string Repo, string File) lm) : ITextEncoderModel, IDisposable
{
    public const int Width = 1024, Layers = 28;

    Qwen2Tokenizer? _text, _lmTokenizer;

    public string Name => name;
    public SdFamily Family => SdFamily.SD1;
    public BPETokenizer Tokenizer => throw new NotSupportedException("ACE-Step's Qwen3 has no CLIP tokenizer.");
    public int FirstTowerWidth => Width;
    public SdTextEncoder? TextEncoder => null;

    public Task<TextEncoding> EncodeAsync(int[] ids, int[]? ids2 = null, int? layer = null, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException("ACE-Step's text encoder is used through Text Encode ACE-Step Audio 1.5, not CLIP Text Encode.");

    public Task<TextEncoding> EmptyPromptAsync(int? layer, IReadOnlyList<WeightStep> weights) =>
        throw new NotSupportedException("ACE-Step's text encoder is used through Text Encode ACE-Step Audio 1.5.");

    public Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) =>
        weights.Count == 0 ? Task.CompletedTask : throw new NotSupportedException("LoRAs on ACE-Step's text encoders are not available yet.");

    async Task<Qwen2Tokenizer> TokenizerAsync(string file, CancellationToken ct) =>
        new(System.Text.Encoding.UTF8.GetString(await runtime.ModelSource.FetchBytesAsync(AceStepFiles.Repo, file, ct)));

    /// <summary>The caption, lyric and (codes on) LM prompt ids, as ComfyUI's node feeds its embedding tables.</summary>
    public async Task<(int[] Caption, int[] Lyrics, int[]? LmPositive, int[]? LmNegative)> TokenizeAsync(AceStepEncodeSettings s, CancellationToken ct)
    {
        int seconds = Seconds(s.Duration);
        _text ??= await TokenizerAsync(AceStepFiles.TextTokenizer, ct);
        var captionIds = _text.Encode(AceStepPrompts.Caption(s.Tags, s.Bpm, s.TimeSignature, s.KeyScale, seconds));
        var lyricIds = _text.Encode(AceStepPrompts.Lyrics(s.Language, s.Lyrics));
        if (!s.GenerateAudioCodes) return (captionIds, lyricIds, null, null);
        _lmTokenizer ??= await TokenizerAsync(AceStepFiles.LmTokenizer, ct);
        return (captionIds, lyricIds, _lmTokenizer.Encode(AceStepPrompts.LmPositive(s.Tags, s.Lyrics, s.Bpm, s.TimeSignature, s.KeyScale, seconds)),
            _lmTokenizer.Encode(AceStepPrompts.LmNegative(s.Tags, s.Lyrics)));
    }

    /// <summary>The prompts print whole seconds (measured at 10, 30 and 120); a fractional duration was not measured, so it is refused.</summary>
    public static int Seconds(double duration) =>
        duration == Math.Floor(duration) && duration >= 1
            ? (int)duration
            : throw new NotSupportedException($"A duration of {duration} s: SpawnGen takes whole seconds for ACE-Step's prompts so far (how ComfyUI prints a fraction is not measured).");

    public async Task<CondEntry> EncodeAsync(AceStepEncodeSettings s, NodeContext context)
    {
        var ct = context.CancellationToken;
        if (s.TopK != 0 || s.MinP != 0)
            throw new NotSupportedException("top_k and min_p are not available for ACE-Step's code LM in SpawnDev.ILGPU.ML yet (cfg, temperature and top_p are); leave them at 0.");
        var (captionIds, lyricIds, lmPos, lmNeg) = await TokenizeAsync(s, ct);
        var acc = runtime.Accelerator;
        ModelResidency.ReleaseTransformers();
        int[]? codes = null;
        if (lmPos is not null)
        {
            int count = 5 * Seconds(s.Duration);
            var negative = AceStepCodeLm.StripLeftPadding(lmNeg!);
            using var lmModel = await AceStepCodeLm.LoadAsync(acc, runtime.ModelSource, lm.Repo, lm.File, Math.Max(lmPos.Length, negative.Length), count, ct: ct);
            int made = 0;
            codes = await lmModel.GenerateCodesAsync(lmPos, negative, count, (int)(s.Seed & int.MaxValue), (float)s.CfgScale, (float)s.Temperature, (float)s.TopP,
                (_, _) => context.Progress(++made, count, "Generating audio codes"), ct);
        }
        using var weights = await Flux2Weights.LoadAsync(acc, runtime.ModelSource, caption.Repo, caption.File, Layers, finalNorm: true, ct);
        using var encoder = new Flux2TextEncoder(acc, weights, [Layers], finalNorm: true, maxTokens: captionIds.Length) { PreciseActivations = true };
        // The caption is real up to its first <|endoftext|> (padding from there on, measured by the library's gate); every row is kept, as ComfyUI's.
        int real = Array.IndexOf(captionIds, AceStepGenerator.EndOfText) is int first and >= 0 ? first : captionIds.Length;
        var states = encoder.Encode(captionIds, real);
        using var lyricIdBuffer = acc.Allocate1D(lyricIds);
        using var lyrics = acc.Allocate1D<float>((long)lyricIds.Length * Width);
        new SkinTokensKernels(acc).Embed(lyricIdBuffer.View, weights.Vector("embed_tokens.weight"), lyrics.View, lyricIds.Length, Width);
        await acc.SynchronizeAsync();
        var captionRows = await states.SubView(0, (long)captionIds.Length * Width).CopyToHostAsync();
        var lyricRows = await lyrics.View.CopyToHostAsync();
        return new CondEntry(captionRows, [1, captionIds.Length, Width])
        {
            Lyrics = new HostTensor([1, lyricIds.Length, Width], lyricRows),
            AudioCodes = codes,
        };
    }

    public void Dispose() { }
}

/// <summary>
/// ACE-Step 1.5's DiT as a MODEL (Load Diffusion Model: ComfyUI's acestep_v1.5_turbo.safetensors: DiT, condition encoder and code decoder).
/// Measured with ComfyUI: LATENT [b, 64, frames] is the model's space; a call takes the latent at sigma and returns the velocity (denoised = x -
/// sigma v); the model's own schedule is a flow shift of 3 (simple, 8 steps: 1, 0.9545, 0.9, 0.8333, 0.75, 0.6429, 0.5, 0.3, 0). The encoder states
/// come from the caption, the lyric embeddings and the reference latent (the silence latent); the context is [source | ones], the source being the
/// silence latent or, with audio codes, their decoded hints.
/// </summary>
internal sealed class AceStepDiffusionModel(MlRuntime runtime, string name, string repo, string file) : IDiffusionModel, IDisposable
{
    /// <summary>Whether tensor names are ACE-Step 1.5's DiT file (its detokenizer and the decoder's condition embedder).</summary>
    public static bool Is(IReadOnlyCollection<string> names) =>
        names.Any(n => n.StartsWith("detokenizer.", StringComparison.Ordinal)) && names.Any(n => n.StartsWith("decoder.condition_embedder.", StringComparison.Ordinal));

    AceStepDiT? _dit;
    AceStepConditionEncoder? _encoder;
    AceStepCodeDecoder? _codes;
    (int Frames, int Tokens) _size;
    (float[] Values, int Frames)? _silence;

    public string Name => name;
    public SdFamily Family => SdFamily.SD1;
    public LatentFormat LatentFormat => LatentFormat.AceStep15;
    public ModelSampling Sampling { get; } = new(ModelSigmas.Flow(ModelSigmas.Sd3Shift(3.0), 1), PredictionType.Const);
    public SdUNet? UNet => null;

    public Task UseWeightsAsync(IReadOnlyList<WeightStep> weights, CancellationToken cancellationToken) =>
        weights.Count == 0 ? Task.CompletedTask : throw new NotSupportedException("LoRAs and model merges on ACE-Step are not available yet.");

    public UNetRun BeginRun(Accelerator accelerator) => new AceStepRun(this, accelerator);

    /// <summary>ACE-Step's MIT silence latent, read once.</summary>
    public async Task<(float[] Values, int Frames)> SilenceAsync(CancellationToken ct)
    {
        if (_silence is { } s) return s;
        using var pt = new MemoryStream(await runtime.ModelSource.FetchBytesAsync(AceStepFiles.Repo, AceStepFiles.SilenceLatent, ct));
        var read = AceStepSilenceLatent.ReadPt(pt);
        _silence = read;
        return read;
    }

    /// <summary>The parts, sized for at least this many latent frames and encoder tokens (opened again when a run needs more).</summary>
    public async Task<(AceStepDiT DiT, AceStepConditionEncoder Encoder, AceStepCodeDecoder Codes)> PartsAsync(int frames, int tokens, CancellationToken ct)
    {
        if (_dit is not null && _size.Frames >= frames && _size.Tokens >= tokens) return (_dit, _encoder!, _codes!);
        ModelResidency.ReleaseTransformers();
        var acc = runtime.Accelerator;
        _encoder = await AceStepConditionEncoder.LoadAsync(acc, runtime.ModelSource, repo, file, Math.Max(tokens, frames), ct);
        _dit = await AceStepDiT.LoadAsync(acc, runtime.ModelSource, repo, file, frames, tokens, ct);
        _codes = await AceStepCodeDecoder.LoadAsync(acc, runtime.ModelSource, repo, file, (frames + AceStepCodeDecoder.Pool - 1) / AceStepCodeDecoder.Pool, ct);
        _size = (frames, tokens);
        AceStepResidency.Holding(this);
        return (_dit, _encoder, _codes);
    }

    public void Unload()
    {
        _dit?.Dispose(); _encoder?.Dispose(); _codes?.Dispose();
        (_dit, _encoder, _codes) = (null, null, null);
    }

    public void Dispose()
    {
        AceStepResidency.Forget(this);
        Unload();
    }

    sealed class AceStepRun(AceStepDiffusionModel model, Accelerator acc) : UNetRun
    {
        const int C = AceStepDiT.LatentChannels, Text = AceStepTextModel.Width;
        readonly Dictionary<CondEntry, (MemoryBuffer1D<float, Stride1D.Dense> Encoder, MemoryBuffer1D<float, Stride1D.Dense> Context, int Tokens)> _entries
            = new(ReferenceEqualityComparer.Instance);
        MemoryBuffer1D<float, Stride1D.Dense>? _x, _xt, _vt, _v;
        DiTKernels? _k;

        /// <summary>The velocity (PredictionType.Const: denoised = x - sigma v), [64, frames] channel-major.</summary>
        public override async Task<float[]> PredictAsync(float[] input, int h, int w, float timestep, CondEntry entry, IReadOnlyList<ControlCall>? controls,
            UNetOptions options, float[]? concat, CancellationToken cancellationToken)
        {
            if (controls is { Count: > 0 }) throw new NotSupportedException("ControlNets do not apply to ACE-Step.");
            if (entry.Shape[^1] != Text || entry.Lyrics is null)
                throw new ArgumentException($"ACE-Step takes Text Encode ACE-Step Audio 1.5's conditioning (caption [1, rows, {Text}] with lyrics); this one is [{string.Join(", ", entry.Shape)}].");
            int frames = w;
            if (h != 1 || input.Length != C * frames) throw new ArgumentException($"An ACE-Step latent is [64, frames] audio; this one has {input.Length / Math.Max(1, h * w)} channels over {h} x {w}.");
            int captionRows = entry.Shape[^2], lyricRows = entry.Lyrics.Shape[^2], tokens = lyricRows + 1 + captionRows;
            var (dit, encoder, codeDecoder) = await model.PartsAsync(frames, tokens, cancellationToken);
            _k ??= new DiTKernels(acc);
            if (!_entries.TryGetValue(entry, out var prepared))
                _entries[entry] = prepared = await PrepareAsync(entry, encoder, codeDecoder, frames, captionRows, lyricRows, tokens, cancellationToken);
            int n = C * frames;
            if (_x is null || _x.Length != n)
            {
                _x?.Dispose(); _xt?.Dispose(); _vt?.Dispose(); _v?.Dispose();
                _x = acc.Allocate1D<float>(n); _xt = acc.Allocate1D<float>(n); _vt = acc.Allocate1D<float>(n); _v = acc.Allocate1D<float>(n);
            }
            _x.View.CopyFromCPU(input);
            _k.Transpose(_x.View, _xt!.View, C, frames);                               // [64, T] -> [T, 64]
            dit.Forward(_xt.View, prepared.Context.View, prepared.Encoder.View, frames, prepared.Tokens, timestep, timestep, _vt!.View);
            _k.Transpose(_vt.View, _v!.View, frames, C);                               // [T, 64] -> [64, T]
            await acc.SynchronizeAsync();
            return await _v.CopyToHostAsync(0, n);
        }

        /// <summary>One conditioning entry's encoder states and context (the silence latent for this length, or the codes' hints in its place).</summary>
        async Task<(MemoryBuffer1D<float, Stride1D.Dense>, MemoryBuffer1D<float, Stride1D.Dense>, int)> PrepareAsync(CondEntry entry, AceStepConditionEncoder encoder,
            AceStepCodeDecoder codeDecoder, int frames, int captionRows, int lyricRows, int tokens, CancellationToken ct)
        {
            var (values, total) = await model.SilenceAsync(ct);
            var silence = AceStepSilenceLatent.TimeMajor(values, total, frames);       // [T, 64]
            using var caption = acc.Allocate1D(entry.Embeddings);
            using var lyrics = acc.Allocate1D(entry.Lyrics!.Data);
            using var refer = acc.Allocate1D(silence);
            var states = acc.Allocate1D<float>((long)tokens * AceStepDiT.Dim);
            encoder.Encode(caption.View, captionRows, lyrics.View, lyricRows, refer.View, frames, states.View);
            var context = acc.Allocate1D<float>((long)frames * AceStepDiT.ContextChannels);
            var ones = new float[frames * C];
            Array.Fill(ones, 1f);
            using var onesBuffer = acc.Allocate1D(ones);
            using var source = acc.Allocate1D(silence);
            if (entry.AudioCodes is { Length: > 0 } codes)
            {
                int hintFrames = codes.Length * AceStepCodeDecoder.Pool;
                using var hints = acc.Allocate1D<float>((long)hintFrames * C);
                codeDecoder.Decode(codes, hints.View);
                long copy = (long)Math.Min(hintFrames, frames) * C;
                source.View.SubView(0, copy).CopyFrom(hints.View.SubView(0, copy));
                await acc.SynchronizeAsync();
            }
            new AceStepKernels(acc).Concat(source.View, onesBuffer.View, context.View, frames, C, C);
            await acc.SynchronizeAsync();
            return (states, context, tokens);
        }

        public override void Dispose()
        {
            _x?.Dispose(); _xt?.Dispose(); _vt?.Dispose(); _v?.Dispose();
            foreach (var (e, c, _) in _entries.Values) { e.Dispose(); c.Dispose(); }
            _entries.Clear();
        }
    }
}

/// <summary>ACE-Step 1.5's audio VAE in ComfyUI's layout (Load VAE: ace_1.5_vae.safetensors, Oobleck): an audio latent [b, 64, frames] to stereo 48 kHz.</summary>
internal sealed class AceStepVaeModel(MlRuntime runtime, string name, string repo, string file) : IAudioVaeModel, IDisposable
{
    /// <summary>Whether tensor names are the Oobleck VAE (its encoder and decoder layer stacks, nothing else at the top).</summary>
    public static bool Is(IReadOnlyCollection<string> names) =>
        names.Count > 0 && names.All(n => n.StartsWith("encoder.layers.", StringComparison.Ordinal) || n.StartsWith("decoder.layers.", StringComparison.Ordinal));

    OobleckDecoder? _decoder;

    public string Name => name;
    public SdVae? SdVae => null;
    /// <summary>Measured with ComfyUI: VAE Decode Audio gives 480000 samples a channel for 10 s (250 frames x 1920), sample_rate 48000.</summary>
    public int SampleRate => 48000;

    public Task<float[]> DecodeAsync(float[] latent, int h, int w, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException("ACE-Step's VAE decodes audio latents (VAE Decode Audio).");

    public Task<float[]> EncodeImageAsync(float[] image, int H, int W, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException("ACE-Step's VAE encodes audio, not pictures.");

    public async Task<AudioClip> DecodeAudioAsync(LatentBatch latent, CancellationToken cancellationToken = default)
    {
        if (!latent.IsAudio || latent.Channels != AceStepDiT.LatentChannels)
            throw new ArgumentException($"ACE-Step's VAE decodes audio latents [b, 64, frames]; this latent is [{string.Join(", ", latent.Shape)}].");
        if (latent.Batch != 1)
            throw new NotSupportedException($"A batch of {latent.Batch} audio latents: SpawnGen's AUDIO holds one clip so far, so VAE Decode Audio takes batch 1.");
        ModelResidency.ReleaseTransformers();
        _decoder ??= await OobleckDecoder.LoadAsync(runtime.Accelerator, runtime.ModelSource, repo, file, ct: cancellationToken);
        int frames = latent.Width;
        var acc = runtime.Accelerator;
        using var input = acc.Allocate1D(latent.Data.AsSpan(0, latent.ItemSize).ToArray());
        using var audio = await _decoder.DecodeAsync(input.View, frames, cancellationToken);
        await acc.SynchronizeAsync();
        int samples = _decoder.HopLength * frames, channels = _decoder.AudioChannels;
        var all = await audio.View.SubView(0, (long)channels * samples).CopyToHostAsync();
        return new AudioClip([.. Enumerable.Range(0, channels).Select(c => all.AsSpan(c * samples, samples).ToArray())], SampleRate);
    }

    public void Dispose()
    {
        _decoder?.Dispose();
        _decoder = null;
    }
}
