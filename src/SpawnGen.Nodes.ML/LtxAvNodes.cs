using System.Text.Json;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

// LTX-2's host-side nodes (ComfyUI's video_ltx2_* templates), measured black box against ComfyUI bef11fc (2026-11-17; Fixtures/ltx2, probe
// _scratch/probe/ltx2/host_run*.py). The LTX-2 model itself (Gemma text encoder, the audio + video transformer, the VAEs) comes with
// SpawnDev.ILGPU.ML; until then the nodes that need it say so.

/// <summary>
/// Empty LTXV Latent Video: zeros [batch, 128, (length - 1) / 8 + 1, height / 32, width / 32] (divisions rounded down), with ComfyUI's key
/// downscale_ratio_spacial 32. Measured: 64 x 64 x 9 gives [1, 128, 2, 2, 2]; 768 x 512 x 97 [1, 128, 13, 16, 24]; 100 x 70 x 10 [1, 128, 2, 2, 3];
/// lengths 1 and 8 one latent frame.
/// </summary>
[Node("EmptyLTXVLatentVideo", Category = "model/latent/ltxv", DisplayName = "Empty LTXV Latent Video")]
public sealed class EmptyLTXVLatentVideo
{
    public static int[] Shape(int width, int height, int length, int batchSize) => [batchSize, 128, (length - 1) / 8 + 1, height / 32, width / 32];

    public LatentBatch Execute([Input(Min = 64, Max = 16384, Step = 32)] int width = 768, [Input(Min = 64, Max = 16384, Step = 32)] int height = 512,
        [Input(Min = 1, Max = 16384, Step = 8)] int length = 97, [Input(Min = 1, Max = 4096)] int batchSize = 1) =>
        new(Shape(width, height, length, batchSize)) { DownscaleRatioSpacial = 32 };
}

/// <summary>What LTX-2's audio latent is shaped by, read from the checkpoint's own config (the "audio_vae" entry of its safetensors metadata).</summary>
public sealed record LtxAudioLatentLayout(int Channels, int Bins, double FramesPerSecond)
{
    /// <summary>
    /// From the checkpoint's audio_vae config: z_channels latent channels; mel_bins and the sample rate / STFT hop (mel frames a second) both
    /// halved per downsampling level (one fewer than ch_mult's entries). LTX-2 19B: 8 channels, 64 / 4 = 16 bins, 16000 / 160 / 4 = 25 a second.
    /// </summary>
    public static LtxAudioLatentLayout FromConfig(JsonElement audioVae)
    {
        var dd = audioVae.GetProperty("model").GetProperty("params").GetProperty("ddconfig");
        int down = 1 << (dd.GetProperty("ch_mult").GetArrayLength() - 1);
        var pre = audioVae.GetProperty("preprocessing");
        double melPerSecond = pre.GetProperty("audio").GetProperty("sampling_rate").GetDouble() / pre.GetProperty("stft").GetProperty("hop_length").GetDouble();
        return new(dd.GetProperty("z_channels").GetInt32(), dd.GetProperty("mel_bins").GetInt32() / down, melPerSecond / down);
    }

    /// <summary>The latent frames for a video of this many frames at this rate: Python's round (half to even) of its seconds x frames a second (measured).</summary>
    public int Frames(int videoFrames, double frameRate) => (int)Math.Round(videoFrames / frameRate * FramesPerSecond, MidpointRounding.ToEven);
}

/// <summary>LTX-2's audio VAE (Load LTXV Audio VAE, from an LTX-2 checkpoint): the audio latent's layout; decoding comes with SpawnDev.ILGPU.ML's LTX-2.</summary>
internal sealed class LtxAudioVaeModel(string name, LtxAudioLatentLayout layout) : IAudioVaeModel
{
    public string Name => name;
    public LtxAudioLatentLayout Layout => layout;
    public SdVae? SdVae => null;
    /// <summary>Measured with ComfyUI: LTXV Audio VAE Decode gives 24000 Hz (the vocoder's 240 samples per mel frame at 100 mel frames a second).</summary>
    public int SampleRate => 24000;

    public Task<float[]> DecodeAsync(float[] latent, int h, int w, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException("LTX-2's audio VAE decodes audio latents (LTXV Audio VAE Decode).");

    public Task<float[]> EncodeImageAsync(float[] image, int H, int W, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException("LTX-2's audio VAE encodes audio, not pictures.");

    public Task<AudioClip> DecodeAudioAsync(LatentBatch latent, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException("LTX-2's audio VAE decoder is not in SpawnDev.ILGPU.ML yet (coming with its LTX-2 port).");
}

/// <summary>Load LTXV Audio VAE: LTX-2's audio VAE from an LTX-2 checkpoint (its audio_vae.* tensors and config).</summary>
[Node("LTXVAudioVAELoader", Category = "model/loaders", DisplayName = "Load LTXV Audio VAE", OutputNames = ["Audio VAE"])]
public sealed class LTXVAudioVAELoader
{
    public static string[] CheckpointNames() => [.. MlRuntime.KnownCheckpoints.Keys];

    public async Task<VaeHandle> ExecuteAsync([Input(OptionsProvider = nameof(CheckpointNames), Tooltip = "Audio VAE checkpoint to load.")] string ckptName,
        IServiceProvider services)
    {
        var runtime = await GpuHelpers.RuntimeAsync(services);
        if (!MlRuntime.KnownCheckpoints.TryGetValue(ckptName, out var repo) || MlRuntime.CheckpointDetails[ckptName].File is not { } file)
            throw new FileNotFoundException($"Checkpoint '{ckptName}' was not found.");
        var metadata = await SafetensorsHeader.MetadataAsync(runtime.ModelSource, repo, file);
        if (metadata?.TryGetValue("config", out var config) != true)
            throw new NotSupportedException($"{ckptName} has no config in its metadata: Load LTXV Audio VAE reads an LTX-2 checkpoint.");
        using var doc = JsonDocument.Parse(config!);
        if (!doc.RootElement.TryGetProperty("audio_vae", out var audioVae))
            throw new NotSupportedException($"{ckptName} is not an LTX-2 checkpoint (no audio VAE in its config).");
        return new VaeHandle(new LtxAudioVaeModel(ckptName, LtxAudioLatentLayout.FromConfig(audioVae)));
    }
}

/// <summary>
/// Empty LTXV Latent Audio: zeros [batch, channels, frames, bins] typed "audio", the audio VAE's layout (LTX-2 19B: 8 channels, 16 bins, 25 frames a
/// second). Measured: frames = Python's round of frames_number / frame_rate x 25 (half to even: 1 frame at 2 fps 12, 3 at 2 fps 38, 5 at 2 fps 62);
/// 97 at 25 fps 97, 121 at 24 fps 126, 10 at 30 fps 8.
/// </summary>
[Node("LTXVEmptyLatentAudio", Category = "model/latent/ltxv", DisplayName = "Empty LTXV Latent Audio", OutputNames = ["Latent"])]
public sealed class LTXVEmptyLatentAudio
{
    public LatentBatch Execute([Input(Default = 97, Min = 1, Max = 1000, Tooltip = "Number of frames.")] int framesNumber,
        [Input(Type = "FLOAT,INT", Default = 25.0, Min = 1.0, Max = 1000.0, Step = 0.01, Tooltip = "Number of frames per second.")] double frameRate,
        [Input(Default = 1, Min = 1, Max = 4096, Tooltip = "The number of latent audio samples in the batch.")] int batchSize,
        [Input(Tooltip = "The Audio VAE model to get configuration from.")] VaeHandle audioVae)
    {
        if (audioVae.Vae is not LtxAudioVaeModel vae)
            throw new NotSupportedException("Empty LTXV Latent Audio needs LTX-2's audio VAE (Load LTXV Audio VAE).");
        int frames = vae.Layout.Frames(framesNumber, frameRate);
        if (frames < 1)
            throw new ArgumentException($"{framesNumber} frames at {frameRate} fps is {framesNumber / frameRate * vae.Layout.FramesPerSecond:0.##} audio latent frames, which rounds to none; ComfyUI makes an empty latent here.");
        return new([batchSize, vae.Layout.Channels, frames, vae.Layout.Bins]) { Type = "audio" };
    }
}

/// <summary>
/// Concat AV Latent: LTX-2's audio + video latent (ComfyUI's NestedTensor), packed as the model sees it ([batch, 1, video values then audio values],
/// measured), with both latents' keys (type "audio", downscale_ratio_spacial 32).
/// </summary>
[VideoLatents]
[AudioLatents]
[Node("LTXVConcatAVLatent", Category = "model/latent/ltxv", DisplayName = "Concat AV Latent", OutputNames = ["latent"])]
public sealed class LTXVConcatAVLatent
{
    public LatentBatch Execute(LatentBatch videoLatent, LatentBatch audioLatent)
    {
        if (!videoLatent.IsVideo) throw new ArgumentException($"video_latent is [{string.Join(", ", videoLatent.Shape)}]; Concat AV Latent takes a video latent there.");
        if (!audioLatent.IsAudio) throw new ArgumentException($"audio_latent is [{string.Join(", ", audioLatent.Shape)}]; Concat AV Latent takes an audio latent there.");
        if (videoLatent.NoiseMask is not null || audioLatent.NoiseMask is not null || videoLatent.BatchIndex is not null || audioLatent.BatchIndex is not null)
            throw new NotSupportedException("Concat AV Latent with a noise mask or batch indexes is not measured yet in SpawnGen.");
        return LatentBatch.Nested(videoLatent, audioLatent);
    }
}

/// <summary>Separate AV Latent: the video and audio latents of an audio + video latent, each with all of its keys (measured: the video part keeps type "audio").</summary>
[NestedLatents]
[Node("LTXVSeparateAVLatent", Category = "model/latent/ltxv", DisplayName = "Separate AV Latent", OutputNames = ["video_latent", "audio_latent"])]
public sealed class LTXVSeparateAVLatent
{
    public (LatentBatch VideoLatent, LatentBatch AudioLatent) Execute(LatentBatch avLatent)
    {
        // ComfyUI fails on a plain latent ("tuple index out of range", measured); SpawnGen says what it needs
        if (!avLatent.IsNested) throw new ArgumentException("Separate AV Latent takes an audio + video latent (Concat AV Latent, or an LTX-2 sampler's output).");
        var parts = avLatent.SplitParts();
        return (parts[0], parts[1]);
    }
}

/// <summary>LTXVConditioning: the frame rate set on every entry of both conditionings (ComfyUI's "frame_rate"; the tensors unchanged, measured).</summary>
[Node("LTXVConditioning", Category = "model/conditioning/ltxv", DisplayName = "LTXVConditioning", OutputNames = ["positive", "negative"])]
public sealed class LTXVConditioning
{
    public (Conditioning Positive, Conditioning Negative) Execute(Conditioning positive, Conditioning negative,
        [Input(Min = 0.0, Max = 1000.0, Step = 0.01)] double frameRate = 25.0) =>
        (positive.Map(e => e with { FrameRate = frameRate }), negative.Map(e => e with { FrameRate = frameRate }));
}

/// <summary>
/// LTXVCropGuides: removes keyframe guides from the conditioning and their frames from the latent. SpawnGen has no keyframe guides yet, so the
/// conditioning passes through; the latent comes back as ComfyUI gives it (measured, no guides): its keys dropped and a noise mask of ones over every
/// latent frame. ComfyUI fails on an audio + video latent (measured), so the latent is the video one (Separate AV Latent first).
/// </summary>
[VideoLatents]
[Node("LTXVCropGuides", Category = "model/conditioning/ltxv", DisplayName = "LTXVCropGuides", OutputNames = ["positive", "negative", "latent"])]
public sealed class LTXVCropGuides
{
    public (Conditioning Positive, Conditioning Negative, LatentBatch Latent) Execute(Conditioning positive, Conditioning negative, LatentBatch latent)
    {
        if (!latent.IsVideo) throw new ArgumentException($"LTXVCropGuides takes a video latent; this one is [{string.Join(", ", latent.Shape)}].");
        var ones = MaskBatch.Filled(latent.Batch, latent.Frames, 1, 1f);
        return (positive, negative, new LatentBatch(latent.Shape, latent.Data) { NoiseMask = ones, BatchIndex = latent.BatchIndex });
    }
}

/// <summary>
/// LTXVScheduler: steps + 1 sigmas from 1 to 0, time-shifted by mu, interpolated by the latent's tokens between base_shift at 1024 and max_shift at
/// 4096 (no latent: 4096; a video latent: frames x height x width; an audio + video latent: its video part's; an audio latent: frames x bins), then
/// with stretch the non-zero ones mapped so the last is terminal. Measured to float32 precision (16 schedules: steps 2 to 30, shifts, stretch,
/// terminals, five latent sizes).
/// </summary>
[VideoLatents]
[AudioLatents]
[NestedLatents]
[Node("LTXVScheduler", Category = "model/sampling/schedulers", DisplayName = "LTXVScheduler")]
public sealed class LTXVScheduler
{
    /// <summary>The token count the shift is interpolated by.</summary>
    public static int Tokens(LatentBatch? latent)
    {
        if (latent is null) return 4096;
        var shape = latent.IsNested ? latent.Parts![0] : latent.Shape;
        int n = 1;
        for (int d = 2; d < shape.Length; d++) n *= shape[d];
        return n;
    }

    public static float[] Schedule(int steps, double maxShift, double baseShift, bool stretch, double terminal, int tokens)
    {
        double mu = baseShift + (maxShift - baseShift) * (tokens - 1024) / (4096.0 - 1024.0), e = Math.Exp(mu);
        var s = new double[steps + 1];
        for (int i = 0; i <= steps; i++)
        {
            double t = 1.0 - (double)i / steps;
            s[i] = t > 0 ? e / (e + (1 / t - 1)) : 0;
        }
        if (stretch)
        {
            int nonZero = s.Count(v => v != 0);
            double last = 1 - s[nonZero - 1];
            // One step: the stretch would divide 0 by 0. ComfyUI gives NaN there except, by float rounding, at mu 2.05 ([terminal, 0], measured);
            // SpawnGen gives that limit for every shift.
            if (last == 0) { for (int i = 0; i < nonZero; i++) s[i] = terminal; }
            else
            {
                double scale = last / (1 - terminal);
                for (int i = 0; i < nonZero; i++) s[i] = 1 - (1 - s[i]) / scale;
            }
        }
        return [.. s.Select(v => (float)v)];
    }

    public Sigmas Execute([Input(Min = 1, Max = 10000)] int steps = 20, [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double maxShift = 2.05,
        [Input(Min = 0.0, Max = 100.0, Step = 0.01)] double baseShift = 0.95,
        [Input(Tooltip = "Stretch the sigmas to be in the range [terminal, 1].")] bool stretch = true,
        [Input(Min = 0.0, Max = 0.99, Step = 0.01, Tooltip = "The terminal value of the sigmas after stretching.")] double terminal = 0.1,
        LatentBatch? latent = null) =>
        new(Schedule(steps, maxShift, baseShift, stretch, terminal, Tokens(latent)));
}

/// <summary>LTXV Audio VAE Decode: an LTX-2 audio latent to audio through its audio VAE and vocoder (stereo, 24 kHz).</summary>
[AudioLatents]
[Node("LTXVAudioVAEDecode", Category = "model/latent/ltxv", DisplayName = "LTXV Audio VAE Decode", OutputNames = ["Audio"])]
public sealed class LTXVAudioVAEDecode
{
    public async Task<AudioClip> ExecuteAsync([Input(Tooltip = "The latent to be decoded.")] LatentBatch samples,
        [Input(Tooltip = "The Audio VAE model used for decoding the latent.")] VaeHandle audioVae, NodeContext context)
    {
        if (audioVae.Vae is not LtxAudioVaeModel vae)
            throw new NotSupportedException($"{audioVae.Vae.Name} is not LTX-2's audio VAE (Load LTXV Audio VAE).");
        return await vae.DecodeAudioAsync(samples, context.CancellationToken);
    }
}
