using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

/// <summary>A VAE that decodes audio latents (ACE-Step 1.5's Oobleck VAE) for VAE Decode Audio.</summary>
public interface IAudioVaeModel : IVaeModel
{
    int SampleRate { get; }
    Task<AudioClip> DecodeAudioAsync(LatentBatch latent, CancellationToken cancellationToken = default);
}

/// <summary>
/// Empty Ace Step 1.5 Latent Audio (ComfyUI's EmptyAceStep1.5LatentAudio): zeros [batch, 64, frames], 25 frames per second. Measured with ComfyUI
/// (2026-11-17): frames = Python's round (half to even) of seconds x 25: 10.01 s and 10.02 s give 250, 10.03 s and 10.05 s 251, 10.06 s 252,
/// 7.77 s 194, 1000 s 25000; batch_size gives the batch.
/// </summary>
[AudioLatents]
[Node("EmptyAceStep1.5LatentAudio", Category = "model/latent/ace", DisplayName = "Empty Ace Step 1.5 Latent Audio")]
public sealed class EmptyAceStep15LatentAudio
{
    public static int Frames(double seconds) => (int)Math.Round(seconds * 25.0, MidpointRounding.ToEven);

    public LatentBatch Execute([Input(Min = 1.0, Max = 1000.0, Step = 0.01)] double seconds = 120.0,
        [Input(Min = 1, Max = 4096, Tooltip = "The number of latent images in the batch.")] int batchSize = 1) =>
        // ComfyUI's keys, measured: type "audio", downscale_ratio_temporal 1764
        new([batchSize, 64, Frames(seconds)]) { Type = "audio", DownscaleRatioTemporal = 1764 };
}

/// <summary>
/// TextEncodeAceStepAudio1.5: ACE-Step 1.5's conditioning from tags, lyrics and the music's metadata: the caption's hidden states, the lyric
/// embeddings and, with generate_audio_codes, the 1.7B code LM's audio codes (5 per second, sampled with cfg / temperature / top_p from the seed).
/// The code LM's random numbers are not torch's, so a seed picks different codes than ComfyUI's; the rest follows ComfyUI (measured).
/// </summary>
[Node("TextEncodeAceStepAudio1.5", Category = "model/conditioning/ace", DisplayName = "TextEncodeAceStepAudio1.5")]
public sealed class TextEncodeAceStepAudio15
{
    public static string[] TimeSignatures() => ["2", "3", "4", "6"];

    public static string[] Languages() =>
        ["ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu", "id", "is", "it", "ja", "ko", "la", "lt",
         "ms", "ne", "nl", "no", "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh", "unknown"];

    public static string[] KeyScales() =>
        ["C major", "C# major", "Db major", "D major", "D# major", "Eb major", "E major", "F major", "F# major", "Gb major", "G major", "G# major", "Ab major",
         "A major", "A# major", "Bb major", "B major", "C minor", "C# minor", "Db minor", "D minor", "D# minor", "Eb minor", "E minor", "F minor", "F# minor",
         "Gb minor", "G minor", "G# minor", "Ab minor", "A minor", "A# minor", "Bb minor", "B minor"];

    public async Task<Conditioning> ExecuteAsync(ClipHandle clip, NodeContext context,
        [Input(Multiline = true)] string tags = "", [Input(Multiline = true)] string lyrics = "",
        [Input(Min = 0, Max = 18446744073709551615d)] long seed = 0,
        [Input(Min = 10, Max = 300)] int bpm = 120,
        [Input(Min = 0.0, Max = 2000.0, Step = 0.1)] double duration = 120.0,
        [Input(OptionsProvider = nameof(TimeSignatures))] string timesignature = "2",
        [Input(OptionsProvider = nameof(Languages))] string language = "en",
        [Input(OptionsProvider = nameof(KeyScales))] string keyscale = "C major",
        [Input(Tooltip = "Enable the LLM that generates audio codes. This can be slow but will increase the quality of the generated audio. Turn this off if you are giving the model an audio reference.")]
        bool generateAudioCodes = true,
        [Input(Min = 0.0, Max = 100.0, Step = 0.1)] double cfgScale = 2.0,
        [Input(Min = 0.0, Max = 2.0, Step = 0.01)] double temperature = 0.85,
        [Input(Min = 0.0, Max = 2000.0, Step = 0.01)] double topP = 0.9,
        [Input(Min = 0, Max = 100)] long topK = 0,
        [Input(Min = 0.0, Max = 1.0, Step = 0.001)] double minP = 0.0)
    {
        if (clip.Encoder is not AceStepTextModel model)
            throw new NotSupportedException($"TextEncodeAceStepAudio1.5 takes ACE-Step 1.5's text encoders (Load CLIP (Dual) type ace); '{clip.Encoder.Name}' is not one.");
        var entry = await model.EncodeAsync(new AceStepEncodeSettings(tags, lyrics, seed, bpm, duration, timesignature, language, keyscale,
            generateAudioCodes, cfgScale, temperature, topP, topK, minP), context);
        return new Conditioning([entry]);
    }
}

/// <summary>VAE Decode Audio: an audio latent to its audio through an audio VAE (ACE-Step 1.5's: stereo, 48 kHz, 1920 samples a frame).</summary>
[AudioLatents]
[Node("VAEDecodeAudio", Category = "model/latent", DisplayName = "VAE Decode Audio")]
public sealed class VAEDecodeAudio
{
    public async Task<AudioClip> ExecuteAsync(LatentBatch samples, VaeHandle vae, NodeContext context)
    {
        if (vae.Vae is not IAudioVaeModel audio)
            throw new NotSupportedException($"{vae.Vae.Name} is not an audio VAE; VAE Decode Audio needs one (ACE-Step 1.5's ace_1.5_vae).");
        return await audio.DecodeAudioAsync(samples, context.CancellationToken);
    }
}
