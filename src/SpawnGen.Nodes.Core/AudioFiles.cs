using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// The FLAC files the audio save nodes write, like ComfyUI's AudioSaveHelper: PyAV hands the clip to FFmpeg's FLAC encoder as 16-bit samples
/// (<see cref="Samples16"/>) with the run's metadata as Vorbis comments. A host with an accelerator sets <see cref="Flac"/> (SpawnDev.Codecs); without it the
/// nodes write the same 16-bit samples as WAV.
/// </summary>
public static class AudioFiles
{
    /// <summary>Encodes a clip as FLAC with these Vorbis comments (in order: prompt, then the editor's extras such as workflow).</summary>
    public static Func<AudioClip, IReadOnlyList<KeyValuePair<string, string>>, Task<byte[]>>? Flac { get; set; }

    /// <summary>The clip interleaved as 16-bit values: x * 32768 rounded half to even and clipped (FFmpeg's flt -> s16, which PyAV picks for FLAC).</summary>
    public static int[] Samples16(AudioClip clip)
    {
        int n = clip.Length, c = clip.ChannelCount;
        var samples = new int[n * c];
        for (int f = 0; f < n; f++)
            for (int k = 0; k < c; k++)
                samples[f * c + k] = (int)Math.Clamp(MathF.Round(clip.Channels[k][f] * 32768f), -32768f, 32767f);
        return samples;
    }

    /// <summary>
    /// Encodes a clip as Ogg Opus at an Opus rate (<see cref="OpusRate"/>, already applied) with this bit rate (null = FFmpeg's default) and these
    /// comments. A host with an accelerator sets it (SpawnDev.Codecs).
    /// </summary>
    public static Func<AudioClip, int?, IReadOnlyList<KeyValuePair<string, string>>, Task<byte[]>>? Opus { get; set; }

    /// <summary>
    /// Decodes a sound file Load Audio is given that is not WAV (FLAC), or null when the format is not one it reads. A host with an accelerator sets it
    /// (SpawnDev.Codecs).
    /// </summary>
    public static Func<byte[], Task<AudioClip?>>? Decode { get; set; }

    /// <summary>The sound file extensions Load Audio lists.</summary>
    public static readonly string[] LoadableExtensions = [".wav", ".flac"];

    /// <summary>A sound file as ComfyUI's Load Audio reads it: WAV here, other formats through <see cref="Decode"/>.</summary>
    public static async Task<AudioClip> LoadAsync(byte[] data)
    {
        if (data.Length >= 12 && data.AsSpan(0, 4).SequenceEqual("RIFF"u8) && data.AsSpan(8, 4).SequenceEqual("WAVE"u8)) return WavFile.Decode(data);
        if (Decode is { } decode && await decode(data) is { } clip) return clip;
        throw new NotSupportedException(Decode is null
            ? "Load Audio reads WAV here; FLAC needs SpawnDev.Codecs on an accelerator (the server or the browser engine)."
            : "Load Audio reads WAV and FLAC; this file is neither.");
    }

    /// <summary>The Opus quality choices and their bit rates (ComfyUI's SaveAudioOpus / Save Audio (Advanced)).</summary>
    public static readonly string[] OpusQualities = ["64k", "96k", "128k", "192k", "320k"];
    public static readonly string[] Mp3Qualities = ["V0", "128k", "320k"];

    static readonly int[] OpusRates = [8000, 12000, 16000, 24000, 48000];

    /// <summary>ComfyUI's rate for Opus: above 48 kHz is 48 kHz; a rate Opus does not have becomes the next higher one it has.</summary>
    public static int OpusRate(int rate) => rate > 48000 ? 48000 : OpusRates.Contains(rate) ? rate : OpusRates.FirstOrDefault(r => r > rate, 48000);

    /// <summary>The clip as ComfyUI saves it (FLAC when <see cref="Flac"/> is set, else WAV), with the file extension.</summary>
    public static async Task<(byte[] Data, string Extension)> EncodeAsync(AudioClip clip, IReadOnlyDictionary<string, string> metadata) =>
        Flac is { } flac ? (await flac(clip, [.. metadata]), "flac") : (WavFile.Encode(clip), "wav");

    /// <summary>
    /// ComfyUI's AudioSaveHelper.save_audio for a format ("flac", "mp3" or "opus") and quality: the file and its extension. Opus resamples the clip to an Opus
    /// rate first (comfy.audio.resample, <see cref="AudioDsp.Resample(AudioClip, int)"/>). MP3 is not available yet: SpawnDev.Codecs has no MP3 encoder.
    /// </summary>
    public static async Task<(byte[] Data, string Extension)> EncodeAsync(AudioClip clip, IReadOnlyDictionary<string, string> metadata, string format, string? quality)
    {
        switch (format)
        {
            case "flac":
                return await EncodeAsync(clip, metadata);
            case "opus":
                if (Opus is not { } opus) throw new NotSupportedException("Saving Opus needs SpawnDev.Codecs on an accelerator (the server or the browser engine).");
                if (clip.ChannelCount > 2) throw new ArgumentException($"Opus saving takes mono or stereo audio; this clip has {clip.ChannelCount} channels.");
                int? bitrate = quality switch { "64k" => 64000, "96k" => 96000, "128k" => 128000, "192k" => 192000, "320k" => 320000, _ => null };
                return (await opus(AudioDsp.Resample(clip, OpusRate(clip.SampleRate)), bitrate, [.. metadata]), "opus");
            case "mp3":
                throw new NotSupportedException("Saving MP3 is not available yet: SpawnDev.Codecs has no MP3 encoder.");
            default:
                throw new ArgumentException($"Unsupported audio format: '{format}'");
        }
    }
}
