using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md). The rules come from the tests and fixtures: the 16-bit samples are the ones whose MD5 ComfyUI's FLAC files
// carry (Fixtures/codecs/save_audio: x * 32768 rounded half to even, clipped), the Opus rates from SaveAudioOpusTests, the file choices from the
// save tests. The FLAC and Opus encoders and the FLAC decoder are SpawnDev.Codecs', installed as hooks (CodecsAudio).

/// <summary>Reading and writing audio files: WAV here, FLAC and Opus through SpawnDev.Codecs when it is installed.</summary>
public static class AudioFiles
{
    /// <summary>The audio files Load Audio lists.</summary>
    public static readonly string[] LoadableExtensions = [".wav", ".flac"];

    /// <summary>Save Audio's Opus and MP3 quality choices.</summary>
    public static readonly string[] OpusQualities = ["64k", "96k", "128k", "192k", "320k"];
    public static readonly string[] Mp3Qualities = ["V0", "128k", "320k"];

    /// <summary>The sample rates Opus encodes at.</summary>
    static readonly string[] OpusRates = ["8000", "12000", "16000", "24000", "48000"];

    /// <summary>The FLAC encoder: the clip and the comments (prompt, then the workflow's keys) to the file. Without one, Save Audio writes WAV.</summary>
    public static Func<AudioClip, IReadOnlyList<KeyValuePair<string, string>>, Task<byte[]>>? Flac { get; set; }

    /// <summary>The Opus encoder: the clip (at an Opus rate), the bitrate in bits per second (null: the encoder's default) and the comments.</summary>
    public static Func<AudioClip, int?, IReadOnlyList<KeyValuePair<string, string>>, Task<byte[]>>? Opus { get; set; }

    /// <summary>The MP3 encoder: the clip, the quality ("V0", "128k" or "320k") and the comments.</summary>
    public static Func<AudioClip, string, IReadOnlyList<KeyValuePair<string, string>>, Task<byte[]>>? Mp3 { get; set; }

    /// <summary>
    /// The sample rates MP3 carries (MPEG-1, 2 and 2.5 Layer III). Measured with ComfyUI: any other rate (96000) fails in its encoder; nothing is
    /// resampled.
    /// </summary>
    public static readonly int[] Mp3Rates = [8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000];

    /// <summary>The decoder for files that are not WAV (FLAC); null when it does not know the file.</summary>
    public static Func<byte[], Task<AudioClip?>>? Decode { get; set; }

    /// <summary>The clip as interleaved 16-bit samples: x * 32768 rounded half to even, clipped to the 16-bit range.</summary>
    public static int[] Samples16(AudioClip clip)
    {
        int channels = clip.Channels.Length, n = clip.Length;
        var r = new int[channels * n];
        for (int i = 0; i < n; i++)
            for (int c = 0; c < channels; c++)
            {
                float v = clip.Channels[c][i] * 32768f;
                r[i * channels + c] = (int)Math.Clamp(MathF.Round(v, MidpointRounding.ToEven), -32768f, 32767f);
            }
        return r;
    }

    /// <summary>A file as a clip: WAV directly, anything else through <see cref="Decode"/>.</summary>
    public static async Task<AudioClip> LoadAsync(byte[] data)
    {
        if (WavFile.LooksLikeWav(data)) return WavFile.Decode(data);
        if (Decode is { } decode && await decode(data) is { } clip) return clip;
        throw new NotSupportedException("This audio file is not WAV; reading FLAC needs SpawnDev.Codecs, which is not installed in this build.");
    }

    /// <summary>The Opus rate a clip is resampled to: the lowest Opus rate at or above its rate, at most 48000.</summary>
    public static int OpusRate(int rate)
    {
        foreach (var r in OpusRates.Select(int.Parse))
            if (r >= rate) return r;
        return 48000;
    }

    /// <summary>The clip as Save Audio's default file: FLAC through the encoder, else WAV. Returns the bytes and the extension.</summary>
    public static Task<(byte[] Data, string Extension)> EncodeAsync(AudioClip clip, IReadOnlyDictionary<string, string> metadata) =>
        EncodeAsync(clip, metadata, "flac", null);

    /// <summary>The clip in <paramref name="format"/> (flac, opus or mp3) at <paramref name="quality"/>; the run's prompt and workflow go in its comments.</summary>
    public static async Task<(byte[] Data, string Extension)> EncodeAsync(AudioClip clip, IReadOnlyDictionary<string, string> metadata, string format, string? quality)
    {
        var comments = Comments(metadata);
        switch (format)
        {
            case "flac":
                return Flac is { } flac ? (await flac(clip, comments), "flac") : (WavFile.Encode(clip), "wav");
            case "opus":
                var opus = Opus ?? throw new NotSupportedException("Saving Opus needs SpawnDev.Codecs, which is not installed in this build.");
                if (quality is not null && !OpusQualities.Contains(quality)) throw new ArgumentException($"Unknown Opus quality '{quality}'.", nameof(quality));
                int rate = OpusRate(clip.SampleRate);
                var resampled = rate == clip.SampleRate ? clip : AudioDsp.Resample(clip, rate);
                return (await opus(resampled, quality is null ? null : Bitrate(quality), comments), "opus");
            case "mp3":
                if (quality is not null && !Mp3Qualities.Contains(quality)) throw new ArgumentException($"Unknown MP3 quality '{quality}'.", nameof(quality));
                var mp3 = Mp3 ?? throw new NotSupportedException("Saving MP3 needs SpawnDev.Codecs, which is not installed in this build.");
                if (!Mp3Rates.Contains(clip.SampleRate))
                    throw new NotSupportedException($"MP3 cannot carry {clip.SampleRate} Hz audio (it takes {string.Join(", ", Mp3Rates)}); resample it first.");
                if (clip.ChannelCount is not (1 or 2)) throw new NotSupportedException($"MP3 carries mono or stereo audio, not {clip.ChannelCount} channels.");
                return (await mp3(clip, quality ?? "V0", comments), "mp3");
            default:
                throw new ArgumentException($"Unknown audio format '{format}'. The choices are flac, mp3 and opus.", nameof(format));
        }
    }

    /// <summary>"128k" as 128000.</summary>
    static int Bitrate(string quality) => int.Parse(quality.TrimEnd('k')) * 1000;

    /// <summary>The comments in ComfyUI's order: the prompt, then the workflow's keys.</summary>
    static List<KeyValuePair<string, string>> Comments(IReadOnlyDictionary<string, string> metadata)
    {
        var list = new List<KeyValuePair<string, string>>();
        if (metadata.TryGetValue("prompt", out var prompt)) list.Add(new("prompt", prompt));
        foreach (var kv in metadata)
            if (kv.Key != "prompt") list.Add(kv);
        return list;
    }
}
