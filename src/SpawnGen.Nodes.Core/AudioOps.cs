using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// ComfyUI's model-free audio nodes (comfy_extras/nodes_audio.py). ComfyUI's AUDIO is {"waveform": [batch, channels, samples], "sample_rate"}; an AudioClip is
// one batch item, which is all LoadAudio, RecordAudio and EmptyAudio ever make. Like ComfyUI, a missing clip (null: a video without sound) passes through as null.

[Node(Category = "audio", DisplayName = "Record Audio")]
public sealed class RecordAudio
{
    /// <summary>
    /// The clip the Studio recorded from the microphone. ComfyUI's frontend records a WAV, uploads it and sets the input to its path
    /// (<c>audio/recording-1700000000000.wav [temp]</c>); here the recording is a file in the input store, found by its name (folder and annotation are dropped).
    /// </summary>
    public async Task<AudioClip> ExecuteAsync(string audio)
    {
        var name = StripAnnotation(audio ?? "");
        if (name.Length == 0) throw new ArgumentException("Nothing was recorded: record a clip with the microphone first.", nameof(audio));
        return WavFile.Decode(await InputFiles.Store.ReadAsync(Path.GetFileName(name.Replace('\\', '/'))));
    }

    /// <summary>ComfyUI's annotated file names end in <c> [input]</c>, <c> [output]</c> or <c> [temp]</c>.</summary>
    internal static string StripAnnotation(string name)
    {
        foreach (var tag in (string[])[" [input]", " [output]", " [temp]"])
            if (name.EndsWith(tag, StringComparison.Ordinal)) return name[..^tag.Length];
        return name;
    }
}

[Node(Category = "audio", DisplayName = "Trim Audio Duration", Description = "Trim audio tensor into chosen time range.")]
public sealed class TrimAudioDuration
{
    public AudioClip? Execute(AudioClip? audio,
        [Input(Min = -18446744073709551615.0, Max = 18446744073709551615.0, Step = 0.01, Tooltip = "Start time in seconds, can be negative to count from the end (supports sub-seconds).")] double startIndex = 0.0,
        [Input(Min = 0.0, Step = 0.01, Tooltip = "Duration in seconds")] double duration = 60.0)
    {
        if (audio is null) return null;
        int length = audio.Length, rate = audio.SampleRate;
        if (length == 0) return audio;

        // int(round(x)) is half-to-even; working in doubles keeps huge widget values from overflowing before the clamp.
        double start = Math.Round(startIndex * rate);
        if (startIndex < 0) start += length;
        start = Math.Clamp(start, 0, length);
        double end = Math.Clamp(start + Math.Round(duration * rate), 0, length);
        if (start >= end) throw new ArgumentException("TrimAudioDuration: Start time must be less than end time and be within the audio length.");

        int from = (int)start, count = (int)end - from;
        return new AudioClip([.. audio.Channels.Select(c => c.AsSpan(from, count).ToArray())], rate);
    }
}

[Node(Category = "audio", DisplayName = "Split Audio Channels", Description = "Separates the audio into left and right channels.")]
public sealed class SplitAudioChannels
{
    public (AudioClip? Left, AudioClip? Right) Execute(AudioClip? audio)
    {
        if (audio is null) return (null, null);
        if (audio.ChannelCount != 2) throw new ArgumentException($"AudioSplit: Input audio must be stereo (2 channels), got {audio.ChannelCount} channel(s).");
        return (new AudioClip([audio.Channels[0]], audio.SampleRate), new AudioClip([audio.Channels[1]], audio.SampleRate));
    }
}

[Node(Category = "audio", DisplayName = "Join Audio Channels", Description = "Joins left and right mono audio channels into a stereo audio.", OutputNames = ["audio"])]
public sealed class JoinAudioChannels
{
    /// <summary>The lower sample rate is resampled to the higher one and the longer channel is trimmed to the shorter.</summary>
    public AudioClip? Execute(AudioClip? audioLeft, AudioClip? audioRight)
    {
        if (audioLeft is null || audioRight is null) return audioLeft ?? audioRight;
        if (audioLeft.ChannelCount != 1 || audioRight.ChannelCount != 1) throw new ArgumentException("AudioJoin: Both input audios must be mono.");
        var (left, right, rate) = AudioDsp.MatchSampleRates(audioLeft, audioRight);
        int length = Math.Min(left.Length, right.Length);
        return new AudioClip([left.Channels[0][..length], right.Channels[0][..length]], rate);
    }
}

[Node(Category = "audio", DisplayName = "Concatenate Audio", Description = "Concatenates the audio1 to audio2 in the specified direction.")]
public sealed class AudioConcat
{
    public AudioClip? Execute(AudioClip? audio1, AudioClip? audio2,
        [Input(Options = ["after", "before"], Tooltip = "Whether to append audio2 after or before audio1.")] string direction = "after")
    {
        if (audio1 is null || audio2 is null) return audio1 ?? audio2;
        // Mono becomes stereo by duplicating the channel, then the lower rate is resampled to the higher one.
        var (first, second, rate) = AudioDsp.MatchSampleRates(Stereo(audio1), Stereo(audio2));
        if (first.ChannelCount != second.ChannelCount)
            throw new ArgumentException($"AudioConcat: the clips have {first.ChannelCount} and {second.ChannelCount} channels and cannot be joined.");
        if (direction == "before") (first, second) = (second, first);
        else if (direction != "after") throw new ArgumentException($"Unknown direction '{direction}'.", nameof(direction));
        return new AudioClip([.. first.Channels.Select((c, i) => (float[])[.. c, .. second.Channels[i]])], rate);
    }

    static AudioClip Stereo(AudioClip clip) => clip.ChannelCount == 1 ? new AudioClip([clip.Channels[0], (float[])clip.Channels[0].Clone()], clip.SampleRate) : clip;
}

[Node(Category = "audio", DisplayName = "Merge Audio", Description = "Combine two audio tracks by overlaying their waveforms.")]
public sealed class AudioMerge
{
    /// <summary>
    /// audio2 is resampled to match, then trimmed or padded with silence to audio1's length and combined sample by sample (a mono clip is applied to every
    /// channel of the other). A result that peaks above 1 is scaled down so its loudest sample is 1.
    /// </summary>
    public AudioClip? Execute(AudioClip? audio1, AudioClip? audio2,
        [Input(Options = ["add", "mean", "subtract", "multiply"], Tooltip = "The method used to combine the audio waveforms.")] string mergeMethod = "add")
    {
        if (audio1 is null || audio2 is null) return audio1 ?? audio2;
        var (first, second, rate) = AudioDsp.MatchSampleRates(audio1, audio2);
        int length = first.Length;
        if (length == 0 || second.Length == 0) return first;
        Func<float, float, float> op = mergeMethod switch
        {
            "add" => (a, b) => a + b,
            "subtract" => (a, b) => a - b,
            "multiply" => (a, b) => a * b,
            "mean" => (a, b) => (a + b) / 2,
            _ => throw new ArgumentException($"Unknown merge method '{mergeMethod}'.", nameof(mergeMethod)),
        };
        // Broadcasting over channels: equal counts pair up, a single channel goes with each of the other's.
        int c1 = first.ChannelCount, c2 = second.ChannelCount, channels = Math.Max(c1, c2);
        if (c1 != c2 && c1 != 1 && c2 != 1) throw new ArgumentException($"AudioMerge: the clips have {c1} and {c2} channels and cannot be combined.");

        var result = new float[channels][];
        float peak = 0;
        for (int c = 0; c < channels; c++)
        {
            float[] a = first.Channels[c1 == 1 ? 0 : c], b = second.Channels[c2 == 1 ? 0 : c];
            var o = result[c] = new float[length];
            for (int i = 0; i < length; i++)
            {
                o[i] = op(a[i], i < b.Length ? b[i] : 0f);
                peak = MathF.Max(peak, MathF.Abs(o[i])); // NaN wins, as torch's max does, and then nothing is scaled
            }
        }
        if (peak > 1f)
            foreach (var o in result)
                for (int i = 0; i < o.Length; i++) o[i] /= peak;
        return new AudioClip(result, rate);
    }
}

[Node(Category = "audio", DisplayName = "Adjust Audio Volume", Description = "Adjust the volume of the audio by a specified amount in decibels (dB).")]
public sealed class AudioAdjustVolume
{
    public AudioClip? Execute(AudioClip? audio,
        [Input(Min = -100, Max = 100, Tooltip = "Volume adjustment in decibels (dB). 0 = no change, +6 = double, -6 = half, etc")] int volume = 1)
    {
        if (audio is null) return null;
        if (volume == 0) return audio;
        // 10 ** (volume / 20) in doubles; torch multiplies float32 samples by it as a float32.
        float gain = (float)Math.Pow(10, volume / 20.0);
        return new AudioClip([.. audio.Channels.Select(c => c.Select(s => s * gain).ToArray())], audio.SampleRate);
    }
}

[Node(Category = "audio", DisplayName = "Empty Audio")]
public sealed class EmptyAudio
{
    public AudioClip Execute(
        [Input(Min = 0.0, Max = 18446744073709551615.0, Step = 0.01, Tooltip = "Duration of the empty audio clip in seconds")] double duration = 60.0,
        [Input(Min = 1, Max = 192000, Tooltip = "Sample rate of the empty audio clip.")] int sampleRate = 44100,
        [Input(Min = 1, Max = 2, Tooltip = "Number of audio channels (1 for mono, 2 for stereo).")] int channels = 2)
    {
        double samples = Math.Round(duration * sampleRate);
        if (samples > Array.MaxLength) throw new ArgumentException($"Empty Audio: {duration} seconds at {sampleRate} Hz is too long.", nameof(duration));
        return new AudioClip([.. Enumerable.Range(0, channels).Select(_ => new float[(int)samples])], sampleRate);
    }
}

[Node(Category = "audio", DisplayName = "Audio Equalizer (3-Band)")]
public sealed class AudioEqualizer3Band
{
    /// <summary>
    /// A low shelf, a peaking filter and a high shelf in that order (torchaudio's bass, equalizer and treble biquads), each skipped at 0 dB.
    /// Every stage clamps its result to -1..1, as ComfyUI's does.
    /// </summary>
    public AudioClip? Execute(AudioClip? audio,
        [Input(Name = "low_gain_dB", Min = -24.0, Max = 24.0, Step = 0.1, Tooltip = "Gain for Low frequencies (Bass)")] double lowGainDb = 0.0,
        [Input(Min = 20, Max = 500, Tooltip = "Cutoff frequency for Low shelf")] int lowFreq = 100,
        [Input(Name = "mid_gain_dB", Min = -24.0, Max = 24.0, Step = 0.1, Tooltip = "Gain for Mid frequencies")] double midGainDb = 0.0,
        [Input(Min = 200, Max = 4000, Tooltip = "Center frequency for Mids")] int midFreq = 1000,
        [Input(Min = 0.1, Max = 10.0, Step = 0.1, Tooltip = "Q factor (bandwidth) for Mids")] double midQ = 0.707,
        [Input(Name = "high_gain_dB", Min = -24.0, Max = 24.0, Step = 0.1, Tooltip = "Gain for High frequencies (Treble)")] double highGainDb = 0.0,
        [Input(Min = 1000, Max = 15000, Tooltip = "Cutoff frequency for High shelf")] int highFreq = 5000)
    {
        if (audio is null) return null;
        if (audio.Length == 0) return audio;
        int rate = audio.SampleRate;
        return new AudioClip([.. audio.Channels.Select(c =>
        {
            var x = (float[])c.Clone();
            if (lowGainDb != 0) x = AudioDsp.Bass(x, rate, lowGainDb, lowFreq, 0.707);
            if (midGainDb != 0) x = AudioDsp.Equalizer(x, rate, midFreq, midGainDb, midQ);
            if (highGainDb != 0) x = AudioDsp.Treble(x, rate, highGainDb, highFreq, 0.707);
            return x;
        })], rate);
    }
}
