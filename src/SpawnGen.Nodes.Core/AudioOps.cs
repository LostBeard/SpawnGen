using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries (names, widgets, tooltips, descriptions) and black-box measurements of its
// audio nodes (tools/parity/make_audio_fixtures.py -> Fixtures/comfy/comfy_audio_nodes.json.gz); the DSP is AudioDsp (torchaudio's, BSD-2-Clause).
// SpawnGen's AUDIO is one clip (no batch dimension). A missing (null) input passes the other one on, where ComfyUI would fail.

/// <summary>Record Audio: the recording the editor uploaded, by its file name (an " [input]" / " [temp]" annotation and any folder are dropped).</summary>
[Node(Category = "audio", DisplayName = "Record Audio")]
public sealed class RecordAudio
{
    public async Task<AudioClip> ExecuteAsync(string audio)
    {
        if (string.IsNullOrWhiteSpace(audio)) throw new ArgumentException("Record Audio has no recording yet: record one in the node first.", nameof(audio));
        var name = StripAnnotation(audio).Replace('\\', '/');
        return WavFile.Decode(await InputFiles.Store.ReadAsync(name[(name.LastIndexOf('/') + 1)..]));
    }

    /// <summary>A file name without ComfyUI's trailing " [input]", " [output]" or " [temp]" annotation.</summary>
    internal static string StripAnnotation(string name)
    {
        name = name.Trim();
        foreach (var tag in (string[])[" [input]", " [output]", " [temp]"])
            if (name.EndsWith(tag, StringComparison.Ordinal)) return name[..^tag.Length].Trim();
        return name;
    }
}

/// <summary>
/// Trim Audio Duration, as measured: start = round(start_index * rate) (ties to even), counted from the end when negative and then clamped to 0;
/// end = start + round(duration * rate), at most the length. An empty range, or a start at or past the end, is ComfyUI's ValueError. An empty clip passes.
/// </summary>
[Node(Category = "audio", DisplayName = "Trim Audio Duration", Description = "Trim audio tensor into chosen time range.")]
public sealed class TrimAudioDuration
{
    public AudioClip? Execute(AudioClip? audio,
        [Input(Min = -1.8446744073709552E19, Max = 1.8446744073709552E19, Step = 0.01,
            Tooltip = "Start time in seconds, can be negative to count from the end (supports sub-seconds).")] double startIndex = 0.0,
        [Input(Min = 0.0, Step = 0.01, Tooltip = "Duration in seconds")] double duration = 60.0)
    {
        if (audio is null || audio.Length == 0) return audio;
        long length = audio.Length;
        long start = PyValues.Round(startIndex * audio.SampleRate);
        if (start < 0) start = Math.Max(0, length + start);
        long end = Math.Min(start + PyValues.Round(duration * audio.SampleRate), length);
        if (start >= end || start >= length)
            throw new PyException("ValueError", "TrimAudioDuration: Start time must be less than end time and be within the audio length.");
        return new AudioClip([.. audio.Channels.Select(c => c[(int)start..(int)end])], audio.SampleRate);
    }
}

/// <summary>Split Audio Channels: a stereo clip as two mono clips; anything but 2 channels is ComfyUI's ValueError.</summary>
[Node(Category = "audio", DisplayName = "Split Audio Channels", Description = "Separates the audio into left and right channels.")]
public sealed class SplitAudioChannels
{
    public (AudioClip? Left, AudioClip? Right) Execute(AudioClip? audio)
    {
        if (audio is null) return (null, null);
        if (audio.ChannelCount != 2)
            throw new PyException("ValueError", $"AudioSplit: Input audio must be stereo (2 channels), got {audio.ChannelCount} channel(s).");
        return (AudioClip.Mono((float[])audio.Channels[0].Clone(), audio.SampleRate), AudioClip.Mono((float[])audio.Channels[1].Clone(), audio.SampleRate));
    }
}

/// <summary>
/// Join Audio Channels: two mono clips as one stereo clip. The one at the lower rate is resampled up first; both are cut to the shorter length.
/// </summary>
[Node(Category = "audio", DisplayName = "Join Audio Channels", Description = "Joins left and right mono audio channels into a stereo audio.", OutputNames = ["audio"])]
public sealed class JoinAudioChannels
{
    public AudioClip? Execute(AudioClip? audioLeft, AudioClip? audioRight)
    {
        if (audioLeft is null || audioRight is null) return audioLeft ?? audioRight;
        if (audioLeft.ChannelCount != 1 || audioRight.ChannelCount != 1) throw new PyException("ValueError", "AudioJoin: Both input audios must be mono.");
        var (left, right, rate) = AudioDsp.MatchSampleRates(audioLeft, audioRight);
        int n = Math.Min(left.Length, right.Length);
        return new AudioClip([left.Channels[0][..n], right.Channels[0][..n]], rate);
    }
}

/// <summary>
/// Concatenate Audio: mono clips become stereo (the channel twice), the lower rate is resampled up, and audio2 goes after (or before) audio1. Clips of
/// different channel counts cannot be joined.
/// </summary>
[Node(Category = "audio", DisplayName = "Concatenate Audio", Description = "Concatenates the audio1 to audio2 in the specified direction.")]
public sealed class AudioConcat
{
    public AudioClip? Execute(AudioClip? audio1, AudioClip? audio2,
        [Input(Options = ["after", "before"], Tooltip = "Whether to append audio2 after or before audio1.")] string direction = "after")
    {
        if (audio1 is null || audio2 is null) return audio1 ?? audio2;
        var (a, b, rate) = AudioDsp.MatchSampleRates(Stereo(audio1), Stereo(audio2));
        if (a.ChannelCount != b.ChannelCount)
            throw new ArgumentException($"Cannot concatenate audio of {a.ChannelCount} and {b.ChannelCount} channels.");
        var (first, second) = direction == "before" ? (b, a) : (a, b);
        return new AudioClip([.. first.Channels.Select((c, i) => (float[])[.. c, .. second.Channels[i]])], rate);
    }

    static AudioClip Stereo(AudioClip clip) => clip.ChannelCount == 1 ? new AudioClip([clip.Channels[0], clip.Channels[0]], clip.SampleRate) : clip;
}

/// <summary>
/// Merge Audio: the lower rate is resampled up; a mono clip goes with each channel of a stereo one; audio2 is cut or padded with silence to audio1's
/// length; the waveforms are added, averaged, subtracted (audio1 - audio2) or multiplied, and divided by the loudest sample when it exceeds 1. An
/// empty audio1 passes.
/// </summary>
[Node(Category = "audio", DisplayName = "Merge Audio", Description = "Combine two audio tracks by overlaying their waveforms.")]
public sealed class AudioMerge
{
    public AudioClip? Execute(AudioClip? audio1, AudioClip? audio2,
        [Input(Options = ["add", "mean", "subtract", "multiply"], Tooltip = "The method used to combine the audio waveforms.")] string mergeMethod = "add")
    {
        if (audio1 is null || audio2 is null) return audio1 ?? audio2;
        if (audio1.Length == 0) return audio1;
        Func<float, float, float> op = mergeMethod switch
        {
            "add" => (x, y) => x + y,
            "mean" => (x, y) => (x + y) / 2f,
            "subtract" => (x, y) => x - y,
            "multiply" => (x, y) => x * y,
            _ => throw new ArgumentException($"Unknown merge method '{mergeMethod}'.", nameof(mergeMethod)),
        };
        var (a, b, rate) = AudioDsp.MatchSampleRates(audio1, audio2);
        int channels = Math.Max(a.ChannelCount, b.ChannelCount), n = a.Length;
        float[] Channel(AudioClip clip, int ch) => clip.Channels[clip.ChannelCount == 1 ? 0 : ch];
        var merged = new float[channels][];
        float peak = 0;
        for (int ch = 0; ch < channels; ch++)
        {
            float[] x = Channel(a, ch), y = Channel(b, ch), m = new float[n];
            for (int i = 0; i < n; i++)
            {
                m[i] = op(x[i], i < y.Length ? y[i] : 0f);
                peak = Math.Max(peak, Math.Abs(m[i]));
            }
            merged[ch] = m;
        }
        if (peak > 1f)
            foreach (var m in merged)
                for (int i = 0; i < n; i++) m[i] /= peak;
        return new AudioClip(merged, rate);
    }
}

/// <summary>Adjust Audio Volume: every sample times 10^(volume / 20) (in float32); 0 dB passes the clip as it is.</summary>
[Node(Category = "audio", DisplayName = "Adjust Audio Volume", Description = "Adjust the volume of the audio by a specified amount in decibels (dB).")]
public sealed class AudioAdjustVolume
{
    public AudioClip? Execute(AudioClip? audio,
        [Input(Min = -100, Max = 100, Tooltip = "Volume adjustment in decibels (dB). 0 = no change, +6 = double, -6 = half, etc")] int volume = 1)
    {
        if (audio is null || volume == 0) return audio;
        float gain = (float)Math.Pow(10, volume / 20.0);
        return new AudioClip([.. audio.Channels.Select(c => c.Select(v => v * gain).ToArray())], audio.SampleRate);
    }
}

/// <summary>Empty Audio: silence of round(duration * sample_rate) samples (ties to even).</summary>
[Node(Category = "audio", DisplayName = "Empty Audio")]
public sealed class EmptyAudio
{
    public AudioClip Execute(
        [Input(Min = 0.0, Max = 1.8446744073709552E19, Step = 0.01, Tooltip = "Duration of the empty audio clip in seconds")] double duration = 60.0,
        [Input(Min = 1, Max = 192000, Tooltip = "Sample rate of the empty audio clip.")] int sampleRate = 44100,
        [Input(Min = 1, Max = 2, Tooltip = "Number of audio channels (1 for mono, 2 for stereo).")] int channels = 2)
    {
        int n = (int)PyValues.Round(duration * sampleRate);
        return new AudioClip([.. Enumerable.Range(0, channels).Select(_ => new float[n])], sampleRate);
    }
}

/// <summary>
/// Audio Equalizer (3-Band): per channel, a low shelf (Q 0.707), a peaking filter and a high shelf (Q 0.707), each skipped at 0 dB; every filter clamps
/// its output to -1..1, as torchaudio's lfilter does.
/// </summary>
[Node(Category = "audio", DisplayName = "Audio Equalizer (3-Band)")]
public sealed class AudioEqualizer3Band
{
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
        var channels = audio.Channels.Select(c =>
        {
            var w = (float[])c.Clone();
            if (lowGainDb != 0) w = AudioDsp.Bass(w, audio.SampleRate, lowGainDb, lowFreq);
            if (midGainDb != 0) w = AudioDsp.Equalizer(w, audio.SampleRate, midFreq, midGainDb, midQ);
            if (highGainDb != 0) w = AudioDsp.Treble(w, audio.SampleRate, highGainDb, highFreq);
            return w;
        }).ToArray();
        return new AudioClip(channels, audio.SampleRate);
    }
}
