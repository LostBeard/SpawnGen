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

    /// <summary>The clip as ComfyUI saves it (FLAC when <see cref="Flac"/> is set, else WAV), with the file extension.</summary>
    public static async Task<(byte[] Data, string Extension)> EncodeAsync(AudioClip clip, IReadOnlyDictionary<string, string> metadata) =>
        Flac is { } flac ? (await flac(clip, [.. metadata]), "flac") : (WavFile.Encode(clip), "wav");
}
