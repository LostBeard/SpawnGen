using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Types;

/// <summary>Sound moving between nodes (ComfyUI's AUDIO): one array of samples in -1..1 per channel, all the same length, at one sample rate.</summary>
[Socket("AUDIO")]
public sealed class AudioClip
{
    public AudioClip(float[][] channels, int sampleRate)
    {
        if (channels.Length == 0) throw new ArgumentException("A clip needs at least one channel.", nameof(channels));
        if (channels.Any(c => c.Length != channels[0].Length)) throw new ArgumentException("All channels must have the same length.", nameof(channels));
        if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));
        Channels = channels;
        SampleRate = sampleRate;
    }

    public float[][] Channels { get; }
    public int SampleRate { get; }
    public int ChannelCount => Channels.Length;
    /// <summary>Samples per channel.</summary>
    public int Length => Channels[0].Length;
    public double Seconds => (double)Length / SampleRate;

    public static AudioClip Mono(float[] samples, int sampleRate) => new([samples], sampleRate);
}
