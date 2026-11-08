using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.Codecs.Audio.Opus;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// FLAC for Save Audio and Preview Audio, encoded by SpawnDev.Codecs (a port of FFmpeg's flacenc and FLAC muxer) on the host's accelerator:
/// <see cref="Install"/> sets <see cref="AudioFiles.Flac"/>. The file is what ComfyUI's PyAV writes: 16-bit samples at FFmpeg's default compression level 5,
/// the vendor string and trailing <c>encoder</c> comment of the libavformat ComfyUI ships with, and an 8192-byte padding block.
/// </summary>
public static class CodecsAudio
{
    /// <summary>libavformat's version string in the files ComfyUI's PyAV 18.1 (FFmpeg 8) writes, as vendor and as the muxer's <c>encoder</c> comment.</summary>
    public const string Lavf = "Lavf62.12.102";

    static readonly SemaphoreSlim Gate = new(1, 1);
    static FlacEncoderGpu? _encoder;
    static Accelerator? _accelerator;

    /// <summary>Lets the audio save nodes write FLAC on the accelerator <paramref name="accelerator"/> gives (asked for on the first save).</summary>
    public static void Install(Func<Task<Accelerator>> accelerator)
    {
        AudioFiles.Flac = async (clip, metadata) => await EncodeFlacAsync(await accelerator(), clip, metadata);
        AudioFiles.Opus = async (clip, bitrate, metadata) => await EncodeOpusAsync(await accelerator(), clip, bitrate, metadata);
        AudioFiles.Decode = async data => IsFlac(data) ? await DecodeFlacAsync(await accelerator(), data) : null;
    }

    /// <summary>A FLAC stream, possibly after an ID3v2 tag.</summary>
    public static bool IsFlac(byte[] data) => data.Length >= 4 && (data.AsSpan(0, 4).SequenceEqual("fLaC"u8) || data.AsSpan(0, 3).SequenceEqual("ID3"u8));

    /// <summary>
    /// A FLAC file as ComfyUI's Load Audio reads it: FFmpeg's decoder gives int16 for up to 16 bits and int32 above (the samples shifted up to fill the type),
    /// and f32_pcm turns them to float32 and divides by 2^15 or 2^31. SpawnDev.Codecs does that conversion on the accelerator (DecodeFloatAsync).
    /// </summary>
    public static async Task<AudioClip> DecodeFlacAsync(Accelerator accelerator, byte[] data)
    {
        FlacDecodeResult result;
        await Gate.WaitAsync();
        try
        {
            using var decoder = new FlacDecoderGpu(accelerator);
            result = await decoder.DecodeFloatAsync(data);
        }
        finally { Gate.Release(); }
        int c = result.Channels;
        var planar = result.PlanarFloat;
        int n = planar.Length / c;
        return new AudioClip([.. Enumerable.Range(0, c).Select(k => planar.AsSpan(k * n, n).ToArray())], result.SampleRateHz);
    }

    /// <summary>
    /// Ogg Opus as ComfyUI's PyAV writes it around the packets: FFmpeg's Ogg muxer, OpusTags vendor "Lavf62.12.102" with the comments and a trailing
    /// <c>encoder</c>, a random stream serial (as FFmpeg's). The samples go in as floats; the encoder rounds them to 16 bits as FFmpeg's swresample does.
    /// The packets come from SpawnDev.Codecs' port of libopus' fixed-point build (ComfyUI's PyAV ships the float build), so they are valid but not
    /// byte-identical. <paramref name="serial"/> fixes the serial for tests.
    /// </summary>
    public static async Task<byte[]> EncodeOpusAsync(Accelerator accelerator, AudioClip clip, int? bitrate, IReadOnlyList<KeyValuePair<string, string>> metadata, uint? serial = null)
    {
        int n = clip.Length, c = clip.ChannelCount;
        var interleaved = new float[n * c];
        for (int f = 0; f < n; f++)
            for (int k = 0; k < c; k++) interleaved[f * c + k] = clip.Channels[k][f];
        var options = new OggOpusEncodeOptions
        {
            Bitrate = bitrate,
            Vendor = Lavf,
            Tags = [.. metadata, new("encoder", Lavf)],
            Serial = serial ?? (uint)Random.Shared.NextInt64(0, 1L << 32),
        };
        await Gate.WaitAsync();
        try { return await OggOpusEncoder.EncodeAsync(accelerator, interleaved, clip.SampleRate, c, options); }
        finally { Gate.Release(); }
    }

    public static async Task<byte[]> EncodeFlacAsync(Accelerator accelerator, AudioClip clip, IReadOnlyList<KeyValuePair<string, string>> metadata)
    {
        var samples = AudioFiles.Samples16(clip);
        var options = new FlacEncodeOptions
        {
            Vendor = Lavf,
            Tags = [.. metadata, new("encoder", Lavf)],
        };
        await Gate.WaitAsync();
        try
        {
            if (_encoder is null || _accelerator != accelerator)
            {
                _encoder?.Dispose();
                _encoder = new FlacEncoderGpu(accelerator);
                _accelerator = accelerator;
            }
            return await _encoder.EncodeAsync(samples, clip.ChannelCount, clip.SampleRate, 16, options);
        }
        finally { Gate.Release(); }
    }
}
