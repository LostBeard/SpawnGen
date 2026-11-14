using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Flac;
using SpawnDev.Codecs.Audio.Mp3;
using SpawnDev.Codecs.Audio.Opus;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// FLAC for Save Audio and Preview Audio, encoded by SpawnDev.Codecs (its own MIT FLAC encoder) on the host's accelerator:
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
        AudioFiles.Mp3 = async (clip, quality, metadata) => await EncodeMp3Async(await accelerator(), clip, quality, metadata);
        AudioFiles.Decode = async data => IsFlac(data) ? await DecodeFlacAsync(await accelerator(), data) : null;
    }

    /// <summary>What <see cref="WarmUpAsync"/> can prepare.</summary>
    [Flags]
    public enum Work { None = 0, FlacEncode = 1, FlacDecode = 2, Opus = 4, Mp3 = 8 }

    /// <summary>
    /// Compiles the kernels of the audio work a graph is about to need (SpawnDev.Codecs' warm-ups: a short silent encode or decode, thrown away), so the
    /// first real save or load does not wait for them. In the browser the first compile of the Opus stages takes minutes on some GPUs. Not called
    /// automatically yet: in the browser the compile holds the page's only thread (measured: 30 s + 104 s without an answer on the RX 590, non-AOT build),
    /// so warming up early would freeze the editor instead of the run.
    /// </summary>
    public static async Task WarmUpAsync(Accelerator accelerator, Work work)
    {
        await Gate.WaitAsync();
        try
        {
            if (work.HasFlag(Work.FlacEncode))
            {
                if (_encoder is null || _accelerator != accelerator)
                {
                    _encoder?.Dispose();
                    _encoder = new FlacEncoderGpu(accelerator);
                    _accelerator = accelerator;
                }
                await _encoder.WarmUpAsync();
            }
            if (work.HasFlag(Work.FlacDecode))
            {
                using var decoder = new FlacDecoderGpu(accelerator);
                await decoder.WarmUpAsync();
            }
            if (work.HasFlag(Work.Opus))
                await OggOpusEncoder.WarmUpAsync(accelerator);
            if (work.HasFlag(Work.Mp3))
                await Mp3Encoder(accelerator).WarmUpAsync();
        }
        finally { Gate.Release(); }
    }

    /// <summary>The audio work a node type needs (Save / Preview Audio: FLAC; the Opus and Advanced saves: Opus too; Load Audio: FLAC decoding).</summary>
    public static Work WorkFor(string classType) => classType switch
    {
        "SaveAudio" or "PreviewAudio" => Work.FlacEncode,
        "SaveAudioOpus" => Work.Opus,
        "SaveAudioMP3" => Work.Mp3,
        "SaveAudioAdvanced" => Work.FlacEncode | Work.Opus | Work.Mp3,
        "LoadAudio" => Work.FlacDecode,
        _ => Work.None,
    };

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

    static Mp3EncoderGpu? _mp3;
    static Accelerator? _mp3Accelerator;

    /// <summary>The MP3 encoder for <paramref name="accelerator"/> (kept between saves; call inside <see cref="Gate"/>).</summary>
    static Mp3EncoderGpu Mp3Encoder(Accelerator accelerator)
    {
        if (_mp3 is null || _mp3Accelerator != accelerator)
        {
            _mp3?.Dispose();
            _mp3 = new Mp3EncoderGpu(accelerator);
            _mp3Accelerator = accelerator;
        }
        return _mp3;
    }

    /// <summary>
    /// MP3 as ComfyUI's PyAV (libmp3lame) writes it, measured 2026-11-13 (Fixtures/codecs/save_audio_mp3): an ID3v2.4 tag with the comments as TXXX
    /// frames and TSSE "Lavf62.12.102", an Info (CBR) or Xing (VBR) frame with the "Lavf" encoder string, joint stereo. "V0" is variable bit rate at
    /// quality 0; "128k" and "320k" are constant, at most 160 kbps at the MPEG-2 rates (below 32 kHz), as libmp3lame clamps them. The frames come from
    /// SpawnDev.Codecs' own encoder (not LAME), so they are valid but not byte-identical.
    /// </summary>
    public static async Task<byte[]> EncodeMp3Async(Accelerator accelerator, AudioClip clip, string quality, IReadOnlyList<KeyValuePair<string, string>> metadata)
    {
        int n = clip.Length, c = clip.ChannelCount;
        var interleaved = new float[n * c];
        for (int f = 0; f < n; f++)
            for (int k = 0; k < c; k++) interleaved[f * c + k] = clip.Channels[k][f];
        var options = new Mp3EncodeOptions { Tags = [.. metadata], Id3Encoder = Lavf };
        options = quality switch
        {
            "V0" => options with { VbrQuality = 0 },
            "128k" or "320k" => options with { BitrateKbps = Math.Min(int.Parse(quality.TrimEnd('k')), clip.SampleRate < 32000 ? 160 : 320) },
            _ => throw new ArgumentException($"Unknown MP3 quality '{quality}'.", nameof(quality)),
        };
        await Gate.WaitAsync();
        try { return await Mp3Encoder(accelerator).EncodeAsync(interleaved, c, clip.SampleRate, options); }
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
