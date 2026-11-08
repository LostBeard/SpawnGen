using ILGPU.Runtime;
using SpawnDev.Codecs.Audio.Flac;
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
    public static void Install(Func<Task<Accelerator>> accelerator) =>
        AudioFiles.Flac = async (clip, metadata) => await EncodeFlacAsync(await accelerator(), clip, metadata);

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
