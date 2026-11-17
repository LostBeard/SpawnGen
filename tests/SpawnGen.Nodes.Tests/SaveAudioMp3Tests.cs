using System.Buffers.Binary;
using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnDev.Codecs.Audio.Mp3;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Save Audio's MP3 against ComfyUI's own saving (Fixtures/codecs/save_audio, make_save_audio.py: PyAV with libmp3lame). SpawnDev.Codecs' encoder is
/// not LAME, so the audio frames differ; everything around them is ComfyUI's: the ID3v2.4 tag byte for byte, the Info (CBR) or Xing (VBR) frame with
/// the "Lavf" encoder string, the MPEG version, rate, channel mode and constant bit rate of every frame. The sound is checked by decoding both files
/// (SpawnDev.Codecs' MP3 decoder, gapless trim from the Info frame) against the source clip.
/// </summary>
public class SaveAudioMp3Tests
{
    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "codecs", "save_audio");
    static JsonElement Cases => JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "cases.json"))).RootElement;

    static AudioClip Clip(string name)
    {
        var c = Cases.GetProperty("cases").EnumerateArray().First(c => c.GetProperty("name").GetString() == name);
        int channels = c.GetProperty("channels").GetInt32(), n = c.GetProperty("samples").GetInt32();
        var raw = File.ReadAllBytes(Path.Combine(Dir, name + ".f32"));
        var all = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, all, 0, raw.Length);
        return new AudioClip([.. Enumerable.Range(0, channels).Select(k => all[(k * n)..((k + 1) * n)])], c.GetProperty("sample_rate").GetInt32());
    }

    static List<KeyValuePair<string, string>> Metadata()
    {
        var all = Cases;
        var metadata = new List<KeyValuePair<string, string>> { new("prompt", all.GetProperty("prompt").GetString()!) };
        foreach (var e in all.GetProperty("extra").EnumerateObject()) metadata.Add(new(e.Name, e.Value.GetString()!));
        return metadata;
    }

    static readonly int[] Mpeg1Rates = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320];
    static readonly int[] Mpeg2Rates = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160];

    sealed record Frame(int Version, int Kbps, int SampleRate, int Mode, bool Crc, int Offset, int Length);

    sealed record Mp3File(byte[] Tag, List<Frame> Frames, string InfoTag, int InfoFlags, int InfoFrames, int InfoBytes, string Encoder, byte[] Bytes);

    static Mp3File Parse(byte[] d)
    {
        Assert.That(d.AsSpan(0, 3).SequenceEqual("ID3"u8), "an ID3v2 tag first");
        int tagSize = 10 + (d[6] << 21 | d[7] << 14 | d[8] << 7 | d[9]);
        var frames = new List<Frame>();
        for (int p = tagSize; p + 4 <= d.Length;)
        {
            uint h = BinaryPrimitives.ReadUInt32BigEndian(d.AsSpan(p));
            Assert.That(h >> 21, Is.EqualTo(0x7FFu), $"frame sync at {p}");
            int version = (int)(h >> 19) & 3, bri = (int)(h >> 12) & 15, sri = (int)(h >> 10) & 3, pad = (int)(h >> 9) & 1;
            int kbps = (version == 3 ? Mpeg1Rates : Mpeg2Rates)[bri];
            int rate = version switch { 3 => new[] { 44100, 48000, 32000 }[sri], 2 => new[] { 22050, 24000, 16000 }[sri], _ => new[] { 11025, 12000, 8000 }[sri] };
            int length = (version == 3 ? 144000 : 72000) * kbps / rate + pad;
            frames.Add(new(version, kbps, rate, (int)(h >> 6) & 3, ((h >> 16) & 1) == 0, p, length));
            p += length;
        }
        var first = d.AsSpan(frames[0].Offset, frames[0].Length).ToArray();
        int at = first.AsSpan().IndexOf("Xing"u8) is var x and >= 0 ? x : first.AsSpan().IndexOf("Info"u8);
        Assert.That(at, Is.GreaterThanOrEqualTo(0), "an Info / Xing frame first");
        int flags = BinaryPrimitives.ReadInt32BigEndian(first.AsSpan(at + 4));
        int count = BinaryPrimitives.ReadInt32BigEndian(first.AsSpan(at + 8)), bytes = BinaryPrimitives.ReadInt32BigEndian(first.AsSpan(at + 12));
        var encoder = System.Text.Encoding.ASCII.GetString(first, at + 8 + 4 + 4 + 100 + 4, 9).TrimEnd('\0');
        return new(d[..tagSize], frames, System.Text.Encoding.ASCII.GetString(first, at, 4), flags, count, bytes, encoder, d);
    }

    static async Task<Mp3DecodeResult> DecodeAsync(Accelerator acc, byte[] file)
    {
        using var decoder = new Mp3DecoderGpu(acc);
        return await decoder.DecodeAsync(file);
    }

    /// <summary>Signal to noise ratio (dB) of the decoded samples against the source clip, sample for sample.</summary>
    static double Snr(AudioClip source, Mp3DecodeResult decoded)
    {
        double signal = 0, noise = 0;
        int c = source.ChannelCount;
        for (int i = 0; i < source.Length; i++)
            for (int k = 0; k < c; k++)
            {
                double s = source.Channels[k][i], e = s - decoded.Interleaved[i * c + k];
                signal += s * s;
                noise += e * e;
            }
        return 10 * Math.Log10(signal / noise);
    }

    [TestCase("stereo_44100", "V0")]
    [TestCase("stereo_44100", "128k")]
    [TestCase("mono_22050_odd", "V0")]
    [TestCase("mono_22050_odd", "320k")]
    public async Task The_file_around_the_frames_is_ComfyUIs_and_it_sounds_like_the_clip(string name, string quality)
    {
        var theirs = Parse(File.ReadAllBytes(Path.Combine(Dir, $"{name}_{quality}.mp3")));
        using var context = Context.Create(b => b.CPU());
        using var acc = context.CreateCPUAccelerator(0);
        var clip = Clip(name);
        var ours = Parse(await CodecsAudio.EncodeMp3Async(acc, clip, quality, Metadata()));

        Assert.That(ours.Tag, Is.EqualTo(theirs.Tag), "ID3v2.4: TXXX prompt, TXXX workflow, TSSE Lavf62.12.102, 10 bytes of padding");
        Assert.That((ours.InfoTag, ours.InfoFlags, ours.Encoder), Is.EqualTo((theirs.InfoTag, theirs.InfoFlags, theirs.Encoder)), "Xing for V0, Info for CBR; Lavf");
        Assert.That(ours.InfoFrames, Is.EqualTo(ours.Frames.Count - 1), "the Info frame counts the audio frames");
        Assert.That(ours.InfoBytes, Is.EqualTo(ours.Bytes.Length - ours.Tag.Length), "and the bytes from the Info frame on");
        Assert.That(ours.Frames.Select(f => (f.Version, f.SampleRate, f.Crc)).Distinct(), Is.EqualTo(theirs.Frames.Select(f => (f.Version, f.SampleRate, f.Crc)).Distinct()));
        Assert.That(ours.Frames.Skip(1).Select(f => f.Mode).Distinct(), Is.EqualTo(theirs.Frames.Skip(1).Select(f => f.Mode).Distinct()), "joint stereo / mono");
        if (quality != "V0")
            Assert.That(ours.Frames.Select(f => f.Kbps).Distinct(), Is.EqualTo(theirs.Frames.Select(f => f.Kbps).Distinct()), "constant bit rate (320k is 160 below 32 kHz)");
        else
            Assert.That(ours.Frames.Skip(1).Select(f => f.Kbps).Distinct().Count(), Is.GreaterThan(1), "variable bit rate");

        var decodedOurs = await DecodeAsync(acc, ours.Bytes);
        var decodedTheirs = await DecodeAsync(acc, theirs.Bytes);
        Assert.That(decodedOurs.TotalSamplesPerChannel, Is.EqualTo(clip.Length), "the Info frame's delay and padding give back the clip's length");
        double snrOurs = Snr(clip, decodedOurs), snrTheirs = Snr(clip, decodedTheirs);
        TestContext.Out.WriteLine($"{name} {quality}: {ours.Bytes.Length} bytes (ComfyUI {theirs.Bytes.Length}), SNR {snrOurs:F1} dB (ComfyUI's file {snrTheirs:F1} dB)");
        // Measured 2026-11-13 (CPU accelerator): ours 25.3 / 19.3 / 26.9 / 47.4 dB, ComfyUI's LAME files 24.4 / 17.9 / 24.4 / 23.5 dB.
        Assert.That(snrOurs, Is.GreaterThan(snrTheirs - 3), "as close to the clip as ComfyUI's own file, or closer");
    }

    [Test]
    public async Task Rates_MP3_cannot_carry_are_refused_as_ComfyUI_refuses_them()
    {
        // Measured: ComfyUI's MP3 save of a 96 kHz clip fails in libmp3lame ("avcodec_open2 returned 22"); nothing is resampled.
        var installed = AudioFiles.Mp3;
        AudioFiles.Mp3 = (_, _, _) => Task.FromResult(Array.Empty<byte>());
        try
        {
            var clip = new AudioClip([new float[960]], 96000);
            var error = Assert.ThrowsAsync<NotSupportedException>(() => AudioFiles.EncodeAsync(clip, new Dictionary<string, string>(), "mp3", "V0"));
            Assert.That(error!.Message, Does.Contain("96000 Hz"));
            Assert.That((await AudioFiles.EncodeAsync(new AudioClip([new float[480]], 48000), new Dictionary<string, string>(), "mp3", "V0")).Extension, Is.EqualTo("mp3"));
        }
        finally { AudioFiles.Mp3 = installed; }
    }
}
