using System.Buffers.Binary;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

public class AudioTests
{
    static byte[] Riff(ushort format, ushort channels, int rate, ushort bits, byte[] data, bool extraChunk = false)
    {
        using var m = new MemoryStream();
        using var w = new BinaryWriter(m);
        w.Write("RIFF"u8); w.Write(0); w.Write("WAVE"u8);
        if (extraChunk) { w.Write("LIST"u8); w.Write(5); w.Write(new byte[] { 1, 2, 3, 4, 5 }); w.Write((byte)0); } // odd-sized chunks are padded
        w.Write("fmt "u8); w.Write(16); w.Write(format); w.Write(channels); w.Write(rate); w.Write(rate * channels * bits / 8); w.Write((ushort)(channels * bits / 8)); w.Write(bits);
        w.Write("data"u8); w.Write(data.Length); w.Write(data);
        return m.ToArray();
    }

    [Test]
    public void A_clip_round_trips_through_16_bit_WAV_within_one_step()
    {
        var left = Enumerable.Range(0, 100).Select(i => MathF.Sin(i / 7f) * 0.8f).ToArray();
        var right = Enumerable.Range(0, 100).Select(i => MathF.Cos(i / 5f) * 0.5f).ToArray();
        var clip = new AudioClip([left, right], 22050);
        var back = WavFile.Decode(WavFile.Encode(clip));
        Assert.That((back.ChannelCount, back.Length, back.SampleRate), Is.EqualTo((2, 100, 22050)));
        for (int i = 0; i < 100; i++)
        {
            Assert.That(back.Channels[0][i], Is.EqualTo(left[i]).Within(1 / 32767f));
            Assert.That(back.Channels[1][i], Is.EqualTo(right[i]).Within(1 / 32767f));
        }
        Assert.That(clip.Seconds, Is.EqualTo(100 / 22050.0).Within(1e-9));
    }

    [Test]
    public void Out_of_range_samples_are_clipped_when_writing()
    {
        var back = WavFile.Decode(WavFile.Encode(AudioClip.Mono([2f, -3f, 0.5f], 8000)));
        Assert.That(back.Channels[0][0], Is.EqualTo(1f).Within(1e-4));
        Assert.That(back.Channels[0][1], Is.EqualTo(-1f).Within(1e-4));
    }

    [Test]
    public void Other_sample_formats_decode_to_the_same_scale()
    {
        // 8 bit unsigned: 128 is silence.
        var eight = WavFile.Decode(Riff(1, 1, 8000, 8, [128, 255, 0]));
        Assert.That(eight.Channels[0], Is.EqualTo(new[] { 0f, 127 / 128f, -1f }).Within(1e-6));
        // 24 bit signed little endian: 0x400000 is half scale, -0x400000 is minus half.
        var twentyFour = WavFile.Decode(Riff(1, 1, 8000, 24, [0x00, 0x00, 0x40, 0x00, 0x00, 0xC0]));
        Assert.That(twentyFour.Channels[0], Is.EqualTo(new[] { 0.5f, -0.5f }).Within(1e-6));
        // 32 bit float.
        var floats = new byte[8];
        BinaryPrimitives.WriteSingleLittleEndian(floats, 0.25f); BinaryPrimitives.WriteSingleLittleEndian(floats.AsSpan(4), -0.75f);
        Assert.That(WavFile.Decode(Riff(3, 1, 8000, 32, floats)).Channels[0], Is.EqualTo(new[] { 0.25f, -0.75f }));
        // Stereo frames are interleaved; chunks other than fmt/data are skipped (including a padded odd-sized one).
        var stereo = WavFile.Decode(Riff(1, 2, 8000, 16, [0x00, 0x40, 0x00, 0xC0, 0x00, 0x20, 0x00, 0xE0], extraChunk: true));
        Assert.That(stereo.Channels[0], Is.EqualTo(new[] { 0.5f, 0.25f }).Within(1e-6));
        Assert.That(stereo.Channels[1], Is.EqualTo(new[] { -0.5f, -0.25f }).Within(1e-6));
    }

    [Test]
    public void Files_that_are_not_readable_WAV_say_so()
    {
        Assert.That(() => WavFile.Decode([1, 2, 3]), Throws.TypeOf<InvalidDataException>());
        Assert.That(() => WavFile.Decode(Riff(2, 1, 8000, 4, [0, 0])), Throws.TypeOf<NotSupportedException>(), "ADPCM");
        var noData = Riff(1, 1, 8000, 16, []);
        Assert.That(WavFile.Decode(noData).Length, Is.EqualTo(0));
    }

    [Test]
    public void A_clip_needs_equal_channels_and_a_rate()
    {
        Assert.That(() => new AudioClip([[1f], [1f, 2f]], 8000), Throws.ArgumentException);
        Assert.That(() => new AudioClip([], 8000), Throws.ArgumentException);
        Assert.That(() => new AudioClip([[1f]], 0), Throws.InstanceOf<ArgumentOutOfRangeException>());
    }

    [Test]
    public async Task Save_Audio_writes_numbered_wav_files_and_Load_Audio_reads_them_back()
    {
        var dir = Path.Combine(Path.GetTempPath(), "spawngen-audio-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(dir, "input"));
        InputFiles.Directory = Path.Combine(dir, "input");
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly);
        var options = new OutputOptions { Directory = Path.Combine(dir, "output"), TempDirectory = Path.Combine(dir, "temp") };

        // A source clip: write a WAV into the input folder, load it, save it twice and preview it.
        File.WriteAllBytes(Path.Combine(dir, "input", "tone.wav"), WavFile.Encode(AudioClip.Mono(Enumerable.Range(0, 800).Select(i => MathF.Sin(i * 0.2f) * 0.5f).ToArray(), 8000)));
        var g = new PromptGraph();
        g.Add("1", "LoadAudio", ("audio", "tone.wav"));
        g.Add("2", "SaveAudio", ("audio", PromptInput.Link("1")), ("filename_prefix", "clip"));
        g.Add("3", "PreviewAudio", ("audio", PromptInput.Link("1")));
        g.Add("4", "AudioInfo", ("audio", PromptInput.Link("1")));
        var services = new TestServices(options);
        var executor = new PromptExecutor(registry, services);
        var result = await executor.ExecuteAsync(g);
        var saved = (object[])result.Outputs["2"].Ui!["audio"]!;
        Assert.That(((ResultFile)saved[0]).Filename, Is.EqualTo("clip_00001_.wav"));
        Assert.That(File.Exists(Path.Combine(dir, "output", "clip_00001_.wav")), Is.True);
        Assert.That(((ResultFile)((object[])result.Outputs["3"].Ui!["audio"]!)[0]).Type, Is.EqualTo("temp"));

        // A second run with another picture of the graph numbers on: change the prefix to get a fresh cache key.
        var again = await new PromptExecutor(registry, services).ExecuteAsync(g);
        Assert.That(((ResultFile)((object[])again.Outputs["2"].Ui!["audio"]!)[0]).Filename, Is.EqualTo("clip_00002_.wav"), "never overwritten");
        Assert.That(LoadAudio.AudioNames(), Is.EqualTo(new[] { "tone.wav" }));
        Assert.That(LoadImage.ImageNames(), Is.Empty, "pictures and sound are listed apart");

        var saved1 = WavFile.Decode(File.ReadAllBytes(Path.Combine(dir, "output", "clip_00001_.wav")));
        Assert.That((saved1.SampleRate, saved1.Length, saved1.ChannelCount), Is.EqualTo((8000, 800, 1)));
        Directory.Delete(dir, true);
    }

    [Test]
    public void The_memory_stores_keep_wav_files_next_to_pictures_and_list_them_apart()
    {
        var results = new MemoryResultStore();
        var a = results.SaveAsync("output", "x", [1, 2], "wav").Result;
        var b = results.SaveAsync("output", "x", [3], "wav").Result;
        var p = results.SaveAsync("output", "x", [4]).Result;
        Assert.That((a.Filename, b.Filename, p.Filename), Is.EqualTo(("x_00001_.wav", "x_00002_.wav", "x_00001_.png")));
        Assert.That(results.Get("output", "x_00002_.wav"), Is.EqualTo(new byte[] { 3 }));

        var inputs = new MemoryInputStore();
        inputs.SaveAsync("a.png", [1]).Wait();
        inputs.SaveAsync("b.wav", [2]).Wait();
        Assert.That(inputs.List(), Is.EqualTo(new[] { "a.png" }));
        Assert.That(inputs.List(n => n.EndsWith(".wav")), Is.EqualTo(new[] { "b.wav" }));
    }

    sealed class TestServices(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }
}
