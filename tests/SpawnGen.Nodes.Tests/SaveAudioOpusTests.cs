using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Save Audio (Opus) against ComfyUI's own saving (Fixtures/codecs/save_audio, make_save_audio.py). ComfyUI resamples to an Opus rate with
/// comfy.audio.resample and encodes with PyAV's libopus (the float build); SpawnDev.Codecs ports libopus' fixed-point build, so the audio packets differ.
/// Everything around them is ComfyUI's: the Ogg pages' layout and flags, the final granule position, and the OpusHead and OpusTags pages byte for byte.
/// </summary>
public class SaveAudioOpusTests
{
    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "codecs", "save_audio");
    static JsonElement Cases => JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "cases.json"))).RootElement;

    static float[] F32(string file)
    {
        var raw = File.ReadAllBytes(Path.Combine(Dir, file));
        var f = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, f, 0, raw.Length);
        return f;
    }

    static AudioClip Clip(string name)
    {
        var c = Cases.GetProperty("cases").EnumerateArray().First(c => c.GetProperty("name").GetString() == name);
        int channels = c.GetProperty("channels").GetInt32(), n = c.GetProperty("samples").GetInt32();
        var all = F32(name + ".f32");
        return new AudioClip([.. Enumerable.Range(0, channels).Select(k => all[(k * n)..((k + 1) * n)])], c.GetProperty("sample_rate").GetInt32());
    }

    sealed record Page(byte Flags, long Granule, uint Serial, uint Sequence, byte[] Bytes);

    static List<Page> Pages(byte[] file)
    {
        var pages = new List<Page>();
        for (int i = 0; i < file.Length;)
        {
            Assert.That(file.AsSpan(i, 4).SequenceEqual("OggS"u8), $"page at {i}");
            int segments = file[i + 26], body = 0;
            for (int s = 0; s < segments; s++) body += file[i + 27 + s];
            int length = 27 + segments + body;
            pages.Add(new(file[i + 5], BitConverter.ToInt64(file, i + 6), BitConverter.ToUInt32(file, i + 14), BitConverter.ToUInt32(file, i + 18), file[i..(i + length)]));
            i += length;
        }
        return pages;
    }

    [TestCase(44100, 48000)]
    [TestCase(22050, 24000)]
    [TestCase(96000, 48000)]
    [TestCase(48000, 48000)]
    [TestCase(8000, 8000)]
    [TestCase(11025, 12000)]
    [TestCase(32000, 48000)]
    [TestCase(4000, 8000)]
    public void The_Opus_rate_is_ComfyUIs(int rate, int expected) => Assert.That(AudioFiles.OpusRate(rate), Is.EqualTo(expected));

    [TestCase("stereo_44100")]
    [TestCase("mono_22050_odd")]
    public void Resampling_to_the_Opus_rate_is_comfy_audio_resample(string name)
    {
        var o = Cases.GetProperty("opus").EnumerateArray().First(o => o.GetProperty("name").GetString() == name);
        var ours = AudioDsp.Resample(Clip(name), o.GetProperty("rate").GetInt32());
        var theirs = F32(o.GetProperty("resampled").GetString()!);
        int n = ours.Length;
        Assert.That(n * ours.ChannelCount, Is.EqualTo(theirs.Length));
        double worst = 0;
        for (int k = 0; k < ours.ChannelCount; k++)
            for (int i = 0; i < n; i++) worst = Math.Max(worst, Math.Abs(ours.Channels[k][i] - theirs[k * n + i]));
        Assert.That(worst, Is.LessThan(1e-5), "largest difference");
    }

    [TestCase("stereo_44100")]
    [TestCase("mono_22050_odd")]
    public async Task The_Ogg_pages_around_the_packets_are_ComfyUIs(string name)
    {
        var o = Cases.GetProperty("opus").EnumerateArray().First(o => o.GetProperty("name").GetString() == name);
        var theirs = Pages(File.ReadAllBytes(Path.Combine(Dir, o.GetProperty("file").GetString()!)));
        var all = Cases;
        var metadata = new List<KeyValuePair<string, string>> { new("prompt", all.GetProperty("prompt").GetString()!) };
        foreach (var e in all.GetProperty("extra").EnumerateObject()) metadata.Add(new(e.Name, e.Value.GetString()!));
        int bitrate = o.GetProperty("quality").GetString() switch { "64k" => 64000, "128k" => 128000, var q => throw new InvalidOperationException(q) };

        using var context = Context.Create(b => b.CPU());
        using var acc = context.CreateCPUAccelerator(0);
        var clip = AudioDsp.Resample(Clip(name), AudioFiles.OpusRate(Clip(name).SampleRate));
        var ours = Pages(await CodecsAudio.EncodeOpusAsync(acc, clip, bitrate, metadata, serial: theirs[0].Serial));

        Assert.That(ours[0].Bytes, Is.EqualTo(theirs[0].Bytes), "OpusHead page");
        Assert.That(ours[1].Bytes, Is.EqualTo(theirs[1].Bytes), "OpusTags page (vendor, prompt, workflow, encoder)");
        Assert.That(ours.Select(p => (p.Flags, p.Sequence)), Is.EqualTo(theirs.Select(p => (p.Flags, p.Sequence))), "pages and their flags");
        Assert.That(ours[^1].Granule, Is.EqualTo(theirs[^1].Granule), "final granule position (samples + pre-skip)");
        TestContext.Out.WriteLine($"{name}: {ours.Sum(p => p.Bytes.Length)} bytes vs ComfyUI's {theirs.Sum(p => p.Bytes.Length)}");
    }

    [TestCase("opus", "64k", true)]
    [TestCase("opus", "320k", true)]
    [TestCase("mp3", "V0", true)]
    [TestCase("mp3", "64k", false)]
    [TestCase("opus", "V0", false)]
    public void Save_Audio_Advanced_checks_quality_against_the_chosen_format(string format, string quality, bool valid)
    {
        // mp3 and opus both bring "format.quality" with their own choices (the prompt from ComfyUI's editor sends the chosen one's).
        var registry = new SpawnGen.Engine.Nodes.NodeRegistry().AddAssembly(typeof(SaveAudio).Assembly);
        var g = new SpawnGen.Engine.Graph.PromptGraph();
        g.Add("1", "LoadAudio", ("audio", "x.wav"));
        g.Add("2", "SaveAudioAdvanced", ("audio", SpawnGen.Engine.Graph.PromptInput.Link("1")), ("filename_prefix", "audio/ComfyUI"), ("format", format), ("format.quality", quality));
        var errors = SpawnGen.Engine.Execution.PromptValidator.Validate(registry, g).Errors.Where(e => e.Input == "format.quality").ToList();
        Assert.That(errors, valid ? Is.Empty : Is.Not.Empty, string.Join(" | ", errors.Select(e => e.Message)));
    }
}
