using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.OpenCL;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Load Audio's FLAC against ComfyUI's own reading (Fixtures/codecs/load_audio, make_load_audio.py: comfy_extras.nodes_audio.load, PyAV 18.1 / FFmpeg 8,
/// then f32_pcm): 16- and 24-bit, mono, stereo and 5.1, decoded by SpawnDev.Codecs. The floats must be ComfyUI's exactly.
/// </summary>
public class LoadAudioFlacTests
{
    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "codecs", "load_audio");
    static JsonElement Case(string name) =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "cases.json"))).RootElement.GetProperty("cases").EnumerateArray().First(c => c.GetProperty("name").GetString() == name);

    static void AssertComfyUIs(string name, AudioClip clip)
    {
        var c = Case(name);
        int channels = c.GetProperty("channels").GetInt32(), n = c.GetProperty("samples").GetInt32();
        Assert.That((clip.ChannelCount, clip.Length, clip.SampleRate), Is.EqualTo((channels, n, c.GetProperty("sample_rate").GetInt32())));
        var raw = File.ReadAllBytes(Path.Combine(Dir, name + ".f32"));
        var expected = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, expected, 0, raw.Length);
        for (int k = 0; k < channels; k++)
            for (int i = 0; i < n; i++)
                if (clip.Channels[k][i] != expected[k * n + i])
                    Assert.Fail($"{name}: channel {k} sample {i} is {clip.Channels[k][i]:R}, ComfyUI's {expected[k * n + i]:R}");
    }

    [TestCase("stereo16_44100")]
    [TestCase("mono16_22050")]
    [TestCase("stereo24_48000")]
    [TestCase("surround24_48000")]
    public async Task FLAC_reads_as_ComfyUIs_floats(string name)
    {
        using var context = Context.Create(b => b.CPU());
        using var acc = context.CreateCPUAccelerator(0);
        AssertComfyUIs(name, await CodecsAudio.DecodeFlacAsync(acc, File.ReadAllBytes(Path.Combine(Dir, name + ".flac"))));
    }

    [TestCase("stereo16_44100")]
    [TestCase("surround24_48000")]
    [Explicit("Needs an OpenCL GPU.")]
    public async Task FLAC_reads_as_ComfyUIs_floats_on_OpenCL(string name)
    {
        using var context = Context.Create(b => b.OpenCL());
        using var acc = context.Devices.First(d => d.AcceleratorType == AcceleratorType.OpenCL).CreateAccelerator(context);
        TestContext.Out.WriteLine(acc.Name);
        AssertComfyUIs(name, await CodecsAudio.DecodeFlacAsync(acc, File.ReadAllBytes(Path.Combine(Dir, name + ".flac"))));
    }

    [Test]
    public async Task Load_Audio_lists_and_reads_FLAC_through_the_hook()
    {
        var (savedStore, savedDecode) = (InputFiles.Store, AudioFiles.Decode);
        using var context = Context.Create(b => b.CPU());
        using var acc = context.CreateCPUAccelerator(0);
        try
        {
            var inputs = new MemoryInputStore();
            await inputs.SaveAsync("song.flac", File.ReadAllBytes(Path.Combine(Dir, "stereo24_48000.flac")));
            await inputs.SaveAsync("notes.txt", [1]);
            InputFiles.Store = inputs;
            CodecsAudio.Install(() => Task.FromResult<Accelerator>(acc));
            Assert.That(LoadAudio.AudioNames(), Is.EqualTo(new[] { "song.flac" }));
            AssertComfyUIs("stereo24_48000", await new LoadAudio().ExecuteAsync("song.flac"));

            AudioFiles.Decode = null;
            Assert.That(async () => await new LoadAudio().ExecuteAsync("song.flac"), Throws.TypeOf<NotSupportedException>().With.Message.Contains("SpawnDev.Codecs"));
        }
        finally { (InputFiles.Store, AudioFiles.Decode) = (savedStore, savedDecode); }
    }
}
