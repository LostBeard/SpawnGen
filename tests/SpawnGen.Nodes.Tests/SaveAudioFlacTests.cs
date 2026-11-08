using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.OpenCL;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Save Audio's FLAC against ComfyUI's own saving (metadata byte for byte, frames lossless; see AssertComfyUIsFlac) (Fixtures/codecs/save_audio, make_save_audio.py: AudioSaveHelper.save_audio, PyAV 18.1 / FFmpeg 8:
/// 16-bit samples, compression level 5, vendor and encoder comment "Lavf62.12.102", 8192 bytes of padding), encoded by SpawnDev.Codecs on the CPU accelerator.
/// </summary>
public class SaveAudioFlacTests
{
    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "codecs", "save_audio");
    static JsonElement Cases => JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "cases.json"))).RootElement;

    static AudioClip Clip(string name)
    {
        var c = Cases.GetProperty("cases").EnumerateArray().First(c => c.GetProperty("name").GetString() == name);
        int channels = c.GetProperty("channels").GetInt32(), n = c.GetProperty("samples").GetInt32();
        var raw = File.ReadAllBytes(Path.Combine(Dir, name + ".f32"));
        var data = new float[channels][];
        for (int k = 0; k < channels; k++)
        {
            data[k] = new float[n];
            Buffer.BlockCopy(raw, k * n * 4, data[k], 0, n * 4);
        }
        return new AudioClip(data, c.GetProperty("sample_rate").GetInt32());
    }

    /// <summary>ComfyUI's comments: prompt, then each extra_pnginfo key (none when the run had no metadata).</summary>
    static List<KeyValuePair<string, string>> Metadata(bool present)
    {
        if (!present) return [];
        var all = Cases;
        var list = new List<KeyValuePair<string, string>> { new("prompt", all.GetProperty("prompt").GetString()!) };
        foreach (var e in all.GetProperty("extra").EnumerateObject()) list.Add(new(e.Name, e.Value.GetString()!));
        return list;
    }

    [TestCase("stereo_44100")]
    [TestCase("mono_22050_odd")]
    [TestCase("loud_clipping")]
    [TestCase("stereo_no_metadata")]
    public async Task FLAC_is_ComfyUIs_file(string name)
    {
        var c = Cases.GetProperty("cases").EnumerateArray().First(c => c.GetProperty("name").GetString() == name);
        using var context = Context.Create(b => b.CPU());
        using var acc = context.CreateCPUAccelerator(0);
        var ours = await CodecsAudio.EncodeFlacAsync(acc, Clip(name), Metadata(c.GetProperty("metadata").GetBoolean()));
        await AssertComfyUIsFlac(acc, ours, name);

    }

    [TestCase("stereo_44100")]
    [TestCase("loud_clipping")]
    [Explicit("Needs an OpenCL GPU.")]
    public async Task FLAC_is_ComfyUIs_file_on_OpenCL(string name)
    {
        var c = Cases.GetProperty("cases").EnumerateArray().First(c => c.GetProperty("name").GetString() == name);
        using var context = Context.Create(b => b.OpenCL());
        using var acc = context.Devices.First(d => d.AcceleratorType == AcceleratorType.OpenCL).CreateAccelerator(context);
        TestContext.Out.WriteLine(acc.Name);
        var ours = await CodecsAudio.EncodeFlacAsync(acc, Clip(name), Metadata(c.GetProperty("metadata").GetBoolean()));
        await AssertComfyUIsFlac(acc, ours, name);
    }

    [Test]
    public async Task Warming_up_leaves_the_files_ComfyUIs()
    {
        using var context = Context.Create(b => b.CPU());
        using var acc = context.CreateCPUAccelerator(0);
        await CodecsAudio.WarmUpAsync(acc, CodecsAudio.Work.FlacEncode | CodecsAudio.Work.FlacDecode | CodecsAudio.Work.Opus);
        foreach (var name in new[] { "stereo_44100", "mono_22050_odd" })
            await AssertComfyUIsFlac(acc, await CodecsAudio.EncodeFlacAsync(acc, Clip(name), Metadata(true)), name);
        Assert.That(CodecsAudio.WorkFor("SaveAudioAdvanced"), Is.EqualTo(CodecsAudio.Work.FlacEncode | CodecsAudio.Work.Opus));
        Assert.That(CodecsAudio.WorkFor("LoadAudio"), Is.EqualTo(CodecsAudio.Work.FlacDecode));
        Assert.That(CodecsAudio.WorkFor("KSampler"), Is.EqualTo(CodecsAudio.Work.None));
    }

    [Test]
    public async Task Save_Audio_writes_the_FLAC_with_the_runs_prompt_and_workflow()
    {
        var saved = AudioFiles.Flac;
        using var context = Context.Create(b => b.CPU());
        using var acc = context.CreateCPUAccelerator(0);
        CodecsAudio.Install(() => Task.FromResult<Accelerator>(acc));
        try
        {
            var store = new MemoryResultStore();
            var services = new Services(store);
            var all = Cases;
            string prompt = all.GetProperty("prompt").GetString()!;
            var extra = all.GetProperty("extra").EnumerateObject().ToDictionary(e => e.Name, e => e.Value.GetString()!);
            var node = (NodeContext)Activator.CreateInstance(typeof(NodeContext), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, null,
                ["1", "SaveAudio", services, CancellationToken.None, null, new Lazy<string>(() => prompt), extra, null, null, 0], null)!;
            var (_, ui) = await new SaveAudio().ExecuteAsync(Clip("stereo_44100"), node, services);
            var file = (ResultFile)((object[])ui["audio"]!)[0];
            Assert.That((file.Subfolder, file.Filename, file.Type), Is.EqualTo(("audio", "ComfyUI_00001.flac", "output")));
            await AssertComfyUIsFlac(acc, store.Get("output", file.Filename, "audio")!, "stereo_44100");
        }
        finally { AudioFiles.Flac = saved; }
    }

    /// <summary>
    /// A FLAC as ComfyUI saves it, minus the frames' bytes (TJ's choice: SpawnDev.Codecs' clean-room encoder makes its own lossless frames): the
    /// metadata blocks (STREAMINFO with the MD5 of the samples, apart from its min / max frame sizes; VORBIS_COMMENT with vendor, prompt, workflow and encoder, PADDING) are ComfyUI's byte for
    /// byte, and the frames decode to exactly the 16-bit samples ComfyUI encodes.
    /// </summary>
    static async Task AssertComfyUIsFlac(Accelerator acc, byte[] ours, string name)
    {
        var theirs = File.ReadAllBytes(Path.Combine(Dir, name + ".flac"));
        // STREAMINFO's minimum and maximum frame sizes (file bytes 12..17) describe the frames, which are the encoder's own; everything else matches.
        static byte[] Metadata(byte[] f) { var m = f.AsSpan(0, MetadataLength(f)).ToArray(); m.AsSpan(12, 6).Clear(); return m; }
        Assert.That(Metadata(ours), Is.EqualTo(Metadata(theirs)), $"{name}: metadata blocks (frame sizes aside)");
        using var decoder = new SpawnDev.Codecs.Audio.Flac.FlacDecoderGpu(acc);
        var decoded = await decoder.DecodeAsync(ours);
        Assert.That(decoded.InterleavedSamples, Is.EqualTo(AudioFiles.Samples16(Clip(name))), $"{name}: lossless");
        Assert.That(decoded.Md5Matches, Is.True, $"{name}: STREAMINFO MD5");
    }

    /// <summary>"fLaC" and the metadata blocks, up to the first frame.</summary>
    static int MetadataLength(byte[] flac)
    {
        int i = 4;
        while (true)
        {
            bool last = (flac[i] & 0x80) != 0;
            i += 4 + (flac[i + 1] << 16 | flac[i + 2] << 8 | flac[i + 3]);
            if (last) return i;
        }
    }

    sealed class Services(IResultStore store) : IServiceProvider
    {
        public object? GetService(Type t) => t == typeof(IResultStore) ? store : null;
    }
}
