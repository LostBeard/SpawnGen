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
/// Save Audio's FLAC against ComfyUI's own saving (Fixtures/codecs/save_audio, make_save_audio.py: AudioSaveHelper.save_audio, PyAV 18.1 / FFmpeg 8:
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
        var theirs = File.ReadAllBytes(Path.Combine(Dir, name + ".flac"));
        int first = Enumerable.Range(0, Math.Min(ours.Length, theirs.Length)).FirstOrDefault(i => ours[i] != theirs[i], -1);
        Assert.That(ours, Is.EqualTo(theirs), $"{ours.Length} vs {theirs.Length} bytes, first difference at {first}");
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
        Assert.That(ours, Is.EqualTo(File.ReadAllBytes(Path.Combine(Dir, name + ".flac"))));
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
            Assert.That(store.Get("output", file.Filename, "audio"), Is.EqualTo(File.ReadAllBytes(Path.Combine(Dir, "stereo_44100.flac"))));
        }
        finally { AudioFiles.Flac = saved; }
    }

    sealed class Services(IResultStore store) : IServiceProvider
    {
        public object? GetService(Type t) => t == typeof(IResultStore) ? store : null;
    }
}
