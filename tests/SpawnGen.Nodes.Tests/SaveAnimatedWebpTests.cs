using System.Text.Json;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Save Animated WEBP against ComfyUI's own saving (Fixtures/codecs/save_webp, make_save_webp.py: ImageSaveHelper.save_animated_webp, Pillow 13 / libwebp
/// 1.6.0). EXIF: byte for byte with Pillow's Exif.tobytes(). Lossy animation (SpawnDev.Codecs' port of WebPAnimEncoder): byte for byte with Pillow's file.
/// A single lossy picture (Pillow's still path, EXIF without its prefix): byte for byte too. Lossless (SpawnDev.Codecs' own VP8L, whole frames): the pictures
/// decode exactly.
/// </summary>
public class SaveAnimatedWebpTests
{
    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "codecs", "save_webp");
    static JsonElement Cases => JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "cases.json"))).RootElement;

    [Test]
    public void Exif_is_Pillows_bytes_for_ComfyUIs_tags()
    {
        foreach (var c in Cases.GetProperty("exif").EnumerateArray())
        {
            var meta = c.GetProperty("metadata").EnumerateArray().Select(kv => (kv[0].GetString()!, kv[1].GetString()!))
                .ToDictionary(kv => kv.Item1, kv => kv.Item2);
            Assert.That(Convert.ToHexStringLower(ExifWriter.ComfyMetadata(meta)), Is.EqualTo(c.GetProperty("hex").GetString()), string.Join(",", meta.Keys));
        }
    }

    sealed class Services(IResultStore store, MlRuntime runtime) : IServiceProvider
    {
        public object? GetService(Type t) => t == typeof(IResultStore) ? store : t == typeof(MlRuntime) ? runtime : null;
    }

    static ImageBatch Frames(int count)
    {
        var all = Cases;
        int w = all.GetProperty("width").GetInt32(), h = all.GetProperty("height").GetInt32();
        var raw = File.ReadAllBytes(Path.Combine(Dir, "frames.f32"));
        var data = new float[count * h * w * 3];
        Buffer.BlockCopy(raw, 0, data, 0, data.Length * 4);
        return new ImageBatch(count, h, w, 3, data);
    }

    static NodeContext Context(IServiceProvider services)
    {
        var all = Cases;
        var extra = all.GetProperty("extra").EnumerateArray().ToDictionary(kv => kv[0].GetString()!, kv => kv[1].GetString()!);
        string prompt = all.GetProperty("prompt").GetString()!;
        return (NodeContext)Activator.CreateInstance(typeof(NodeContext), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, null,
            ["1", "SaveAnimatedWEBP", services, CancellationToken.None, null, new Lazy<string>(() => prompt), extra, null, null, 0], null)!;
    }

    [TestCase("anim_lossy", 3, false)]
    [TestCase("anim_lossless", 3, true)]
    [TestCase("one_lossy", 1, false)]
    [TestCase("one_lossless", 1, true)]
    public async Task Save_animated_webp_matches_ComfyUI_on_the_CPU_accelerator(string name, int frames, bool lossless)
    {
        using var ilgpu = ILGPU.Context.Create(b => b.CPU());
        using var accelerator = ilgpu.CreateCPUAccelerator(0);
        await Check(accelerator, name, frames, lossless);
    }

    [TestCase("anim_lossy", 3, false)]
    [TestCase("anim_lossless", 3, true)]
    [Explicit("Needs a GPU (OpenCL / CUDA).")]
    public async Task Save_animated_webp_matches_ComfyUI_on_the_GPU(string name, int frames, bool lossless)
    {
        using var runtime = new MlRuntime(new FolderModelSource(Path.GetTempPath()));
        TestContext.Out.WriteLine(runtime.Accelerator.Name);
        await Check(runtime.Accelerator, name, frames, lossless);
    }

    static async Task Check(Accelerator accelerator, string name, int count, bool lossless)
    {
        using var runtime = new MlRuntime(new FolderModelSource(Path.GetTempPath()), accelerator);
        var store = new MemoryResultStore();
        var services = new Services(store, runtime);
        var images = Frames(count);
        var (passed, ui) = await new SaveAnimatedWEBP().ExecuteAsync(images, Context(services), services, "anim", 6.0, lossless, 80, "default");
        Assert.That(passed, Is.SameAs(images));
        var file = (ResultFile)((object[])ui["images"]!)[0];
        Assert.That(file.Filename, Is.EqualTo("anim_00001_.webp"));
        Assert.That(ui.ContainsKey("animated"), Is.EqualTo(count > 1));
        var ours = store.Get("output", file.Filename)!;
        var pillow = File.ReadAllBytes(Path.Combine(Dir, name + ".webp"));
        if (name is "anim_lossy" or "one_lossy")
        {
            Assert.That(ours, Is.EqualTo(pillow), "byte for byte with Pillow / libwebp");
            return;
        }
        // Otherwise: the same pictures as Pillow's file (lossless: exactly the input; lossy still: what libwebp decodes from Pillow's file).
        var mine = await CodecsImages.DecodeAsync(accelerator, ours);
        var theirs = await CodecsImages.DecodeAsync(accelerator, pillow);
        Assert.That(mine!.Rgba, Has.Count.EqualTo(count));
        for (int f = 0; f < count; f++) Assert.That(mine.Rgba[f], Is.EqualTo(theirs!.Rgba[f]), $"frame {f}");
        TestContext.Out.WriteLine($"{name}: ours {ours.Length} bytes, Pillow's {pillow.Length}");
    }
}
