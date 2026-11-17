using System.Security.Cryptography;
using System.Text.Json;
using ILGPU;
using ILGPU.Runtime.CPU;
using SpawnDev.Codecs.Video;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Load Video against ComfyUI's LoadVideo (black-box dumps, Fixtures/video/comfyui_loadvideo.json: a hash of its float32 frames, the fps, the sound):
/// SpawnDev.Codecs' WebM loader on the CPU accelerator, through Load Video's own frame and sound layout.
/// </summary>
public class LoadVideoTests
{
    static readonly string Fixtures = Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "video");

    static JsonElement Reference(string file) =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixtures, "comfyui_loadvideo.json"))).RootElement.GetProperty("files").GetProperty(file);

    [Test]
    public async Task A_VP9_WebM_loads_bit_for_bit_as_ComfyUI_loads_it()
    {
        const string file = "comfyui_vp9_rgba_512x320_crf32.webm";
        using var context = Context.Create(b => b.CPU());
        using var accelerator = context.CreateCPUAccelerator(0);
        var video = await new WebMVideoLoaderGpu(accelerator).LoadAsync(File.ReadAllBytes(Path.Combine(Fixtures, file)));
        var frames = LoadVideo.Frames(video);
        var reference = Reference(file);

        Assert.That(frames.Shape, Is.EqualTo(reference.GetProperty("frames_shape").EnumerateArray().Select(e => e.GetInt32()).ToArray()), "RGB: the alpha is dropped");
        Assert.That(video.FrameRate, Is.EqualTo(reference.GetProperty("fps").GetDouble()));
        var bytes = new byte[frames.Data.Length * 4];
        Buffer.BlockCopy(frames.Data, 0, bytes, 0, bytes.Length);
        Assert.That(Convert.ToHexStringLower(SHA256.HashData(bytes)), Is.EqualTo(reference.GetProperty("frames_sha256").GetString()));
        Assert.That(LoadVideo.Audio(video.Audio), Is.Null, "no sound in the file: none, as ComfyUI");
    }

    [Test]
    public void Load_Video_lists_ComfyUIs_video_extensions() =>
        Assert.That(LoadVideo.Extensions, Is.EqualTo(new[] { ".3gp", ".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ogv", ".webm", ".wmv" }));

    [Test]
    public void Interleaved_sound_becomes_channels_by_samples()
    {
        var clip = LoadVideo.Audio(new LoadedAudio(48000, 2, [1f, -1f, 2f, -2f, 3f, -3f]))!;
        Assert.That(clip.SampleRate, Is.EqualTo(48000));
        Assert.That(clip.Channels[0], Is.EqualTo(new[] { 1f, 2f, 3f }));
        Assert.That(clip.Channels[1], Is.EqualTo(new[] { -1f, -2f, -3f }));
    }
}
