using System.IO.Compression;
using System.Text.Json;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Load Image with animated PNGs against ComfyUI's own Load Image (Fixtures/codecs/apng, make_apng.py: Pillow-made APNGs, decoded by ComfyUI through
/// PyAV, i.e. ffmpeg's APNG decoder, whose RGBA frames reach the IMAGE exactly): every frame, every pixel and the mask, exactly, for over / source blending
/// and none / background / previous disposal, plus hand-built probes of every pair of disposals (ffmpeg shows a frame disposed to the background
/// unblended).
/// </summary>
public class ApngTests
{
    static readonly string[] Files = ["over_none.png", "over_background.png", "over_previous.png", "source_none.png", "rgb.png",
        .. from d0 in Enumerable.Range(0, 3) from d1 in Enumerable.Range(0, 3) select $"probe_{d0}_{d1}.png"];

    [TestCaseSource(nameof(Files))]
    public async Task Load_Image_composites_the_frames_like_ComfyUI(string file)
    {
        var dir = Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "codecs", "apng");
        using var gz = new GZipStream(File.OpenRead(Path.Combine(dir, "load_apng.json.gz")), CompressionMode.Decompress);
        var fx = JsonDocument.Parse(gz).RootElement.GetProperty(file);
        static (int[] Shape, float[] Data) T(JsonElement t) =>
            (t.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray(), t.GetProperty("data").EnumerateArray().Select(e => e.GetSingle()).ToArray());
        var (imageShape, image) = T(fx.GetProperty("image"));
        var (maskShape, mask) = T(fx.GetProperty("mask"));

        var store = new MemoryInputStore();
        await store.SaveAsync(file, File.ReadAllBytes(Path.Combine(dir, file)));
        var saved = InputFiles.Store;
        InputFiles.Store = store;
        try
        {
            var (ours, ourMask) = await new LoadImage().ExecuteAsync(file);
            Assert.That(ours.Shape, Is.EqualTo(imageShape), "IMAGE shape");
            Assert.That(ourMask.Shape, Is.EqualTo(maskShape), "MASK shape");
            int off = Enumerable.Range(0, image.Length).Count(i => Math.Abs(ours.Data[i] - image[i]) * 255 > 0.01);
            int maskOff = Enumerable.Range(0, mask.Length).Count(i => Math.Abs(ourMask.Data[i] - mask[i]) * 255 > 0.01);
            Assert.That((off, maskOff), Is.EqualTo((0, 0)), "colour and mask values that differ from ComfyUI's");
        }
        finally { InputFiles.Store = saved; }
    }
}
