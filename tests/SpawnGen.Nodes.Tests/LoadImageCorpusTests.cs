using System.Text.Json;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Load Image on one small file per PNG / JPEG case (Fixtures/decode_corpus, written by tools/decode-corpus/make_corpus.py) against the
/// spec-correct picture (tools/decode-corpus/make_reference.py): TJ 2026-11-20, picture decoding is held to correctness, not to ComfyUI's
/// FFmpeg-based Load Image (whose measured differences are listed in docs/PARITY.md). PNG must match exactly; JPEG within the JPEG standard's
/// freedom against libjpeg (inverse-DCT rounding, chroma upsampling), with the limits JpegTests uses.
/// </summary>
public class LoadImageCorpusTests
{
    static string Folder => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "decode_corpus");

    static JsonElement Reference => JsonDocument.Parse(File.ReadAllText(Path.Combine(Folder, "reference", "reference.json"))).RootElement;

    public static IEnumerable<string> Files() =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "decode_corpus", "reference", "reference.json")))
            .RootElement.EnumerateObject().Select(p => p.Name);

    static float[] Values(string file, string what, JsonElement entry)
    {
        var path = Path.Combine(Folder, "reference", $"{file}.{what}");
        if (entry.TryGetProperty(what + "_float", out var f) && f.GetBoolean())
        {
            var bytes = File.ReadAllBytes(path + ".f32");
            var values = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
            return values;
        }
        return File.ReadAllBytes(path).Select(b => b / 255f).ToArray();
    }

    /// <summary>The reference IMAGE / MASK and the limits Load Image is held to: exact for PNG, libjpeg's freedom for JPEG (mean, max in 1/255).</summary>
    internal static (int[] ImageShape, float[] Image, int[] MaskShape, float[] Mask, bool Exact) Expected(string file)
    {
        var entry = Reference.GetProperty(file);
        int h = entry.GetProperty("height").GetInt32(), w = entry.GetProperty("width").GetInt32();
        int mh = entry.GetProperty("mask_height").GetInt32(), mw = entry.GetProperty("mask_width").GetInt32();
        return ([1, h, w, 3], Values(file, "image", entry), [1, mh, mw], Values(file, "mask", entry), entry.GetProperty("exact").GetBoolean());
    }

    internal static void Check(string file, ImageBatch image, MaskBatch mask)
    {
        var (imageShape, expected, maskShape, expectedMask, exact) = Expected(file);
        Assert.That(image.Shape, Is.EqualTo(imageShape), "IMAGE shape");
        Assert.That(mask.Shape, Is.EqualTo(maskShape), "MASK shape");
        double sum = 0, worst = 0;
        for (int i = 0; i < expected.Length; i++) { double d = Math.Abs(image.Data[i] - expected[i]) * 255; sum += d; worst = Math.Max(worst, d); }
        double mean = sum / expected.Length;
        double maskWorst = Enumerable.Range(0, expectedMask.Length).Max(i => Math.Abs(mask.Data[i] - expectedMask[i])) * 255;
        TestContext.Out.WriteLine($"{file}: IMAGE mean {mean:F4}/255, largest {worst:F4}/255; MASK largest {maskWorst:F4}/255");
        if (exact) Assert.That(worst, Is.LessThanOrEqualTo(1e-3), "IMAGE: a PNG decodes exactly");
        else
        {
            Assert.That(mean, Is.LessThan(file.Contains("444") || file.Contains("gray") ? 0.2 : 0.6), "IMAGE: mean difference from libjpeg");
            Assert.That(worst, Is.LessThan(file.Contains("444") || file.Contains("gray") ? 3.0 : 4.0), "IMAGE: largest difference from libjpeg");
        }
        Assert.That(maskWorst, Is.LessThanOrEqualTo(1e-3), "MASK");
    }

    [TestCaseSource(nameof(Files))]
    public async Task Load_Image_decodes_the_file_correctly(string file)
    {
        var store = new MemoryInputStore();
        await store.SaveAsync(file, File.ReadAllBytes(Path.Combine(Folder, file)));
        var saved = InputFiles.Store;
        InputFiles.Store = store;
        try
        {
            var (image, mask) = await new LoadImage().ExecuteAsync(file);
            await image.ToHostAsync();
            Check(file, image, mask);
        }
        finally { InputFiles.Store = saved; }
    }
}
