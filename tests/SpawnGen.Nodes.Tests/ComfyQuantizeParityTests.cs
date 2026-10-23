using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// ImageQuantize against ComfyUI's own Quantize.execute (comfy_extras/nodes_post_processing.py) run on torch, numpy and Pillow 13:
/// Fixtures/comfy/comfy_image_quantize.json.gz, written by tools/comfy-parity/gen_quantize_fixtures.py. Every case must agree byte for byte,
/// both the palette Pillow's median cut builds and the picture after the palette step (plain, Floyd-Steinberg, bayer 2 to 16).
/// </summary>
public class ComfyQuantizeParityTests
{
    public sealed record Case(JsonElement Json, int Index)
    {
        public override string ToString() => $"{Json.GetProperty("picture").GetString()} {Json.GetProperty("colors").GetInt32()} {Json.GetProperty("dither").GetString()}";
    }

    static JsonDocument? doc;

    static JsonElement Root()
    {
        lock (typeof(ComfyQuantizeParityTests))
        {
            if (doc is null)
            {
                var path = Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "comfy", "comfy_image_quantize.json.gz");
                using var gz = new GZipStream(File.OpenRead(path), CompressionMode.Decompress);
                doc = JsonDocument.Parse(gz);
            }
            return doc.RootElement;
        }
    }

    static IEnumerable<TestCaseData> Cases() =>
        Root().GetProperty("cases").EnumerateArray().Select((c, i) => new TestCaseData(new Case(c, i)).SetName($"ImageQuantize_{i:D3}_{new Case(c, i)}"));

    static int[] Ints(JsonElement a) => a.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    /// <summary>The picture the generator fed ComfyUI: float32 k / 65535, either stored or from the same xorshift32 stream.</summary>
    static ImageBatch Input(JsonElement c)
    {
        int[] shape, k;
        if (c.TryGetProperty("procedural", out var p))
        {
            shape = Ints(p.GetProperty("shape"));
            k = Procedural(p.GetProperty("kind").GetString()!, shape, p.GetProperty("seed").GetUInt32());
        }
        else
        {
            var t = Root().GetProperty("tensors")[c.GetProperty("input").GetInt32()];
            shape = Ints(t.GetProperty("shape"));
            k = Ints(t.GetProperty("k"));
        }
        var img = new ImageBatch(shape[0], shape[1], shape[2], shape[3]);
        bool outside = c.GetProperty("picture").GetString() == "outside";
        for (int i = 0; i < k.Length; i++)
        {
            float v = k[i] / 65535f;
            img.Data[i] = outside ? v * 1.5f - 0.25f : v;
        }
        return img;
    }

    static int[] Procedural(string kind, int[] shape, uint seed)
    {
        int b = shape[0], h = shape[1], w = shape[2], c = shape[3];
        var r = new int[b * h * w * c];
        uint s = seed;
        for (int i = 0; i < r.Length; i++)
        {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            r[i] = (int)(s >> 16);
        }
        if (kind == "random") return r;
        Assert.That(kind, Is.EqualTo("gradnoise"));
        var k = new int[r.Length];
        for (int bi = 0; bi < b; bi++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    for (int ch = 0; ch < c; ch++)
                    {
                        long bas = (ch % 4) switch
                        {
                            0 => x * 65535L / (w - 1),
                            1 => y * 65535L / (h - 1),
                            2 => (x + y) * 65535L / (w + h - 2),
                            _ => 65535 - x * 65535L / (w - 1),
                        };
                        int i = ((bi * h + y) * w + x) * c + ch;
                        k[i] = (int)Math.Clamp(bas + r[i] % 2049 - 1024, 0, 65535);
                    }
        return k;
    }

    [TestCaseSource(nameof(Cases))]
    public void Matches_ComfyUI(Case test)
    {
        var c = test.Json;
        int colors = c.GetProperty("colors").GetInt32();
        string dither = c.GetProperty("dither").GetString()!;
        var image = Input(c);
        int n = image.Width * image.Height;

        // The median-cut palette of every picture of the batch.
        var palettes = c.GetProperty("palettes").EnumerateArray().Select(Ints).ToArray();
        for (int b = 0; b < image.Batch; b++)
        {
            var rgb = new byte[n * 3];
            for (int i = 0; i < n; i++)
                for (int ch = 0; ch < 3; ch++) rgb[i * 3 + ch] = ImageQuantize.TorchToUInt8(image.Data[(b * n + i) * image.Channels + ch] * 255f);
            Assert.That(PilQuantize.MedianCutPalette(rgb, colors).Select(v => (int)v).ToArray(), Is.EqualTo(palettes[b]), $"palette of picture {b}");
        }

        var result = new ImageQuantize().Execute(image, colors, dither);
        Assert.That((result.Batch, result.Height, result.Width, result.Channels), Is.EqualTo((image.Batch, image.Height, image.Width, image.Channels == 4 ? 4 : 3)));
        var bytes = new byte[result.Batch * n * 3];
        for (int i = 0; i < result.Batch * n; i++)
            for (int ch = 0; ch < 3; ch++)
            {
                float v = result.Data[i * result.Channels + ch];
                bytes[i * 3 + ch] = (byte)MathF.Round(v * 255f);
                Assert.That(v, Is.EqualTo(bytes[i * 3 + ch] / 255f), "an exact byte / 255");
            }
        if (result.Channels == 4)
            for (int i = 0; i < result.Batch * n; i++) Assert.That(result.Data[i * 4 + 3], Is.EqualTo(image.Data[i * 4 + 3]), "alpha passes through");

        if (c.TryGetProperty("sha256", out var sha))
            Assert.That(Convert.ToHexStringLower(SHA256.HashData(bytes)), Is.EqualTo(sha.GetString()));
        else
        {
            var expected = Ints(Root().GetProperty("tensors")[c.GetProperty("out").GetInt32()].GetProperty("bytes"));
            int wrong = Enumerable.Range(0, bytes.Length).Count(i => bytes[i] != expected[i]);
            int first = Enumerable.Range(0, bytes.Length).FirstOrDefault(i => bytes[i] != expected[i], -1);
            Assert.That(wrong, Is.Zero, $"{wrong} of {bytes.Length} bytes differ, the first at {first}: {(first >= 0 ? bytes[first] : 0)} vs {(first >= 0 ? expected[first] : 0)}");
        }

        if (test.Index % 7 == 0 || c.TryGetProperty("procedural", out _))
        {
            var dir = Path.Combine(TestContext.CurrentContext.WorkDirectory, "comfy-quantize");
            Directory.CreateDirectory(dir);
            File.WriteAllBytes(Path.Combine(dir, $"{test.Index:D3}_{c.GetProperty("picture").GetString()}_{colors}_{dither}.png"), PngWriter.Encode(result));
        }
    }

    [Test]
    public void Fixture_covers_every_dither_and_the_colour_hash_coarsening()
    {
        var cases = Root().GetProperty("cases").EnumerateArray().ToArray();
        Assert.That(cases.Select(c => c.GetProperty("dither").GetString()).Distinct(), Is.EquivalentTo(ImageQuantize.Dithers()));
        Assert.That(cases.Select(c => c.GetProperty("colors").GetInt32()).Distinct(), Is.SupersetOf(new[] { 1, 2, 16, 256 }));
        Assert.That(cases.Count(c => c.TryGetProperty("procedural", out _)), Is.GreaterThan(0), "pictures of more than 65536 colours");
        // A picture with fewer colours than asked for gets a shorter palette (bayer's spread follows it).
        Assert.That(cases.Any(c => c.GetProperty("palettes")[0].GetArrayLength() / 3 < c.GetProperty("colors").GetInt32()), Is.True);
    }
}
