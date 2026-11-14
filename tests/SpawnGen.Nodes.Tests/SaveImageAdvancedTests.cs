using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Save Image (Advanced) against files ComfyUI's node wrote (Fixtures/image_save_advanced, ComfyUI bef11fc on the CPU, 2026-11-13). The input
/// pictures are raw float32 files a probe node loaded unchanged: src (2 x 24 x 40 RGB, values from -0.1 to 1.4 and a few chosen near rounding
/// edges), rgba (1 x 8 x 10 RGBA), and k: every 16-bit level k / 65535 (red), k / 65535 + half a level (green), min(k, 255) / 255 (blue).
/// </summary>
public class SaveImageAdvancedTests
{
    static readonly string Fixtures = Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "image_save_advanced");

    static byte[] Fixture(string name) => File.ReadAllBytes(Path.Combine(Fixtures, name));

    static ImageBatch Raw(string name, int batch, int h, int w, int ch)
    {
        var bytes = Fixture(name);
        var data = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
        return new ImageBatch(batch, h, w, ch, data);
    }

    static ImageBatch Src => Raw("src_f32_2x24x40x3.raw", 2, 24, 40, 3);
    static ImageBatch Rgba => Raw("rgba_f32_1x8x10x4.raw", 1, 8, 10, 4);

    static ImageBatch Levels()
    {
        var img = new ImageBatch(1, 256, 256, 3);
        for (int k = 0; k < 65536; k++)
        {
            img.Data[k * 3] = (float)(k / 65535.0);
            img.Data[k * 3 + 1] = (float)(k / 65535.0 + 0.5 / 65535.0);
            img.Data[k * 3 + 2] = (float)(Math.Min(k, 255) / 255.0);
        }
        return img;
    }

    // ---- PNG ----

    [TestCase("png8_0.png", 0, 8)]
    [TestCase("png8_1.png", 1, 8)]
    [TestCase("png16_0.png", 0, 16)]
    [TestCase("png16_1.png", 1, 16)]
    public void PNG_samples_are_ComfyUIs(string file, int index, int bits)
    {
        var comfy = Fixture(file);
        var ours = PngWriter.Encode(Src, index, PngText(comfy), bits, aspectChunk: true);
        Assert.That(Samples(ours), Is.EqualTo(Samples(comfy)));
        Assert.That(Chunks(ours).Select(c => c.Type), Is.EqualTo(Chunks(comfy).Select(c => c.Type)), "IHDR, tEXt prompt, tEXt workflow, pHYs, IDAT, IEND");
        foreach (var type in new[] { "IHDR", "tEXt", "pHYs" })
            Assert.That(Chunks(ours).Where(c => c.Type == type).Select(c => c.Data), Is.EqualTo(Chunks(comfy).Where(c => c.Type == type).Select(c => c.Data)), type);
    }

    [TestCase("k_png16.png", 16)]
    [TestCase("k_png8.png", 8)]
    public void Every_16_bit_level_is_truncated_as_ComfyUI_does(string file, int bits)
    {
        // A float64 product would differ at 32,895 of the red levels: ComfyUI multiplies in float32, then truncates.
        Assert.That(Samples(PngWriter.Encode(Levels(), 0, null, bits)), Is.EqualTo(Samples(Fixture(file))));
    }

    [TestCase("rgba_png8.png", 8)]
    [TestCase("rgba_png16.png", 16)]
    public void RGBA_PNGs_keep_their_alpha_as_ComfyUI(string file, int bits)
    {
        var comfy = Fixture(file);
        var ours = PngWriter.Encode(Rgba, 0, null, bits);
        Assert.That(Chunks(ours)[0].Data, Is.EqualTo(Chunks(comfy)[0].Data), "IHDR: RGBA (colour type 6) at the same depth");
        Assert.That(Samples(ours), Is.EqualTo(Samples(comfy)));
    }

    // ---- OpenEXR ----

    [TestCase("exr32_linear_0.exr", 0, ExrInput.Linear, false)]
    [TestCase("exr32_linear_1.exr", 1, ExrInput.Linear, false)]
    [TestCase("exr16_srgb_0.exr", 0, ExrInput.Srgb, true)]
    [TestCase("exr16_srgb_1.exr", 1, ExrInput.Srgb, true)]
    public void EXR_values_are_ComfyUIs_exactly(string file, int index, ExrInput input, bool half)
    {
        var comfy = Exr.Read(Fixture(file));
        var ours = Exr.Read(ExrWriter.Encode(Src, index, input, half));
        AssertSameLayout(ours, comfy);
        foreach (var name in comfy.Channels.Keys) Assert.That(ours.Channels[name], Is.EqualTo(comfy.Channels[name]), name);
    }

    [TestCase("exr32_srgb_0.exr", 0, ExrInput.Srgb)]
    [TestCase("exr32_srgb_1.exr", 1, ExrInput.Srgb)]
    [TestCase("exr32_hdr_0.exr", 0, ExrInput.Hlg)]
    [TestCase("exr32_hdr_1.exr", 1, ExrInput.Hlg)]
    public void EXR_float_transfers_match_ComfyUI_to_its_last_bit_or_two(string file, int index, ExrInput input)
    {
        // torch's pow and exp and .NET's MathF differ in the last bit now and then (measured: at most 1.7e-7 relative); nothing more.
        var comfy = Exr.Read(Fixture(file));
        var ours = Exr.Read(ExrWriter.Encode(Src, index, input, half: false));
        AssertSameLayout(ours, comfy);
        foreach (var name in comfy.Channels.Keys)
            for (int i = 0; i < comfy.Channels[name].Length; i++)
            {
                float c = comfy.Channels[name][i], o = ours.Channels[name][i];
                Assert.That(Math.Abs(o - c), Is.LessThanOrEqualTo(3e-7 * Math.Max(Math.Abs(c), 1e-30f)), $"{name}[{i}]: ComfyUI {c:R}, ours {o:R}");
            }
    }

    [TestCase("rgba_exr32.exr", ExrInput.Srgb, false)]
    [TestCase("rgba_exr16.exr", ExrInput.Linear, true)]
    [TestCase("rgba_exrhdr.exr", ExrInput.Hlg, false)]
    public void RGBA_EXRs_write_alpha_unchanged_and_straight_colours(string file, ExrInput input, bool half)
    {
        var comfy = Exr.Read(Fixture(file));
        var ours = Exr.Read(ExrWriter.Encode(Rgba, 0, input, half));
        AssertSameLayout(ours, comfy);
        Assert.That(ours.Channels.Keys, Is.EqualTo(new[] { "A", "B", "G", "R" }));
        Assert.That(ours.Channels["A"], Is.EqualTo(comfy.Channels["A"]));
        foreach (var name in new[] { "B", "G", "R" })
            for (int i = 0; i < comfy.Channels[name].Length; i++)
                Assert.That(ours.Channels[name][i], Is.EqualTo(comfy.Channels[name][i]).Within(3e-7 * Math.Abs(comfy.Channels[name][i]) + 1e-30), $"{name}[{i}]");
    }

    [Test]
    public void EXR_header_carries_the_metadata_and_the_gamut()
    {
        var text = new Dictionary<string, string> { ["prompt"] = "{\"1\":{}}", ["workflow"] = "{\"nodes\":[\"é\"]}" };
        var exr = Exr.Read(ExrWriter.Encode(Src, 0, ExrInput.Hlg, half: true, text));
        Assert.That(exr.Strings["prompt"], Is.EqualTo("{\"1\":{}}"));
        Assert.That(exr.Strings["workflow"], Is.EqualTo("{\"nodes\":[\"é\"]}"));
        Assert.That(exr.Attributes, Does.Contain("pixelAspectRatio"), "required by the file layout (ComfyUI's libavcodec files leave it out)");
        Assert.That(exr.Chromaticities, Is.EqualTo(Exr.Read(Fixture("exr32_hdr_0.exr")).Chromaticities), "Rec.2020 for HLG input");
        Assert.That(Exr.Read(ExrWriter.Encode(Src, 0, ExrInput.Linear, true)).Chromaticities, Is.EqualTo(Exr.Read(Fixture("exr32_linear_0.exr")).Chromaticities));
    }

    [Test]
    public void A_chunk_that_does_not_compress_is_stored_as_it_is()
    {
        // Noise does not compress: the chunk's size then equals its raw size, which is how a reader knows.
        var noise = new ImageBatch(1, 3, 5, 3);
        var rng = new Random(5);
        for (int i = 0; i < noise.Data.Length; i++) noise.Data[i] = (float)rng.NextDouble();
        var exr = Exr.Read(ExrWriter.Encode(noise, 0, ExrInput.Linear, half: false));
        Assert.That(exr.StoredChunks, Is.EqualTo(1));
        for (int i = 0; i < 15; i++) Assert.That(exr.Channels["R"][i], Is.EqualTo(noise.Data[i * 3]));
    }

    // ---- the node ----

    [Test]
    public async Task The_node_names_files_without_the_trailing_underscore_and_shares_Save_Images_counter()
    {
        var (services, store) = Services();
        var node = new SaveImageAdvanced();
        var png = new DynamicChoice("png", new Dictionary<string, object?> { ["bit_depth"] = "16-bit", ["input_color_space"] = "sRGB" });
        var src = Src;
        var (passed, ui) = await node.ExecuteAsync(src, Context(services), services, "cnt_a", png);
        Assert.That(passed, Is.SameAs(src), "the pictures are passed on");
        Assert.That(Names(ui), Is.EqualTo(new[] { "cnt_a_00001.png", "cnt_a_00002.png" }));
        // Measured: Save Image goes on after these (cnt_a_00003_.png ...), and this node after Save Image's.
        var (_, classic) = await new SaveImage().ExecuteAsync(Src, Context(services), services, "cnt_a");
        Assert.That(Names(classic), Is.EqualTo(new[] { "cnt_a_00003_.png", "cnt_a_00004_.png" }));
        var exr = new DynamicChoice("exr", new Dictionary<string, object?> { ["bit_depth"] = "32-bit float", ["input_color_space"] = "linear" });
        (_, ui) = await node.ExecuteAsync(Src, Context(services), services, "cnt_a", exr);
        Assert.That(Names(ui), Is.EqualTo(new[] { "cnt_a_00005.exr", "cnt_a_00006.exr" }));

        var saved = store.List("output", "")!;
        var file = await store.ReadAsync("output", "", "cnt_a_00001.png");
        Assert.That(Chunks(file!)[0].Data[8], Is.EqualTo(16), "16-bit, as chosen");
        Assert.That(PngText(file!)["prompt"], Is.EqualTo("{\"1\":{\"class_type\":\"X\"}}"));
        var entry = (ResultFile)((List<object>)ui["images"]!)[0];
        Assert.That(entry.Metadata, Is.EqualTo(new ResultMetadata("image", 40, 24)));
        var written = Exr.Read((await store.ReadAsync("output", "", "cnt_a_00006.exr"))!);
        Assert.That(written.Channels["R"], Is.EqualTo(Exr.Read(ExrWriter.Encode(Src, 1, ExrInput.Linear, false)).Channels["R"]), "the second picture");
        Assert.That(written.Strings["workflow"], Is.EqualTo("{\"nodes\":[\"é\"]}"));
        Assert.That(saved, Has.Count.EqualTo(6));
    }

    [Test]
    public async Task Batch_num_names_start_at_one_on_every_run_as_in_ComfyUI()
    {
        var (services, _) = Services();
        var png = new DynamicChoice("png", new Dictionary<string, object?>());
        for (int run = 0; run < 2; run++)
        {
            var (_, ui) = await new SaveImageAdvanced().ExecuteAsync(Src, Context(services), services, "b_%batch_num%", png);
            Assert.That(Names(ui), Is.EqualTo(new[] { "b_0_00001.png", "b_1_00002.png" }));
        }
    }

    [Test]
    public void AVIF_is_refused_with_the_reason()
    {
        var (services, _) = Services();
        var avif = new DynamicChoice("avif", new Dictionary<string, object?> { ["crf"] = 18L });
        var error = Assert.ThrowsAsync<NotSupportedException>(() => new SaveImageAdvanced().ExecuteAsync(Src, Context(services), services, "a", avif));
        Assert.That(error!.Message, Does.Contain("AVIF").And.Contain("png or exr"));
    }

    [Test]
    public void The_schema_is_ComfyUIs()
    {
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImageAdvanced).Assembly);
        var schema = registry.GetSchema("SaveImageAdvanced")!;
        Assert.That(schema.DisplayName, Is.EqualTo("Save Image (Advanced)"));
        Assert.That(schema.Category, Is.EqualTo("image"));
        Assert.That(schema.Outputs.Select(o => o.Name), Is.EqualTo(new[] { "images" }));
        var formats = SaveImageAdvanced.Formats().ToList();
        Assert.That(formats.Select(f => f.Key), Is.EqualTo(new[] { "png", "exr", "avif" }));
        Assert.That(formats[2].Inputs.Select(i => i.Name), Is.EqualTo(new[] { "bit_depth", "input_color_space", "crf", "save_mode" }));
        Assert.That(formats[2].Inputs[3].Options!.Select(o => o.Key), Is.EqualTo(new[] { "still images", "animated" }));
    }

    [TestCase("png", "8-bit", true)]
    [TestCase("png", "16-bit float", false)]
    [TestCase("exr", "16-bit float", true)]
    [TestCase("exr", "32-bit float", true)]
    [TestCase("exr", "8-bit", false)]
    [TestCase("avif", "10-bit YUV420", true)]
    public void A_prompt_is_checked_against_the_chosen_formats_choices(string format, string bitDepth, bool valid)
    {
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImageAdvanced).Assembly);
        var g = new SpawnGen.Engine.Graph.PromptGraph();
        g.Add("1", "EmptyImage", ("width", 8L), ("height", 8L), ("batch_size", 1L), ("color", 0L));
        g.Add("2", "SaveImageAdvanced", ("images", SpawnGen.Engine.Graph.PromptInput.Link("1")), ("filename_prefix", "x"), ("format", format), ("format.bit_depth", bitDepth));
        var errors = new SpawnGen.Engine.Execution.PromptExecutor(registry).Validate(g).Errors;
        Assert.That(errors, valid ? Is.Empty : Has.Some.Matches<SpawnGen.Engine.Execution.ValidationError>(e => e.Input == "format.bit_depth"),
            string.Join("; ", errors.Select(e => e.Message)));
    }

    // ---- helpers ----

    static void AssertSameLayout(Exr ours, Exr comfy)
    {
        Assert.That(ours.Width, Is.EqualTo(comfy.Width));
        Assert.That(ours.Height, Is.EqualTo(comfy.Height));
        Assert.That(ours.Compression, Is.EqualTo(comfy.Compression));
        Assert.That(ours.PixelTypes, Is.EqualTo(comfy.PixelTypes));
        Assert.That(ours.Channels.Keys, Is.EqualTo(comfy.Channels.Keys));
        Assert.That(ours.Chromaticities, Is.EqualTo(comfy.Chromaticities));
    }

    static string[] Names(UiOutput ui) => ((List<object>)ui["images"]!).Cast<ResultFile>().Select(f => f.Filename).ToArray();

    sealed class StoreServices(IResultStore store) : IServiceProvider
    {
        public object? GetService(Type serviceType) => serviceType == typeof(IResultStore) ? store : null;
    }

    static (IServiceProvider Services, MemoryResultStore Store) Services()
    {
        var store = new MemoryResultStore();
        return (new StoreServices(store), store);
    }

    static NodeContext Context(IServiceProvider services, string prompt = "{\"1\":{\"class_type\":\"X\"}}") =>
        (NodeContext)Activator.CreateInstance(typeof(NodeContext), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, null,
            ["1", "X", services, CancellationToken.None, null, new Lazy<string>(() => prompt), new Dictionary<string, string> { ["workflow"] = "{\"nodes\":[\"é\"]}" }, null, null, 0], null)!;

    static List<(string Type, byte[] Data)> Chunks(byte[] png)
    {
        var list = new List<(string, byte[])>();
        for (int p = 8; p < png.Length;)
        {
            int len = BinaryPrimitives.ReadInt32BigEndian(png.AsSpan(p));
            list.Add((Encoding.ASCII.GetString(png, p + 4, 4), png[(p + 8)..(p + 8 + len)]));
            p += 12 + len;
        }
        return list;
    }

    static Dictionary<string, string> PngText(byte[] png)
    {
        var text = new Dictionary<string, string>();
        foreach (var (type, data) in Chunks(png).Where(c => c.Type == "tEXt"))
        {
            int z = Array.IndexOf(data, (byte)0);
            text[Encoding.Latin1.GetString(data, 0, z)] = Encoding.Latin1.GetString(data, z + 1, data.Length - z - 1);
        }
        return text;
    }

    /// <summary>A non-interlaced PNG's samples as integers (8 or 16 bits, any filters), straight from IDAT.</summary>
    static int[] Samples(byte[] png)
    {
        var chunks = Chunks(png);
        var ihdr = chunks[0].Data;
        int w = BinaryPrimitives.ReadInt32BigEndian(ihdr), h = BinaryPrimitives.ReadInt32BigEndian(ihdr.AsSpan(4)), depth = ihdr[8];
        int channels = ihdr[9] switch { 0 => 1, 2 => 3, 4 => 2, 6 => 4, _ => throw new NotSupportedException() };
        int bpp = channels * depth / 8, stride = w * bpp;
        using var z = new ZLibStream(new MemoryStream([.. chunks.Where(c => c.Type == "IDAT").SelectMany(c => c.Data)]), CompressionMode.Decompress);
        var raw = new MemoryStream();
        z.CopyTo(raw);
        var data = raw.ToArray();
        var prev = new byte[stride];
        var outBytes = new byte[h * stride];
        for (int y = 0; y < h; y++)
        {
            int f = data[y * (stride + 1)];
            var line = data.AsSpan(y * (stride + 1) + 1, stride).ToArray();
            for (int x = 0; x < stride; x++)
            {
                int a = x >= bpp ? line[x - bpp] : 0, b = prev[x], c = x >= bpp ? prev[x - bpp] : 0;
                int p = f switch
                {
                    0 => 0, 1 => a, 2 => b, 3 => (a + b) >> 1,
                    4 => Math.Abs(b - c) <= Math.Abs(a - c) && Math.Abs(b - c) <= Math.Abs(a + b - 2 * c) ? a : Math.Abs(a - c) <= Math.Abs(a + b - 2 * c) ? b : c,
                    _ => throw new InvalidDataException(),
                };
                line[x] = (byte)(line[x] + p);
            }
            line.CopyTo(outBytes, y * stride);
            prev = line;
        }
        return depth == 8 ? [.. outBytes.Select(v => (int)v)] : [.. Enumerable.Range(0, outBytes.Length / 2).Select(i => outBytes[2 * i] << 8 | outBytes[2 * i + 1])];
    }

    /// <summary>A test-side OpenEXR reader for single-part scan-line files with no compression or ZIP compression.</summary>
    sealed class Exr
    {
        public int Width, Height, Compression, StoredChunks;
        public List<string> Attributes = [];
        public List<int> PixelTypes = [];
        public Dictionary<string, string> Strings = [];
        public float[] Chromaticities = [];
        public Dictionary<string, float[]> Channels = [];

        public static Exr Read(byte[] d)
        {
            var e = new Exr();
            Assert.That(BinaryPrimitives.ReadInt32LittleEndian(d), Is.EqualTo(20000630), "magic");
            Assert.That(BinaryPrimitives.ReadInt32LittleEndian(d.AsSpan(4)), Is.EqualTo(2), "version 2, single-part scan-line");
            int p = 8;
            var names = new List<string>();
            while (d[p] != 0)
            {
                string Str() { int z = Array.IndexOf(d, (byte)0, p); var s = Encoding.UTF8.GetString(d, p, z - p); p = z + 1; return s; }
                var name = Str();
                var type = Str();
                int size = BinaryPrimitives.ReadInt32LittleEndian(d.AsSpan(p));
                var value = d.AsSpan(p + 4, size).ToArray();
                p += 4 + size;
                e.Attributes.Add(name);
                switch (type)
                {
                    case "chlist":
                        for (int q = 0; value[q] != 0;)
                        {
                            int z = Array.IndexOf(value, (byte)0, q);
                            names.Add(Encoding.ASCII.GetString(value, q, z - q));
                            e.PixelTypes.Add(BinaryPrimitives.ReadInt32LittleEndian(value.AsSpan(z + 1)));
                            q = z + 17;
                        }
                        break;
                    case "compression": e.Compression = value[0]; break;
                    case "box2i" when name == "dataWindow":
                        e.Width = BinaryPrimitives.ReadInt32LittleEndian(value.AsSpan(8)) - BinaryPrimitives.ReadInt32LittleEndian(value) + 1;
                        e.Height = BinaryPrimitives.ReadInt32LittleEndian(value.AsSpan(12)) - BinaryPrimitives.ReadInt32LittleEndian(value.AsSpan(4)) + 1;
                        break;
                    case "chromaticities":
                        e.Chromaticities = [.. Enumerable.Range(0, 8).Select(i => BinaryPrimitives.ReadSingleLittleEndian(value.AsSpan(i * 4)))];
                        break;
                    case "string": e.Strings[name] = Encoding.UTF8.GetString(value); break;
                }
            }
            p++;
            int lines = e.Compression == 3 ? 16 : 1, chunks = (e.Height + lines - 1) / lines;
            foreach (var n in names) e.Channels[n] = new float[e.Width * e.Height];
            for (int c = 0; c < chunks; c++)
            {
                int at = (int)BinaryPrimitives.ReadInt64LittleEndian(d.AsSpan(p + 8 * c));
                int y0 = BinaryPrimitives.ReadInt32LittleEndian(d.AsSpan(at)), size = BinaryPrimitives.ReadInt32LittleEndian(d.AsSpan(at + 4));
                int n = Math.Min(lines, e.Height - y0);
                int rawSize = n * e.Width * e.PixelTypes.Sum(t => t == 1 ? 2 : 4);
                var data = d.AsSpan(at + 8, size).ToArray();
                if (size < rawSize)
                {
                    using var z = new ZLibStream(new MemoryStream(data), CompressionMode.Decompress);
                    var ms = new MemoryStream();
                    z.CopyTo(ms);
                    var t = ms.ToArray();
                    for (int i = 1; i < t.Length; i++) t[i] = (byte)(t[i - 1] + t[i] - 128);
                    data = new byte[t.Length];
                    int half = (t.Length + 1) / 2;
                    for (int i = 0; i < t.Length; i++) data[i] = (i & 1) == 0 ? t[i / 2] : t[half + i / 2];
                }
                else e.StoredChunks++;
                int o = 0;
                for (int y = y0; y < y0 + n; y++)
                    for (int k = 0; k < names.Count; k++)
                        for (int x = 0; x < e.Width; x++)
                        {
                            e.Channels[names[k]][y * e.Width + x] = e.PixelTypes[k] == 1 ? (float)BinaryPrimitives.ReadHalfLittleEndian(data.AsSpan(o)) : BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(o));
                            o += e.PixelTypes[k] == 1 ? 2 : 4;
                        }
            }
            return e;
        }
    }
}
