using System.Buffers.Binary;
using System.Text;
using ILGPU;
using ILGPU.Runtime.CPU;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Export PSD: a composition written with SpawnDev.Codecs' PSD writer and read back here with a small reader of the PSD layout (header, layer
/// records, PackBits channels, the flattened picture): every layer's rectangle, name, blend key, opacity, flags and pixels as ComfyUI's
/// export stores them (measured: probe-comfy-compositor-psd*.js).
/// </summary>
public class CompositorPsdTests
{
    sealed record ReadLayer(string Name, string UnicodeName, int Left, int Top, int Right, int Bottom, int[] ChannelIds, string Key, byte Opacity, byte Flags, byte[] Rgba);
    sealed record ReadPsd(int Channels, int Width, int Height, int Depth, int Mode, List<ReadLayer> Layers, byte[][] Composite);

    static ReadPsd Read(byte[] d)
    {
        int p = 0;
        int U16() { int v = BinaryPrimitives.ReadUInt16BigEndian(d.AsSpan(p)); p += 2; return v; }
        int I16() { int v = BinaryPrimitives.ReadInt16BigEndian(d.AsSpan(p)); p += 2; return v; }
        int I32() { int v = BinaryPrimitives.ReadInt32BigEndian(d.AsSpan(p)); p += 4; return v; }
        string Ascii(int n) { var s = Encoding.ASCII.GetString(d, p, n); p += n; return s; }
        byte[] Plane(int w, int h, int end)
        {
            int compression = U16();
            var plane = new byte[w * h];
            if (compression == 0) { d.AsSpan(p, w * h).CopyTo(plane); p = end; return plane; }
            Assert.That(compression, Is.EqualTo(1), "RLE");
            var lengths = new int[h];
            for (int y = 0; y < h; y++) lengths[y] = U16();
            for (int y = 0; y < h; y++)
            {
                int stop = p + lengths[y], x = 0;
                while (p < stop)
                {
                    sbyte n = (sbyte)d[p++];
                    if (n >= 0) { for (int k = 0; k <= n; k++) plane[y * w + x++] = d[p++]; }
                    else if (n != -128) { byte v = d[p++]; for (int k = 0; k < 1 - n; k++) plane[y * w + x++] = v; }
                }
                Assert.That(x, Is.EqualTo(w), $"row {y} unpacks to the layer's width");
            }
            if (end >= 0) Assert.That(p, Is.EqualTo(end), "a channel's data is as long as its record says");
            return plane;
        }

        Assert.That(Ascii(4), Is.EqualTo("8BPS"));
        Assert.That(U16(), Is.EqualTo(1), "version");
        p += 6;
        int channels = U16(), height = I32(), width = I32(), depth = U16(), mode = U16();
        // colour mode data, then the image resources
        p = 26; p += 4 + BinaryPrimitives.ReadInt32BigEndian(d.AsSpan(p));
        p += 4 + BinaryPrimitives.ReadInt32BigEndian(d.AsSpan(p));
        int layerAndMask = I32(), sectionEnd = p + layerAndMask;
        var layers = new List<ReadLayer>();
        if (layerAndMask > 0)
        {
            int info = I32(), infoEnd = p + info, count = Math.Abs(I16());
            var records = new List<(int T, int L, int B, int R, (int Id, int Size)[] Ch, string Key, byte Op, byte Flags, string Name, string Uni)>();
            for (int k = 0; k < count; k++)
            {
                int top = I32(), left = I32(), bottom = I32(), right = I32(), nc = U16();
                var ch = new (int, int)[nc];
                for (int c = 0; c < nc; c++) ch[c] = (I16(), I32());
                Assert.That(Ascii(4), Is.EqualTo("8BIM"));
                string key = Ascii(4);
                byte opacity = d[p], flags = d[p + 2]; p += 4;
                int extra = I32(), extraEnd = p + extra;
                p += 4 + BinaryPrimitives.ReadInt32BigEndian(d.AsSpan(p));   // layer mask data
                p += 4 + BinaryPrimitives.ReadInt32BigEndian(d.AsSpan(p));   // blending ranges
                int len = d[p];
                string name = Encoding.Latin1.GetString(d, p + 1, len);
                p += 1 + len; p += (4 - (1 + len) % 4) % 4;
                string uni = "";
                while (p < extraEnd)
                {
                    p += 4; string tag = Ascii(4); int size = I32();
                    if (tag == "luni") { int chars = BinaryPrimitives.ReadInt32BigEndian(d.AsSpan(p)); uni = Encoding.BigEndianUnicode.GetString(d, p + 4, chars * 2).TrimEnd('\0'); }
                    p += size + (size & 1);
                }
                p = extraEnd;
                records.Add((top, left, bottom, right, ch, key, opacity, flags, name, uni));
            }
            foreach (var r in records)
            {
                int w = r.R - r.L, h = r.B - r.T;
                var rgba = new byte[w * h * 4];
                for (int i = 3; i < rgba.Length; i += 4) rgba[i] = 255;
                foreach (var (id, size) in r.Ch)
                {
                    var plane = Plane(w, h, p + size);
                    int slot = id == -1 ? 3 : id;
                    for (int i = 0; i < w * h; i++) rgba[i * 4 + slot] = plane[i];
                }
                layers.Add(new ReadLayer(r.Name, r.Uni, r.L, r.T, r.R, r.B, [.. r.Ch.Select(c => c.Id)], r.Key, r.Op, r.Flags, rgba));
            }
        }
        p = sectionEnd;
        // the flattened picture: one compression word, then every row length of every channel, then the rows
        int comp = U16();
        var planes = new byte[channels][];
        if (comp == 0) for (int c = 0; c < channels; c++) { planes[c] = d.AsSpan(p, width * height).ToArray(); p += width * height; }
        else
        {
            var lengths = new int[channels * height];
            for (int i = 0; i < lengths.Length; i++) lengths[i] = U16();
            for (int c = 0; c < channels; c++)
            {
                planes[c] = new byte[width * height];
                for (int y = 0; y < height; y++)
                {
                    int stop = p + lengths[c * height + y], x = 0;
                    while (p < stop)
                    {
                        sbyte n = (sbyte)d[p++];
                        if (n >= 0) { for (int k = 0; k <= n; k++) planes[c][y * width + x++] = d[p++]; }
                        else if (n != -128) { byte v = d[p++]; for (int k = 0; k < 1 - n; k++) planes[c][y * width + x++] = v; }
                    }
                }
            }
        }
        Assert.That(p, Is.EqualTo(d.Length), "the file ends with the flattened picture");
        return new ReadPsd(channels, width, height, depth, mode, layers, planes);
    }

    static (byte[] Rgba, int Width, int Height) Picture(int w, int h, int seed)
    {
        var px = new byte[w * h * 4];
        var random = new Random(seed);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int i = (y * w + x) * 4;
                // runs and noise, so both PackBits cases are met; a transparent corner
                px[i] = (byte)(x < w / 2 ? 200 : random.Next(256)); px[i + 1] = (byte)(y * 255 / Math.Max(1, h - 1)); px[i + 2] = (byte)random.Next(256);
                px[i + 3] = (byte)(x < 3 && y < 3 ? 0 : x % 5 == 0 ? 128 : 255);
            }
        return (px, w, h);
    }

    static DictValue Layer(string name, double x, double y, double w, double h, double rotation = 0, double opacity = 1, string blend = "normal", bool visible = true) => new()
    {
        ["name"] = name, ["visible"] = visible, ["opacity"] = opacity, ["blend"] = blend,
        ["transform"] = new DictValue { ["x"] = x, ["y"] = y, ["w"] = w, ["h"] = h, ["rotation"] = rotation }, ["flipH"] = false, ["flipV"] = false,
    };

    static DictValue Composition(int w, int h, bool background, params DictValue[] layers) => new()
    {
        ["version"] = 1L, ["canvas"] = new DictValue { ["w"] = (long)w, ["h"] = (long)h }, ["inputs"] = new ArrayValue(),
        ["background"] = new DictValue { ["color"] = "#336699", ["opacity"] = 1L, ["visible"] = background }, ["layers"] = new ArrayValue(layers.Cast<object?>()),
    };

    [Test]
    public void The_blend_ids_map_to_the_writers_modes_one_to_one()
    {
        var modes = CompositorBlend.ModeNames.Select(CompositorPsd.BlendMode).ToList();
        Assert.Multiple(() =>
        {
            Assert.That(modes.Distinct().Count(), Is.EqualTo(26));
            Assert.That(modes.Select(m => (int)m), Is.EqualTo(Enumerable.Range(0, 26)), "the writer's enum is in the compositor's order");
            Assert.That(() => CompositorPsd.BlendMode("nope"), Throws.ArgumentException);
            // measured on ComfyUI's export: 0.5 -> 128, 0.6 -> 153, 0.7 -> 179, 0.8 -> 204, 0.9 -> 230
            Assert.That(new[] { 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0 }.Select(CompositorPsd.OpacityByte), Is.EqualTo(new byte[] { 128, 153, 179, 204, 230, 255, 0 }));
            Assert.That(CompositorPsd.FileName(new DateTime(2026, 11, 25, 3, 18, 44)), Is.EqualTo("spawngen-layers-20261125-031844.psd"));
        });
    }

    [TestCase(false)]
    [TestCase(true)]
    public async Task A_composition_is_written_as_a_layered_PSD_that_reads_back_exactly(bool background)
    {
        var a = Picture(40, 30, 1);
        var b = Picture(16, 16, 2);
        var export = ImageCompositor.Export([a, b], Composition(48, 36, background,
            Layer("coat", 0, 0, 40, 30), Layer("Ünïcødé 層 ✓", -6, -4, 16, 16, opacity: 0.8, blend: "multiply")));
        using var context = Context.Create(builder => builder.CPU());
        using var accelerator = context.CreateCPUAccelerator(0);
        var bytes = await CompositorPsd.EncodeAsync(accelerator, export);
        var psd = Read(bytes);
        Assert.Multiple(() =>
        {
            Assert.That((psd.Width, psd.Height, psd.Depth, psd.Mode), Is.EqualTo((48, 36, 8, 3)));
            Assert.That(psd.Channels, Is.EqualTo(background ? 3 : 4), "the merged transparency is there only when the picture has one");
            Assert.That(psd.Layers.Select(x => x.UnicodeName), Is.EqualTo(new[] { "Background", "coat", "Ünïcødé 層 ✓" }));
            var bg = psd.Layers[0];
            Assert.That((bg.Left, bg.Top, bg.Right, bg.Bottom, bg.Key, bg.Opacity), Is.EqualTo((0, 0, 48, 36, "norm", (byte)255)));
            Assert.That(bg.Flags & 2, Is.EqualTo(background ? 0 : 2), "bit 1 set = hidden");
            var coat = psd.Layers[1];
            Assert.That((coat.Left, coat.Top, coat.Right, coat.Bottom, coat.Key, coat.Opacity, coat.Flags & 2), Is.EqualTo((0, 0, 40, 30, "norm", (byte)255, 0)));
            Assert.That(coat.ChannelIds, Is.EqualTo(new[] { -1, 0, 1, 2 }));
            var top = psd.Layers[2];
            Assert.That((top.Left, top.Top, top.Right, top.Bottom, top.Key, top.Opacity), Is.EqualTo((-6, -4, 10, 12, "mul ", (byte)204)));
            // pixels: alpha everywhere, colour wherever the pixel is not fully transparent (the colour under alpha 0 is not defined)
            foreach (var (read, source) in new[] { (coat, a.Rgba), (top, b.Rgba) })
                for (int i = 0; i < source.Length; i += 4)
                {
                    if (read.Rgba[i + 3] != source[i + 3]) { Assert.Fail($"{read.UnicodeName}: alpha differs at pixel {i / 4}"); break; }
                    if (source[i + 3] != 0 && !read.Rgba.AsSpan(i, 3).SequenceEqual(source.AsSpan(i, 3))) { Assert.Fail($"{read.UnicodeName}: colour differs at pixel {i / 4}"); break; }
                }
        });
        // the flattened picture is the run's
        for (int i = 0; i < 48 * 36; i++)
        {
            if (!background && psd.Composite[3][i] != export.Composite[i * 4 + 3]) { Assert.Fail($"merged alpha differs at pixel {i}"); break; }
            if (background && (psd.Composite[0][i] != export.Composite[i * 4] || psd.Composite[1][i] != export.Composite[i * 4 + 1] || psd.Composite[2][i] != export.Composite[i * 4 + 2]))
            { Assert.Fail($"flattened colour differs at pixel {i}"); break; }
        }
    }
}
