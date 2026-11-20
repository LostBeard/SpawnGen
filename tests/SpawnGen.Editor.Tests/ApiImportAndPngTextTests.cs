using SpawnGen.Editor;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Editor.Tests;

public class ApiPromptImportTests
{
    /// <summary>The API form of ComfyUI's stock workflow: numeric ids, links as [node, output], widgets as plain inputs.</summary>
    const string Api = """
    {
      "4": { "class_type": "CheckpointLoaderSimple", "inputs": { "ckpt_name": "sd_turbo" } },
      "5": { "class_type": "EmptyLatentImage", "inputs": { "width": 512, "height": 512, "batch_size": 1 } },
      "6": { "class_type": "CLIPTextEncode", "inputs": { "text": "a red apple", "clip": ["4", 1] } },
      "7": { "class_type": "CLIPTextEncode", "inputs": { "text": "", "clip": ["4", 1] } },
      "3": { "class_type": "KSampler", "inputs": { "seed": 7, "steps": 4, "cfg": 1.5, "sampler_name": "euler_ancestral", "scheduler": "normal",
              "denoise": 1.0, "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0] } },
      "8": { "class_type": "VAEDecode", "inputs": { "samples": ["3", 0], "vae": ["4", 2] } },
      "9": { "class_type": "SaveImage", "inputs": { "filename_prefix": "api", "images": ["8", 0] } }
    }
    """;

    static EditorDocument Imported(out IReadOnlyList<string> warnings, string json = Api)
    {
        var doc = new EditorDocument(Fixtures.Registry());
        warnings = ApiPromptImport.Load(json, doc);
        return doc;
    }

    [Test]
    public void Opens_an_API_prompt_as_a_workflow_with_ids_widgets_and_links()
    {
        var doc = Imported(out var warnings);
        Assert.That(warnings, Is.Empty);
        Assert.That((doc.Graph.Nodes.Count, doc.Graph.Links.Count), Is.EqualTo((7, 9)));
        var sampler = doc.Graph.GetNode(3);
        Assert.That(sampler.Widgets["steps"], Is.EqualTo(4L));
        Assert.That(sampler.Widgets["cfg"], Is.EqualTo(1.5));
        Assert.That(sampler.Widgets["sampler_name"], Is.EqualTo("euler_ancestral"));
        Assert.That(doc.Graph.GetNode(6).Widgets["text"], Is.EqualTo("a red apple"));
        var link = doc.Graph.LinkInto(8, "vae")!;
        Assert.That((link.FromNode, link.FromOutput), Is.EqualTo((4, 2)));
        Assert.That(doc.CanUndo, Is.False);
    }

    [Test]
    public void Lays_nodes_out_left_to_right_by_dependency_without_overlap()
    {
        var doc = Imported(out _);
        double X(int id) => doc.Graph.GetNode(id).X;
        Assert.That(X(4), Is.LessThan(X(6)), "the loader is left of its consumers");
        Assert.That(X(6), Is.LessThan(X(3)));
        Assert.That(X(3), Is.LessThan(X(8)));
        Assert.That(X(8), Is.LessThan(X(9)));

        var hits = new HitTester(doc);
        var rects = doc.Graph.Nodes.Select(n => (n.Id, Bounds: hits.Geometry(n).Bounds)).ToList();
        foreach (var a in rects)
            foreach (var b in rects.Where(b => b.Id > a.Id))
                Assert.That(a.Bounds.Intersects(b.Bounds), Is.False, $"nodes {a.Id} and {b.Id} overlap");
    }

    [Test]
    public void Compiling_the_imported_workflow_gives_back_the_same_prompt()
    {
        var doc = Imported(out _);
        var original = PromptJson.Parse(Api);
        var again = GraphCompiler.Compile(doc.Graph, doc.Registry).Graph;
        Assert.That(again.Nodes.Keys.Order(), Is.EqualTo(original.Nodes.Keys.Order()));
        foreach (var (id, node) in original.Nodes)
        {
            Assert.That(again.Nodes[id].ClassType, Is.EqualTo(node.ClassType));
            foreach (var (name, input) in node.Inputs)
                Assert.That(again.Nodes[id].Inputs[name], Is.EqualTo(input), $"node {id} input {name}");
        }
    }

    [Test]
    public void Unknown_node_types_and_dangling_links_are_reported_not_fatal()
    {
        var doc = Imported(out var warnings, """
        { "1": { "class_type": "FromAPack", "inputs": { "x": 1 } },
          "2": { "class_type": "SaveImage", "inputs": { "images": ["99", 0] } } }
        """);
        Assert.That(warnings, Has.Some.Contain("FromAPack"));
        Assert.That(warnings, Has.Some.Contain("missing node 99"));
        Assert.That(doc.Graph.Nodes.Count, Is.EqualTo(2));
    }

    [Test]
    public void Looks_tells_a_workflow_from_an_API_prompt()
    {
        Assert.That(ApiPromptImport.Looks(Api), Is.True);
        Assert.That(ApiPromptImport.Looks(ComfySamples.DefaultWorkflow), Is.False);
        Assert.That(ApiPromptImport.Looks("[1,2]"), Is.False);
        Assert.That(ApiPromptImport.Looks("not json"), Is.False);
    }
}

public class PngTextTests
{
    static ImageBatch Tiny() => new(1, 4, 4, 3);

    static byte[] BigEndian(uint v) => [(byte)(v >> 24), (byte)(v >> 16), (byte)(v >> 8), (byte)v];

    [Test]
    public void Text_written_into_a_PNG_reads_back_and_the_picture_still_decodes()
    {
        var png = PngWriter.Encode(Tiny(), 0, new Dictionary<string, string> { ["prompt"] = """{"1":{"class_type":"X"}}""", ["workflow"] = "{\"nodes\":[]}" });
        var text = PngReader.ReadText(png);
        Assert.That(text["prompt"], Is.EqualTo("""{"1":{"class_type":"X"}}"""));
        Assert.That(text["workflow"], Is.EqualTo("{\"nodes\":[]}"));
        Assert.That(PngReader.Decode(png).Width, Is.EqualTo(4));
    }

    [Test]
    public void Text_outside_Latin1_survives_as_UTF8()
    {
        var value = "a prompt with 日本語 and an emoji 🍎 and “quotes”";
        var png = PngWriter.Encode(Tiny(), 0, new Dictionary<string, string> { ["parameters"] = value });
        Assert.That(PngReader.ReadText(png)["parameters"], Is.EqualTo(value));
    }

    /// <summary>
    /// The page looks for a dropped or opened picture's workflow chunk by chunk (Studio.OpenFileAsync over the browser File's slices), so the
    /// pixels never come to .NET: the same text as <see cref="PngReader.ReadText"/>, including a text chunk after the picture data, and not one
    /// byte of an IDAT chunk's data read.
    /// </summary>
    [Test]
    public async Task Text_read_piece_by_piece_matches_and_never_reads_the_pixels()
    {
        var rng = new Random(3);
        var picture = new ImageBatch(1, 256, 256, 3, Enumerable.Range(0, 256 * 256 * 3).Select(_ => (float)rng.NextDouble()).ToArray());
        var png = PngWriter.Encode(picture, 0, new Dictionary<string, string> { ["workflow"] = "{\"nodes\":[]}", ["prompt"] = "{}" });
        // a tEXt chunk after the picture data, just before IEND (allowed by the PNG spec; ReadText finds it too)
        var late = System.Text.Encoding.ASCII.GetBytes("tEXt").Concat(System.Text.Encoding.Latin1.GetBytes("late\0after the pixels")).ToArray();
        uint crc = 0xFFFFFFFF;
        foreach (var b in late) { crc ^= b; for (int k = 0; k < 8; k++) crc = (crc & 1) != 0 ? 0xEDB88320 ^ (crc >> 1) : crc >> 1; }
        var chunk = BigEndian((uint)(late.Length - 4)).Concat(late).Concat(BigEndian(crc ^ 0xFFFFFFFF)).ToArray();
        png = png.Take(png.Length - 12).Concat(chunk).Concat(png.Skip(png.Length - 12)).ToArray();

        var idat = new List<(long From, long To)>();
        for (long pos = 8; pos + 12 <= png.Length;)
        {
            long len = (png[pos] << 24) | (png[pos + 1] << 16) | (png[pos + 2] << 8) | png[pos + 3];
            if (System.Text.Encoding.ASCII.GetString(png, (int)pos + 4, 4) == "IDAT") idat.Add((pos + 8, pos + 8 + len));
            pos += 12 + len;
        }
        Assert.That(idat, Is.Not.Empty);

        long readBytes = 0;
        var text = await PngReader.ReadTextAsync(png.Length, (offset, count) =>
        {
            foreach (var (from, to) in idat)
                Assert.That(offset + count <= from || offset >= to, Is.True, $"read {offset}+{count} overlaps picture data {from}..{to}");
            readBytes += count;
            return Task.FromResult(png.AsSpan((int)offset, count).ToArray());
        });
        Assert.That(text, Is.EqualTo(PngReader.ReadText(png)));
        Assert.That(text!["late"], Is.EqualTo("after the pixels"));
        Assert.That(text["workflow"], Is.EqualTo("{\"nodes\":[]}"));
        Assert.That(readBytes, Is.LessThan(png.Length / 10), "only headers and text were read");
        Assert.That(await PngReader.ReadTextAsync(3, (o, c) => Task.FromResult(new byte[c])), Is.Null, "too short to be a PNG");
        Assert.That(await PngReader.ReadTextAsync(20, (o, c) => Task.FromResult(new byte[c])), Is.Null, "not a PNG");
    }

    [Test]
    public void A_PNG_without_text_and_a_non_PNG_give_an_empty_map()
    {
        Assert.That(PngReader.ReadText(PngWriter.Encode(Tiny())), Is.Empty);
        Assert.That(PngReader.ReadText([1, 2, 3]), Is.Empty);
    }

    [Test]
    public void Compressed_text_chunks_from_other_tools_are_read()
    {
        // A zTXt chunk (what some tools write), built by hand around a real picture.
        var png = PngWriter.Encode(Tiny());
        var compressed = new MemoryStream();
        using (var z = new System.IO.Compression.ZLibStream(compressed, System.IO.Compression.CompressionLevel.Optimal, true))
            z.Write(System.Text.Encoding.Latin1.GetBytes("hello workflow"));
        var body = new List<byte>(System.Text.Encoding.Latin1.GetBytes("workflow")) { 0, 0 };
        body.AddRange(compressed.ToArray());

        var chunk = new List<byte>();
        chunk.AddRange(BigEndian((uint)body.Count));
        var typed = System.Text.Encoding.ASCII.GetBytes("zTXt").Concat(body).ToArray();
        chunk.AddRange(typed);
        uint crc = 0xFFFFFFFF;
        foreach (var b in typed) { crc ^= b; for (int k = 0; k < 8; k++) crc = (crc & 1) != 0 ? 0xEDB88320 ^ (crc >> 1) : crc >> 1; }
        chunk.AddRange(BigEndian(crc ^ 0xFFFFFFFF));

        // Insert right after the 33-byte signature + IHDR.
        var withText = png.Take(33).Concat(chunk).Concat(png.Skip(33)).ToArray();
        Assert.That(PngReader.ReadText(withText)["workflow"], Is.EqualTo("hello workflow"));
        Assert.That(PngReader.Decode(withText).Height, Is.EqualTo(4));
    }
}
