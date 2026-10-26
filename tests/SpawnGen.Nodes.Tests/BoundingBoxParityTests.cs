using System.IO.Compression;
using System.Text.Json;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Create Bounding Boxes, Crop Image (ImageCropV2), Draw BBoxes, Crop By Bounding Boxes and the tile list nodes against what ComfyUI's own node code
/// gave for the same arguments (Fixtures/bbox/make_bbox.py -> bbox.json.gz, run on torch 2.14 CPU and Pillow 13): pictures pixel for pixel (the tile merge
/// to float rounding), boxes and elements value for value, and the same Python errors. ComfyUI writes text into its previews (number tags, descriptions,
/// detection labels) with a font; SpawnGen has no font engine yet, so the pixels inside those text rectangles are the only ones not compared.
/// </summary>
public class BoundingBoxParityTests
{
    public sealed record Tensor(int[] Shape, float[] Data);

    public sealed record Case(int Index, JsonElement Json, Tensor[] Tensors)
    {
        public string Node => Json.GetProperty("node").GetString()!;
        public override string ToString() => $"{Node} {Json.GetProperty("args").GetRawText()}";
    }

    static (Tensor[] Tensors, List<Case> Cases)? _loaded;

    static (Tensor[] Tensors, List<Case> Cases) Load()
    {
        lock (typeof(BoundingBoxParityTests))
        {
            if (_loaded is { } done) return done;
            var path = Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "bbox", "bbox.json.gz");
            using var gz = new GZipStream(File.OpenRead(path), CompressionMode.Decompress);
            using var doc = JsonDocument.Parse(gz);
            var tensors = doc.RootElement.GetProperty("tensors").EnumerateArray().Select(ReadTensor).ToArray();
            var cases = doc.RootElement.GetProperty("cases").EnumerateArray().Select((c, i) => new Case(i, c.Clone(), tensors)).ToList();
            _loaded = (tensors, cases);
            return _loaded.Value;
        }
    }

    static Tensor ReadTensor(JsonElement t)
    {
        var shape = t.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        if (t.TryGetProperty("u8", out var u8)) return new Tensor(shape, Convert.FromBase64String(u8.GetString()!).Select(b => b / 255f).ToArray());
        var raw = Convert.FromBase64String(t.GetProperty("f32").GetString()!);
        var data = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, data, 0, raw.Length);
        return new Tensor(shape, data);
    }

    static IEnumerable<TestCaseData> Cases(string node) =>
        Load().Cases.Where(c => c.Node == node).Select(c => new TestCaseData(c).SetName($"{node}_{c.Index:D3}"));

    static IEnumerable<TestCaseData> CreateCases() => Cases("CreateBoundingBoxes");
    static IEnumerable<TestCaseData> CropCases() => Cases("ImageCropV2");
    static IEnumerable<TestCaseData> DrawCases() => Cases("DrawBBoxes");
    static IEnumerable<TestCaseData> CropByCases() => Cases("CropByBBoxes");
    static IEnumerable<TestCaseData> SplitCases() => Cases("SplitImageToTileList");
    static IEnumerable<TestCaseData> MergeCases() => Cases("ImageMergeTileList");

    static object? Arg(Case c, string name) =>
        c.Json.GetProperty("args").TryGetProperty(name, out var v) ? JsonLiteral.FromJson(v) : null;

    static ImageBatch? Input(Case c, string name) =>
        c.Json.GetProperty("inputs").TryGetProperty(name, out var i) ? Img(c.Tensors[i.GetInt32()]) : null;

    static ImageBatch Img(Tensor t) => new(t.Shape[0], t.Shape[1], t.Shape[2], t.Shape[3], (float[])t.Data.Clone());

    static JsonElement Out(Case c, int i) => c.Json.GetProperty("out")[i];

    /// <summary>Runs the case; when ComfyUI raised, the port must raise the same Python error (an error torch itself raised only needs to be an error).</summary>
    static void Expect(Case c, Action run)
    {
        if (!c.Json.TryGetProperty("error", out var error))
        {
            run();
            return;
        }
        var ex = Assert.Catch(() => run(), $"ComfyUI raised {error.GetString()}: {c.Json.GetProperty("message").GetString()}");
        if (ex is PyException py)
        {
            Assert.That(py.PyType, Is.EqualTo(error.GetString()), py.Message);
            Assert.That(py.PyMessage, Is.EqualTo(c.Json.GetProperty("message").GetString()));
        }
        else Assert.That(error.GetString(), Is.EqualTo("RuntimeError"), $"{ex!.GetType().Name}: {ex.Message} (ComfyUI: {c.Json.GetProperty("message").GetString()})");
    }

    /// <summary>The picture against ComfyUI's, every value within <paramref name="tolerance"/>, outside the given inclusive rectangles (in every picture of the batch).</summary>
    static void Same(ImageBatch actual, Tensor expected, double tolerance, IReadOnlyList<int[]>? skip = null, string what = "")
    {
        Assert.That(actual.Shape, Is.EqualTo(expected.Shape), "shape " + what);
        int h = actual.Shape[1], w = actual.Shape[2], c = actual.Shape[3];
        double worst = 0;
        int at = -1, compared = 0;
        for (int i = 0; i < expected.Data.Length; i++)
        {
            int p = i / c, x = p % w, y = p / w % h;
            if (skip is not null && skip.Any(r => x >= r[0] && x <= r[2] && y >= r[1] && y <= r[3])) continue;
            compared++;
            double d = Math.Abs(actual.Data[i] - expected.Data[i]);
            if (!(d <= worst)) { worst = d; at = i; }
        }
        Assert.That(worst, Is.LessThanOrEqualTo(tolerance), $"{what} worst at {at}: {(at >= 0 ? actual.Data[at] : 0)} vs {(at >= 0 ? expected.Data[at] : 0)}");
        if (skip is { Count: > 0 }) Assert.That(compared, Is.GreaterThan(expected.Data.Length / 3), "text covers most of the picture: the case tests too little");
    }

    static List<int[]> TextRects(Case c) =>
        c.Json.TryGetProperty("text_rects", out var r) ? r.EnumerateArray().Select(a => a.EnumerateArray().Select(v => v.GetInt32()).ToArray()).ToList() : [];

    static void SameValue(object? actual, JsonElement expected, string what) =>
        Assert.That(JsonLiteral.ToJson(actual), Is.EqualTo(JsonLiteral.ToJson(JsonLiteral.FromJson(expected))), what);

    [TestCaseSource(nameof(CreateCases))]
    public void CreateBoundingBoxes_matches_ComfyUI(Case c)
    {
        Expect(c, () =>
        {
            var (preview, bboxes, elements, ui) = new CreateBoundingBoxes().ExecuteAsync(Input(c, "background"), Arg(c, "bboxes"),
                (long)Arg(c, "width")!, (long)Arg(c, "height")!, Arg(c, "editor_state"), Arg(c, "last_incoming")).GetAwaiter().GetResult();
            Same(preview, c.Tensors[Out(c, 0).GetProperty("tensor").GetInt32()], 0, TextRects(c), "preview");
            SameValue(bboxes, Out(c, 1).GetProperty("value"), "bboxes");
            SameValue(elements, Out(c, 2).GetProperty("value"), "elements");
            Assert.That(ui["dims"], Is.EqualTo(new[] { (long)Arg(c, "width")!, (long)Arg(c, "height")! }));
        });
    }

    [TestCaseSource(nameof(CropCases))]
    public void ImageCropV2_matches_ComfyUI(Case c) =>
        Expect(c, () => Same(ImageCropV2.Crop(Input(c, "image")!, Arg(c, "crop_region")), c.Tensors[Out(c, 0).GetProperty("tensor").GetInt32()], 0));

    [TestCaseSource(nameof(DrawCases))]
    public void DrawBBoxes_matches_ComfyUI(Case c) =>
        Expect(c, () =>
        {
            var image = Input(c, "image");
            // A picture of out-of-range values keeps them in the fixture as float32, as given.
            Same(new DrawBBoxes().Execute(Arg(c, "bboxes")!, image), c.Tensors[Out(c, 0).GetProperty("tensor").GetInt32()], 0, TextRects(c));
        });

    [TestCaseSource(nameof(CropByCases))]
    public void CropByBBoxes_matches_ComfyUI(Case c) =>
        Expect(c, () => Same(new CropByBBoxes().Execute(Input(c, "image")!, Arg(c, "bboxes")!, (long)Arg(c, "output_width")!, (long)Arg(c, "output_height")!,
            (long)Arg(c, "padding")!, (string)Arg(c, "keep_aspect")!), c.Tensors[Out(c, 0).GetProperty("tensor").GetInt32()], 2e-6));

    [TestCaseSource(nameof(SplitCases))]
    public void SplitImageToTileList_matches_ComfyUI(Case c) =>
        Expect(c, () =>
        {
            var tiles = new SplitImageToTileList().Execute(Input(c, "image")!, (long)Arg(c, "tile_width")!, (long)Arg(c, "tile_height")!, (long)Arg(c, "overlap")!);
            var expected = Out(c, 0).GetProperty("tensors").EnumerateArray().Select(t => c.Tensors[t.GetInt32()]).ToList();
            Assert.That(tiles, Has.Count.EqualTo(expected.Count), "tile count");
            for (int i = 0; i < tiles.Count; i++) Same(tiles[i], expected[i], 0, what: $"tile {i}");
        });

    [TestCaseSource(nameof(MergeCases))]
    public void ImageMergeTileList_matches_ComfyUI(Case c) =>
        Expect(c, () =>
        {
            var tiles = c.Json.GetProperty("tiles").EnumerateArray().Select(t => Img(c.Tensors[t.GetInt32()])).ToList();
            var merged = new ImageMergeTileList().Execute(tiles, [(long)Arg(c, "final_width")!], [(long)Arg(c, "final_height")!], [(long)Arg(c, "overlap")!]);
            // torch's sin (SLEEF) and .NET's differ in the last bit; the rest is the same float32 arithmetic in the same order.
            Same(merged, c.Tensors[Out(c, 0).GetProperty("tensor").GetInt32()], 2e-6);
        });

    [Test]
    public void The_fixture_covers_every_node_and_its_errors()
    {
        var cases = Load().Cases;
        foreach (var node in new[] { "CreateBoundingBoxes", "ImageCropV2", "DrawBBoxes", "CropByBBoxes", "SplitImageToTileList", "ImageMergeTileList" })
            Assert.That(cases.Count(c => c.Node == node && !c.Json.TryGetProperty("error", out _)), Is.GreaterThanOrEqualTo(4), node);
        Assert.That(cases.Count(c => c.Json.TryGetProperty("error", out _)), Is.GreaterThanOrEqualTo(15));
        Assert.That(cases.Count(c => TextRects(c).Count > 0), Is.GreaterThanOrEqualTo(10), "previews with text");
    }
}
