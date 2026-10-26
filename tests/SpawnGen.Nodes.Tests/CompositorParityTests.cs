using System.IO.Compression;
using System.Text.Json;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Add Layer, Layers From Bounding Boxes and Create Layered Image against what ComfyUI's own compositor code gave for the same arguments
/// (Fixtures/bbox/make_compositor.py -> compositor.json.gz, torch 2.14 CPU, numpy 2.4, Pillow 13): the layer documents item by item, the composite and its
/// mask to float32 rounding (every blend mode, turns, flips, resizes, masks, saved compositions), the layer fingerprints the editor matches saved
/// compositions by, and the same Python errors.
/// </summary>
public class CompositorParityTests
{
    public sealed record Tensor(int[] Shape, float[] Data);

    public sealed record Case(int Index, JsonElement Json, Tensor[] Tensors)
    {
        public string Node => Json.GetProperty("node").GetString()!;
        public override string ToString() => $"{Node} {Index}";
    }

    static List<Case>? _cases;
    static (JsonElement Json, Tensor[] Tensors) _blend;

    static List<Case> Load()
    {
        lock (typeof(CompositorParityTests))
        {
            if (_cases is not null) return _cases;
            var path = Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "bbox", "compositor.json.gz");
            using var gz = new GZipStream(File.OpenRead(path), CompressionMode.Decompress);
            using var doc = JsonDocument.Parse(gz);
            var tensors = doc.RootElement.GetProperty("tensors").EnumerateArray().Select(ReadTensor).ToArray();
            _blend = (doc.RootElement.GetProperty("blend").Clone(), tensors);
            return _cases = doc.RootElement.GetProperty("cases").EnumerateArray().Select((c, i) => new Case(i, c.Clone(), tensors)).ToList();
        }
    }

    static Tensor BlendTensor(JsonElement index) => _blend.Tensors[index.GetInt32()];

    /// <summary>Equal within <paramref name="tolerance"/> relative to the size of the value (NaN equals NaN, an infinity itself).</summary>
    static void Close(float[] actual, float[] expected, double tolerance, string what)
    {
        Assert.That(actual.Length, Is.EqualTo(expected.Length), what);
        for (int k = 0; k < expected.Length; k++)
        {
            float a = actual[k], e = expected[k];
            if (float.IsNaN(e) || float.IsInfinity(e)) { Assert.That(a, Is.EqualTo(e), $"{what} [{k}]"); continue; }
            Assert.That(Math.Abs(a - e), Is.LessThanOrEqualTo(tolerance * Math.Max(1, Math.Abs(e))), $"{what} [{k}]: {a} vs {e}");
        }
    }

    [Test]
    public void Every_blend_mode_computes_what_compositor_blend_computes_on_edge_values()
    {
        Load();
        var (b, _) = _blend;
        var i = BlendTensor(b.GetProperty("i")).Data;
        var l = BlendTensor(b.GetProperty("l")).Data;
        foreach (var mode in b.GetProperty("modes").EnumerateObject())
        {
            var actual = new float[i.Length];
            for (int p = 0; p < i.Length; p += 3) CompositorBlend.BlendPixel(mode.Name, i.AsSpan(p, 3), l.AsSpan(p, 3), actual.AsSpan(p, 3));
            Close(actual, BlendTensor(mode.Value).Data, 0, mode.Name); // the same float32 operations in the same order
        }
        Assert.That(b.GetProperty("modes").EnumerateObject().Select(m => m.Name), Is.EquivalentTo(CompositorBlend.ModeNames));
    }

    [Test]
    public void The_sRGB_curves_and_the_composites_match_compositor_blend()
    {
        Load();
        var (b, _) = _blend;
        var ramp = BlendTensor(b.GetProperty("ramp")).Data;
        // numpy's float32 power is the C runtime's powf, as MathF.Pow.
        Close(ramp.Select(CompositorBlend.SrgbToLinear).ToArray(), BlendTensor(b.GetProperty("to_linear")).Data, 1.5e-7, "srgb_to_linear");
        Close(ramp.Select(CompositorBlend.LinearToSrgb).ToArray(), BlendTensor(b.GetProperty("to_srgb")).Data, 1.5e-7, "linear_to_srgb");
        var back = BlendTensor(b.GetProperty("back")).Data;
        var layer = BlendTensor(b.GetProperty("layer")).Data;
        foreach (var c in b.GetProperty("composites").EnumerateArray())
        {
            var canvas = (float[])back.Clone();
            CompositorBlend.Composite(c.GetProperty("mode").GetString()!, canvas, 0, 0, layer, 0, 0, back.Length / 4, 1, c.GetProperty("opacity").GetDouble());
            // Linear modes are plain float32 arithmetic (exact); perceptual ones go through the curves.
            var (linear, _) = CompositorBlend.Resolve(c.GetProperty("mode").GetString()!);
            Close(canvas, BlendTensor(c.GetProperty("out")).Data, linear ? 0 : 1.5e-7, $"{c.GetProperty("mode").GetString()} at {c.GetProperty("opacity").GetDouble()}");
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
        Load().Where(c => c.Node == node).Select(c => new TestCaseData(c).SetName($"{node}_{c.Index:D3}"));

    static IEnumerable<TestCaseData> AddCases() => Cases("AddLayer");
    static IEnumerable<TestCaseData> BoxCases() => Cases("LayersFromBoundingBoxes");
    static IEnumerable<TestCaseData> CompositeCases() => Cases("ImageCompositor");

    static Tensor T(Case c, JsonElement e) => c.Tensors[e.GetProperty("tensor").GetInt32()];
    static ImageBatch Img(Tensor t) => new(t.Shape[0], t.Shape[1], t.Shape[2], t.Shape[3], (float[])t.Data.Clone());
    static MaskBatch Mask(Tensor t) => new(t.Shape[0], t.Shape[1], t.Shape[2], (float[])t.Data.Clone());

    static bool Present(JsonElement e, string name, out JsonElement v) => e.TryGetProperty(name, out v) && v.ValueKind != JsonValueKind.Null;

    /// <summary>A LAYERS document as the fixture keeps it (ComfyUI's dict, pictures by tensor index).</summary>
    static LayerDocument? Doc(Case c, JsonElement e)
    {
        if (e.ValueKind == JsonValueKind.Null) return null;
        (int, int)? canvas = Present(e, "canvas", out var cv) ? (cv[0].GetInt32(), cv[1].GetInt32()) : null;
        var layers = e.GetProperty("layers").EnumerateArray().Select(i => new LayerItem(Img(T(c, i.GetProperty("image"))))
        {
            Type = Present(i, "type", out var type) ? type.GetString()! : "raster",
            X = Present(i, "x", out var x) ? x.GetInt64() : 0,
            Y = Present(i, "y", out var y) ? y.GetInt64() : 0,
            ZIndex = Present(i, "z_index", out var z) ? z.GetInt64() : 0,
            Mask = Present(i, "mask", out var m) ? Mask(T(c, m)) : null,
            Name = Present(i, "name", out var n) ? n.GetString() : null,
            Opacity = Present(i, "opacity", out var o) ? o.GetDouble() : null,
            BlendMode = Present(i, "blend_mode", out var b) ? b.GetString() : null,
            Visible = Present(i, "visible", out var vis) ? vis.GetBoolean() : null,
            FlipH = Present(i, "flip_h", out var fh) && fh.GetBoolean(),
            FlipV = Present(i, "flip_v", out var fv) && fv.GetBoolean(),
            Rotation = Present(i, "rotation", out var r) ? r.GetDouble() : null,
            W = Present(i, "w", out var w) ? w.GetInt64() : null,
            H = Present(i, "h", out var h) ? h.GetInt64() : null,
        }).ToList();
        return new LayerDocument { Version = e.GetProperty("version").GetInt32(), Canvas = canvas, Layers = layers };
    }

    static object? Arg(Case c, string name) => c.Json.GetProperty("args").TryGetProperty(name, out var v) ? JsonLiteral.FromJson(v) : null;

    static void Same(HostTensor actual, Tensor expected, double tolerance, string what)
    {
        Assert.That(actual.Shape, Is.EqualTo(expected.Shape), "shape " + what);
        double worst = 0;
        int at = -1;
        for (int i = 0; i < expected.Data.Length; i++)
        {
            double d = Math.Abs(actual.Data[i] - expected.Data[i]);
            if (!(d <= worst)) { worst = d; at = i; }
        }
        Assert.That(worst, Is.LessThanOrEqualTo(tolerance), $"{what} worst at {at}: {(at >= 0 ? actual.Data[at] : 0)} vs {(at >= 0 ? expected.Data[at] : 0)}");
    }

    /// <summary>The node's document against ComfyUI's, item by item: only the keys ComfyUI wrote are set, pictures and masks exactly.</summary>
    static void SameDoc(Case c, LayerDocument actual, JsonElement expected)
    {
        var e = Doc(c, expected)!;
        Assert.That(actual.Canvas, Is.EqualTo(e.Canvas), "canvas");
        Assert.That(actual.Layers, Has.Count.EqualTo(e.Layers.Count), "layer count");
        for (int i = 0; i < e.Layers.Count; i++)
        {
            LayerItem a = actual.Layers[i], x = e.Layers[i];
            Assert.That(a with { Image = x.Image, Mask = x.Mask }, Is.EqualTo(x), $"item {i}");
            Same(a.Image, ReadBack(x.Image), 0, $"item {i} picture");
            Assert.That(a.Mask is null, Is.EqualTo(x.Mask is null), $"item {i} mask");
            if (x.Mask is not null) Same(a.Mask!, ReadBack(x.Mask), 0, $"item {i} mask");
        }
    }

    static Tensor ReadBack(HostTensor t) => new(t.Shape, t.Data);

    static void Expect(Case c, Action run)
    {
        if (!c.Json.TryGetProperty("error", out var error))
        {
            run();
            return;
        }
        var ex = Assert.Catch(() => run(), $"ComfyUI raised {error.GetString()}: {c.Json.GetProperty("message").GetString()}");
        Assert.That(ex, Is.InstanceOf<PyException>(), ex!.Message);
        Assert.That(((PyException)ex).PyType, Is.EqualTo(error.GetString()));
        Assert.That(((PyException)ex).PyMessage, Is.EqualTo(c.Json.GetProperty("message").GetString()));
    }

    static long L(Case c, string name, long fallback) => Arg(c, name) is { } v ? Convert.ToInt64(v) : fallback;
    static double D(Case c, string name, double fallback) => Arg(c, name) is { } v ? Convert.ToDouble(v) : fallback;

    [TestCaseSource(nameof(AddCases))]
    public void AddLayer_matches_ComfyUI(Case c)
    {
        var args = c.Json.GetProperty("args");
        Expect(c, () =>
        {
            var doc = new AddLayer().Execute(Img(T(c, args.GetProperty("image"))), Doc(c, args.GetProperty("layers")),
                Present(args, "mask", out var m) ? Mask(T(c, m)) : null, Arg(c, "name") as string ?? "", L(c, "x", 0), L(c, "y", 0), D(c, "opacity", 1.0),
                Arg(c, "blend_mode") as string ?? "normal", D(c, "rotation", 0.0), L(c, "width", 0), L(c, "height", 0), L(c, "z_index", 0),
                Arg(c, "flip_h") as bool? ?? false, Arg(c, "flip_v") as bool? ?? false);
            SameDoc(c, doc, c.Json.GetProperty("out")[0]);
        });
    }

    [TestCaseSource(nameof(BoxCases))]
    public void LayersFromBoundingBoxes_matches_ComfyUI(Case c)
    {
        var args = c.Json.GetProperty("args");
        Expect(c, () =>
        {
            var doc = new LayersFromBoundingBoxes().Execute(Img(T(c, args.GetProperty("image"))), Arg(c, "bboxes")!,
                Present(args, "mask", out var m) ? Mask(T(c, m)) : null, Present(args, "layers", out var l) ? Doc(c, l) : null,
                Arg(c, "crop_to_content") as bool? ?? true, L(c, "canvas_width", 0), L(c, "canvas_height", 0));
            SameDoc(c, doc, c.Json.GetProperty("out")[0]);
        });
    }

    [TestCaseSource(nameof(CompositeCases))]
    public void ImageCompositor_matches_ComfyUI(Case c)
    {
        var args = c.Json.GetProperty("args");
        Expect(c, () =>
        {
            var (image, mask, ui) = new ImageCompositor().ExecuteAsync(Doc(c, args.GetProperty("layers"))!, Arg(c, "compositor")).GetAwaiter().GetResult();
            var o = c.Json.GetProperty("out");
            // numpy's float32 arithmetic in the same order; powf in the sRGB curves can differ in the last bit.
            Same(image, T(c, o[0]), 2e-6, "image");
            Same(mask, T(c, o[1]), 2e-6, "mask");
            var expectedUi = c.Json.GetProperty("ui");
            foreach (var key in new[] { "compositor_inputs", "compositor_bboxes", "compositor_canvas", "compositor_state_stale" })
            {
                if (!expectedUi.TryGetProperty(key, out var e)) { Assert.That(ui.ContainsKey(key), Is.False, key); continue; }
                Assert.That(JsonLiteral.ToJson(ui[key]), Is.EqualTo(JsonLiteral.ToJson(JsonLiteral.FromJson(e))), key);
            }
        });
    }

    /// <summary>The nodes in a prompt as the server receives it: object-valued widgets as JSON, previews stored, the editor's data in the node's UI.</summary>
    [Test]
    public async Task The_nodes_run_in_a_prompt_and_show_their_previews()
    {
        var registry = new SpawnGen.Engine.Nodes.NodeRegistry().AddAssembly(typeof(ImageCompositor).Assembly);
        var store = new MemoryResultStore();
        var executor = new SpawnGen.Engine.Execution.PromptExecutor(registry, new Services(store));
        var prompt = SpawnGen.Engine.Graph.PromptJson.Parse("""
        {
          "1": { "class_type": "EmptyImage", "inputs": { "width": 40, "height": 30, "batch_size": 1, "color": 3368601 } },
          "2": { "class_type": "ImageCropV2", "inputs": { "image": ["1", 0], "crop_region": {"x": 5, "y": 6, "width": 20, "height": 10} } },
          "3": { "class_type": "AddLayer", "inputs": { "image": ["1", 0] } },
          "4": { "class_type": "AddLayer", "inputs": { "image": ["2", 0], "layers": ["3", 0], "x": 4, "y": 3, "opacity": 0.5, "blend_mode": "screen", "rotation": 20 } },
          "5": { "class_type": "ImageCompositor", "inputs": { "layers": ["4", 0], "compositor": {} } },
          "6": { "class_type": "CreateBoundingBoxes", "inputs": { "background": ["2", 0], "width": 64, "height": 64,
                 "editor_state": {"__value__": [{"x": 2, "y": 2, "width": 10, "height": 5}]}, "last_incoming": {"__value__": []} } },
          "7": { "class_type": "PreviewImage", "inputs": { "images": ["2", 0] } }
        }
        """);
        var result = await executor.ExecuteAsync(prompt);

        var composite = (ImageBatch)result.Outputs["5"].Values[0]!;
        Assert.That((composite.Width, composite.Height, composite.Channels), Is.EqualTo((40, 30, 3)), "an opaque composite is RGB on the base layer's extent");
        var ui = result.Outputs["5"].Ui!;
        Assert.That(ui["images"], Is.InstanceOf<List<object>>().And.Count.EqualTo(1));
        Assert.That(ui["compositor_layers"], Is.InstanceOf<List<object>>().And.Count.EqualTo(2));
        Assert.That(((ArrayValue)ui["compositor_inputs"]!).Count, Is.EqualTo(2));
        Assert.That(JsonLiteral.ToJson(ui["compositor_canvas"]), Is.EqualTo("""[{"w":40,"h":30}]"""));
        Assert.That(ui.ContainsKey("compositor_state_stale"), Is.False, "an empty saved state is not a stale one");

        var boxes = result.Outputs["6"];
        Assert.That(((ImageBatch)boxes.Values[0]!).Width, Is.EqualTo(20), "the preview takes the background's size");
        Assert.That(boxes.Ui!["background_images"], Is.InstanceOf<List<object>>().And.Count.EqualTo(1));
        Assert.That(JsonLiteral.ToJson(boxes.Values[1]), Is.EqualTo("""[[{"x":2,"y":2,"width":10,"height":5}]]"""));
        Assert.That(((ImageBatch)result.Outputs["7"].Values[0]!).Shape, Is.EqualTo(new[] { 1, 10, 20, 3 }));
    }

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }

    [Test]
    public void The_fixture_covers_every_mode_and_the_saved_compositions()
    {
        var composites = Load().Where(c => c.Node == "ImageCompositor").ToList();
        var modes = composites.SelectMany(c => c.Json.GetProperty("args").GetProperty("layers") is { ValueKind: JsonValueKind.Object } l
            ? l.GetProperty("layers").EnumerateArray().Select(i => i.TryGetProperty("blend_mode", out var b) ? b.GetString() : "normal") : []).ToHashSet();
        Assert.That(modes, Is.SupersetOf(CompositorBlend.ModeNames));
        Assert.That(composites.Count(c => c.Json.TryGetProperty("ui", out var u) && !u.TryGetProperty("compositor_state_stale", out _)
            && c.Json.GetProperty("args").GetProperty("compositor").ValueKind is JsonValueKind.Object or JsonValueKind.String), Is.GreaterThanOrEqualTo(2), "replayed");
        Assert.That(composites.Count(c => c.Json.TryGetProperty("ui", out var u) && u.TryGetProperty("compositor_state_stale", out _)), Is.GreaterThanOrEqualTo(4), "stale");
    }
}
