using System.IO.Compression;
using System.Text.Json;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Pad Image, Convert SDR to HDR and Transfer Color against ComfyUI's own code: Fixtures/comfy/comfy_pad_color_nodes.json.gz holds inputs and the
/// outputs of each node's <c>execute</c> as torch returned them (float32), made by tools/parity/gen_pad_color_fixtures.py on torch 2.14 (CPU),
/// kornia 0.8.3 and numpy 2.4.
/// </summary>
public class PadColorParityTests
{
    public sealed record Tensor(int[] Shape, float[] Data);

    public sealed record Case(string Node, JsonElement Args, Dictionary<string, Tensor> Inputs, Tensor Out)
    {
        public override string ToString() => Node + " " + Args.GetRawText();
    }

    static List<Case>? loaded;

    static List<Case> Load()
    {
        if (loaded is not null) return loaded;
        var path = Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "comfy", "comfy_pad_color_nodes.json.gz");
        using var gz = new GZipStream(File.OpenRead(path), CompressionMode.Decompress);
        using var doc = JsonDocument.Parse(gz);
        var tensors = doc.RootElement.GetProperty("tensors").EnumerateArray().Select(ReadTensor).ToArray();
        return loaded = doc.RootElement.GetProperty("cases").EnumerateArray().Select(c => new Case(
            c.GetProperty("node").GetString()!,
            c.GetProperty("args").Clone(),
            c.GetProperty("inputs").EnumerateObject().ToDictionary(p => p.Name, p => tensors[p.Value.GetInt32()]),
            ReadTensor(c.GetProperty("out")))).ToList();
    }

    static Tensor ReadTensor(JsonElement t)
    {
        var shape = t.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        var bytes = Convert.FromBase64String(t.GetProperty("b64").GetString()!);
        var data = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
        return new Tensor(shape, data);
    }

    static IEnumerable<TestCaseData> Cases(string node) => Load().Where(c => c.Node == node).Select((c, i) => new TestCaseData(c).SetName($"{node}_{i:D3}"));
    static IEnumerable<TestCaseData> PadCases() => Cases("PadImageNode");
    static IEnumerable<TestCaseData> HdrCases() => Cases("ImageSDRToHDR");
    // A flat target frame under a Lab method is compared by Flat_frames_take_the_reference_mean instead (see there).
    static IEnumerable<TestCaseData> TransferCases() => Cases("ColorTransfer").Where(t => !IsFlatLab((Case)t.Arguments[0]!));

    static bool IsFlatLab(Case c) => c.Node == "ColorTransfer" && c.Args.GetProperty("method").GetString() != "histogram" && c.Inputs["image_target"].Data.Distinct().Count() == 1;

    static ImageBatch Img(Tensor t) => new(t.Shape[0], t.Shape[1], t.Shape[2], t.Shape[3], (float[])t.Data.Clone());

    /// <summary>A dynamic combo's value as ComfyUI passes it ({"padding_type": "pad dimensions", "left": 2, ...}): the key under the combo's name.</summary>
    static DynamicChoice Choice(JsonElement e, string name)
    {
        var values = new Dictionary<string, object?>();
        foreach (var p in e.EnumerateObject())
        {
            if (p.Name == name) continue;
            values[p.Name] = p.Value.ValueKind switch
            {
                JsonValueKind.String => p.Value.GetString(),
                JsonValueKind.Number when p.Value.GetRawText().Contains('.') => p.Value.GetDouble(),
                JsonValueKind.Number => p.Value.GetInt64(),
                _ => throw new InvalidDataException(p.Value.GetRawText()),
            };
        }
        return new DynamicChoice(e.GetProperty(name).GetString()!, values);
    }

    static void Same(ImageBatch actual, Tensor expected, double tolerance)
    {
        Assert.That(actual.Shape, Is.EqualTo(expected.Shape), "shape");
        double worst = 0;
        int at = -1;
        for (int i = 0; i < expected.Data.Length; i++)
        {
            double d = Math.Abs(actual.Data[i] - expected.Data[i]);
            if (!(d <= worst)) { worst = d; at = i; }
        }
        Assert.That(worst, Is.LessThanOrEqualTo(tolerance), $"worst at {at}: {(at >= 0 ? actual.Data[at] : 0)} vs {(at >= 0 ? expected.Data[at] : 0)}");
    }

    [TestCaseSource(nameof(PadCases))]
    public void Pad_Image_matches_ComfyUI(Case c)
    {
        var r = new PadImageNode().Execute(Img(c.Inputs["image"]), Choice(c.Args.GetProperty("padding_type"), "padding_type"),
            Choice(c.Args.GetProperty("padding_style"), "padding_style"));
        Same(r, c.Out, 0); // copies, the fill colour and a float32 lerp: identical
    }

    [TestCaseSource(nameof(HdrCases))]
    public void SDR_to_HDR_matches_ComfyUI(Case c)
    {
        var r = new ImageSDRToHDR().Execute(Img(c.Inputs["image"]), Choice(c.Args.GetProperty("algorithm"), "algorithm"));
        Same(r, c.Out, 2.4e-7); // float32 step by step: at most an ulp or two (torch's SIMD pow / exp / log)
    }

    [TestCaseSource(nameof(TransferCases))]
    public void Transfer_Color_matches_ComfyUI(Case c)
    {
        var method = c.Args.GetProperty("method").GetString()!;
        var r = new ColorTransfer().Execute(Img(c.Inputs["image_target"]), Img(c.Inputs["image_ref"]), method,
            Choice(c.Args.GetProperty("source_stats"), "source_stats"), c.Args.GetProperty("strength").GetDouble());
        // histogram: bin counting and a lookup, identical (a lerp may round an ulp apart where torch's SIMD kernel fuses it); Lab: float32 pow / matmul
        // rounding and LAPACK's float eigh differ in the last bits.
        double strength = c.Args.GetProperty("strength").GetDouble();
        Same(r, c.Out, method != "histogram" ? 1e-5 : strength == 1 ? 0 : 1.2e-7);
    }

    [Test]
    public void The_fixture_covers_every_option()
    {
        var cases = Load();
        Assert.That(cases.Where(c => c.Node == "PadImageNode").Select(c => c.Args.GetProperty("padding_type").GetProperty("padding_type").GetString()).Distinct(),
            Is.EquivalentTo(PadImageNode.PaddingTypes().Select(o => o.Key)));
        Assert.That(cases.Where(c => c.Node == "PadImageNode").Select(c => c.Args.GetProperty("padding_style").GetProperty("padding_style").GetString()).Distinct(),
            Is.EquivalentTo(PadImageNode.PaddingStyles().Select(o => o.Key)));
        Assert.That(cases.Where(c => c.Node == "ColorTransfer").Select(c => c.Args.GetProperty("method").GetString()).Distinct(), Is.EquivalentTo(ColorTransfer.MethodOptions()));
        Assert.That(cases.Where(c => c.Node == "ColorTransfer").Select(c => c.Args.GetProperty("source_stats").GetProperty("source_stats").GetString()).Distinct(),
            Is.EquivalentTo(ColorTransfer.SourceStats().Select(o => o.Key)));
    }

    // ---- by hand ----

    [Test]
    public void Pad_sizes_follow_Pythons_rounding_and_modulo()
    {
        var img = new ImageBatch(1, 5, 7, 3);
        static DynamicChoice T(string key, params (string, object)[] v) => new(key, v.ToDictionary(x => x.Item1, x => (object?)x.Item2));
        var mirror = T("mirror image");
        // round(7 * 1.5) = round(10.5) = 10 (half to even): 3 more, 1 left / 2 right; round(7.5) = 8: 3 more, 1 top / 2 bottom.
        var r = new PadImageNode().Execute(img, T("pad by multiplier", ("multiplier", 1.5)), mirror);
        Assert.That((r.Width, r.Height), Is.EqualTo((10, 8)));
        // -7 % 4 = 1 (Python), -5 % 4 = 3: 0 left / 1 right, 1 top / 2 bottom.
        r = new PadImageNode().Execute(img, T("pad to multiple", ("multiple", 4L)), mirror);
        Assert.That((r.Width, r.Height), Is.EqualTo((8, 8)));
        Assert.That(new PadImageNode().Execute(img, T("pad dimensions"), mirror), Is.SameAs(img), "no padding: the input itself");
    }

    [Test]
    public void Pad_colour_is_parsed_like_ComfyUI()
    {
        var img = new ImageBatch(1, 1, 1, 4, [0.5f, 0.5f, 0.5f, 0.5f]);
        float[] Corner(string color) => new PadImageNode().Execute(img, new DynamicChoice("pad dimensions", new Dictionary<string, object?> { ["left"] = 1L }),
            new DynamicChoice("color", new Dictionary<string, object?> { ["color"] = color, ["feathering"] = 0L })).Data[..4];
        Assert.That(Corner("#ff8000"), Is.EqualTo(new[] { 1f, (float)(128 / 255.0), 0f, 1f }), "a fourth channel is filled with 1");
        Assert.That(Corner("#336699cc"), Is.EqualTo(new[] { (float)(0x33 / 255.0), (float)(0x66 / 255.0), (float)(0x99 / 255.0), 1f }), "alpha ignored");
        Assert.That(Corner("#zz0000"), Is.EqualTo(new[] { 1f, 1f, 1f, 1f }), "not hex: white");
    }

    [Test]
    public void ColorTransfer_with_no_strength_returns_the_target()
    {
        var a = new ImageBatch(1, 2, 2, 3, Enumerable.Range(0, 12).Select(i => i / 4f - 1f).ToArray());
        Assert.That(new ColorTransfer().Execute(a, a, "mkl_lab", new DynamicChoice("uniform", new Dictionary<string, object?>()), 0), Is.SameAs(a));
    }

    /// <summary>
    /// A frame of one colour has no spread: the eps clamps make the reinhard ratio (and the mkl matrix) about 1e6, applied to (Lab - mean). That is 0, so the
    /// frame takes the reference mean, which is what SpawnGen computes. ComfyUI's float32 torch.mean of the equal values is itself an ulp off (its
    /// SIMD lanes are added one after another: 16 x 26.982914 sums to 431.72659), and 1e6 x that ulp moves the colour by a visible amount; how far
    /// depends on torch's reduction order (SIMD width, thread count), so it is not something to reproduce.
    /// </summary>
    [Test]
    public void Flat_frames_take_the_reference_mean()
    {
        var flatCases = Load().Where(IsFlatLab).ToList();
        Assert.That(flatCases, Has.Count.EqualTo(2));
        ImageBatch? first = null;
        foreach (var c in flatCases)
        {
            var r = new ColorTransfer().Execute(Img(c.Inputs["image_target"]), Img(c.Inputs["image_ref"]), c.Args.GetProperty("method").GetString()!,
                Choice(c.Args.GetProperty("source_stats"), "source_stats"), 1.0);
            for (int i = 3; i < r.Data.Length; i++) Assert.That(r.Data[i], Is.EqualTo(r.Data[i % 3]), "one colour");
            // ComfyUI's output is one colour too (the shift is the same for every pixel).
            for (int i = 3; i < c.Out.Data.Length; i++) Assert.That(c.Out.Data[i], Is.EqualTo(c.Out.Data[i % 3]));
            if (first is null) first = r;
            else Assert.That(r.Data[..3], Is.EqualTo(first.Data[..3]).Within(1e-5), "reinhard and mkl agree: both are the reference frame's mean");
        }
    }

    [Test]
    public void Mkl_matrix_maps_one_covariance_onto_the_other()
    {
        // T Cs T^T = Cr for the Monge-Kantorovich map, whatever the eigenvector signs.
        float[] cs = [4, 1, 0.5f, 1, 3, 0.2f, 0.5f, 0.2f, 2], cr = [1, -0.3f, 0, -0.3f, 2, 0.4f, 0, 0.4f, 5];
        var t = (float[])typeof(ColorTransfer).GetMethod("MklMatrix", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!.Invoke(null, [cs, cr])!;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
            {
                double s = 0;
                for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) s += t[r * 3 + i] * cs[i * 3 + j] * t[c * 3 + j];
                Assert.That(s, Is.EqualTo(cr[r * 3 + c]).Within(1e-5));
            }
        // and T is symmetric (the map is the gradient of a convex function).
        Assert.That(t[1], Is.EqualTo(t[3]).Within(1e-6));
    }

    [Test]
    public void The_schemas_list_ComfyUIs_inputs_in_order()
    {
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly);
        Assert.That(registry.GetSchema("PadImageNode")!.Inputs.Select(i => i.Name), Is.EqualTo(new[]
        {
            "image", "padding_type", "padding_type.left", "padding_type.top", "padding_type.right", "padding_type.bottom", "padding_type.multiplier",
            "padding_type.multiple", "padding_style", "padding_style.color", "padding_style.feathering",
        }));
        var color = registry.GetSchema("PadImageNode")!.FindInput("padding_style.color")!;
        Assert.That((color.Type.Name, color.Widget!.Kind, color.Widget.Default), Is.EqualTo(("COLOR", WidgetKind.Color, (object)"#000000")));
        Assert.That(registry.GetSchema("ColorTransfer")!.Inputs.Select(i => i.Name),
            Is.EqualTo(new[] { "image_target", "image_ref", "method", "source_stats", "source_stats.target_index", "strength" }));
        Assert.That(registry.GetSchema("ImageSDRToHDR")!.Inputs.Select(i => i.Name), Is.EqualTo(new[] { "image", "algorithm", "algorithm.target_luminance" }));
    }

    [Test]
    public async Task Pad_Image_runs_from_a_prompt_with_its_colour()
    {
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly);
        var g = new PromptGraph();
        g.Add("1", "EmptyImage", ("width", 3L), ("height", 2L), ("batch_size", 1L), ("color", 0L));
        g.Add("2", "PadImageNode", ("image", PromptInput.Link("1")), ("padding_type", "pad to multiple"), ("padding_type.multiple", 4L),
            ("padding_style", "color"), ("padding_style.color", "#ff0000"), ("padding_style.feathering", 0L));
        g.Add("3", "GetImageSize", ("image", PromptInput.Link("2")));
        g.Add("4", "PreviewAny", ("source", PromptInput.Link("3")));
        var r = await new PromptExecutor(registry).ExecuteAsync(g);
        // 3 wide padded to 4: the COLOR widget value passes validation and reaches the node.
        Assert.That(((IEnumerable<string>)r.Outputs["4"].Ui!["text"]!).Single(), Is.EqualTo("4"));
    }

    // ---- pictures to look at ----

    static ImageBatch Scene(int w, int h, float warm)
    {
        var img = new ImageBatch(1, h, w, 3);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                float u = x / (w - 1f), v = y / (h - 1f);
                bool disc = (u - 0.6f) * (u - 0.6f) + (v - 0.4f) * (v - 0.4f) < 0.04f;
                int p = (y * w + x) * 3;
                img.Data[p] = disc ? 0.95f : 0.2f + 0.6f * u * warm;
                img.Data[p + 1] = disc ? 0.85f * warm : 0.3f + 0.5f * v;
                img.Data[p + 2] = disc ? 0.2f : 0.7f - 0.4f * u * warm;
            }
        return img;
    }

    [Test]
    public void Writes_pictures_of_real_results()
    {
        var dir = Path.Combine(TestContext.CurrentContext.WorkDirectory, "pad-color-nodes");
        Directory.CreateDirectory(dir);
        void Write(string name, ImageBatch img) { var path = Path.Combine(dir, name); File.WriteAllBytes(path, PngWriter.Encode(img)); TestContext.Out.WriteLine(path); }

        var scene = Scene(96, 64, 1f);
        Write("scene.png", scene);
        Write("pad_mirror.png", new PadImageNode().Execute(scene, new DynamicChoice("pad by multiplier", new Dictionary<string, object?> { ["multiplier"] = 1.6 }),
            new DynamicChoice("mirror image", new Dictionary<string, object?>())));
        Write("pad_color_feathered.png", new PadImageNode().Execute(scene,
            new DynamicChoice("pad dimensions", new Dictionary<string, object?> { ["left"] = 24L, ["top"] = 8L, ["right"] = 0L, ["bottom"] = 16L }),
            new DynamicChoice("color", new Dictionary<string, object?> { ["color"] = "#3366cc", ["feathering"] = 12L })));
        Write("sdr_to_hdr_hlg.png", new ImageSDRToHDR().Execute(scene, new DynamicChoice("lottes", new Dictionary<string, object?> { ["target_luminance"] = 650.0 })));

        var reference = Scene(64, 48, -0.6f);
        Write("transfer_ref.png", reference);
        foreach (var method in ColorTransfer.MethodOptions())
        {
            var r = new ColorTransfer().Execute(scene, reference, method, new DynamicChoice("per_frame", new Dictionary<string, object?>()));
            Assert.That(r.Data, Is.All.InRange(0f, 1f));
            Write($"transfer_{method}.png", r);
        }
    }
}
