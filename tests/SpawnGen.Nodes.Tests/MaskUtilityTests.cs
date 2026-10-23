using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Preview Mask, Resolution Selector, Image Histogram, and the ComfyUI details of the mask nodes (nodes_mask.py): ids, names, categories,
/// widget order and edge cases, against values worked out from ComfyUI's Python.
/// </summary>
public class MaskUtilityTests
{
    static MaskBatch Mask(int w, int h, params float[] v) => new(1, h, w, v);

    static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(MaskPreview).Assembly);

    // ---- registration ----

    [TestCase("SolidMask", "Create Solid Mask", "image/mask")]
    [TestCase("InvertMask", "Invert Mask", "image/mask")]
    [TestCase("CropMask", "Crop Mask", "image/mask")]
    [TestCase("MaskComposite", "Combine Masks", "image/mask")]
    [TestCase("FeatherMask", "Feather Mask", "image/mask")]
    [TestCase("GrowMask", "Grow Mask", "image/mask")]
    [TestCase("ThresholdMask", "Threshold Mask", "image/mask")]
    [TestCase("ImageToMask", "Convert Image to Mask", "image/mask")]
    [TestCase("MaskToImage", "Convert Mask to Image", "image/mask")]
    [TestCase("ImageColorToMask", "Convert Image Color to Mask", "image/mask")]
    [TestCase("ImageCompositeMasked", "Image Composite Masked", "image/compositing")]
    [TestCase("MaskPreview", "Preview Mask", "image/mask")]
    [TestCase("ResolutionSelector", "Resolution Selector", "utilities")]
    [TestCase("ImageHistogram", "Image Histogram", "utilities")]
    public void Nodes_have_ComfyUIs_display_name_and_category(string id, string display, string category)
    {
        var schema = Registry().GetSchema(id);
        Assert.That(schema, Is.Not.Null, id);
        Assert.That((schema!.DisplayName, schema.Category), Is.EqualTo((display, category)));
    }

    [Test]
    public void Inputs_widgets_and_outputs_are_in_ComfyUIs_order()
    {
        var registry = Registry();
        string[] Inputs(string id) => registry.GetSchema(id)!.Inputs.Select(i => i.Name).ToArray();
        Assert.That(Inputs("SolidMask"), Is.EqualTo(new[] { "value", "width", "height" }));
        Assert.That(Inputs("CropMask"), Is.EqualTo(new[] { "mask", "x", "y", "width", "height" }));
        Assert.That(Inputs("MaskComposite"), Is.EqualTo(new[] { "destination", "source", "x", "y", "operation" }));
        Assert.That(Inputs("FeatherMask"), Is.EqualTo(new[] { "mask", "left", "top", "right", "bottom" }));
        Assert.That(Inputs("GrowMask"), Is.EqualTo(new[] { "mask", "expand", "tapered_corners" }));
        Assert.That(Inputs("ThresholdMask"), Is.EqualTo(new[] { "mask", "value" }));
        Assert.That(Inputs("ImageCompositeMasked"), Is.EqualTo(new[] { "destination", "source", "x", "y", "resize_source", "mask" }));
        Assert.That(Inputs("ImageColorToMask"), Is.EqualTo(new[] { "image", "color" }));
        Assert.That(registry.GetSchema("ImageToMask")!.FindInput("channel")!.Widget!.ResolveOptions(), Is.EqualTo(new[] { "red", "green", "blue", "alpha" }));
        Assert.That(registry.GetSchema("GrowMask")!.FindInput("expand")!.Widget!.Min, Is.EqualTo(-16384));

        var preview = registry.GetSchema("MaskPreview")!;
        Assert.That(Inputs("MaskPreview"), Is.EqualTo(new[] { "mask" }));
        Assert.That(preview.IsOutputNode, Is.True);
        Assert.That(preview.Outputs.Select(o => (o.Name, o.Type.Name)), Is.EqualTo(new[] { ("mask", "MASK") }));

        // The "preview" widget of ComfyUI's node is frontend-only (serialize: false), so widgets_values holds just these three.
        var resolution = registry.GetSchema("ResolutionSelector")!;
        Assert.That(Inputs("ResolutionSelector"), Is.EqualTo(new[] { "aspect_ratio", "megapixels", "multiple" }));
        Assert.That(resolution.FindInput("aspect_ratio")!.Widget!.ResolveOptions(), Is.EqualTo(new[]
        {
            "1:1 (Square)", "2:3 (Portrait Photo)", "3:2 (Photo)", "3:4 (Portrait Standard)", "4:3 (Standard)",
            "9:16 (Portrait Widescreen)", "16:9 (Widescreen)", "21:9 (Ultrawide)",
        }));
        var multiple = resolution.FindInput("multiple")!.Widget!;
        Assert.That((multiple.Default, multiple.Min, multiple.Max, multiple.Step), Is.EqualTo(((object)8L, (double?)8, (double?)128, (double?)4)));
        var mp = resolution.FindInput("megapixels")!.Widget!;
        Assert.That((mp.Default, mp.Min, mp.Max, mp.Step), Is.EqualTo(((object)1.0, (double?)0.1, (double?)16, (double?)0.1)));
        Assert.That(resolution.Outputs.Select(o => (o.Name, o.Type.Name)), Is.EqualTo(new[] { ("width", "INT"), ("height", "INT") }));

        Assert.That(registry.GetSchema("ImageHistogram")!.Outputs.Select(o => (o.Name, o.Type.Name)), Is.EqualTo(new[]
        {
            ("rgb", "HISTOGRAM"), ("luminance", "HISTOGRAM"), ("red", "HISTOGRAM"), ("green", "HISTOGRAM"), ("blue", "HISTOGRAM"),
        }));
    }

    // ---- mask node edge cases (nodes_mask.py) ----

    [Test]
    public void CropMask_slices_like_Python_and_a_corner_past_the_edge_gives_an_empty_mask()
    {
        var ramp = Mask(4, 2, 0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f);
        // mask[:, 0:5, 3:8] -> the last column.
        var last = new CropMask().Execute(ramp, 3, 0, 5, 5);
        Assert.That((last.Width, last.Height), Is.EqualTo((1, 2)));
        Assert.That(last.Data, Is.EqualTo(new[] { 0.3f, 0.7f }));
        // mask[:, 0:1, 4:5] and mask[:, 2:3, 0:1] are empty (ComfyUI does not clamp the corner like Image Crop does).
        var right = new CropMask().Execute(ramp, 4, 0, 1, 1);
        Assert.That((right.Batch, right.Width, right.Height), Is.EqualTo((1, 0, 1)));
        var below = new CropMask().Execute(ramp, 0, 2, 1, 1);
        Assert.That((below.Width, below.Height), Is.EqualTo((1, 0)));
    }

    [Test]
    public void MaskComposite_clamps_the_whole_result_not_only_the_pasted_area()
    {
        // torch.clamp(output, 0, 1) runs over the destination outside the source too.
        var r = new MaskComposite().Execute(Mask(3, 1, 1.5f, -0.5f, 0.5f), Mask(1, 1, 0.25f), 2, 0, "add");
        Assert.That(r.Data, Is.EqualTo(new[] { 1f, 0f, 0.75f }).Within(1e-6));
        // and/or/xor round half to even: 0.5 counts as 0.
        var and = new MaskComposite().Execute(Mask(2, 1, 0.5f, 0.51f), Mask(2, 1, 1f, 1f), 0, 0, "and");
        Assert.That(and.Data, Is.EqualTo(new[] { 0f, 1f }));
    }

    [Test]
    public void GrowMask_handles_the_border_like_scipys_reflect_mode()
    {
        // grey_erosion with mode="reflect": the pixel outside the edge mirrors the edge pixel, so an edge row of ones survives one erosion step.
        var m = new MaskBatch(1, 3, 3, [1, 1, 1, 1, 1, 1, 0, 0, 0]);
        var eroded = new GrowMask().Execute(m, -1, true);
        Assert.That(eroded.Data, Is.EqualTo(new float[] { 1, 1, 1, 0, 0, 0, 0, 0, 0 }));
        var grown = new GrowMask().Execute(new MaskBatch(1, 1, 4, [0, 0.25f, 0, 0.75f]), 1, true);
        Assert.That(grown.Data, Is.EqualTo(new[] { 0.25f, 0.25f, 0.75f, 0.75f }), "grey dilation takes the maximum, values are kept");
    }

    [Test]
    public void ImageCompositeMasked_gives_an_RGB_source_an_opaque_alpha_and_does_not_clamp_the_mask()
    {
        // node_helpers.image_alpha_fix pads the RGB source with alpha 1 for an RGBA destination.
        var dest = new ImageBatch(1, 1, 1, 4, [0.2f, 0.2f, 0.2f, 0f]);
        var src = new ImageBatch(1, 1, 1, 3, [0.8f, 0.6f, 0.4f]);
        var r = new ImageCompositeMasked().Execute(dest, src);
        Assert.That(r.Data, Is.EqualTo(new[] { 0.8f, 0.6f, 0.4f, 1f }).Within(1e-6));

        // mask * source + (1 - mask) * destination with the mask as given: 2 * 0.8 - 1 * 0.2 = 1.4.
        var rgb = new ImageBatch(1, 1, 1, 3, [0.2f, 0.2f, 0.2f]);
        var over = new ImageCompositeMasked().Execute(rgb, src, 0, 0, false, Mask(1, 1, 2f));
        Assert.That(over.Data[0], Is.EqualTo(1.4f).Within(1e-6));

        // An RGBA source over an RGB destination: the alpha is dropped.
        var rgba = new ImageBatch(1, 1, 1, 4, [0.9f, 0.9f, 0.9f, 0.1f]);
        Assert.That(new ImageCompositeMasked().Execute(rgb, rgba).Data, Is.EqualTo(new[] { 0.9f, 0.9f, 0.9f }));
    }

    [Test]
    public void ImageColorToMask_clamps_before_rounding_to_bytes()
    {
        // (clamp(image, 0, 1) * 255).round(): 1.2 counts as 255, -0.1 as 0; 0.5 * 255 = 127.5 rounds half to even (128).
        var img = new ImageBatch(1, 1, 3, 3, [1.2f, -0.1f, 0f, 1f, 0f, 0f, 0.5f, 0.5f, 0.5f]);
        Assert.That(new ImageColorToMask().Execute(img, 0xFF0000).Data, Is.EqualTo(new[] { 1f, 1f, 0f }));
        Assert.That(new ImageColorToMask().Execute(img, 0x808080).Data, Is.EqualTo(new[] { 0f, 0f, 1f }));
    }

    // ---- Preview Mask ----

    [Test]
    public async Task MaskPreview_stores_a_grey_picture_of_the_mask_and_passes_the_mask_on()
    {
        // A horizontal ramp, feathered at the top: dark left, bright right, fading in from the top edge.
        const int w = 64, h = 32;
        var g = new PromptGraph();
        g.Add("1", "SolidMask", ("value", 1.0), ("width", w), ("height", h));
        g.Add("2", "FeatherMask", ("mask", PromptInput.Link("1")), ("left", 64), ("top", 16), ("right", 0), ("bottom", 0));
        g.Add("3", "MaskPreview", ("mask", PromptInput.Link("2")));
        var store = new MemoryResultStore();
        var saved = new List<byte[]>();
        store.Saved += (_, png) => saved.Add(png);
        var result = await new PromptExecutor(Registry(), new Services(store)).ExecuteAsync(g);

        var outputs = result.Outputs["3"];
        var mask = (MaskBatch)outputs.Values.Single()!;
        Assert.That((mask.Width, mask.Height), Is.EqualTo((w, h)));
        Assert.That(outputs.Ui!.ContainsKey("images"), Is.True);
        Assert.That(saved, Has.Count.EqualTo(1));

        var picture = PngReader.Decode(saved[0]);
        Assert.That((picture.Width, picture.Height, picture.Channels), Is.EqualTo((w, h, 3)));
        // Bottom-right pixel: feather (64/64) * (row 31 is past the top ramp) = 1 -> white; top-left: (1/64) * (1/16).
        int br = ((h - 1) * w + w - 1) * 3;
        Assert.That(picture.Data[br], Is.EqualTo(1f).Within(1 / 255f));
        Assert.That(picture.Data[0], Is.EqualTo(1f / 64 / 16).Within(1 / 255f));
        Assert.That(picture.Data[br + 1], Is.EqualTo(picture.Data[br]), "grey: the three channels are the mask");

        var file = Path.Combine(TestContext.CurrentContext.WorkDirectory, "mask_preview.png");
        File.WriteAllBytes(file, saved[0]);
        TestContext.Out.WriteLine("Preview Mask picture: " + file);
    }

    // ---- Resolution Selector ----

    [TestCase("1:1 (Square)", 1.0, 8, 1024, 1024)]
    [TestCase("16:9 (Widescreen)", 1.0, 8, 1368, 768)] // 170.67 -> 171, 96 -> 96
    [TestCase("21:9 (Ultrawide)", 2.0, 64, 2240, 960)] // 34.56 -> 35, 14.81 -> 15
    [TestCase("2:3 (Portrait Photo)", 0.5, 16, 592, 880)] // 36.95 -> 37, 55.43 -> 55
    [TestCase("1:1 (Square)", 0.2384185791015625, 8, 496, 496)] // sqrt(250000) / 8 = 62.5 -> 62 (Python rounds half to even)
    [TestCase("1:1 (Square)", 1.0, 128, 1024, 1024)]
    public void ResolutionSelector_matches_ComfyUIs_rounding(string ratio, double megapixels, int multiple, int width, int height)
    {
        Assert.That(new ResolutionSelector().Execute(ratio, megapixels, multiple), Is.EqualTo((width, height)));
    }

    [Test]
    public void ResolutionSelector_rejects_an_unknown_ratio()
    {
        Assert.That(() => new ResolutionSelector().Execute("5:4", 1.0, 8), Throws.ArgumentException);
    }

    // ---- Image Histogram ----

    [Test]
    public void ImageHistogram_counts_truncated_levels_and_luminance_in_single_precision()
    {
        // Six pixels; values worked out with float32 arithmetic as numpy does (np.clip(img * 255, 0, 255).astype(np.uint8) truncates).
        float g53 = 53f / 255f;
        var img = new ImageBatch(1, 2, 3, 3,
        [
            1f, 0.5f, 0f,   0.2f, 0.2f, 0.2f,   1.5f, -0.1f, 0.999f,
            0f, 0f, 1f,     g53, g53, g53,      0.7f, 0.7f, 0.7f,
        ]);
        // A second picture in the batch is ignored (ComfyUI reads image[0]).
        var batch = new ImageBatch(2, 2, 3, 3, [.. img.Data, .. Enumerable.Repeat(0.9f, 18)]);
        var (rgb, lum, red, green, blue) = new ImageHistogram().Execute(batch);

        Dictionary<int, long> Counts(Histogram hist)
        {
            Assert.That(hist.Bins, Has.Length.EqualTo(256));
            return Enumerable.Range(0, 256).Where(i => hist.Bins[i] != 0).ToDictionary(i => i, i => hist.Bins[i]);
        }
        Assert.That(Counts(red), Is.EqualTo(new Dictionary<int, long> { [0] = 1, [51] = 1, [53] = 1, [178] = 1, [255] = 2 }));
        Assert.That(Counts(green), Is.EqualTo(new Dictionary<int, long> { [0] = 2, [51] = 1, [53] = 1, [127] = 1, [178] = 1 }));
        Assert.That(Counts(blue), Is.EqualTo(new Dictionary<int, long> { [0] = 1, [51] = 1, [53] = 1, [178] = 1, [254] = 1, [255] = 1 }));
        // (r + g + b) // 3 per level: level 0 has 1 + 2 + 1 = 4 -> 1, level 127 has 1 -> 0.
        Assert.That(Counts(rgb), Is.EqualTo(new Dictionary<int, long> { [0] = 1, [51] = 1, [53] = 1, [178] = 1, [255] = 1 }));
        // The grey 53/255 lands on 52: 0.2126 r + 0.7152 g + 0.0722 b in float32 is just under 53/255 (in double it would be 53).
        Assert.That(Counts(lum), Is.EqualTo(new Dictionary<int, long> { [18] = 1, [51] = 1, [52] = 1, [81] = 1, [145] = 1, [178] = 1 }));
        Assert.That(red.ToString(), Does.StartWith("[1, 0, 0,"));
    }

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }
}
