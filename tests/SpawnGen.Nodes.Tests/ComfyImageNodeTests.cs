using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// The comfy_extras image nodes: registration as ComfyUI declares them, values worked out by hand from the Python, the saved files, and pictures of
/// real results written to the test folder (comfy-image-nodes/*.png) to look at.
/// </summary>
public class ComfyImageNodeTests
{
    static readonly NodeRegistry Registry = new NodeRegistry().AddAssembly(typeof(ImageScale).Assembly);

    static NodeSchema Schema(string id) => Registry.GetSchema(id) ?? throw new AssertionException($"{id} is not registered");

    static string[] Inputs(string id) => Schema(id).Inputs.Select(i => i.Name).ToArray();

    [TestCase("ImageAddNoise", "Add Noise to Image", "image/filters")]
    [TestCase("ImageCropToMask", "Crop Image to Mask", "image/transform")]
    [TestCase("ImageScaleToMaxDimension", "Scale Image to Max Dimension", "image/upscaling")]
    [TestCase("ImageStitch", "Stitch Images", "image/transform")]
    [TestCase("ResizeAndPadImage", "Resize And Pad Image", "image/transform")]
    [TestCase("ImageColorSpace", "Convert Image Color Space", "image/color")]
    [TestCase("SaveAnimatedPNG", "Save Animated PNG", "image")]
    [TestCase("SaveSVGNode", "Save SVG", "image")]
    [TestCase("PrimitiveBoundingBox", "Bounding Box", "utilities/primitive")]
    [TestCase("JoinImageWithAlpha", "Join Image with Alpha", "image/compositing")]
    [TestCase("PorterDuffImageComposite", "Porter-Duff Image Composite", "image/compositing")]
    [TestCase("SplitImageWithAlpha", "Split Image with Alpha", "image/compositing")]
    [TestCase("ImageRGBToYUV", "Image RGB to YUV", "image/color")]
    [TestCase("ImageYUVToRGB", "Image YUV to RGB", "image/color")]
    [TestCase("Canny", "Detect Edges (Canny)", "image/filters")]
    [TestCase("ColorToRGBInt", "Color Picker", "utilities")]
    [TestCase("Morphology", "Apply Morphology", "image/filters")]
    public void Registers_with_ComfyUIs_id_name_and_category(string id, string display, string category)
    {
        var s = Schema(id);
        Assert.That((s.DisplayName, s.Category), Is.EqualTo((display, category)));
    }

    [Test]
    public void Inputs_outputs_defaults_and_combo_lists_are_ComfyUIs()
    {
        Assert.That(Inputs("ImageAddNoise"), Is.EqualTo(new[] { "image", "seed", "strength" }));
        Assert.That(Inputs("ImageCropToMask"), Is.EqualTo(new[] { "images", "masks", "width", "height", "pad_factor", "grow_mask", "background" }));
        Assert.That(Inputs("ImageScaleToMaxDimension"), Is.EqualTo(new[] { "image", "upscale_method", "largest_size" }));
        Assert.That(Inputs("ImageStitch"), Is.EqualTo(new[] { "image1", "direction", "match_image_size", "spacing_width", "spacing_color", "image2" }));
        Assert.That(Inputs("ResizeAndPadImage"), Is.EqualTo(new[] { "image", "target_width", "target_height", "padding_color", "interpolation" }));
        Assert.That(Inputs("ImageColorSpace"), Is.EqualTo(new[] { "image", "source", "destination" }));
        Assert.That(Inputs("SaveAnimatedPNG"), Is.EqualTo(new[] { "images", "filename_prefix", "fps", "compress_level" }));
        Assert.That(Inputs("SaveSVGNode"), Is.EqualTo(new[] { "svg", "filename_prefix" }));
        Assert.That(Inputs("PrimitiveBoundingBox"), Is.EqualTo(new[] { "x", "y", "width", "height" }));
        Assert.That(Inputs("JoinImageWithAlpha"), Is.EqualTo(new[] { "image", "alpha" }));
        Assert.That(Inputs("PorterDuffImageComposite"), Is.EqualTo(new[] { "source", "source_alpha", "destination", "destination_alpha", "mode" }));
        Assert.That(Inputs("ImageYUVToRGB"), Is.EqualTo(new[] { "Y", "U", "V" }));
        Assert.That(Inputs("Canny"), Is.EqualTo(new[] { "image", "low_threshold", "high_threshold" }));
        Assert.That(Inputs("ColorToRGBInt"), Is.EqualTo(new[] { "color" }));
        Assert.That(Inputs("Morphology"), Is.EqualTo(new[] { "image", "operation", "kernel_size" }));

        WidgetSpec W(string id, string input) => Schema(id).FindInput(input)!.Widget!;
        Assert.That(Schema("ImageStitch").FindInput("image2")!.Required, Is.False);
        Assert.That(W("ImageStitch", "match_image_size").Default, Is.EqualTo(true));
        Assert.That((W("ImageStitch", "spacing_width").Max, W("ImageStitch", "spacing_width").Step), Is.EqualTo(((double?)1024, (double?)2)));
        Assert.That(W("ImageStitch", "spacing_color").ResolveOptions(), Is.EqualTo(new[] { "white", "black", "red", "green", "blue" }));
        Assert.That(W("ImageScaleToMaxDimension", "upscale_method").ResolveOptions(), Is.EqualTo(new[] { "area", "lanczos", "bilinear", "nearest-exact", "bilinear", "bicubic" }));
        Assert.That(W("ImageScaleToMaxDimension", "upscale_method").Default, Is.EqualTo("area"));
        Assert.That(W("ResizeAndPadImage", "interpolation").ResolveOptions(), Is.EqualTo(new[] { "area", "bicubic", "nearest-exact", "bilinear", "lanczos" }));
        Assert.That(W("ResizeAndPadImage", "padding_color").Default, Is.EqualTo("white"));
        Assert.That(W("PorterDuffImageComposite", "mode").ResolveOptions(), Has.Count.EqualTo(18).And.Contains("SRC_OVER"));
        Assert.That(W("PorterDuffImageComposite", "mode").Default, Is.EqualTo("DST"));
        Assert.That(W("ImageColorSpace", "source").ResolveOptions(), Is.EqualTo(new[] { "sRGB", "HDR", "HDR PQ", "linear", "HDR LogC3", "HDR ACEScct" }));
        Assert.That((W("Canny", "low_threshold").Default, W("Canny", "high_threshold").Default), Is.EqualTo((0.4f, 0.8f)).Using<(object?, object?)>((a, b) =>
            Math.Abs(Convert.ToDouble(a.Item1) - Convert.ToDouble(b.Item1)) < 1e-6 && Math.Abs(Convert.ToDouble(a.Item2) - Convert.ToDouble(b.Item2)) < 1e-6));
        Assert.That(W("Morphology", "operation").Default, Is.EqualTo("erode"));
        Assert.That((W("ImageCropToMask", "width").Min, W("ImageCropToMask", "width").Max, W("ImageCropToMask", "width").Step), Is.EqualTo(((double?)64, (double?)4096, (double?)8)));
        Assert.That(W("ImageCropToMask", "background").Default, Is.EqualTo("#000000"));
        Assert.That(W("ColorToRGBInt", "color").Default, Is.EqualTo("#ffffff"));
        Assert.That(W("SaveSVGNode", "filename_prefix").Default, Is.EqualTo("svg/ComfyUI"));
        Assert.That((W("SaveAnimatedPNG", "fps").Default, W("SaveAnimatedPNG", "compress_level").Default), Is.EqualTo((6.0, 4L)));

        string[] Outs(string id) => Schema(id).Outputs.Select(o => o.Type.Name).ToArray();
        Assert.That(Outs("PorterDuffImageComposite"), Is.EqualTo(new[] { "IMAGE", "MASK" }));
        Assert.That(Outs("SplitImageWithAlpha"), Is.EqualTo(new[] { "IMAGE", "MASK" }));
        Assert.That(Outs("ImageRGBToYUV"), Is.EqualTo(new[] { "IMAGE", "IMAGE", "IMAGE" }));
        Assert.That(Outs("ColorToRGBInt"), Is.EqualTo(new[] { "INT", "STRING", "FLOAT" }));
        Assert.That(Schema("ColorToRGBInt").Outputs.Select(o => o.Name), Is.EqualTo(new[] { "rgb_int", "hex", "alpha" }));
        Assert.That(Outs("PrimitiveBoundingBox"), Is.EqualTo(new[] { "BOUNDING_BOX" }));
        Assert.That(Outs("SaveSVGNode"), Is.EqualTo(new[] { "SVG" }));
        Assert.That(Outs("SaveAnimatedPNG"), Is.EqualTo(new[] { "IMAGE" }));
        Assert.That(Schema("SaveAnimatedPNG").IsOutputNode && Schema("SaveSVGNode").IsOutputNode, Is.True);
    }

    // ---- values worked out by hand from the Python ----

    [Test]
    public void Torch_randn_gives_the_well_known_numbers_for_seed_0_and_42()
    {
        // torch.manual_seed(0); torch.randn(4, 4) (16 values: normal_fill) and torch.manual_seed(42); torch.randn(5) (fewer than 16: scalar path).
        Assert.That(TorchRandom.Randn(0, 16).Take(8), Is.EqualTo(new[] { -1.1258f, -1.1524f, -0.2506f, -0.4339f, 0.8487f, 0.6920f, -0.3160f, -2.1152f }).Within(1e-4f));
        Assert.That(TorchRandom.Randn(42, 5), Is.EqualTo(new[] { 0.3367f, 0.1288f, 0.2345f, 0.2303f, -1.1229f }).Within(1e-4f));
        Assert.That(TorchRandom.Randn(0, 4), Is.EqualTo(new[] { 1.5410f, -0.2934f, -2.1788f, 0.5684f }).Within(1e-4f));
        Assert.That(TorchRandom.Randn((1L << 32) + 42, 5), Is.EqualTo(TorchRandom.Randn(42, 5)), "only the low 32 bits seed mt19937");
    }

    [Test]
    public void Add_noise_is_the_image_plus_strength_times_randn_clamped_and_keeps_alpha()
    {
        var img = new ImageBatch(1, 2, 2, 4, Enumerable.Repeat(0.5f, 16).ToArray());
        var r = new ImageAddNoise().Execute(img, 0, 0.1f);
        var n = TorchRandom.Randn(0, 16);
        Assert.That(r.Data[0], Is.EqualTo(0.5f + 0.1f * n[0]).Within(1e-7));        // 0.5 - 0.11258
        Assert.That(r.Data[0], Is.EqualTo(0.38742f).Within(1e-4));
        Assert.That(r.Data[3], Is.EqualTo(0.5f), "alpha untouched");
        var grey = new ImageBatch(1, 4, 4, 1, Enumerable.Repeat(0.5f, 16).ToArray());
        Assert.That(new ImageAddNoise().Execute(grey, 0, 1f).Data[7], Is.EqualTo(0f), "0.5 - 2.1152 clamps to 0");
        Assert.That(new ImageAddNoise().Execute(grey, 0, 1f).Data[4], Is.EqualTo(1f), "0.5 + 0.8487 clamps to 1");
    }

    [Test]
    public void Porter_duff_over_and_the_alpha_convention()
    {
        // Source red at alpha 0.75 (mask 0.25) over destination blue at alpha 0.5 (mask 0.5):
        // out alpha = 0.75 + 0.25 * 0.5 = 0.875; premultiplied red = 0.75, blue = 0.25 * 0.5 = 0.125; un-premultiplied 0.857..., 0.142...
        var src = new ImageBatch(1, 1, 1, 3, [1f, 0f, 0f]);
        var dst = new ImageBatch(1, 1, 1, 3, [0f, 0f, 1f]);
        var (image, mask) = new PorterDuffImageComposite().Execute(src, new MaskBatch(1, 1, 1, [0.25f]), dst, new MaskBatch(1, 1, 1, [0.5f]), "SRC_OVER");
        Assert.That(image.Data, Is.EqualTo(new[] { 0.75f / 0.875f, 0f, 0.125f / 0.875f }).Within(1e-6));
        Assert.That(mask.Data[0], Is.EqualTo(0.125f).Within(1e-6));
        var (_, clear) = new PorterDuffImageComposite().Execute(src, new MaskBatch(1, 1, 1, [0.25f]), dst, new MaskBatch(1, 1, 1, [0.5f]), "CLEAR");
        Assert.That(clear.Data[0], Is.EqualTo(1f));
        Assert.Throws<ArgumentException>(() => new PorterDuffImageComposite().Execute(src, new MaskBatch(1, 1, 1), dst, new MaskBatch(1, 1, 1), "src_over"));
    }

    [Test]
    public void Split_and_join_alpha_round_trip_with_the_mask_inverted()
    {
        var rgba = new ImageBatch(1, 1, 2, 4, [0.1f, 0.2f, 0.3f, 1f, 0.4f, 0.5f, 0.6f, 0.25f]);
        var (rgb, mask) = new SplitImageWithAlpha().Execute(rgba);
        Assert.That(rgb.Data, Is.EqualTo(new[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f }));
        Assert.That(mask.Data, Is.EqualTo(new[] { 0f, 0.75f }));
        Assert.That(new JoinImageWithAlpha().Execute(rgb, mask).Data, Is.EqualTo(rgba.Data));
        Assert.That(new SplitImageWithAlpha().Execute(rgb).Mask.Data, Is.EqualTo(new[] { 0f, 0f }), "no alpha: opaque");
    }

    [Test]
    public void Yuv_of_pure_red_and_back()
    {
        // Y = 0.299; U = (0 - 0.299) * 0.564 + 0.5 = 0.331364; V = (1 - 0.299) * 0.713 + 0.5 = 0.999813.
        var (y, u, v) = new ImageRGBToYUV().Execute(new ImageBatch(1, 1, 1, 3, [1f, 0f, 0f]));
        Assert.That(y.Data, Is.EqualTo(new[] { 0.299f, 0.299f, 0.299f }).Within(1e-6));
        Assert.That(u.Data[0], Is.EqualTo(0.331364f).Within(1e-6));
        Assert.That(v.Data[0], Is.EqualTo(0.999813f).Within(1e-6));
        // Back: R = 0.299 + 1.403 * 0.499813 = 1.000238 (kornia clamps it to 1), G = 0.299 - 0.714 * 0.499813 - 0.344 * (-0.168636) = 0.000145,
        // B = 0.299 + 1.773 * (-0.168636) = 0.000008.
        var rgb = new ImageYUVToRGB().Execute(y, u, v);
        Assert.That(rgb.Data, Is.EqualTo(new[] { 1f, 0.000145f, 0.000008f }).Within(2e-6));
    }

    [Test]
    public void Scale_to_max_dimension_sizes()
    {
        // 9 wide, 6 tall, largest 13: height = round(6 / 9 * 13) = round(8.667) = 9; a 5x5 square becomes 13x13.
        var r = new ImageScaleToMaxDimension().Execute(new ImageBatch(1, 6, 9, 3), "bilinear", 13);
        Assert.That((r.Width, r.Height), Is.EqualTo((13, 9)));
        // Python's round is half to even: 2 wide, 1 tall... 3 tall x 2 wide at largest 3: width = round(2/3 * 3) = round(2.0) = 2; 4 tall x 2 wide at 5: round(2.5) = 2.
        Assert.That(new ImageScaleToMaxDimension().Execute(new ImageBatch(1, 4, 2, 3), "area", 5).Width, Is.EqualTo(2));
        Assert.That(new ImageScaleToMaxDimension().Execute(new ImageBatch(1, 4, 6, 3), "area", 5).Height, Is.EqualTo(3), "round(3.333)");
    }

    [Test]
    public void Resize_and_pad_centres_the_fitted_picture()
    {
        // 8x5 into 10x4: scale = min(1.25, 0.8) = 0.8 -> int(6.4) = 6 by int(4.0) = 4, offset x (10 - 6) / 2 = 2.
        var img = new ImageBatch(1, 5, 8, 3, Enumerable.Repeat(0.3f, 120).ToArray());
        var r = new ResizeAndPadImage().Execute(img, 10, 4, "black", "nearest-exact");
        Assert.That((r.Width, r.Height), Is.EqualTo((10, 4)));
        float At(int x, int y) => r.Data[(y * 10 + x) * 3];
        Assert.That(new[] { At(0, 0), At(1, 3), At(2, 0), At(7, 3), At(8, 0), At(9, 3) }, Is.EqualTo(new[] { 0f, 0f, 0.3f, 0.3f, 0f, 0f }));
    }

    [Test]
    public void Stitch_sizes_spacing_and_channels()
    {
        var a = new ImageBatch(2, 4, 5, 3);
        var b = new ImageBatch(1, 6, 3, 4);
        // Right, matched: image2 becomes 4 tall and int(4 * 3 / 6) = 2 wide; spacing 3 rounds up to 4; channels padded to 4 (alpha 1).
        var r = new ImageStitch().Execute(a, "right", true, 3, "red", b);
        Assert.That(r.Shape, Is.EqualTo(new[] { 2, 4, 5 + 4 + 2, 4 }));
        Assert.That(r.Data.Skip((5 * 4) + 0).Take(4), Is.EqualTo(new[] { 1f, 0f, 0f, 1f }), "the spacing is red and opaque");
        Assert.That(r.Data[3], Is.EqualTo(1f), "image1 got an alpha of 1");
        // Down, not matched: widths padded to 5 (image2: 1 left, 1 right), heights add up.
        var d = new ImageStitch().Execute(a, "down", false, 0, "white", b);
        Assert.That(d.Shape, Is.EqualTo(new[] { 2, 10, 5, 4 }));
        Assert.That(new ImageStitch().Execute(a, "left", true, 0, "white"), Is.SameAs(a), "no image2: image1 as it is");
    }

    [Test]
    public void Color_picker_parses_like_python()
    {
        Assert.That(new ColorToRGBInt().Execute("#ff8000"), Is.EqualTo((0xFF8000, "#ff8000", 1.0)));
        Assert.That(new ColorToRGBInt().Execute("#12345680"), Is.EqualTo((0x123456, "#123456", 128 / 255.0)));
        Assert.That(new ColorToRGBInt().Execute("#ff_fff"), Is.EqualTo((0xFFFFFF, "#ff_fff", 1.0)), "int() accepts it, hex_to_rgb falls back to white");
        Assert.Throws<ArgumentException>(() => new ColorToRGBInt().Execute("ff8000"));
        Assert.Throws<ArgumentException>(() => new ColorToRGBInt().Execute("#ff80zz"));
    }

    [Test]
    public void Morphology_even_kernels_reach_one_pixel_further_back_like_kornia()
    {
        // One bright pixel at x = 3 of a 7x1 row, kernel 4: the window of x is x-2 .. x+1, so dilation lights x = 2..5.
        var img = new ImageBatch(1, 1, 7, 1, [0, 0, 0, 1, 0, 0, 0]);
        Assert.That(new Morphology().Execute(img, "dilate", 4).Data, Is.EqualTo(new float[] { 0, 0, 1, 1, 1, 1, 0 }));
        Assert.That(new Morphology().Execute(img, "dilate", 3).Data, Is.EqualTo(new float[] { 0, 0, 1, 1, 1, 0, 0 }));
    }

    [Test]
    public void Canny_finds_a_vertical_step_and_thresholds_are_checked()
    {
        // Left half black, right half white: one edge column where the gradient peaks, nothing else.
        var img = new ImageBatch(1, 8, 10, 3);
        for (int y = 0; y < 8; y++) for (int x = 5; x < 10; x++) for (int k = 0; k < 3; k++) img.Data[(y * 10 + x) * 3 + k] = 1f;
        var e = new Canny().Execute(img, 0.4f, 0.8f);
        var columns = Enumerable.Range(0, 10).Where(x => Enumerable.Range(0, 8).All(y => e.Data[(y * 10 + x) * 3] == 1f)).ToArray();
        Assert.That(columns, Has.Length.EqualTo(1));
        Assert.That(columns[0], Is.EqualTo(4).Or.EqualTo(5));
        Assert.That(e.Data.Count(v => v == 1f), Is.EqualTo(8 * 3));
        Assert.Throws<ArgumentException>(() => new Canny().Execute(img, 0.8f, 0.4f));
    }

    [Test]
    public void Color_space_round_trips_and_white_maps_to_reference_levels()
    {
        var img = new ImageBatch(1, 1, 3, 3, [1f, 1f, 1f, 0.5f, 0.25f, 0.75f, 0.2f, 0.2f, 0.2f]);
        foreach (var space in new[] { "linear", "HDR", "HDR PQ", "HDR LogC3", "HDR ACEScct" })
        {
            var back = new ImageColorSpace().Execute(new ImageColorSpace().Execute(img, "sRGB", space), space, "sRGB");
            Assert.That(back.Data, Is.EqualTo(img.Data).Within(2e-4), space);
        }
        // sRGB white is 203 nits: HLG 0.75 (BT.2408 reference white), PQ 0.58.
        Assert.That(new ImageColorSpace().Execute(img, "sRGB", "HDR").Data[0], Is.EqualTo(0.75f).Within(0.002));
        Assert.That(new ImageColorSpace().Execute(img, "sRGB", "HDR PQ").Data[0], Is.EqualTo(0.58f).Within(0.002));
        Assert.That(new ImageColorSpace().Execute(img, "sRGB", "sRGB"), Is.SameAs(img));
    }

    [Test]
    public void Bounding_box_primitive()
    {
        Assert.That(new PrimitiveBoundingBox().Execute(1, 2, 3, 4), Is.EqualTo(new BoundingBox(1, 2, 3, 4)));
    }

    // ---- saved files ----

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
            ["1", "X", services, CancellationToken.None, null, new Lazy<string>(() => prompt), new Dictionary<string, string> { ["workflow"] = "{\"nodes\":[\"é\"]}" }, null], null)!;

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

    [Test]
    public async Task Save_animated_png_writes_one_apng_with_merged_repeats_changed_rectangles_and_comf_chunks()
    {
        var (services, store) = Services();
        // Three 4x3 frames: the second equal to the first (merged, so it shows twice as long), the third with one pixel changed at (2, 1).
        var images = new ImageBatch(3, 3, 4, 3);
        for (int i = 0; i < images.Data.Length; i++) images.Data[i] = (i % 36) / 40f;
        images.Data[2 * 36 + (1 * 4 + 2) * 3] = 1f;
        var (passed, ui) = await new SaveAnimatedPNG().ExecuteAsync(images, Context(services), services, "anim", 4.0, 9);
        Assert.That(passed, Is.SameAs(images));
        var file = (ResultFile)((object[])ui["images"]!)[0];
        Assert.That(file.Filename, Is.EqualTo("anim_00001_.png"));
        Assert.That(ui["animated"], Is.EqualTo(new object[] { true }));
        var chunks = Chunks(store.Get("output", file.Filename)!);
        Assert.That(chunks.Select(c => c.Type), Is.EqualTo(new[] { "IHDR", "acTL", "fcTL", "IDAT", "fcTL", "fdAT", "comf", "comf", "IEND" }));
        Assert.That(BinaryPrimitives.ReadUInt32BigEndian(chunks[1].Data), Is.EqualTo(2u), "two frames after merging");
        var f0 = chunks[2].Data;
        Assert.That(BinaryPrimitives.ReadUInt16BigEndian(f0.AsSpan(20)), Is.EqualTo(500), "int(1000 / 4) ms, twice");
        var f1 = chunks[4].Data;
        Assert.That(new[] { BinaryPrimitives.ReadUInt32BigEndian(f1), BinaryPrimitives.ReadUInt32BigEndian(f1.AsSpan(4)), BinaryPrimitives.ReadUInt32BigEndian(f1.AsSpan(8)),
            BinaryPrimitives.ReadUInt32BigEndian(f1.AsSpan(12)), BinaryPrimitives.ReadUInt32BigEndian(f1.AsSpan(16)) }, Is.EqualTo(new uint[] { 1, 1, 1, 2, 1 }),
            "sequence 1, a 1x1 rectangle at (2, 1)");
        Assert.That(BinaryPrimitives.ReadUInt32BigEndian(chunks[5].Data), Is.EqualTo(2u), "fdAT sequence number");
        Assert.That(Encoding.Latin1.GetString(chunks[6].Data), Is.EqualTo("prompt\0{\"1\":{\"class_type\":\"X\"}}"));
        Assert.That(Encoding.Latin1.GetString(chunks[7].Data), Is.EqualTo("workflow\0{\"nodes\":[\"\\u00e9\"]}"), "json.dumps escapes non-ASCII");
        // Pixels are 255 * x truncated (0.1 * 255 = 25.5 -> 25), as ComfyUI's PIL conversion does.
        using var z = new ZLibStream(new MemoryStream(chunks[3].Data), CompressionMode.Decompress);
        var raw = new MemoryStream();
        z.CopyTo(raw);
        Assert.That(raw.ToArray()[1 + 3], Is.EqualTo((byte)(3 / 40f * 255f)));
        WriteResult("save_animated_png.png", store.Get("output", file.Filename)!);

        // One picture (or all equal) is a still PNG, without "animated".
        var (_, still) = await new SaveAnimatedPNG().ExecuteAsync(new ImageBatch(1, 2, 2, 3), Context(services), services, "anim", 6.0, 4);
        Assert.That(still.ContainsKey("animated"), Is.False);
        Assert.That(Chunks(store.Get("output", ((ResultFile)((object[])still["images"]!)[0]).Filename)!).Select(c => c.Type), Is.EqualTo(new[] { "IHDR", "IDAT", "comf", "comf", "IEND" }));
    }

    [Test]
    public async Task Save_svg_puts_comfyuis_metadata_block_after_the_svg_tag()
    {
        var (services, store) = Services();
        var svg = new SvgImages([Encoding.UTF8.GetBytes("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"4\"><rect/></svg>"), Encoding.UTF8.GetBytes("<svg><g/></svg>")]);
        var (passed, ui) = await new SaveSVGNode().ExecuteAsync(svg, Context(services), services, "svg/test_%batch_num%");
        Assert.That(passed, Is.SameAs(svg));
        var files = ((List<object>)ui["images"]!).Cast<ResultFile>().ToArray();
        Assert.That(files.Select(f => (f.Subfolder, f.Filename)), Is.EqualTo(new[] { ("svg", "test_0_00001_.svg"), ("svg", "test_1_00001_.svg") }));
        var text = Encoding.UTF8.GetString(store.Get("output", files[0].Filename, "svg")!);
        // json.dumps({"prompt": ..., "workflow": ...}, indent=2) inside ComfyUI's CDATA block.
        const string json = "{\n  \"prompt\": {\n    \"1\": {\n      \"class_type\": \"X\"\n    }\n  },\n  \"workflow\": {\n    \"nodes\": [\n      \"\\u00e9\"\n    ]\n  }\n}";
        Assert.That(text, Is.EqualTo("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"4\">\n  <metadata>\n                <![CDATA[\n            " + json +
            "\n                ]]>\n            </metadata>\n            <rect/></svg>"));
    }

    // ---- pictures to look at ----

    static string OutputDir()
    {
        var dir = Path.Combine(TestContext.CurrentContext.WorkDirectory, "comfy-image-nodes");
        Directory.CreateDirectory(dir);
        return dir;
    }

    static void WriteResult(string name, byte[] png)
    {
        var path = Path.Combine(OutputDir(), name);
        File.WriteAllBytes(path, png);
        TestContext.Out.WriteLine(path);
    }

    static void Write(string name, ImageBatch img, int index = 0) => WriteResult(name, PngWriter.Encode(img.Channels == 3 || img.Channels == 4 ? img : ToRgb(img), index));

    static ImageBatch ToRgb(ImageBatch img)
    {
        var r = new ImageBatch(img.Batch, img.Height, img.Width, 3);
        for (int p = 0; p < img.Batch * img.Height * img.Width; p++) for (int k = 0; k < 3; k++) r.Data[p * 3 + k] = img.Data[p * img.Channels];
        return r;
    }

    /// <summary>A 192x128 test scene: a sky-to-grass gradient, a red disc, a blue square and a thin white line.</summary>
    static (ImageBatch Image, MaskBatch Disc) Scene()
    {
        int w = 192, h = 128;
        var img = new ImageBatch(1, h, w, 3);
        var disc = new MaskBatch(1, h, w);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                float t = y / (float)h;
                float r = 0.4f * (1 - t), g = 0.5f + 0.3f * t, b = 0.9f * (1 - t);
                bool inDisc = (x - 60) * (x - 60) + (y - 60) * (y - 60) < 30 * 30;
                if (inDisc) { r = 0.9f; g = 0.15f; b = 0.1f; disc.Data[y * w + x] = 1f; }
                if (x > 110 && x < 160 && y > 30 && y < 80) { r = 0.1f; g = 0.2f; b = 0.8f; }
                if (y == 100 && x > 20) { r = g = b = 1f; }
                img.Data[(y * w + x) * 3] = r; img.Data[(y * w + x) * 3 + 1] = g; img.Data[(y * w + x) * 3 + 2] = b;
            }
        return (img, disc);
    }

    [Test]
    public void Pictures_of_real_results()
    {
        var (scene, disc) = Scene();
        Write("scene.png", scene);
        Write("canny.png", new Canny().Execute(scene, 0.4f, 0.8f));
        Write("canny_low.png", new Canny().Execute(new ImageAddNoise().Execute(scene, 3, 0.05f), 0.1f, 0.3f));
        Write("add_noise.png", new ImageAddNoise().Execute(scene, 0, 0.2f));
        var flipped = new ImageFlip().Execute(scene, "y-axis: horizontally");
        Write("stitch_right_red_spacing.png", new ImageStitch().Execute(scene, "right", true, 8, "red", new ImageScaleToMaxDimension().Execute(flipped, "lanczos", 64)));
        Write("stitch_down_unmatched.png", new ImageStitch().Execute(scene, "down", false, 4, "black", new ImageScaleToMaxDimension().Execute(flipped, "bicubic", 96)));
        Write("resize_and_pad.png", new ResizeAndPadImage().Execute(scene, 128, 128, "white", "lanczos"));
        Write("scale_to_max_96_area.png", new ImageScaleToMaxDimension().Execute(scene, "area", 96));
        var inverted = new MaskBatch(1, disc.Height, disc.Width, disc.Data.Select(v => 1f - v).ToArray());
        Write("crop_to_mask.png", new ImageCropToMask().Execute(scene, disc, 128, 128, 1.2, 2, "#ffffff"));
        // Porter-Duff: the scene's disc (source, opaque only on the disc) over a dark copy, and XOR.
        var dark = new ImageBatch(1, scene.Height, scene.Width, 3, scene.Data.Select(v => v * 0.3f).ToArray());
        var halfMask = MaskBatch.Filled(1, scene.Height, scene.Width, 0.5f);
        Write("porter_duff_src_over.png", new PorterDuffImageComposite().Execute(scene, inverted, dark, halfMask, "SRC_OVER").Image);
        Write("porter_duff_xor_mask.png", new MaskToImage().Execute(new PorterDuffImageComposite().Execute(scene, inverted, dark, halfMask, "XOR").Mask));
        Write("join_alpha.png", new JoinImageWithAlpha().Execute(scene, inverted));
        var (y, u, v) = new ImageRGBToYUV().Execute(scene);
        Write("yuv_y.png", y); Write("yuv_u.png", u); Write("yuv_v.png", v);
        Write("yuv_back.png", new ImageYUVToRGB().Execute(y, u, v));
        Write("srgb_to_linear.png", new ImageColorSpace().Execute(scene, "sRGB", "linear"));
        Write("srgb_to_hdr_hlg.png", new ImageColorSpace().Execute(scene, "sRGB", "HDR"));
        Write("srgb_to_logc3.png", new ImageColorSpace().Execute(scene, "sRGB", "HDR LogC3"));
        Write("morphology_gradient_4.png", new Morphology().Execute(scene, "gradient", 4));
    }
}
