using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>The host-side node library: every node registers with ComfyUI's id and widget order, and does what ComfyUI's node does on small hand-checkable inputs.</summary>
public class NodeLibraryTests
{
    static ImageBatch Img(int w, int h, params float[] rgb) => new(1, h, w, 3, rgb.Length == w * h * 3 ? rgb : Enumerable.Range(0, w * h * 3).Select(i => rgb[i % rgb.Length]).ToArray());
    static MaskBatch Mask(int w, int h, params float[] v) => new(1, h, w, v);
    static LatentBatch Lat(int c, int h, int w, Func<int, float> f) => new(1, c, h, w, Enumerable.Range(0, c * h * w).Select(f).ToArray());

    // ---- registration ----

    [Test]
    public void Every_core_node_registers_with_ComfyUIs_id_and_widget_order()
    {
        var registry = new NodeRegistry().AddAssembly(typeof(ImageScale).Assembly);
        string[] expected =
        [
            "ImageScale", "ImageScaleBy", "ImageScaleToTotalPixels", "ImageInvert", "ImageBatch", "ImageCrop", "ImageFromBatch", "RepeatImageBatch", "EmptyImage",
            "ImageBlend", "ImageBlur", "ImageSharpen", "ImageCompositeMasked", "ImageToMask", "MaskToImage", "ImagePadForOutpaint",
            "SolidMask", "InvertMask", "CropMask", "MaskComposite", "FeatherMask", "GrowMask", "ThresholdMask",
            "LatentUpscale", "LatentUpscaleBy", "LatentFromBatch", "RepeatLatentBatch", "LatentComposite", "LatentFlip", "LatentRotate", "LatentCrop", "LatentBlend", "SetLatentNoiseMask",
            "PrimitiveString", "PrimitiveStringMultiline", "PrimitiveInt", "PrimitiveFloat", "PrimitiveBoolean",
            "LoadImage", "SaveImage", "PreviewImage", "EmptyLatentImage",
        ];
        foreach (var id in expected) Assert.That(registry.GetSchema(id), Is.Not.Null, id);

        // Workflow files store widget values by position, so the order must be ComfyUI's.
        string[] Widgets(string id) => registry.GetSchema(id)!.Inputs.Where(i => i.Widget is not null).Select(i => i.Name).ToArray();
        Assert.That(Widgets("ImageScale"), Is.EqualTo(new[] { "upscale_method", "width", "height", "crop" }));
        Assert.That(Widgets("ImageBlend"), Is.EqualTo(new[] { "blend_factor", "blend_mode" }));
        Assert.That(Widgets("ImageBlur"), Is.EqualTo(new[] { "blur_radius", "sigma" }));
        Assert.That(Widgets("ImagePadForOutpaint"), Is.EqualTo(new[] { "left", "top", "right", "bottom", "feathering" }));
        Assert.That(Widgets("MaskComposite"), Is.EqualTo(new[] { "x", "y", "operation" }));
        Assert.That(Widgets("GrowMask"), Is.EqualTo(new[] { "expand", "tapered_corners" }));
        Assert.That(Widgets("LatentUpscale"), Is.EqualTo(new[] { "upscale_method", "width", "height", "crop" }));
        Assert.That(Widgets("LatentComposite"), Is.EqualTo(new[] { "x", "y", "feather" }));
        Assert.That(registry.GetSchema("ImageScale")!.FindInput("width")!.Widget!.Default, Is.EqualTo(512L));
        Assert.That(registry.GetSchema("ImagePadForOutpaint")!.Outputs.Select(o => o.Type.Name), Is.EqualTo(new[] { "IMAGE", "MASK" }));
        Assert.That(registry.GetSchema("ImageBlend")!.FindInput("blend_mode")!.Widget!.ResolveOptions(), Does.Contain("soft_light"));
    }

    // ---- image ----

    [Test]
    public void ImageScale_keeps_the_aspect_ratio_when_one_side_is_zero_and_the_size_when_both_are()
    {
        var src = Img(8, 4, 0.5f);
        var s = new ImageScale();
        var r = s.Execute(src, "bilinear", 16, 0);
        Assert.That((r.Width, r.Height), Is.EqualTo((16, 8)));
        Assert.That(s.Execute(src, "bilinear", 0, 0), Is.SameAs(src));
        Assert.That(s.Execute(src, "bilinear", 0, 12).Width, Is.EqualTo(24));
    }

    [Test]
    public void ImageScale_with_center_crop_fills_the_target_exactly_from_the_middle()
    {
        // 4x2 picture: left half 0, right half 1. Cropping to a 2x2 square from the middle takes one 0 column and one 1 column.
        var src = new ImageBatch(1, 2, 4, 3);
        for (int y = 0; y < 2; y++) for (int x = 2; x < 4; x++) for (int c = 0; c < 3; c++) src.Data[(y * 4 + x) * 3 + c] = 1;
        var r = new ImageScale().Execute(src, "nearest-exact", 2, 2, "center");
        Assert.That((r.Width, r.Height), Is.EqualTo((2, 2)));
        Assert.That(new[] { r.Data[0], r.Data[3] }, Is.EqualTo(new[] { 0f, 1f }));
    }

    [Test]
    public void ImageScaleBy_and_total_pixels_set_the_expected_sizes()
    {
        var src = Img(10, 6, 0.2f);
        var by = new ImageScaleBy().Execute(src, "nearest-exact", 2.5f);
        Assert.That((by.Width, by.Height), Is.EqualTo((25, 15)));
        var total = new ImageScaleToTotalPixels().Execute(Img(400, 300, 0.1f), "area", 1f);
        Assert.That(total.Width * total.Height, Is.EqualTo(1024 * 1024).Within(2000));
        Assert.That((double)total.Width / total.Height, Is.EqualTo(4.0 / 3.0).Within(0.01));
        var stepped = new ImageScaleToTotalPixels().Execute(Img(400, 300, 0.1f), "area", 1f, 64);
        Assert.That((stepped.Width, stepped.Height), Is.EqualTo((1152, 896)), "ComfyUI: round(400 * 2.956 / 64) * 64 and round(300 * 2.956 / 64) * 64");
    }

    [Test]
    public void ImageInvert_twice_is_the_identity()
    {
        var src = Img(3, 2, 0.1f, 0.5f, 0.9f);
        var inv = new ImageInvert().Execute(src);
        Assert.That(inv.Data[0], Is.EqualTo(0.9f).Within(1e-6));
        Assert.That(new ImageInvert().Execute(inv).Data, Is.EqualTo(src.Data).Within(1e-6));
    }

    [Test]
    public void ImageBatch_stacks_and_scales_a_picture_of_another_size_to_the_first()
    {
        var a = Img(4, 4, 0.2f);
        var b = Img(8, 8, 0.8f);
        var r = new ImageBatchNode().Execute(a, b);
        Assert.That((r.Batch, r.Width, r.Height), Is.EqualTo((2, 4, 4)));
        Assert.That(r.Data[0], Is.EqualTo(0.2f).Within(1e-5));
        Assert.That(r.Data[4 * 4 * 3], Is.EqualTo(0.8f).Within(1e-3));
    }

    [Test]
    public void ImageCrop_cuts_the_region_and_clamps_to_the_picture()
    {
        var src = new ImageBatch(1, 4, 4, 3);
        for (int i = 0; i < 16; i++) src.Data[i * 3] = i;
        var r = new ImageCrop().Execute(src, 2, 2, 1, 1);
        Assert.That(new[] { r.Data[0], r.Data[3], r.Data[6], r.Data[9] }, Is.EqualTo(new float[] { 5, 6, 9, 10 }));
        var edge = new ImageCrop().Execute(src, 10, 10, 3, 3);
        Assert.That((edge.Width, edge.Height), Is.EqualTo((1, 1)));
    }

    [Test]
    public void ImageFromBatch_and_Repeat_pick_and_multiply_frames()
    {
        var two = new ImageBatchNode().Execute(Img(2, 2, 0.1f), Img(2, 2, 0.9f));
        Assert.That(new ImageFromBatch().Execute(two, 1, 1).Data[0], Is.EqualTo(0.9f));
        Assert.That(new ImageFromBatch().Execute(two, 5, 3).Batch, Is.EqualTo(1), "A start past the end clamps to the last frame.");
        var rep = new RepeatImageBatch().Execute(two, 3);
        Assert.That(rep.Batch, Is.EqualTo(6));
        Assert.That(rep.Data[2 * 2 * 2 * 3], Is.EqualTo(0.1f));
    }

    [Test]
    public void EmptyImage_fills_with_the_colour()
    {
        var r = new EmptyImage().Execute(2, 2, 3, 0xFF8000);
        Assert.That(r.Batch, Is.EqualTo(3));
        Assert.That(r.Data.Take(3), Is.EqualTo(new[] { 1f, 128 / 255f, 0f }).Within(1e-6));
    }

    [TestCase("normal", 0.3f, 0.6f, 0.6f)]
    [TestCase("multiply", 0.3f, 0.6f, 0.18f)]
    [TestCase("screen", 0.3f, 0.6f, 0.72f)]
    [TestCase("overlay", 0.3f, 0.6f, 0.36f)]
    [TestCase("overlay", 0.8f, 0.6f, 0.84f)]
    [TestCase("difference", 0.8f, 0.6f, 0.2f)]
    [TestCase("soft_light", 0.8f, 0.6f, 0.6f + 0.4f * 0.6f * 0.6f)]
    public void ImageBlend_modes_follow_ComfyUIs_formulas(string mode, float a, float b, float expected)
    {
        var r = new ImageBlend().Execute(Img(1, 1, a), Img(1, 1, b), 1f, mode);
        Assert.That(r.Data[0], Is.EqualTo(Math.Clamp(expected, 0f, 1f)).Within(1e-5));
    }

    [Test]
    public void ImageBlend_factor_mixes_with_the_first_picture_and_rejects_unknown_modes()
    {
        var half = new ImageBlend().Execute(Img(1, 1, 0.2f), Img(1, 1, 0.8f), 0.5f, "normal");
        Assert.That(half.Data[0], Is.EqualTo(0.5f).Within(1e-6));
        Assert.That(new ImageBlend().Execute(Img(1, 1, 0.2f), Img(1, 1, 0.8f), 0f, "multiply").Data[0], Is.EqualTo(0.2f).Within(1e-6));
        Assert.Throws<ArgumentException>(() => new ImageBlend().Execute(Img(1, 1, 0.2f), Img(1, 1, 0.8f), 1f, "dissolve"));
    }

    [Test]
    public void ImageBlur_spreads_an_impulse_symmetrically_and_conserves_its_total()
    {
        var src = new ImageBatch(1, 9, 9, 3);
        for (int c = 0; c < 3; c++) src.Data[(4 * 9 + 4) * 3 + c] = 1;
        var r = new ImageBlur().Execute(src, 2, 0.5f);
        Assert.That(r.Data[(4 * 9 + 4) * 3], Is.LessThan(1f).And.GreaterThan(0.1f));
        Assert.That(r.Data[(4 * 9 + 3) * 3], Is.EqualTo(r.Data[(4 * 9 + 5) * 3]).Within(1e-6));
        Assert.That(r.Data[(3 * 9 + 4) * 3], Is.EqualTo(r.Data[(5 * 9 + 4) * 3]).Within(1e-6));
        Assert.That(r.Data.Where((_, i) => i % 3 == 0).Sum(), Is.EqualTo(1f).Within(1e-4), "A normalised kernel keeps the total brightness.");
        Assert.That(new ImageBlur().Execute(Img(5, 5, 0.4f), 3, 1f).Data.All(v => Math.Abs(v - 0.4f) < 1e-5), Is.True, "A flat picture stays flat (edges reflect).");
    }

    [Test]
    public void ImageSharpen_leaves_flat_areas_alone_and_exaggerates_an_edge()
    {
        Assert.That(new ImageSharpen().Execute(Img(5, 5, 0.4f), 1, 1f, 1f).Data.All(v => Math.Abs(v - 0.4f) < 1e-5), Is.True);
        var edge = new ImageBatch(1, 1, 8, 3);
        for (int x = 4; x < 8; x++) for (int c = 0; c < 3; c++) edge.Data[x * 3 + c] = 0.6f;
        for (int x = 0; x < 4; x++) for (int c = 0; c < 3; c++) edge.Data[x * 3 + c] = 0.4f;
        var r = new ImageSharpen().Execute(edge, 1, 1f, 0.5f);
        Assert.That(r.Data[3 * 3], Is.LessThan(0.4f), "the dark side of the edge gets darker");
        Assert.That(r.Data[4 * 3], Is.GreaterThan(0.6f), "the bright side gets brighter");
    }

    [Test]
    public void ImageCompositeMasked_pastes_through_the_mask_and_cuts_what_hangs_over()
    {
        var dest = Img(4, 4, 0f);
        var src = Img(2, 2, 1f);
        var full = new ImageCompositeMasked().Execute(dest, src, 3, 3);
        Assert.That(full.Data[(3 * 4 + 3) * 3], Is.EqualTo(1f));
        Assert.That(full.Data.Sum(), Is.EqualTo(3f), "Only the one pixel that fits was pasted.");

        var halfMask = Mask(2, 2, 0.5f, 0.5f, 0.5f, 0.5f);
        var blended = new ImageCompositeMasked().Execute(dest, src, 0, 0, false, halfMask);
        Assert.That(blended.Data[0], Is.EqualTo(0.5f).Within(1e-6));
        var resized = new ImageCompositeMasked().Execute(dest, Img(1, 1, 1f), 0, 0, true);
        Assert.That(resized.Data.All(v => v == 1f), Is.True, "resize_source stretches the source over the destination.");
    }

    [Test]
    public void Image_to_mask_and_back_and_a_missing_alpha_is_opaque()
    {
        var src = Img(2, 1, 0.1f, 0.2f, 0.3f);
        Assert.That(new ImageToMask().Execute(src, "green").Data, Is.EqualTo(new[] { 0.2f, 0.2f }));
        Assert.That(new ImageToMask().Execute(src, "alpha").Data, Is.EqualTo(new[] { 1f, 1f }));
        var back = new MaskToImage().Execute(Mask(2, 1, 0.25f, 0.75f));
        Assert.That(back.Data, Is.EqualTo(new[] { 0.25f, 0.25f, 0.25f, 0.75f, 0.75f, 0.75f }));
    }

    [Test]
    public void ImagePadForOutpaint_adds_grey_borders_and_a_feathered_mask()
    {
        var src = Img(20, 20, 1f);
        var (img, mask) = new ImagePadForOutpaint().Execute(src, 10, 0, 0, 0, 8);
        Assert.That((img.Width, img.Height), Is.EqualTo((30, 20)));
        Assert.That(img.Data[0], Is.EqualTo(0.5f), "the new area is grey");
        Assert.That(img.Data[10 * 3], Is.EqualTo(1f), "the old picture is kept");
        Assert.That(mask.Data[0], Is.EqualTo(1f), "1 on the new area");
        Assert.That(mask.Data[10 + 19], Is.EqualTo(0f), "0 deep inside the old picture (only padded edges feather)");
        float next = mask.Data[10], inner = mask.Data[10 + 4];
        Assert.That(next, Is.GreaterThan(inner).And.LessThanOrEqualTo(1f), "the mask fades inside the picture next to the new area");
        Assert.That(mask.Data[10 + 8], Is.EqualTo(0f), "and is gone after `feathering` pixels");
    }

    // ---- masks ----

    [Test]
    public void SolidMask_Invert_Crop_and_Threshold()
    {
        var m = new SolidMask().Execute(0.25f, 4, 3);
        Assert.That((m.Width, m.Height, m.Data[0]), Is.EqualTo((4, 3, 0.25f)));
        Assert.That(new InvertMask().Execute(m).Data[0], Is.EqualTo(0.75f));
        var ramp = Mask(4, 1, 0f, 0.4f, 0.6f, 1f);
        Assert.That(new ThresholdMask().Execute(ramp, 0.5f).Data, Is.EqualTo(new[] { 0f, 0f, 1f, 1f }));
        Assert.That(new CropMask().Execute(ramp, 1, 0, 2, 5).Data, Is.EqualTo(new[] { 0.4f, 0.6f }));
    }

    [TestCase("multiply", 0.5f, 0.5f, 0.25f)]
    [TestCase("add", 0.7f, 0.6f, 1f)]
    [TestCase("subtract", 0.3f, 0.6f, 0f)]
    [TestCase("and", 1f, 0f, 0f)]
    [TestCase("or", 1f, 0f, 1f)]
    [TestCase("xor", 1f, 1f, 0f)]
    [TestCase("xor", 1f, 0f, 1f)]
    public void MaskComposite_operations(string op, float d, float s, float expected)
    {
        var r = new MaskComposite().Execute(Mask(2, 1, d, d), Mask(1, 1, s), 1, 0, op);
        Assert.That(r.Data, Is.EqualTo(new[] { d, expected }).Within(1e-6), "only the overlapped pixel changes");
    }

    [Test]
    public void FeatherMask_ramps_the_edges()
    {
        var r = new FeatherMask().Execute(Mask(5, 1, 1f, 1f, 1f, 1f, 1f), 2, 0, 2, 0);
        Assert.That(r.Data, Is.EqualTo(new[] { 0.5f, 1f, 1f, 1f, 0.5f }).Within(1e-6));
        var top = new FeatherMask().Execute(new MaskBatch(1, 2, 1, [1f, 1f]), 0, 2, 0, 0);
        Assert.That(top.Data, Is.EqualTo(new[] { 0.5f, 1f }).Within(1e-6));
    }

    [Test]
    public void GrowMask_dilates_and_erodes_with_either_footprint()
    {
        var dot = new MaskBatch(1, 5, 5);
        dot.Data[2 * 5 + 2] = 1;
        var cross = new GrowMask().Execute(dot, 1, true);
        Assert.That(cross.Data.Sum(), Is.EqualTo(5f), "tapered corners: a plus shape");
        Assert.That(cross.Data[1 * 5 + 1], Is.EqualTo(0f));
        var square = new GrowMask().Execute(dot, 1, false);
        Assert.That(square.Data.Sum(), Is.EqualTo(9f), "otherwise a 3x3 square");
        Assert.That(new GrowMask().Execute(square, -1, false).Data.Sum(), Is.EqualTo(1f), "negative expand erodes back");
        Assert.That(new GrowMask().Execute(dot, 0).Data, Is.EqualTo(dot.Data));
    }

    // ---- latents ----

    [Test]
    public void LatentFlip_and_Rotate_are_their_own_inverses_and_move_pixels_the_right_way()
    {
        var l = Lat(1, 2, 3, i => i); // [[0,1,2],[3,4,5]]
        Assert.That(new LatentFlip().Execute(l, "x-axis: vertically").Data, Is.EqualTo(new float[] { 3, 4, 5, 0, 1, 2 }));
        Assert.That(new LatentFlip().Execute(l, "y-axis: horizontally").Data, Is.EqualTo(new float[] { 2, 1, 0, 5, 4, 3 }));
        var r90 = new LatentRotate().Execute(l, "90 degrees");
        Assert.That((r90.Height, r90.Width), Is.EqualTo((3, 2)));
        Assert.That(r90.Data, Is.EqualTo(new float[] { 3, 0, 4, 1, 5, 2 }), "clockwise");
        var back = l;
        for (int i = 0; i < 4; i++) back = new LatentRotate().Execute(back, "90 degrees");
        Assert.That(back.Data, Is.EqualTo(l.Data));
        Assert.That(new LatentRotate().Execute(l, "180 degrees").Data, Is.EqualTo(new float[] { 5, 4, 3, 2, 1, 0 }));
        Assert.That(new LatentRotate().Execute(l, "none"), Is.SameAs(l));
    }

    [Test]
    public void LatentCrop_and_FromBatch_and_Repeat()
    {
        var l = Lat(1, 4, 4, i => i);
        // Pixel units: 128 px = 16 latent. ComfyUI pulls the start back to width - 8 = -4 (a 4-wide latent), and Python's [-4:12] is the whole row.
        var crop = new LatentCrop().Execute(l, 128, 128, 8 * 1, 8 * 1);
        Assert.That((crop.Width, crop.Height), Is.EqualTo((4, 4)));
        Assert.That(crop.Data[0], Is.EqualTo(0f));
        var big = Lat(1, 12, 12, i => i);
        var inner = new LatentCrop().Execute(big, 64, 64, 8 * 6, 8 * 2); // x 6 > 12 - 8: pulled back to 4; y 2 stays
        Assert.That((inner.Width, inner.Height), Is.EqualTo((8, 8)));
        Assert.That(inner.Data[0], Is.EqualTo(2 * 12 + 4f));
        var two = new RepeatLatentBatch().Execute(l, 2);
        Assert.That(two.Batch, Is.EqualTo(2));
        Assert.That(new LatentFromBatch().Execute(two, 1, 1).Data, Is.EqualTo(l.Data));
    }

    [Test]
    public void LatentUpscale_changes_the_size_in_latent_units_and_keeps_the_noise_mask_as_it_is()
    {
        var l = new SetLatentNoiseMask().Execute(Lat(2, 4, 4, i => 1f), Mask(4, 4, Enumerable.Repeat(1f, 16).ToArray()));
        var up = new LatentUpscale().Execute(l, "nearest-exact", 64, 64); // 64 px = 8 latent
        Assert.That((up.Width, up.Height, up.Channels), Is.EqualTo((8, 8, 2)));
        Assert.That(up.NoiseMask, Is.SameAs(l.NoiseMask), "ComfyUI copies the dict: the mask is not resized (the sampler fits it to the latent)");
        var by = new LatentUpscaleBy().Execute(l, "bilinear", 1.5f);
        Assert.That(by.Width, Is.EqualTo(6));
        Assert.That(new LatentUpscale().Execute(l, "nearest-exact", 0, 0), Is.SameAs(l));
    }

    [Test]
    public void LatentComposite_pastes_and_feathers_only_inner_edges()
    {
        var to = Lat(1, 4, 4, _ => 0f);
        var from = Lat(1, 2, 2, _ => 1f);
        var hard = new LatentComposite().Execute(to, from, 8, 8, 0); // latent (1,1)
        Assert.That(hard.Data[1 * 4 + 1], Is.EqualTo(1f));
        Assert.That(hard.Data.Sum(), Is.EqualTo(4f));
        var soft = new LatentComposite().Execute(to, from, 8, 8, 16);
        Assert.That(soft.Data[1 * 4 + 1], Is.LessThan(1f), "edges inside the target fade");
        var corner = new LatentComposite().Execute(to, from, 0, 0, 16);
        Assert.That(corner.Data[0], Is.EqualTo(1f), "an edge on the border of the target is not faded");
    }

    [Test]
    public void LatentBlend_mixes_and_SetLatentNoiseMask_copies()
    {
        var blend = new LatentBlend().Execute(Lat(1, 2, 2, _ => 1f), Lat(1, 2, 2, _ => 3f), 0.25f);
        Assert.That(blend.Data.All(v => Math.Abs(v - 2.5f) < 1e-6), Is.True);
        var src = Lat(1, 2, 2, _ => 7f);
        var masked = new SetLatentNoiseMask().Execute(src, Mask(2, 2, 1, 0, 1, 0));
        Assert.That(masked.NoiseMask!.Data, Is.EqualTo(new float[] { 1, 0, 1, 0 }));
        masked.Data[0] = -1;
        Assert.That(src.Data[0], Is.EqualTo(7f), "the original latent is not modified");
    }

    // ---- primitives ----

    [Test]
    public void Primitive_nodes_return_their_value()
    {
        Assert.That(new PrimitiveInt().Execute(42), Is.EqualTo(42L));
        Assert.That(new PrimitiveFloat().Execute(1.5), Is.EqualTo(1.5));
        Assert.That(new PrimitiveString().Execute("hi"), Is.EqualTo("hi"));
        Assert.That(new PrimitiveStringMultiline().Execute("a\nb"), Is.EqualTo("a\nb"));
        Assert.That(new PrimitiveBoolean().Execute(true), Is.True);
    }
}
