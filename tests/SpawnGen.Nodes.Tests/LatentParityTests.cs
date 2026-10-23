using System.Globalization;
using System.Text;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// The latent nodes against ComfyUI's code (nodes.py, comfy_extras/nodes_latent.py, nodes_mask.py, nodes_images.py, comfy/utils.py): schemas as ComfyUI declares them,
/// and results on small inputs whose expected numbers are worked out by hand from the Python and PyTorch's kernels.
/// </summary>
public class LatentParityTests
{
    static LatentBatch Lat(int b, int c, int h, int w, params float[] v) =>
        new(b, c, h, w, v.Length == b * c * h * w ? v : Enumerable.Range(0, b * c * h * w).Select(i => v[i % v.Length]).ToArray());
    static LatentBatch Ramp(int b, int c, int h, int w) => new(b, c, h, w, Enumerable.Range(0, b * c * h * w).Select(i => (float)i).ToArray());

    // ---- schemas ----

    static string Signature(NodeSchema s)
    {
        static string N(double? v) => v is null ? "" : v.Value.ToString(CultureInfo.InvariantCulture);
        var sb = new StringBuilder();
        foreach (var i in s.Inputs)
        {
            if (sb.Length > 0) sb.Append(' ');
            var w = i.Widget;
            if (w is null) { sb.Append($"{i.Name}:{i.Type.Name}{(i.Required ? "" : "?")}"); continue; }
            sb.Append(w.Kind switch
            {
                WidgetKind.Int or WidgetKind.Float => $"{i.Name}:{(w.Kind == WidgetKind.Int ? "INT" : "FLOAT")}={N(Convert.ToDouble(w.Default, CultureInfo.InvariantCulture))}({N(w.Min)}..{N(w.Max)}/{N(w.Step)})",
                WidgetKind.Combo => $"{i.Name}:COMBO={w.Default}[{string.Join("|", w.ResolveOptions())}]",
                WidgetKind.Boolean => $"{i.Name}:BOOLEAN={w.Default}",
                _ => $"{i.Name}:STRING={w.Default}",
            });
        }
        sb.Append(" -> ").Append(string.Join(" ", s.Outputs.Select(o => $"{o.Name}:{o.Type.Name}")));
        return sb.ToString();
    }

    static readonly string Methods = "[nearest-exact|bilinear|area|bicubic|bislerp]";

    static readonly (string Id, string Category, string Display, string Signature)[] Expected =
    [
        ("EmptyLatentImage", "model/latent", "Empty Latent Image", "width:INT=1024(16..16384/8) height:INT=1024(16..16384/8) batch_size:INT=1(1..4096/1) -> LATENT:LATENT"),
        ("LatentUpscale", "model/latent", "Upscale Latent", $"samples:LATENT upscale_method:COMBO=nearest-exact{Methods} width:INT=512(0..16384/8) height:INT=512(0..16384/8) crop:COMBO=disabled[disabled|center] -> LATENT:LATENT"),
        ("LatentUpscaleBy", "model/latent", "Upscale Latent By", $"samples:LATENT upscale_method:COMBO=nearest-exact{Methods} scale_by:FLOAT=1.5(0.01..8/0.01) -> LATENT:LATENT"),
        ("LatentFromBatch", "model/latent/batch", "Get Latent From Batch", "samples:LATENT batch_index:INT=0(-16384..16384/1) length:INT=1(1..64/1) -> LATENT:LATENT"),
        ("RepeatLatentBatch", "model/latent/batch", "Repeat Latent Batch", "samples:LATENT amount:INT=1(1..64/1) -> LATENT:LATENT"),
        ("LatentComposite", "model/latent", "Latent Composite", "samples_to:LATENT samples_from:LATENT x:INT=0(0..16384/8) y:INT=0(0..16384/8) feather:INT=0(0..16384/8) -> LATENT:LATENT"),
        ("LatentFlip", "model/latent/transform", "Flip Latent", "samples:LATENT flip_method:COMBO=x-axis: vertically[x-axis: vertically|y-axis: horizontally] -> LATENT:LATENT"),
        ("LatentRotate", "model/latent/transform", "Rotate Latent", "samples:LATENT rotation:COMBO=none[none|90 degrees|180 degrees|270 degrees] -> LATENT:LATENT"),
        ("LatentCrop", "model/latent/transform", "Crop Latent", "samples:LATENT width:INT=512(64..16384/8) height:INT=512(64..16384/8) x:INT=0(0..16384/8) y:INT=0(0..16384/8) -> LATENT:LATENT"),
        ("LatentBlend", "experimental", "Latent Blend", "samples1:LATENT samples2:LATENT blend_factor:FLOAT=0.5(0..1/0.01) -> LATENT:LATENT"),
        ("SetLatentNoiseMask", "model/latent", "Set Latent Noise Mask", "samples:LATENT mask:MASK -> LATENT:LATENT"),
        ("LatentCompositeMasked", "model/latent", "Latent Composite Masked", "destination:LATENT source:LATENT x:INT=0(0..16384/8) y:INT=0(0..16384/8) resize_source:BOOLEAN=False mask:MASK? -> LATENT:LATENT"),
        ("LatentAdd", "model/latent/advanced", "LatentAdd", "samples1:LATENT samples2:LATENT -> LATENT:LATENT"),
        ("LatentSubtract", "model/latent/advanced", "LatentSubtract", "samples1:LATENT samples2:LATENT -> LATENT:LATENT"),
        ("LatentMultiply", "model/latent/advanced", "LatentMultiply", "samples:LATENT multiplier:FLOAT=1(-10..10/0.01) -> LATENT:LATENT"),
        ("LatentInterpolate", "model/latent/advanced", "LatentInterpolate", "samples1:LATENT samples2:LATENT ratio:FLOAT=1(0..1/0.01) -> LATENT:LATENT"),
        ("LatentConcat", "model/latent/advanced", "LatentConcat", "samples1:LATENT samples2:LATENT dim:COMBO=x[x|-x|y|-y|t|-t] -> LATENT:LATENT"),
        ("LatentCut", "model/latent/advanced", "LatentCut", "samples:LATENT dim:COMBO=x[x|y|t] index:INT=0(-16384..16384/1) amount:INT=1(1..16384/1) -> LATENT:LATENT"),
        ("LatentCutToBatch", "model/latent/advanced", "LatentCutToBatch", "samples:LATENT dim:COMBO=t[t|x|y] slice_size:INT=1(1..16384/1) -> LATENT:LATENT"),
        ("LatentBatch", "model/latent/batch", "Batch Latents (DEPRECATED)", "samples1:LATENT samples2:LATENT -> LATENT:LATENT"),
        ("LatentBatchSeedBehavior", "model/latent/advanced", "LatentBatchSeedBehavior", "samples:LATENT seed_behavior:COMBO=fixed[random|fixed] -> LATENT:LATENT"),
        ("LatentApplyOperation", "model/latent/advanced/operations", "LatentApplyOperation", "samples:LATENT operation:LATENT_OPERATION -> LATENT:LATENT"),
        ("LatentOperationBlend", "model/latent/advanced/operations", "Latent Operation Blend", "reference:LATENT strength:FLOAT=1(0..1/0.0001) -> LATENT_OPERATION:LATENT_OPERATION"),
        ("LatentOperationTonemapReinhard", "model/latent/advanced/operations", "LatentOperationTonemapReinhard", "multiplier:FLOAT=1(0..100/0.01) -> LATENT_OPERATION:LATENT_OPERATION"),
        ("LatentOperationSharpen", "model/latent/advanced/operations", "LatentOperationSharpen", "sharpen_radius:INT=9(1..31/1) sigma:FLOAT=1(0.1..10/0.1) alpha:FLOAT=0.1(0..5/0.01) -> LATENT_OPERATION:LATENT_OPERATION"),
        ("GetLatentSize", "latent", "Get Image/Video Latent Size", "latent:LATENT -> width:INT height:INT frames:INT batch_size:INT"),
        ("SaveLatent", "model/latent", "Save Latent", "samples:LATENT filename_prefix:STRING=latents/ComfyUI -> samples:LATENT"),
        ("LatentApplyOperationCFG", "model/latent/advanced/operations", "LatentApplyOperationCFG", "model:MODEL operation:LATENT_OPERATION start_percent:FLOAT=0(0..1/0.001) end_percent:FLOAT=1(0..1/0.001) -> MODEL:MODEL"),
    ];

    [Test]
    public void Every_latent_node_has_ComfyUIs_id_category_name_inputs_and_outputs()
    {
        var registry = new NodeRegistry().AddAssembly(typeof(LatentUpscale).Assembly).AddAssembly(typeof(SpawnGen.Nodes.ML.KSampler).Assembly);
        foreach (var (id, category, display, signature) in Expected)
        {
            var s = registry.GetSchema(id);
            Assert.That(s, Is.Not.Null, id);
            Assert.That((s!.Category, s.DisplayName), Is.EqualTo((category, display)), id);
            Assert.That(Signature(s), Is.EqualTo(signature), id);
        }
        Assert.That(registry.GetSchema("LoadLatent")!.Category, Is.EqualTo("model/latent"));
        Assert.That(registry.GetSchema("SaveLatent")!.IsOutputNode, Is.True);
        Assert.That(registry.CanonicalId("LatentBatchNode"), Is.EqualTo("LatentBatch"), "the earlier SpawnGen id still opens");
    }

    // ---- PyTorch interpolate and ComfyUI's common_upscale ----

    [Test]
    public void Interpolate_matches_PyTorchs_kernels_on_small_rows()
    {
        // F.interpolate([[0, 1]], size=4): bilinear [0, .25, .75, 1]; bicubic (a = -0.75, edge taps repeat the border sample) overshoots.
        Assert.That(TorchInterpolate.Resize([0f, 1f], 1, 1, 2, 1, 4, "bilinear"), Is.EqualTo(new[] { 0f, 0.25f, 0.75f, 1f }));
        Assert.That(TorchInterpolate.Resize([0f, 1f], 1, 1, 2, 1, 4, "bicubic"), Is.EqualTo(new[] { -0.10546875f, 0.2265625f, 0.7734375f, 1.10546875f }).Within(1e-6));
        // area is adaptive_avg_pool: cells floor(o*3/2) .. ceil((o+1)*3/2), plain mean: [1,2] and [2,3] (not a weighted box).
        Assert.That(TorchInterpolate.Resize([1f, 2f, 3f], 1, 1, 3, 1, 2, "area"), Is.EqualTo(new[] { 1.5f, 2.5f }));
        // nearest-exact: floor((o + .5) * 1.5) = 0, 2; nearest: floor(o * 1.5) = 0, 1.
        Assert.That(TorchInterpolate.Resize([0f, 1f, 2f], 1, 1, 3, 1, 2, "nearest-exact"), Is.EqualTo(new[] { 0f, 2f }));
        Assert.That(TorchInterpolate.Resize([0f, 1f, 2f], 1, 1, 3, 1, 2, "nearest"), Is.EqualTo(new[] { 0f, 1f }));
        // Same size is the identity for every mode.
        foreach (var mode in new[] { "bilinear", "bicubic", "area", "nearest-exact" })
            Assert.That(TorchInterpolate.Resize([3f, -1f, 2f, 5f], 1, 2, 2, 2, 2, mode), Is.EqualTo(new[] { 3f, -1f, 2f, 5f }), mode);
    }

    [Test]
    public void Bislerp_spherically_interpolates_the_channel_vector()
    {
        // Two channels, a row of two positions: (1, 0) and (0, 2), widened to 3. The middle sample is half way round with the mean length 1.5.
        var r = TorchInterpolate.Bislerp([1f, 0f, 0f, 2f], 1, 2, 1, 2, 1, 3);
        float mid = MathF.Sqrt(0.5f) * 1.5f;
        Assert.That(r, Is.EqualTo(new[] { 1f, mid, 0f, 0f, mid, 2f }).Within(1e-6));
    }

    [Test]
    public void Common_upscale_center_crops_to_the_new_aspect_before_resizing()
    {
        // 4x2 to 2x2: old aspect 2, new 1, so x = round((4 - 4 * 0.5) / 2) = 1 columns come off each side, and the 2x2 middle is used as it is.
        var (data, h, w) = TorchInterpolate.CommonUpscale([0, 1, 2, 3, 4, 5, 6, 7], 1, 1, 2, 4, 2, 2, "bilinear", "center");
        Assert.That((w, h), Is.EqualTo((2, 2)));
        Assert.That(data, Is.EqualTo(new float[] { 1, 2, 5, 6 }));
    }

    [Test]
    public void Upscale_methods_side_by_side_picture()
    {
        // A 3-channel 8x8 latent (a disc on a ramp) upscaled x16 with every method, channels shown as RGB, written for a look: the shapes must be the same,
        // nearest blocky, bilinear/bicubic smooth (bicubic slightly ringing), area like nearest when enlarging, bislerp smooth and as bright as the source.
        const int n = 8, scale = 16, size = n * scale;
        var src = new LatentBatch(1, 3, n, n);
        for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
            {
                bool disc = (x - 3.5) * (x - 3.5) + (y - 3.5) * (y - 3.5) < 6;
                src.Data[y * n + x] = disc ? 0.9f : x / 8f;
                src.Data[n * n + y * n + x] = disc ? 0.2f : 0.5f;
                src.Data[2 * n * n + y * n + x] = disc ? 0.1f : 1 - x / 8f;
            }
        var methods = TorchInterpolate.LatentMethods;
        var img = new ImageBatch(1, size, size * methods.Length, 3);
        for (int m = 0; m < methods.Length; m++)
        {
            var up = new LatentUpscale().Execute(src, methods[m], size * 8, size * 8);
            Assert.That((up.Width, up.Height), Is.EqualTo((size, size)));
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                    for (int c = 0; c < 3; c++)
                        img.Data[(y * size * methods.Length + m * size + x) * 3 + c] = Math.Clamp(up.Data[(c * size + y) * size + x], 0f, 1f);
        }
        var path = Path.Combine(TestContext.CurrentContext.WorkDirectory, "latent-upscale-methods.png");
        File.WriteAllBytes(path, PngWriter.Encode(img));
        TestContext.Out.WriteLine(path);
    }

    // ---- nodes.py latents ----

    [Test]
    public void LatentUpscale_follows_ComfyUIs_size_rules()
    {
        var l = Ramp(1, 1, 2, 3); // 3 wide, 2 high (24 x 16 px)
        // A zero side follows the latent's aspect: width = max(64, round(3 * 100 / 2)) = 150 px, so 18 x 12 cells.
        var a = new LatentUpscale().Execute(l, "bilinear", 0, 100);
        Assert.That((a.Width, a.Height), Is.EqualTo((18, 12)));
        // Both sides are at least 64 px: 8 -> 64.
        var b = new LatentUpscale().Execute(l, "nearest-exact", 8, 8);
        Assert.That((b.Width, b.Height), Is.EqualTo((8, 8)));
        Assert.That(new LatentUpscale().Execute(l, "bilinear", 0, 0), Is.SameAs(l));
        // Python rounds half to even: 3 * 1.5 = 4.5 -> 4.
        var by = new LatentUpscaleBy().Execute(Ramp(1, 1, 3, 3), "nearest-exact", 1.5);
        Assert.That((by.Width, by.Height), Is.EqualTo((4, 4)));
        // Upscale By has no 64 px minimum (Upscale Latent would turn 2 cells into 8).
        var small = new LatentUpscaleBy().Execute(Ramp(1, 1, 2, 2), "bilinear", 2);
        Assert.That((small.Width, small.Height), Is.EqualTo((4, 4)));
        Assert.That(small.Data.Take(4), Is.EqualTo(new[] { 0f, 0.25f, 0.75f, 1f }));
        Assert.That(new LatentUpscale().Execute(l, "bislerp", 64, 64).Width, Is.EqualTo(8));
    }

    [Test]
    public void LatentComposite_feathers_like_ComfyUIs_loop()
    {
        var to = Lat(1, 1, 4, 4, 0f);
        var from = Lat(1, 1, 2, 2, 1f);
        // Inside the target on all four sides with feather 2 cells: every cell gets 0.5 from its row and 0.5 from its column.
        var inner = new LatentComposite().Execute(to, from, 8, 8, 16);
        Assert.That(new[] { inner.Data[5], inner.Data[6], inner.Data[9], inner.Data[10] }, Is.All.EqualTo(0.25f));
        // At the corner only the inner edges fade: rows and columns 1 get 0.5.
        var corner = new LatentComposite().Execute(to, from, 0, 0, 16);
        Assert.That(new[] { corner.Data[0], corner.Data[1], corner.Data[4], corner.Data[5] }, Is.EqualTo(new[] { 1f, 0.5f, 0.5f, 0.25f }));
        // Feather 4 cells over a 2-cell paste: the loop's factors .25, .5 hit both rows from both ends, .75 falls outside, and 1 (t = 3) wraps to row 0 by
        // Python's negative slice, so every row ends at .25 * .5 and every column too.
        var wide = new LatentComposite().Execute(to, from, 8, 8, 32);
        Assert.That(wide.Data[5], Is.EqualTo(0.125f * 0.125f).Within(1e-7));
        // A from latent that sticks out is cut to what fits.
        var edge = new LatentComposite().Execute(to, Lat(1, 1, 3, 3, 2f), 16, 16, 0);
        Assert.That(edge.Data.Sum(), Is.EqualTo(2f * 4));
    }

    [Test]
    public void LatentCrop_pulls_the_start_back_to_leave_eight_cells()
    {
        var l = Ramp(1, 1, 16, 16);
        var c = new LatentCrop().Execute(l, 64, 64, 8 * 12, 0); // x 12 > 16 - 8: becomes 8
        Assert.That((c.Width, c.Height), Is.EqualTo((8, 8)));
        Assert.That(c.Data[0], Is.EqualTo(8f));
        var keep = new LatentCrop().Execute(new SetLatentNoiseMask().Execute(l, MaskBatch.Filled(1, 2, 2, 1f)), 64, 64, 0, 0);
        Assert.That(keep.NoiseMask, Is.Not.Null, "the dict is copied: the noise mask stays");
    }

    [Test]
    public void LatentFromBatch_counts_from_the_end_repeats_a_short_mask_and_records_the_batch_index()
    {
        var l = Ramp(4, 1, 1, 1);
        var masks = new MaskBatch(2, 1, 1, [0.25f, 0.75f]);
        var last = new LatentFromBatch().Execute(new SetLatentNoiseMask().Execute(l, masks), -1, 2);
        Assert.That(last.Data, Is.EqualTo(new[] { 3f }), "-1 is the last picture, and the length is cut to what is left");
        Assert.That(last.BatchIndex, Is.EqualTo(new[] { 3 }));
        Assert.That(last.NoiseMask!.Data, Is.EqualTo(new[] { 0.75f }), "masks [a, b] repeat to [a, b, a, b]: picture 3 has b");
        var two = new LatentFromBatch().Execute(last, 0, 5);
        Assert.That(two.BatchIndex, Is.EqualTo(new[] { 3 }), "an existing batch index is sliced, not renumbered");
        var single = new SetLatentNoiseMask().Execute(l, MaskBatch.Filled(1, 1, 1, 0.5f));
        Assert.That(new LatentFromBatch().Execute(single, 2, 1).NoiseMask, Is.SameAs(single.NoiseMask), "a one-picture mask is kept whole");
    }

    [Test]
    public void RepeatLatentBatch_continues_the_batch_index_and_keeps_a_single_mask()
    {
        var picked = new LatentFromBatch().Execute(Ramp(4, 1, 1, 1), 2, 2); // batch index [2, 3]
        var repeated = new RepeatLatentBatch().Execute(picked, 3);
        Assert.That(repeated.Data, Is.EqualTo(new[] { 2f, 3f, 2f, 3f, 2f, 3f }));
        Assert.That(repeated.BatchIndex, Is.EqualTo(new[] { 2, 3, 4, 5, 6, 7 }), "offset = max - min + 1 = 2 per copy");
        var masked = new SetLatentNoiseMask().Execute(Ramp(2, 1, 1, 1), MaskBatch.Filled(1, 1, 1, 1f));
        Assert.That(new RepeatLatentBatch().Execute(masked, 2).NoiseMask!.Batch, Is.EqualTo(1));
        var masked2 = new SetLatentNoiseMask().Execute(Ramp(2, 1, 1, 1), new MaskBatch(2, 1, 1, [0f, 1f]));
        Assert.That(new RepeatLatentBatch().Execute(masked2, 2).NoiseMask!.Data, Is.EqualTo(new[] { 0f, 1f, 0f, 1f }));
        Assert.That(new RepeatLatentBatch().Execute(Ramp(1, 1, 1, 1), 2).BatchIndex, Is.Null, "no batch index, none made");
    }

    [Test]
    public void LatentRotate_and_Flip_keep_the_noise_mask_and_batch_index()
    {
        var l = new LatentFromBatch().Execute(new SetLatentNoiseMask().Execute(Ramp(2, 1, 2, 3), MaskBatch.Filled(1, 2, 3, 1f)), 1, 1);
        var r = new LatentRotate().Execute(l, "90 degrees");
        Assert.That((r.NoiseMask, r.BatchIndex), Is.EqualTo((l.NoiseMask, l.BatchIndex)));
        var f = new LatentFlip().Execute(l, "y-axis: horizontally");
        Assert.That((f.NoiseMask, f.BatchIndex), Is.EqualTo((l.NoiseMask, l.BatchIndex)));
    }

    [Test]
    public void LatentBlend_broadcasts_a_single_first_latent_over_the_second_batch()
    {
        // samples1 * .25 + samples2 * .75; the shapes differ (batch), so samples2 goes through a same-size bicubic resize, which leaves it alone.
        var r = new LatentBlend().Execute(Lat(1, 1, 1, 1, 1f), Lat(2, 1, 1, 1, 3f, 5f), 0.25);
        Assert.That(r.Batch, Is.EqualTo(2));
        Assert.That(r.Data, Is.EqualTo(new[] { 2.5f, 4f }));
        // Another size: bicubic with centre crop to the first's size.
        var wide = new LatentBlend().Execute(Lat(1, 1, 1, 4, 0f), Lat(1, 1, 1, 2, 0f, 1f), 0);
        Assert.That(wide.Data, Is.EqualTo(new[] { -0.10546875f, 0.2265625f, 0.7734375f, 1.10546875f }).Within(1e-6));
    }

    // ---- nodes_latent.py ----

    [Test]
    public void LatentAdd_resizes_the_second_bilinearly_and_repeats_its_batch()
    {
        var sum = new LatentAdd().Execute(Lat(1, 1, 1, 4, 0f), Lat(1, 1, 1, 2, 0f, 1f));
        Assert.That(sum.Data, Is.EqualTo(new[] { 0f, 0.25f, 0.75f, 1f }), "reshape_latent_to is bilinear, not bicubic");
        var cut = new LatentSubtract().Execute(Lat(1, 1, 1, 1, 10f), Lat(3, 1, 1, 1, 1f, 2f, 3f));
        Assert.That(cut.Data, Is.EqualTo(new[] { 9f }), "a longer second batch is cut to the first's");
        Assert.That(new LatentMultiply().Execute(Lat(1, 1, 1, 2, 2f, -4f), 0.5).Data, Is.EqualTo(new[] { 1f, -2f }));
    }

    [Test]
    public void LatentInterpolate_reproduces_ComfyUIs_broadcast_for_batches()
    {
        // Batch 2, 2 channels, one position. ComfyUI's norms have shape (batch, h, w) and broadcast against the channels, so channel c uses picture c's norm.
        var s = Lat(2, 2, 1, 1, 3f, 4f, 0f, 1f); // picture 0 = (3, 4), length 5; picture 1 = (0, 1), length 1
        var r = new LatentInterpolate().Execute(s, s, 1.0);
        // t = (3/5, 4/1) and (0/5, 1/1); |t0| = sqrt(.36 + 16); out[n, c] = t[n, c] / |t_c| * m1[c].
        double t0 = Math.Sqrt(0.36 + 16);
        Assert.That(r.Data, Is.EqualTo(new[] { (float)(0.6 / t0 * 5), 4f, 0f, 1f }).Within(1e-5));
        Assert.That(() => new LatentInterpolate().Execute(Lat(3, 2, 1, 1, 1f), Lat(3, 2, 1, 1, 1f), 0.5), Throws.ArgumentException, "PyTorch cannot broadcast 3 against 2");
        // A batch of one is the plain per-position interpolation.
        var one = new LatentInterpolate().Execute(Lat(1, 2, 1, 1, 3f, 4f), Lat(1, 2, 1, 1, 0f, 10f), 0.5).Data;
        Assert.That(Math.Sqrt(one[0] * one[0] + one[1] * one[1]), Is.EqualTo(7.5).Within(1e-5));
    }

    [Test]
    public void LatentConcat_joins_along_x_y_and_t()
    {
        var a = Lat(2, 1, 1, 2, 1f, 2f, 5f, 6f);
        var b = Lat(1, 1, 1, 1, 9f);
        Assert.That(new LatentConcat().Execute(a, b, "x").Data, Is.EqualTo(new[] { 1f, 2f, 9f, 5f, 6f, 9f }), "the second is repeated to the first's batch");
        Assert.That(new LatentConcat().Execute(a, b, "-x").Data, Is.EqualTo(new[] { 9f, 1f, 2f, 9f, 5f, 6f }));
        var y = new LatentConcat().Execute(Lat(1, 1, 1, 2, 1f, 2f), Lat(1, 1, 1, 2, 3f, 4f), "y");
        Assert.That((y.Height, y.Data), Is.EqualTo((2, new[] { 1f, 2f, 3f, 4f })));
        // t is the third axis from the end: the channels of a picture latent.
        var t = new LatentConcat().Execute(Lat(1, 1, 1, 2, 1f, 2f), Lat(1, 1, 1, 2, 3f, 4f), "-t");
        Assert.That((t.Channels, t.Data), Is.EqualTo((2, new[] { 3f, 4f, 1f, 2f })));
        Assert.That(() => new LatentConcat().Execute(Lat(1, 1, 1, 2, 1f), Lat(1, 1, 2, 2, 1f), "x"), Throws.ArgumentException);
    }

    [Test]
    public void LatentCut_clamps_and_counts_negative_indexes_from_the_end()
    {
        var l = Ramp(1, 1, 1, 5);
        Assert.That(new LatentCut().Execute(l, "x", 1, 2).Data, Is.EqualTo(new[] { 1f, 2f }));
        Assert.That(new LatentCut().Execute(l, "x", -2, 5).Data, Is.EqualTo(new[] { 3f, 4f }), "amount = min(-index, amount)");
        Assert.That(new LatentCut().Execute(l, "x", 10, 3).Data, Is.EqualTo(new[] { 4f }), "index = min(index, size - 1)");
        Assert.That(new LatentCut().Execute(l, "x", -10, 1).Data, Is.EqualTo(new[] { 0f }), "index = max(index, -size)");
        var y = new LatentCut().Execute(Ramp(1, 1, 3, 2), "y", 1, 1);
        Assert.That((y.Height, y.Data), Is.EqualTo((1, new[] { 2f, 3f })));
        var t = new LatentCut().Execute(Ramp(1, 4, 1, 1), "t", 1, 2);
        Assert.That((t.Channels, t.Data), Is.EqualTo((2, new[] { 1f, 2f })));
    }

    [Test]
    public void LatentCutToBatch_slices_into_pictures_and_drops_the_remainder()
    {
        var l = Ramp(2, 1, 2, 5); // two pictures of [[0..4], [5..9]] (+10 for the second)
        var x = new LatentCutToBatch().Execute(l, "x", 2);
        Assert.That((x.Batch, x.Height, x.Width), Is.EqualTo((4, 2, 2)));
        Assert.That(x.Data, Is.EqualTo(new float[] { 0, 1, 5, 6, 2, 3, 7, 8, 10, 11, 15, 16, 12, 13, 17, 18 }));
        var y = new LatentCutToBatch().Execute(l, "y", 1);
        Assert.That((y.Batch, y.Height, y.Width), Is.EqualTo((4, 1, 5)));
        Assert.That(y.Data.Skip(5).Take(5), Is.EqualTo(new float[] { 5, 6, 7, 8, 9 }));
        Assert.That(new LatentCutToBatch().Execute(l, "x", 9).Batch, Is.EqualTo(2), "a slice longer than the axis is the whole axis");
        Assert.That(new LatentCutToBatch().Execute(l, "t", 1), Is.SameAs(l), "t is axis 1 of a picture latent, which ComfyUI passes through");
    }

    [Test]
    public void LatentBatch_joins_batch_indexes_and_keeps_the_first_mask()
    {
        var mask = MaskBatch.Filled(1, 1, 1, 1f);
        var a = new SetLatentNoiseMask().Execute(Ramp(1, 1, 1, 1), mask);
        var r = new LatentBatchNode().Execute(a, Ramp(2, 1, 1, 1));
        Assert.That(r.BatchIndex, Is.EqualTo(new[] { 0, 0, 1 }), "[0] + [0, 1]: the second latent's pictures keep their own noise");
        Assert.That(r.NoiseMask, Is.SameAs(mask));
    }

    [Test]
    public async Task GetLatentSize_reports_latent_cells_and_one_frame()
    {
        var registry = new NodeRegistry().AddAssembly(typeof(GetLatentSize).Assembly);
        var g = new PromptGraph();
        g.Add("1", "EmptyLatentImage", ("width", 64), ("height", 32), ("batch_size", 3));
        g.Add("2", "GetLatentSize", ("latent", PromptInput.Link("1")));
        var result = await new PromptExecutor(registry, new Services()).ExecuteAsync(g, new ExecutionOptions { OutputNodeIds = ["2"] });
        Assert.That(result.Outputs["2"].Values, Is.EqualTo(new object[] { 8, 4, 1, 3 }));
    }

    [Test]
    public async Task SaveLatent_passes_the_latent_through_and_stores_the_prompt()
    {
        var dir = Path.Combine(Path.GetTempPath(), "spawngen-latentparity-" + Guid.NewGuid().ToString("N"));
        try
        {
            var registry = new NodeRegistry().AddAssembly(typeof(SaveLatent).Assembly);
            var options = new OutputOptions { Directory = Path.Combine(dir, "output"), TempDirectory = Path.Combine(dir, "temp") };
            var g = new PromptGraph();
            g.Add("1", "EmptyLatentImage", ("width", 64), ("height", 64), ("batch_size", 1));
            g.Add("2", "SaveLatent", ("samples", PromptInput.Link("1")), ("filename_prefix", "x"));
            var result = await new PromptExecutor(registry, new Services(options)).ExecuteAsync(g);
            Assert.That(result.Outputs["2"].Values[0], Is.InstanceOf<LatentBatch>());
            var bytes = await File.ReadAllBytesAsync(Path.Combine(dir, "output", "x_00001_.latent"));
            Safetensors.Read(bytes, out var metadata);
            Assert.That(metadata["prompt"], Does.Contain("EmptyLatentImage"));
        }
        finally { if (Directory.Exists(dir)) Directory.Delete(dir, true); }
    }

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }
}
