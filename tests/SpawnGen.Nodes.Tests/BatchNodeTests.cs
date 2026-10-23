using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

public class BatchNodeTests
{
    static ImageBatch Solid(int w, int h, int c, float v) => new(1, h, w, c, Enumerable.Repeat(v, w * h * c).ToArray());

    [Test]
    public void Common_upscale_crops_to_the_target_aspect_before_resizing_like_ComfyUI()
    {
        // 4x2 -> 2x2: old aspect 2 > 1, x = round((4 - 4 * 0.5) / 2) = 1, so columns 1 and 2 are kept and nothing is resized.
        float[] row = [10, 20, 30, 40];
        var src = row.Concat(row).ToArray();
        Assert.That(Resample.CommonUpscale(src, 4, 2, 1, 2, 2, ResampleMethod.Bilinear, true), Is.EqualTo(new float[] { 20, 30, 20, 30 }));
        // 5x2 -> 2x2: x = round((5 - 2) / 2) = round(1.5) = 2 (Python rounds half to even), so only column 2 is kept, then stretched.
        float[] five = [1, 2, 3, 4, 5];
        Assert.That(Resample.CommonUpscale(five.Concat(five).ToArray(), 5, 2, 1, 2, 2, ResampleMethod.Bilinear, true), Is.EqualTo(new float[] { 3, 3, 3, 3 }));
    }

    [Test]
    public void Batch_images_pads_alpha_and_scales_to_the_first_picture()
    {
        var rgba = Solid(4, 4, 4, 0.25f);
        var rgb = Solid(8, 8, 3, 0.5f);
        var r = new BatchImagesNode().Execute([rgba, rgb]);
        Assert.That((r.Batch, r.Height, r.Width, r.Channels), Is.EqualTo((2, 4, 4, 4)));
        var second = r.Data[(4 * 4 * 4)..];
        Assert.That(second[..4], Is.EqualTo(new[] { 0.5f, 0.5f, 0.5f, 1f }), "RGB gets full alpha, then is scaled down");
        Assert.That(new ImageBatchNode().Execute(rgb, rgba).Channels, Is.EqualTo(4), "the two-input Image Batch follows the same rule");
    }

    [Test]
    public void Batch_masks_and_latents_scale_to_the_first()
    {
        var m = new BatchMasksNode().Execute([new MaskBatch(1, 2, 2, [1, 1, 1, 1]), new MaskBatch(2, 4, 4, Enumerable.Repeat(0.5f, 32).ToArray())]);
        Assert.That((m.Batch, m.Height, m.Width), Is.EqualTo((3, 2, 2)));
        Assert.That(m.Data[4..], Is.All.EqualTo(0.5f));

        var l = new BatchLatentsNode().Execute([new LatentBatch(1, 4, 8, 8), new LatentBatch(2, 4, 16, 16)]);
        Assert.That((l.Batch, l.Channels, l.Height, l.Width), Is.EqualTo((3, 4, 8, 8)));
    }

    [Test]
    public async Task Create_list_makes_one_list_of_its_inputs_and_get_item_reads_it()
    {
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly);
        var g = new PromptGraph();
        g.Add("1", "PrimitiveString", ("value", "a"));
        g.Add("2", "PrimitiveString", ("value", "b"));
        g.Add("3", "CreateList", ("inputs.input0", PromptInput.Link("1")), ("inputs.input1", PromptInput.Link("2")), ("inputs.input2", PromptInput.Link("1")));
        g.Add("4", "PreviewAny", ("source", PromptInput.Link("3")));
        var r = await new PromptExecutor(registry).ExecuteAsync(g);
        Assert.That((IEnumerable<object?>)r.Outputs["4"].Ui!["text"]!, Is.EqualTo(new[] { "a", "b", "a" }), "Preview Any ran once per item");

        var s = registry.GetSchema("CreateList")!;
        Assert.That((s.InputIsList, s.Outputs.Single().IsList, s.Outputs.Single().Name, s.Inputs[0].Name), Is.EqualTo((true, true, "list", "inputs.input0")));
    }
}
