using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

public class ResizeNodeTests
{
    static ImageBatch Img(int w, int h) => new(1, h, w, 3, Enumerable.Range(0, w * h * 3).Select(i => (i % 7) / 7f).ToArray());

    static (int W, int H) Run(object input, string type, params (string Name, object Value)[] values)
    {
        var r = new ResizeImageMaskNode().Execute(input, new DynamicChoice(type, values.ToDictionary(v => v.Name, v => (object?)v.Value)), "bilinear");
        return r switch { ImageBatch i => (i.Width, i.Height), MaskBatch m => (m.Width, m.Height), _ => (0, 0) };
    }

    [Test]
    public void Every_resize_type_gives_ComfyUIs_size()
    {
        var img = Img(300, 200);
        Assert.That(Run(img, "scale dimensions", ("width", 64L), ("height", 0L), ("crop", "center")), Is.EqualTo((64, 43)), "round(200 * 64 / 300) = round(42.67)");
        Assert.That(Run(img, "scale dimensions", ("width", 0L), ("height", 0L)), Is.EqualTo((300, 200)), "both 0: unchanged");
        Assert.That(Run(img, "scale by multiplier", ("multiplier", 0.25)), Is.EqualTo((75, 50)));
        Assert.That(Run(img, "scale longer dimension", ("longer_size", 90L)), Is.EqualTo((90, 60)));
        Assert.That(Run(img, "scale shorter dimension", ("shorter_size", 50L)), Is.EqualTo((75, 50)));
        Assert.That(Run(img, "scale width", ("width", 150L)), Is.EqualTo((150, 100)));
        Assert.That(Run(img, "scale height", ("height", 100L)), Is.EqualTo((150, 100)));
        // int(0.01 * 1048576) = 10485; sqrt(10485 / 60000) = 0.41803; round(125.41) = 125, round(83.61) = 84.
        Assert.That(Run(img, "scale total pixels", ("megapixels", 0.01)), Is.EqualTo((125, 84)));
        Assert.That(Run(img, "match size", ("match", new MaskBatch(1, 32, 48)), ("crop", "center")), Is.EqualTo((48, 32)));
        // 300x200 to a multiple of 64: target 256x192; scales 0.853 / 0.96 -> height wins: 192 high, ceil(300 * 0.96) = 288 wide, then the middle 256 is cut.
        Assert.That(Run(img, "scale to multiple", ("multiple", 64L)), Is.EqualTo((256, 192)));
    }

    [Test]
    public void A_mask_comes_out_as_a_mask()
    {
        var mask = new MaskBatch(2, 4, 4, Enumerable.Repeat(0.5f, 32).ToArray());
        var r = new ResizeImageMaskNode().Execute(mask, new DynamicChoice("scale by multiplier", new Dictionary<string, object?> { ["multiplier"] = 2.0 }), "nearest-exact");
        Assert.That(r, Is.InstanceOf<MaskBatch>());
        var m = (MaskBatch)r;
        Assert.That((m.Batch, m.Width, m.Height), Is.EqualTo((2, 8, 8)));
        Assert.That(m.Data, Is.All.EqualTo(0.5f));
    }

    [Test]
    public async Task It_runs_from_a_prompt_with_ComfyUIs_input_names()
    {
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly);
        var g = new PromptGraph();
        g.Add("1", "EmptyImage", ("width", 300L), ("height", 200L), ("batch_size", 1L), ("color", 0L));
        g.Add("2", "ResizeImageMaskNode", ("input", PromptInput.Link("1")), ("resize_type", "scale width"), ("resize_type.width", 120L), ("scale_method", "area"));
        g.Add("3", "GetImageSize", ("image", PromptInput.Link("2")));
        g.Add("4", "PreviewAny", ("source", PromptInput.Link("3")));
        var r = await new PromptExecutor(registry).ExecuteAsync(g);
        Assert.That(((IEnumerable<string>)r.Outputs["4"].Ui!["text"]!).Single(), Is.EqualTo("120"));
    }
}
