using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Painter against what ComfyUI's own node saved for the same painting (Fixtures/painter, made by tools/studio-browser-test/
/// probe-comfy-painter-run.js on 2026-11-25: the painting ComfyUI's widget uploaded and Save Image of the node's IMAGE and of its MASK
/// through Convert Mask to Image). "white" is one hard white stroke; "mixed" adds a soft red stroke at 50 % and an erased gap.
/// Save Image writes floor(value x 255), so the comparison is on those bytes.
/// </summary>
public class PainterTests
{
    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "painter");

    static (byte[] Rgb, int W, int H) Png(string name)
    {
        var image = ImageDecoder.Decode(File.ReadAllBytes(Path.Combine(Dir, name)));
        var px = new byte[image.Width * image.Height * 3];
        for (int i = 0; i < px.Length; i++) px[i] = (byte)MathF.Round(image.Data[i / 3 * image.Channels + i % 3] * 255f);
        return (px, image.Width, image.Height);
    }

    /// <summary>What Save Image writes for a value: clip(255 x v) truncated to a byte.</summary>
    static byte Saved(float v) => (byte)Math.Clamp(v * 255f, 0f, 255f);

    static async Task<(ImageBatch Image, MaskBatch Mask)> Run(string painting, long width = 512, long height = 512, string colour = "#000000", ImageBatch? image = null)
    {
        var store = new MemoryInputStore();
        if (painting.Length > 0) await store.SaveAsync(painting, File.ReadAllBytes(Path.Combine(Dir, painting)));
        var saved = InputFiles.Store;
        InputFiles.Store = store;
        try { return await new Painter().ExecuteAsync(painting.Length > 0 ? painting + " [input]" : "", width, height, colour, image); }
        finally { InputFiles.Store = saved; }
    }

    static void Same(ImageBatch ours, string fixture, int tolerance = 0)
    {
        var (want, w, h) = Png(fixture);
        Assert.That((ours.Width, ours.Height, ours.Batch), Is.EqualTo((w, h, 1)), fixture + " size");
        int worst = 0, off = 0;
        for (int i = 0; i < want.Length; i++)
        {
            int d = Math.Abs(Saved(ours.Data[i / 3 * ours.Channels + i % 3]) - want[i]);
            if (d > 0) off++;
            worst = Math.Max(worst, d);
        }
        TestContext.Out.WriteLine($"{fixture}: {off} of {want.Length} bytes differ, largest {worst}");
        Assert.That(worst, Is.LessThanOrEqualTo(tolerance), fixture + ": largest difference from ComfyUI's saved bytes");
    }

    static void SameMask(MaskBatch ours, string fixture)
    {
        var (want, w, h) = Png(fixture);
        Assert.That((ours.Width, ours.Height, ours.Batch), Is.EqualTo((w, h, 1)), fixture + " size");
        int worst = 0;
        for (int i = 0; i < w * h; i++) worst = Math.Max(worst, Math.Abs(Saved(ours.Data[i]) - want[i * 3]));
        Assert.That(worst, Is.EqualTo(0), fixture + ": largest mask difference from ComfyUI's saved bytes");
    }

    [TestCase("white")]
    [TestCase("mixed")]
    public async Task The_painting_over_the_background_colour(string tag)
    {
        var (image, mask) = await Run($"{tag}-uploaded.png");
        Same(image, $"{tag}-asis-image.png");
        SameMask(mask, $"{tag}-asis-mask.png");
        var (blue, blueMask) = await Run($"{tag}-uploaded.png", colour: "#336699");
        Same(blue, $"{tag}-bg-image.png");
        SameMask(blueMask, $"{tag}-asis-mask.png");   // the mask does not depend on the background
    }

    [Test]
    public async Task Another_width_and_height_resize_the_painting_with_Pillows_Lanczos()
    {
        var (image, mask) = await Run("white-uploaded.png", width: 256, height: 384);
        Same(image, "white-size-image.png");
        SameMask(mask, "white-size-mask.png");
    }

    [TestCase("white")]
    [TestCase("mixed")]
    public async Task A_linked_picture_is_painted_over_and_sets_the_size(string tag)
    {
        var baseImage = ImageDecoder.Decode(File.ReadAllBytes(Path.Combine(Dir, "base.png")));
        // width and height are ignored with a picture linked (measured: 512 x 512 out with 256 x 384 set)
        var (image, mask) = await Run($"{tag}-uploaded.png", width: 256, height: 384, image: baseImage);
        Same(image, $"{tag}-image-image.png");
        SameMask(mask, $"{tag}-asis-mask.png");
    }

    [Test]
    public async Task A_picture_of_another_size_stretches_the_painting_and_only_its_first_frame_is_used()
    {
        // two 300 x 200 frames: the painting (512 x 512) is resized to 300 x 200, the output is one picture
        var frames = new ImageBatch(2, 200, 300, 3);
        Array.Fill(frames.Data, 0.25f, 0, 200 * 300 * 3);
        Array.Fill(frames.Data, 0.75f, 200 * 300 * 3, 200 * 300 * 3);
        var (image, mask) = await Run("white-uploaded.png", image: frames);
        SameMask(mask, "white-imageother-mask.png");
        Assert.Multiple(() =>
        {
            Assert.That((image.Batch, image.Width, image.Height), Is.EqualTo((1, 300, 200)));
            Assert.That(image.Data[0], Is.EqualTo(0.25f), "an unpainted pixel shows the first frame");
            Assert.That(image.Data[(100 * 300 + 150) * 3], Is.EqualTo(1f), "the stroke's white");
        });
    }

    [Test]
    public async Task Without_a_painting_the_base_comes_out_and_the_mask_is_empty()
    {
        var (image, mask) = await Run("", width: 128, height: 64, colour: "#336699");
        Assert.Multiple(() =>
        {
            Assert.That((image.Width, image.Height, mask.Width, mask.Height), Is.EqualTo((128, 64, 128, 64)));
            Assert.That(new[] { Saved(image.Data[0]), Saved(image.Data[1]), Saved(image.Data[2]) }, Is.EqualTo(new byte[] { 51, 102, 153 }));
            Assert.That(mask.Data.All(v => v == 0f), Is.True);
        });
        var black = await Run("");
        Assert.That(black.Image.Data.All(v => v == 0f) && (black.Image.Width, black.Image.Height) == (512, 512), Is.True, "defaults: 512 x 512 of black");
    }

    [Test]
    public void The_schema_is_ComfyUIs()
    {
        var schema = new SpawnGen.Engine.Nodes.NodeRegistry().AddAssembly(typeof(Painter).Assembly).GetSchema("Painter")!;
        Assert.Multiple(() =>
        {
            Assert.That(schema.DisplayName, Is.EqualTo("Painter"));
            Assert.That(schema.Category, Is.EqualTo("image"));
            Assert.That(schema.Inputs.Select(i => $"{i.Name}:{i.Type.Name}:{(i.Required ? "required" : "optional")}"),
                Is.EqualTo(new[] { "mask:STRING:required", "width:INT:required", "height:INT:required", "bg_color:COLOR:required", "image:IMAGE:optional" }));
            var mask = schema.Inputs[0].Widget!;
            Assert.That((mask.WidgetType, mask.Default, mask.Socketless), Is.EqualTo(("PAINTER", (object)"", (bool?)true)));
            var width = schema.Inputs[1].Widget!;
            Assert.That((width.Hidden, width.HiddenStated, width.Default, width.Min, width.Max, width.Step), Is.EqualTo((true, true, (object)512L, (double?)64, (double?)4096, (double?)64)));
            Assert.That(schema.Inputs[3].Widget!.Default, Is.EqualTo("#000000"));
            Assert.That(schema.Outputs.Select(o => $"{o.Name}:{o.Type.Name}"), Is.EqualTo(new[] { "IMAGE:IMAGE", "MASK:MASK" }));
        });
    }
}
