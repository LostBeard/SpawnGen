using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Detect Edges (Canny) against ComfyUI's own node (kornia.filters.canny): Fixtures/canny holds ComfyUI's SD 1.5 apple and the edge maps ComfyUI made from it at
/// two threshold pairs (the default 0.4 / 0.8, and 0.1 / 0.3 for a dense map).
/// </summary>
public class CannyTests
{
    static string Fixture(string name) => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "canny", name);

    [TestCase(0.4, 0.8, "apple_canny_0.4_0.8.png")]
    [TestCase(0.1, 0.3, "apple_canny_0.1_0.3.png")]
    public void Edges_are_ComfyUIs_pixel_for_pixel(double low, double high, string expected)
    {
        var image = ImageDecoder.Decode(File.ReadAllBytes(Fixture("apple.png")));
        var theirs = ImageDecoder.Decode(File.ReadAllBytes(Fixture(expected)));
        var ours = new Canny().Execute(image, (float)low, (float)high);
        Assert.That((ours.Height, ours.Width, ours.Channels), Is.EqualTo((theirs.Height, theirs.Width, 3)));
        int edgePixels = 0, different = 0;
        for (int p = 0; p < ours.Height * ours.Width; p++)
        {
            bool mine = ours.Data[p * 3] > 0.5f, comfy = theirs.Data[p * theirs.Channels] > 0.5f;
            if (comfy) edgePixels++;
            if (mine != comfy) different++;
        }
        TestContext.Out.WriteLine($"{expected}: {edgePixels} edge pixels in ComfyUI's map, {different} pixels differ");
        Assert.That(edgePixels, Is.GreaterThan(100), "the reference has edges");
        Assert.That(different, Is.EqualTo(0));
    }

    [Test]
    public void Output_is_black_and_white_in_three_channels_and_thresholds_are_checked()
    {
        var image = new ImageBatch(1, 16, 16, 3);
        for (int y = 0; y < 16; y++) for (int x = 8; x < 16; x++) for (int c = 0; c < 3; c++) image.Data[(y * 16 + x) * 3 + c] = 1f;
        var edges = new Canny().Execute(image, 0.1f, 0.2f);
        Assert.That(edges.Data.Distinct().OrderBy(v => v), Is.EqualTo(new[] { 0f, 1f }));
        for (int p = 0; p < 256; p++) Assert.That(edges.Data[p * 3], Is.EqualTo(edges.Data[p * 3 + 2]));
        Assert.That(Enumerable.Range(0, 16).Any(y => edges.Data[(y * 16 + 7) * 3] == 1f || edges.Data[(y * 16 + 8) * 3] == 1f), "the step between columns 7 and 8 is an edge");
        Assert.That(() => new Canny().Execute(image, 0.5f, 0.4f), Throws.ArgumentException);
    }
}
