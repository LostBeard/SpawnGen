using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

public class LatentPreviewTests
{
    [Test]
    public void A_zero_latent_is_mid_grey_and_one_channel_shifts_the_colours_by_its_factors()
    {
        var zero = LatentPreview.Image(new float[4 * 2 * 3], 2, 3);
        Assert.That(zero.Data, Is.All.EqualTo(0.5f));

        var latent = new float[4 * 1 * 1];
        latent[0] = 1f; // only channel 0
        var one = LatentPreview.Image(latent, 1, 1);
        Assert.That(one.Data[0], Is.EqualTo((0.3512f + 1) / 2).Within(1e-6));
        Assert.That(one.Data[1], Is.EqualTo((0.2297f + 1) / 2).Within(1e-6));
        Assert.That(one.Data[2], Is.EqualTo((0.3227f + 1) / 2).Within(1e-6));

        latent = [10f, 10f, 10f, -10f];
        Assert.That(LatentPreview.Image(latent, 1, 1).Data, Is.All.InRange(0f, 1f), "extreme values are clamped");
    }

    [Test]
    public void The_preview_is_a_PNG_of_the_latents_size()
    {
        var png = LatentPreview.Png(new float[4 * 8 * 8], 8, 8);
        var back = PngReader.Decode(png);
        Assert.That((back.Width, back.Height), Is.EqualTo((8, 8)));
    }
}
