using SpawnGen.Engine.Types;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>ComfyUI's latent formats (comfy/latent_formats.py): scale factors, the empty-latent rule and the latent2rgb preview.</summary>
public class LatentFormatTests
{
    [Test]
    public void Scale_factors_are_ComfyUIs()
    {
        Assert.That(LatentFormat.SD15.ScaleFactor, Is.EqualTo(0.18215f));
        Assert.That(LatentFormat.SDXL.ScaleFactor, Is.EqualTo(0.13025f));
        Assert.That(LatentFormat.SDXL.ProcessIn([1f, -2f]), Is.EqualTo(new[] { 0.13025f, -2f * 0.13025f }));
        Assert.That(LatentFormat.SDXL.ProcessOut([0.13025f]), Is.EqualTo(new[] { 0.13025f / 0.13025f }));
    }

    [Test]
    public void The_empty_latent_is_not_processed_in()
    {
        var empty = new float[16];
        var r = LatentFormat.SD15.ProcessIn(empty);
        Assert.That(r, Is.EqualTo(empty));
        Assert.That(r, Is.Not.SameAs(empty), "a copy: the caller may write to it");
    }

    [Test]
    public void The_SDXL_preview_adds_its_bias_and_uses_its_own_factors()
    {
        // One pixel, channel 0 = 1: SDXL's row 0 plus its bias; (v + 1) / 2 as ComfyUI's preview_to_image.
        var sdxl = LatentPreview.Image([1f, 0f, 0f, 0f], 1, 1, LatentFormat.SDXL);
        Assert.That(sdxl.Data[0], Is.EqualTo((0.3651f + 0.1084f + 1f) / 2f).Within(1e-6));
        Assert.That(sdxl.Data[1], Is.EqualTo((0.4232f - 0.0175f + 1f) / 2f).Within(1e-6));
        Assert.That(sdxl.Data[2], Is.EqualTo((0.4341f - 0.0011f + 1f) / 2f).Within(1e-6));
        var sd = LatentPreview.Image([1f, 0f, 0f, 0f], 1, 1);
        Assert.That(sd.Data[0], Is.EqualTo((0.3512f + 1f) / 2f).Within(1e-6), "SD 1.5 by default, no bias");
    }

    [Test]
    public void AddNoise_on_the_empty_latent_gives_the_noise_in_the_VAEs_units()
    {
        var model = new ModelHandle(null!, new ModelSampling(ModelSigmas.LinearSchedule()));
        var latent = new LatentBatch(1, 4, 8, 8);
        var noisy = new AddNoise().Execute(model, new RandomNoise().Execute(5), new Sigmas([2f]), latent);
        var noise = SamplingCore.InitialNoise(5, 0, latent);
        for (int i = 0; i < noise.Length; i++) Assert.That(noisy.Data[i], Is.EqualTo(noise[i] * 2f / 0.18215f).Within(1e-4));
    }
}
