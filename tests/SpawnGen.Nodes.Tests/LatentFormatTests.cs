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

    [TestCase(1024, 1024, 1, new[] { 1, 16, 128, 128 })]
    [TestCase(100, 36, 2, new[] { 2, 16, 4, 12 })]
    [TestCase(272, 160, 1, new[] { 1, 16, 20, 34 })]
    public void EmptySD3LatentImage_is_ComfyUIs_zeros(int w, int h, int batch, int[] shape)
    {
        // Measured with ComfyUI (2026-11-13, SpawnGenDumpLatent): these shapes, every value 0.
        var latent = new SpawnGen.Nodes.Core.EmptySD3LatentImage().Execute(w, h, batch);
        Assert.That(latent.Shape, Is.EqualTo(shape));
        Assert.That(latent.Data, Is.All.EqualTo(0f));
    }

    [Test]
    public void An_empty_latent_takes_the_models_channel_count()
    {
        // ComfyUI sampled SD 1.5 from Empty SD3 Latent Image (16 channels) and from Empty Latent Image (4): the same 4-channel result to the last bit.
        var mask = new MaskBatch(2, 4, 12);
        var sd3 = new SpawnGen.Nodes.Core.EmptySD3LatentImage().Execute(100, 36, 2);
        var withMask = new LatentBatch(2, 16, 4, 12) { NoiseMask = mask, BatchIndex = [3, 5] };
        var r = SamplingCore.ToModelChannels(withMask, LatentFormat.SD15);
        Assert.That(r.Shape, Is.EqualTo(new[] { 2, 4, 4, 12 }));
        Assert.That(r.Data, Is.All.EqualTo(0f));
        Assert.That(r.NoiseMask, Is.SameAs(mask));
        Assert.That(r.BatchIndex, Is.EqualTo(new[] { 3, 5 }));
        Assert.That(SamplingCore.ToModelChannels(sd3, LatentFormat.Wan22).Channels, Is.EqualTo(48));

        var video = new LatentBatch(1, 16, 3, 4, 6);
        Assert.That(SamplingCore.ToModelChannels(video, LatentFormat.Wan22).Shape, Is.EqualTo(new[] { 1, 48, 3, 4, 6 }));

        var four = new LatentBatch(1, 4, 4, 12);
        Assert.That(SamplingCore.ToModelChannels(four, LatentFormat.SD15), Is.SameAs(four), "already the model's: untouched");
        var nonZero = new LatentBatch(1, 16, 4, 12);
        nonZero.Data[7] = 0.5f;
        Assert.That(SamplingCore.ToModelChannels(nonZero, LatentFormat.SD15), Is.SameAs(nonZero), "only an all-zero latent is replaced");
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
