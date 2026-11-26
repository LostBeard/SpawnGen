using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Does the same picture encode to the same latent every time on this machine's GPU? Found 2026-11-26 on the RX 590 under OpenCL: image to
/// image and inpainting (the workflows that go through VAE Encode) came out 1/255 different in a few hundred pixels between two runs of
/// the same build, while every workflow without an encode was identical byte for byte. Explicit: it needs the GPU and the SD-Turbo
/// checkpoint in the models folder (SPAWNGEN_MODELS, default %LOCALAPPDATA%\SpawnGen\models).
/// </summary>
public class VaeEncodeRepeatTests
{
    static string Models => Environment.GetEnvironmentVariable("SPAWNGEN_MODELS")
        ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "SpawnGen", "models");

    /// <summary>A fixed picture in the VAE's layout (3 planes, -1..1): smooth ramps with a fine texture on top.</summary>
    static float[] Picture(int h, int w)
    {
        var px = new float[3 * h * w];
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    px[(c * h + y) * w + x] = (float)(0.6 * Math.Sin(0.013 * x * (c + 1) + 0.021 * y) + 0.3 * Math.Cos(0.37 * x - 0.29 * y * (c + 1)) + 0.05 * ((x * 7 + y * 13 + c * 5) % 11 - 5) / 5);
        return px;
    }

    [TestCase("sd_turbo", 512)]
    [TestCase("sd_turbo", 256)]
    [TestCase("sd_v1-5", 512)]
    [Explicit("Needs the GPU and the checkpoint in the models folder. Reports how many latent values differ between repeated encodes of one picture.")]
    public async Task The_same_picture_encodes_to_the_same_latent_every_time(string checkpointName, int size)
    {
        using var runtime = new MlRuntime(new DesktopModelSource(Models, MlRuntime.RepoFolders));
        TestContext.Out.WriteLine($"accelerator: {runtime.Accelerator.AcceleratorType} {runtime.Accelerator.Name}");
        var checkpoint = await runtime.GetCheckpointAsync(checkpointName);
        var picture = Picture(size, size);
        var first = await checkpoint.EncodeImageAsync(picture, size, size);
        int worstCount = 0;
        float worst = 0;
        for (int run = 2; run <= 6; run++)
        {
            var again = await checkpoint.EncodeImageAsync(picture, size, size);
            Assert.That(again.Length, Is.EqualTo(first.Length));
            int differ = 0; float largest = 0;
            for (int i = 0; i < first.Length; i++)
            {
                float d = Math.Abs(again[i] - first[i]);
                if (d != 0) { differ++; largest = Math.Max(largest, d); }
            }
            TestContext.Out.WriteLine($"encode {run} vs encode 1: {differ} of {first.Length} latent values differ, largest difference {largest:G6}");
            worstCount = Math.Max(worstCount, differ); worst = Math.Max(worst, largest);
        }
        Assert.That(worstCount, Is.EqualTo(0), $"repeated encodes of one picture differ (largest {worst:G6}): something in the encoder depends on an order that is not fixed");
    }
}
