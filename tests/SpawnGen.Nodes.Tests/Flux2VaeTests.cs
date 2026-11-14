using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// FLUX.2's VAE (ComfyUI's flux2-vae.safetensors) through Load VAE / VAE Encode / VAE Decode against ComfyUI's own results (bef11fc, CPU, float32;
/// Fixtures/flux2_vae, the same dumps SpawnDev.ILGPU.ML's gate uses): LATENT [b, 128, H/16, W/16], patchified and batch-normalised.
/// </summary>
public class Flux2VaeTests
{
    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "flux2_vae");

    static float[] Raw(string name)
    {
        var b = File.ReadAllBytes(Path.Combine(Dir, name));
        var d = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, d, 0, b.Length);
        return d;
    }

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }

    static double RelRms(float[] ours, float[] theirs)
    {
        double num = 0, den = 0;
        for (int i = 0; i < ours.Length; i++) { double d = ours[i] - theirs[i]; num += d * d; den += (double)theirs[i] * theirs[i]; }
        return Math.Sqrt(num / den);
    }

    [Test]
    public void VAE_Encode_cuts_pictures_to_the_VAEs_downscale_around_the_middle()
    {
        // Measured: ComfyUI's FLUX.2 VAE Encode of 200 x 120 equals its encode of the centred 192 x 112 crop (offset 4, 4) exactly.
        var picture = new ImageBatch(1, 120, 200, 3, Raw("f2_img_1x120x200x3.raw"));
        var cropped = VAEEncode.CropForVae(picture, 16);
        Assert.That((cropped.Width, cropped.Height), Is.EqualTo((192, 112)));
        Assert.That(cropped.Data[0], Is.EqualTo(picture.Data[(4 * 200 + 4) * 3]));
        Assert.That(VAEEncode.CropForVae(picture).Width, Is.EqualTo(200), "an 8x VAE keeps 200 x 120");
    }

    [Test, Explicit("Needs SPAWNGEN_MODELS_ROOT with vae/flux2-vae.safetensors (Comfy-Org's) and a GPU.")]
    public async Task Encode_and_decode_match_ComfyUIs_FLUX2_VAE()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        ModelFolders.Scan(root!);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var services = new Services(runtime);
        var context = (NodeContext)Activator.CreateInstance(typeof(NodeContext), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, null,
            ["1", "X", services, CancellationToken.None, null, new Lazy<string>(() => "{}"), new Dictionary<string, string>(), null, null, 0], null)!;
        var vae = await new VAELoader().ExecuteAsync("flux2-vae.safetensors", context, services);
        Assert.That(vae.Vae.Downscale, Is.EqualTo(16));

        var a = new ImageBatch(1, 192, 256, 3, Raw("f2_img_1x192x256x3.raw"));
        var latentA = await new VAEEncode().ExecuteAsync(a, vae, context, services);
        Assert.That(latentA.Shape, Is.EqualTo(new[] { 1, 128, 12, 16 }));
        double encA = RelRms(latentA.Data, Raw("enc_256x192_samples_f32_1x128x12x16.raw"));

        var b = new ImageBatch(1, 120, 200, 3, Raw("f2_img_1x120x200x3.raw"));
        var latentB = await new VAEEncode().ExecuteAsync(b, vae, context, services);
        Assert.That(latentB.Shape, Is.EqualTo(new[] { 1, 128, 7, 12 }));
        double encB = RelRms(latentB.Data, Raw("enc_200x120_samples_f32_1x128x7x12.raw"));

        // Decode ComfyUI's own latent of A, and the synthetic latent.
        var comfyA = new LatentBatch(1, 128, 12, 16, Raw("enc_256x192_samples_f32_1x128x12x16.raw"));
        var decA = await new VAEDecode().ExecuteAsync(comfyA, vae, context, services);
        Assert.That((decA.Width, decA.Height), Is.EqualTo((256, 192)));
        double decodeA = RelRms(decA.Data, Raw("dec_of_enc_256x192_images_f32_1x192x256x3.raw"));
        var bytes = new byte[decA.Data.Length * 4];
        Buffer.BlockCopy(decA.Data, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(Path.Combine(TestContext.CurrentContext.WorkDirectory, "flux2_dec_A_ours_1x192x256x3.raw"), bytes);
        var synthetic = new LatentBatch(1, 128, 12, 16, Raw("f2_lat_1x128x12x16.raw"));
        var decS = await new VAEDecode().ExecuteAsync(synthetic, vae, context, services);
        double decodeS = RelRms(decS.Data, Raw("dec_synthetic_12x16_images_f32_1x192x256x3.raw"));

        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: encode A {encA:E2}, encode B {encB:E2}, decode A {decodeA:E2}, decode synthetic {decodeS:E2} (relative RMS vs ComfyUI)");
        // The library's gate (CUDA, RTX 4070): encode A 5.8e-4, B 9.3e-4, decode 1.1e-4. A device with other float rounding lands near that.
        Assert.That(encA, Is.LessThan(5e-3));
        Assert.That(encB, Is.LessThan(5e-3));
        Assert.That(decodeA, Is.LessThan(5e-3));
        Assert.That(decodeS, Is.LessThan(5e-3));
    }
}
