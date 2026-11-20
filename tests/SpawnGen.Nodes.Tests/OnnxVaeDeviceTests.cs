using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// The ONNX checkpoint's VAE (sd_turbo: the site's first example) on the device (PLANS "Tensors stay on the GPU", 2026-11-19): VAE Decode gives the
/// host path's IMAGE bit for bit and leaves it on the GPU.
/// </summary>
public class OnnxVaeDeviceTests
{
    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }

    [Test, Explicit("Needs SPAWNGEN_MODELS (the models folder with schmuell/sd-turbo-ort-web cached) and a GPU.")]
    public async Task The_ONNX_VAE_decodes_on_the_device_like_the_host_path()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS is not set.");
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var services = new Services(runtime);
        var context = (NodeContext)Activator.CreateInstance(typeof(NodeContext), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, null,
            ["1", "X", services, CancellationToken.None, null, new Lazy<string>(() => "{}"), new Dictionary<string, string>(), null, null, 0], null)!;
        var (_, _, vae) = await new CheckpointLoaderSimple().ExecuteAsync("sd_turbo", context, services);
        Assert.That(vae.Vae.DecodesOnDevice);

        var rng = new Random(3);
        var values = Enumerable.Range(0, 2 * 4 * 32 * 40).Select(_ => (float)(rng.NextDouble() * 4 - 2)).ToArray();
        var latent = new LatentBatch(2, 4, 32, 40, values);
        var device = await new VAEDecode().ExecuteAsync(latent, vae, context, services);
        Assert.That(device.OnHost, Is.False, "the pictures stay on the GPU");
        await device.ToHostAsync();

        var host = new ImageBatch(2, 256, 320, 3);
        for (int b = 0; b < 2; b++)
            VaePictures.FromVae(await vae.Vae.DecodeAsync(values.AsSpan(b * 4 * 32 * 40, 4 * 32 * 40).ToArray(), 32, 40), host, b);
        Assert.That(device.Data, Is.EqualTo(host.Data));
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: {device.Data.Length} values equal");
    }
}
