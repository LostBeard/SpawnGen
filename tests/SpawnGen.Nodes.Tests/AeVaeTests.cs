using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// FLUX.1 / Z-Image's VAE (ComfyUI's ae.safetensors, also loaded by the HiDream, Lumina, OmniGen2 and Chroma templates) through Load VAE / VAE Encode /
/// VAE Decode against ComfyUI's own results (bef11fc, CPU, float32; Fixtures/ae_vae, the dumps sent to SpawnDev.ILGPU.ML). Measured: LATENT is
/// the raw posterior mean [b, 16, H/8, W/8] (no scale or shift: FLUX.1's and Z-Image's go on the way into the model), cut to multiples of 8 around
/// the picture's middle.
/// </summary>
public class AeVaeTests
{
    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "ae_vae");

    /// <summary>Pictures A (256 x 192) and B (200 x 120) are the FLUX.2 VAE fixtures' (the same bytes).</summary>
    static string Flux2Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "flux2_vae");

    static float[] Raw(string dir, string name)
    {
        var b = File.ReadAllBytes(Path.Combine(dir, name));
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
    public void VAE_Encode_cuts_205x125_to_its_middle_200x120()
    {
        // Measured: ComfyUI's encode of C (205 x 125) equals its encode of the centred 200 x 120 crop, offset (2, 2).
        var c = new ImageBatch(1, 125, 205, 3, Raw(Dir, "ae_img_1x125x205x3.raw"));
        var cropped = VAEEncode.CropForVae(c);
        Assert.That((cropped.Width, cropped.Height), Is.EqualTo((200, 120)));
        Assert.That(cropped.Data[0], Is.EqualTo(c.Data[(2 * 205 + 2) * 3]));
    }

    [Test, Explicit("Needs SPAWNGEN_MODELS_ROOT with vae/ae.safetensors (Comfy-Org's) and a GPU.")]
    public async Task Encode_and_decode_match_ComfyUIs_ae_VAE()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        ModelFolders.Scan(root!);
        // SPAWNGEN_TEST_CPU=1 runs it on ILGPU's CPU accelerator instead (slow; tells a device's arithmetic from the library's).
        using var cpuContext = Environment.GetEnvironmentVariable("SPAWNGEN_TEST_CPU") == "1" ? ILGPU.Context.Create(b => b.CPU()) : null;
        using var cpu = cpuContext is null ? null : cpuContext.CreateCPUAccelerator(0);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders), cpu);
        var services = new Services(runtime);
        var context = (NodeContext)Activator.CreateInstance(typeof(NodeContext), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, null,
            ["1", "X", services, CancellationToken.None, null, new Lazy<string>(() => "{}"), new Dictionary<string, string>(), null, null, 0], null)!;
        var vae = await new VAELoader().ExecuteAsync("ae.safetensors", context, services);
        Assert.That(vae.Vae.Downscale, Is.EqualTo(8));

        var a = new ImageBatch(1, 192, 256, 3, Raw(Flux2Dir, "f2_img_1x192x256x3.raw"));
        var latentA = await new VAEEncode().ExecuteAsync(a, vae, context, services);
        Assert.That(latentA.Shape, Is.EqualTo(new[] { 1, 16, 24, 32 }));
        double encA = RelRms(latentA.Data, Raw(Dir, "ae_enc_256x192_samples_f32_1x16x24x32.raw"));

        var b = new ImageBatch(1, 120, 200, 3, Raw(Flux2Dir, "f2_img_1x120x200x3.raw"));
        var latentB = await new VAEEncode().ExecuteAsync(b, vae, context, services);
        Assert.That(latentB.Shape, Is.EqualTo(new[] { 1, 16, 15, 25 }));
        double encB = RelRms(latentB.Data, Raw(Dir, "ae_enc_200x120_samples_f32_1x16x15x25.raw"));

        var c = new ImageBatch(1, 125, 205, 3, Raw(Dir, "ae_img_1x125x205x3.raw"));
        var latentC = await new VAEEncode().ExecuteAsync(c, vae, context, services);
        Assert.That(latentC.Shape, Is.EqualTo(new[] { 1, 16, 15, 25 }));
        double encC = RelRms(latentC.Data, Raw(Dir, "ae_enc_205x125_samples_f32_1x16x15x25.raw"));

        // Decode ComfyUI's own latent of A, and the synthetic latent.
        var comfyA = new LatentBatch(1, 16, 24, 32, Raw(Dir, "ae_enc_256x192_samples_f32_1x16x24x32.raw"));
        var decA = await new VAEDecode().ExecuteAsync(comfyA, vae, context, services);
        Assert.That((decA.Width, decA.Height), Is.EqualTo((256, 192)));
        double decodeA = RelRms(decA.Data, Raw(Dir, "ae_dec_of_enc_256x192_images_f32_1x192x256x3.raw"));
        var synthetic = new LatentBatch(1, 16, 24, 32, Raw(Dir, "ae_lat_1x16x24x32.raw"));
        var decS = await new VAEDecode().ExecuteAsync(synthetic, vae, context, services);
        double decodeS = RelRms(decS.Data, Raw(Dir, "ae_dec_synthetic_24x32_images_f32_1x192x256x3.raw"));

        // Ours, for a look at where any difference sits (ae_ours_*.raw next to the test assembly).
        foreach (var (tag, l) in new[] { ("encA", latentA), ("encB", latentB), ("encC", latentC) })
        {
            var bytes = new byte[l.Data.Length * 4];
            Buffer.BlockCopy(l.Data, 0, bytes, 0, bytes.Length);
            File.WriteAllBytes(Path.Combine(TestContext.CurrentContext.WorkDirectory, $"ae_ours_{tag}_{string.Join("x", l.Shape)}.raw"), bytes);
        }
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: encode A {encA:E2}, B {encB:E2}, C (cut) {encC:E2}, decode A {decodeA:E2}, decode synthetic {decodeS:E2} (relative RMS vs ComfyUI)");
        // Measured on the RX 590 (OpenCL, ML 5.8.18-local.3 on SpawnDev.ILGPU 5.4.32): encode A 3.0e-4, B 1.75e-5, C 4.5e-5, decode A 9.6e-7, synthetic
        // 1.4e-5. With 5.8.17 encode A was 4.85e-3 at a few single latent cells: the library kept the weights as fp16 (proven by Tuvok on the host);
        // it now keeps fp32 weights as fp16 hi + lo with a second conv pass.
        Assert.That(encA, Is.LessThan(1e-3));
        Assert.That(encB, Is.LessThan(2e-4));
        Assert.That(encC, Is.LessThan(5e-4));
        Assert.That(decodeA, Is.LessThan(1e-4));
        Assert.That(decodeS, Is.LessThan(1e-4));
    }

    /// <summary>The library's ae encoder, stage by stage, on picture A (SdVae.Tap): ae_tap_&lt;stage&gt;_&lt;h&gt;x&lt;w&gt;x&lt;c&gt;.raw, channel-last.</summary>
    [Test, Explicit("Needs SPAWNGEN_MODELS_ROOT with vae/ae.safetensors and a GPU; a diagnostic.")]
    public async Task Encoder_stage_taps_for_picture_A()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        ModelFolders.Scan(root!);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var (repo, file) = ModelFolders.Locate(ModelFolders.Vae, "ae.safetensors");
        using var vae = await SpawnDev.ILGPU.ML.StableDiffusion.SdVae.LoadAsync(runtime.Accelerator, runtime.ModelSource, repo, file);
        var dir = Path.Combine(TestContext.CurrentContext.WorkDirectory, "ae_taps");
        Directory.CreateDirectory(dir);
        var names = new List<string>();
        vae.Tap = (stage, view, h, w, c) =>
        {
            runtime.Accelerator.Synchronize();
            var data = new float[(long)h * w * c];
            view.SubView(0, data.Length).CopyToCPU(data);
            var bytes = new byte[data.Length * 4];
            Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
            File.WriteAllBytes(Path.Combine(dir, $"ae_tap_{stage.Replace('.', '_')}_{h}x{w}x{c}.raw"), bytes);
            names.Add($"{stage} {h}x{w}x{c}");
        };
        var a = new ImageBatch(1, 192, 256, 3, Raw(Flux2Dir, "f2_img_1x192x256x3.raw"));
        var chw = new float[3 * 192 * 256];
        for (int y = 0; y < 192; y++)
            for (int x = 0; x < 256; x++)
                for (int c = 0; c < 3; c++) chw[(c * 192 + y) * 256 + x] = a.Data[(y * 256 + x) * 3 + c] * 2 - 1;
        var mean = await vae.EncodeAsync(chw, 192, 256);
        var latentBytes = new byte[mean.Length * 4];
        Buffer.BlockCopy(mean, 0, latentBytes, 0, latentBytes.Length);
        File.WriteAllBytes(Path.Combine(dir, "ae_mean_16x24x32.raw"), latentBytes);
        TestContext.Out.WriteLine(string.Join("\n", names) + "\n" + dir);
    }
}
