using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>ComfyUI's component folders (diffusion_models / unet, vae, text_encoders / clip) and the loaders' choices SpawnGen does not run.</summary>
[NonParallelizable]
public class ComponentLoaderTests
{
    [Test]
    public void Component_files_are_listed_from_the_current_and_the_older_folder_and_open_from_where_they_are()
    {
        var root = Path.Combine(Path.GetTempPath(), "spawngen-parts-" + Guid.NewGuid().ToString("N"));
        var tag = Guid.NewGuid().ToString("N")[..8];
        try
        {
            void Put(string folder, string name) { Directory.CreateDirectory(Path.GetDirectoryName(Path.Combine(root, folder, name))!); File.WriteAllText(Path.Combine(root, folder, name), folder); }
            Put("diffusion_models", $"u_{tag}.safetensors");
            Put("unet", $"old_{tag}.safetensors");
            Put("vae", Path.Combine("sdxl", $"v_{tag}.safetensors"));
            Put("clip", $"c_{tag}.safetensors");
            Put("text_encoders", $"t_{tag}.ckpt"); // not a safetensors file: not listed

            Assert.That(ModelFolders.Scan(root), Is.EqualTo(4));
            Assert.That(ModelFolders.Names(ModelFolders.DiffusionModels), Does.Contain($"u_{tag}.safetensors").And.Contain($"old_{tag}.safetensors"));
            string nested = Path.Combine("sdxl", $"v_{tag}.safetensors");
            Assert.That(VAELoader.VaeNames(), Does.Contain(nested).And.Contain("pixel_space"));
            Assert.That(CLIPLoader.ClipNames(), Does.Contain($"c_{tag}.safetensors").And.Not.Contain($"t_{tag}.ckpt"));
            Assert.That(ModelFolders.Locate(ModelFolders.DiffusionModels, $"old_{tag}.safetensors"), Is.EqualTo(("local/unet", $"old_{tag}.safetensors")));
            Assert.That(ModelFolders.Scan(root), Is.EqualTo(0), "a second scan adds nothing");

            var source = new DesktopModelSource(root, MlRuntime.RepoFolders);
            using var s = source.OpenAsync("local/clip", $"c_{tag}.safetensors").GetAwaiter().GetResult();
            Assert.That(new StreamReader(s).ReadToEnd(), Is.EqualTo("clip"));
            Assert.That(() => ModelFolders.Locate(ModelFolders.Vae, "missing.safetensors"), Throws.ArgumentException.With.Message.Contains("vae folder"));
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }

    [Test]
    public void Choices_SpawnGen_does_not_run_fail_with_a_reason_before_anything_loads()
    {
        Assert.That(async () => await new UNETLoader().ExecuteAsync("x.safetensors", "fp8_e4m3fn", null!, null!),
            Throws.TypeOf<NotSupportedException>().With.Message.Contains("default"));
        Assert.That(async () => await new CLIPLoader().ExecuteAsync("x.safetensors", "sd3", null!, null!),
            Throws.TypeOf<NotSupportedException>().With.Message.Contains("stable_diffusion"));
        Assert.That(async () => await new DualCLIPLoader().ExecuteAsync("a.safetensors", "b.safetensors", "flux", null!, null!),
            Throws.TypeOf<NotSupportedException>().With.Message.Contains("sdxl"));
        Assert.That(async () => await new VAELoader().ExecuteAsync("pixel_space", null!, null!), Throws.TypeOf<NotSupportedException>());
    }

    [Test]
    public void The_loaders_offer_ComfyUIs_full_type_lists_so_workflows_open()
    {
        Assert.That(CLIPLoader.Types(), Has.Length.EqualTo(32).And.Contain("stable_diffusion"));
        Assert.That(DualCLIPLoader.Types(), Has.Length.EqualTo(12).And.Contain("sdxl"));
        Assert.That(UNETLoader.WeightDtypes(), Is.EqualTo(new[] { "default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2" }));
    }
}
