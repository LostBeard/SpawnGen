using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// The hosted model files (HostedModels, wwwroot/model-catalog.json): each listed by the loader of its ComfyUI folder and opened from its Hugging
/// Face repo, a file of the same name in the models folder winning. Names are unique to this test (the catalogs are process-wide).
/// </summary>
public class HostedModelsTests
{
    static string Repo()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "SpawnGen.slnx"))) dir = dir.Parent;
        return dir!.FullName;
    }

    [Test]
    public void Each_file_is_listed_by_the_loader_of_its_folder_and_opened_from_its_repo()
    {
        int listed = HostedModels.Register([
            new("hosted_test_dit.safetensors", "Org/dit-repo", "split_files/diffusion_models/hosted_test_dit.safetensors", "diffusion_models"),
            new("hosted_test_te.safetensors", "Org/te-repo", "split_files/text_encoders/hosted_test_te.safetensors", "clip"),
            new("hosted_test_vae.safetensors", "Org/vae-repo", "vae/hosted_test_vae.safetensors", "vae"),
            new("hosted_test_ckpt.safetensors", "Org/ckpt-repo", "hosted_test_ckpt.safetensors", "checkpoints"),
            new("hosted_test_lora.safetensors", "Org/lora-repo", "loras/hosted_test_lora.safetensors", "loras"),
            new("hosted_test_other.safetensors", "Org/x", "x.safetensors", "geometry_estimation"),
        ]);
        Assert.That(listed, Is.EqualTo(5), "a folder no loader here reads is skipped");
        Assert.That(ModelFolders.Names(ModelFolders.DiffusionModels), Does.Contain("hosted_test_dit.safetensors"));
        Assert.That(ModelFolders.Locate(ModelFolders.DiffusionModels, "hosted_test_dit.safetensors"),
            Is.EqualTo(("Org/dit-repo", "split_files/diffusion_models/hosted_test_dit.safetensors")));
        Assert.That(ModelFolders.Locate(ModelFolders.TextEncoders, "hosted_test_te.safetensors").Repo, Is.EqualTo("Org/te-repo"), "clip is text_encoders' older name");
        Assert.That(ModelFolders.Locate(ModelFolders.Vae, "hosted_test_vae.safetensors").Repo, Is.EqualTo("Org/vae-repo"));
        Assert.That(CheckpointCatalog.Repos["hosted_test_ckpt.safetensors"], Is.EqualTo("Org/ckpt-repo"));
        Assert.That(CheckpointCatalog.Details["hosted_test_ckpt.safetensors"].File, Is.EqualTo("hosted_test_ckpt.safetensors"));
        Assert.That(CheckpointCatalog.Loras, Does.Contain("hosted_test_lora.safetensors"));
        Assert.That(CheckpointCatalog.LocateLora("hosted_test_lora.safetensors"), Is.EqualTo(("Org/lora-repo", "loras/hosted_test_lora.safetensors")));
    }

    [Test]
    public void A_file_in_the_models_folder_wins_over_a_hosted_one_of_the_same_name()
    {
        HostedModels.Register([
            new("hosted_local_vae.safetensors", "Org/vae-repo", "vae/hosted_local_vae.safetensors", "vae"),
            new("hosted_local_lora.safetensors", "Org/lora-repo", "hosted_local_lora.safetensors", "loras"),
        ]);
        var models = Path.Combine(Path.GetTempPath(), "spawngen-hosted-" + Guid.NewGuid().ToString("N"));
        try
        {
            Directory.CreateDirectory(Path.Combine(models, "vae"));
            Directory.CreateDirectory(Path.Combine(models, "loras"));
            File.WriteAllBytes(Path.Combine(models, "vae", "hosted_local_vae.safetensors"), [0]);
            File.WriteAllBytes(Path.Combine(models, "loras", "hosted_local_lora.safetensors"), [0]);
            ModelFolders.Scan(models);
            CheckpointCatalog.ScanLoraFiles(models);
            Assert.That(ModelFolders.Locate(ModelFolders.Vae, "hosted_local_vae.safetensors"), Is.EqualTo(("local/vae", "hosted_local_vae.safetensors")));
            Assert.That(CheckpointCatalog.LocateLora("hosted_local_lora.safetensors"), Is.EqualTo((CheckpointCatalog.LocalLorasRepo, "hosted_local_lora.safetensors")));

            // Registering the hosted file again keeps the local one.
            HostedModels.Register([new("hosted_local_vae.safetensors", "Org/vae-repo", "vae/hosted_local_vae.safetensors", "vae")]);
            Assert.That(ModelFolders.Locate(ModelFolders.Vae, "hosted_local_vae.safetensors").Repo, Is.EqualTo("local/vae"));
        }
        finally { Directory.Delete(models, true); }
    }

    [Test]
    public void The_shipped_catalog_lists_the_LTX_2_template_files_where_ComfyUI_gets_them()
    {
        var json = File.ReadAllText(Path.Combine(Repo(), "src", "SpawnGen.Studio", "wwwroot", "model-catalog.json"));
        using var doc = System.Text.Json.JsonDocument.Parse(json);
        var models = doc.RootElement.GetProperty("models");
        Assert.That(models.EnumerateObject().Count(), Is.GreaterThan(300));
        var ckpt = models.GetProperty("ltx-2-19b-dev-fp8.safetensors");
        Assert.That(ckpt.GetProperty("repo").GetString(), Is.EqualTo("Lightricks/LTX-2"));
        Assert.That(ckpt.GetProperty("folder").GetString(), Is.EqualTo("checkpoints"));
        var gemma = models.GetProperty("gemma_3_12B_it_fp4_mixed.safetensors");
        Assert.That(gemma.GetProperty("repo").GetString(), Is.EqualTo("Comfy-Org/ltx-2"));
        Assert.That(gemma.GetProperty("path").GetString(), Is.EqualTo("split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors"));
        using var needs = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(Repo(), "src", "SpawnGen.Studio", "wwwroot", "template-needs.json")));
        Assert.That(doc.RootElement.GetProperty("commit").GetString(), Is.EqualTo(needs.RootElement.GetProperty("commit").GetString()),
            "the catalog and the template list come from one commit");
    }
}
