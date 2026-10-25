using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>Single-file checkpoints in &lt;models&gt;/checkpoints, and the CLIP nodes that come with them.</summary>
[NonParallelizable]
public class SafetensorsCatalogTests
{
    [Test]
    public void Safetensors_files_in_the_checkpoints_folder_are_listed_by_their_path_there_and_open_from_it()
    {
        var root = Path.Combine(Path.GetTempPath(), "spawngen-ckpts-" + Guid.NewGuid().ToString("N"));
        var tag = Guid.NewGuid().ToString("N")[..8];
        try
        {
            var dir = Path.Combine(root, "checkpoints");
            Directory.CreateDirectory(Path.Combine(dir, "SDXL"));
            File.WriteAllText(Path.Combine(dir, $"a_{tag}.safetensors"), "x");
            File.WriteAllText(Path.Combine(dir, "SDXL", $"b_{tag}.safetensors"), "x");
            File.WriteAllText(Path.Combine(dir, $"c_{tag}.ckpt"), "x"); // pickled checkpoints are not loaded

            Assert.That(CheckpointCatalog.ScanCheckpointFiles(root), Is.EqualTo(2));
            string nested = Path.Combine("SDXL", $"b_{tag}.safetensors"); // ComfyUI's name: the path with the platform's separator
            Assert.That(CheckpointCatalog.Repos.Keys, Does.Contain($"a_{tag}.safetensors").And.Contain(nested).And.Not.Contain($"c_{tag}.ckpt"));
            Assert.That(CheckpointCatalog.Details[nested].File, Is.EqualTo(nested));
            Assert.That(CheckpointCatalog.Repos[nested], Is.EqualTo(CheckpointCatalog.LocalCheckpointsRepo));
            Assert.That(MlRuntime.RepoFolders[CheckpointCatalog.LocalCheckpointsRepo], Is.EqualTo("checkpoints"));
            Assert.That(CheckpointCatalog.ScanCheckpointFiles(root), Is.EqualTo(0), "a second scan adds nothing");

            // The desktop source reads the file from the checkpoints folder instead of downloading it.
            var source = new DesktopModelSource(root, MlRuntime.RepoFolders);
            using var s = source.OpenAsync(CheckpointCatalog.LocalCheckpointsRepo, nested).GetAwaiter().GetResult();
            Assert.That(new StreamReader(s).ReadToEnd(), Is.EqualTo("x"));
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }

    [Test]
    public void The_built_in_single_files_download_and_a_local_file_of_the_same_name_replaces_the_download()
    {
        Assert.That(CheckpointCatalog.Repos["sd_xl_base_1.0.safetensors"], Is.EqualTo("stabilityai/stable-diffusion-xl-base-1.0"));
        Assert.That(CheckpointCatalog.Details["sd_turbo.safetensors"].File, Is.EqualTo("sd_turbo.safetensors"));
        Assert.That(MlRuntime.RepoFolders.ContainsKey("stabilityai/stable-diffusion-xl-base-1.0"), Is.False, "a download is never looked up in a sub-folder");
        const string name = "v1-5-pruned-emaonly-fp16.safetensors";
        if (CheckpointCatalog.Repos[name] == CheckpointCatalog.LocalCheckpointsRepo) Assert.Inconclusive("another test already put a local file of this name in the catalog");
        Assert.That(CheckpointCatalog.Repos[name], Is.EqualTo("Comfy-Org/stable-diffusion-v1-5-archive"));

        var root = Path.Combine(Path.GetTempPath(), "spawngen-ckpts-" + Guid.NewGuid().ToString("N"));
        try
        {
            Directory.CreateDirectory(Path.Combine(root, "checkpoints"));
            File.WriteAllText(Path.Combine(root, "checkpoints", name), "x");
            Assert.That(CheckpointCatalog.ScanCheckpointFiles(root), Is.EqualTo(1));
            Assert.That(CheckpointCatalog.Repos[name], Is.EqualTo(CheckpointCatalog.LocalCheckpointsRepo), "the local file is used, not downloaded again");
            Assert.That(CheckpointCatalog.Details[name].File, Is.EqualTo(name));
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }

    [Test]
    public void A_missing_checkpoints_folder_adds_nothing() =>
        Assert.That(CheckpointCatalog.ScanCheckpointFiles(Path.Combine(Path.GetTempPath(), "spawngen-none-" + Guid.NewGuid().ToString("N"))), Is.EqualTo(0));

    [Test]
    public void CLIP_Set_Last_Layer_returns_a_new_CLIP_with_the_layer_and_leaves_the_input_alone()
    {
        var clip = new ClipHandle(null!);
        var set = new CLIPSetLastLayer().Execute(clip, -2);
        Assert.That(set.Layer, Is.EqualTo(-2));
        Assert.That(set, Is.Not.SameAs(clip));
        Assert.That(clip.Layer, Is.Null);
    }

    [Test]
    public void Prompt_weights_apply_per_tower_columns()
    {
        // Two rows of width 4: columns 0..1 are tower L, 2..3 tower G. Only row 1 has a weight, in tower G only.
        var z = new float[] { 1, 1, 1, 1, 2, 2, 2, 2 };
        var empty = new float[8];
        PromptWeights.Apply(z, empty, [1f, 1f], 4, 0, 2);
        PromptWeights.Apply(z, empty, [1f, 3f], 4, 2, 2);
        Assert.That(z, Is.EqualTo(new float[] { 1, 1, 1, 1, 2, 2, 6, 6 }));
    }
}
