using SpawnGen.Engine.Nodes;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>Checkpoints the user adds: checkpoints.json and folders in the models folder.</summary>
[NonParallelizable]
public class CheckpointCatalogTests
{
    string _dir = null!;
    readonly List<string> _added = [];

    [SetUp]
    public void SetUp()
    {
        _dir = Path.Combine(Path.GetTempPath(), "spawngen-catalog-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    [TearDown]
    public void TearDown() => Directory.Delete(_dir, true);

    static void Touch(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, "x");
    }

    [Test]
    public void The_built_in_checkpoints_are_there_with_their_settings()
    {
        Assert.That(CheckpointCatalog.Repos["sd_turbo"], Is.EqualTo("schmuell/sd-turbo-ort-web"));
        Assert.That(CheckpointCatalog.Details["sd_turbo"].TextWidth, Is.EqualTo(1024));
        Assert.That(CheckpointCatalog.Details["sd_v1-5"], Is.EqualTo(new MlRuntime.CheckpointInfo(768, "unet/model.onnx_data")));
    }

    [Test]
    public void A_checkpoints_json_adds_entries_and_skips_the_ones_it_cannot_read()
    {
        File.WriteAllText(Path.Combine(_dir, "checkpoints.json"), """
            { "catalog_test_a": { "repo": "someone/sd15-onnx", "textWidth": 768, "unetData": "unet/model.onnx_data" },
              "catalog_test_b": { "repo": "someone/sd21-onnx" },
              "catalog_test_bad1": { "textWidth": 768 },
              "catalog_test_bad2": "text",
              "catalog_test_bad3": { "repo": "x/y", "textWidth": 3 } }
            """);
        Assert.That(CheckpointCatalog.LoadJson(Path.Combine(_dir, "checkpoints.json")), Is.EqualTo(2));
        Assert.That(CheckpointCatalog.Repos["catalog_test_a"], Is.EqualTo("someone/sd15-onnx"));
        Assert.That(CheckpointCatalog.Details["catalog_test_a"], Is.EqualTo(new MlRuntime.CheckpointInfo(768, "unet/model.onnx_data")));
        Assert.That(CheckpointCatalog.Details["catalog_test_b"], Is.EqualTo(new MlRuntime.CheckpointInfo(1024)));
        Assert.That(CheckpointCatalog.Repos.ContainsKey("catalog_test_bad1"), Is.False);
        Assert.That(CheckpointCatalog.Repos.ContainsKey("catalog_test_bad3"), Is.False);
        Assert.That(MlRuntime.KnownCheckpoints.ContainsKey("catalog_test_a"), Is.True, "the Load Checkpoint list follows the catalog");
        Assert.That(CheckpointCatalog.LoadJson(Path.Combine(_dir, "missing.json")), Is.EqualTo(0));
        File.WriteAllText(Path.Combine(_dir, "broken.json"), "{ not json");
        Assert.That(CheckpointCatalog.LoadJson(Path.Combine(_dir, "broken.json")), Is.EqualTo(0));
    }

    [Test]
    public void A_folder_with_the_ONNX_layout_becomes_a_checkpoint_and_other_folders_do_not()
    {
        var good = Path.Combine(_dir, "catalog_test_folder");
        foreach (var f in new[] { "unet/model.onnx", "unet/model.onnx_data", "text_encoder/model.onnx", "vae_decoder/model.onnx", "tokenizer/vocab.json" }) Touch(Path.Combine(good, f));
        File.WriteAllText(Path.Combine(good, "checkpoint.json"), "{ \"textWidth\": 768 }");
        Touch(Path.Combine(_dir, "catalog_test_incomplete", "unet", "model.onnx"));
        Touch(Path.Combine(_dir, "hub", "x"));

        Assert.That(CheckpointCatalog.ScanFolder(_dir), Is.EqualTo(1));
        Assert.That(CheckpointCatalog.Repos["catalog_test_folder"], Is.EqualTo("local/catalog_test_folder"));
        Assert.That(CheckpointCatalog.Details["catalog_test_folder"], Is.EqualTo(new MlRuntime.CheckpointInfo(768, "unet/model.onnx_data")), "the data file is found, the width comes from checkpoint.json");
        Assert.That(CheckpointCatalog.Repos.ContainsKey("catalog_test_incomplete"), Is.False);
        Assert.That(CheckpointCatalog.RepoFolders["local/catalog_test_folder"], Is.EqualTo("catalog_test_folder"));
        Assert.That(CheckpointCatalog.ScanFolder(_dir), Is.EqualTo(0), "a second scan adds nothing");
    }

    [Test]
    public void Register_refuses_empty_names_and_silly_widths()
    {
        Assert.That(CheckpointCatalog.Register("", "a/b"), Is.False);
        Assert.That(CheckpointCatalog.Register("catalog_test_c", ""), Is.False);
        Assert.That(CheckpointCatalog.Register("catalog_test_c", "a/b", textWidth: 10), Is.False);
        Assert.That(CheckpointCatalog.Register("catalog_test_c", "a/b", 768), Is.True);
    }

    [Test]
    public void Replacing_a_combo_changes_its_choices_and_the_old_widget_comes_back()
    {
        var registry = new NodeRegistry().AddAssembly(typeof(CheckpointLoaderSimple).Assembly);
        var old = registry.GetSchema("CheckpointLoaderSimple")!.FindInput("ckpt_name")!.Widget!;
        var returned = registry.ReplaceWidget("CheckpointLoaderSimple", "ckpt_name", old with { Options = ["only_this"], OptionsProvider = null });
        Assert.That(returned, Is.SameAs(old));
        Assert.That(registry.GetSchema("CheckpointLoaderSimple")!.FindInput("ckpt_name")!.Widget!.ResolveOptions(), Is.EqualTo(new[] { "only_this" }));
        Assert.That(registry.TryGet("CheckpointLoaderSimple", out var node) && node.Schema.Outputs.Count == 3, Is.True, "the rest of the node is untouched");
        registry.ReplaceWidget("CheckpointLoaderSimple", "ckpt_name", returned!);
        Assert.That(registry.GetSchema("CheckpointLoaderSimple")!.FindInput("ckpt_name")!.Widget!.ResolveOptions(), Does.Contain("sd_turbo"));
        Assert.That(registry.ReplaceWidget("Nope", "x", old), Is.Null);
        Assert.That(registry.ReplaceWidget("CheckpointLoaderSimple", "nope", old), Is.Null);
    }
}
