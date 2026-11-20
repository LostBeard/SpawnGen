using System.Text.Json.Nodes;
using SpawnGen.Editor;
using SpawnGen.Engine.Execution;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Editor.Tests;

/// <summary>The example workflows the Studio offers (src/SpawnGen.Studio/wwwroot/examples): each must open without warnings and pass validation.</summary>
public class ExamplesTests
{
    static string ExamplesFolder()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "SpawnGen.slnx"))) dir = dir.Parent;
        return Path.Combine(dir!.FullName, "src", "SpawnGen.Studio", "wwwroot", "examples");
    }

    static IEnumerable<string> Files() => JsonNode.Parse(File.ReadAllText(Path.Combine(ExamplesFolder(), "index.json")))!.AsArray().Select(e => (string)e!["file"]!);

    [Test]
    public void The_index_lists_every_file_in_the_folder_with_a_name_and_a_note()
    {
        var index = JsonNode.Parse(File.ReadAllText(Path.Combine(ExamplesFolder(), "index.json")))!.AsArray();
        Assert.That(index, Has.Count.GreaterThanOrEqualTo(10));
        foreach (var e in index)
        {
            Assert.That((string?)e!["name"], Is.Not.Null.And.Not.Empty);
            Assert.That((string?)e["note"], Is.Not.Null.And.Not.Empty);
            Assert.That(File.Exists(Path.Combine(ExamplesFolder(), (string)e["file"]!)), Is.True, (string)e["file"]!);
        }
        var listed = index.Select(e => (string)e!["file"]!).ToHashSet();
        foreach (var f in Directory.GetFiles(ExamplesFolder(), "*.json").Select(Path.GetFileName).Where(n => n != "index.json"))
            Assert.That(listed, Does.Contain(f), $"{f} is not in the index");
    }

    /// <summary>The pictures the browser and server engines put in the inputs (BrowserEngine.ExamplePictures, ServerApp.CopyExamplePicture).</summary>
    static readonly string[] ShippedPictures = ["example.png", "example_inpaint.png"];

    [TestCaseSource(nameof(Files))]
    public void Every_example_picture_is_one_the_site_ships(string file)
    {
        // A first-time visitor has only the shipped pictures: an example naming any other file fails on its first Queue.
        var prompt = JsonNode.Parse(File.ReadAllText(Path.Combine(ExamplesFolder(), file)))!.AsObject();
        foreach (var (_, node) in prompt)
            if ((string?)node!["class_type"] is "LoadImage" or "LoadImageMask")
            {
                var name = (string)node["inputs"]!["image"]!;
                Assert.That(ShippedPictures, Does.Contain(name), $"{file} loads {name}");
                Assert.That(File.Exists(Path.Combine(ExamplesFolder(), name)), Is.True, $"examples/{name} is shipped");
                // An example that uses the MASK output needs a picture with a mask, or it runs and redraws nothing (the Inpainting example
                // on example.png did that until 2026-11-19).
                var id = prompt.First(kv => kv.Value == node).Key;
                bool maskUsed = prompt.Any(kv => kv.Value!["inputs"]!.AsObject().Any(i => i.Value is JsonArray link && (string?)link[0] == id && (int)link[1]! == 1));
                if (maskUsed)
                {
                    ImageDecoder.Decode(File.ReadAllBytes(Path.Combine(ExamplesFolder(), name)), out var alpha);
                    Assert.That(alpha is not null && alpha.Any(a => a < 0.5f), Is.True, $"{file} uses {name}'s mask, which is empty");
                }
            }
    }

    [Test]
    public void The_inpainting_example_picture_has_a_mask()
    {
        // LoadImage's mask is 1 - alpha: the coat is transparent, the rest opaque, so the Inpainting example redraws something out of the box.
        ImageDecoder.Decode(File.ReadAllBytes(Path.Combine(ExamplesFolder(), "example_inpaint.png")), out var alpha);
        Assert.That(alpha, Is.Not.Null, "the picture has an alpha channel");
        double masked = alpha!.Count(a => a < 0.5f) / (double)alpha!.Length;
        Assert.That(masked, Is.InRange(0.05, 0.3), "a coat-sized part of the picture is masked");
        Assert.That(alpha.Count(a => a > 0f && a < 1f), Is.Zero, "the mask is hard-edged (grow_mask_by softens it)");
    }

    [TestCaseSource(nameof(Files))]
    public void Every_example_opens_without_warnings_and_validates(string file)
    {
        var json = File.ReadAllText(Path.Combine(ExamplesFolder(), file));
        // The examples that start from a picture name a shipped example picture: make them exist for the check, as the engines seed them.
        var inputs = new MemoryInputStore();
        foreach (var name in ShippedPictures) inputs.SaveAsync(name, PngWriter.Encode(new SpawnGen.Engine.Types.ImageBatch(1, 8, 8, 3))).Wait();
        var previous = InputFiles.Store;
        InputFiles.Store = inputs;
        try
        {
            var registry = EditorDocumentFixtures.Registry();
            var doc = new EditorDocument(registry);
            var warnings = ApiPromptImport.Load(json, doc);
            Assert.That(warnings, Is.Empty, string.Join("; ", warnings));
            Assert.That(doc.Graph.Nodes.Count, Is.GreaterThanOrEqualTo(2));
            var compiled = GraphCompiler.Compile(doc.Graph, registry);
            var result = new PromptExecutor(registry).Validate(compiled.Graph);
            Assert.That(result.Errors, Is.Empty, string.Join("; ", result.Errors.Select(e => e.Message)));
        }
        finally { InputFiles.Store = previous; }
    }

    [TestCaseSource(nameof(Files))]
    public void No_two_nodes_of_an_example_overlap_once_its_pictures_show(string file)
    {
        // The page lays an example out, then shows each Load Image's chosen picture under its widgets (Studio.SyncPicturePreviews): the
        // Upscale example's Load Image grew over Load Upscale Model below it (live site, 2026-11-20). Give each node the picture it will show,
        // at the shipped picture's real size, and each output node taking an IMAGE its result picture after a run (Remove the background's
        // first Save Image ended up under the second); the examples' results are square or wider than tall, so square is the tallest. No two
        // nodes may touch.
        var registry = EditorDocumentFixtures.Registry();
        var doc = new EditorDocument(registry);
        ApiPromptImport.Load(File.ReadAllText(Path.Combine(ExamplesFolder(), file)), doc);
        foreach (var node in doc.Graph.Nodes)
            if (node.Widgets.GetValueOrDefault("image") is string name && ShippedPictures.Contains(name))
            {
                var picture = ImageDecoder.Decode(File.ReadAllBytes(Path.Combine(ExamplesFolder(), name)), out _);
                node.Previews = [new NodePreview(name, picture.Width, picture.Height)];
            }
            else if (registry.GetSchema(node.ClassType) is { IsOutputNode: true } schema && schema.Inputs.Any(i => i.Type.Name == "IMAGE"))
                node.Previews = [new NodePreview("result", 512, 512)];
        var hits = new HitTester(doc);
        var nodes = doc.Graph.Nodes.ToList();
        for (int i = 0; i < nodes.Count; i++)
            for (int j = i + 1; j < nodes.Count; j++)
                Assert.That(hits.Geometry(nodes[i]).Bounds.Intersects(hits.Geometry(nodes[j]).Bounds), Is.False,
                    $"{file}: {nodes[i].ClassType} and {nodes[j].ClassType} overlap");
    }
}

/// <summary>The registry the editor tests use (the Fixtures class of the document tests).</summary>
static class EditorDocumentFixtures
{
    public static SpawnGen.Engine.Nodes.NodeRegistry Registry() => Fixtures.Registry();
}
