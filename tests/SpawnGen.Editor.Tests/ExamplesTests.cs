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

    [TestCaseSource(nameof(Files))]
    public void Every_example_opens_without_warnings_and_validates(string file)
    {
        var json = File.ReadAllText(Path.Combine(ExamplesFolder(), file));
        // The examples that start from a picture name "example.png": make it exist for the check, as an uploaded picture would.
        var inputs = new MemoryInputStore();
        inputs.SaveAsync("example.png", PngWriter.Encode(new SpawnGen.Engine.Types.ImageBatch(1, 8, 8, 3))).Wait();
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
}

/// <summary>The registry the editor tests use (the Fixtures class of the document tests).</summary>
static class EditorDocumentFixtures
{
    public static SpawnGen.Engine.Nodes.NodeRegistry Registry() => Fixtures.Registry();
}
