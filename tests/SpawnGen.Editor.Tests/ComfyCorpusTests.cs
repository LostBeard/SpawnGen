using System.Text.Json.Nodes;
using SpawnGen.Editor;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor.Tests;

/// <summary>
/// Opens every workflow in a folder of ComfyUI's own files (its blueprints, its templates) and reports what does not come through. The files are
/// ComfyUI's (GPL), so they are not in this repository: point SPAWNGEN_COMFY_WORKFLOWS at a folder of them to run this.
/// </summary>
[Explicit("Needs SPAWNGEN_COMFY_WORKFLOWS: a folder of ComfyUI workflow files")]
public class ComfyCorpusTests
{
    static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(SpawnGen.Nodes.Core.SaveImage).Assembly).AddAssembly(typeof(SpawnGen.Nodes.ML.KSampler).Assembly);

    [Test]
    public void Every_workflow_opens_and_saves_back_the_same_graph()
    {
        var folder = Environment.GetEnvironmentVariable("SPAWNGEN_COMFY_WORKFLOWS");
        if (string.IsNullOrEmpty(folder) || !Directory.Exists(folder)) Assert.Ignore("SPAWNGEN_COMFY_WORKFLOWS is not set.");
        var failures = new List<string>();
        var missing = new Dictionary<string, int>();
        int files = 0, droppedLinks = 0;
        foreach (var file in Directory.EnumerateFiles(folder, "*.json", SearchOption.AllDirectories).OrderBy(f => f))
        {
            var text = File.ReadAllText(file);
            if (JsonNode.Parse(text) is not JsonObject root || root["nodes"] is not JsonArray) continue;
            files++;
            try
            {
                var doc = new EditorDocument(Registry());
                var warnings = ComfyWorkflow.Load(text, doc);
                foreach (var w in warnings)
                {
                    if (w.Contains("is not installed")) { var t = w[(w.IndexOf('\'') + 1)..]; t = t[..t.IndexOf('\'')]; missing[t] = missing.GetValueOrDefault(t) + 1; }
                    if (w.StartsWith("Link #")) droppedLinks++;
                }
                // Saving and opening again keeps every node and link.
                var again = new EditorDocument(Registry());
                ComfyWorkflow.Load(ComfyWorkflow.Save(doc), again);
                if (again.Graph.Nodes.Count != doc.Graph.Nodes.Count || again.Graph.Links.Count != doc.Graph.Links.Count)
                    failures.Add($"{Path.GetFileName(file)}: round trip changed {doc.Graph.Nodes.Count}/{doc.Graph.Links.Count} to {again.Graph.Nodes.Count}/{again.Graph.Links.Count}");
                foreach (var n in doc.Graph.Nodes.Where(n => n.ClassType.StartsWith(SubgraphDefinition.Prefix)))
                    if (SubgraphLibrary.Get(n.ClassType) is not { } def) failures.Add($"{Path.GetFileName(file)}: subgraph {n.ClassType} not registered");
                    else if (def.Nodes.Count == 0) failures.Add($"{Path.GetFileName(file)}: subgraph '{def.Name}' has no nodes");
            }
            catch (Exception ex)
            {
                failures.Add($"{Path.GetFileName(file)}: {ex.GetType().Name}: {ex.Message}");
            }
        }
        TestContext.Out.WriteLine($"{files} workflows, {failures.Count} failures, {droppedLinks} dropped links");
        foreach (var f in failures) TestContext.Out.WriteLine("  FAIL " + f);
        TestContext.Out.WriteLine("Node types not installed (workflows using them):");
        foreach (var (t, c) in missing.OrderByDescending(kv => kv.Value)) TestContext.Out.WriteLine($"  {c,4}  {t}");
        Assert.That(failures, Is.Empty);
    }
}
