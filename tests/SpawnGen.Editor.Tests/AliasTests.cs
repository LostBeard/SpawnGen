using System.Text.Json.Nodes;
using SpawnGen.Editor;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor.Tests;

/// <summary>Node types and inputs renamed to ComfyUI's ids keep their earlier names as aliases, so workflows saved before the rename still open.</summary>
public class AliasTests
{
    static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(SpawnGen.Nodes.Core.SaveImage).Assembly).AddAssembly(typeof(SpawnGen.Nodes.ML.KSampler).Assembly);

    [Test]
    public void ComfyUI_ids_are_used_and_the_old_ones_still_resolve()
    {
        var r = Registry();
        Assert.That(r.GetSchema("RegexMatch")?.Id, Is.EqualTo("RegexMatch"));
        Assert.That(r.GetSchema("RegexMatchNode")?.Id, Is.EqualTo("RegexMatch"));
        Assert.That(r.CanonicalId("LatentBatchNode"), Is.EqualTo("LatentBatch"));
        Assert.That(r.CanonicalId("KSampler"), Is.EqualTo("KSampler"));
        Assert.That(r.GetSchema("ConditioningCombine")!.Inputs.Select(i => i.Name), Is.EqualTo(new[] { "conditioning_1", "conditioning_2" }));
        Assert.That(r.GetSchema("ConditioningCombine")!.CanonicalInput("conditioning2"), Is.EqualTo("conditioning_2"));
    }

    [Test]
    public void A_workflow_with_old_names_opens_with_the_new_ones_and_its_links_connected()
    {
        // Saved before the rename: a Regex Match node under its old id, and a combine fed through the old input names.
        var (doc, ids) = Fixtures.Txt2Img();
        var combine = doc.AddNode("ConditioningCombine", 600, 300);
        doc.Connect(ids["pos"], 0, combine.Id, "conditioning_1", out _);
        doc.Connect(ids["neg"], 0, combine.Id, "conditioning_2", out _);
        var regex = doc.AddNode("RegexMatch", 600, 600);
        var json = JsonNode.Parse(ComfyWorkflow.Save(doc))!;
        foreach (var n in json["nodes"]!.AsArray())
        {
            if ((int)n!["id"]! == regex.Id) n["type"] = "RegexMatchNode";
            if ((int)n["id"]! == combine.Id)
                foreach (var input in n["inputs"]!.AsArray()) input!["name"] = ((string)input["name"]!).Replace("_", "");
        }

        var reopened = new EditorDocument(Registry());
        ComfyWorkflow.Load(json.ToJsonString(), reopened);
        Assert.That(reopened.Graph.GetNode(regex.Id).ClassType, Is.EqualTo("RegexMatch"));
        Assert.That(reopened.Graph.LinkInto(combine.Id, "conditioning_1")?.FromNode, Is.EqualTo(ids["pos"]));
        Assert.That(reopened.Graph.LinkInto(combine.Id, "conditioning_2")?.FromNode, Is.EqualTo(ids["neg"]));
    }

    [Test]
    public void An_API_prompt_with_old_names_imports_with_the_new_ones()
    {
        const string prompt = """
        {
          "1": { "class_type": "RegexReplaceNode", "inputs": { "string": "a1b2", "regex_pattern": "\\d", "replace": "#" } },
          "2": { "class_type": "CLIPTextEncode", "inputs": { "text": "x", "clip": ["9", 1] } },
          "3": { "class_type": "ConditioningCombine", "inputs": { "conditioning1": ["2", 0], "conditioning2": ["2", 0] } },
          "9": { "class_type": "CheckpointLoaderSimple", "inputs": { "ckpt_name": "sd_turbo" } }
        }
        """;
        var doc = new EditorDocument(Registry());
        ApiPromptImport.Load(prompt, doc);
        var replace = doc.Graph.Nodes.Single(n => n.ClassType == "RegexReplace");
        Assert.That(replace.Widgets["regex_pattern"], Is.EqualTo("\\d"));
        var combine = doc.Graph.Nodes.Single(n => n.ClassType == "ConditioningCombine");
        Assert.That(doc.Graph.LinkInto(combine.Id, "conditioning_1"), Is.Not.Null);
        Assert.That(doc.Graph.LinkInto(combine.Id, "conditioning_2"), Is.Not.Null);
    }
}
