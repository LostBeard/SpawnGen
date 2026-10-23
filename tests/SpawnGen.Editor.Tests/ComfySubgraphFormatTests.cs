using System.Text.Json.Nodes;
using SpawnGen.Editor;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor.Tests;

/// <summary>Subgraphs in ComfyUI's own file format: definitions with links from node -10 and to node -20, instances typed by the definition's id.</summary>
public class ComfySubgraphFormatTests
{
    const string SubgraphId = "6f1c2c1e-3a4b-4c5d-8e9f-0a1b2c3d4e5f";

    /// <summary>Load Image -> [subgraph: Invert Image] -> Preview Image, written the way ComfyUI's frontend saves it.</summary>
    static readonly string Workflow = $$"""
    {
      "last_node_id": "3", "last_link_id": 2,
      "nodes": [
        { "id": 1, "type": "LoadImage", "pos": [0, 0], "size": [200, 100], "flags": {}, "order": 0, "mode": 0, "inputs": [],
          "outputs": [{ "name": "IMAGE", "type": "IMAGE", "links": [1] }, { "name": "MASK", "type": "MASK", "links": null }],
          "properties": {}, "widgets_values": ["apple.png", "image"] },
        { "id": 2, "type": "{{SubgraphId}}", "pos": [300, 0], "size": [200, 60], "flags": {}, "order": 1, "mode": 0,
          "inputs": [{ "label": "picture", "name": "picture", "type": "IMAGE", "link": 1 }],
          "outputs": [{ "name": "inverted", "type": "IMAGE", "links": [2] }], "properties": { "proxyWidgets": [] }, "title": "Invert it" },
        { "id": 3, "type": "PreviewImage", "pos": [600, 0], "size": [200, 100], "flags": {}, "order": 2, "mode": 0,
          "inputs": [{ "name": "images", "type": "IMAGE", "link": 2 }], "outputs": [], "properties": {}, "widgets_values": [] }
      ],
      "links": [[1, 1, 0, 2, 0, "IMAGE"], [2, 2, 0, 3, 0, "IMAGE"]],
      "groups": [], "config": {}, "extra": {}, "version": 0.4,
      "definitions": { "subgraphs": [ {
        "id": "{{SubgraphId}}", "version": 1, "state": { "lastGroupId": 0, "lastNodeId": 10, "lastLinkId": 12, "lastRerouteId": 0 },
        "revision": 0, "config": {}, "name": "Inverter",
        "inputNode": { "id": -10, "bounding": [0, 0, 120, 60] }, "outputNode": { "id": -20, "bounding": [500, 0, 120, 60] },
        "inputs": [{ "id": "a", "name": "picture", "type": "IMAGE", "linkIds": [11], "pos": [100, 20] }],
        "outputs": [{ "id": "b", "name": "inverted", "type": "IMAGE", "linkIds": [12], "pos": [520, 20] }],
        "widgets": [],
        "nodes": [ { "id": 10, "type": "ImageInvert", "pos": [200, 0], "size": [200, 60], "flags": {}, "order": 0, "mode": 0,
                     "inputs": [{ "name": "image", "type": "IMAGE", "link": 11 }], "outputs": [{ "name": "IMAGE", "type": "IMAGE", "links": [12] }],
                     "properties": {}, "widgets_values": [] } ],
        "groups": [],
        "links": [ { "id": 11, "origin_id": -10, "origin_slot": 0, "target_id": 10, "target_slot": 0, "type": "IMAGE" },
                   { "id": 12, "origin_id": 10, "origin_slot": 0, "target_id": -20, "target_slot": 0, "type": "IMAGE" } ],
        "extra": {} } ] }
    }
    """;

    static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(SpawnGen.Nodes.Core.SaveImage).Assembly).AddAssembly(typeof(SpawnGen.Nodes.ML.KSampler).Assembly);

    [Test]
    public void A_ComfyUI_subgraph_opens_as_a_working_node()
    {
        var doc = new EditorDocument(Registry());
        var warnings = ComfyWorkflow.Load(Workflow, doc);
        Assert.That(warnings, Is.Empty);
        var instance = doc.Graph.GetNode(2);
        Assert.That(instance.ClassType, Is.EqualTo(SubgraphDefinition.Prefix + SubgraphId));
        var def = SubgraphLibrary.Get(instance.ClassType)!;
        Assert.That((def.Name, def.Inputs[0].Targets.Single(), def.Outputs[0].Source), Is.EqualTo(("Inverter", (10, "image"), (10, 0))));
        Assert.That(doc.Graph.LinkInto(2, "picture")?.FromNode, Is.EqualTo(1));

        var prompt = GraphCompiler.Compile(doc.Graph, doc.Registry).Graph;
        var invert = prompt.Nodes.Values.Single(n => n.ClassType == "ImageInvert");
        Assert.That(invert.Inputs["image"].NodeId, Is.EqualTo("1"), "the inner node reads what fed the subgraph");
        Assert.That(prompt.Nodes["3"].Inputs["images"].NodeId, Is.EqualTo(prompt.Nodes.Single(kv => kv.Value.ClassType == "ImageInvert").Key));
    }

    [Test]
    public void Saving_writes_ComfyUIs_format_and_it_opens_again()
    {
        var doc = new EditorDocument(Registry());
        ComfyWorkflow.Load(Workflow, doc);
        var saved = JsonNode.Parse(ComfyWorkflow.Save(doc))!;
        Assert.That((string?)saved["nodes"]!.AsArray().First(n => (int)n!["id"]! == 2)!["type"], Is.EqualTo(SubgraphId), "an instance is saved under the definition's id");
        var def = saved["definitions"]!["subgraphs"]![0]!;
        Assert.That(((int)def["inputNode"]!["id"]!, (int)def["outputNode"]!["id"]!), Is.EqualTo((-10, -20)));
        var links = def["links"]!.AsArray().Select(l => ((int)l!["origin_id"]!, (int)l["target_id"]!, (int)l["target_slot"]!)).ToList();
        Assert.That(links, Is.EquivalentTo(new[] { (-10, 10, 0), (10, -20, 0) }));
        Assert.That(def["inputs"]![0]!["linkIds"]!.AsArray().Count, Is.EqualTo(1));
        var inner = def["nodes"]![0]!;
        Assert.That((string?)inner["type"], Is.EqualTo("ImageInvert"));
        Assert.That((int?)inner["inputs"]![0]!["link"], Is.EqualTo((int)def["links"]!.AsArray().First(l => (int)l!["origin_id"]! == -10)!["id"]!), "the inner input names the boundary link");

        var again = new EditorDocument(Registry());
        Assert.That(ComfyWorkflow.Load(saved.ToJsonString(), again), Is.Empty);
        Assert.That(GraphCompiler.Compile(again.Graph, again.Registry).Graph.Nodes.Values.Count(n => n.ClassType == "ImageInvert"), Is.EqualTo(1));
    }
}
