using System.Text.Json;
using System.Text.Json.Nodes;

namespace SpawnGen.Editor;

/// <summary>A saved group of nodes (the canvas menu's Save Selected as Template): its name and ComfyUI's template data (a JSON text).</summary>
public sealed record NodeTemplate(string Name, string Data);

/// <summary>
/// Node templates in ComfyUI's format (measured from its frontend): the user's <c>comfy.templates.json</c> is a list of <c>{name, data}</c>, where
/// <c>data</c> is the text of <c>{nodes, groups, reroutes, links, subgraphs}</c>: workflow nodes, links as <c>{id, origin_id, origin_slot, target_id,
/// target_slot, type}</c>, and the subgraph definitions the nodes use. Inserting one adds its nodes with the links among them.
/// </summary>
public static class NodeTemplates
{
    /// <summary>The file's name in the user's folder (beside comfy.settings.json).</summary>
    public const string FileName = "comfy.templates.json";

    static readonly JsonSerializerOptions Indented = new() { WriteIndented = true, IndentSize = 4 };

    /// <summary>Reads the templates file; a missing, damaged or odd file gives what can be read (nothing, at worst).</summary>
    public static List<NodeTemplate> Parse(string? text)
    {
        var list = new List<NodeTemplate>();
        if (string.IsNullOrWhiteSpace(text)) return list;
        try
        {
            if (JsonNode.Parse(text) is not JsonArray array) return list;
            foreach (var item in array.OfType<JsonObject>())
                if (item["name"] is JsonValue n && n.TryGetValue<string>(out var name) && item["data"] is JsonValue d && d.TryGetValue<string>(out var data))
                    list.Add(new NodeTemplate(name, data));
        }
        catch (JsonException) { }
        return list;
    }

    public static string Serialize(IEnumerable<NodeTemplate> templates) =>
        new JsonArray([.. templates.Select(t => (JsonNode)new JsonObject { ["name"] = t.Name, ["data"] = t.Data })]).ToJsonString(Indented);

    /// <summary>The template data for some nodes of a document: those nodes as the workflow saves them, the links among them, their subgraph definitions.</summary>
    public static string Build(EditorDocument doc, IEnumerable<int> nodeIds, Func<string, double>? measureText = null)
    {
        var ids = nodeIds.Where(id => doc.Graph.FindNode(id) is not null).ToHashSet();
        var part = new EditorDocument(doc.Registry);
        foreach (var id in ids) part.Graph.AddNodeRaw(doc.Graph.GetNode(id).Clone());
        foreach (var l in doc.Graph.Links.Where(l => ids.Contains(l.FromNode) && ids.Contains(l.ToNode)))
            part.Graph.AddLinkRaw(new EditorLink { Id = l.Id, FromNode = l.FromNode, FromOutput = l.FromOutput, ToNode = l.ToNode, ToInput = l.ToInput });
        var saved = JsonNode.Parse(ComfyWorkflow.Save(part, measureText))!.AsObject();
        var links = new JsonArray();
        foreach (var l in (saved["links"] as JsonArray ?? []).OfType<JsonArray>())
            links.Add(new JsonObject { ["id"] = l[0]?.DeepClone(), ["origin_id"] = l[1]?.DeepClone(), ["origin_slot"] = l[2]?.DeepClone(), ["target_id"] = l[3]?.DeepClone(), ["target_slot"] = l[4]?.DeepClone(), ["type"] = l[5]?.DeepClone() });
        return new JsonObject
        {
            ["nodes"] = saved["nodes"]?.DeepClone(), ["groups"] = new JsonArray(), ["reroutes"] = new JsonArray(), ["links"] = links,
            ["subgraphs"] = saved["definitions"]?["subgraphs"]?.DeepClone() ?? new JsonArray(),
        }.ToJsonString();
    }

    /// <summary>
    /// Opens template data as a document of its own (its subgraph definitions registered), ready to be pasted. Throws <see cref="JsonException"/> or
    /// <see cref="InvalidDataException"/> for data that is not a template.
    /// </summary>
    public static EditorDocument Open(string data, SpawnGen.Engine.Nodes.NodeRegistry registry, out IReadOnlyList<string> warnings)
    {
        if (JsonNode.Parse(data) is not JsonObject t || t["nodes"] is not JsonArray nodes) throw new InvalidDataException("A node template has a list of nodes.");
        var workflow = new JsonObject
        {
            ["nodes"] = nodes.DeepClone(), ["links"] = t["links"]?.DeepClone() ?? new JsonArray(), ["groups"] = new JsonArray(),
            ["version"] = 0.4,
        };
        if (t["subgraphs"] is JsonArray { Count: > 0 } subgraphs) workflow["definitions"] = new JsonObject { ["subgraphs"] = subgraphs.DeepClone() };
        var doc = new EditorDocument(registry);
        warnings = ComfyWorkflow.Load(workflow.ToJsonString(), doc);
        return doc;
    }
}
