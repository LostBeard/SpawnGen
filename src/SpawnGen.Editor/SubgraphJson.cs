using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor;

/// <summary>
/// The subgraph definitions inside a workflow file (<c>definitions.subgraphs</c>), in ComfyUI's format: each definition has its own nodes and links
/// (links as objects), and its inputs and outputs are the links from node -10 and to node -20. An instance is a node whose type is the definition's id.
/// SpawnGen's earlier layout (ports with "targets") is still read.
/// </summary>
public static class SubgraphJson
{
    /// <summary>The virtual node a subgraph's inputs come from, and the one its outputs go to (ComfyUI's ids).</summary>
    public const int InputNodeId = -10, OutputNodeId = -20;

    public static JsonArray Write(IEnumerable<SubgraphDefinition> definitions, NodeRegistry registry)
    {
        var array = new JsonArray();
        foreach (var def in definitions) array.Add(Write(def, registry));
        return array;
    }

    static JsonObject Write(SubgraphDefinition def, NodeRegistry registry)
    {
        // The inner graph with the boundary as links: from node -10 (output slot = input port) and to node -20 (the port index in ToInput).
        var graph = new EditorGraph();
        foreach (var n in def.Nodes) graph.AddNodeRaw(n);
        foreach (var l in def.Links) graph.AddLinkRaw(l);
        int linkId = def.Links.Select(l => l.Id).DefaultIfEmpty(0).Max() + 1;
        var inputLinks = def.Inputs.Select(_ => new List<int>()).ToList();
        var outputLinks = def.Outputs.Select(_ => new List<int>()).ToList();
        for (int p = 0; p < def.Inputs.Count; p++)
            foreach (var (node, input) in def.Inputs[p].Targets)
            {
                inputLinks[p].Add(linkId);
                graph.AddLinkRaw(new EditorLink { Id = linkId++, FromNode = InputNodeId, FromOutput = p, ToNode = node, ToInput = input });
            }
        for (int p = 0; p < def.Outputs.Count; p++)
        {
            outputLinks[p].Add(linkId);
            graph.AddLinkRaw(new EditorLink { Id = linkId++, FromNode = def.Outputs[p].Source.Node, FromOutput = def.Outputs[p].Source.Output, ToNode = OutputNodeId, ToInput = p.ToString(System.Globalization.CultureInfo.InvariantCulture) });
        }

        var nodes = ComfyWorkflow.WriteNodes(graph, registry, n => registry.GetSchema(n.ClassType) is { } s ? NodeLayout.Compute(n, s, graph).Bounds : new Rect(n.X, n.Y, 200, 100));
        var links = ComfyWorkflow.WriteLinks(graph, registry, p => def.Inputs.ElementAtOrDefault(p)?.TypeName ?? "*")
            .Select(l => (JsonNode)new JsonObject { ["id"] = l.Id, ["origin_id"] = l.From, ["origin_slot"] = l.FromSlot, ["target_id"] = l.To, ["target_slot"] = l.ToSlot, ["type"] = l.Type }).ToArray();

        double left = def.Nodes.Select(n => n.X).DefaultIfEmpty(0).Min() - 200, right = def.Nodes.Select(n => n.X + (n.Width ?? 220)).DefaultIfEmpty(0).Max() + 80;
        double top = def.Nodes.Select(n => n.Y).DefaultIfEmpty(0).Min();
        JsonArray Ports(List<SubgraphPort> ports, List<List<int>> ids, string side, double x) => new(ports.Select((p, i) => (JsonNode)new JsonObject
        {
            ["id"] = PortId(def.Id, side, i), ["name"] = p.Name, ["type"] = p.TypeName,
            ["linkIds"] = new JsonArray(ids[i].Select(id => (JsonNode)id).ToArray()), ["localized_name"] = p.Name, ["pos"] = new JsonArray(x, top + 20 + 20 * i),
        }).ToArray());

        return new JsonObject
        {
            ["id"] = def.Id, ["version"] = 1,
            ["state"] = new JsonObject { ["lastGroupId"] = 0, ["lastNodeId"] = def.Nodes.Select(n => n.Id).DefaultIfEmpty(0).Max(), ["lastLinkId"] = linkId - 1, ["lastRerouteId"] = 0 },
            ["revision"] = 0, ["config"] = new JsonObject(), ["name"] = def.Name,
            ["inputNode"] = new JsonObject { ["id"] = InputNodeId, ["bounding"] = new JsonArray(left, top, 120, 60) },
            ["outputNode"] = new JsonObject { ["id"] = OutputNodeId, ["bounding"] = new JsonArray(right, top, 120, 60) },
            ["inputs"] = Ports(def.Inputs, inputLinks, "in", left + 100),
            ["outputs"] = Ports(def.Outputs, outputLinks, "out", right + 20),
            ["widgets"] = new JsonArray(), ["nodes"] = nodes, ["groups"] = new JsonArray(), ["links"] = new JsonArray(links), ["extra"] = new JsonObject(),
        };
    }

    /// <summary>A stable id for a port (ComfyUI gives each one a uuid; ours are made from the definition and the slot so saving twice gives the same file).</summary>
    static string PortId(string definitionId, string side, int index) =>
        new Guid(MD5.HashData(Encoding.UTF8.GetBytes($"{definitionId}/{side}/{index}"))).ToString();

    /// <summary>
    /// Reads the definitions and registers them in the <see cref="SubgraphLibrary"/>. Every definition's ports are known before any inner node is
    /// read, so instances nested inside definitions get their sockets. Returns the ids read; entries that are not understood are skipped with a warning.
    /// </summary>
    public static IReadOnlySet<string> Register(JsonArray? array, NodeRegistry registry, List<string>? warnings = null)
    {
        warnings ??= [];
        var ids = new HashSet<string>(StringComparer.Ordinal);
        if (array is null) return ids;
        var items = array.OfType<JsonObject>().Where(i => i["id"] is JsonValue).ToList();

        // 1. Shells: name and ports, so instance types resolve while the inner nodes are read.
        var shells = new List<(JsonObject Json, SubgraphDefinition Def)>();
        foreach (var item in items)
        {
            try
            {
                var def = new SubgraphDefinition { Id = (string)item["id"]!, Name = (string?)item["name"] ?? "Subgraph" };
                foreach (var p in (item["inputs"] as JsonArray ?? []).OfType<JsonObject>()) def.Inputs.Add(new SubgraphPort { Name = (string)p["name"]!, TypeName = (string?)p["type"] ?? "*" });
                foreach (var p in (item["outputs"] as JsonArray ?? []).OfType<JsonObject>()) def.Outputs.Add(new SubgraphPort { Name = (string)p["name"]!, TypeName = (string?)p["type"] ?? "*" });
                SubgraphLibrary.Register(def);
                ids.Add(def.Id);
                shells.Add((item, def));
            }
            catch (Exception ex) when (ex is InvalidCastException or FormatException or NullReferenceException or InvalidOperationException)
            {
                warnings.Add($"A subgraph definition could not be read ({ex.Message}).");
            }
        }

        // 2. Inner nodes and links.
        foreach (var (item, def) in shells)
        {
            try
            {
                if (IsOldLayout(item)) ReadOldLayout(item, def);
                else ReadComfyLayout(item, def, registry, ids, warnings);
            }
            catch (Exception ex) when (ex is InvalidCastException or FormatException or NullReferenceException or InvalidOperationException or KeyNotFoundException)
            {
                warnings.Add($"Subgraph '{def.Name}' could not be read completely ({ex.Message}).");
            }
        }
        return ids;
    }

    static void ReadComfyLayout(JsonObject item, SubgraphDefinition def, NodeRegistry registry, IReadOnlySet<string> ids, List<string> warnings)
    {
        var graph = new EditorGraph();
        var nodeJson = ComfyWorkflow.ReadNodes(item["nodes"] as JsonArray ?? [], registry, graph, ids, warnings);
        def.Nodes.AddRange(graph.Nodes);
        foreach (var l in ComfyWorkflow.Links(item["links"]))
        {
            if (l.From == InputNodeId)
            {
                if (l.FromSlot < def.Inputs.Count && nodeJson.TryGetValue(l.To, out var target) && ComfyWorkflow.InputName(target, l.ToSlot, registry) is { } name)
                    def.Inputs[l.FromSlot].Targets.Add((l.To, name));
            }
            else if (l.To == OutputNodeId)
            {
                if (l.ToSlot < def.Outputs.Count) def.Outputs[l.ToSlot].Source = (l.From, l.FromSlot);
            }
            else if (ComfyWorkflow.ReadLink(l, nodeJson, registry, warnings) is { } link) def.Links.Add(link);
        }
    }

    /// <summary>SpawnGen's layout before it wrote ComfyUI's: ports carry "targets", links are {from, from_output, to, to_input}, widgets are by name.</summary>
    static bool IsOldLayout(JsonObject item) =>
        (item["inputs"] as JsonArray)?.OfType<JsonObject>().Any(p => p["targets"] is not null) == true
        || (item["outputs"] as JsonArray)?.OfType<JsonObject>().Any(p => p["node"] is not null) == true
        || (item["links"] as JsonArray)?.OfType<JsonObject>().Any(l => l["from"] is not null) == true;

    static void ReadOldLayout(JsonObject item, SubgraphDefinition def)
    {
        foreach (var n in (item["nodes"] as JsonArray ?? []).OfType<JsonObject>())
        {
            var node = new EditorNode
            {
                Id = (int)n["id"]!, ClassType = (string)n["type"]!, Title = (string?)n["title"], Color = (string?)n["color"], BgColor = (string?)n["bgcolor"],
                X = (double?)n["pos"]?[0] ?? 0, Y = (double?)n["pos"]?[1] ?? 0, Collapsed = (bool?)n["collapsed"] ?? false, Width = (double?)n["width"],
                Mode = Enum.IsDefined(typeof(NodeMode), (int?)n["mode"] ?? 0) ? (NodeMode)((int?)n["mode"] ?? 0) : NodeMode.Always,
            };
            foreach (var (name, value) in n["widgets"] as JsonObject ?? [])
                node.Widgets[name] = value switch
                {
                    null => null,
                    JsonValue v when v.TryGetValue<bool>(out var b) => b,
                    JsonValue v when v.TryGetValue<long>(out var l) => l,
                    JsonValue v when v.TryGetValue<double>(out var d) => d,
                    JsonValue v when v.TryGetValue<string>(out var s) => s,
                    _ => null,
                };
            def.Nodes.Add(node);
        }
        int linkId = 1;
        foreach (var l in (item["links"] as JsonArray ?? []).OfType<JsonObject>())
            def.Links.Add(new EditorLink { Id = linkId++, FromNode = (int)l["from"]!, FromOutput = (int)l["from_output"]!, ToNode = (int)l["to"]!, ToInput = (string)l["to_input"]! });
        var inputs = (item["inputs"] as JsonArray ?? []).OfType<JsonObject>().ToList();
        for (int i = 0; i < inputs.Count && i < def.Inputs.Count; i++)
            foreach (var t in (inputs[i]["targets"] as JsonArray ?? []).OfType<JsonObject>()) def.Inputs[i].Targets.Add(((int)t["node"]!, (string)t["input"]!));
        var outputs = (item["outputs"] as JsonArray ?? []).OfType<JsonObject>().ToList();
        for (int i = 0; i < outputs.Count && i < def.Outputs.Count; i++) def.Outputs[i].Source = ((int)outputs[i]["node"]!, (int)outputs[i]["output"]!);
    }
}
