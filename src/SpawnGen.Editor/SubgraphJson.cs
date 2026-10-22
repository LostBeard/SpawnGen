using System.Text.Json.Nodes;

namespace SpawnGen.Editor;

/// <summary>
/// The subgraph definitions inside a workflow file (<c>definitions.subgraphs</c>). This is SpawnGen's own layout, not ComfyUI's: ComfyUI ignores the block and shows
/// the instances as missing nodes.
/// </summary>
public static class SubgraphJson
{
    public static JsonArray Write(IEnumerable<SubgraphDefinition> definitions)
    {
        var array = new JsonArray();
        foreach (var def in definitions)
        {
            var nodes = new JsonArray();
            foreach (var n in def.Nodes)
            {
                var widgets = new JsonObject();
                foreach (var (name, value) in n.Widgets)
                    widgets[name] = value switch
                    {
                        null => null,
                        long l => JsonValue.Create(l),
                        int i => JsonValue.Create((long)i),
                        double d => JsonValue.Create(d),
                        float f => JsonValue.Create((double)f),
                        bool b => JsonValue.Create(b),
                        _ => JsonValue.Create(value.ToString()),
                    };
                var node = new JsonObject { ["id"] = n.Id, ["type"] = n.ClassType, ["pos"] = new JsonArray(n.X, n.Y), ["mode"] = (int)n.Mode, ["widgets"] = widgets };
                if (n.Title is not null) node["title"] = n.Title;
                if (n.Color is not null) node["color"] = n.Color;
                if (n.BgColor is not null) node["bgcolor"] = n.BgColor;
                if (n.Collapsed) node["collapsed"] = true;
                if (n.Width is { } w) node["width"] = w;
                nodes.Add(node);
            }
            var links = new JsonArray(def.Links.Select(l => (JsonNode)new JsonObject { ["from"] = l.FromNode, ["from_output"] = l.FromOutput, ["to"] = l.ToNode, ["to_input"] = l.ToInput }).ToArray());
            var inputs = new JsonArray(def.Inputs.Select(p => (JsonNode)new JsonObject
            {
                ["name"] = p.Name, ["type"] = p.TypeName,
                ["targets"] = new JsonArray(p.Targets.Select(t => (JsonNode)new JsonObject { ["node"] = t.Node, ["input"] = t.Input }).ToArray()),
            }).ToArray());
            var outputs = new JsonArray(def.Outputs.Select(p => (JsonNode)new JsonObject { ["name"] = p.Name, ["type"] = p.TypeName, ["node"] = p.Source.Node, ["output"] = p.Source.Output }).ToArray());
            array.Add(new JsonObject { ["id"] = def.Id, ["name"] = def.Name, ["nodes"] = nodes, ["links"] = links, ["inputs"] = inputs, ["outputs"] = outputs });
        }
        return array;
    }

    /// <summary>Reads definitions and registers them in the <see cref="SubgraphLibrary"/>; entries that are not understood are skipped.</summary>
    public static int Register(JsonArray? array)
    {
        if (array is null) return 0;
        int count = 0;
        foreach (var item in array.OfType<JsonObject>())
        {
            try
            {
                var def = new SubgraphDefinition { Id = (string)item["id"]!, Name = (string?)item["name"] ?? "Subgraph" };
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
                foreach (var p in (item["inputs"] as JsonArray ?? []).OfType<JsonObject>())
                {
                    var port = new SubgraphPort { Name = (string)p["name"]!, TypeName = (string?)p["type"] ?? "*" };
                    foreach (var t in (p["targets"] as JsonArray ?? []).OfType<JsonObject>()) port.Targets.Add(((int)t["node"]!, (string)t["input"]!));
                    def.Inputs.Add(port);
                }
                foreach (var p in (item["outputs"] as JsonArray ?? []).OfType<JsonObject>())
                    def.Outputs.Add(new SubgraphPort { Name = (string)p["name"]!, TypeName = (string?)p["type"] ?? "*", Source = ((int)p["node"]!, (int)p["output"]!) });
                SubgraphLibrary.Register(def);
                count++;
            }
            catch (Exception ex) when (ex is InvalidCastException or FormatException or NullReferenceException or InvalidOperationException) { }
        }
        return count;
    }
}
