using System.Text.Json;
using System.Text.Json.Nodes;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor;

/// <summary>
/// Reads and writes ComfyUI's workflow ("UI") JSON, so a workflow saved from ComfyUI opens here and the other way round. Positions, sizes, titles,
/// collapse and mute/bypass state, widget values and links are kept; node types this build does not know are kept as placeholders with their raw
/// widget values so saving does not lose them. Groups and the <c>extra</c> block pass through untouched.
/// </summary>
public static class ComfyWorkflow
{
    /// <summary>ComfyUI adds a "control after generate" combo after these integer widgets and stores its value in <c>widgets_values</c> too.</summary>
    static readonly HashSet<string> SeedWidgets = ["seed", "noise_seed"];
    static readonly HashSet<string> ControlValues = ["fixed", "increment", "decrement", "randomize"];

    public const string RawWidgetsKey = "__widgets_values";
    /// <summary>The node property ComfyUI's search-and-replace in file names uses (%KSampler.seed%); written with the node type when the node has none.</summary>
    public const string NameForSearchReplace = "Node name for S&R";
    static readonly JsonSerializerOptions Pretty = new() { WriteIndented = true };

    // ---- load ----

    /// <summary>Replaces the document's graph with the workflow. Returns warnings (unknown node types, values that did not fit a widget). Clears undo history.</summary>
    public static IReadOnlyList<string> Load(string json, EditorDocument doc)
    {
        var warnings = new List<string>();
        var root = JsonNode.Parse(json) as JsonObject ?? throw new InvalidDataException("Not a workflow: the JSON is not an object.");
        if (root["nodes"] is not JsonArray nodes) throw new InvalidDataException("Not a workflow: there is no \"nodes\" list (an API-format prompt cannot be opened as a workflow).");

        // Before the nodes: an instance needs its definition to have sockets.
        var subgraphIds = SubgraphJson.Register(root["definitions"]?["subgraphs"] as JsonArray, doc.Registry, warnings);
        doc.Reset(graph =>
        {
            var nodeJson = ReadNodes(nodes, doc.Registry, graph, subgraphIds, warnings);
            foreach (var link in Links(root["links"]))
                if (ReadLink(link, nodeJson, doc.Registry, warnings) is { } l) graph.AddLinkRaw(l);

            FoldPrimitiveNodes(doc, graph, warnings);
            ReadGroups(root["groups"] as JsonArray, graph);

            graph.NextNodeId = Math.Max(graph.NextNodeId, (Int(root["last_node_id"]) ?? 0) + 1);
            graph.NextLinkId = Math.Max(graph.NextLinkId, (Int(root["last_link_id"]) ?? 0) + 1);
        });
        doc.Extra = root["extra"] as JsonObject;
        return warnings;
    }

    /// <summary>The nodes of a workflow (or of a subgraph definition) into <paramref name="graph"/>; returns each node's JSON by id, for reading links.</summary>
    internal static Dictionary<int, JsonObject> ReadNodes(JsonArray nodes, NodeRegistry registry, EditorGraph graph, IReadOnlySet<string> subgraphIds, List<string> warnings)
    {
        var nodeJson = new Dictionary<int, JsonObject>();
        int z = 0;
        foreach (var n in nodes.OfType<JsonObject>().OrderBy(n => (int?)n["order"] ?? 0))
        {
            int id = Int(n["id"]) ?? throw new InvalidDataException("A node has no id.");
            var saved = (string?)n["type"] ?? "";
            // A subgraph instance is saved under its definition's id; a node saved under an earlier id opens as the current one.
            string type = subgraphIds.Contains(saved) ? SubgraphDefinition.Prefix + saved : registry.CanonicalId(saved);
            nodeJson[id] = n;
            var node = new EditorNode
            {
                Id = id, ClassType = type, Title = (string?)n["title"], Color = (string?)n["color"], BgColor = (string?)n["bgcolor"],
                X = Pair(n["pos"]).X, Y = Pair(n["pos"]).Y,
                Collapsed = (bool?)n["flags"]?["collapsed"] ?? false,
                Pinned = (bool?)n["flags"]?["pinned"] ?? false,
                Mode = Enum.IsDefined(typeof(NodeMode), (int?)n["mode"] ?? 0) ? (NodeMode)((int?)n["mode"] ?? 0) : NodeMode.Always,
                ZIndex = ++z,
                Properties = n["properties"] is JsonObject props ? (JsonObject)props.DeepClone() : [],
            };
            if (Pair(n["size"]).X is > 0 and var w) node.Width = w;

            var values = n["widgets_values"];
            if (registry.GetSchema(type) is { } schema) ReadWidgets(node, schema, values, warnings);
            else
            {
                warnings.Add($"Node #{id}: type '{type}' is not installed; it is kept as a placeholder.");
                if (values is not null) node.Widgets[RawWidgetsKey] = values.DeepClone();
            }
            graph.AddNodeRaw(node);
        }
        return nodeJson;
    }

    /// <summary>One saved link, from either form ComfyUI writes: <c>[id, from, from_slot, to, to_slot, type]</c> or <c>{id, origin_id, origin_slot, target_id, target_slot, type}</c>.</summary>
    internal readonly record struct SavedLink(int Id, int From, int FromSlot, int To, int ToSlot, string? Type);

    internal static IEnumerable<SavedLink> Links(JsonNode? links)
    {
        foreach (var l in (links as JsonArray) ?? [])
        {
            if (l is JsonArray a && a.Count >= 5 && Int(a[0]) is { } id && Int(a[1]) is { } from && Int(a[2]) is { } fromSlot && Int(a[3]) is { } to && Int(a[4]) is { } toSlot)
                yield return new(id, from, fromSlot, to, toSlot, a.Count > 5 ? a[5]?.ToString() : null);
            else if (l is JsonObject o && Int(o["id"]) is { } oid && Int(o["origin_id"]) is { } ofrom && Int(o["origin_slot"]) is { } oslot && Int(o["target_id"]) is { } oto && Int(o["target_slot"]) is { } otoSlot)
                yield return new(oid, ofrom, oslot, oto, otoSlot, o["type"]?.ToString());
        }
    }

    /// <summary>The editor link for a saved link between two read nodes: the target slot becomes the input's name. Null (with a warning) when it cannot be placed.</summary>
    internal static EditorLink? ReadLink(SavedLink l, Dictionary<int, JsonObject> nodeJson, NodeRegistry registry, List<string> warnings)
    {
        if (!nodeJson.TryGetValue(l.To, out var target) || !nodeJson.ContainsKey(l.From)) { warnings.Add($"Link #{l.Id} points at a missing node and was dropped."); return null; }
        var name = InputName(target, l.ToSlot, registry);
        if (name is null) { warnings.Add($"Link #{l.Id} has no matching input on node #{l.To} and was dropped."); return null; }
        return new EditorLink { Id = l.Id, FromNode = l.From, FromOutput = l.FromSlot, ToNode = l.To, ToInput = name };
    }

    /// <summary>The input a saved node lists at <paramref name="slot"/>, by its current name.</summary>
    internal static string? InputName(JsonObject node, int slot, NodeRegistry registry)
    {
        var name = (string?)(node["inputs"] as JsonArray)?.ElementAtOrDefault(slot)?["name"];
        var schema = registry.GetSchema(registry.CanonicalId((string?)node["type"] ?? ""));
        if (name is not null && schema is not null) name = schema.CanonicalInput(name);
        // ComfyUI's Reroute names its only input "": that is our node's single input.
        if (string.IsNullOrEmpty(name) && schema is { Inputs.Count: 1 }) name = schema.Inputs[0].Name;
        return name;
    }

    static void ReadGroups(JsonArray? groups, EditorGraph graph)
    {
        foreach (var g in groups?.OfType<JsonObject>() ?? [])
        {
            if (g["bounding"] is not JsonArray b || b.Count < 4) continue;
            var extra = new JsonObject();
            foreach (var (key, value) in g)
                if (key is not ("id" or "title" or "bounding" or "color" or "font_size")) extra[key] = value?.DeepClone();
            graph.AddGroupRaw(new EditorGroup
            {
                Id = (int?)g["id"] is { } id && graph.FindGroup(id) is null ? id : graph.NextGroupId,
                Title = (string?)g["title"] ?? "Group",
                X = (double)b[0]!, Y = (double)b[1]!, Width = Math.Max((double)b[2]!, EditorGroup.MinWidth), Height = Math.Max((double)b[3]!, EditorGroup.MinHeight),
                Color = (string?)g["color"] ?? (string?)g["groupcolor"] ?? EditorGroup.DefaultColor,
                FontSize = (double?)g["font_size"] ?? 24,
                Extra = extra.Count > 0 ? extra : null,
            });
        }
    }

    static JsonObject NodeFlags(EditorNode n)
    {
        var flags = new JsonObject();
        if (n.Collapsed) flags["collapsed"] = true;
        if (n.Pinned) flags["pinned"] = true;
        return flags;
    }

    static JsonArray WriteGroups(EditorGraph graph)
    {
        var list = new JsonArray();
        foreach (var g in graph.Groups)
        {
            var o = new JsonObject
            {
                ["id"] = g.Id, ["title"] = g.Title, ["bounding"] = new JsonArray(g.X, g.Y, g.Width, g.Height),
                ["color"] = g.Color, ["font_size"] = g.FontSize,
            };
            if (g.Extra is not null) foreach (var (key, value) in g.Extra) o[key] = value?.DeepClone();
            list.Add(o);
        }
        return list;
    }

    /// <summary>
    /// ComfyUI's legacy <c>PrimitiveNode</c> is a front-end device that turns into the widget it is linked to. Here its value is written into every widget it
    /// fed and the node is dropped (the Primitive Int / Float / String nodes are the real equivalents).
    /// </summary>
    static void FoldPrimitiveNodes(EditorDocument doc, EditorGraph graph, List<string> warnings)
    {
        foreach (var primitive in graph.Nodes.Where(n => n.ClassType == "PrimitiveNode").ToList())
        {
            var value = (primitive.Widgets.GetValueOrDefault(RawWidgetsKey) as JsonArray)?.FirstOrDefault();
            foreach (var link in graph.LinksFrom(primitive.Id).ToList())
            {
                if (graph.FindNode(link.ToNode) is { } target && doc.Registry.GetSchema(target.ClassType)?.FindInput(link.ToInput)?.Widget is { } spec && value is not null)
                    target.Widgets[link.ToInput] = Coerce(spec, value, target, link.ToInput, warnings);
                graph.RemoveLinkRaw(link.Id);
            }
            graph.RemoveNodeRaw(primitive.Id);
            warnings.Add($"Primitive node #{primitive.Id} was folded into the widgets it fed (its value is kept).");
        }
    }

    /// <summary>An integer that ComfyUI may have saved as a number or as text ("34"); null when there is none.</summary>
    internal static int? Int(JsonNode? node) => node switch
    {
        JsonValue v when v.TryGetValue<int>(out var i) => i,
        JsonValue v when v.TryGetValue<double>(out var d) && d == Math.Floor(d) && Math.Abs(d) < int.MaxValue => (int)d,
        JsonValue v when v.TryGetValue<string>(out var s) && int.TryParse(s, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var p) => p,
        _ => null,
    };

    /// <summary>ComfyUI saves a pair either as <c>[a, b]</c> or, in older files, as <c>{"0": a, "1": b}</c>.</summary>
    static (double X, double Y) Pair(JsonNode? node) => node switch
    {
        JsonArray a => ((double?)a.ElementAtOrDefault(0) ?? 0, (double?)a.ElementAtOrDefault(1) ?? 0),
        JsonObject o => ((double?)o["0"] ?? 0, (double?)o["1"] ?? 0),
        _ => (0, 0),
    };

    static void ReadWidgets(EditorNode node, NodeSchema schema, JsonNode? values, List<string> warnings)
    {
        var widgetInputs = schema.Inputs.Where(i => i.Widget is not null).ToList();
        if (values is JsonObject map) // some nodes (and newer frontends) store widget values by name
        {
            foreach (var w in widgetInputs)
                if (map[w.Name] is { } v) node.Widgets[w.Name] = Coerce(w.Widget!, v, node, w.Name, warnings);
            if (map[ControlAfterGenerate.Name] is JsonValue mode && mode.TryGetValue<string>(out var named) && ControlValues.Contains(named)) node.Widgets[ControlAfterGenerate.Name] = named;
            return;
        }
        var list = values as JsonArray ?? [];
        int pos = 0;
        // A dynamic combo's values follow it: only the chosen option's inputs are in the list (the walk reads the combo's value as it reaches it).
        foreach (var w in NodeLayout.OrderedInputs(schema, node.Widgets).Where(i => i.Widget is not null))
        {
            node.Widgets[w.Name] = w.Widget!.Default is { } d && pos >= list.Count ? d : node.Widgets.GetValueOrDefault(w.Name);
            if (pos >= list.Count) continue;
            node.Widgets[w.Name] = Coerce(w.Widget!, list[pos++], node, w.Name, warnings);
            // ComfyUI stores the "control after generate" mode right after a seed widget.
            if (w.Widget!.Kind == WidgetKind.Int && SeedWidgets.Contains(w.Name) && pos < list.Count
                && list[pos] is JsonValue cv && cv.TryGetValue<string>(out var s) && ControlValues.Contains(s))
            {
                node.Widgets[ControlAfterGenerate.Name] = s;
                pos++;
            }
        }
    }

    static object? Coerce(WidgetSpec spec, JsonNode? value, EditorNode node, string name, List<string> warnings)
    {
        if (value is null) return spec.Default;
        try
        {
            return spec.Kind switch
            {
                WidgetKind.Int => value is JsonValue iv && iv.TryGetValue<double>(out var d) ? (object)(long)Math.Round(d) : throw new FormatException(),
                WidgetKind.Float => (double)value,
                WidgetKind.Boolean => (bool)value,
                WidgetKind.Json => SpawnGen.Engine.Types.JsonLiteral.FromJson(value),
                _ => (string?)value ?? "",
            };
        }
        catch (Exception ex) when (ex is FormatException or InvalidOperationException)
        {
            warnings.Add($"Node #{node.Id} widget '{name}': could not read {value.ToJsonString()}, using the default.");
            return spec.Default;
        }
    }

    // ---- save ----

    public static string Save(EditorDocument doc, Func<string, double>? measureText = null)
    {
        var graph = doc.Graph;
        var hits = new HitTester(doc, measureText);
        var nodes = WriteNodes(graph, doc.Registry, n => hits.Geometry(n).Bounds);
        var links = new JsonArray(WriteLinks(graph, doc.Registry, null).Select(l => (JsonNode)new JsonArray(l.Id, l.From, l.FromSlot, l.To, l.ToSlot, l.Type)).ToArray());

        var root = new JsonObject
        {
            ["last_node_id"] = graph.NextNodeId - 1, ["last_link_id"] = graph.NextLinkId - 1,
            ["nodes"] = nodes, ["links"] = links,
            ["groups"] = WriteGroups(doc.Graph),
            ["config"] = new JsonObject(),
            ["extra"] = doc.Extra?.DeepClone() ?? new JsonObject(),
            ["version"] = 0.4,
        };
        var used = SubgraphLibrary.Used(graph.Nodes.Select(n => n.ClassType));
        if (used.Count > 0) root["definitions"] = new JsonObject { ["subgraphs"] = SubgraphJson.Write(used, doc.Registry) };
        return root.ToJsonString(Pretty);
    }

    /// <summary>
    /// The nodes of a graph as ComfyUI saves them. Links may come from or go to the subgraph boundary (node -10 for the subgraph's inputs, -20 for its
    /// outputs): those count like any other link.
    /// </summary>
    internal static JsonArray WriteNodes(EditorGraph graph, NodeRegistry registry, Func<EditorNode, Rect> bounds)
    {
        var order = TopologicalOrder(graph);
        var nodes = new JsonArray();
        foreach (var n in graph.Nodes.OrderBy(n => n.Id))
        {
            var schema = registry.GetSchema(n.ClassType);
            var size = bounds(n);
            var obj = new JsonObject
            {
                ["id"] = n.Id, ["type"] = SubgraphLibrary.Get(n.ClassType) is { } def ? def.Id : n.ClassType, // an instance is saved under its definition's id
                ["pos"] = new JsonArray(n.X, n.Y),
                ["size"] = new JsonArray(size.Width, size.Height),
                ["flags"] = NodeFlags(n),
                ["order"] = order.GetValueOrDefault(n.Id),
                ["mode"] = (int)n.Mode,
            };
            if (n.Title is not null) obj["title"] = n.Title;
            if (n.Color is not null) obj["color"] = n.Color;
            if (n.BgColor is not null) obj["bgcolor"] = n.BgColor;

            var inputs = new JsonArray();
            var outputs = new JsonArray();
            if (schema is not null)
            {
                foreach (var i in NodeLayout.VisibleInputs(schema, n, graph))
                {
                    var link = graph.LinkInto(n.Id, i.Name);
                    if (i.Widget is null || link is not null)
                    {
                        var entry = new JsonObject { ["name"] = i.Name, ["type"] = i.Type.Name, ["link"] = link?.Id };
                        if (i.Widget is not null) entry["widget"] = new JsonObject { ["name"] = i.Name };
                        inputs.Add(entry);
                    }
                }
                for (int o = 0; o < schema.Outputs.Count; o++)
                {
                    var ids = graph.LinksFrom(n.Id, o).Select(l => l.Id).OrderBy(x => x).ToList();
                    outputs.Add(new JsonObject
                    {
                        ["name"] = schema.Outputs[o].Name, ["type"] = schema.Outputs[o].Type.Name,
                        ["links"] = ids.Count == 0 ? null : new JsonArray(ids.Select(x => (JsonNode)x).ToArray()), ["slot_index"] = o,
                    });
                }
                obj["inputs"] = inputs;
                obj["outputs"] = outputs;
                obj["widgets_values"] = WriteWidgets(schema, n);
            }
            else
            {
                // A node type we do not know: keep what we have so opening and saving does not destroy it.
                foreach (var l in graph.Links.Where(l => l.ToNode == n.Id)) inputs.Add(new JsonObject { ["name"] = l.ToInput, ["type"] = "*", ["link"] = l.Id });
                obj["inputs"] = inputs;
                obj["outputs"] = outputs;
                if (n.Widgets.GetValueOrDefault(RawWidgetsKey) is JsonNode raw) obj["widgets_values"] = raw.DeepClone();
            }
            var properties = (JsonObject)n.Properties.DeepClone();
            if (!properties.ContainsKey(NameForSearchReplace)) properties.Insert(0, NameForSearchReplace, n.ClassType);
            obj["properties"] = properties;
            nodes.Add(obj);
        }
        return nodes;
    }

    /// <summary>
    /// The links of a graph with the target slot as ComfyUI counts it (the position in the target's saved "inputs" list). A link to the subgraph's
    /// outputs (node -20) keeps its port index as the slot; <paramref name="boundaryType"/> names the type of a link from the inputs (node -10).
    /// </summary>
    internal static List<SavedLink> WriteLinks(EditorGraph graph, NodeRegistry registry, Func<int, string>? boundaryType)
    {
        var result = new List<SavedLink>();
        foreach (var l in graph.Links.OrderBy(l => l.Id))
        {
            var from = graph.FindNode(l.FromNode);
            var type = from is not null ? registry.GetSchema(from.ClassType)?.Outputs.ElementAtOrDefault(l.FromOutput)?.Type.Name ?? "*"
                : boundaryType?.Invoke(l.FromOutput) ?? "*";
            int slot = l.ToNode < 0 ? int.Parse(l.ToInput, System.Globalization.CultureInfo.InvariantCulture) : SlotIndex(graph, registry, l);
            result.Add(new(l.Id, l.FromNode, l.FromOutput, l.ToNode, slot, type));
        }
        return result;
    }

    static int SlotIndex(EditorGraph graph, NodeRegistry registry, EditorLink l)
    {
        var to = graph.GetNode(l.ToNode);
        if (registry.GetSchema(to.ClassType) is not { } schema)
            return graph.Links.Where(x => x.ToNode == l.ToNode).OrderBy(x => x.Id).ToList().FindIndex(x => x.Id == l.Id);
        int idx = 0;
        foreach (var i in NodeLayout.VisibleInputs(schema, to, graph))
        {
            bool listed = i.Widget is null || graph.LinkInto(l.ToNode, i.Name) is not null;
            if (i.Name == l.ToInput) return listed ? idx : -1;
            if (listed) idx++;
        }
        return -1;
    }

    static JsonArray WriteWidgets(NodeSchema schema, EditorNode n)
    {
        var list = new JsonArray();
        foreach (var w in NodeLayout.OrderedInputs(schema, n.Widgets).Where(i => i.Widget is not null))
        {
            var v = n.Widgets.GetValueOrDefault(w.Name) ?? w.Widget!.Default;
            list.Add(v switch
            {
                null => null,
                long l => JsonValue.Create(l),
                int i => JsonValue.Create((long)i),
                double d => JsonValue.Create(d),
                float f => JsonValue.Create((double)f),
                bool b => JsonValue.Create(b),
                SpawnGen.Engine.Types.DictValue or SpawnGen.Engine.Types.ArrayValue => SpawnGen.Engine.Types.JsonLiteral.ToNode(v),
                _ => JsonValue.Create(v.ToString()),
            });
            if (w.Widget!.Kind == WidgetKind.Int && SeedWidgets.Contains(w.Name)) list.Add(ControlAfterGenerate.ModeOf(n));
        }
        return list;
    }

    /// <summary>Execution order of the nodes (inputs before consumers), the "order" field ComfyUI stores.</summary>
    static Dictionary<int, int> TopologicalOrder(EditorGraph graph)
    {
        var order = new Dictionary<int, int>();
        var visiting = new HashSet<int>();
        void Visit(int id)
        {
            if (order.ContainsKey(id) || !visiting.Add(id)) return;
            foreach (var l in graph.Links.Where(l => l.ToNode == id)) Visit(l.FromNode);
            order[id] = order.Count;
        }
        foreach (var n in graph.Nodes.OrderBy(n => n.Id)) Visit(n.Id);
        return order;
    }
}
