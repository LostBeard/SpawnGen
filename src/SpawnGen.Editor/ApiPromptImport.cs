using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor;

/// <summary>
/// Opens a ComfyUI API-format prompt (what ComfyUI embeds in generated pictures, and what "Save (API format)" writes) as an editable workflow.
/// An API prompt has no positions, so the nodes are laid out left to right by how far they are from the loaders.
/// </summary>
public static class ApiPromptImport
{
    const double ColumnWidth = 430;
    const double RowGap = 36;
    const double Margin = 40;

    /// <summary>Replaces the document's graph. Returns warnings (unknown node types, links to missing nodes). Clears undo history.</summary>
    public static IReadOnlyList<string> Load(string apiJson, EditorDocument doc)
    {
        var prompt = PromptJson.Parse(apiJson);
        var warnings = new List<string>();

        // Node ids: ComfyUI uses numbers as strings; anything else gets a fresh number.
        var ids = new Dictionary<string, int>();
        int next = prompt.Nodes.Keys.Select(k => int.TryParse(k, out var n) ? n : 0).DefaultIfEmpty(0).Max() + 1;
        foreach (var key in prompt.Nodes.Keys) ids[key] = int.TryParse(key, out var n) && !ids.ContainsValue(n) ? n : next++;

        // Longest distance from a node with no inputs: that is its column.
        var depth = new Dictionary<string, int>();
        int Depth(string key, HashSet<string>? visiting = null)
        {
            if (depth.TryGetValue(key, out var d)) return d;
            visiting ??= [];
            if (!visiting.Add(key)) return 0; // a cycle cannot be laid out; the validator rejects it at run time
            d = 0;
            foreach (var input in prompt.Nodes[key].Inputs.Values)
                if (input.IsLink && prompt.Nodes.ContainsKey(input.NodeId!)) d = Math.Max(d, Depth(input.NodeId!, visiting) + 1);
            return depth[key] = d;
        }
        foreach (var key in prompt.Nodes.Keys) Depth(key);

        doc.Reset(graph =>
        {
            foreach (var (key, node) in prompt.Nodes)
            {
                var editor = new EditorNode { Id = ids[key], ClassType = node.ClassType, Title = node.Title };
                if (doc.Registry.GetSchema(node.ClassType) is { } schema)
                {
                    foreach (var input in schema.Inputs)
                    {
                        if (input.Widget is null) continue;
                        editor.Widgets[input.Name] = node.Inputs.TryGetValue(input.Name, out var v) && !v.IsLink && v.Literal is not null
                            ? Coerce(input.Widget, v.Literal) : input.Widget.Default;
                    }
                }
                else warnings.Add($"Node {key}: type '{node.ClassType}' is not installed; it is kept as a placeholder.");
                graph.AddNodeRaw(editor);
            }

            foreach (var (key, node) in prompt.Nodes)
                foreach (var (name, input) in node.Inputs.Where(i => i.Value.IsLink))
                {
                    if (!ids.TryGetValue(input.NodeId!, out var from)) { warnings.Add($"Node {key} input '{name}' points at missing node {input.NodeId}."); continue; }
                    graph.AddLinkRaw(new EditorLink { Id = graph.NextLinkId, FromNode = from, FromOutput = input.OutputIndex, ToNode = ids[key], ToInput = name });
                }
        });

        // Lay out by column, top to bottom within a column, using the real node sizes.
        var hits = new HitTester(doc);
        var rowTop = new Dictionary<int, double>();
        foreach (var key in prompt.Nodes.Keys.OrderBy(k => depth[k]).ThenBy(k => ids[k]))
        {
            var node = doc.Graph.GetNode(ids[key]);
            int column = depth[key];
            node.X = Margin + column * ColumnWidth;
            node.Y = Margin + rowTop.GetValueOrDefault(column);
            rowTop[column] = rowTop.GetValueOrDefault(column) + hits.Geometry(node).Bounds.Height + RowGap;
        }
        doc.ClearHistory();
        return warnings;
    }

    /// <summary>True when the JSON is an API-format prompt (an object of nodes, each with a <c>class_type</c>) rather than a workflow.</summary>
    public static bool Looks(string json)
    {
        try
        {
            using var doc = System.Text.Json.JsonDocument.Parse(json);
            if (doc.RootElement.ValueKind != System.Text.Json.JsonValueKind.Object || doc.RootElement.TryGetProperty("nodes", out _)) return false;
            foreach (var p in doc.RootElement.EnumerateObject())
                return p.Value.ValueKind == System.Text.Json.JsonValueKind.Object && p.Value.TryGetProperty("class_type", out _);
            return false;
        }
        catch (System.Text.Json.JsonException) { return false; }
    }

    static object? Coerce(WidgetSpec spec, object literal) => spec.Kind switch
    {
        WidgetKind.Int when literal is double d => (long)Math.Round(d),
        WidgetKind.Float when literal is long l => (double)l,
        _ => literal,
    };
}
