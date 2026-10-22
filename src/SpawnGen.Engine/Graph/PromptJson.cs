using System.Text.Json;

namespace SpawnGen.Engine.Graph;

/// <summary>Reads and writes ComfyUI API-format prompt JSON.</summary>
public static class PromptJson
{
    public static PromptGraph Parse(string json)
    {
        using var doc = JsonDocument.Parse(json, new JsonDocumentOptions { AllowTrailingCommas = true, CommentHandling = JsonCommentHandling.Skip });
        // Some tools wrap the graph: { "prompt": { ... } }.
        var root = doc.RootElement;
        if (root.ValueKind == JsonValueKind.Object && root.TryGetProperty("prompt", out var inner) && inner.ValueKind == JsonValueKind.Object
            && !root.TryGetProperty("class_type", out _))
            root = inner;
        return Parse(root);
    }

    public static PromptGraph Parse(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Object)
            throw new FormatException("A prompt must be a JSON object keyed by node id.");

        var graph = new PromptGraph();
        foreach (var nodeProp in root.EnumerateObject())
        {
            var n = nodeProp.Value;
            if (n.ValueKind != JsonValueKind.Object || !n.TryGetProperty("class_type", out var ct) || ct.ValueKind != JsonValueKind.String)
                throw new FormatException($"Node '{nodeProp.Name}' is missing a string 'class_type'.");

            var node = new PromptNode { ClassType = ct.GetString()! };
            if (n.TryGetProperty("_meta", out var meta) && meta.ValueKind == JsonValueKind.Object
                && meta.TryGetProperty("title", out var title) && title.ValueKind == JsonValueKind.String)
                node.Title = title.GetString();

            if (n.TryGetProperty("inputs", out var inputs) && inputs.ValueKind == JsonValueKind.Object)
                foreach (var input in inputs.EnumerateObject())
                    node.Inputs[input.Name] = ParseInput(nodeProp.Name, input.Name, input.Value);

            graph.Nodes[nodeProp.Name] = node;
        }
        return graph;
    }

    static PromptInput ParseInput(string nodeId, string name, JsonElement v)
    {
        switch (v.ValueKind)
        {
            case JsonValueKind.Number:
                return v.TryGetInt64(out var l) ? PromptInput.Value(l) : PromptInput.Value(v.GetDouble());
            case JsonValueKind.String: return PromptInput.Value(v.GetString());
            case JsonValueKind.True: return PromptInput.Value(true);
            case JsonValueKind.False: return PromptInput.Value(false);
            case JsonValueKind.Null: return PromptInput.Value(null);
            case JsonValueKind.Array when v.GetArrayLength() == 2:
                var id = v[0];
                var idx = v[1];
                if (idx.ValueKind == JsonValueKind.Number && idx.TryGetInt32(out var index)
                    && id.ValueKind is JsonValueKind.String or JsonValueKind.Number)
                    return PromptInput.Link(id.ValueKind == JsonValueKind.String ? id.GetString()! : id.GetRawText(), index);
                break;
        }
        throw new FormatException($"Node '{nodeId}' input '{name}': unsupported value {v.GetRawText()}.");
    }

    public static string ToJson(PromptGraph graph, bool indented = false)
    {
        using var ms = new MemoryStream();
        using (var w = new Utf8JsonWriter(ms, new JsonWriterOptions { Indented = indented }))
        {
            w.WriteStartObject();
            foreach (var (id, node) in graph.Nodes)
            {
                w.WriteStartObject(id);
                w.WriteString("class_type", node.ClassType);
                w.WriteStartObject("inputs");
                foreach (var (name, input) in node.Inputs)
                {
                    w.WritePropertyName(name);
                    if (input.IsLink)
                    {
                        w.WriteStartArray();
                        w.WriteStringValue(input.NodeId);
                        w.WriteNumberValue(input.OutputIndex);
                        w.WriteEndArray();
                    }
                    else WriteLiteral(w, input.Literal);
                }
                w.WriteEndObject();
                if (node.Title is not null)
                {
                    w.WriteStartObject("_meta");
                    w.WriteString("title", node.Title);
                    w.WriteEndObject();
                }
                w.WriteEndObject();
            }
            w.WriteEndObject();
        }
        return System.Text.Encoding.UTF8.GetString(ms.ToArray());
    }

    static void WriteLiteral(Utf8JsonWriter w, object? v)
    {
        switch (v)
        {
            case null: w.WriteNullValue(); break;
            case long l: w.WriteNumberValue(l); break;
            case double d: w.WriteNumberValue(d); break;
            case bool b: w.WriteBooleanValue(b); break;
            case string s: w.WriteStringValue(s); break;
            default: throw new NotSupportedException($"Cannot serialize literal of type {v.GetType().Name}.");
        }
    }
}
