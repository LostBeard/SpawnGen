using System.Text.Json;
using System.Text.Json.Nodes;

namespace SpawnGen.Engine.Nodes;

/// <summary>Reads ComfyUI's <c>object_info</c> catalogue (what the SpawnGen server serves) back into schemas, so the editor can show a server's custom nodes.</summary>
public static class NodeSchemaJson
{
    public static IReadOnlyList<NodeSchema> FromObjectInfo(JsonObject info)
    {
        var result = new List<NodeSchema>();
        foreach (var (id, node) in info)
        {
            if (node is not JsonObject n) continue;
            try { result.Add(Parse(id, n)); }
            catch (Exception ex) when (ex is InvalidCastException or InvalidOperationException or FormatException or KeyNotFoundException) { } // one odd entry must not hide the rest
        }
        return result;
    }

    static NodeSchema Parse(string id, JsonObject n)
    {
        var inputs = new List<InputSpec>();
        foreach (var section in new[] { "required", "optional" })
        {
            if (n["input"]?[section] is not JsonObject group) continue;
            var order = n["input_order"]?[section] as JsonArray;
            var names = order is null ? group.Select(kv => kv.Key) : order.Select(o => (string)o!).Where(group.ContainsKey);
            foreach (var name in names)
                if (group[name] is JsonArray entry && entry.Count > 0) inputs.Add(Input(name, entry, section == "required"));
        }
        var types = n["output"] as JsonArray ?? [];
        var names2 = n["output_name"] as JsonArray;
        var lists = n["output_is_list"] as JsonArray;
        var outputs = types.Select((t, i) => new OutputSpec(names2 is not null && i < names2.Count && (string?)names2[i] is { Length: > 0 } name ? name : (string)t!, new SocketType((string)t!))
        {
            IsList = lists is not null && i < lists.Count && lists[i] is JsonValue v && v.TryGetValue<bool>(out var isList) && isList,
        }).ToList();
        return new NodeSchema
        {
            Id = id,
            DisplayName = (string?)n["display_name"] is { Length: > 0 } dn ? dn : id,
            Category = (string?)n["category"] is { Length: > 0 } cat ? cat : "custom",
            Description = (string?)n["description"],
            Inputs = inputs,
            Outputs = outputs,
            IsOutputNode = (bool?)n["output_node"] ?? false,
            InputIsList = n["input_is_list"] is JsonValue l && l.TryGetValue<bool>(out var inputIsList) && inputIsList,
        };
    }

    static InputSpec Input(string name, JsonArray entry, bool required)
    {
        var config = entry.Count > 1 ? entry[1] as JsonObject : null;
        return Plain(name, entry, required, config) with { Lazy = config?["lazy"] is JsonValue lazy && lazy.TryGetValue<bool>(out var isLazy) && isLazy, Optional = !required,
            Group = (string?)config?["autogrow_group"], GroupMin = (int?)config?["autogrow_min"] ?? 0,
            DynamicParent = (string?)config?["dynamic_parent"],
            DynamicOptions = config?["dynamic_options"] is JsonArray options ? options.Select(o => (string)o!).ToList() : [] };
    }

    static InputSpec Plain(string name, JsonArray entry, bool required, JsonObject? config)
    {
        if (entry[0] is JsonArray options)
            return new InputSpec(name, SocketType.String, required, new WidgetSpec(WidgetKind.Combo) { Options = options.Select(o => (string)o!).ToList(), Default = options.Count > 0 ? (string?)options[0] : null });
        var type = (string)entry[0]!;
        if (config is null) return new InputSpec(name, new SocketType(type), required);
        var kind = type switch { "INT" => WidgetKind.Int, "FLOAT" => WidgetKind.Float, "BOOLEAN" => WidgetKind.Boolean, "STRING" => WidgetKind.String, "COLOR" => WidgetKind.Color, _ => (WidgetKind?)null };
        if (kind is null) return new InputSpec(name, new SocketType(type), required);
        return new InputSpec(name, new SocketType(type), required, new WidgetSpec(kind.Value)
        {
            Default = Value(kind.Value, config["default"]),
            Min = (double?)config["min"], Max = (double?)config["max"], Step = (double?)config["step"],
            Multiline = (bool?)config["multiline"] ?? false,
        });
    }

    static object? Value(WidgetKind kind, JsonNode? node)
    {
        if (node is null) return null;
        return kind switch
        {
            WidgetKind.Int => (object)(long)Math.Round((double)node),
            WidgetKind.Float => (double)node,
            WidgetKind.Boolean => (bool)node,
            _ => (string?)node,
        };
    }
}
