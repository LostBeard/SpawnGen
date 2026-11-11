using System.Text.Json.Nodes;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor;

// App Mode's configuration, in ComfyUI's workflow format (measured from the workflow files ComfyUI's frontend writes; docs.comfy.org/interface/app-mode
// describes the feature): extra.linearData = { "inputs": [[node id, widget name], ...], "outputs": [node id, ...] } in display order, the node ids as
// strings (numbers are read too), possibly naming nodes inside a subgraph's definition; extra.linearMode = true when the workflow opens as an app.

/// <summary>One app input: a widget of a node.</summary>
public sealed record AppInput(string NodeId, string Widget);

/// <summary>A workflow's App Mode setup: which widgets the app shows as inputs, which output nodes it shows, and whether it opens as an app.</summary>
public sealed class AppModeConfig
{
    public List<AppInput> Inputs { get; } = [];
    public List<string> Outputs { get; } = [];

    /// <summary>extra.linearMode: true opens the workflow as an app, false in the node graph, null when never chosen.</summary>
    public bool? OpensAsApp { get; set; }

    /// <summary>The app has been built (the workflow carries linearData, or the builder ran on it).</summary>
    public bool Configured { get; set; }

    public static AppModeConfig Read(JsonObject? extra)
    {
        var config = new AppModeConfig { OpensAsApp = extra?["linearMode"] is JsonValue m && m.TryGetValue<bool>(out var b) ? b : null };
        if (extra?["linearData"] is not JsonObject data) return config;
        config.Configured = true;
        if (data["inputs"] is JsonArray inputs)
            foreach (var entry in inputs)
                if (entry is JsonArray { Count: >= 2 } pair && Id(pair[0]) is { } id && pair[1] is JsonValue w && w.TryGetValue<string>(out var widget))
                    config.Inputs.Add(new AppInput(id, widget));
        if (data["outputs"] is JsonArray outputs)
            foreach (var entry in outputs)
                if (Id(entry) is { } id) config.Outputs.Add(id);
        return config;
    }

    static string? Id(JsonNode? node) => node is JsonValue v
        ? v.TryGetValue<string>(out var s) ? s : v.TryGetValue<long>(out var l) ? l.ToString(System.Globalization.CultureInfo.InvariantCulture) : null
        : null;

    /// <summary>Writes the setup into the document's extra block, keeping anything else linearData holds.</summary>
    public void Write(EditorDocument doc)
    {
        var extra = doc.Extra ??= [];
        var data = extra["linearData"] as JsonObject ?? [];
        data["inputs"] = new JsonArray([.. Inputs.Select(i => (JsonNode)new JsonArray(i.NodeId, i.Widget))]);
        data["outputs"] = new JsonArray([.. Outputs.Select(o => (JsonNode)JsonValue.Create(o)!)]);
        extra["linearData"] = data;
        if (OpensAsApp is { } opens) extra["linearMode"] = opens;
        else extra.Remove("linearMode");
    }

    public static string IdOf(EditorNode node) => node.Id.ToString(System.Globalization.CultureInfo.InvariantCulture);

    public bool HasInput(int nodeId, string widget) => Inputs.Contains(new AppInput(nodeId.ToString(System.Globalization.CultureInfo.InvariantCulture), widget));
    public bool HasOutput(int nodeId) => Outputs.Contains(nodeId.ToString(System.Globalization.CultureInfo.InvariantCulture));

    /// <summary>Adds the widget as the last input, or takes it out when it is one already. True when it is now an input.</summary>
    public bool ToggleInput(int nodeId, string widget)
    {
        var input = new AppInput(nodeId.ToString(System.Globalization.CultureInfo.InvariantCulture), widget);
        if (Inputs.Remove(input)) return false;
        Inputs.Add(input);
        return true;
    }

    /// <summary>Adds the node as the last output, or takes it out. True when it is now an output.</summary>
    public bool ToggleOutput(int nodeId)
    {
        var id = nodeId.ToString(System.Globalization.CultureInfo.InvariantCulture);
        if (Outputs.Remove(id)) return false;
        Outputs.Add(id);
        return true;
    }

    /// <summary>Moves entry <paramref name="index"/> of a list by <paramref name="delta"/> places (clamped).</summary>
    public static void Move<T>(List<T> list, int index, int delta)
    {
        int to = Math.Clamp(index + delta, 0, list.Count - 1);
        if (index < 0 || index >= list.Count || to == index) return;
        var item = list[index];
        list.RemoveAt(index);
        list.Insert(to, item);
    }

    /// <summary>The node an id names: in the workflow's own graph first, else inside one of the subgraph definitions it uses.</summary>
    public static EditorNode? Resolve(EditorGraph graph, string id)
    {
        if (!int.TryParse(id, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var n)) return null;
        if (graph.FindNode(n) is { } top) return top;
        foreach (var def in SubgraphLibrary.Used(graph.Nodes.Select(x => x.ClassType)))
            if (def.Nodes.FirstOrDefault(x => x.Id == n) is { } inner) return inner;
        return null;
    }

    /// <summary>An output node: one the app can show (it saves or previews something).</summary>
    public static bool IsOutputNode(NodeRegistry registry, EditorNode node) => registry.GetSchema(node.ClassType) is { IsOutputNode: true };

    /// <summary>Whether an executed event's node (a plain id, or "outer:inner" for a node inside a subgraph) is the app output <paramref name="id"/>.</summary>
    public static bool IsNode(string? eventNode, string id) =>
        eventNode is not null && (eventNode == id || eventNode.EndsWith(":" + id, StringComparison.Ordinal));
}
