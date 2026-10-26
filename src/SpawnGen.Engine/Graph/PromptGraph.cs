namespace SpawnGen.Engine.Graph;

/// <summary>An input value: either a literal or a link to another node's output.</summary>
public readonly record struct PromptInput
{
    PromptInput(object? literal, string? nodeId, int outputIndex)
    {
        Literal = literal;
        NodeId = nodeId;
        OutputIndex = outputIndex;
    }

    /// <summary>
    /// Literal value: <see cref="long"/>, <see cref="double"/>, <see cref="bool"/>, <see cref="string"/>, a JSON-like <see cref="Types.DictValue"/> or
    /// <see cref="Types.ArrayValue"/> (object-valued widgets), or null. Unused for links.
    /// </summary>
    public object? Literal { get; }
    public string? NodeId { get; }
    public int OutputIndex { get; }
    public bool IsLink => NodeId is not null;

    public static PromptInput Link(string nodeId, int outputIndex = 0) => new(null, nodeId, outputIndex);

    public static PromptInput Value(object? literal) => new(Normalize(literal), null, 0);

    /// <summary>A <see cref="PromptInput"/> passes through; anything else becomes a literal.</summary>
    public static PromptInput From(object? value) => value is PromptInput p ? p : Value(value);

    static object? Normalize(object? v) => v switch
    {
        int i => (long)i,
        short s => (long)s,
        byte b => (long)b,
        uint u => (long)u,
        float f => (double)f,
        Enum e => e.ToString(),
        _ => v,
    };
}

public sealed class PromptNode
{
    public required string ClassType { get; init; }
    public Dictionary<string, PromptInput> Inputs { get; init; } = new(StringComparer.Ordinal);
    /// <summary>Optional editor title (<c>_meta.title</c>); not part of the cache signature.</summary>
    public string? Title { get; set; }
}

/// <summary>
/// An executable graph in ComfyUI's API format: node id -> class type + inputs. Order of insertion is preserved
/// so serialization is stable.
/// </summary>
public sealed class PromptGraph
{
    public Dictionary<string, PromptNode> Nodes { get; } = new(StringComparer.Ordinal);

    /// <summary>Adds a node. Input values that are <see cref="PromptInput"/> are used as-is; anything else is a literal.</summary>
    public PromptNode Add(string id, string classType, params (string Name, object? Value)[] inputs)
    {
        var node = new PromptNode { ClassType = classType };
        foreach (var (name, value) in inputs)
            node.Inputs[name] = PromptInput.From(value);
        Nodes[id] = node;
        return node;
    }

    public PromptGraph Clone()
    {
        var copy = new PromptGraph();
        foreach (var (id, n) in Nodes)
            copy.Nodes[id] = new PromptNode { ClassType = n.ClassType, Title = n.Title, Inputs = new(n.Inputs, StringComparer.Ordinal) };
        return copy;
    }
}
