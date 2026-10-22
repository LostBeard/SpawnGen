using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor;

/// <summary>How a node takes part in a run. Values match ComfyUI's workflow <c>mode</c>.</summary>
public enum NodeMode
{
    Always = 0,
    /// <summary>Muted: the node and everything that depends only on it is skipped.</summary>
    Never = 2,
    /// <summary>Bypassed: the node is skipped and its inputs pass through to matching outputs.</summary>
    Bypass = 4,
}

public sealed class EditorNode
{
    public required int Id { get; init; }
    public required string ClassType { get; init; }
    public string? Title { get; set; }
    public double X { get; set; }
    public double Y { get; set; }
    /// <summary>Width the user dragged the node to; null = sized from its content.</summary>
    public double? Width { get; set; }
    public bool Collapsed { get; set; }
    public NodeMode Mode { get; set; }
    /// <summary>Widget values by input name (long, double, bool or string).</summary>
    public Dictionary<string, object?> Widgets { get; init; } = new(StringComparer.Ordinal);

    public EditorNode Clone(int? newId = null) => new()
    {
        Id = newId ?? Id, ClassType = ClassType, Title = Title, X = X, Y = Y, Width = Width, Collapsed = Collapsed, Mode = Mode,
        Widgets = new Dictionary<string, object?>(Widgets, StringComparer.Ordinal),
    };
}

/// <summary>A link from an output slot of one node to a named input of another. An input takes at most one link.</summary>
public sealed class EditorLink
{
    public required int Id { get; init; }
    public required int FromNode { get; init; }
    public required int FromOutput { get; init; }
    public required int ToNode { get; init; }
    public required string ToInput { get; init; }
}

/// <summary>The node graph being edited: positions, widget values and links. Mutate it through <see cref="EditorDocument"/> so changes can be undone.</summary>
public sealed class EditorGraph
{
    readonly Dictionary<int, EditorNode> _nodes = new();
    readonly Dictionary<int, EditorLink> _links = new();

    public IReadOnlyCollection<EditorNode> Nodes => _nodes.Values;
    public IReadOnlyCollection<EditorLink> Links => _links.Values;
    public int NextNodeId { get; set; } = 1;
    public int NextLinkId { get; set; } = 1;

    public EditorNode? FindNode(int id) => _nodes.GetValueOrDefault(id);
    public EditorLink? FindLink(int id) => _links.GetValueOrDefault(id);

    public EditorNode GetNode(int id) => FindNode(id) ?? throw new KeyNotFoundException($"No node {id}.");

    public EditorLink? LinkInto(int nodeId, string input) =>
        _links.Values.FirstOrDefault(l => l.ToNode == nodeId && l.ToInput == input);

    public IEnumerable<EditorLink> LinksFrom(int nodeId, int? output = null) =>
        _links.Values.Where(l => l.FromNode == nodeId && (output is null || l.FromOutput == output));

    public IEnumerable<EditorLink> LinksOf(int nodeId) => _links.Values.Where(l => l.FromNode == nodeId || l.ToNode == nodeId);

    // Raw mutations. They keep the id counters ahead of every id in use; callers (commands) own the undo bookkeeping.
    internal void AddNodeRaw(EditorNode node)
    {
        _nodes[node.Id] = node;
        NextNodeId = Math.Max(NextNodeId, node.Id + 1);
    }

    internal void RemoveNodeRaw(int id) => _nodes.Remove(id);

    internal void AddLinkRaw(EditorLink link)
    {
        _links[link.Id] = link;
        NextLinkId = Math.Max(NextLinkId, link.Id + 1);
    }

    internal void RemoveLinkRaw(int id) => _links.Remove(id);

    /// <summary>True when following links downstream from <paramref name="from"/> reaches <paramref name="target"/> (a new link target -> from would close a cycle).</summary>
    public bool Reaches(int from, int target)
    {
        var seen = new HashSet<int>();
        var stack = new Stack<int>();
        stack.Push(from);
        while (stack.Count > 0)
        {
            var n = stack.Pop();
            if (n == target) return true;
            if (!seen.Add(n)) continue;
            foreach (var l in LinksFrom(n)) stack.Push(l.ToNode);
        }
        return false;
    }

    /// <summary>Why <paramref name="fromNode"/>.<paramref name="fromOutput"/> cannot feed <paramref name="toNode"/>.<paramref name="toInput"/>, or null when it can.</summary>
    public string? WhyCannotConnect(NodeRegistry registry, int fromNode, int fromOutput, int toNode, string toInput)
    {
        var from = FindNode(fromNode);
        var to = FindNode(toNode);
        if (from is null || to is null) return "Node not found.";
        if (fromNode == toNode) return "A node cannot feed itself.";
        var outSchema = registry.GetSchema(from.ClassType);
        var inSchema = registry.GetSchema(to.ClassType);
        if (outSchema is null) return $"Unknown node type '{from.ClassType}'.";
        if (inSchema is null) return $"Unknown node type '{to.ClassType}'.";
        if (fromOutput < 0 || fromOutput >= outSchema.Outputs.Count) return "No such output.";
        var input = inSchema.FindInput(toInput);
        if (input is null) return $"'{to.ClassType}' has no input '{toInput}'.";
        var outType = outSchema.Outputs[fromOutput].Type;
        // A widget input is fed through its value type (INT, FLOAT, STRING, BOOLEAN or a combo's string).
        var inType = input.Type;
        if (!inType.Accepts(outType)) return $"{outType} cannot connect to {inType}.";
        if (Reaches(toNode, fromNode)) return "That link would create a cycle.";
        return null;
    }
}
