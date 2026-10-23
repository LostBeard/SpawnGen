using System.Collections.Concurrent;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor;

/// <summary>One input or output of a subgraph. An input feeds one or more inner node inputs; an output is one inner node output.</summary>
public sealed class SubgraphPort
{
    public required string Name { get; set; }
    /// <summary>Socket type name (IMAGE, LATENT ...).</summary>
    public required string TypeName { get; set; }
    /// <summary>Inputs only: the inner (node, input) pairs this port feeds.</summary>
    public List<(int Node, string Input)> Targets { get; } = [];
    /// <summary>Outputs only: the inner node output this port exposes.</summary>
    public (int Node, int Output) Source { get; set; }
}

/// <summary>
/// A group of nodes packed into one node: the inner nodes and links, and the ports (the links that crossed the boundary when it was made). An instance in a graph is a node of type
/// <see cref="InstanceType"/>; running the graph replaces each instance by a copy of its inner nodes (see <see cref="SubgraphExpander"/>).
/// </summary>
public sealed class SubgraphDefinition
{
    public const string Prefix = "subgraph:";

    public required string Id { get; init; }
    public string Name { get; set; } = "Subgraph";
    public List<EditorNode> Nodes { get; } = [];
    public List<EditorLink> Links { get; } = [];
    public List<SubgraphPort> Inputs { get; } = [];
    public List<SubgraphPort> Outputs { get; } = [];
    /// <summary>
    /// Counts edits (ComfyUI's <c>revision</c>). A workflow read later that carries an older copy of this definition does not replace it, so a definition
    /// edited inside one workflow stays edited when another tab, or the workflow it came from, is opened again.
    /// </summary>
    public int Revision { get; set; }

    public string InstanceType => Prefix + Id;

    public NodeSchema ToSchema() => new()
    {
        Id = InstanceType,
        DisplayName = Name,
        Category = "subgraph",
        Inputs = Inputs.Select(p => new InputSpec(p.Name, new SocketType(p.TypeName), Required: false)).ToList(),
        Outputs = Outputs.Select(p => new OutputSpec(p.Name, new SocketType(p.TypeName))).ToList(),
    };

    /// <summary>The ids of the subgraphs this one contains (one level).</summary>
    public IEnumerable<string> ContainedIds() => Nodes.Where(n => n.ClassType.StartsWith(Prefix, StringComparison.Ordinal)).Select(n => n.ClassType[Prefix.Length..]).Distinct();
}

/// <summary>The subgraph definitions known to this process, by id. A workflow that uses subgraphs carries its definitions and registers them here when it opens.</summary>
public static class SubgraphLibrary
{
    static readonly ConcurrentDictionary<string, SubgraphDefinition> Definitions = new(StringComparer.Ordinal);

    public static void Register(SubgraphDefinition definition) => Definitions[definition.Id] = definition;

    /// <summary>Registers <paramref name="definition"/> unless a newer revision of it is already known. Returns the definition in use.</summary>
    public static SubgraphDefinition RegisterUnlessOlder(SubgraphDefinition definition) =>
        Definitions.AddOrUpdate(definition.Id, definition, (_, known) => known.Revision > definition.Revision ? known : definition);

    public static SubgraphDefinition? Get(string classTypeOrId) =>
        Definitions.TryGetValue(classTypeOrId.StartsWith(SubgraphDefinition.Prefix, StringComparison.Ordinal) ? classTypeOrId[SubgraphDefinition.Prefix.Length..] : classTypeOrId, out var d) ? d : null;

    /// <summary>Lets a registry answer for subgraph instance types (their schema comes from the definition). Safe to call more than once.</summary>
    public static void Install(NodeRegistry registry) => registry.SchemaResolver = type =>
        type.StartsWith(SubgraphDefinition.Prefix, StringComparison.Ordinal) ? Get(type)?.ToSchema() : SubgraphEditing.BoundarySchema(type);

    /// <summary>The definitions a graph needs, including those nested inside them, each once, innermost first.</summary>
    public static IReadOnlyList<SubgraphDefinition> Used(IEnumerable<string> classTypes)
    {
        var result = new List<SubgraphDefinition>();
        var seen = new HashSet<string>();
        void Visit(string id)
        {
            if (!seen.Add(id) || Get(id) is not { } def) return;
            foreach (var inner in def.ContainedIds()) Visit(inner);
            result.Add(def);
        }
        foreach (var t in classTypes) if (t.StartsWith(SubgraphDefinition.Prefix, StringComparison.Ordinal)) Visit(t[SubgraphDefinition.Prefix.Length..]);
        return result;
    }
}

/// <summary>Replaces subgraph instances by copies of their inner nodes, so a graph with subgraphs can be compiled and run like any other.</summary>
public static class SubgraphExpander
{
    const int MaxDepth = 16;

    /// <summary>
    /// The graph with every active subgraph instance replaced (repeatedly, for subgraphs inside subgraphs). Muted and bypassed instances stay as they are. The ids of ordinary nodes do not change, so
    /// run events still find their nodes; the copies of inner nodes get fresh ids above all existing ones. Returns <paramref name="graph"/> itself when there is nothing to expand.
    /// </summary>
    public static EditorGraph Flatten(EditorGraph graph)
    {
        var current = graph;
        for (int depth = 0; depth < MaxDepth; depth++)
        {
            var instances = current.Nodes.Where(n => n.Mode == NodeMode.Always && SubgraphLibrary.Get(n.ClassType) is not null).ToList();
            if (instances.Count == 0) return current;
            current = ExpandOnce(current, instances);
        }
        throw new InvalidOperationException("Subgraphs are nested too deeply (does one contain itself?).");
    }

    static EditorGraph ExpandOnce(EditorGraph g, List<EditorNode> instances)
    {
        var flat = new EditorGraph();
        var instanceIds = instances.ToDictionary(i => i.Id, i => SubgraphLibrary.Get(i.ClassType)!);
        int nodeId = g.NextNodeId, linkId = g.NextLinkId;
        foreach (var n in g.Nodes) if (!instanceIds.ContainsKey(n.Id)) flat.AddNodeRaw(n);

        var map = new Dictionary<(int Instance, int Inner), int>();
        foreach (var (instanceId, def) in instanceIds)
        {
            foreach (var inner in def.Nodes)
            {
                int id = nodeId++;
                map[(instanceId, inner.Id)] = id;
                var copy = inner.Clone(id);
                copy.ZIndex = id;
                flat.AddNodeRaw(copy);
            }
            foreach (var l in def.Links)
                flat.AddLinkRaw(new EditorLink { Id = linkId++, FromNode = map[(instanceId, l.FromNode)], FromOutput = l.FromOutput, ToNode = map[(instanceId, l.ToNode)], ToInput = l.ToInput });
        }

        // Where an output really comes from once instances are looked through.
        (int Node, int Output) Resolve(int fromNode, int fromOutput)
        {
            if (!instanceIds.TryGetValue(fromNode, out var def) || fromOutput >= def.Outputs.Count) return (fromNode, fromOutput);
            var port = def.Outputs[fromOutput];
            return (map[(fromNode, port.Source.Node)], port.Source.Output);
        }

        foreach (var l in g.Links)
        {
            var (srcNode, srcOutput) = Resolve(l.FromNode, l.FromOutput);
            if (instanceIds.TryGetValue(l.ToNode, out var target))
            {
                var port = target.Inputs.FirstOrDefault(p => p.Name == l.ToInput);
                if (port is null) continue;
                foreach (var (node, input) in port.Targets)
                    flat.AddLinkRaw(new EditorLink { Id = linkId++, FromNode = srcNode, FromOutput = srcOutput, ToNode = map[(l.ToNode, node)], ToInput = input });
            }
            else flat.AddLinkRaw(new EditorLink { Id = linkId++, FromNode = srcNode, FromOutput = srcOutput, ToNode = l.ToNode, ToInput = l.ToInput });
        }
        return flat;
    }
}

/// <summary>
/// Editing a subgraph's inside on the canvas, as ComfyUI does when one is opened: the inner nodes and links, with two boundary nodes standing for the
/// subgraph's inputs (left, its outputs are the input ports) and its outputs (right, its inputs are the output ports). <see cref="Commit"/> writes the
/// edited inside back to the definition, which every instance shares.
/// </summary>
public static class SubgraphEditing
{
    public const string InputsPrefix = "subgraph-inputs:", OutputsPrefix = "subgraph-outputs:";
    /// <summary>The boundary nodes' ids, ComfyUI's -10 and -20.</summary>
    public const int InputsNodeId = -10, OutputsNodeId = -20;

    /// <summary>The schema of a boundary node type (editor-only: never compiled), or null for any other type.</summary>
    public static NodeSchema? BoundarySchema(string type)
    {
        if (type.StartsWith(InputsPrefix, StringComparison.Ordinal) && SubgraphLibrary.Get(type[InputsPrefix.Length..]) is { } inDef)
            return new NodeSchema
            {
                Id = type, DisplayName = "Subgraph Inputs", Category = "subgraph", FrontendOnly = true,
                Outputs = inDef.Inputs.Select(p => new OutputSpec(p.Name, new SocketType(p.TypeName))).ToList(),
            };
        if (type.StartsWith(OutputsPrefix, StringComparison.Ordinal) && SubgraphLibrary.Get(type[OutputsPrefix.Length..]) is { } outDef)
            return new NodeSchema
            {
                Id = type, DisplayName = "Subgraph Outputs", Category = "subgraph", FrontendOnly = true,
                Inputs = outDef.Outputs.Select(p => new InputSpec(p.Name, new SocketType(p.TypeName), Required: false)).ToList(),
            };
        return null;
    }

    /// <summary>Fills <paramref name="doc"/> with the inside of <paramref name="def"/> (copies: nothing changes until <see cref="Commit"/>). Clears undo.</summary>
    public static void Open(EditorDocument doc, SubgraphDefinition def)
    {
        doc.Reset(graph =>
        {
            foreach (var n in def.Nodes) graph.AddNodeRaw(n.Clone());
            foreach (var l in def.Links) graph.AddLinkRaw(new EditorLink { Id = l.Id, FromNode = l.FromNode, FromOutput = l.FromOutput, ToNode = l.ToNode, ToInput = l.ToInput });
            double left = def.Nodes.Select(n => n.X).DefaultIfEmpty(0).Min() - 260, right = def.Nodes.Select(n => n.X + (n.Width ?? 240)).DefaultIfEmpty(400).Max() + 80;
            double top = def.Nodes.Select(n => n.Y).DefaultIfEmpty(0).Min();
            graph.AddNodeRaw(new EditorNode { Id = InputsNodeId, ClassType = InputsPrefix + def.Id, X = left, Y = top, Pinned = true });
            graph.AddNodeRaw(new EditorNode { Id = OutputsNodeId, ClassType = OutputsPrefix + def.Id, X = right, Y = top, Pinned = true });
            int linkId = def.Links.Select(l => l.Id).DefaultIfEmpty(0).Max() + 1;
            for (int p = 0; p < def.Inputs.Count; p++)
                foreach (var (node, input) in def.Inputs[p].Targets)
                    graph.AddLinkRaw(new EditorLink { Id = linkId++, FromNode = InputsNodeId, FromOutput = p, ToNode = node, ToInput = input });
            for (int p = 0; p < def.Outputs.Count; p++)
                if (def.Nodes.Any(n => n.Id == def.Outputs[p].Source.Node))
                    graph.AddLinkRaw(new EditorLink { Id = linkId++, FromNode = def.Outputs[p].Source.Node, FromOutput = def.Outputs[p].Source.Output, ToNode = OutputsNodeId, ToInput = def.Outputs[p].Name });
        });
    }

    /// <summary>True when <paramref name="doc"/> holds the inside of a subgraph (it has the boundary nodes); the definition's id, or null.</summary>
    public static string? EditedId(EditorDocument doc) =>
        doc.Graph.FindNode(InputsNodeId)?.ClassType is { } t && t.StartsWith(InputsPrefix, StringComparison.Ordinal) ? t[InputsPrefix.Length..] : null;

    /// <summary>
    /// Writes the edited inside back into <paramref name="def"/>: its nodes, inner links, which inner inputs each input port feeds and which inner output
    /// each output port shows. The ports themselves stay as they are. Raises the revision, so older copies of the definition no longer replace it.
    /// </summary>
    public static void Commit(EditorDocument doc, SubgraphDefinition def)
    {
        var graph = doc.Graph;
        bool Boundary(int id) => id is InputsNodeId or OutputsNodeId;
        def.Nodes.Clear();
        def.Nodes.AddRange(graph.Nodes.Where(n => !Boundary(n.Id)).Select(n => n.Clone()));
        def.Links.Clear();
        def.Links.AddRange(graph.Links.Where(l => !Boundary(l.FromNode) && !Boundary(l.ToNode))
            .Select(l => new EditorLink { Id = l.Id, FromNode = l.FromNode, FromOutput = l.FromOutput, ToNode = l.ToNode, ToInput = l.ToInput }));
        for (int p = 0; p < def.Inputs.Count; p++)
        {
            def.Inputs[p].Targets.Clear();
            foreach (var l in graph.LinksFrom(InputsNodeId, p).OrderBy(l => l.Id)) def.Inputs[p].Targets.Add((l.ToNode, l.ToInput));
        }
        foreach (var port in def.Outputs)
            port.Source = graph.LinkInto(OutputsNodeId, port.Name) is { } l ? (l.FromNode, l.FromOutput) : (0, 0);
        def.Revision++;
    }
}
