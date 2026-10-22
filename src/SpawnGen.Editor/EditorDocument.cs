using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor;

/// <summary>One undoable change to the graph.</summary>
interface IEditorCommand
{
    void Do(EditorGraph graph);
    void Undo(EditorGraph graph);
    /// <summary>Folds <paramref name="next"/> into this command when they are one gesture (a drag, a slider scrub); returns false when they are separate steps.</summary>
    bool TryMerge(IEditorCommand next) => false;
    /// <summary>True for edits that arrive in bursts as one gesture (drag, scrub, resize): successive ones merge into one undo step until <c>EndGesture</c>.</summary>
    bool IsGesture => false;
}

/// <summary>
/// The graph plus its undo history and change notification. Every edit the user can make goes through here so it can be undone, and the
/// UI redraws from <see cref="Changed"/>.
/// </summary>
public sealed class EditorDocument(NodeRegistry registry)
{
    readonly List<IEditorCommand> _undo = [];
    readonly List<IEditorCommand> _redo = [];
    List<IEditorCommand>? _group;

    public NodeRegistry Registry { get; } = InstallSubgraphs(registry);

    static NodeRegistry InstallSubgraphs(NodeRegistry registry)
    {
        SubgraphLibrary.Install(registry);
        return registry;
    }

    /// <summary>Part of a workflow the editor does not edit but must not lose when it is opened and saved again: ComfyUI's "extra" block.</summary>
    public System.Text.Json.Nodes.JsonObject? Extra { get; set; }
    public EditorGraph Graph { get; } = new();
    public int UndoLimit { get; set; } = 200;
    public bool CanUndo => _undo.Count > 0;
    public bool CanRedo => _redo.Count > 0;

    /// <summary>Raised after every change, including undo and redo.</summary>
    public event Action? Changed;

    // ---- edits ----

    /// <summary>Creates a node of <paramref name="classType"/> at the position with its widgets at their defaults.</summary>
    public EditorNode AddNode(string classType, double x, double y)
    {
        var schema = Registry.GetSchema(classType) ?? throw new ArgumentException($"Unknown node type '{classType}'.", nameof(classType));
        var node = new EditorNode { Id = Graph.NextNodeId, ClassType = classType, X = x, Y = y };
        foreach (var input in schema.Inputs)
            if (input.Widget is { } w) node.Widgets[input.Name] = DefaultValue(w);
        Run(new AddNodeCommand(node));
        return node;
    }

    static object? DefaultValue(WidgetSpec w) => w.Default switch
    {
        null => w.Kind switch
        {
            WidgetKind.Int => (object)(long)(w.Min ?? 0),
            WidgetKind.Float => w.Min ?? 0.0,
            WidgetKind.Boolean => false,
            WidgetKind.Combo => w.ResolveOptions().FirstOrDefault() ?? "",
            _ => "",
        },
        int i => (long)i,
        float f => (double)f,
        var v => v,
    };

    /// <summary>Deletes the nodes and every link touching them.</summary>
    public void DeleteNodes(IEnumerable<int> ids)
    {
        var list = ids.Distinct().Where(i => Graph.FindNode(i) is not null).ToList();
        if (list.Count > 0) Run(new DeleteNodesCommand(list));
    }

    // ---- subgraphs ----

    /// <summary>
    /// Packs the nodes into one subgraph node (one undo step). Links from outside into the nodes become the subgraph's inputs (links from the same output share one input) and links from
    /// the nodes to the outside become its outputs. Returns the new node, or null with <paramref name="error"/> when nothing sensible can be packed.
    /// </summary>
    public EditorNode? ConvertToSubgraph(IEnumerable<int> nodeIds, string name, out string? error)
    {
        var ids = nodeIds.Distinct().Where(i => Graph.FindNode(i) is not null).ToHashSet();
        error = null;
        if (ids.Count == 0) { error = "Select the nodes to pack first."; return null; }
        var selected = ids.Select(Graph.GetNode).ToList();
        if (selected.Any(n => Registry.GetSchema(n.ClassType) is null)) { error = "A node of an unknown type cannot be packed."; return null; }

        var def = new SubgraphDefinition { Id = Guid.NewGuid().ToString("N"), Name = name };
        foreach (var n in selected.OrderBy(n => n.Id)) def.Nodes.Add(n.Clone());
        var inLinks = Graph.Links.Where(l => ids.Contains(l.ToNode) && !ids.Contains(l.FromNode)).OrderBy(l => l.ToNode).ThenBy(l => l.ToInput).ToList();
        var outLinks = Graph.Links.Where(l => ids.Contains(l.FromNode) && !ids.Contains(l.ToNode)).OrderBy(l => l.FromNode).ThenBy(l => l.FromOutput).ThenBy(l => l.ToNode).ToList();
        foreach (var l in Graph.Links.Where(l => ids.Contains(l.FromNode) && ids.Contains(l.ToNode)))
            def.Links.Add(new EditorLink { Id = l.Id, FromNode = l.FromNode, FromOutput = l.FromOutput, ToNode = l.ToNode, ToInput = l.ToInput });

        string Unique(string wanted, IEnumerable<string> taken)
        {
            var set = taken.ToHashSet();
            if (!set.Contains(wanted)) return wanted;
            for (int i = 2; ; i++) if (!set.Contains($"{wanted}_{i}")) return $"{wanted}_{i}";
        }
        // One input per distinct source output: everything it fed inside becomes a target of that input.
        var inputPorts = new List<(SubgraphPort Port, EditorLink Outer)>();
        foreach (var group in inLinks.GroupBy(l => (l.FromNode, l.FromOutput)))
        {
            var first = group.First();
            var srcSchema = Registry.GetSchema(Graph.GetNode(first.FromNode).ClassType);
            var type = srcSchema is not null && first.FromOutput < srcSchema.Outputs.Count ? srcSchema.Outputs[first.FromOutput].Type.Name : "*";
            var port = new SubgraphPort { Name = Unique(first.ToInput, def.Inputs.Select(p => p.Name)), TypeName = type };
            foreach (var l in group) port.Targets.Add((l.ToNode, l.ToInput));
            def.Inputs.Add(port);
            inputPorts.Add((port, first));
        }
        var outputPorts = new List<(SubgraphPort Port, List<EditorLink> Outer)>();
        foreach (var group in outLinks.GroupBy(l => (l.FromNode, l.FromOutput)))
        {
            var first = group.First();
            var srcSchema = Registry.GetSchema(Graph.GetNode(first.FromNode).ClassType)!;
            var output = first.FromOutput < srcSchema.Outputs.Count ? srcSchema.Outputs[first.FromOutput] : null;
            var port = new SubgraphPort { Name = Unique(output?.Name ?? $"output{first.FromOutput}", def.Outputs.Select(p => p.Name)), TypeName = output?.Type.Name ?? "*", Source = (first.FromNode, first.FromOutput) };
            def.Outputs.Add(port);
            outputPorts.Add((port, group.ToList()));
        }

        SubgraphLibrary.Register(def);
        double x = selected.Min(n => n.X), y = selected.Min(n => n.Y);
        EditorNode? instance = null;
        Group(() =>
        {
            instance = AddNode(def.InstanceType, x, y);
            SetTitle(instance.Id, name);
            DeleteNodes(ids);
            foreach (var (port, outer) in inputPorts) Connect(outer.FromNode, outer.FromOutput, instance.Id, port.Name, out _);
            for (int k = 0; k < outputPorts.Count; k++)
                foreach (var l in outputPorts[k].Outer) Connect(instance.Id, k, l.ToNode, l.ToInput, out _);
        });
        return instance;
    }

    /// <summary>Puts a subgraph instance's inner nodes back into the graph in its place, rewired to what the instance was connected to (one undo step).</summary>
    public bool UnpackSubgraph(int instanceId, out string? error)
    {
        error = null;
        var instance = Graph.FindNode(instanceId);
        if (instance is null || SubgraphLibrary.Get(instance.ClassType) is not { } def) { error = "That is not a subgraph."; return false; }
        double minX = def.Nodes.Count == 0 ? 0 : def.Nodes.Min(n => n.X), minY = def.Nodes.Count == 0 ? 0 : def.Nodes.Min(n => n.Y);
        var outerIn = Graph.Links.Where(l => l.ToNode == instanceId).ToList();
        var outerOut = Graph.Links.Where(l => l.FromNode == instanceId).ToList();
        Group(() =>
        {
            var map = new Dictionary<int, int>();
            foreach (var inner in def.Nodes)
            {
                var copy = AddNode(inner.ClassType, instance.X + inner.X - minX, instance.Y + inner.Y - minY);
                map[inner.Id] = copy.Id;
                foreach (var (name, value) in inner.Widgets) SetWidget(copy.Id, name, value);
                if (inner.Title is not null) SetTitle(copy.Id, inner.Title);
                if (inner.Mode != NodeMode.Always) SetMode(copy.Id, inner.Mode);
                if (inner.Collapsed) SetCollapsed(copy.Id, true);
                if (inner.Width is { } w) SetWidth(copy.Id, w);
                if (inner.Color is not null || inner.BgColor is not null) SetNodeColor(copy.Id, inner.Color, inner.BgColor);
            }
            foreach (var l in def.Links) Connect(map[l.FromNode], l.FromOutput, map[l.ToNode], l.ToInput, out _);
            foreach (var l in outerIn)
                if (def.Inputs.FirstOrDefault(p => p.Name == l.ToInput) is { } port)
                    foreach (var (node, input) in port.Targets) Connect(l.FromNode, l.FromOutput, map[node], input, out _);
            foreach (var l in outerOut)
                if (l.FromOutput < def.Outputs.Count) Connect(map[def.Outputs[l.FromOutput].Source.Node], def.Outputs[l.FromOutput].Source.Output, l.ToNode, l.ToInput, out _);
            DeleteNodes([instanceId]);
        });
        return true;
    }

    public void MoveNodes(IEnumerable<int> ids, double dx, double dy)
    {
        var list = ids.Distinct().Where(i => Graph.FindNode(i) is { Pinned: false }).ToList();
        if (list.Count > 0 && (dx != 0 || dy != 0)) Run(new MoveNodesCommand(list, dx, dy));
    }

    /// <summary>
    /// Connects an output to an input, replacing the link the input already had. Returns the new link, or null with <paramref name="error"/> set
    /// when the types do not match or the link would close a cycle.
    /// </summary>
    public EditorLink? Connect(int fromNode, int fromOutput, int toNode, string toInput, out string? error)
    {
        error = Graph.WhyCannotConnect(Registry, fromNode, fromOutput, toNode, toInput);
        if (error is not null) return null;
        var link = new EditorLink { Id = Graph.NextLinkId, FromNode = fromNode, FromOutput = fromOutput, ToNode = toNode, ToInput = toInput };
        Run(new ConnectCommand(link, Graph.LinkInto(toNode, toInput)));
        return link;
    }

    public void Disconnect(int linkId)
    {
        if (Graph.FindLink(linkId) is { } link) Run(new DisconnectCommand(link));
    }

    public void SetWidget(int nodeId, string name, object? value)
    {
        var node = Graph.GetNode(nodeId);
        node.Widgets.TryGetValue(name, out var old);
        if (Equals(old, value)) return;
        Run(new SetWidgetCommand(nodeId, name, old, value));
    }

    public void SetTitle(int nodeId, string? title)
    {
        var node = Graph.GetNode(nodeId);
        if (node.Title != title) Run(new SetNodePropertyCommand<string?>(nodeId, "title", node.Title, title,
            (n, v) => n.Title = v));
    }

    /// <summary>Colours a node (null for both = back to the theme's). One undo step.</summary>
    public void SetNodeColor(int nodeId, string? color, string? bgColor)
    {
        var node = Graph.GetNode(nodeId);
        if (node.Color == color && node.BgColor == bgColor) return;
        Run(new SetNodePropertyCommand<(string?, string?)>(nodeId, "color", (node.Color, node.BgColor), (color, bgColor), (n, v) => (n.Color, n.BgColor) = v));
    }

    public void SetMode(int nodeId, NodeMode mode)
    {
        var node = Graph.GetNode(nodeId);
        if (node.Mode != mode) Run(new SetNodePropertyCommand<NodeMode>(nodeId, "mode", node.Mode, mode, (n, v) => n.Mode = v));
    }

    public void SetCollapsed(int nodeId, bool collapsed)
    {
        var node = Graph.GetNode(nodeId);
        if (node.Collapsed != collapsed) Run(new SetNodePropertyCommand<bool>(nodeId, "collapsed", node.Collapsed, collapsed, (n, v) => n.Collapsed = v));
    }

    public void SetPinned(int nodeId, bool pinned)
    {
        var node = Graph.GetNode(nodeId);
        if (node.Pinned != pinned) Run(new SetNodePropertyCommand<bool>(nodeId, "pinned", node.Pinned, pinned, (n, v) => n.Pinned = v));
    }

    public void SetWidth(int nodeId, double? width)
    {
        var node = Graph.GetNode(nodeId);
        if (node.Width != width) Run(new SetNodePropertyCommand<double?>(nodeId, "width", node.Width, width, (n, v) => n.Width = v));
    }

    // ---- groups ----

    /// <summary>Adds an empty group box.</summary>
    public EditorGroup AddGroup(double x, double y, double width, double height, string? title = null, string? color = null)
    {
        var group = new EditorGroup
        {
            Id = Graph.NextGroupId, X = x, Y = y, Width = Math.Max(width, EditorGroup.MinWidth), Height = Math.Max(height, EditorGroup.MinHeight),
            Title = title ?? "Group", Color = color ?? EditorGroup.DefaultColor,
        };
        Run(new AddGroupCommand(group));
        return group;
    }

    /// <summary>A group that just fits around the given nodes (with a margin and room for its title), as Ctrl+G makes. Null when there are no nodes.</summary>
    public EditorGroup? AddGroupAround(IEnumerable<EditorNode> nodes, Func<EditorNode, Rect> bounds, double margin = 12)
    {
        Rect? box = null;
        foreach (var n in nodes) box = box is null ? bounds(n) : Rect.Union(box.Value, bounds(n));
        if (box is not { } b) return null;
        return AddGroup(b.X - margin, b.Y - margin - EditorGroup.TitleHeight, b.Width + 2 * margin, b.Height + 2 * margin + EditorGroup.TitleHeight);
    }

    public void DeleteGroup(int id)
    {
        if (Graph.FindGroup(id) is { } g) Run(new DeleteGroupCommand(g));
    }

    /// <summary>Moves a group, and with it the nodes inside (listed by the caller at the start of a drag so they stay the same set during it).</summary>
    public void MoveGroup(int id, double dx, double dy, IReadOnlyList<int> carriedNodes)
    {
        if (Graph.FindGroup(id) is not { Pinned: false }) return;
        if (dx != 0 || dy != 0) Run(new MoveGroupCommand(id, dx, dy, [.. carriedNodes.Where(n => Graph.FindNode(n) is { Pinned: false })]));
    }

    public void SetGroupBounds(int id, double x, double y, double width, double height)
    {
        var g = Graph.GetGroup(id);
        if (g.Pinned) return;
        width = Math.Max(width, EditorGroup.MinWidth);
        height = Math.Max(height, EditorGroup.MinHeight);
        if ((g.X, g.Y, g.Width, g.Height) != (x, y, width, height)) Run(new SetGroupBoundsCommand(id, (g.X, g.Y, g.Width, g.Height), (x, y, width, height)));
    }

    public void SetGroupTitle(int id, string title)
    {
        var g = Graph.GetGroup(id);
        if (g.Title != title) Run(new SetGroupPropertyCommand<string>(id, g.Title, title, (grp, v) => grp.Title = v));
    }

    public void SetGroupPinned(int id, bool pinned)
    {
        var g = Graph.GetGroup(id);
        if (g.Pinned != pinned) Run(new SetGroupPropertyCommand<bool>(id, g.Pinned, pinned, (grp, v) => grp.Pinned = v));
    }

    public void SetGroupColor(int id, string color)
    {
        var g = Graph.GetGroup(id);
        if (g.Color != color) Run(new SetGroupPropertyCommand<string>(id, g.Color, color, (grp, v) => grp.Color = v));
    }

    /// <summary>Several edits as one undo step (paste, delete-and-reconnect, ...).</summary>
    public void Group(Action edits)
    {
        if (_group is not null) { edits(); return; }
        _group = [];
        try { edits(); }
        finally
        {
            var done = _group;
            _group = null;
            if (done.Count > 0) Push(new GroupCommand(done));
        }
    }

    // ---- history ----

    public void Undo()
    {
        if (_undo.Count == 0) return;
        var cmd = _undo[^1];
        _undo.RemoveAt(_undo.Count - 1);
        cmd.Undo(Graph);
        _redo.Add(cmd);
        Changed?.Invoke();
    }

    public void Redo()
    {
        if (_redo.Count == 0) return;
        var cmd = _redo[^1];
        _redo.RemoveAt(_redo.Count - 1);
        cmd.Do(Graph);
        _undo.Add(cmd);
        Changed?.Invoke();
    }

    /// <summary>Forgets the history (after loading a workflow).</summary>
    public void ClearHistory()
    {
        _undo.Clear();
        _redo.Clear();
    }

    /// <summary>Marks the end of a gesture so the next edit of the same kind is a new undo step instead of merging into the previous one.</summary>
    public void EndGesture() => _lastMergeable = null;

    IEditorCommand? _lastMergeable;

    void Run(IEditorCommand cmd)
    {
        cmd.Do(Graph);
        if (_group is not null) _group.Add(cmd);
        else Push(cmd);
        Changed?.Invoke();
    }

    void Push(IEditorCommand cmd)
    {
        _redo.Clear();
        if (_lastMergeable is not null && _undo.Count > 0 && ReferenceEquals(_undo[^1], _lastMergeable) && _lastMergeable.TryMerge(cmd))
            return;
        _undo.Add(cmd);
        _lastMergeable = cmd.IsGesture ? cmd : null;
        if (_undo.Count > UndoLimit) _undo.RemoveAt(0);
    }

    /// <summary>Removes every node and link (a new workflow). Not undoable; clears the history.</summary>
    public void Clear() => Reset(_ => { });

    /// <summary>Replaces the whole graph (workflow load). Not undoable; clears the history.</summary>
    internal void Reset(Action<EditorGraph> fill)
    {
        foreach (var l in Graph.Links.ToList()) Graph.RemoveLinkRaw(l.Id);
        foreach (var n in Graph.Nodes.ToList()) Graph.RemoveNodeRaw(n.Id);
        foreach (var g in Graph.Groups.ToList()) Graph.RemoveGroupRaw(g.Id);
        Graph.NextGroupId = 1;
        Graph.NextNodeId = 1;
        Graph.NextLinkId = 1;
        fill(Graph);
        ClearHistory();
        _lastMergeable = null;
        Changed?.Invoke();
    }

    // ---- commands ----

    sealed class AddNodeCommand(EditorNode node) : IEditorCommand
    {
        public void Do(EditorGraph g) => g.AddNodeRaw(node);
        public void Undo(EditorGraph g) => g.RemoveNodeRaw(node.Id);
    }

    sealed class DeleteNodesCommand(List<int> ids) : IEditorCommand
    {
        List<EditorNode> _nodes = [];
        List<EditorLink> _links = [];

        public void Do(EditorGraph g)
        {
            _nodes = ids.Select(g.GetNode).ToList();
            _links = ids.SelectMany(g.LinksOf).DistinctBy(l => l.Id).ToList();
            foreach (var l in _links) g.RemoveLinkRaw(l.Id);
            foreach (var n in _nodes) g.RemoveNodeRaw(n.Id);
        }

        public void Undo(EditorGraph g)
        {
            foreach (var n in _nodes) g.AddNodeRaw(n);
            foreach (var l in _links) g.AddLinkRaw(l);
        }
    }

    sealed class MoveNodesCommand(List<int> ids, double dx, double dy) : IEditorCommand
    {
        double _dx = dx, _dy = dy;

        public bool IsGesture => true;
        public void Do(EditorGraph g) => Shift(g, _dx, _dy);
        public void Undo(EditorGraph g) => Shift(g, -_dx, -_dy);

        void Shift(EditorGraph g, double x, double y)
        {
            foreach (var id in ids)
                if (g.FindNode(id) is { } n) { n.X += x; n.Y += y; }
        }

        public bool TryMerge(IEditorCommand next)
        {
            if (next is not MoveNodesCommand m || !m.Ids.SequenceEqual(ids)) return false;
            _dx += m._dx;
            _dy += m._dy;
            return true;
        }

        List<int> Ids => ids;
    }

    sealed class ConnectCommand(EditorLink link, EditorLink? replaced) : IEditorCommand
    {
        public void Do(EditorGraph g)
        {
            if (replaced is not null) g.RemoveLinkRaw(replaced.Id);
            g.AddLinkRaw(link);
        }

        public void Undo(EditorGraph g)
        {
            g.RemoveLinkRaw(link.Id);
            if (replaced is not null) g.AddLinkRaw(replaced);
        }
    }

    sealed class DisconnectCommand(EditorLink link) : IEditorCommand
    {
        public void Do(EditorGraph g) => g.RemoveLinkRaw(link.Id);
        public void Undo(EditorGraph g) => g.AddLinkRaw(link);
    }

    sealed class SetWidgetCommand(int nodeId, string name, object? before, object? after) : IEditorCommand
    {
        object? _after = after;

        public bool IsGesture => true;
        public void Do(EditorGraph g) => g.GetNode(nodeId).Widgets[name] = _after;
        public void Undo(EditorGraph g) => g.GetNode(nodeId).Widgets[name] = before;

        public bool TryMerge(IEditorCommand next)
        {
            if (next is not SetWidgetCommand w || w.NodeId != nodeId || w.Name != name) return false;
            _after = w._after;
            return true;
        }

        int NodeId => nodeId;
        string Name => name;
    }

    sealed class SetNodePropertyCommand<T>(int nodeId, string property, T before, T after, Action<EditorNode, T> set) : IEditorCommand
    {
        T _after = after;
        public int NodeId { get; } = nodeId;
        public string Property { get; } = property;
        /// <summary>Resizing fires on every pointer move; the other properties are one-shot toggles.</summary>
        public bool IsGesture => Property == "width";
        public void Do(EditorGraph g) => set(g.GetNode(NodeId), _after);
        public void Undo(EditorGraph g) => set(g.GetNode(NodeId), before);

        public bool TryMerge(IEditorCommand next)
        {
            if (next is not SetNodePropertyCommand<T> p || p.NodeId != NodeId || p.Property != Property) return false;
            _after = p._after;
            return true;
        }
    }

    sealed class AddGroupCommand(EditorGroup group) : IEditorCommand
    {
        public void Do(EditorGraph g) => g.AddGroupRaw(group);
        public void Undo(EditorGraph g) => g.RemoveGroupRaw(group.Id);
    }

    sealed class DeleteGroupCommand(EditorGroup group) : IEditorCommand
    {
        int _index;
        public void Do(EditorGraph g) => _index = g.RemoveGroupRaw(group.Id);
        public void Undo(EditorGraph g) => g.AddGroupRaw(group, _index);
    }

    sealed class MoveGroupCommand(int id, double dx, double dy, List<int> nodes) : IEditorCommand
    {
        double _dx = dx, _dy = dy;
        public int Id => id;
        public bool IsGesture => true;
        public void Do(EditorGraph g) => Shift(g, _dx, _dy);
        public void Undo(EditorGraph g) => Shift(g, -_dx, -_dy);

        void Shift(EditorGraph g, double x, double y)
        {
            if (g.FindGroup(id) is { } grp) { grp.X += x; grp.Y += y; }
            foreach (var n in nodes) if (g.FindNode(n) is { } node) { node.X += x; node.Y += y; }
        }

        public bool TryMerge(IEditorCommand next)
        {
            if (next is not MoveGroupCommand m || m.Id != id || !m.Nodes.SequenceEqual(nodes)) return false;
            _dx += m._dx;
            _dy += m._dy;
            return true;
        }

        List<int> Nodes => nodes;
    }

    sealed class SetGroupBoundsCommand(int id, (double X, double Y, double W, double H) before, (double X, double Y, double W, double H) after) : IEditorCommand
    {
        (double X, double Y, double W, double H) _after = after;
        public int Id => id;
        public bool IsGesture => true;
        public void Do(EditorGraph g) => Apply(g, _after);
        public void Undo(EditorGraph g) => Apply(g, before);

        void Apply(EditorGraph g, (double X, double Y, double W, double H) b)
        {
            var grp = g.GetGroup(id);
            (grp.X, grp.Y, grp.Width, grp.Height) = b;
        }

        public bool TryMerge(IEditorCommand next)
        {
            if (next is not SetGroupBoundsCommand s || s.Id != id) return false;
            _after = s._after;
            return true;
        }
    }

    sealed class SetGroupPropertyCommand<T>(int id, T before, T after, Action<EditorGroup, T> set) : IEditorCommand
    {
        public void Do(EditorGraph g) => set(g.GetGroup(id), after);
        public void Undo(EditorGraph g) => set(g.GetGroup(id), before);
    }

    sealed class GroupCommand(List<IEditorCommand> commands) : IEditorCommand
    {
        public void Do(EditorGraph g) { foreach (var c in commands) c.Do(g); }
        public void Undo(EditorGraph g) { for (int i = commands.Count - 1; i >= 0; i--) commands[i].Undo(g); }
    }
}
