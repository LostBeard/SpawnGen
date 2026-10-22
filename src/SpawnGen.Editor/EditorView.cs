using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor;

public enum PointerButton { None, Left, Middle, Right }

[Flags]
public enum Modifiers { None = 0, Shift = 1, Ctrl = 2, Alt = 4, Space = 8 }

/// <summary>A link being dragged out of a slot. <see cref="Target"/> is where it would land if released now.</summary>
public sealed class LinkDraft
{
    /// <summary>True when the drag started on an output slot (and is looking for an input), false when it started on an input (looking for an output).</summary>
    public required bool StartsAtOutput { get; init; }
    public required int NodeId { get; init; }
    /// <summary>Output index when <see cref="StartsAtOutput"/>.</summary>
    public int Output { get; init; }
    /// <summary>Input name when not <see cref="StartsAtOutput"/>.</summary>
    public string? Input { get; init; }
    /// <summary>The existing link this drag picked up from a connected input; it is replaced or removed on release.</summary>
    public EditorLink? PickedUp { get; init; }
    public Point Cursor { get; set; }
    /// <summary>The slot the link would connect to, or null.</summary>
    public Hit? Target { get; set; }
    /// <summary>Why the slot under the cursor refuses the link (shown near the cursor), or null.</summary>
    public string? Problem { get; set; }
}

/// <summary>
/// The editing session on top of an <see cref="EditorDocument"/>: viewport, selection, and what the pointer and keyboard do. A UI shell forwards
/// raw events here and draws from the state; all behaviour lives here so it is unit-tested without a browser.
/// </summary>
public sealed class EditorView
{
    enum Mode { Idle, Panning, DraggingNodes, Resizing, DraggingLink, BoxSelecting, PressedWidget }

    const double DragThresholdPixels = 3;

    readonly HashSet<int> _selected = [];
    Mode _mode;
    Point _downScreen, _lastScreen, _lastWorld;
    bool _moved;
    Hit _pressed;
    double _resizeGrabOffset;
    bool _boxAdds;
    HashSet<int> _boxBase = [];
    Clipboard? _clipboard;

    public EditorView(EditorDocument document, Func<string, double>? measureText = null)
    {
        Document = document;
        Hits = new HitTester(document, measureText);
        document.Changed += OnDocumentChanged;
    }

    public EditorDocument Document { get; }
    public Viewport Viewport { get; } = new();
    public HitTester Hits { get; }

    public IReadOnlyCollection<int> SelectedNodes => _selected;
    public int? SelectedLink { get; private set; }
    public LinkDraft? Draft { get; private set; }
    /// <summary>The drag-select rectangle in world coordinates while one is being drawn.</summary>
    public Rect? SelectionBox { get; private set; }
    public Hit Hover { get; private set; }
    /// <summary>The CSS cursor the canvas should show.</summary>
    public string Cursor { get; private set; } = "default";
    /// <summary>The canvas size in pixels, for fit-to-view. The UI keeps it current.</summary>
    public Point ScreenSize { get; set; } = new(1200, 800);

    /// <summary>Selection, viewport, drafts or hover changed: redraw.</summary>
    public event Action? ViewChanged;
    /// <summary>The user released a link on empty canvas or double-clicked it: show the node search at this world point. The draft (if any) says what to connect the new node to.</summary>
    public event Action<Point, LinkDraft?>? SearchRequested;
    /// <summary>A widget was clicked: the UI edits it in place.</summary>
    public event Action<int, string>? WidgetActivated;
    public event Action<int>? TitleEditRequested;
    public event Action<Hit, Point>? ContextMenuRequested;
    /// <summary>A short message for a toast (a refused link, a paste problem).</summary>
    public event Action<string>? Message;

    // ---- selection ----

    public void SelectOnly(int nodeId)
    {
        if (_selected.Count == 1 && _selected.Contains(nodeId) && SelectedLink is null) return;
        _selected.Clear();
        _selected.Add(nodeId);
        SelectedLink = null;
        Document.Graph.BringToFront(nodeId);
        ViewChanged?.Invoke();
    }

    public void ToggleSelect(int nodeId)
    {
        if (!_selected.Remove(nodeId)) { _selected.Add(nodeId); Document.Graph.BringToFront(nodeId); }
        SelectedLink = null;
        ViewChanged?.Invoke();
    }

    public void SelectAll()
    {
        _selected.Clear();
        foreach (var n in Document.Graph.Nodes) _selected.Add(n.Id);
        SelectedLink = null;
        ViewChanged?.Invoke();
    }

    public void ClearSelection()
    {
        if (_selected.Count == 0 && SelectedLink is null) return;
        _selected.Clear();
        SelectedLink = null;
        ViewChanged?.Invoke();
    }

    void OnDocumentChanged()
    {
        _selected.RemoveWhere(id => Document.Graph.FindNode(id) is null);
        if (SelectedLink is { } l && Document.Graph.FindLink(l) is null) SelectedLink = null;
        ViewChanged?.Invoke();
    }

    // ---- placing nodes ----

    /// <summary>Adds a node (from the search box) at a world point; when a link was being dragged, connects the new node to it.</summary>
    public EditorNode PlaceNode(string classType, Point world, LinkDraft? draft = null)
    {
        EditorNode? node = null;
        Document.Group(() =>
        {
            node = Document.AddNode(classType, world.X, world.Y);
            if (draft is null) return;
            if (draft.StartsAtOutput) ConnectToNode(draft.NodeId, draft.Output, node.Id, out _);
            else ConnectFromNode(node.Id, draft.NodeId, draft.Input!, out _);
        });
        SelectOnly(node!.Id);
        return node;
    }

    // ---- pointer ----

    public void PointerDown(Point screen, PointerButton button, Modifiers mods = Modifiers.None, int clickCount = 1)
    {
        _downScreen = _lastScreen = screen;
        var world = _lastWorld = Viewport.ScreenToWorld(screen);
        _moved = false;

        if (button == PointerButton.Middle || (button == PointerButton.Left && mods.HasFlag(Modifiers.Space)))
        {
            _mode = Mode.Panning;
            SetCursor("grabbing");
            return;
        }

        var hit = Hits.Test(world);
        if (button == PointerButton.Right)
        {
            if (hit.Kind is HitKind.NodeBody or HitKind.TitleBar or HitKind.Widget && !_selected.Contains(hit.NodeId)) SelectOnly(hit.NodeId);
            ContextMenuRequested?.Invoke(hit, screen);
            return;
        }
        if (button != PointerButton.Left) return;

        _pressed = hit;
        switch (hit.Kind)
        {
            case HitKind.OutputSlot:
                Draft = new LinkDraft { StartsAtOutput = true, NodeId = hit.NodeId, Output = hit.Index, Cursor = world };
                _mode = Mode.DraggingLink;
                break;

            case HitKind.InputSlot:
                var existing = Document.Graph.LinkInto(hit.NodeId, hit.Name!);
                // Dragging a connected input picks its link up (to move it elsewhere or drop it); an empty input starts a link backwards.
                Draft = existing is not null
                    ? new LinkDraft { StartsAtOutput = true, NodeId = existing.FromNode, Output = existing.FromOutput, PickedUp = existing, Cursor = world }
                    : new LinkDraft { StartsAtOutput = false, NodeId = hit.NodeId, Input = hit.Name, Cursor = world };
                _mode = Mode.DraggingLink;
                break;

            case HitKind.ResizeHandle:
                var node = Document.Graph.GetNode(hit.NodeId);
                var g = Hits.Geometry(node);
                _resizeGrabOffset = g.Bounds.Right - world.X;
                SelectOnly(hit.NodeId);
                _mode = Mode.Resizing;
                break;

            case HitKind.CollapseButton:
                Document.SetCollapsed(hit.NodeId, !Document.Graph.GetNode(hit.NodeId).Collapsed);
                Document.EndGesture();
                break;

            case HitKind.Widget:
                SelectOnly(hit.NodeId);
                _mode = Mode.PressedWidget;
                break;

            case HitKind.TitleBar when clickCount >= 2:
                TitleEditRequested?.Invoke(hit.NodeId);
                break;

            case HitKind.TitleBar:
            case HitKind.NodeBody:
                if (mods.HasFlag(Modifiers.Shift) || mods.HasFlag(Modifiers.Ctrl)) ToggleSelect(hit.NodeId);
                else if (!_selected.Contains(hit.NodeId)) SelectOnly(hit.NodeId);
                else Document.Graph.BringToFront(hit.NodeId);
                _mode = _selected.Contains(hit.NodeId) ? Mode.DraggingNodes : Mode.Idle;
                break;

            case HitKind.Link:
                _selected.Clear();
                SelectedLink = hit.LinkId;
                ViewChanged?.Invoke();
                break;

            default: // empty canvas
                if (clickCount >= 2) { SearchRequested?.Invoke(world, null); break; }
                if (mods.HasFlag(Modifiers.Ctrl))
                {
                    _boxAdds = mods.HasFlag(Modifiers.Shift);
                    _boxBase = _boxAdds ? [.. _selected] : [];
                    SelectionBox = new Rect(world.X, world.Y, 0, 0);
                    _mode = Mode.BoxSelecting;
                }
                else
                {
                    _mode = Mode.Panning; // a plain drag on the background pans, like ComfyUI
                    SetCursor("grabbing");
                }
                break;
        }
    }

    public void PointerMove(Point screen, Modifiers mods = Modifiers.None)
    {
        var world = Viewport.ScreenToWorld(screen);
        if (!_moved && _mode != Mode.Idle && screen.DistanceTo(_downScreen) >= DragThresholdPixels) _moved = true;

        switch (_mode)
        {
            case Mode.Panning:
                Viewport.Pan(screen.X - _lastScreen.X, screen.Y - _lastScreen.Y);
                ViewChanged?.Invoke();
                break;

            case Mode.DraggingNodes when _moved:
                Document.MoveNodes(_selected, world.X - _lastWorld.X, world.Y - _lastWorld.Y);
                SetCursor("move");
                break;

            case Mode.Resizing:
                var node = Document.Graph.GetNode(_pressed.NodeId);
                Document.SetWidth(node.Id, Math.Max(90, world.X + _resizeGrabOffset - node.X));
                break;

            case Mode.DraggingLink:
                UpdateDraft(world);
                break;

            case Mode.BoxSelecting:
                SelectionBox = Rect.FromCorners(Viewport.ScreenToWorld(_downScreen), world);
                _selected.Clear();
                _selected.UnionWith(_boxBase);
                foreach (var n in Document.Graph.Nodes)
                    if (Hits.Geometry(n).Bounds.Intersects(SelectionBox.Value)) _selected.Add(n.Id);
                ViewChanged?.Invoke();
                break;

            case Mode.Idle:
                var hover = Hits.Test(world);
                if (hover != Hover)
                {
                    Hover = hover;
                    SetCursor(hover.Kind switch
                    {
                        HitKind.OutputSlot or HitKind.InputSlot => "crosshair",
                        HitKind.Widget or HitKind.CollapseButton or HitKind.Link => "pointer",
                        HitKind.ResizeHandle => "nwse-resize",
                        HitKind.TitleBar => "move",
                        _ => "default",
                    });
                    ViewChanged?.Invoke();
                }
                break;
        }
        _lastScreen = screen;
        _lastWorld = world;
    }

    public void PointerUp(Point screen, PointerButton button, Modifiers mods = Modifiers.None)
    {
        var world = Viewport.ScreenToWorld(screen);
        var mode = _mode;
        _mode = Mode.Idle;

        switch (mode)
        {
            case Mode.Panning:
                if (!_moved && button == PointerButton.Left) ClearSelection(); // a click on the background deselects
                break;
            case Mode.DraggingNodes:
            case Mode.Resizing:
                Document.EndGesture();
                break;
            case Mode.PressedWidget:
                if (!_moved) WidgetActivated?.Invoke(_pressed.NodeId, _pressed.Name!);
                break;
            case Mode.DraggingLink:
                ReleaseLink(world);
                break;
            case Mode.BoxSelecting:
                SelectionBox = null;
                break;
        }
        Draft = null;
        _pressed = default;
        SetCursor("default");
        ViewChanged?.Invoke();
    }

    public void Wheel(Point screen, double deltaY, Modifiers mods = Modifiers.None)
    {
        Viewport.ZoomAt(screen, Math.Pow(1.001, -deltaY));
        ViewChanged?.Invoke();
    }

    // ---- link dragging ----

    void UpdateDraft(Point world)
    {
        var d = Draft!;
        d.Cursor = world;
        d.Target = null;
        d.Problem = null;
        var hit = Hits.Test(world);

        if (d.StartsAtOutput)
        {
            switch (hit.Kind)
            {
                case HitKind.InputSlot:
                    d.Problem = Document.Graph.WhyCannotConnect(Document.Registry, d.NodeId, d.Output, hit.NodeId, hit.Name!);
                    if (d.Problem is null) d.Target = hit;
                    break;
                case HitKind.NodeBody or HitKind.TitleBar or HitKind.Widget:
                    // Over a node but not a slot: the link would go to its first matching input.
                    if (FirstInputFor(d.NodeId, d.Output, hit.NodeId) is { } input) d.Target = new Hit(HitKind.InputSlot, hit.NodeId, input);
                    break;
            }
        }
        else
        {
            switch (hit.Kind)
            {
                case HitKind.OutputSlot:
                    d.Problem = Document.Graph.WhyCannotConnect(Document.Registry, hit.NodeId, hit.Index, d.NodeId, d.Input!);
                    if (d.Problem is null) d.Target = hit;
                    break;
                case HitKind.NodeBody or HitKind.TitleBar or HitKind.Widget:
                    if (FirstOutputFor(hit.NodeId, d.NodeId, d.Input!) is { } output) d.Target = new Hit(HitKind.OutputSlot, hit.NodeId, Index: output);
                    break;
            }
        }
        ViewChanged?.Invoke();
    }

    void ReleaseLink(Point world)
    {
        var d = Draft!;
        UpdateDraft(world);

        if (d.Target is { } target)
        {
            Document.Group(() =>
            {
                if (d.PickedUp is not null) Document.Disconnect(d.PickedUp.Id);
                string? error;
                if (d.StartsAtOutput) Document.Connect(d.NodeId, d.Output, target.NodeId, target.Name!, out error);
                else Document.Connect(target.NodeId, target.Index, d.NodeId, d.Input!, out error);
                if (error is not null) Message?.Invoke(error);
            });
            return;
        }

        if (d.Problem is not null)
        {
            Message?.Invoke(d.Problem); // dropped on a slot that refuses it: the link stays where it was
            return;
        }

        if (!_moved) return; // a click on a slot, not a drag

        var onCanvas = Hits.Test(world).Kind is HitKind.None or HitKind.Link;
        if (d.PickedUp is not null)
        {
            Document.Disconnect(d.PickedUp.Id); // dragged a link off its input and let go: it is removed
            if (onCanvas) SearchRequested?.Invoke(world, new LinkDraft { StartsAtOutput = true, NodeId = d.NodeId, Output = d.Output });
        }
        else if (onCanvas)
        {
            SearchRequested?.Invoke(world, d);
        }
    }

    /// <summary>The first input of <paramref name="toNode"/> that accepts the output, preferring one that is still free.</summary>
    string? FirstInputFor(int fromNode, int fromOutput, int toNode)
    {
        var to = Document.Graph.FindNode(toNode);
        var schema = to is null ? null : Document.Registry.GetSchema(to.ClassType);
        if (schema is null) return null;
        var candidates = schema.Inputs
            .Where(i => Document.Graph.WhyCannotConnect(Document.Registry, fromNode, fromOutput, toNode, i.Name) is null).ToList();
        return (candidates.FirstOrDefault(i => Document.Graph.LinkInto(toNode, i.Name) is null) ?? candidates.FirstOrDefault())?.Name;
    }

    int? FirstOutputFor(int fromNode, int toNode, string toInput)
    {
        var from = Document.Graph.FindNode(fromNode);
        var schema = from is null ? null : Document.Registry.GetSchema(from.ClassType);
        if (schema is null) return null;
        for (int i = 0; i < schema.Outputs.Count; i++)
            if (Document.Graph.WhyCannotConnect(Document.Registry, fromNode, i, toNode, toInput) is null) return i;
        return null;
    }

    bool ConnectToNode(int fromNode, int fromOutput, int toNode, out string? error)
    {
        var input = FirstInputFor(fromNode, fromOutput, toNode);
        error = input is null ? "No matching input." : null;
        return input is not null && Document.Connect(fromNode, fromOutput, toNode, input, out error) is not null;
    }

    bool ConnectFromNode(int fromNode, int toNode, string toInput, out string? error)
    {
        var output = FirstOutputFor(fromNode, toNode, toInput);
        error = output is null ? "No matching output." : null;
        return output is not null && Document.Connect(fromNode, output.Value, toNode, toInput, out error) is not null;
    }

    // ---- keyboard ----

    /// <summary>Handles an editor shortcut. <paramref name="key"/> is the lower-case key name ("z", "delete", "escape"). Returns true when it was used.</summary>
    public bool Key(string key, Modifiers mods = Modifiers.None)
    {
        bool ctrl = mods.HasFlag(Modifiers.Ctrl), shift = mods.HasFlag(Modifiers.Shift);
        switch (key)
        {
            case "delete" or "backspace" when !ctrl:
                DeleteSelection();
                return true;
            case "escape":
                if (Draft is not null) { Draft = null; _mode = Mode.Idle; ViewChanged?.Invoke(); }
                else ClearSelection();
                return true;
            case "z" when ctrl && !shift:
                Document.Undo();
                return true;
            case "y" when ctrl:
            case "z" when ctrl && shift:
                Document.Redo();
                return true;
            case "a" when ctrl:
                SelectAll();
                return true;
            case "c" when ctrl:
                Copy();
                return true;
            case "v" when ctrl:
                Paste(_lastWorld, keepExternalInputs: shift);
                return true;
            case "m" when ctrl:
                ToggleMode(NodeMode.Never);
                return true;
            case "b" when ctrl:
                ToggleMode(NodeMode.Bypass);
                return true;
            case "f" or "home" when !ctrl:
                FitToView(selectionOnly: _selected.Count > 0);
                return true;
        }
        return false;
    }

    public void DeleteSelection()
    {
        if (_selected.Count == 0 && SelectedLink is null) return;
        Document.Group(() =>
        {
            if (SelectedLink is { } l) Document.Disconnect(l);
            Document.DeleteNodes(_selected.ToList());
        });
        _selected.Clear();
        SelectedLink = null;
        ViewChanged?.Invoke();
    }

    void ToggleMode(NodeMode mode)
    {
        if (_selected.Count == 0) return;
        // If every selected node already has the mode, switch it off; otherwise switch it on for all.
        bool allHave = _selected.All(id => Document.Graph.GetNode(id).Mode == mode);
        Document.Group(() =>
        {
            foreach (var id in _selected) Document.SetMode(id, allHave ? NodeMode.Always : mode);
        });
    }

    /// <summary>Asks the UI to redraw (something outside the editor changed what is shown: a result arrived, a picture loaded).</summary>
    public void Invalidate() => ViewChanged?.Invoke();

    public void FitToView(bool selectionOnly = false)
    {
        var nodes = selectionOnly ? Document.Graph.Nodes.Where(n => _selected.Contains(n.Id)) : Document.Graph.Nodes;
        Rect? all = null;
        foreach (var n in nodes)
        {
            var b = Hits.Geometry(n).Bounds;
            all = all is null ? b : Rect.Union(all.Value, b);
        }
        if (all is null) Viewport.Reset();
        else Viewport.Fit(all.Value, ScreenSize.X, ScreenSize.Y);
        ViewChanged?.Invoke();
    }

    // ---- clipboard ----

    sealed record Clipboard(List<EditorNode> Nodes, List<EditorLink> InternalLinks, List<EditorLink> ExternalInputLinks);

    public bool CanPaste => _clipboard is { Nodes.Count: > 0 };

    public void Copy()
    {
        if (_selected.Count == 0) return;
        var nodes = _selected.Select(Document.Graph.GetNode).Select(n => n.Clone()).ToList();
        var links = Document.Graph.Links.ToList();
        _clipboard = new Clipboard(
            nodes,
            links.Where(l => _selected.Contains(l.FromNode) && _selected.Contains(l.ToNode)).ToList(),
            links.Where(l => !_selected.Contains(l.FromNode) && _selected.Contains(l.ToNode)).ToList());
    }

    /// <summary>Pastes the copied nodes with their top-left at <paramref name="at"/>, with the links among them; optionally also re-links their inputs to the original sources.</summary>
    public void Paste(Point at, bool keepExternalInputs = false)
    {
        if (_clipboard is not { Nodes.Count: > 0 } clip) return;
        double minX = clip.Nodes.Min(n => n.X), minY = clip.Nodes.Min(n => n.Y);
        var map = new Dictionary<int, int>();
        var created = new List<int>();

        Document.Group(() =>
        {
            foreach (var original in clip.Nodes)
            {
                var added = Document.AddNode(original.ClassType, at.X + original.X - minX, at.Y + original.Y - minY);
                foreach (var (name, value) in original.Widgets) Document.SetWidget(added.Id, name, value);
                Document.SetTitle(added.Id, original.Title);
                if (original.Mode != NodeMode.Always) Document.SetMode(added.Id, original.Mode);
                if (original.Width is { } w) Document.SetWidth(added.Id, w);
                if (original.Collapsed) Document.SetCollapsed(added.Id, true);
                map[original.Id] = added.Id;
                created.Add(added.Id);
            }
            foreach (var l in clip.InternalLinks)
                Document.Connect(map[l.FromNode], l.FromOutput, map[l.ToNode], l.ToInput, out _);
            if (keepExternalInputs)
                foreach (var l in clip.ExternalInputLinks.Where(l => Document.Graph.FindNode(l.FromNode) is not null))
                    Document.Connect(l.FromNode, l.FromOutput, map[l.ToNode], l.ToInput, out _);
        });
        Document.EndGesture();
        _selected.Clear();
        _selected.UnionWith(created);
        SelectedLink = null;
        ViewChanged?.Invoke();
    }

    void SetCursor(string cursor)
    {
        if (Cursor == cursor) return;
        Cursor = cursor;
        ViewChanged?.Invoke();
    }
}
