using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor;

public enum HitKind { None, NodeBody, TitleBar, CollapseButton, InputSlot, OutputSlot, Widget, ResizeHandle, Link, GroupTitle, GroupResize }

/// <summary>What is under a point. <see cref="Name"/> is the slot's or widget's input/output name; <see cref="LinkId"/> is set for links; for groups <see cref="NodeId"/> is the group's id.</summary>
public readonly record struct Hit(HitKind Kind, int NodeId = 0, string? Name = null, int Index = -1, int LinkId = 0);

/// <summary>Finds what is under a world-space point: slots first, then resize handle, widgets, title, body, and last the links behind the nodes.</summary>
public sealed class HitTester(EditorDocument document, Func<string, double>? measureText = null)
{
    /// <summary>How close (world units) the pointer has to be to a slot dot to grab it.</summary>
    public double SlotRadius { get; set; } = 12;
    /// <summary>How close (world units) the pointer has to be to a link curve to pick it.</summary>
    public double LinkTolerance { get; set; } = 6;

    public NodeGeometry Geometry(EditorNode node) =>
        NodeLayout.Compute(node, document.Registry.GetSchema(node.ClassType) ?? Unknown(node.ClassType), document.Graph, measureText);

    /// <summary>A node whose type is no longer registered (a workflow from a newer version or a missing node pack) still has a body to show and move.</summary>
    static NodeSchema Unknown(string classType) => new() { Id = classType, DisplayName = classType + " (missing)" };

    /// <summary>Where a link attaches for the given endpoint, or null when the node or slot does not exist.</summary>
    public Point? OutputPosition(int nodeId, int output) =>
        document.Graph.FindNode(nodeId) is { } n ? Geometry(n).Outputs.ElementAtOrDefault(output)?.Position : null;

    public Point? InputPosition(int nodeId, string input) =>
        document.Graph.FindNode(nodeId) is { } n ? Geometry(n).Input(input)?.Position : null;

    public Hit Test(Point world)
    {
        // Topmost node first.
        foreach (var node in document.Graph.Nodes.OrderByDescending(n => n.ZIndex))
        {
            var g = Geometry(node);
            var slotZone = g.Bounds with { X = g.Bounds.X - SlotRadius, Width = g.Bounds.Width + 2 * SlotRadius };
            if (!slotZone.Contains(world)) continue;

            foreach (var s in g.Outputs)
                if (world.DistanceTo(s.Position) <= SlotRadius) return new Hit(HitKind.OutputSlot, node.Id, s.Name, s.Index);
            foreach (var s in g.Inputs)
                if (world.DistanceTo(s.Position) <= SlotRadius) return new Hit(HitKind.InputSlot, node.Id, s.Name, s.Index);

            if (!g.Bounds.Contains(world)) continue;
            if (!g.Collapsed && !node.Pinned && g.ResizeHandle.Contains(world)) return new Hit(HitKind.ResizeHandle, node.Id);
            foreach (var w in g.Widgets)
                if (w.Bounds.Contains(world)) return new Hit(HitKind.Widget, node.Id, w.Name);
            if (g.TitleBar.Contains(world))
                return !node.Pinned && world.X <= g.TitleBar.X + 22 ? new Hit(HitKind.CollapseButton, node.Id) : new Hit(HitKind.TitleBar, node.Id);
            return new Hit(HitKind.NodeBody, node.Id);
        }

        // Group boxes are behind the nodes: only their title bar and resize corner are grabbed (the rest of the box is the canvas), topmost group first.
        foreach (var group in document.Graph.Groups.Reverse())
        {
            if (!group.Pinned && group.ResizeHandle.Contains(world)) return new Hit(HitKind.GroupResize, group.Id);
            if (group.TitleBar.Contains(world)) return new Hit(HitKind.GroupTitle, group.Id);
        }

        // Links are drawn behind nodes, so they are tested last.
        EditorLink? best = null;
        double bestDistance = LinkTolerance;
        foreach (var link in document.Graph.Links)
        {
            if (OutputPosition(link.FromNode, link.FromOutput) is not { } from || InputPosition(link.ToNode, link.ToInput) is not { } to) continue;
            double d = LinkGeometry.Distance(world, from, to);
            if (d <= bestDistance) { best = link; bestDistance = d; }
        }
        return best is null ? default : new Hit(HitKind.Link, LinkId: best.Id);
    }
}
