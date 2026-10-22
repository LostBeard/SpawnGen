using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor;

public enum SlotKind { Input, Output }

/// <summary>A connection point on a node: where links attach, and what it carries.</summary>
public sealed record SlotGeometry(SlotKind Kind, string Name, int Index, SocketType Type, Point Position, bool IsWidgetInput, bool IsConnected);

/// <summary>One widget row (a number, text box, combo or toggle) inside a node body.</summary>
public sealed record WidgetRow(string Name, WidgetSpec Spec, Rect Bounds);

/// <summary>Everything the renderer and hit tester need to know about where one node's parts are, in world units.</summary>
public sealed class NodeGeometry
{
    public required int NodeId { get; init; }
    public required string Title { get; init; }
    public required Rect Bounds { get; init; }
    public required Rect TitleBar { get; init; }
    public required bool Collapsed { get; init; }
    public required IReadOnlyList<SlotGeometry> Inputs { get; init; }
    public required IReadOnlyList<SlotGeometry> Outputs { get; init; }
    public required IReadOnlyList<WidgetRow> Widgets { get; init; }
    /// <summary>Where the result image is drawn, when the node has one.</summary>
    public Rect? PreviewArea { get; init; }
    /// <summary>Where the node's result text is drawn (null when it has none); <see cref="TextLines"/> is how many lines fit.</summary>
    public Rect? TextArea { get; init; }
    /// <summary>The bottom-right grab area for resizing the width.</summary>
    public required Rect ResizeHandle { get; init; }

    public SlotGeometry? Input(string name) => Inputs.FirstOrDefault(s => s.Name == name);
}

/// <summary>Sizes and places the parts of nodes. The metrics are world units; the renderer scales them with the viewport.</summary>
public static class NodeLayout
{
    public const double TitleHeight = 30;
    public const double SlotRowHeight = 22;
    public const double WidgetRowHeight = 26;
    public const double MultilineWidgetHeight = 92;
    public const double Padding = 10;
    public const double BottomPadding = 8;
    public const double MinWidth = 190;
    public const double ResizeSize = 14;
    public const double MaxPreviewHeight = 520;
    public const int MaxTextLines = 16;
    public const double TextLineHeight = 16;
    /// <summary>Average glyph width used to size nodes before the real text is measured; the UI may pass a better measure.</summary>
    public const double EstimatedCharWidth = 7.2;

    /// <summary>
    /// Computes the geometry of <paramref name="node"/>. Socket inputs and any widget input that currently has a link get a slot row on the left,
    /// outputs on the right; the remaining widgets stack below. A widget input without a link still has a drop target at the left edge of its row.
    /// </summary>
    public static NodeGeometry Compute(EditorNode node, NodeSchema schema, EditorGraph graph, Func<string, double>? measureText = null)
    {
        measureText ??= s => s.Length * EstimatedCharWidth;
        var title = node.Title ?? schema.DisplayName;
        double titleWidth = measureText(title) + 2 * Padding + 20;

        if (node.Collapsed)
        {
            var pill = new Rect(node.X, node.Y, Math.Max(titleWidth, 90), TitleHeight);
            var left = new Point(pill.X, pill.Center.Y);
            var right = new Point(pill.Right, pill.Center.Y);
            return new NodeGeometry
            {
                NodeId = node.Id, Title = title, Bounds = pill, TitleBar = pill, Collapsed = true, ResizeHandle = new Rect(pill.Right, pill.Bottom, 0, 0),
                // A collapsed node keeps all its links: they attach to the ends of the pill.
                Inputs = schema.Inputs.Where(i => IsSlotRow(i, node, graph) || graph.LinkInto(node.Id, i.Name) is not null)
                    .Select((i, n) => new SlotGeometry(SlotKind.Input, i.Name, n, i.Type, left, i.Widget is not null, graph.LinkInto(node.Id, i.Name) is not null)).ToList(),
                Outputs = schema.Outputs.Select((o, n) => new SlotGeometry(SlotKind.Output, o.Name, n, o.Type, right, false, graph.LinksFrom(node.Id, n).Any())).ToList(),
                Widgets = [],
            };
        }

        if (schema.Id == "Reroute") return ComputeReroute(node, schema, graph);

        var slotInputs = schema.Inputs.Where(i => IsSlotRow(i, node, graph)).ToList();
        var widgetInputs = schema.Inputs.Where(i => i.Widget is not null && !IsSlotRow(i, node, graph)).ToList();
        int slotRows = Math.Max(slotInputs.Count, schema.Outputs.Count);

        double width = node.Width ?? Math.Max(MinWidth, Math.Max(titleWidth,
            slotInputs.Select(i => measureText(i.Name)).DefaultIfEmpty(0).Max() + schema.Outputs.Select(o => measureText(o.Name)).DefaultIfEmpty(0).Max() + 3 * Padding + 16));
        width = Math.Max(width, 90);

        double slotsTop = node.Y + TitleHeight;
        var inputs = new List<SlotGeometry>();
        for (int i = 0; i < slotInputs.Count; i++)
        {
            var spec = slotInputs[i];
            inputs.Add(new SlotGeometry(SlotKind.Input, spec.Name, i, spec.Type,
                new Point(node.X, slotsTop + i * SlotRowHeight + SlotRowHeight / 2), spec.Widget is not null, graph.LinkInto(node.Id, spec.Name) is not null));
        }
        var outputs = new List<SlotGeometry>();
        for (int i = 0; i < schema.Outputs.Count; i++)
        {
            var spec = schema.Outputs[i];
            outputs.Add(new SlotGeometry(SlotKind.Output, spec.Name, i, spec.Type,
                new Point(node.X + width, slotsTop + i * SlotRowHeight + SlotRowHeight / 2), false, graph.LinksFrom(node.Id, i).Any()));
        }

        double y = slotsTop + slotRows * SlotRowHeight + (slotRows > 0 ? 4 : 0);
        var widgets = new List<WidgetRow>();
        foreach (var spec in widgetInputs)
        {
            double h = spec.Widget!.Multiline ? MultilineWidgetHeight : WidgetRowHeight;
            var bounds = new Rect(node.X + Padding, y, width - 2 * Padding, h - 4);
            widgets.Add(new WidgetRow(spec.Name, spec.Widget, bounds));
            // The drop target for linking a value into this widget: the left edge of its row.
            inputs.Add(new SlotGeometry(SlotKind.Input, spec.Name, inputs.Count, spec.Type, new Point(node.X, bounds.Center.Y), true, false));
            y += h;
            if (ControlAfterGenerate.HasControl(spec))
            {
                // ComfyUI's "control after generate" combo sits right under the seed. It is not an input: no slot, no link.
                widgets.Add(new WidgetRow(ControlAfterGenerate.Name, ControlAfterGenerate.Spec, new Rect(node.X + Padding, y, width - 2 * Padding, WidgetRowHeight - 4)));
                y += WidgetRowHeight;
            }
        }

        // The result picture, under the widgets: as wide as the node, as tall as its aspect ratio asks (until it has loaded it is assumed square).
        Rect? preview = null;
        if (node.Previews.Count > 0)
        {
            var first = node.Previews[0];
            double pw = Math.Max(40, width - 2 * Padding);
            double aspect = first.Width > 0 && first.Height > 0 ? first.Height / first.Width : 1;
            double ph = Math.Clamp(pw * aspect, 40, MaxPreviewHeight);
            preview = new Rect(node.X + Padding, y + 2, pw, ph);
            y += ph + 8;
        }

        Rect? text = null;
        if (!string.IsNullOrEmpty(node.ResultText))
        {
            int lines = Math.Clamp(node.ResultText.Split('\n').Length, 1, MaxTextLines);
            double th = lines * TextLineHeight + 12;
            text = new Rect(node.X + Padding, y + 2, Math.Max(40, width - 2 * Padding), th);
            y += th + 8;
        }

        double height = (y - node.Y) + BottomPadding;
        var boundsAll = new Rect(node.X, node.Y, width, height);
        return new NodeGeometry
        {
            NodeId = node.Id, Title = title, Bounds = boundsAll, TitleBar = new Rect(node.X, node.Y, width, TitleHeight), Collapsed = false,
            Inputs = inputs, Outputs = outputs, Widgets = widgets, PreviewArea = preview, TextArea = text,
            ResizeHandle = new Rect(boundsAll.Right - ResizeSize, boundsAll.Bottom - ResizeSize, ResizeSize, ResizeSize),
        };
    }

    public const double RerouteWidth = 75, RerouteHeight = 26;

    /// <summary>A reroute is just its two dots in a small body: no title, no widgets.</summary>
    static NodeGeometry ComputeReroute(EditorNode node, NodeSchema schema, EditorGraph graph)
    {
        var bounds = new Rect(node.X, node.Y, RerouteWidth, RerouteHeight);
        var input = schema.Inputs[0];
        var output = schema.Outputs[0];
        return new NodeGeometry
        {
            NodeId = node.Id, Title = "", Bounds = bounds, TitleBar = new Rect(node.X, node.Y, RerouteWidth, 0), Collapsed = false,
            Inputs = [new SlotGeometry(SlotKind.Input, input.Name, 0, input.Type, new Point(bounds.X, bounds.Center.Y), false, graph.LinkInto(node.Id, input.Name) is not null)],
            Outputs = [new SlotGeometry(SlotKind.Output, output.Name, 0, output.Type, new Point(bounds.Right, bounds.Center.Y), false, graph.LinksFrom(node.Id, 0).Any())],
            Widgets = [], ResizeHandle = new Rect(bounds.Right, bounds.Bottom, 0, 0),
        };
    }

    /// <summary>A socket input, or a widget input that is fed by a link right now (its value is shown by the link, not by a widget).</summary>
    static bool IsSlotRow(InputSpec input, EditorNode node, EditorGraph graph) =>
        input.Widget is null || graph.LinkInto(node.Id, input.Name) is not null;
}
