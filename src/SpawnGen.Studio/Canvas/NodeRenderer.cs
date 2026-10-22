using SpawnDev.SpawnJS.JSObjects;
using SpawnGen.Editor;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Studio.Canvas;

/// <summary>Draws the editor state (<see cref="EditorView"/>) onto a 2D canvas in world coordinates. All state lives in the view; this only paints.</summary>
public sealed class NodeRenderer(EditorView view)
{
    readonly Dictionary<string, double> _textWidths = [];
    CanvasRenderingContext2D? _measureContext;

    /// <summary>Below this zoom the detail (text, widgets) is skipped: unreadable anyway and the cost of the interop calls adds up.</summary>
    const double DetailScale = 0.45;

    /// <summary>Measures text for layout with a cache; the layout asks for the same labels on every pointer move.</summary>
    public double MeasureText(string text)
    {
        if (_textWidths.TryGetValue(text, out var w)) return w;
        if (_measureContext is null) return text.Length * NodeLayout.EstimatedCharWidth;
        _measureContext.Font = Theme.Font;
        using var metrics = _measureContext.MeasureText(text);
        return _textWidths[text] = metrics.Width;
    }

    public void UseMeasureContext(CanvasRenderingContext2D ctx) => _measureContext = ctx;

    public void Draw(CanvasRenderingContext2D ctx, double cssWidth, double cssHeight, double pixelRatio)
    {
        var vp = view.Viewport;
        var doc = view.Document;

        // Screen space: clear, then the world transform (zoom + pan) for everything below.
        ctx.SetTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
        ctx.FillStyle = Theme.Background;
        ctx.FillRect(0, 0, cssWidth, cssHeight);
        ctx.SetTransform(vp.Scale * pixelRatio, 0, 0, vp.Scale * pixelRatio, vp.OffsetX * pixelRatio, vp.OffsetY * pixelRatio);

        var visible = vp.VisibleWorld(cssWidth, cssHeight);
        DrawGrid(ctx, visible, vp.Scale);

        var geometries = doc.Graph.Nodes.OrderBy(n => n.ZIndex).Select(n => (Node: n, Geo: view.Hits.Geometry(n))).ToList();
        var byId = geometries.ToDictionary(g => g.Node.Id, g => g.Geo);

        DrawLinks(ctx, byId, vp.Scale);
        foreach (var (node, geo) in geometries)
            if (geo.Bounds.Intersects(Pad(visible, 40))) DrawNode(ctx, node, geo, vp.Scale);

        DrawDraft(ctx, byId);
        if (view.SelectionBox is { } box)
        {
            ctx.FillStyle = "rgba(79,143,232,0.12)";
            ctx.FillRect(box.X, box.Y, box.Width, box.Height);
            ctx.StrokeStyle = "rgba(79,143,232,0.8)";
            ctx.LineWidth = 1 / vp.Scale;
            ctx.StrokeRect(box.X, box.Y, box.Width, box.Height);
        }
    }

    static Rect Pad(Rect r, double p) => new(r.X - p, r.Y - p, r.Width + 2 * p, r.Height + 2 * p);

    // ---- grid ----

    static void DrawGrid(CanvasRenderingContext2D ctx, Rect visible, double scale)
    {
        const double minor = 20, major = 100;
        ctx.LineWidth = 1 / scale;
        if (scale >= 0.5) DrawGridLines(ctx, visible, minor, Theme.GridMinor);
        DrawGridLines(ctx, visible, major, Theme.GridMajor);
    }

    static void DrawGridLines(CanvasRenderingContext2D ctx, Rect v, double step, string color)
    {
        ctx.StrokeStyle = color;
        ctx.BeginPath();
        for (double x = Math.Floor(v.X / step) * step; x <= v.Right; x += step) { ctx.MoveTo(x, v.Y); ctx.LineTo(x, v.Bottom); }
        for (double y = Math.Floor(v.Y / step) * step; y <= v.Bottom; y += step) { ctx.MoveTo(v.X, y); ctx.LineTo(v.Right, y); }
        ctx.Stroke();
    }

    // ---- links ----

    void DrawLinks(CanvasRenderingContext2D ctx, Dictionary<int, NodeGeometry> byId, double scale)
    {
        foreach (var link in view.Document.Graph.Links)
        {
            if (!byId.TryGetValue(link.FromNode, out var from) || !byId.TryGetValue(link.ToNode, out var to)) continue;
            var output = from.Outputs.ElementAtOrDefault(link.FromOutput);
            var input = to.Input(link.ToInput);
            if (output is null || input is null) continue;
            bool selected = view.SelectedLink == link.Id;
            DrawCurve(ctx, output.Position, input.Position, Theme.SocketColor(output.Type.Name), selected ? 4.5 : 3, selected ? 1 : 0.9, scale);
        }
    }

    static void DrawCurve(CanvasRenderingContext2D ctx, Point from, Point to, string color, double width, double alpha, double scale)
    {
        var (p0, c1, c2, p3) = LinkGeometry.Curve(from, to);
        ctx.GlobalAlpha = alpha;
        ctx.StrokeStyle = color;
        ctx.LineWidth = width;
        ctx.BeginPath();
        ctx.MoveTo(p0.X, p0.Y);
        ctx.BezierCurveTo(c1.X, c1.Y, c2.X, c2.Y, p3.X, p3.Y);
        ctx.Stroke();
        ctx.GlobalAlpha = 1;
    }

    void DrawDraft(CanvasRenderingContext2D ctx, Dictionary<int, NodeGeometry> byId)
    {
        if (view.Draft is not { } d || !byId.TryGetValue(d.NodeId, out var anchorNode)) return;
        Point start;
        string type;
        if (d.StartsAtOutput)
        {
            var o = anchorNode.Outputs.ElementAtOrDefault(d.Output);
            if (o is null) return;
            (start, type) = (o.Position, o.Type.Name);
        }
        else
        {
            var i = anchorNode.Input(d.Input!);
            if (i is null) return;
            (start, type) = (i.Position, i.Type.Name);
        }

        var end = d.Cursor;
        if (d.Target is { } t && byId.TryGetValue(t.NodeId, out var targetNode))
        {
            var slot = t.Kind == HitKind.InputSlot ? targetNode.Input(t.Name!) : targetNode.Outputs.ElementAtOrDefault(t.Index);
            if (slot is not null) end = slot.Position;
        }
        var color = d.Problem is null ? Theme.SocketColor(type) : Theme.Error;
        if (d.StartsAtOutput) DrawCurve(ctx, start, end, color, 3, 0.85, view.Viewport.Scale);
        else DrawCurve(ctx, end, start, color, 3, 0.85, view.Viewport.Scale);

        if (d.Target is { } target && byId.TryGetValue(target.NodeId, out var tn))
        {
            var slot = target.Kind == HitKind.InputSlot ? tn.Input(target.Name!) : tn.Outputs.ElementAtOrDefault(target.Index);
            if (slot is not null) Ring(ctx, slot.Position, 9, "#ffffff");
        }
        else if (d.Problem is not null)
        {
            ctx.FillStyle = Theme.Error;
            ctx.Font = Theme.Font;
            ctx.TextBaseline = "middle";
            ctx.FillText(d.Problem, d.Cursor.X + 14, d.Cursor.Y);
        }
    }

    // ---- nodes ----

    void DrawNode(CanvasRenderingContext2D ctx, EditorNode node, NodeGeometry geo, double scale)
    {
        bool selected = view.SelectedNodes.Contains(node.Id);
        var schema = view.Document.Registry.GetSchema(node.ClassType);
        bool detailed = scale >= DetailScale;
        var b = geo.Bounds;

        ctx.GlobalAlpha = node.Mode == NodeMode.Never ? 0.4 : node.Mode == NodeMode.Bypass ? 0.55 : 1;
        // Body and title.
        RoundRect(ctx, b, 8);
        ctx.FillStyle = node.Mode == NodeMode.Bypass ? "#4a3a5a" : Theme.NodeBody;
        ctx.Fill();
        if (!geo.Collapsed || true)
        {
            ctx.Save();
            RoundRect(ctx, b, 8);
            ctx.Clip();
            ctx.FillStyle = schema is null ? "#5a2a2a" : Theme.TitleColor(schema.Category);
            ctx.FillRect(geo.TitleBar.X, geo.TitleBar.Y, geo.TitleBar.Width, geo.TitleBar.Height);
            ctx.Restore();
        }
        RoundRect(ctx, b, 8);
        ctx.LineWidth = selected ? 2 / scale + 0.5 : 1.5;
        ctx.StrokeStyle = selected ? Theme.NodeSelected : (schema is null ? Theme.Error : Theme.NodeBorder);
        ctx.Stroke();

        if (detailed)
        {
            ctx.Font = Theme.TitleFont;
            ctx.TextBaseline = "middle";
            ctx.FillStyle = Theme.TitleText;
            ctx.FillText(geo.Title, b.X + 26, geo.TitleBar.Y + geo.TitleBar.Height / 2);
            // Collapse arrow.
            ctx.FillStyle = Theme.LabelText;
            ctx.BeginPath();
            double ax = b.X + 12, ay = geo.TitleBar.Y + geo.TitleBar.Height / 2;
            if (geo.Collapsed) { ctx.MoveTo(ax - 3, ay - 5); ctx.LineTo(ax + 4, ay); ctx.LineTo(ax - 3, ay + 5); }
            else { ctx.MoveTo(ax - 5, ay - 3); ctx.LineTo(ax + 5, ay - 3); ctx.LineTo(ax, ay + 4); }
            ctx.Fill();
        }

        if (!geo.Collapsed)
        {
            foreach (var s in geo.Inputs) DrawSlot(ctx, s, detailed, view.Hover is { Kind: HitKind.InputSlot } h && h.NodeId == node.Id && h.Name == s.Name);
            foreach (var s in geo.Outputs) DrawSlot(ctx, s, detailed, view.Hover is { Kind: HitKind.OutputSlot } h && h.NodeId == node.Id && h.Index == s.Index);
            if (detailed) foreach (var w in geo.Widgets) DrawWidget(ctx, node, w);
            // Resize grip.
            var r = geo.ResizeHandle;
            ctx.StrokeStyle = "#5a5a5a";
            ctx.LineWidth = 1.2;
            ctx.BeginPath();
            ctx.MoveTo(r.Right - 3, r.Y + 4); ctx.LineTo(r.X + 4, r.Bottom - 3);
            ctx.MoveTo(r.Right - 3, r.Y + 9); ctx.LineTo(r.X + 9, r.Bottom - 3);
            ctx.Stroke();
        }
        else
        {
            foreach (var s in geo.Outputs.Take(1)) Dot(ctx, s.Position, 5, Theme.SocketColor(s.Type.Name), s.IsConnected);
            foreach (var s in geo.Inputs.Take(1)) Dot(ctx, s.Position, 5, Theme.SocketColor(s.Type.Name), s.IsConnected);
        }
        ctx.GlobalAlpha = 1;
    }

    void DrawSlot(CanvasRenderingContext2D ctx, SlotGeometry s, bool detailed, bool hovered)
    {
        // An unlinked widget input is only a drop target: its row is the widget. Show the dot while a link is being dragged, or when pointed at.
        if (s.IsWidgetInput && !s.IsConnected)
        {
            if (view.Draft is { StartsAtOutput: true } || hovered) Dot(ctx, s.Position, hovered ? 6.5 : 4.5, Theme.SocketColor(s.Type.Name), false);
            return;
        }
        var color = Theme.SocketColor(s.Type.Name);
        Dot(ctx, s.Position, hovered ? 6.5 : 5, color, s.IsConnected);
        if (!detailed) return;
        ctx.Font = Theme.Font;
        ctx.TextBaseline = "middle";
        ctx.FillStyle = Theme.LabelText;
        if (s.Kind == SlotKind.Input)
        {
            ctx.TextAlign = "left";
            ctx.FillText(s.Name, s.Position.X + 12, s.Position.Y);
        }
        else
        {
            ctx.TextAlign = "right";
            ctx.FillText(s.Name, s.Position.X - 12, s.Position.Y);
        }
        ctx.TextAlign = "left";
    }

    void DrawWidget(CanvasRenderingContext2D ctx, EditorNode node, WidgetRow w)
    {
        var r = w.Bounds;
        node.Widgets.TryGetValue(w.Name, out var value);
        value ??= w.Spec.Default;

        RoundRect(ctx, r, 6);
        ctx.FillStyle = Theme.WidgetFill;
        ctx.Fill();
        ctx.LineWidth = 1;
        ctx.StrokeStyle = Theme.WidgetBorder;
        ctx.Stroke();

        ctx.Font = Theme.Font;
        ctx.TextBaseline = "middle";
        double cy = r.Y + (w.Spec.Multiline ? 12 : r.Height / 2);

        if (w.Spec.Multiline)
        {
            ctx.FillStyle = Theme.LabelText;
            ctx.FillText(w.Name, r.X + 8, cy);
            ctx.FillStyle = Theme.ValueText;
            // The first lines of the text, clipped to the box.
            ctx.Save();
            ctx.BeginPath();
            ctx.Rect(r.X + 4, r.Y + 22, r.Width - 8, r.Height - 26);
            ctx.Clip();
            int line = 0;
            foreach (var l in WrapLines(value?.ToString() ?? "", r.Width - 16).Take(4))
                ctx.FillText(l, r.X + 8, r.Y + 32 + line++ * 15);
            ctx.Restore();
            return;
        }

        string text = w.Spec.Kind switch
        {
            WidgetKind.Float => Convert.ToDouble(value ?? 0.0).ToString("0.###", System.Globalization.CultureInfo.InvariantCulture),
            WidgetKind.Boolean => (value as bool? ?? false) ? "on" : "off",
            _ => value?.ToString() ?? "",
        };
        ctx.FillStyle = Theme.LabelText;
        ctx.TextAlign = "left";
        ctx.FillText(w.Name, r.X + 10, cy);
        ctx.FillStyle = Theme.ValueText;
        ctx.TextAlign = "right";
        double room = r.Width - MeasureText(w.Name) - 34;
        ctx.FillText(Ellipsize(text, room), r.Right - 10, cy);
        ctx.TextAlign = "left";

        if (w.Spec.Kind is WidgetKind.Int or WidgetKind.Float or WidgetKind.Combo)
        {
            ctx.FillStyle = "#6f6f6f";
            ctx.BeginPath(); // little arrows at the edges, as ComfyUI's number and combo widgets have
            ctx.MoveTo(r.X + 4, cy); ctx.LineTo(r.X + 8, cy - 3); ctx.LineTo(r.X + 8, cy + 3);
            ctx.MoveTo(r.Right - 4, cy); ctx.LineTo(r.Right - 8, cy - 3); ctx.LineTo(r.Right - 8, cy + 3);
            ctx.Fill();
        }
    }

    string Ellipsize(string text, double width)
    {
        if (width <= 8 || MeasureText(text) <= width) return text;
        while (text.Length > 1 && MeasureText(text + "...") > width) text = text[..^1];
        return text + "...";
    }

    IEnumerable<string> WrapLines(string text, double width)
    {
        foreach (var paragraph in text.Split('\n'))
        {
            var line = "";
            foreach (var word in paragraph.Split(' '))
            {
                var next = line.Length == 0 ? word : line + " " + word;
                if (MeasureText(next) > width && line.Length > 0) { yield return line; line = word; }
                else line = next;
            }
            yield return line;
        }
    }

    // ---- primitives ----

    static void Dot(CanvasRenderingContext2D ctx, Point p, double radius, string color, bool filled)
    {
        ctx.BeginPath();
        ctx.Arc(p.X, p.Y, radius, 0, Math.PI * 2);
        if (filled) { ctx.FillStyle = color; ctx.Fill(); }
        else { ctx.FillStyle = Theme.NodeBody; ctx.Fill(); }
        ctx.LineWidth = 2;
        ctx.StrokeStyle = color;
        ctx.Stroke();
    }

    static void Ring(CanvasRenderingContext2D ctx, Point p, double radius, string color)
    {
        ctx.BeginPath();
        ctx.Arc(p.X, p.Y, radius, 0, Math.PI * 2);
        ctx.LineWidth = 2;
        ctx.StrokeStyle = color;
        ctx.Stroke();
    }

    /// <summary>Starts a new path shaped as a rounded rectangle (canvas 2D has no portable RoundRect, so it is built from arcs).</summary>
    static void RoundRect(CanvasRenderingContext2D ctx, Rect r, double radius)
    {
        double rad = Math.Min(radius, Math.Min(r.Width, r.Height) / 2);
        ctx.BeginPath();
        ctx.MoveTo(r.X + rad, r.Y);
        ctx.ArcTo(r.Right, r.Y, r.Right, r.Bottom, rad);
        ctx.ArcTo(r.Right, r.Bottom, r.X, r.Bottom, rad);
        ctx.ArcTo(r.X, r.Bottom, r.X, r.Y, rad);
        ctx.ArcTo(r.X, r.Y, r.Right, r.Y, rad);
        ctx.ClosePath();
    }
}
