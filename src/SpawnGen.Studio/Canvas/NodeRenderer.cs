using SpawnDev.SpawnJS.JSObjects;
using SpawnGen.Editor;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Studio.Canvas;

/// <summary>Draws the editor state (<see cref="EditorView"/>) onto a 2D canvas in world coordinates. All state lives in the view; this only paints.</summary>
public sealed class NodeRenderer(EditorView view, SpawnGen.Studio.Run.RunVisuals visuals, ImageCache images)
{
    CanvasRenderingContext2D? _measureContext;

    /// <summary>Below this zoom the detail (text, widgets) is skipped: unreadable anyway and the cost of the interop calls adds up.</summary>
    const double DetailScale = 0.45;

    readonly Dictionary<char, double> _charWidths = new();

    /// <summary>
    /// Measures text for layout. Each distinct character is measured once (a call into the browser) and strings are summed from those widths, so laying out hundreds of nodes
    /// with their own numbers and names costs no more calls once the alphabet has been seen. Kerning is ignored: layout only needs the width to a pixel or two.
    /// </summary>
    public double MeasureText(string text)
    {
        if (_measureContext is null) return text.Length * NodeLayout.EstimatedCharWidth;
        double total = 0;
        foreach (char c in text)
        {
            if (!_charWidths.TryGetValue(c, out var w))
            {
                _measureContext.Font = Theme.Font;
                using var metrics = _measureContext.MeasureText(c.ToString());
                _charWidths[c] = w = metrics.Width;
            }
            total += w;
        }
        return total;
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

        DrawGroups(ctx, visible, vp.Scale);

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

        if (view.MinimapVisible) DrawMinimap(ctx, geometries.Select(g => (g.Node, g.Geo)).ToList(), cssWidth, cssHeight, pixelRatio);
    }

    /// <summary>The overview in the bottom right corner: groups, nodes (the selected ones lit) and the rectangle of what the canvas shows now.</summary>
    void DrawMinimap(CanvasRenderingContext2D ctx, List<(EditorNode Node, NodeGeometry Geo)> geometries, double cssWidth, double cssHeight, double pixelRatio)
    {
        ctx.SetTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
        var r = view.MinimapScreenRect;
        var (scale, _) = view.MinimapMapping();
        RoundRect(ctx, r, 6);
        ctx.FillStyle = "rgba(20,20,20,0.96)";
        ctx.Fill();
        ctx.StrokeStyle = "#555";
        ctx.LineWidth = 1;
        ctx.Stroke();
        ctx.Save();
        RoundRect(ctx, r, 6);
        ctx.Clip();
        foreach (var g in view.Document.Graph.Groups)
        {
            var a = view.WorldToMinimap(new Point(g.X, g.Y));
            ctx.FillStyle = "rgba(120,150,190,0.25)";
            ctx.FillRect(a.X, a.Y, g.Width * scale, g.Height * scale);
        }
        var selected = view.SelectedNodes;
        foreach (var (node, geo) in geometries)
        {
            var a = view.WorldToMinimap(new Point(geo.Bounds.X, geo.Bounds.Y));
            ctx.FillStyle = selected.Contains(node.Id) ? "#4f8fe8" : "#7a7a7a";
            ctx.FillRect(a.X, a.Y, Math.Max(2, geo.Bounds.Width * scale), Math.Max(2, geo.Bounds.Height * scale));
        }
        var vis = view.Viewport.VisibleWorld(cssWidth, cssHeight);
        var v = view.WorldToMinimap(new Point(vis.X, vis.Y));
        ctx.StrokeStyle = "#f0f0f0";
        ctx.LineWidth = 1.5;
        ctx.StrokeRect(v.X, v.Y, vis.Width * scale, vis.Height * scale);
        ctx.Restore();
    }

    // ---- groups ----

    /// <summary>A small push-pin: a round head over a needle, tilted.</summary>
    static void DrawPin(CanvasRenderingContext2D ctx, double x, double y)
    {
        ctx.StrokeStyle = "#d9d9d9";
        ctx.FillStyle = "#d9d9d9";
        ctx.LineWidth = 1.6;
        ctx.BeginPath();
        ctx.MoveTo(x - 2, y + 6); ctx.LineTo(x + 1.5, y + 0.5);
        ctx.Stroke();
        ctx.BeginPath();
        ctx.Arc(x + 3, y - 2.5, 3.6, 0, Math.PI * 2);
        ctx.Fill();
    }

    void DrawGroups(CanvasRenderingContext2D ctx, Rect visible, double scale)
    {
        foreach (var group in view.Document.Graph.Groups)
        {
            var b = group.Bounds;
            if (!b.Intersects(visible)) continue;
            bool selected = view.SelectedGroup == group.Id;

            RoundRect(ctx, b, 10);
            ctx.FillStyle = Rgba(group.Color, 0.16);
            ctx.Fill();
            // The title bar is a stronger band of the same colour.
            ctx.Save();
            RoundRect(ctx, b, 10);
            ctx.Clip();
            ctx.FillStyle = Rgba(group.Color, 0.38);
            ctx.FillRect(b.X, b.Y, b.Width, EditorGroup.TitleHeight);
            ctx.Restore();
            RoundRect(ctx, b, 10);
            ctx.LineWidth = selected ? 3 : 1.5;
            ctx.StrokeStyle = selected ? Theme.NodeSelected : Rgba(group.Color, 0.75);
            ctx.Stroke();

            if (scale >= DetailScale)
            {
                ctx.Font = $"600 {Math.Min(group.FontSize, 20):0}px system-ui, 'Segoe UI', sans-serif";
                ctx.TextBaseline = "middle";
                ctx.FillStyle = Theme.TitleText;
                ctx.FillText(Ellipsize(group.Title, b.Width - 44), b.X + 12, b.Y + EditorGroup.TitleHeight / 2);
                if (group.Pinned) DrawPin(ctx, b.Right - 16, b.Y + EditorGroup.TitleHeight / 2);
                var r = group.ResizeHandle;
                ctx.StrokeStyle = Rgba(group.Color, 0.9);
                ctx.LineWidth = 1.5;
                ctx.BeginPath();
                ctx.MoveTo(r.Right - 4, r.Y + 6); ctx.LineTo(r.X + 6, r.Bottom - 4);
                ctx.MoveTo(r.Right - 4, r.Y + 11); ctx.LineTo(r.X + 11, r.Bottom - 4);
                ctx.Stroke();
            }
        }
    }

    /// <summary>"#rgb" or "#rrggbb" as a CSS colour with the given opacity.</summary>
    static string Rgba(string hex, double alpha)
    {
        var h = hex.TrimStart('#');
        if (h.Length == 3) h = string.Concat(h.Select(c => new string(c, 2)));
        if (h.Length != 6 || !int.TryParse(h, System.Globalization.NumberStyles.HexNumber, null, out var v)) return $"rgba(63,120,158,{alpha})";
        return $"rgba({(v >> 16) & 255},{(v >> 8) & 255},{v & 255},{alpha.ToString(System.Globalization.CultureInfo.InvariantCulture)})";
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

    /// <summary>One link in the current link style; a link being dragged (<paramref name="draft"/>) is always shown, even when links are hidden.</summary>
    static void DrawCurve(CanvasRenderingContext2D ctx, Point from, Point to, string color, double width, double alpha, double scale, bool draft = false)
    {
        var style = LinkGeometry.Style;
        if (style == LinkStyle.Hidden && !draft) return;
        ctx.GlobalAlpha = alpha;
        ctx.StrokeStyle = color;
        ctx.LineWidth = width;
        ctx.BeginPath();
        if (style is LinkStyle.Straight or LinkStyle.Linear)
        {
            var pts = LinkGeometry.Sample(from, to);
            ctx.MoveTo(pts[0].X, pts[0].Y);
            for (int i = 1; i < pts.Count; i++) ctx.LineTo(pts[i].X, pts[i].Y);
        }
        else
        {
            var (p0, c1, c2, p3) = LinkGeometry.Curve(from, to);
            ctx.MoveTo(p0.X, p0.Y);
            ctx.BezierCurveTo(c1.X, c1.Y, c2.X, c2.Y, p3.X, p3.Y);
        }
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
        if (d.StartsAtOutput) DrawCurve(ctx, start, end, color, 3, 0.85, view.Viewport.Scale, draft: true);
        else DrawCurve(ctx, end, start, color, 3, 0.85, view.Viewport.Scale, draft: true);

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
        bool isNote = schema is { FrontendOnly: true } && schema.Id != "Reroute";
        ctx.FillStyle = node.Mode == NodeMode.Bypass ? "#4a3a5a" : node.BgColor ?? (isNote ? "#4d4630" : Theme.NodeBody);
        ctx.Fill();
        if (!geo.Collapsed || true)
        {
            ctx.Save();
            RoundRect(ctx, b, 8);
            ctx.Clip();
            ctx.FillStyle = schema is null ? "#5a2a2a" : node.Color ?? (isNote ? "#7a6a2a" : Theme.TitleColor(schema.Category));
            ctx.FillRect(geo.TitleBar.X, geo.TitleBar.Y, geo.TitleBar.Width, geo.TitleBar.Height);
            ctx.Restore();
        }
        bool running = visuals.ExecutingNode == node.Id;
        bool failed = visuals.Errors.ContainsKey(node.Id);
        RoundRect(ctx, b, 8);
        ctx.LineWidth = running || failed ? 3 : selected ? 2 / scale + 0.5 : 1.5;
        ctx.StrokeStyle = failed ? Theme.Error : running ? Theme.Running : selected ? Theme.NodeSelected : (schema is null ? Theme.Error : Theme.NodeBorder);
        ctx.Stroke();

        // Progress along the bottom of the title bar while this node runs (a node that reports no steps just shows the green outline).
        if (running && visuals.ProgressMax > 0)
        {
            ctx.FillStyle = Theme.Running;
            ctx.FillRect(geo.TitleBar.X + 1, geo.TitleBar.Bottom - 4, (geo.TitleBar.Width - 2) * visuals.Fraction, 3);
        }

        if (detailed && geo.TitleBar.Height > 0)
        {
            ctx.Font = Theme.TitleFont;
            ctx.TextBaseline = "middle";
            ctx.FillStyle = Theme.TitleText;
            ctx.FillText(geo.Title, b.X + 26, geo.TitleBar.Y + geo.TitleBar.Height / 2);
            if (node.Pinned) DrawPin(ctx, b.Right - 14, geo.TitleBar.Y + geo.TitleBar.Height / 2);
            // Collapse arrow.
            ctx.FillStyle = Theme.LabelText;
            ctx.BeginPath();
            double ax = b.X + 12, ay = geo.TitleBar.Y + geo.TitleBar.Height / 2;
            if (geo.Collapsed) { ctx.MoveTo(ax - 3, ay - 5); ctx.LineTo(ax + 4, ay); ctx.LineTo(ax - 3, ay + 5); }
            else { ctx.MoveTo(ax - 5, ay - 3); ctx.LineTo(ax + 5, ay - 3); ctx.LineTo(ax, ay + 4); }
            ctx.Fill();
        }

        if (schema?.Id == "Reroute")
        {
            // Just the two dots: no names, no grip.
            foreach (var s in geo.Inputs.Concat(geo.Outputs)) Dot(ctx, s.Position, 5, Theme.SocketColor(s.Type.Name), s.IsConnected);
        }
        else if (!geo.Collapsed)
        {
            foreach (var s in geo.Inputs) DrawSlot(ctx, s, detailed, view.Hover is { Kind: HitKind.InputSlot } h && h.NodeId == node.Id && h.Name == s.Name);
            foreach (var s in geo.Outputs) DrawSlot(ctx, s, detailed, view.Hover is { Kind: HitKind.OutputSlot } h && h.NodeId == node.Id && h.Index == s.Index);
            if (detailed) foreach (var w in geo.Widgets) DrawWidget(ctx, node, w);
            if (geo.PreviewArea is { } area) DrawPreview(ctx, node, area, detailed);
            if (geo.TextArea is { } textArea && detailed) DrawResultText(ctx, geo.TextLines, textArea);
            if (!node.Pinned)
            {
                // Resize grip.
                var r = geo.ResizeHandle;
                ctx.StrokeStyle = "#5a5a5a";
                ctx.LineWidth = 1.2;
                ctx.BeginPath();
                ctx.MoveTo(r.Right - 3, r.Y + 4); ctx.LineTo(r.X + 4, r.Bottom - 3);
                ctx.MoveTo(r.Right - 3, r.Y + 9); ctx.LineTo(r.X + 9, r.Bottom - 3);
                ctx.Stroke();
            }
        }
        else
        {
            foreach (var s in geo.Outputs.Take(1)) Dot(ctx, s.Position, 5, Theme.SocketColor(s.Type.Name), s.IsConnected);
            foreach (var s in geo.Inputs.Take(1)) Dot(ctx, s.Position, 5, Theme.SocketColor(s.Type.Name), s.IsConnected);
        }
        ctx.GlobalAlpha = 1;

        // A failed node says why, in a red tag under it.
        if (failed && detailed)
        {
            var text = Ellipsize(visuals.Errors[node.Id], Math.Max(b.Width, 220));
            ctx.Font = Theme.Font;
            double w = Math.Max(MeasureText(text) + 16, 60);
            RoundRect(ctx, new Rect(b.X, b.Bottom + 6, w, 22), 5);
            ctx.FillStyle = "#6b2424";
            ctx.Fill();
            ctx.FillStyle = "#ffd6d6";
            ctx.TextBaseline = "middle";
            ctx.FillText(text, b.X + 8, b.Bottom + 17);
        }
    }

    /// <summary>The node's result text, one line per row in a dark box; what does not fit is cut off with an ellipsis.</summary>
    void DrawResultText(CanvasRenderingContext2D ctx, IReadOnlyList<string> lines, Rect area)
    {
        RoundRect(ctx, area, 5);
        ctx.FillStyle = "#141414";
        ctx.Fill();
        ctx.Save();
        RoundRect(ctx, area, 5);
        ctx.Clip();
        ctx.Font = Theme.Font;
        ctx.FillStyle = "#d8d8d8";
        ctx.TextBaseline = "middle";
        for (int i = 0; i < lines.Count; i++)
            ctx.FillText(lines[i], area.X + 8, area.Y + 6 + (i + 0.5) * NodeLayout.TextLineHeight);
        ctx.Restore();
    }

    /// <summary>The node's result picture. Until it has loaded a placeholder shows; once it has, the node is resized to the picture's shape.</summary>
    void DrawPreview(CanvasRenderingContext2D ctx, EditorNode node, Rect area, bool detailed)
    {
        var preview = node.Previews[0];
        var image = images.Get(preview.Url, (w, h) =>
        {
            node.Previews = node.Previews.Select(p => p.Url == preview.Url ? p with { Width = w, Height = h } : p).ToList();
            view.Invalidate();
        });

        RoundRect(ctx, area, 5);
        ctx.FillStyle = "#141414";
        ctx.Fill();
        if (image is not null)
        {
            ctx.Save();
            RoundRect(ctx, area, 5);
            ctx.Clip();
            ctx.DrawImage(image, area.X, area.Y, area.Width, area.Height);
            ctx.Restore();
        }
        else if (detailed)
        {
            ctx.Font = Theme.Font;
            ctx.FillStyle = Theme.LabelText;
            ctx.TextAlign = "center";
            ctx.TextBaseline = "middle";
            ctx.FillText("loading...", area.Center.X, area.Center.Y);
            ctx.TextAlign = "left";
        }
        if (node.Previews.Count > 1 && detailed)
        {
            ctx.Font = Theme.Font;
            ctx.FillStyle = "rgba(0,0,0,0.6)";
            ctx.FillRect(area.Right - 38, area.Y + 4, 34, 18);
            ctx.FillStyle = "#fff";
            ctx.TextBaseline = "middle";
            ctx.FillText($"1/{node.Previews.Count}", area.Right - 34, area.Y + 13);
        }
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
        // The value keeps its room (up to two thirds of the row); a long name gives way to it, as in ComfyUI.
        double avail = r.Width - 34, valueWidth = Math.Min(MeasureText(text), avail * 0.65);
        ctx.FillStyle = Theme.LabelText;
        ctx.TextAlign = "left";
        ctx.FillText(Ellipsize(w.Name, Math.Max(24, avail - valueWidth - 8)), r.X + 10, cy);
        ctx.FillStyle = Theme.ValueText;
        ctx.TextAlign = "right";
        ctx.FillText(Ellipsize(text, valueWidth), r.Right - 10, cy);
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
