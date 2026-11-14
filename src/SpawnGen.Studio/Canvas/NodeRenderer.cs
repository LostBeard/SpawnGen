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
        ctx.FillStyle = Theme.Current.MinimapFill;
        ctx.Fill();
        ctx.StrokeStyle = Theme.Current.MinimapBorder;
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
            ctx.FillStyle = selected.Contains(node.Id) ? Theme.Accent : Theme.Current.MinimapNode;
            ctx.FillRect(a.X, a.Y, Math.Max(2, geo.Bounds.Width * scale), Math.Max(2, geo.Bounds.Height * scale));
        }
        var vis = view.Viewport.VisibleWorld(cssWidth, cssHeight);
        var v = view.WorldToMinimap(new Point(vis.X, vis.Y));
        ctx.StrokeStyle = Theme.Current.MinimapView;
        ctx.LineWidth = 1.5;
        ctx.StrokeRect(v.X, v.Y, vis.Width * scale, vis.Height * scale);
        ctx.Restore();
    }

    // ---- groups ----

    /// <summary>A small push-pin: a round head over a needle, tilted.</summary>
    static void DrawPin(CanvasRenderingContext2D ctx, double x, double y)
    {
        ctx.StrokeStyle = Theme.Current.PinColor;
        ctx.FillStyle = Theme.Current.PinColor;
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
            if (slot is not null) Ring(ctx, slot.Position, 9, Theme.Current.LinkTargetRing);
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
        NodeOutline(ctx, node, b);
        bool isNote = schema is { FrontendOnly: true } && schema.Id != "Reroute";
        ctx.FillStyle = node.Mode == NodeMode.Bypass ? Theme.Current.BypassBody : node.BgColor ?? (isNote ? Theme.Current.NoteBody : Theme.NodeBody);
        ctx.Fill();
        if (!geo.Collapsed || true)
        {
            ctx.Save();
            NodeOutline(ctx, node, b);
            ctx.Clip();
            ctx.FillStyle = schema is null ? Theme.Current.MissingTitle : node.Color ?? (isNote ? Theme.Current.NoteTitle : Theme.TitleColor(schema.Category));
            ctx.FillRect(geo.TitleBar.X, geo.TitleBar.Y, geo.TitleBar.Width, geo.TitleBar.Height);
            ctx.Restore();
        }
        bool running = visuals.ExecutingNode == node.Id;
        bool failed = visuals.Errors.ContainsKey(node.Id);
        NodeOutline(ctx, node, b);
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
            // With the id badge on, a long title gives up its tail (the title font is bolder than the one measured, hence the 1.12).
            var title = view.ShowNodeIds ? Ellipsize(geo.Title, (b.Width - 26 - 40 - (node.Pinned ? 18 : 0)) / 1.12) : geo.Title;
            ctx.FillText(title, b.X + 26, geo.TitleBar.Y + geo.TitleBar.Height / 2);
            if (node.Pinned) DrawPin(ctx, b.Right - 14, geo.TitleBar.Y + geo.TitleBar.Height / 2);
            if (view.ShowNodeIds)
            {
                ctx.Font = Theme.Font;
                ctx.TextAlign = "right";
                ctx.FillStyle = Theme.LabelText;
                ctx.FillText("#" + node.Id, b.Right - (node.Pinned ? 28 : 10), geo.TitleBar.Y + geo.TitleBar.Height / 2);
                ctx.TextAlign = "left";
                ctx.Font = Theme.TitleFont;
            }
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
                ctx.StrokeStyle = Theme.Current.ResizeGrip;
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

        DrawAppPicks(ctx, node, geo, scale, detailed);

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

    const string AppPick = "#4aa8ff";

    /// <summary>The App Builder's outlines: pickable widgets / output nodes dashed, picked ones solid with their place in the app.</summary>
    void DrawAppPicks(CanvasRenderingContext2D ctx, EditorNode node, NodeGeometry geo, double scale, bool detailed)
    {
        if (visuals.ShowAppOutputs && visuals.AppOutputCandidates.Contains(node.Id))
        {
            int place = visuals.AppOutputs.IndexOf(node.Id);
            var r = geo.Bounds;
            RoundRect(ctx, new Rect(r.X - 4, r.Y - 4, r.Width + 8, r.Height + 8), 10);
            ctx.StrokeStyle = AppPick;
            ctx.LineWidth = place >= 0 ? 3 : 1.5;
            if (place < 0) ctx.SetLineDash([6, 4]);
            ctx.Stroke();
            ctx.SetLineDash([]);
            if (place >= 0) Badge(ctx, r.Right - 2, r.Y - 2, place + 1);
        }
        if (visuals.ShowAppInputs && detailed && !geo.Collapsed)
        {
            foreach (var w in geo.Widgets)
            {
                int place = visuals.AppInputs.IndexOf((node.Id, w.Name));
                var r = w.Bounds;
                RoundRect(ctx, new Rect(r.X - 2, r.Y - 1, r.Width + 4, r.Height + 2), 6);
                if (place >= 0)
                {
                    ctx.GlobalAlpha = 0.18;
                    ctx.FillStyle = AppPick;
                    ctx.Fill();
                    ctx.GlobalAlpha = 1;
                }
                ctx.StrokeStyle = AppPick;
                ctx.LineWidth = place >= 0 ? 2 : 1;
                if (place < 0) ctx.SetLineDash([4, 3]);
                ctx.Stroke();
                ctx.SetLineDash([]);
                if (place >= 0) Badge(ctx, r.Right + 2, r.Y + r.Height / 2, place + 1);
            }
        }
    }

    /// <summary>A small numbered disc.</summary>
    void Badge(CanvasRenderingContext2D ctx, double x, double y, int number)
    {
        ctx.BeginPath();
        ctx.Arc(x, y, 9, 0, Math.PI * 2);
        ctx.FillStyle = AppPick;
        ctx.Fill();
        ctx.Font = Theme.Font;
        ctx.FillStyle = "#ffffff";
        ctx.TextAlign = "center";
        ctx.TextBaseline = "middle";
        ctx.FillText(number.ToString(System.Globalization.CultureInfo.InvariantCulture), x, y + 0.5);
        ctx.TextAlign = "left";
    }

    /// <summary>The node's result text, one line per row in a dark box; what does not fit is cut off with an ellipsis.</summary>
    void DrawResultText(CanvasRenderingContext2D ctx, IReadOnlyList<string> lines, Rect area)
    {
        RoundRect(ctx, area, 5);
        ctx.FillStyle = Theme.Current.TextBoxFill;
        ctx.Fill();
        ctx.Save();
        RoundRect(ctx, area, 5);
        ctx.Clip();
        ctx.Font = Theme.Font;
        ctx.FillStyle = Theme.Current.TextBoxText;
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
        ctx.FillStyle = Theme.Current.TextBoxFill;
        ctx.Fill();
        if (image is not null)
        {
            ctx.Save();
            RoundRect(ctx, area, 5);
            ctx.Clip();
            ctx.DrawImage(image, area.X, area.Y, area.Width, area.Height);
            ctx.Restore();
            if (node.CompareWith is { } other && images.Get(other.Url, (_, _) => view.Invalidate()) is { } second)
            {
                // Compare Images: the second picture right of the split, which follows the pointer over the picture (the middle otherwise).
                var p = view.PointerWorld;
                double split = area.Contains(p) ? p.X : area.X + area.Width / 2;
                ctx.Save();
                RoundRect(ctx, area, 5);
                ctx.Clip();
                ctx.BeginPath();
                ctx.Rect(split, area.Y, area.Right - split, area.Height);
                ctx.Clip();
                ctx.DrawImage(second, area.X, area.Y, area.Width, area.Height);
                ctx.Restore();
                ctx.StrokeStyle = "#fff";
                ctx.LineWidth = 1.5;
                ctx.BeginPath();
                ctx.MoveTo(split, area.Y);
                ctx.LineTo(split, area.Bottom);
                ctx.Stroke();
            }
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
        if (s.IsList) Grid(ctx, s.Position, hovered ? 6.5 : 5, color);
        else Dot(ctx, s.Position, hovered ? 6.5 : 5, color, s.IsConnected);
        if (!detailed) return;
        ctx.Font = Theme.Font;
        ctx.TextBaseline = "middle";
        ctx.FillStyle = Theme.LabelText;
        if (s.Kind == SlotKind.Input)
        {
            ctx.TextAlign = "left";
            ctx.FillText(NodeLayout.Label(s.Name), s.Position.X + 12, s.Position.Y);
        }
        else
        {
            ctx.TextAlign = "right";
            ctx.FillText(s.Name, s.Position.X - 12, s.Position.Y);
        }
        ctx.TextAlign = "left";
    }

    /// <summary>A CURVE widget: the plot with a quarter grid, the curve as the node evaluates it (SpawnGen.Nodes.Core.Curve), its points and the interpolation.</summary>
    void DrawCurve(CanvasRenderingContext2D ctx, WidgetRow w, object? value, IReadOnlyList<double>? histogram)
    {
        var plot = CurveWidget.Plot(w.Bounds);
        var points = CurveWidget.Points(value);
        var interpolation = CurveWidget.Interpolation(value);
        ctx.Font = Theme.Font;
        ctx.TextBaseline = "middle";
        ctx.FillStyle = Theme.LabelText;
        ctx.FillText(w.Label, w.Bounds.X + 8, w.Bounds.Y + (CurveWidget.LabelHeight - 6) / 2 + 1);
        ctx.FillStyle = Theme.ValueText;
        ctx.TextAlign = "right";
        ctx.FillText(interpolation == "linear" ? "linear" : "monotone cubic", w.Bounds.Right - 8, w.Bounds.Y + (CurveWidget.LabelHeight - 6) / 2 + 1);
        ctx.TextAlign = "left";

        ctx.StrokeStyle = Theme.WidgetBorder;
        ctx.LineWidth = 1;
        ctx.StrokeRect(plot.X, plot.Y, plot.Width, plot.Height);
        ctx.BeginPath();
        for (int k = 1; k < 4; k++)
        {
            ctx.MoveTo(plot.X + plot.Width * k / 4, plot.Y); ctx.LineTo(plot.X + plot.Width * k / 4, plot.Bottom);
            ctx.MoveTo(plot.X, plot.Y + plot.Height * k / 4); ctx.LineTo(plot.Right, plot.Y + plot.Height * k / 4);
        }
        ctx.Stroke();

        // ComfyUI's faint identity diagonal, then the last run's histogram filled in the curve's colour at 15%.
        ctx.GlobalAlpha = 0.15;
        ctx.StrokeStyle = Theme.ValueText;
        ctx.BeginPath();
        ctx.MoveTo(plot.X, plot.Bottom); ctx.LineTo(plot.Right, plot.Y);
        ctx.Stroke();
        if (histogram is { Count: > 0 } && CurveWidget.HistogramOutline(histogram) is { Count: > 0 } outline)
        {
            ctx.FillStyle = Theme.ValueText;
            ctx.BeginPath();
            ctx.MoveTo(plot.X, plot.Bottom);
            foreach (var o in outline) { var p = CurveWidget.ToWorld(plot, o); ctx.LineTo(p.X, p.Y); }
            ctx.LineTo(plot.Right, plot.Bottom);
            ctx.ClosePath();
            ctx.Fill();
        }
        ctx.GlobalAlpha = 1;

        var curve = new SpawnGen.Nodes.Core.Curve(points, interpolation);
        ctx.StrokeStyle = Theme.ValueText;
        ctx.LineWidth = 2;
        ctx.BeginPath();
        const int steps = 96;
        for (int i = 0; i <= steps; i++)
        {
            double x = i / (double)steps;
            var p = CurveWidget.ToWorld(plot, (x, Math.Clamp(curve.Interp(x), -0.1, 1.1)));
            if (i == 0) ctx.MoveTo(p.X, p.Y); else ctx.LineTo(p.X, p.Y);
        }
        ctx.Stroke();
        ctx.FillStyle = "#e8e8e8";
        foreach (var pt in points)
        {
            var p = CurveWidget.ToWorld(plot, pt);
            ctx.BeginPath();
            ctx.Arc(p.X, p.Y, 4, 0, Math.PI * 2);
            ctx.Fill();
        }
    }

    /// <summary>
    /// A BOUNDING_BOXES widget (Create Bounding Boxes): the toolbar (Grid, Clear all), the canvas at the boxes' aspect with the last run's background,
    /// a dot every grid step while Grid is on, each box with its number tag and palette strip (the selected one highlighted with corner handles, the
    /// one being drawn dashed), and the selected box's panel: obj / text, the text to render, the description, the palette.
    /// </summary>
    void DrawBoxes(CanvasRenderingContext2D ctx, EditorNode node, WidgetRow w, object? value)
    {
        var (cw, ch) = BoxesWidget.CanvasSize(node.Widgets);
        var lay = BoxesWidget.Lay(w.Bounds, cw, ch);
        var boxes = BoxesWidget.Boxes(value);
        bool grid = view.BoxGrid(node.Id, w.Name);
        int sel = view.SelectedBox is { } s && s.NodeId == node.Id && s.Name == w.Name && s.Index < boxes.Count ? s.Index : -1;
        ctx.Font = Theme.Font;
        ctx.TextBaseline = "middle";

        void Button(Rect r, string text, bool on)
        {
            RoundRect(ctx, r, 4);
            ctx.FillStyle = on ? Theme.Accent : Theme.Current.TextBoxFill;
            ctx.Fill();
            ctx.FillStyle = on ? "#ffffff" : Theme.ValueText;
            ctx.TextAlign = "center";
            ctx.FillText(text, r.Center.X, r.Center.Y + 1);
            ctx.TextAlign = "left";
        }
        Button(lay.GridButton, "Grid", grid);
        Button(lay.ClearButton, "Clear all", false);

        // The canvas: the background picture of the last run (or a dark field), the grid's dots, the boxes.
        var c = lay.Canvas;
        ctx.Save();
        ctx.BeginPath();
        ctx.Rect(c.X, c.Y, c.Width, c.Height);
        ctx.Clip();
        ctx.FillStyle = "#202020";
        ctx.FillRect(c.X, c.Y, c.Width, c.Height);
        if (node.BoxesBackground is { } bg && images.Get(bg.Url, (_, _) => view.Invalidate()) is { } picture)
        {
            ctx.GlobalAlpha = 0.85;
            ctx.DrawImage(picture, c.X, c.Y, c.Width, c.Height);
            ctx.GlobalAlpha = 1;
        }
        double step = BoxesWidget.GridStep / cw * c.Width;
        if (grid && step * view.Viewport.Scale >= 4)
        {
            ctx.FillStyle = "rgba(255,255,255,0.18)";
            for (double y = c.Y + step; y < c.Bottom; y += step)
                for (double x = c.X + step; x < c.Right; x += step)
                    ctx.FillRect(x - 0.5, y - 0.5, 1, 1);
        }
        for (int i = 0; i < boxes.Count; i++)
        {
            var b = boxes[i];
            var r = BoxesWidget.BoxRect(c, cw, ch, b);
            var palette = b.Palette;
            string colour = palette.Count > 0 ? palette[0] : "#c8c8c8";
            ctx.FillStyle = "rgba(0,0,0,0.25)";
            ctx.FillRect(r.X, r.Y, r.Width, r.Height);
            for (int k = 0; k < Math.Min(5, palette.Count); k++)
            {
                ctx.FillStyle = palette[k];
                ctx.FillRect(r.X + r.Width * k / Math.Min(5, palette.Count), r.Y, r.Width / Math.Min(5, palette.Count), 4);
            }
            ctx.StrokeStyle = i == sel ? Theme.Accent : colour;
            ctx.LineWidth = i == sel ? 2 : 1.25;
            ctx.StrokeRect(r.X, r.Y, r.Width, r.Height);
            var tag = BoxesWidget.Tag(i);
            ctx.Font = "600 9px ui-monospace, monospace";
            double tw = MeasureText(tag) * 0.8 + 6;
            ctx.FillStyle = i == sel ? Theme.Accent : "#5a5a5a";
            ctx.FillRect(r.X + 1, r.Y + 1, tw, 11);
            ctx.FillStyle = "#ffffff";
            ctx.FillText(tag, r.X + 4, r.Y + 7);
            ctx.Font = Theme.Font;
            if (b.Desc.Length > 0 || (b.Type == "text" && b.Text.Length > 0))
            {
                ctx.FillStyle = "rgba(255,255,255,0.75)";
                var label = b.Type == "text" && b.Text.Length > 0 ? $"\"{b.Text}\" {b.Desc}" : b.Desc;
                ctx.FillText(Ellipsize(label, Math.Max(10, r.Width - tw - 8)), r.X + tw + 5, r.Y + 7);
            }
            if (i == sel)
            {
                ctx.FillStyle = Theme.Accent;
                foreach (var (hx, hy) in new[] { (r.X, r.Y), (r.Right, r.Y), (r.X, r.Bottom), (r.Right, r.Bottom) })
                    ctx.FillRect(hx - 3, hy - 3, 6, 6);
            }
        }
        if (view.DraftBox is { } d && d.NodeId == node.Id && d.Name == w.Name)
        {
            var r = BoxesWidget.BoxRect(c, cw, ch, new BoxesWidget.Box(d.X, d.Y, d.W, d.H, BoxesWidget.NewMetadata()));
            ctx.StrokeStyle = Theme.Accent;
            ctx.LineWidth = 1.5;
            ctx.SetLineDash([5, 3]);
            ctx.StrokeRect(r.X, r.Y, r.Width, r.Height);
            ctx.SetLineDash([]);
        }
        ctx.Restore();
        ctx.StrokeStyle = Theme.WidgetBorder;
        ctx.LineWidth = 1;
        ctx.StrokeRect(c.X, c.Y, c.Width, c.Height);

        // The selected box's panel.
        if (sel < 0)
        {
            ctx.FillStyle = Theme.LabelText;
            ctx.FillText(boxes.Count == 0 ? "Drag on the canvas to draw a box." : "Click a box to edit it.", lay.Panel.X + 8, lay.Panel.Y + 14);
            return;
        }
        var box = boxes[sel];
        Button(lay.ObjButton, "obj", box.Type != "text");
        Button(lay.TextButton, "text", box.Type == "text");
        void Field(Rect r, string text, string placeholder)
        {
            RoundRect(ctx, r, 4);
            ctx.FillStyle = Theme.Current.TextBoxFill;
            ctx.Fill();
            ctx.Save();
            ctx.BeginPath();
            ctx.Rect(r.X, r.Y, r.Width, r.Height);
            ctx.Clip();
            ctx.FillStyle = text.Length > 0 ? Theme.ValueText : Theme.LabelText;
            int line = 0;
            foreach (var l in WrapLines(text.Length > 0 ? text : placeholder, r.Width - 12).Take(Math.Max(1, (int)((r.Height - 6) / 15))))
                ctx.FillText(l, r.X + 6, r.Y + 10 + line++ * 15);
            ctx.Restore();
        }
        if (box.Type == "text") Field(lay.TextField, box.Text, "text to render (verbatim)");
        Field(lay.DescField, box.Desc, "description of this region");
        var palette2 = box.Palette;
        ctx.FillStyle = Theme.LabelText;
        ctx.FillText("color_palette", lay.PaletteRow.X + 2, lay.PaletteRow.Center.Y + 1);
        for (int k = 0; k < palette2.Count; k++)
        {
            var r = BoxesWidget.Swatch(lay.PaletteRow, k);
            ctx.FillStyle = palette2[k];
            ctx.FillRect(r.X, r.Y, r.Width, r.Height);
            ctx.StrokeStyle = Theme.WidgetBorder;
            ctx.StrokeRect(r.X, r.Y, r.Width, r.Height);
        }
        Button(BoxesWidget.Plus(lay.PaletteRow, palette2.Count), "+", false);
    }

    /// <summary>The picture a Crop Image node shows (its input's: the upstream node's last result, or Load Image's file), set by the Studio.</summary>
    public Func<EditorNode, string?>? CropPictureUrl { get; set; }

    /// <summary>The natural size of each picture drawn so far (the editor maps drags on a crop picture to its pixels with it).</summary>
    public Dictionary<string, (double W, double H)> PictureSizes { get; } = [];

    /// <summary>
    /// Crop Image's crop region: the input picture with everything outside the region dimmed, the region's rectangle with its corner handles, the
    /// Ratio choice and its lock, and the X / Y / Width / Height fields.
    /// </summary>
    void DrawCrop(CanvasRenderingContext2D ctx, EditorNode node, WidgetRow w, object? value)
    {
        var lay = CropWidget.Lay(w.Bounds);
        var region = CropWidget.Read(value);
        ctx.Font = Theme.Font;
        ctx.TextBaseline = "middle";
        var box = lay.PictureBox;
        RoundRect(ctx, box, 5);
        ctx.FillStyle = "#1c1c1c";
        ctx.Fill();
        var url = CropPictureUrl?.Invoke(node);
        var picture = url is null ? null : images.Get(url, (pw, ph) => { PictureSizes[url] = (pw, ph); view.Invalidate(); });
        if (picture is not null && url is not null && PictureSizes.TryGetValue(url, out var size))
        {
            var fit = CropWidget.Fit(box, size.W, size.H);
            ctx.DrawImage(picture, fit.X, fit.Y, fit.Width, fit.Height);
            var r = CropWidget.RegionRect(fit, size.W, size.H, region);
            // Dim what the crop leaves out, then the rectangle and its handles.
            ctx.Save();
            ctx.BeginPath();
            ctx.Rect(fit.X, fit.Y, fit.Width, fit.Height);
            ctx.Clip();
            ctx.FillStyle = "rgba(0,0,0,0.5)";
            ctx.FillRect(fit.X, fit.Y, fit.Width, Math.Max(0, r.Y - fit.Y));
            ctx.FillRect(fit.X, r.Bottom, fit.Width, Math.Max(0, fit.Bottom - r.Bottom));
            ctx.FillRect(fit.X, r.Y, Math.Max(0, r.X - fit.X), r.Height);
            ctx.FillRect(r.Right, r.Y, Math.Max(0, fit.Right - r.Right), r.Height);
            ctx.StrokeStyle = "#ffffff";
            ctx.LineWidth = 1.5;
            ctx.StrokeRect(r.X, r.Y, r.Width, r.Height);
            ctx.Restore();
            ctx.FillStyle = "#ffffff";
            foreach (var (hx, hy) in new[] { (r.X, r.Y), (r.Right, r.Y), (r.X, r.Bottom), (r.Right, r.Bottom) })
            {
                ctx.BeginPath();
                ctx.Arc(hx, hy, 3.5, 0, Math.PI * 2);
                ctx.Fill();
            }
        }
        else
        {
            ctx.FillStyle = Theme.LabelText;
            ctx.TextAlign = "center";
            ctx.FillText(url is null ? "No input image connected" : "Loading the picture...", box.Center.X, box.Center.Y);
            ctx.TextAlign = "left";
        }

        ctx.FillStyle = Theme.LabelText;
        ctx.FillText("Ratio", lay.RatioLabel.X, lay.RatioLabel.Center.Y + 1);
        RoundRect(ctx, lay.RatioButton, 4);
        ctx.FillStyle = Theme.Current.TextBoxFill;
        ctx.Fill();
        ctx.FillStyle = Theme.ValueText;
        ctx.FillText(view.CropRatio(node.Id, w.Name) + "  ▾", lay.RatioButton.X + 8, lay.RatioButton.Center.Y + 1);
        // The lock: a padlock, filled when on.
        bool locked = view.CropLocked(node.Id, w.Name);
        var l = lay.LockButton;
        RoundRect(ctx, l, 4);
        ctx.FillStyle = locked ? Theme.Accent : Theme.Current.TextBoxFill;
        ctx.Fill();
        ctx.StrokeStyle = locked ? "#ffffff" : Theme.ValueText;
        ctx.LineWidth = 1.5;
        ctx.BeginPath();
        ctx.Arc(l.Center.X, l.Center.Y - 1, 3.5, Math.PI, locked ? 0 : -0.35 * Math.PI);
        ctx.Stroke();
        ctx.FillStyle = locked ? "#ffffff" : Theme.ValueText;
        ctx.FillRect(l.Center.X - 5, l.Center.Y - 1, 10, 7);

        foreach (var (field, label, valueRect) in lay.FieldRows)
        {
            ctx.FillStyle = Theme.LabelText;
            ctx.FillText(field switch { "x" => "X", "y" => "Y", "width" => "Width", _ => "Height" }, label.X, label.Center.Y + 1);
            RoundRect(ctx, valueRect, 4);
            ctx.FillStyle = Theme.Current.TextBoxFill;
            ctx.Fill();
            ctx.FillStyle = Theme.ValueText;
            var v = field switch { "x" => region.X, "y" => region.Y, "width" => region.W, _ => region.H };
            ctx.FillText(v.ToString("0.##", System.Globalization.CultureInfo.InvariantCulture), valueRect.X + 8, valueRect.Center.Y + 1);
        }
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

        if (w.IsCurve) { DrawCurve(ctx, w, value, node.ResultHistogram); return; }
        if (w.IsBoxes) { DrawBoxes(ctx, node, w, value); return; }
        if (w.IsCrop) { DrawCrop(ctx, node, w, value); return; }

        ctx.Font = Theme.Font;
        ctx.TextBaseline = "middle";
        double cy = r.Y + (w.Spec.Multiline ? 12 : r.Height / 2);

        if (w.Spec.Multiline)
        {
            ctx.FillStyle = Theme.LabelText;
            ctx.FillText(w.Label, r.X + 8, cy);
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

        if (w.Spec.Kind == WidgetKind.Color)
        {
            // A colour: its swatch at the right end of the row, the hex code beside it.
            var hex = value as string ?? "#ffffff";
            ctx.FillStyle = Theme.LabelText;
            ctx.TextAlign = "left";
            ctx.FillText(Ellipsize(w.Label, Math.Max(24, r.Width - 110)), r.X + 10, cy);
            ctx.FillStyle = Theme.ValueText;
            ctx.TextAlign = "right";
            ctx.FillText(hex, r.Right - 34, cy);
            ctx.TextAlign = "left";
            ctx.FillStyle = hex;
            ctx.FillRect(r.Right - 28, r.Y + 4, 20, r.Height - 8);
            ctx.StrokeStyle = Theme.WidgetBorder;
            ctx.LineWidth = 1;
            ctx.StrokeRect(r.Right - 28, r.Y + 4, 20, r.Height - 8);
            return;
        }

        string text = w.Spec.Kind switch
        {
            WidgetKind.Float => Convert.ToDouble(value ?? 0.0).ToString("0.###", System.Globalization.CultureInfo.InvariantCulture),
            WidgetKind.Boolean => (value as bool? ?? false) ? "on" : "off",
            WidgetKind.Json => SpawnGen.Engine.Types.JsonLiteral.ToJson(value),
            _ => value?.ToString() ?? "",
        };
        // The value keeps its room (up to two thirds of the row); a long name gives way to it, as in ComfyUI.
        double avail = r.Width - 34, valueWidth = Math.Min(MeasureText(text), avail * 0.65);
        ctx.FillStyle = Theme.LabelText;
        ctx.TextAlign = "left";
        ctx.FillText(Ellipsize(w.Label, Math.Max(24, avail - valueWidth - 8)), r.X + 10, cy);
        ctx.FillStyle = Theme.ValueText;
        ctx.TextAlign = "right";
        ctx.FillText(Ellipsize(text, valueWidth), r.Right - 10, cy);
        ctx.TextAlign = "left";

        if (w.Spec.Kind is WidgetKind.Int or WidgetKind.Float or WidgetKind.Combo)
        {
            ctx.FillStyle = Theme.Current.WidgetArrow;
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

    /// <summary>A list output: LiteGraph's grid-shaped slot, three by three small squares.</summary>
    static void Grid(CanvasRenderingContext2D ctx, Point p, double radius, string color)
    {
        double cell = radius * 2 / 3;
        ctx.FillStyle = color;
        for (int row = 0; row < 3; row++)
            for (int col = 0; col < 3; col++)
                ctx.FillRect(p.X - radius + col * cell + 0.5, p.Y - radius + row * cell + 0.5, cell - 1, cell - 1);
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
    /// <summary>
    /// A node's outline by its shape, as ComfyUI draws them (measured from its canvas): round (the default) rounds every corner, box none, card only
    /// the top two.
    /// </summary>
    static void NodeOutline(CanvasRenderingContext2D ctx, EditorNode node, Rect b)
    {
        const double r = 8;
        switch (node.Shape)
        {
            case NodeShape.Box: RoundRect(ctx, b, 0, 0, 0, 0); break;
            case NodeShape.Card: RoundRect(ctx, b, r, r, 0, 0); break;
            default: RoundRect(ctx, b, r); break;
        }
    }

    /// <summary>A rectangle path with its own radius per corner: top-left, top-right, bottom-right, bottom-left.</summary>
    static void RoundRect(CanvasRenderingContext2D ctx, Rect r, double topLeft, double topRight, double bottomRight, double bottomLeft)
    {
        double max = Math.Min(r.Width, r.Height) / 2;
        double tl = Math.Min(topLeft, max), tr = Math.Min(topRight, max), br = Math.Min(bottomRight, max), bl = Math.Min(bottomLeft, max);
        ctx.BeginPath();
        ctx.MoveTo(r.X + tl, r.Y);
        if (tr > 0) ctx.ArcTo(r.Right, r.Y, r.Right, r.Bottom, tr); else ctx.LineTo(r.Right, r.Y);
        if (br > 0) ctx.ArcTo(r.Right, r.Bottom, r.X, r.Bottom, br); else ctx.LineTo(r.Right, r.Bottom);
        if (bl > 0) ctx.ArcTo(r.X, r.Bottom, r.X, r.Y, bl); else ctx.LineTo(r.X, r.Bottom);
        if (tl > 0) ctx.ArcTo(r.X, r.Y, r.Right, r.Y, tl); else ctx.LineTo(r.X, r.Y);
        ctx.ClosePath();
    }

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
