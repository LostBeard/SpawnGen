using System.Globalization;
using SpawnDev.SpawnJS.JSObjects;
using SpawnGen.Editor;

namespace SpawnGen.Studio.Canvas;

/// <summary>
/// Painter nodes' paintings, kept in the page: per node an OffscreenCanvas with the painting (RGBA, straight alpha as uploaded, transparent where
/// nothing is painted) and one for the stroke being drawn. Every pixel stays in JS (AGENTS.md rule 4): stamps are canvas operations, the canvas
/// is drawn straight onto the node, and the upload is the canvas's own PNG Blob.
/// </summary>
/// <remarks>
/// The rules are ComfyUI's, measured (PainterWidget's remarks): a stamp is the radial gradient opaque to r x hardness and transparent at
/// r x (1 + (1 - hardness) / 2) (a hard one a filled circle); a stroke's stamps build up on the stroke canvas at full strength and the stroke
/// lands on the painting at the brush's opacity, source-over, when it ends; the Eraser removes alpha with hard full stamps; a new canvas
/// size keeps the painting's pixels from the top-left.
/// </remarks>
public sealed class PainterSurface : IPainterCanvas, IDisposable
{
    sealed class Item : IDisposable
    {
        public required OffscreenCanvas Painting;
        public required CanvasRenderingContext2D Ctx;
        public required OffscreenCanvas StrokeCanvas;
        public required CanvasRenderingContext2D StrokeCtx;
        public int W, H;
        public PainterWidget.Brush? Stroke;
        /// <summary>Painted since the last upload (or never uploaded but loaded or painted).</summary>
        public bool Changed;
        public bool Empty = true;

        public void Dispose()
        {
            Ctx.Dispose(); Painting.Dispose();
            StrokeCtx.Dispose(); StrokeCanvas.Dispose();
        }
    }

    // Paintings by scope (the workflow tab and the subgraph level on the canvas: node ids repeat between them), then by node id. A tab's
    // paintings stay while it is open, so switching tabs and back keeps strokes that were never uploaded.
    readonly Dictionary<string, Dictionary<int, Item>> _scopes = [];

    /// <summary>The scope the canvas shows now (the Studio: its tab id and subgraph path).</summary>
    public Func<string> Scope { get; set; } = () => "";

    Dictionary<int, Item> _items
    {
        get
        {
            var scope = Scope();
            if (!_scopes.TryGetValue(scope, out var items)) _scopes[scope] = items = [];
            return items;
        }
    }

    /// <summary>A workflow tab was closed: its paintings go (every subgraph level of it).</summary>
    public void RemoveTab(string tabId)
    {
        foreach (var scope in _scopes.Keys.Where(k => k == tabId || k.StartsWith(tabId + "/", StringComparison.Ordinal)).ToList())
        {
            foreach (var item in _scopes[scope].Values) item.Dispose();
            _scopes.Remove(scope);
        }
    }

    /// <summary>Raised when a painting changed (the canvas redraws).</summary>
    public event Action? Changed;

    static Item Create(int w, int h)
    {
        var painting = new OffscreenCanvas(w, h);
        var stroke = new OffscreenCanvas(w, h);
        return new Item { Painting = painting, Ctx = painting.Get2DContext(null!), StrokeCanvas = stroke, StrokeCtx = stroke.Get2DContext(null!), W = w, H = h };
    }

    /// <summary>The node's painting at <paramref name="size"/>: created empty, or resized keeping its pixels from the top-left (not scaled).</summary>
    Item Ensure(int nodeId, (int W, int H) size)
    {
        if (_items.TryGetValue(nodeId, out var item))
        {
            if (item.W == size.W && item.H == size.H) return item;
            var resized = Create(size.W, size.H);
            resized.Ctx.DrawImage(item.Painting, 0, 0);
            resized.Changed = true;
            resized.Empty = item.Empty;
            item.Dispose();
            _items[nodeId] = resized;
            return resized;
        }
        _items[nodeId] = item = Create(size.W, size.H);
        return item;
    }

    /// <summary>Brings a node's painting to the size its widget shows now (the Width / Height sliders, a linked picture); call before drawing it.</summary>
    public void Fit(int nodeId, (int W, int H) size)
    {
        if (_items.ContainsKey(nodeId)) Ensure(nodeId, size);
    }

    /// <summary>The colour with an alpha, as the canvas takes it ("#rrggbb" -> "rgba(r,g,b,a)").</summary>
    public static string Rgba(string hex, double alpha)
    {
        var c = PainterWidget.NormalizeColor(hex) ?? PainterWidget.DefaultColor;
        int r = Convert.ToInt32(c[1..3], 16), g = Convert.ToInt32(c[3..5], 16), b = Convert.ToInt32(c[5..7], 16);
        return string.Create(CultureInfo.InvariantCulture, $"rgba({r},{g},{b},{alpha})");
    }

    static void Stamp(CanvasRenderingContext2D ctx, PainterWidget.Brush brush, double x, double y)
    {
        double rout = PainterWidget.OuterRadius(brush.Size, brush.Hardness);
        ctx.BeginPath();
        ctx.Arc(x, y, rout, 0, Math.PI * 2, false);
        if (PainterWidget.GradientStops(brush.Size, brush.Hardness) is { } stops)
        {
            using var gradient = ctx.CreateRadialGradient(x, y, 0, x, y, rout);
            foreach (var (offset, alpha) in stops) gradient.AddColorStop(offset, Rgba(brush.Color, alpha));
            ctx.SetFillStyle(gradient);
        }
        else ctx.FillStyle = Rgba(brush.Color, 1);
        ctx.Fill();
    }

    public void BeginStroke(int nodeId, (int W, int H) size, PainterWidget.Brush brush, double x, double y)
    {
        var item = Ensure(nodeId, size);
        item.Stroke = brush;
        item.StrokeCtx.ClearRect(0, 0, item.W, item.H);
        Stamps(nodeId, [(x, y)]);
    }

    public void Stamps(int nodeId, IReadOnlyList<(double X, double Y)> stamps)
    {
        if (!_items.TryGetValue(nodeId, out var item) || item.Stroke is not { } brush) return;
        if (brush.Tool == PainterWidget.Tool.Eraser)
        {
            // hard and full: erasing straight on the painting is the same as at the stroke's end, and shows at once
            item.Ctx.GlobalCompositeOperation = "destination-out";
            foreach (var (x, y) in stamps) Stamp(item.Ctx, brush with { Color = "#000000" }, x, y);
            item.Ctx.GlobalCompositeOperation = "source-over";
            item.Changed = true;
        }
        else foreach (var (x, y) in stamps) Stamp(item.StrokeCtx, brush, x, y);
        Changed?.Invoke();
    }

    public void EndStroke(int nodeId)
    {
        if (!_items.TryGetValue(nodeId, out var item) || item.Stroke is not { } brush) return;
        if (brush.Tool == PainterWidget.Tool.Brush)
        {
            item.Ctx.GlobalAlpha = brush.Opacity / 100.0;
            item.Ctx.DrawImage(item.StrokeCanvas, 0, 0);
            item.Ctx.GlobalAlpha = 1;
            item.StrokeCtx.ClearRect(0, 0, item.W, item.H);
            item.Empty = false;
        }
        item.Stroke = null;
        item.Changed = true;
        Changed?.Invoke();
    }

    public void Clear(int nodeId)
    {
        if (!_items.TryGetValue(nodeId, out var item)) return;
        item.Ctx.ClearRect(0, 0, item.W, item.H);
        item.Changed = true;
        item.Empty = true;
        Changed?.Invoke();
    }

    /// <summary>Draws the node's painting (and the stroke in progress, at its opacity) into <paramref name="shown"/> on the editor's canvas.</summary>
    public void Draw(CanvasRenderingContext2D ctx, int nodeId, Rect shown)
    {
        if (!_items.TryGetValue(nodeId, out var item)) return;
        ctx.DrawImage(item.Painting, shown.X, shown.Y, shown.Width, shown.Height);
        if (item.Stroke is { Tool: PainterWidget.Tool.Brush } brush)
        {
            ctx.GlobalAlpha = brush.Opacity / 100.0;
            ctx.DrawImage(item.StrokeCanvas, shown.X, shown.Y, shown.Width, shown.Height);
            ctx.GlobalAlpha = 1;
        }
    }

    /// <summary>Whether the node's painting changed since it was last uploaded (a stroke, Clear, a resize, a loaded file).</summary>
    public bool NeedsUpload(int nodeId) => _items.TryGetValue(nodeId, out var item) && item.Changed;

    /// <summary>The painting as a PNG Blob of the canvas size (straight alpha, as ComfyUI's widget uploads it): it stays a JS object.</summary>
    public async Task<Blob?> ToPngAsync(int nodeId) =>
        _items.TryGetValue(nodeId, out var item) ? await item.Painting.ConvertToBlob() : null;

    /// <summary>The upload took the current painting.</summary>
    public void Uploaded(int nodeId)
    {
        if (_items.TryGetValue(nodeId, out var item)) item.Changed = false;
    }

    /// <summary>A saved workflow's painting (the file its <c>mask</c> names) drawn into the node's canvas, the canvas taking the picture's size.</summary>
    public void Load(int nodeId, HTMLImageElement picture)
    {
        var item = Ensure(nodeId, (picture.NaturalWidth, picture.NaturalHeight));
        item.Ctx.ClearRect(0, 0, item.W, item.H);
        item.Ctx.DrawImage(picture, 0, 0);
        item.Changed = false; // it is the file the node already names
        item.Empty = false;
        Changed?.Invoke();
    }

    public bool Has(int nodeId) => _items.ContainsKey(nodeId);

    /// <summary>Drops a node's painting (the node was deleted, the workflow closed).</summary>
    public void Remove(int nodeId)
    {
        if (_items.Remove(nodeId, out var item)) item.Dispose();
    }

    public void Dispose()
    {
        foreach (var items in _scopes.Values) foreach (var item in items.Values) item.Dispose();
        _scopes.Clear();
    }
}
