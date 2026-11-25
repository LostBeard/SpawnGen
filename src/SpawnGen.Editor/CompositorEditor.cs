using System.Globalization;
using SpawnGen.Engine.Types;

namespace SpawnGen.Editor;

/// <summary>
/// The compositor editor of Create Layered Image (the COMPOSITOR widget): the layers of the last run on a canvas, each with a place, a size,
/// a rotation, flips, an opacity, a blend mode and a visibility, a drawing order, a background and the canvas size. This class is the editor's
/// document and its actions, with undo; the Studio draws it. Its value is the JSON the node replays (ImageCompositor, SpawnGen.Nodes.Core).
/// </summary>
/// <remarks>
/// Measured on ComfyUI's own editor (black box, 2026-11-25: tools/studio-browser-test/probe-comfy-compositor*.js; docs/PLANS.md has the steps):
/// the value is <c>{}</c> until the first edit and is rewritten after every action; layers are stored in input order with an "order" key
/// (bottom first) only while the drawing order differs; positions and sizes are stored unrounded and shown rounded; rotation is stored in
/// radians and shown in degrees, opacity stored 0..1 and shown in percent; a body drag moves by the drag over the zoom; a corner or edge
/// handle resizes freely with the opposite side staying put, and with Shift a corner keeps the aspect (the drag projected on the diagonal);
/// the rotation handle sets the angle of the pointer about the layer's centre; alignment is to the canvas; typed values are taken as they
/// are; undo and redo go one action at a time; Restore returns to the composition as it was when the editor opened; arrow keys and Delete
/// do nothing. Not measured (SpawnGen's own choices, marked where they are): resizing a rotated layer, a size dragged through zero,
/// alignment of a rotated layer, the background colour picker and canvas size editing. ComfyUI's look is its own.
/// </remarks>
public sealed class CompositorEditor
{
    public const string TypeName = "COMPOSITOR";

    /// <summary>The blend modes in the editor's list order: the stored id and the name shown.</summary>
    public static readonly (string Id, string Name)[] BlendModes =
    [
        ("normal", "Normal"), ("multiply", "Multiply"), ("screen", "Screen"), ("overlay", "Overlay"), ("darken", "Darken"), ("lighten", "Lighten"),
        ("color-dodge", "Color Dodge"), ("color-burn", "Color Burn"), ("hard-light", "Hard Light"), ("soft-light", "Soft Light"),
        ("difference", "Difference"), ("exclusion", "Exclusion"), ("linear-dodge", "Linear Dodge"), ("linear-burn", "Linear Burn"),
        ("vivid-light", "Vivid Light"), ("pin-light", "Pin Light"), ("linear-light", "Linear Light"), ("hard-mix", "Hard Mix"),
        ("subtract", "Subtract"), ("divide", "Divide"), ("grain-extract", "Grain Extract"), ("grain-merge", "Grain Merge"), ("hue", "Hue"),
        ("saturation", "Saturation"), ("color", "Color"), ("luminosity", "Luminosity"),
    ];

    /// <summary>One layer, as the value stores it (rotation in radians, opacity 0..1; X / Y are the unrotated box's top-left).</summary>
    public sealed class Layer
    {
        public string Name = "";
        public bool Visible = true;
        public double Opacity = 1;
        public string Blend = "normal";
        public double X, Y, W, H, Rotation;
        public bool FlipH, FlipV;

        public Layer Clone() => (Layer)MemberwiseClone();
        public double CenterX => X + W / 2;
        public double CenterY => Y + H / 2;
    }

    /// <summary>The parts of a selected layer's frame a pointer can take hold of.</summary>
    public enum Handle { None, Body, TopLeft, Top, TopRight, Right, BottomRight, Bottom, BottomLeft, Left, Rotate }

    public enum Alignment { Left, HorizontalCenter, Right, Top, VerticalCenter, Bottom }

    sealed record Snapshot(Layer[] Layers, int[] Order, double CanvasW, double CanvasH, string BackgroundColor, double BackgroundOpacity, bool BackgroundVisible, int? Selected);

    readonly List<Layer> _layers = [];
    readonly List<int> _order = [];           // input indices, bottom first
    readonly List<Snapshot> _undo = [], _redo = [];
    readonly Snapshot _opened;
    Snapshot? _gesture;

    /// <summary>The run's layer fingerprints (the node's ui compositor_inputs): a saved composition belongs to these.</summary>
    public IReadOnlyList<string> Inputs { get; }
    public IReadOnlyList<Layer> Layers => _layers;
    /// <summary>Input indices in drawing order, bottom first.</summary>
    public IReadOnlyList<int> Order => _order;
    public double CanvasW { get; private set; }
    public double CanvasH { get; private set; }
    public string BackgroundColor { get; private set; } = "#ffffff";
    public double BackgroundOpacity { get; private set; } = 1;
    public bool BackgroundVisible { get; private set; }
    /// <summary>The selected layer's input index, or null when the Background is selected.</summary>
    public int? Selected { get; private set; }
    /// <summary>Whether anything has been edited since the value was <c>{}</c> (an unedited composition stores nothing).</summary>
    public bool Edited { get; private set; }
    public bool CanUndo => _undo.Count > 0;
    public bool CanRedo => _redo.Count > 0;

    /// <summary>
    /// Opens the editor on a run's layers. <paramref name="inputs"/>, <paramref name="boxes"/> and <paramref name="canvas"/> are the node's ui
    /// outputs (compositor_inputs, compositor_bboxes, compositor_canvas); <paramref name="saved"/> is the widget's value: taken when it is a
    /// version-1 composition of exactly these inputs, otherwise the layers start where the run placed them. The top layer is selected.
    /// </summary>
    public CompositorEditor(IReadOnlyList<string> inputs, IReadOnlyList<DictValue> boxes, (double W, double H) canvas, object? saved = null)
    {
        Inputs = [.. inputs];
        foreach (var b in boxes)
            _layers.Add(new Layer
            {
                Name = b.GetValueOrDefault("name") as string ?? "", Visible = b.GetValueOrDefault("visible") is not false,
                Opacity = Num(b.GetValueOrDefault("opacity"), 1), Blend = b.GetValueOrDefault("blend") as string ?? "normal",
                X = Num(b.GetValueOrDefault("x")), Y = Num(b.GetValueOrDefault("y")), W = Num(b.GetValueOrDefault("width")), H = Num(b.GetValueOrDefault("height")),
                Rotation = Num(b.GetValueOrDefault("rotation")), FlipH = b.GetValueOrDefault("flipH") is true, FlipV = b.GetValueOrDefault("flipV") is true,
            });
        for (int i = 0; i < _layers.Count; i++) _order.Add(i);
        (CanvasW, CanvasH) = canvas;
        if (saved is DictValue d && Fits(d)) { Load(d); Edited = true; }
        Selected = _order.Count > 0 ? _order[^1] : null;
        _opened = Take();
    }

    bool Fits(DictValue d) =>
        d.GetValueOrDefault("version") is long or int or double && Num(d.GetValueOrDefault("version")) == 1
        && d.GetValueOrDefault("inputs") is ArrayValue ins && ins.Count == Inputs.Count && ins.Select(i => i as string).SequenceEqual(Inputs)
        && d.GetValueOrDefault("layers") is ArrayValue ls && ls.Count == _layers.Count
        && d.GetValueOrDefault("canvas") is DictValue c && Num(c.GetValueOrDefault("w")) > 0 && Num(c.GetValueOrDefault("h")) > 0;

    void Load(DictValue d)
    {
        var c = (DictValue)d["canvas"]!;
        (CanvasW, CanvasH) = (Num(c.GetValueOrDefault("w")), Num(c.GetValueOrDefault("h")));
        if (d.GetValueOrDefault("background") is DictValue bg)
        {
            BackgroundColor = bg.GetValueOrDefault("color") as string ?? "#ffffff";
            BackgroundOpacity = Num(bg.GetValueOrDefault("opacity"), 1);
            BackgroundVisible = bg.GetValueOrDefault("visible") is true;
        }
        var saved = (ArrayValue)d["layers"]!;
        for (int i = 0; i < _layers.Count; i++)
        {
            if (saved[i] is not DictValue s) continue;
            var l = _layers[i];
            l.Name = s.GetValueOrDefault("name") as string ?? l.Name;
            l.Visible = s.GetValueOrDefault("visible") is not false;
            l.Opacity = Num(s.GetValueOrDefault("opacity"), 1);
            l.Blend = s.GetValueOrDefault("blend") as string is { } blend && BlendModes.Any(m => m.Id == blend) ? blend : l.Blend;
            l.FlipH = s.GetValueOrDefault("flipH") is true;
            l.FlipV = s.GetValueOrDefault("flipV") is true;
            if (s.GetValueOrDefault("transform") is DictValue t)
            {
                l.X = Num(t.GetValueOrDefault("x")); l.Y = Num(t.GetValueOrDefault("y"));
                if (Num(t.GetValueOrDefault("w")) > 0) l.W = Num(t.GetValueOrDefault("w"));
                if (Num(t.GetValueOrDefault("h")) > 0) l.H = Num(t.GetValueOrDefault("h"));
                l.Rotation = Num(t.GetValueOrDefault("rotation"));
            }
        }
        if (d.GetValueOrDefault("order") is ArrayValue order && order.Count == _layers.Count
            && order.Select(o => (int)Num(o, -1)).Order().SequenceEqual(Enumerable.Range(0, _layers.Count)))
        {
            _order.Clear();
            _order.AddRange(order.Select(o => (int)Num(o)));
        }
    }

    static double Num(object? v, double fallback = 0) => v switch
    {
        null => fallback,
        bool => fallback,
        string s => double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var d) ? d : fallback,
        _ => Convert.ToDouble(v, CultureInfo.InvariantCulture),
    };

    /// <summary>Whole numbers as integers, as the editor's JSON has them (x 120, w 512, but x 170.00027470263439).</summary>
    static object Store(double v) => v == Math.Floor(v) && Math.Abs(v) < 1e15 ? (object)(long)v : v;

    /// <summary>
    /// The widget's value: an empty object while nothing was edited, otherwise the composition, keys in the editor's order
    /// (version, canvas, inputs, background, order only when it differs from the input order, layers in input order).
    /// </summary>
    public DictValue Value()
    {
        if (!Edited) return [];
        var d = new DictValue
        {
            ["version"] = 1L,
            ["canvas"] = new DictValue { ["w"] = Store(CanvasW), ["h"] = Store(CanvasH) },
            ["inputs"] = new ArrayValue(Inputs.Select(i => (object?)i)),
            ["background"] = new DictValue { ["color"] = BackgroundColor, ["opacity"] = Store(BackgroundOpacity), ["visible"] = BackgroundVisible },
        };
        if (!_order.SequenceEqual(Enumerable.Range(0, _layers.Count))) d["order"] = new ArrayValue(_order.Select(o => (object?)(long)o));
        d["layers"] = new ArrayValue(_layers.Select(l => (object?)new DictValue
        {
            ["name"] = l.Name, ["visible"] = l.Visible, ["opacity"] = Store(l.Opacity), ["blend"] = l.Blend,
            ["transform"] = new DictValue { ["x"] = Store(l.X), ["y"] = Store(l.Y), ["w"] = Store(l.W), ["h"] = Store(l.H), ["rotation"] = Store(l.Rotation) },
            ["flipH"] = l.FlipH, ["flipV"] = l.FlipV,
        }));
        return d;
    }

    // ---- what the panels show ----

    /// <summary>JavaScript's Math.round: halves go up.</summary>
    public static double Round(double v) => Math.Floor(v + 0.5);

    /// <summary>The layer list, top-most first: each row's input index, name and line ("N 80%": the blend's first letter and the opacity).</summary>
    public IEnumerable<(int Index, string Name, string Line, bool Visible)> Rows() =>
        _order.AsEnumerable().Reverse().Select(i => (i, _layers[i].Name, RowLine(_layers[i]), _layers[i].Visible));

    static string RowLine(Layer l) =>
        $"{BlendModes.First(m => m.Id == l.Blend).Name[0]} {Round(l.Opacity * 100).ToString(CultureInfo.InvariantCulture)}%";

    /// <summary>The Background row's line: the canvas size, "632×572".</summary>
    public string BackgroundLine => $"{Round(CanvasW).ToString(CultureInfo.InvariantCulture)}×{Round(CanvasH).ToString(CultureInfo.InvariantCulture)}";

    /// <summary>The selected layer's fields as shown: X, Y, W, H, rotation in degrees, opacity in percent (all rounded). Null for the Background.</summary>
    public (double X, double Y, double W, double H, double Degrees, double Percent)? Fields() =>
        Selected is { } i && _layers[i] is var l
            ? (Round(l.X), Round(l.Y), Round(l.W), Round(l.H), Round(l.Rotation * 180 / Math.PI), Round(l.Opacity * 100))
            : null;

    // ---- undo ----

    Snapshot Take() => new([.. _layers.Select(l => l.Clone())], [.. _order], CanvasW, CanvasH, BackgroundColor, BackgroundOpacity, BackgroundVisible, Selected);

    void Put(Snapshot s, bool selection)
    {
        _layers.Clear(); _layers.AddRange(s.Layers.Select(l => l.Clone()));
        _order.Clear(); _order.AddRange(s.Order);
        (CanvasW, CanvasH) = (s.CanvasW, s.CanvasH);
        (BackgroundColor, BackgroundOpacity, BackgroundVisible) = (s.BackgroundColor, s.BackgroundOpacity, s.BackgroundVisible);
        if (selection) Selected = s.Selected;
    }

    /// <summary>One action: the state before it goes on the undo stack, the redo stack empties, and the value starts being written.</summary>
    void Act(Action change)
    {
        var before = _gesture ?? Take();
        change();
        if (_gesture is null) Commit(before);
    }

    void Commit(Snapshot before)
    {
        _undo.Add(before);
        _redo.Clear();
        Edited = true;
    }

    /// <summary>Starts a drag: every change until <see cref="EndGesture"/> is one undo step (a drag is one action).</summary>
    public void BeginGesture() => _gesture ??= Take();

    /// <summary>Ends a drag; it counts as an action only if something changed.</summary>
    public void EndGesture()
    {
        if (_gesture is not { } before) return;
        _gesture = null;
        if (!Same(before, Take())) Commit(before);
    }

    static bool Same(Snapshot a, Snapshot b) =>
        a.Order.SequenceEqual(b.Order) && a.CanvasW == b.CanvasW && a.CanvasH == b.CanvasH && a.BackgroundColor == b.BackgroundColor
        && a.BackgroundOpacity == b.BackgroundOpacity && a.BackgroundVisible == b.BackgroundVisible
        && a.Layers.Zip(b.Layers).All(p => p.First.Name == p.Second.Name && p.First.Visible == p.Second.Visible && p.First.Opacity == p.Second.Opacity
            && p.First.Blend == p.Second.Blend && p.First.X == p.Second.X && p.First.Y == p.Second.Y && p.First.W == p.Second.W && p.First.H == p.Second.H
            && p.First.Rotation == p.Second.Rotation && p.First.FlipH == p.Second.FlipH && p.First.FlipV == p.Second.FlipV);

    public bool Undo()
    {
        if (_undo.Count == 0) return false;
        _redo.Add(Take());
        Put(_undo[^1], selection: false);
        _undo.RemoveAt(_undo.Count - 1);
        return true;
    }

    public bool Redo()
    {
        if (_redo.Count == 0) return false;
        _undo.Add(Take());
        Put(_redo[^1], selection: false);
        _redo.RemoveAt(_redo.Count - 1);
        return true;
    }

    /// <summary>Back to the composition as it was when the editor opened (one action, so it can be undone).</summary>
    public void Restore() => Act(() => Put(_opened, selection: true));

    // ---- selection and the list ----

    /// <summary>Selects a layer by its input index, or the Background with null (a click on a row, on a layer, or on empty space).</summary>
    public void Select(int? index) => Selected = index is { } i && i >= 0 && i < _layers.Count ? i : null;

    public void ToggleVisible(int index) => Act(() => _layers[index].Visible = !_layers[index].Visible);

    public void ToggleBackgroundVisible() => Act(() => BackgroundVisible = !BackgroundVisible);

    /// <summary>Moves the selected layer one place up (towards the top) or down in the drawing order; nothing at the end of the stack.</summary>
    public bool MoveSelected(bool up)
    {
        if (Selected is not { } i) return false;
        int at = _order.IndexOf(i), to = at + (up ? 1 : -1);
        if (to < 0 || to >= _order.Count) return false;
        Act(() => (_order[at], _order[to]) = (_order[to], _order[at]));
        return true;
    }

    // ---- the selected layer's properties ----

    Layer? Current => Selected is { } i ? _layers[i] : null;

    /// <summary>Typed fields: taken as they are (ComfyUI's editor does not clamp them to the canvas). W and H change independently.</summary>
    public void SetPosition(double? x, double? y) { if (Current is { } l) Act(() => { l.X = x ?? l.X; l.Y = y ?? l.Y; }); }

    /// <summary>A typed size. One that is not positive is ignored (not measured: SpawnGen's choice, the node needs a size above 0).</summary>
    public void SetSize(double? w, double? h)
    {
        if (Current is not { } l) return;
        Act(() => { if (w is > 0) l.W = w.Value; if (h is > 0) l.H = h.Value; });
    }

    public void SetRotationDegrees(double degrees) { if (Current is { } l) Act(() => l.Rotation = degrees * Math.PI / 180); }

    /// <summary>A typed opacity in percent, kept within 0..100 (the range of the layer's opacity; the clamp itself was not measured).</summary>
    public void SetOpacityPercent(double percent) { if (Current is { } l) Act(() => l.Opacity = Math.Clamp(percent, 0, 100) / 100); }

    public void SetBlend(string id)
    {
        if (Current is not { } l || !BlendModes.Any(m => m.Id == id)) return;
        Act(() => l.Blend = id);
    }

    public void FlipHorizontal() { if (Current is { } l) Act(() => l.FlipH = !l.FlipH); }

    public void FlipVertical() { if (Current is { } l) Act(() => l.FlipV = !l.FlipV); }

    /// <summary>Aligns the selected layer's box to the canvas (measured on an unrotated layer; a rotated one aligns its unrotated box).</summary>
    public void Align(Alignment how)
    {
        if (Current is not { } l) return;
        Act(() =>
        {
            switch (how)
            {
                case Alignment.Left: l.X = 0; break;
                case Alignment.HorizontalCenter: l.X = (CanvasW - l.W) / 2; break;
                case Alignment.Right: l.X = CanvasW - l.W; break;
                case Alignment.Top: l.Y = 0; break;
                case Alignment.VerticalCenter: l.Y = (CanvasH - l.H) / 2; break;
                case Alignment.Bottom: l.Y = CanvasH - l.H; break;
            }
        });
    }

    // ---- the background and the canvas ----

    public void SetBackgroundColor(string color) => Act(() => BackgroundColor = color);

    /// <summary>The canvas size fields shown when the Background is selected. A size that is not positive is ignored.</summary>
    public void SetCanvasSize(double? w, double? h) => Act(() => { if (w is > 0) CanvasW = w.Value; if (h is > 0) CanvasH = h.Value; });

    // ---- pointer work on the canvas (canvas pixels: the screen drag over the zoom) ----

    /// <summary>A canvas point in a layer's own axes (origin at its centre, rotation undone).</summary>
    static (double X, double Y) ToLocal(Layer l, double x, double y)
    {
        double dx = x - l.CenterX, dy = y - l.CenterY, c = Math.Cos(-l.Rotation), s = Math.Sin(-l.Rotation);
        return (dx * c - dy * s, dx * s + dy * c);
    }

    static (double X, double Y) ToCanvas(double cx, double cy, double rotation, double lx, double ly)
    {
        double c = Math.Cos(rotation), s = Math.Sin(rotation);
        return (cx + lx * c - ly * s, cy + lx * s + ly * c);
    }

    /// <summary>How far above the top edge the rotation handle sits, in screen pixels (the Studio divides by the zoom).</summary>
    public const double RotateHandleOffset = 22;

    /// <summary>
    /// What is under a canvas point: a handle of the selected layer first (within <paramref name="reach"/> canvas pixels; the rotation
    /// handle <paramref name="rotateOffset"/> above the top edge), then the top-most visible layer whose box holds the point, else nothing
    /// (the Background).
    /// </summary>
    public (int? Layer, Handle Part) HitTest(double x, double y, double reach, double rotateOffset)
    {
        if (Current is { } sel)
        {
            var (lx, ly) = ToLocal(sel, x, y);
            double hw = sel.W / 2, hh = sel.H / 2;
            (double X, double Y, Handle H)[] handles =
            [
                (0, -hh - rotateOffset, Handle.Rotate), (-hw, -hh, Handle.TopLeft), (0, -hh, Handle.Top), (hw, -hh, Handle.TopRight), (hw, 0, Handle.Right),
                (hw, hh, Handle.BottomRight), (0, hh, Handle.Bottom), (-hw, hh, Handle.BottomLeft), (-hw, 0, Handle.Left),
            ];
            foreach (var h in handles)
                if (Math.Abs(lx - h.X) <= reach && Math.Abs(ly - h.Y) <= reach) return (Selected, h.H);
        }
        for (int k = _order.Count - 1; k >= 0; k--)
        {
            var l = _layers[_order[k]];
            if (!l.Visible) continue;
            var (lx, ly) = ToLocal(l, x, y);
            if (Math.Abs(lx) <= l.W / 2 && Math.Abs(ly) <= l.H / 2) return (_order[k], Handle.Body);
        }
        return (null, Handle.None);
    }

    /// <summary>Moves the selected layer to where it was at the start of the drag plus the drag (canvas pixels, unrounded).</summary>
    public void DragBody(Layer start, double dx, double dy)
    {
        if (Current is not { } l) return;
        Act(() => { l.X = start.X + dx; l.Y = start.Y + dy; });
    }

    /// <summary>
    /// Resizes the selected layer by a handle moved by (<paramref name="dx"/>, <paramref name="dy"/>) canvas pixels from where the HANDLE was
    /// at the start of the drag: the handle goes to the pointer (measured: a corner pressed 0.45 px off it lands on the pointer, not 0.45 off). The opposite corner or edge stays where it is. A corner resizes freely; with <paramref name="keepAspect"/> (Shift)
    /// the drag is projected on the layer's diagonal and both sides scale together. For a rotated layer the drag is taken in the layer's
    /// own axes (not measured). A side never goes under one pixel (not measured).
    /// </summary>
    public void DragHandle(Layer start, Handle handle, double dx, double dy, bool keepAspect)
    {
        if (Current is not { } l || handle is Handle.None or Handle.Body or Handle.Rotate) return;
        // the drag in the layer's axes
        double c = Math.Cos(-start.Rotation), s = Math.Sin(-start.Rotation);
        double lx = dx * c - dy * s, ly = dx * s + dy * c;
        int sx = handle is Handle.TopLeft or Handle.Left or Handle.BottomLeft ? -1 : handle is Handle.TopRight or Handle.Right or Handle.BottomRight ? 1 : 0;
        int sy = handle is Handle.TopLeft or Handle.Top or Handle.TopRight ? -1 : handle is Handle.BottomLeft or Handle.Bottom or Handle.BottomRight ? 1 : 0;
        double w = start.W, h = start.H;
        if (keepAspect && sx != 0 && sy != 0)
        {
            double diag = Math.Sqrt(start.W * start.W + start.H * start.H);
            double t = (lx * sx * start.W + ly * sy * start.H) / diag;   // the drag along the outward diagonal
            w = start.W + t * start.W / diag;
            h = start.H + t * start.H / diag;
        }
        else
        {
            if (sx != 0) w = start.W + lx * sx;
            if (sy != 0) h = start.H + ly * sy;
        }
        w = Math.Max(1, w); h = Math.Max(1, h);
        // the anchor (the opposite corner or edge middle) keeps its place on the canvas
        double ax = -sx * start.W / 2, ay = -sy * start.H / 2;                    // the anchor in the start layer's axes
        var (wx, wy) = ToCanvas(start.CenterX, start.CenterY, start.Rotation, ax, ay);
        double nax = -sx * w / 2, nay = -sy * h / 2;                              // the same anchor in the resized layer's axes
        double cr = Math.Cos(start.Rotation), sr = Math.Sin(start.Rotation);
        double ncx = wx - (nax * cr - nay * sr), ncy = wy - (nax * sr + nay * cr);
        Act(() => { l.W = w; l.H = h; l.X = ncx - w / 2; l.Y = ncy - h / 2; });
    }

    /// <summary>
    /// The rotation handle: the layer turns about its centre to the angle of the pointer (canvas point), 0 with the pointer straight above the
    /// centre, growing clockwise (a pointer to the right of the centre is a quarter turn: 1.5688 rad measured for that drag).
    /// </summary>
    public void DragRotate(Layer start, double x, double y)
    {
        if (Current is not { } l) return;
        double angle = Math.Atan2(x - start.CenterX, -(y - start.CenterY));
        Act(() => l.Rotation = angle);
    }
}
