using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnGen.Editor;

namespace SpawnGen.Studio.Canvas;

/// <summary>
/// Connects a browser canvas to an <see cref="EditorView"/>: forwards pointer, wheel and key events, and repaints when the view says something
/// changed. There is no Blazor re-render involved while editing, so dragging stays smooth.
/// </summary>
public sealed class StudioCanvas : IDisposable
{
    readonly SpawnJSRuntime _js;
    readonly HTMLCanvasElement _canvas;
    readonly CanvasRenderingContext2D _ctx;
    readonly EditorView _view;
    readonly NodeRenderer _renderer;
    readonly PeriodicTimer _timer = new(TimeSpan.FromMilliseconds(16));
    readonly CancellationTokenSource _stop = new();
    bool _dirty = true;
    bool _space;
    double _cssWidth, _cssHeight, _ratio = 1;
    DateTime _lastDown;
    Point _lastDownAt;

    // The handlers are kept so they can be removed again.
    readonly Action<PointerEvent> _onDown, _onMove, _onUp;
    readonly Action<WheelEvent> _onWheel;
    readonly Action<KeyboardEvent> _onKeyDown, _onKeyUp;
    readonly Action<MouseEvent> _onContext;

    public StudioCanvas(SpawnJSRuntime js, HTMLCanvasElement canvas, EditorView view, NodeRenderer renderer)
    {
        _js = js;
        _canvas = canvas;
        _view = view;
        _renderer = renderer;
        _ctx = canvas.Get2DContext();
        renderer.UseMeasureContext(_ctx);

        _view.ViewChanged += MarkDirty;
        _onDown = OnDown; _onMove = OnMove; _onUp = OnUp; _onWheel = OnWheel; _onKeyDown = OnKeyDown; _onKeyUp = OnKeyUp; _onContext = OnContext;
        canvas.OnPointerDown += _onDown;
        canvas.OnPointerMove += _onMove;
        canvas.OnPointerUp += _onUp;
        canvas.OnWheel += _onWheel;
        canvas.OnKeyDown += _onKeyDown;
        canvas.OnKeyUp += _onKeyUp;
        canvas.OnContextMenu += _onContext;

        _ = RunAsync();
    }

    public EditorView View => _view;
    public void MarkDirty() => _dirty = true;

    /// <summary>The canvas position of a world point in CSS pixels (for placing HTML overlays such as the widget editor).</summary>
    public Point WorldToScreen(Point world) => _view.Viewport.WorldToScreen(world);

    // ---- painting ----

    async Task RunAsync()
    {
        try
        {
            while (await _timer.WaitForNextTickAsync(_stop.Token))
            {
                if (SyncSize()) _dirty = true;
                if (!_dirty) continue;
                _dirty = false;
                _renderer.Draw(_ctx, _cssWidth, _cssHeight, _ratio);
            }
        }
        catch (OperationCanceledException) { }
    }

    /// <summary>Keeps the canvas bitmap matched to its CSS size and the display's pixel ratio. True when it changed.</summary>
    bool SyncSize()
    {
        using var rect = _canvas.GetBoundingClientRect();
        double w = Math.Max(1, Math.Round(rect.Width)), h = Math.Max(1, Math.Round(rect.Height));
        double ratio = _js.Get<double>("devicePixelRatio");
        if (w == _cssWidth && h == _cssHeight && ratio == _ratio) return false;
        (_cssWidth, _cssHeight, _ratio) = (w, h, ratio);
        _canvas.Width = (int)Math.Round(w * ratio);
        _canvas.Height = (int)Math.Round(h * ratio);
        _view.ScreenSize = new Point(w, h);
        if (!_sized)
        {
            _sized = true;
            FirstLayout?.Invoke(); // the first moment the canvas size is real: a fit-to-view before this would use a guess
        }
        return true;
    }

    bool _sized;

    /// <summary>Raised once, when the canvas has been measured for the first time.</summary>
    public event Action? FirstLayout;

    /// <summary>Raised at the start of every pointer press on the canvas, before the editor handles it (lets the page close menus).</summary>
    public event Action? CanvasPressed;

    // ---- input ----

    static Modifiers Mods(MouseEvent e, bool space) => Mods(e.ShiftKey, e.CtrlKey || e.MetaKey, e.AltKey, space);

    static Modifiers Mods(KeyboardEvent e, bool space) => Mods(e.ShiftKey, e.CtrlKey || e.MetaKey, e.AltKey, space);

    static Modifiers Mods(bool shift, bool ctrl, bool alt, bool space) =>
        (shift ? Modifiers.Shift : 0) | (ctrl ? Modifiers.Ctrl : 0) | (alt ? Modifiers.Alt : 0) | (space ? Modifiers.Space : 0);

    static PointerButton Button(MouseButton b) => b switch
    {
        MouseButton.PrimaryButton => PointerButton.Left,
        MouseButton.AuxiliaryButton => PointerButton.Middle,
        MouseButton.SecondaryButton => PointerButton.Right,
        _ => PointerButton.None,
    };

    void OnDown(PointerEvent e)
    {
        try
        {
            var p = new Point(e.OffsetX, e.OffsetY);
            CanvasPressed?.Invoke();
            _canvas.Focus();
            _canvas.SetPointerCapture(e.PointerId);

            // Pointer events carry no click count, so a second press close in time and place is a double click.
            var now = DateTime.UtcNow;
            int clicks = (now - _lastDown).TotalMilliseconds < 350 && p.DistanceTo(_lastDownAt) < 6 ? 2 : 1;
            (_lastDown, _lastDownAt) = clicks == 2 ? (DateTime.MinValue, p) : (now, p);

            _view.PointerDown(p, Button(e.Button), Mods(e, _space), clicks);
        }
        finally { e.Dispose(); }
    }

    void OnMove(PointerEvent e)
    {
        try { _view.PointerMove(new Point(e.OffsetX, e.OffsetY), Mods(e, _space)); }
        finally { e.Dispose(); }
    }

    void OnUp(PointerEvent e)
    {
        try
        {
            _canvas.ReleasePointerCapture(e.PointerId);
            _view.PointerUp(new Point(e.OffsetX, e.OffsetY), Button(e.Button), Mods(e, _space));
        }
        finally { e.Dispose(); }
    }

    void OnWheel(WheelEvent e)
    {
        try
        {
            e.PreventDefault();
            // Line-mode wheels (some mice) report small numbers; scale them to pixels.
            double dy = e.DeltaMode == 1 ? e.DeltaY * 33 : e.DeltaY;
            _view.Wheel(new Point(e.OffsetX, e.OffsetY), dy, Mods(e, _space));
        }
        finally { e.Dispose(); }
    }

    void OnContext(MouseEvent e)
    {
        e.PreventDefault(); // the editor shows its own menu
        e.Dispose();
    }

    void OnKeyDown(KeyboardEvent e)
    {
        try
        {
            var key = e.Key.ToLowerInvariant();
            if (key == " ") { _space = true; e.PreventDefault(); return; }
            if (_view.Key(key, Mods(e, _space))) e.PreventDefault();
        }
        finally { e.Dispose(); }
    }

    void OnKeyUp(KeyboardEvent e)
    {
        try { if (e.Key == " ") _space = false; }
        finally { e.Dispose(); }
    }

    public void Focus() => _canvas.Focus();

    public void Dispose()
    {
        _stop.Cancel();
        _timer.Dispose();
        _view.ViewChanged -= MarkDirty;
        _canvas.OnPointerDown -= _onDown;
        _canvas.OnPointerMove -= _onMove;
        _canvas.OnPointerUp -= _onUp;
        _canvas.OnWheel -= _onWheel;
        _canvas.OnKeyDown -= _onKeyDown;
        _canvas.OnKeyUp -= _onKeyUp;
        _canvas.OnContextMenu -= _onContext;
        _ctx.Dispose();
        _canvas.Dispose();
    }
}
