namespace SpawnGen.Editor;

/// <summary>
/// Where Painter nodes' paintings live: the Studio keeps each in the page (JS canvases, AGENTS.md rule 4) and draws it; the editor only says
/// what the pointer did, in canvas pixels. A stroke draws at full strength while it lasts and lands on the painting at the brush's opacity
/// when it ends (ComfyUI: nothing builds up inside one stroke past its own coverage, separate strokes composite source-over).
/// </summary>
public interface IPainterCanvas
{
    /// <summary>A stroke starts at (x, y) on a canvas of <paramref name="size"/> with this brush (already <see cref="PainterWidget.Brush.Effective"/>).</summary>
    void BeginStroke(int nodeId, (int W, int H) size, PainterWidget.Brush brush, double x, double y);

    /// <summary>More stamps of the current stroke, in order.</summary>
    void Stamps(int nodeId, IReadOnlyList<(double X, double Y)> stamps);

    /// <summary>The stroke is finished: it lands on the painting.</summary>
    void EndStroke(int nodeId);

    /// <summary>The Clear button: the painting becomes empty (transparent).</summary>
    void Clear(int nodeId);
}
