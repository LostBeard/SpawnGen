using System.Text.Json.Nodes;
using SpawnGen.Editor;
using SpawnGen.Engine.Types;

namespace SpawnGen.Studio.Pages;

// Create Layered Image's compositor editor: the node's "Open Compositor" button opens CompositorDialog on the layers of the node's last run;
// every action writes the composition to the node's compositor widget (as ComfyUI's editor does: the value is rewritten after each action,
// not on Close), one graph undo step per action.
public partial class Studio
{
    sealed record OpenCompositor(int NodeId, string Widget, CompositorEditor Editor, IReadOnlyList<string> Urls);
    OpenCompositor? _compositor;

    /// <summary>What a run of Create Layered Image reported for the editor (ui compositor_inputs / _bboxes / _canvas / _layers), or null.</summary>
    CompositorRun? CompositorRunOf(JsonObject data)
    {
        if (data["output"] is not JsonObject output || output["compositor_inputs"] is not JsonArray inputs || output["compositor_bboxes"] is not JsonArray boxes) return null;
        var canvas = (output["compositor_canvas"] as JsonArray)?.OfType<JsonObject>().FirstOrDefault();
        double w = (double?)canvas?["w"] ?? 0, h = (double?)canvas?["h"] ?? 0;
        if (inputs.Count == 0 || boxes.Count != inputs.Count || w <= 0 || h <= 0) return null;
        var urls = (output["compositor_layers"] as JsonArray)?.OfType<JsonObject>()
            .Select(i => FileUrl(_engine!, (string)i["filename"]!, (string?)i["subfolder"], (string?)i["type"])).ToList() ?? [];
        return new CompositorRun([.. inputs.Select(i => (string?)i ?? "")], [.. boxes.Select(b => JsonLiteral.FromJson(b) as DictValue ?? [])], w, h, urls);
    }

    void OpenCompositorFor(EditorNode node, string widget)
    {
        if (node.CompositorRun is not { } run)
        {
            ShowMessage(CompositorEditor.NeedsRun + ", then open the compositor.");
            return;
        }
        var saved = node.Widgets.GetValueOrDefault(widget);
        if (saved is string text && JsonLiteral.TryParse(text, out var parsed)) saved = parsed;
        _compositor = new OpenCompositor(node.Id, widget, new CompositorEditor(run.Inputs, run.Boxes, (run.CanvasW, run.CanvasH), saved), run.LayerUrls);
    }

    void OnCompositorChanged()
    {
        if (_compositor is not { } open || _doc.Graph.FindNode(open.NodeId) is not { } node) return;
        var value = open.Editor.Value();
        if (JsonLiteral.ToJson(value) == JsonLiteral.ToJson(node.Widgets.GetValueOrDefault(open.Widget))) return;
        _doc.SetWidget(open.NodeId, open.Widget, value);
        _doc.EndGesture();
        _view.Invalidate();
    }

    void CloseCompositor()
    {
        _compositor = null;
        _studio?.Focus();
    }
}
