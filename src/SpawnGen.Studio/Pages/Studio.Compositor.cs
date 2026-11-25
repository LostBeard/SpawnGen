using System.Text.Json.Nodes;
using SpawnGen.Editor;
using Microsoft.JSInterop;
using SpawnDev.SpawnJS.JSObjects;
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

    bool _compositorExporting;

    /// <summary>
    /// Export PSD: the layer pictures of the run are read back, drawn as a run draws them (ImageCompositor.Export) and written as a layered
    /// PSD by SpawnDev.Codecs on this browser's GPU, then downloaded. The writer runs as GPU kernels, so it needs WebGPU in this browser
    /// even when a SpawnGen server runs the graphs.
    /// </summary>
    async Task ExportCompositorPsdAsync()
    {
        if (_compositor is not { } open || _compositorExporting) return;
        _compositorExporting = true;
        StateHasChanged();
        try
        {
            var pictures = new List<(byte[] Rgba, int Width, int Height)>();
            foreach (var url in open.Urls)
            {
                using var response = await JS.Fetch(url);
                if (!response.Ok) throw new InvalidOperationException("a layer picture could not be read");
                using var buffer = await response.ArrayBuffer();
                using var data = new Uint8Array(buffer);
                var (image, alpha) = await SpawnGen.Nodes.Core.ImageDecoder.DecodeFirstAsync((byte[])data);
                int n = image.Width * image.Height, c = image.Channels;
                var rgba = new byte[n * 4];
                static byte Q(float v) => (byte)Math.Clamp(MathF.Round(v * 255f, MidpointRounding.ToEven), 0, 255);
                for (int i = 0; i < n; i++)
                {
                    for (int k = 0; k < 3; k++) rgba[i * 4 + k] = Q(image.Data[i * c + (c >= 3 ? k : 0)]);
                    rgba[i * 4 + 3] = alpha is not null ? Q(alpha[i]) : c == 4 ? Q(image.Data[i * c + 3]) : (byte)255;
                }
                pictures.Add((rgba, image.Width, image.Height));
            }
            var export = SpawnGen.Nodes.Core.ImageCompositor.Export(pictures, open.Editor.Composition());
            var runtime = await Browser.GetAsync();
            var bytes = await SpawnGen.Nodes.ML.CompositorPsd.EncodeAsync(runtime.Accelerator, export);
            using var blob = new Blob(new[] { bytes }, new BlobOptions { Type = "image/vnd.adobe.photoshop" });
            var link = URL.CreateObjectURL(blob);
            await Interop.InvokeVoidAsync("spawngenDownloadUrl", SpawnGen.Nodes.ML.CompositorPsd.FileName(DateTime.Now), link);
            _ = Task.Delay(5000).ContinueWith(_ => URL.RevokeObjectURL(link));
            ShowMessage($"Exported a PSD with {export.Layers.Count} layers ({bytes.Length / 1024.0:0} KB).");
        }
        catch (Exception ex)
        {
            ShowMessage("Export PSD failed: " + ex.Message.Split((char)10)[0], 10000);
        }
        finally
        {
            _compositorExporting = false;
            StateHasChanged();
        }
    }

    void CloseCompositor()
    {
        _compositor = null;
        _studio?.Focus();
    }
}
