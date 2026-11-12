using Microsoft.AspNetCore.Components.Forms;
using SpawnGen.Editor;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Studio.Pages;

// App Mode (as docs.comfy.org/interface/app-mode describes ComfyUI's): the workflow run from a form of chosen inputs with its chosen outputs, the node
// graph out of sight. The App Builder picks the inputs (widgets) and outputs (output nodes) on the canvas in four steps (Inputs, Outputs, Preview,
// Default view); the setup is saved in the workflow (extra.linearData / extra.linearMode), so ComfyUI opens it as the same app.
public partial class Studio
{
    enum AppView { Graph, Builder, App }

    /// <summary>The builder's steps, in order.</summary>
    static readonly string[] BuilderSteps = ["Inputs", "Outputs", "Preview", "Default view"];

    AppView _appView = AppView.Graph;
    int _builderStep;            // 0 inputs, 1 outputs, 2 preview, 3 default view
    AppModeConfig _app = new();
    bool _builderOpensAsApp = true;
    int _appRunShown;            // which run the app's output area shows: 0 = the newest

    /// <summary>One result of a run, with the node that made it (App Mode shows the chosen output nodes' results).</summary>
    sealed record RunItem(string? Node, string Kind, string Value);

    /// <summary>Reads the tab's app setup; a workflow saved to open as an app opens as one.</summary>
    void SyncAppFromDocument(bool opening)
    {
        _app = AppModeConfig.Read(_doc.Extra);
        _appRunShown = 0;
        _appView = opening && _app.Configured && _app.OpensAsApp == true ? AppView.App : AppView.Graph;
        if (_appView == AppView.App) _view.ClearSelection();
        ApplyPicking();
    }

    void ToggleAppMode()
    {
        if (_appView != AppView.Graph) { ExitAppMode(); return; }
        EnterAppMode();
    }

    /// <summary>App mode; a workflow never set up as an app goes to the builder first (as in ComfyUI).</summary>
    void EnterAppMode()
    {
        if (_subgraphStack.Count > 0) ExitSubgraphTo(0); // the app runs the whole workflow
        _app = AppModeConfig.Read(_doc.Extra);
        if (!_app.Configured) OpenBuilder();
        else { _appView = AppView.App; _view.ClearSelection(); ApplyPicking(); }
        StateHasChanged();
    }

    void ExitAppMode()
    {
        _appView = AppView.Graph;
        ApplyPicking();
        _studio?.Focus();
        StateHasChanged();
    }

    void OpenBuilder()
    {
        if (_subgraphStack.Count > 0) ExitSubgraphTo(0);
        _app = AppModeConfig.Read(_doc.Extra);
        _builderOpensAsApp = _app.OpensAsApp ?? true;
        _builderStep = 0;
        _appView = AppView.Builder;
        _edit = null;
        _propsOpen = _historyOpen = false; // the builder's own panel takes the right side
        _view.FitToView(left: 0, top: 60, right: 300, bottom: 48); // the whole graph in the part the builder's bars and panel leave free
        ApplyPicking();
        StateHasChanged();
    }

    void BuilderGo(int step)
    {
        _builderStep = Math.Clamp(step, 0, BuilderSteps.Length - 1);
        ApplyPicking();
        StateHasChanged();
    }

    /// <summary>Default view, Apply: the setup goes into the workflow (saved with it), then the app shows.</summary>
    void ApplyApp()
    {
        _app.OpensAsApp = _builderOpensAsApp;
        _app.Configured = true;
        _app.Write(_doc);
        ScheduleAutosave();
        ShowMessage(_builderOpensAsApp ? "App saved in the workflow: it opens as an app." : "App saved in the workflow: it opens in the node graph.");
        _appView = AppView.App;
        _view.ClearSelection();
        ApplyPicking();
        StateHasChanged();
    }

    /// <summary>The canvas picks widgets in the Inputs step and output nodes in the Outputs step, and shows what is picked.</summary>
    void ApplyPicking()
    {
        bool building = _appView == AppView.Builder;
        _view.Picking = building && _builderStep == 0 ? PickTarget.Widgets : building && _builderStep == 1 ? PickTarget.Nodes : PickTarget.None;
        _visuals.ShowAppInputs = _view.Picking == PickTarget.Widgets;
        _visuals.ShowAppOutputs = _view.Picking == PickTarget.Nodes;
        _visuals.AppInputs.Clear();
        foreach (var i in _app.Inputs)
            if (int.TryParse(i.NodeId, out var id)) _visuals.AppInputs.Add((id, i.Widget));
        _visuals.AppOutputs.Clear();
        foreach (var o in _app.Outputs)
            if (int.TryParse(o, out var id)) _visuals.AppOutputs.Add(id);
        _visuals.AppOutputCandidates.Clear();
        foreach (var n in _doc.Graph.Nodes)
            if (AppModeConfig.IsOutputNode(Registry, n)) _visuals.AppOutputCandidates.Add(n.Id);
        _view.Invalidate();
    }

    void OnAppWidgetPicked(int nodeId, string widget)
    {
        bool added = _app.ToggleInput(nodeId, widget);
        ShowMessage(added ? $"Input {_app.Inputs.Count}: {InputLabel(_app.Inputs[^1])}" : "Taken out of the app's inputs.");
        ApplyPicking();
        StateHasChanged();
    }

    void OnAppNodePicked(int nodeId)
    {
        var node = _doc.Graph.FindNode(nodeId);
        if (node is null) return;
        if (!AppModeConfig.IsOutputNode(Registry, node))
        {
            ShowMessage("Only output nodes (Save Image, Preview Image, Save Audio ...) can be app outputs.");
            return;
        }
        bool added = _app.ToggleOutput(nodeId);
        ShowMessage(added ? $"Output {_app.Outputs.Count}: {NodeLabel(node)}" : "Taken out of the app's outputs.");
        ApplyPicking();
        StateHasChanged();
    }

    void MoveAppInput(int index, int delta) { AppModeConfig.Move(_app.Inputs, index, delta); ApplyPicking(); }
    void RemoveAppInput(int index) { _app.Inputs.RemoveAt(index); ApplyPicking(); }
    void MoveAppOutput(int index, int delta) { AppModeConfig.Move(_app.Outputs, index, delta); ApplyPicking(); }
    void RemoveAppOutput(int index) { _app.Outputs.RemoveAt(index); ApplyPicking(); }

    string NodeLabel(EditorNode n) => n.Title ?? Registry.GetSchema(n.ClassType)?.DisplayName ?? n.ClassType;

    string InputLabel(AppInput input) =>
        AppModeConfig.Resolve(_doc.Graph, input.NodeId) is { } n ? $"{NodeLabel(n)} · {input.Widget}" : $"#{input.NodeId} · {input.Widget} (not in this workflow)";

    string OutputLabel(string id) => AppModeConfig.Resolve(_doc.Graph, id) is { } n ? NodeLabel(n) : $"#{id} (not in this workflow)";

    /// <summary>An app input's widget spec on its node, or null when the node or the widget is gone.</summary>
    (EditorNode Node, WidgetSpec Spec, string? Tooltip)? InputField(AppInput input)
    {
        if (AppModeConfig.Resolve(_doc.Graph, input.NodeId) is not { } node || Registry.GetSchema(node.ClassType) is not { } schema) return null;
        if (input.Widget == ControlAfterGenerate.Name) return (node, ControlAfterGenerate.Spec, "What happens to the seed after each run");
        var spec = NodeLayout.OrderedInputs(schema, node.Widgets).FirstOrDefault(i => i.Name == input.Widget);
        return spec?.Widget is { } w ? (node, w, spec.Tooltip) : null;
    }

    /// <summary>Sets an app input: an undo step for a node of the workflow; a node inside a subgraph is changed in its definition.</summary>
    void SetAppInput(EditorNode node, string widget, object? value)
    {
        if (_doc.Graph.FindNode(node.Id) is { } top && ReferenceEquals(top, node))
        {
            _doc.SetWidget(node.Id, widget, value);
            _doc.EndGesture();
        }
        else
        {
            node.Widgets[widget] = value;
            ScheduleAutosave();
        }
        StateHasChanged();
    }

    bool IsPictureInput(EditorNode node, string widget) => node.ClassType is "LoadImage" or "LoadImageMask" && widget == "image";

    string? PictureUrl(EditorNode node, string widget) =>
        _engine is not null && node.Widgets.GetValueOrDefault(widget) is string { Length: > 0 } name ? _engine.ViewUrl(name, "", "input") : null;

    /// <summary>A picture uploaded from the app goes to the engine's inputs and into the input it was uploaded for.</summary>
    async Task OnAppPictureChosenAsync(InputFileChangeEventArgs e, EditorNode node, string widget)
    {
        if (_engine is null) { ShowMessage("There is no engine to upload to."); return; }
        try
        {
            await using var stream = e.File.OpenReadStream(32 * 1024 * 1024);
            using var memory = new MemoryStream();
            await stream.CopyToAsync(memory);
            var stored = await _engine.UploadImageAsync(e.File.Name, memory.ToArray());
            await RefreshInputImagesAsync();
            if (stored is null) { ShowMessage($"Could not upload {e.File.Name}."); return; }
            SetAppInput(node, widget, stored);
        }
        catch (IOException ex) { ShowMessage($"Could not read {e.File.Name}: {ex.Message}"); }
    }

    /// <summary>The run the app's output area shows, and its results from the app's output nodes (all its results when none are chosen).</summary>
    (HistoryEntry? Run, List<RunItem> Items) AppResults()
    {
        var run = _history.ElementAtOrDefault(Math.Min(_appRunShown, Math.Max(_history.Count - 1, 0)));
        if (run is null) return (null, []);
        if (_app.Outputs.Count == 0) return (run, run.Items);
        var items = new List<RunItem>();
        foreach (var id in _app.Outputs) items.AddRange(run.Items.Where(i => AppModeConfig.IsNode(i.Node, id)));
        return (run, items);
    }

    const string InputsHint = "Click a parameter on a node to make it an app input; click it again to take it out. The app shows them in this order.";
    const string OutputsHint = "Click an output node (Save Image, Preview Image, Save Audio ...) to show its results in the app. With none chosen the app shows every result.";

    string StepClass(int step) => step == _builderStep ? "on" : step < _builderStep ? "done" : "";

    string AppProgressText() => RunStatus().TrimEnd(' ', '·');

    static string ProgressWidth(double? fraction) =>
        (fraction is { } f ? (f * 100).ToString("0.#", System.Globalization.CultureInfo.InvariantCulture) : "30") + "%";

    static string EmptyResultText(bool preview, HistoryEntry? run) =>
        preview ? "The results of the chosen outputs show here."
        : run is null ? "Press Run to see the result here."
        : run.State == "running" ? "Working..."
        : run.Error ?? "This run gave no results to show.";

    string RunTitle() => _engine is null ? "No engine" : $"Run on: {_engine.Name} (Ctrl+Enter)";

    /// <summary>A widget of a workflow node that a link sets (its value is not the widget's).</summary>
    bool IsLinkedInput(EditorNode node, string widget) =>
        _doc.Graph.FindNode(node.Id) is { } top && ReferenceEquals(top, node) && _doc.Graph.LinkInto(node.Id, widget) is not null;

    /// <summary>Runs from the app: the workflow as it is, with the app's inputs set; the newest run is shown.</summary>
    async Task AppRunAsync()
    {
        _appRunShown = 0;
        await QueueAsync();
    }
}
