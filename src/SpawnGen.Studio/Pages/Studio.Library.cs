using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Web;
using Microsoft.JSInterop;
using SpawnGen.Editor;
using SpawnGen.Studio.Run;

namespace SpawnGen.Studio.Pages;

// The workflow library, ComfyUI's: saved workflows in the user's workflows/ folder (on the server that serves the Studio, else in this browser),
// apps among them as name.app.json. The Workflows sidebar lists the open tabs and the saved workflows (open, rename, delete); its Apps tab lists
// the apps and makes new ones. Save (Ctrl+S) saves the tab into the library, asking for a name the first time; Export downloads the file.
public partial class Studio
{
    IWorkflowStore? _store;
    bool _workflowsOpen;
    bool _appsTab;                    // the sidebar's Apps tab (else Workflows)
    string _workflowFilter = "";
    List<WorkflowEntry> _saved = [];
    string? _renamingPath;            // a saved file being renamed in the list
    string? _deletingPath;            // a saved file asking "Delete?"

    /// <summary>The Save dialog: the name typed, whether it saves a copy (Save as), and the file it would replace when that name is taken.</summary>
    sealed class SaveDialog
    {
        public string Name = "";
        public bool Copy;
        public string? Replaces;
    }
    SaveDialog? _saveDialog;

    IWorkflowStore Store => _store ??= _serverStore ? new ServerWorkflowStore(Http) : new BrowserWorkflowStore(Interop);
    bool _serverStore;

    async Task ToggleWorkflowsAsync(bool apps)
    {
        if (_workflowsOpen && _appsTab == apps) { _workflowsOpen = false; _studio?.Focus(); return; }
        _workflowsOpen = true;
        _appsTab = apps;
        _libraryOpen = false; // one sidebar on the left
        await RefreshSavedAsync();
    }

    async Task RefreshSavedAsync()
    {
        _saved = [.. await Store.ListAsync()];
        StateHasChanged();
    }

    IEnumerable<WorkflowEntry> SavedShown() =>
        WorkflowLibrary.Sorted(_saved.Where(e => (!_appsTab || e.IsApp) && WorkflowLibrary.Matches(e, _workflowFilter)));

    /// <summary>Opens a saved workflow: the tab that has it open, else a new tab; an app opens as the app.</summary>
    async Task OpenSavedAsync(WorkflowEntry entry)
    {
        if (_tabs.Tabs.FirstOrDefault(t => t.Path == entry.Path) is { } open)
        {
            SwitchTab(open.Id);
            if (entry.IsApp && _appView == AppView.Graph && AppModeConfig.Read(_doc.Extra).Configured) EnterAppMode();
            return;
        }
        var text = await Store.ReadAsync(entry.Path);
        if (text is null) { ShowMessage($"Could not read {entry.Name} {Store.Where}."); await RefreshSavedAsync(); return; }
        if (!OpenWorkflowText(text, System.IO.Path.GetFileName(entry.Name), entry.Name, entry.Path)) return;
        // An app file opens as the app even when it was saved without a default view (ComfyUI's apps list opens apps).
        if (entry.IsApp && _appView == AppView.Graph && _app.Configured) EnterAppMode();
    }

    /// <summary>Save (Ctrl+S): a tab saved before is written again; else the dialog asks for a name.</summary>
    async Task SaveToLibraryAsync()
    {
        if (_tabs.Active is not { } tab) return;
        if (tab.Path is { } path)
        {
            await WriteTabAsync(tab, path, overwrite: true);
            return;
        }
        OpenSaveDialog(copy: false);
    }

    void OpenSaveDialog(bool copy)
    {
        var name = _tabs.Active?.Path is { } p ? WorkflowLibrary.DisplayName(p) : _tabs.Active?.Name ?? "Workflow";
        _saveDialog = new SaveDialog { Name = name, Copy = copy };
        _focusLibraryInput = true;
        StateHasChanged();
    }

    /// <summary>The dialog's Save: a name that is taken asks once more before it replaces that file.</summary>
    async Task ConfirmSaveAsync()
    {
        if (_saveDialog is not { } dialog || _tabs.Active is not { } tab) return;
        bool app = (_app.Configured && _app.OpensAsApp == true) || (tab.Path is { } old && WorkflowLibrary.IsApp(old));
        var path = WorkflowLibrary.PathFor(dialog.Name, app);
        if (path is null) { ShowMessage("Type a name to save the workflow under."); return; }
        bool overwrite = dialog.Replaces == path || path == tab.Path;
        if (!overwrite)
        {
            await RefreshSavedAsync();
            if (_saved.Any(e => e.Path == path)) { dialog.Replaces = path; StateHasChanged(); return; } // asks: replace it?
        }
        // Not overwriting what someone else saved meanwhile: that answer (Exists) asks too.
        if (await WriteTabAsync(tab, path, overwrite) == StoreResult.Exists)
        {
            dialog.Replaces = path; // asks: replace it?
            StateHasChanged();
            return;
        }
        _saveDialog = null;
        StateHasChanged();
    }

    async Task<StoreResult> WriteTabAsync(WorkflowTab tab, string path, bool overwrite)
    {
        var json = ComfyWorkflow.Save(_doc, _renderer.MeasureText);
        var result = await Store.WriteAsync(path, json, overwrite);
        if (result == StoreResult.Exists) return result;
        if (result != StoreResult.Done) { ShowMessage($"Could not save {WorkflowLibrary.DisplayName(path)} {Store.Where}."); return result; }
        foreach (var other in _tabs.Tabs.Where(t => t.Path == path && !ReferenceEquals(t, tab))) other.Path = null; // the file is this tab's now
        tab.Path = path;
        _tabs.Rename(tab.Id, System.IO.Path.GetFileName(WorkflowLibrary.DisplayName(path)));
        PersistTabs();
        ShowMessage($"Saved {WorkflowLibrary.DisplayName(path)} {Store.Where}.");
        await RefreshSavedAsync();
        return result;
    }

    async Task RenameSavedAsync(WorkflowEntry entry, string name)
    {
        _renamingPath = null;
        var to = WorkflowLibrary.Renamed(entry.Path, name);
        if (to is null || to == entry.Path) { StateHasChanged(); return; }
        var result = await Store.MoveAsync(entry.Path, to, overwrite: false);
        if (result == StoreResult.Exists) { ShowMessage($"There is already a workflow named {WorkflowLibrary.DisplayName(to)}."); return; }
        if (result != StoreResult.Done) { ShowMessage($"Could not rename {entry.Name}."); await RefreshSavedAsync(); return; }
        foreach (var tab in _tabs.Tabs.Where(t => t.Path == entry.Path).ToList())
        {
            tab.Path = to;
            _tabs.Rename(tab.Id, System.IO.Path.GetFileName(WorkflowLibrary.DisplayName(to)));
        }
        PersistTabs();
        await RefreshSavedAsync();
    }

    async Task DeleteSavedAsync(WorkflowEntry entry)
    {
        _deletingPath = null;
        if (await Store.DeleteAsync(entry.Path) is not (StoreResult.Done or StoreResult.Missing)) { ShowMessage($"Could not delete {entry.Name}."); return; }
        foreach (var tab in _tabs.Tabs.Where(t => t.Path == entry.Path)) tab.Path = null; // still open, no longer saved
        PersistTabs();
        ShowMessage($"Deleted {entry.Name}.");
        await RefreshSavedAsync();
    }

    /// <summary>Apps, Create app: the App Builder on the workflow in front.</summary>
    void CreateApp()
    {
        _workflowsOpen = false;
        OpenBuilder();
    }

    /// <summary>Export: the workflow downloaded as a file (an app as name.app.json).</summary>
    async Task ExportAsync()
    {
        var json = ComfyWorkflow.Save(_doc, _renderer.MeasureText);
        var name = _tabs.Active?.Path is { } p ? System.IO.Path.GetFileName(p)
            : WorkflowLibrary.PathFor(_tabs.Active?.Name ?? "workflow", _app.Configured && _app.OpensAsApp == true) ?? "workflow.json";
        await Interop.InvokeVoidAsync("spawngenDownload", System.IO.Path.GetFileName(name), json);
        ShowMessage($"Exported {System.IO.Path.GetFileName(name)}.");
    }

    string _renameText = "";
    ElementReference _renameRef, _saveNameRef;
    bool? _focusLibraryInput;         // after the next render: true focuses the Save dialog's name, false the rename box

    void OnWorkflowFilter(ChangeEventArgs e) => _workflowFilter = e.Value?.ToString() ?? "";

    void StartRenameSaved(WorkflowEntry entry)
    {
        _deletingPath = null;
        _renamingPath = entry.Path;
        _renameText = entry.Name;
        _focusLibraryInput = false;
    }

    async Task OnRenameSavedKey(KeyboardEventArgs e, WorkflowEntry entry)
    {
        if (e.Key == "Enter") await CommitRenameAsync(entry);
        else if (e.Key == "Escape") _renamingPath = null;
    }

    /// <summary>Enter or leaving the box renames (once: the box is gone after the first).</summary>
    Task CommitRenameAsync(WorkflowEntry entry) => _renamingPath == entry.Path ? RenameSavedAsync(entry, _renameText) : Task.CompletedTask;

    async Task OnSaveDialogKey(KeyboardEventArgs e)
    {
        if (e.Key == "Enter") await ConfirmSaveAsync();
        else if (e.Key == "Escape") { _saveDialog = null; _studio?.Focus(); }
    }

    static string SavedTitle(WorkflowEntry e) => $"{e.Path}, saved {Ago(e.Modified)}: click to open";

    string EmptyLibraryText() =>
        _workflowFilter.Trim().Length > 0 ? "Nothing matches."
        : _appsTab ? "No apps yet. Create app sets one up from the workflow in front; Save keeps it here."
        : "No saved workflows yet. Save (Ctrl+S) keeps the workflow in front here.";

    static string Ago(long modifiedMs)
    {
        if (modifiedMs <= 0) return "";
        var age = DateTimeOffset.UtcNow - DateTimeOffset.FromUnixTimeMilliseconds(modifiedMs);
        return age.TotalMinutes < 1 ? "just now"
            : age.TotalHours < 1 ? $"{(int)age.TotalMinutes} min ago"
            : age.TotalDays < 1 ? $"{(int)age.TotalHours} h ago"
            : age.TotalDays < 30 ? $"{(int)age.TotalDays} d ago"
            : DateTimeOffset.FromUnixTimeMilliseconds(modifiedMs).ToLocalTime().ToString("yyyy-MM-dd");
    }
}
