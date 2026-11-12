using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Forms;
using Microsoft.AspNetCore.Components.Web;
using Microsoft.JSInterop;
using SpawnGen.Editor;

namespace SpawnGen.Studio.Pages;

// Node templates, as ComfyUI's: the canvas menu's Save Selected as Template keeps the selected nodes (with the links among them) under a name in the
// user's comfy.templates.json (on the server, shared with ComfyUI; in this browser when the Studio runs alone); Node Templates inserts one where the
// menu was opened; Manage renames, deletes, exports and imports them.
public partial class Studio
{
    List<NodeTemplate> _nodeTemplates = [];
    bool _nodeTemplatesLoaded;

    /// <summary>The name dialog: saving the selection as a template (Ids), asking once more when the name is taken.</summary>
    sealed class TemplateDialog
    {
        public string Name = "";
        public required int[] Ids;
        public bool Replace;
    }
    TemplateDialog? _templateDialog;
    bool _templatesManageOpen;
    ElementReference _templateNameRef;
    bool _focusTemplateName;

    async Task LoadNodeTemplatesAsync()
    {
        _nodeTemplates = NodeTemplates.Parse(await Store.ReadUserFileAsync(NodeTemplates.FileName));
        _nodeTemplatesLoaded = true;
        StateHasChanged();
    }

    async Task<bool> SaveNodeTemplatesAsync()
    {
        bool ok = await Store.WriteUserFileAsync(NodeTemplates.FileName, NodeTemplates.Serialize(_nodeTemplates));
        if (!ok) ShowMessage($"Could not save the node templates {Store.Where}.");
        StateHasChanged();
        return ok;
    }

    /// <summary>The canvas menu's template entries: Save Selected as Template (with a selection) and the Node Templates list.</summary>
    void AddTemplateMenu(List<MenuItem> items, Point world)
    {
        if (_view.SelectedNodes.Count > 0)
        {
            var ids = _view.SelectedNodes.ToArray();
            items.Add(new("Save Selected as Template", () =>
            {
                _templateDialog = new TemplateDialog { Ids = ids, Name = "" };
                _focusTemplateName = true;
            }));
        }
        var entries = _nodeTemplates.Select(t => new MenuItem(t.Name, () => InsertTemplate(t, world))).ToList();
        entries.Add(new("Manage...", () => { _templatesManageOpen = true; if (!_nodeTemplatesLoaded) _ = LoadNodeTemplatesAsync(); }));
        items.Add(SubMenu("Node Templates", entries));
    }

    void InsertTemplate(NodeTemplate template, Point world)
    {
        try
        {
            var part = NodeTemplates.Open(template.Data, Registry, out var warnings);
            _view.PasteNodes([.. part.Graph.Nodes], part.Graph.Links, world);
            ShowMessage(warnings.Count == 0 ? $"Inserted {template.Name}." : $"Inserted {template.Name} with {warnings.Count} warning(s): {warnings[0]}");
        }
        catch (Exception ex) when (ex is InvalidDataException or System.Text.Json.JsonException)
        {
            ShowMessage($"{template.Name} cannot be inserted: {ex.Message}");
        }
    }

    async Task ConfirmTemplateAsync()
    {
        if (_templateDialog is not { } d) return;
        var name = d.Name.Trim();
        if (name.Length == 0) { ShowMessage("Type a name for the template."); return; }
        if (!_nodeTemplatesLoaded) await LoadNodeTemplatesAsync();
        int existing = _nodeTemplates.FindIndex(t => t.Name == name);
        if (existing >= 0 && !d.Replace) { d.Replace = true; StateHasChanged(); return; } // asks: replace it?
        var template = new NodeTemplate(name, NodeTemplates.Build(_doc, d.Ids, _renderer.MeasureText));
        if (existing >= 0) _nodeTemplates[existing] = template; else _nodeTemplates.Add(template);
        _templateDialog = null;
        if (await SaveNodeTemplatesAsync()) ShowMessage($"Saved the template {name} {Store.Where}.");
        _studio?.Focus();
    }

    async Task OnTemplateNameKey(KeyboardEventArgs e)
    {
        if (e.Key == "Enter") await ConfirmTemplateAsync();
        else if (e.Key == "Escape") { _templateDialog = null; _studio?.Focus(); }
    }

    void OnTemplateNameInput(ChangeEventArgs e)
    {
        if (_templateDialog is { } d) { d.Name = e.Value?.ToString() ?? ""; d.Replace = false; }
    }

    async Task RenameTemplateAsync(int index, string? name)
    {
        name = name?.Trim();
        if (string.IsNullOrEmpty(name) || index >= _nodeTemplates.Count || _nodeTemplates[index].Name == name) return;
        if (_nodeTemplates.Any(t => t.Name == name)) { ShowMessage($"There is already a template named {name}."); StateHasChanged(); return; }
        _nodeTemplates[index] = _nodeTemplates[index] with { Name = name };
        await SaveNodeTemplatesAsync();
    }

    async Task DeleteTemplateAsync(int index)
    {
        if (index >= _nodeTemplates.Count) return;
        var name = _nodeTemplates[index].Name;
        _nodeTemplates.RemoveAt(index);
        if (await SaveNodeTemplatesAsync()) ShowMessage($"Deleted the template {name}.");
    }

    /// <summary>Export: every template as a comfy.templates.json file (what ComfyUI reads too).</summary>
    async Task ExportTemplatesAsync() =>
        await Interop.InvokeVoidAsync("spawngenDownload", NodeTemplates.FileName, NodeTemplates.Serialize(_nodeTemplates));

    /// <summary>Import: the templates of a templates file are added (a name that is taken gets a number).</summary>
    async Task OnTemplatesImportAsync(InputFileChangeEventArgs e)
    {
        try
        {
            await using var stream = e.File.OpenReadStream(16 * 1024 * 1024);
            using var reader = new StreamReader(stream);
            var added = NodeTemplates.Parse(await reader.ReadToEndAsync());
            if (added.Count == 0) { ShowMessage($"{e.File.Name} has no node templates."); return; }
            foreach (var t in added)
            {
                var name = t.Name;
                for (int i = 2; _nodeTemplates.Any(x => x.Name == name); i++) name = $"{t.Name} ({i})";
                _nodeTemplates.Add(t with { Name = name });
            }
            if (await SaveNodeTemplatesAsync()) ShowMessage($"Imported {added.Count} template(s) from {e.File.Name}.");
        }
        catch (IOException ex) { ShowMessage($"Could not read {e.File.Name}: {ex.Message}"); }
        finally { _fileInputs++; }
    }
}
