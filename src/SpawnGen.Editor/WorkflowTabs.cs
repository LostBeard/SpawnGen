using System.Text.Json.Nodes;

namespace SpawnGen.Editor;

/// <summary>One open workflow: its saved ComfyUI JSON (written when the user switches away or edits) and where its view was.</summary>
public sealed class WorkflowTab
{
    public required string Id { get; init; }
    public required string Name { get; set; }
    /// <summary>ComfyUI workflow JSON; null for a tab nobody has filled yet.</summary>
    public string? Json { get; set; }
    public double Scale { get; set; } = 1;
    public double OffsetX { get; set; }
    public double OffsetY { get; set; }
}

/// <summary>
/// The open workflows as tabs, with the one being edited. Pure data so it can be stored (browser local storage) and restored after a reload
/// without the editor knowing anything about it.
/// </summary>
public sealed class WorkflowTabs
{
    const int Version = 1;
    readonly List<WorkflowTab> _tabs = [];

    public IReadOnlyList<WorkflowTab> Tabs => _tabs;
    public string ActiveId { get; private set; } = "";
    public WorkflowTab? Active => _tabs.FirstOrDefault(t => t.Id == ActiveId);

    /// <summary>Adds a tab named <paramref name="name"/> (made unique) and makes it the active one.</summary>
    public WorkflowTab Add(string name, string? json = null)
    {
        var tab = new WorkflowTab { Id = Guid.NewGuid().ToString("N")[..8], Name = UniqueName(string.IsNullOrWhiteSpace(name) ? "Workflow" : name.Trim()), Json = json };
        _tabs.Add(tab);
        ActiveId = tab.Id;
        return tab;
    }

    public bool Activate(string id)
    {
        if (_tabs.All(t => t.Id != id)) return false;
        ActiveId = id;
        return true;
    }

    /// <summary>Closes a tab; the one to its left (or right, for the first) becomes active when the active one was closed. Returns false when it was the only tab (nothing is closed).</summary>
    public bool Close(string id)
    {
        int index = _tabs.FindIndex(t => t.Id == id);
        if (index < 0 || _tabs.Count == 1) return false;
        _tabs.RemoveAt(index);
        if (ActiveId == id) ActiveId = _tabs[Math.Max(0, index - 1)].Id;
        return true;
    }

    public void Rename(string id, string name)
    {
        var tab = _tabs.FirstOrDefault(t => t.Id == id);
        if (tab is null || string.IsNullOrWhiteSpace(name)) return;
        tab.Name = UniqueName(name.Trim(), except: tab);
    }

    /// <summary>"name", or "name (2)", "name (3)" ... when that is taken by another tab.</summary>
    public string UniqueName(string name, WorkflowTab? except = null)
    {
        bool Taken(string n) => _tabs.Any(t => !ReferenceEquals(t, except) && t.Name == n);
        if (!Taken(name)) return name;
        for (int i = 2; ; i++)
            if (!Taken($"{name} ({i})")) return $"{name} ({i})";
    }

    public string ToJson()
    {
        var tabs = new JsonArray();
        foreach (var t in _tabs)
            tabs.Add(new JsonObject { ["id"] = t.Id, ["name"] = t.Name, ["json"] = t.Json, ["scale"] = t.Scale, ["x"] = t.OffsetX, ["y"] = t.OffsetY });
        return new JsonObject { ["version"] = Version, ["active"] = ActiveId, ["tabs"] = tabs }.ToJsonString();
    }

    /// <summary>Restores what <see cref="ToJson"/> wrote; null when the text is missing, damaged or from an unknown version (the caller starts fresh).</summary>
    public static WorkflowTabs? FromJson(string? text)
    {
        if (string.IsNullOrWhiteSpace(text)) return null;
        try
        {
            if (JsonNode.Parse(text) is not JsonObject root || (int?)root["version"] != Version || root["tabs"] is not JsonArray list) return null;
            var result = new WorkflowTabs();
            foreach (var o in list.OfType<JsonObject>())
            {
                if ((string?)o["id"] is not { Length: > 0 } id || result._tabs.Any(t => t.Id == id)) continue;
                result._tabs.Add(new WorkflowTab
                {
                    Id = id, Name = (string?)o["name"] ?? "Workflow", Json = (string?)o["json"],
                    Scale = Math.Clamp((double?)o["scale"] ?? 1, Viewport.MinScale, Viewport.MaxScale), OffsetX = (double?)o["x"] ?? 0, OffsetY = (double?)o["y"] ?? 0,
                });
            }
            if (result._tabs.Count == 0) return null;
            result.ActiveId = result._tabs.Any(t => t.Id == (string?)root["active"]) ? (string)root["active"]! : result._tabs[0].Id;
            return result;
        }
        catch (Exception ex) when (ex is System.Text.Json.JsonException or InvalidOperationException or FormatException) { return null; }
    }
}
