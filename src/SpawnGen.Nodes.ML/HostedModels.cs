using System.Text.Json;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// The model files ComfyUI's templates name, with the Hugging Face repo and path ComfyUI fetches each one from (wwwroot/model-catalog.json,
/// written by tools/templates-manifest from the templates' properties.models). Registered on both engines, so the loaders list them and the
/// model source fetches them on first use: the desktop into its download cache, the browser into its storage or by streaming
/// (StreamingModelSource; Hugging Face allows cross-origin reads). A file of the same name in the models folder is used instead.
/// </summary>
public static class HostedModels
{
    /// <summary>One hosted file: its Hugging Face repo, its path there, and the ComfyUI models sub-folder it belongs in.</summary>
    public sealed record Entry(string Name, string Repo, string Path, string Folder);

    /// <summary>Reads model-catalog.json ({"models": {name: {"repo", "path", "folder"}}}) and registers every entry; returns how many a loader now lists.</summary>
    public static int LoadJson(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var entries = new List<Entry>();
        foreach (var m in doc.RootElement.GetProperty("models").EnumerateObject())
            if (m.Value.TryGetProperty("repo", out var repo) && m.Value.TryGetProperty("path", out var path) && m.Value.TryGetProperty("folder", out var folder)
                && repo.GetString() is { Length: > 0 } r && path.GetString() is { Length: > 0 } p && folder.GetString() is { Length: > 0 } f)
                entries.Add(new Entry(m.Name, r, p, f));
        return Register(entries);
    }

    /// <summary>Registers each entry with the loader of its folder (ComfyUI's folder names and their older aliases); others are skipped.</summary>
    public static int Register(IEnumerable<Entry> entries)
    {
        int listed = 0;
        foreach (var e in entries)
        {
            bool added = e.Folder switch
            {
                "checkpoints" => CheckpointCatalog.AddHostedCheckpoint(e.Name, e.Repo, e.Path),
                "loras" => CheckpointCatalog.AddHostedLora(e.Name, e.Repo, e.Path),
                _ => ModelFolders.KindOf(e.Folder) is { } kind && ModelFolders.AddHosted(kind, e.Name, e.Repo, e.Path),
            };
            if (added) listed++;
        }
        return listed;
    }
}
