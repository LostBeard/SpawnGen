namespace SpawnGen.Editor;

/// <summary>A saved workflow in the library: its path under workflows/ (<c>sub/name.json</c>), size in bytes and when it was last changed (ms since 1970).</summary>
public sealed record WorkflowEntry(string Path, long Size, long Modified)
{
    /// <summary>A ComfyUI app file (<c>name.app.json</c>): listed under Apps, opens as the app.</summary>
    public bool IsApp => WorkflowLibrary.IsApp(Path);
    public string Name => WorkflowLibrary.DisplayName(Path);
}

/// <summary>
/// Names in the workflow library, which is ComfyUI's: files under the user's <c>workflows/</c> folder, sub-folders allowed, apps saved as
/// <c>.app.json</c>. Pure string rules, so the server store and the browser store name things the same way.
/// </summary>
public static class WorkflowLibrary
{
    public const string Suffix = ".json";
    public const string AppSuffix = ".app.json";

    public static bool IsApp(string path) => path.EndsWith(AppSuffix, StringComparison.OrdinalIgnoreCase);

    /// <summary>The name shown for a file: its path without <c>.app.json</c> / <c>.json</c> (<c>sub/name</c>).</summary>
    public static string DisplayName(string path) =>
        IsApp(path) ? path[..^AppSuffix.Length]
        : path.EndsWith(Suffix, StringComparison.OrdinalIgnoreCase) ? path[..^Suffix.Length]
        : path;

    /// <summary>
    /// The library path for a name the user typed: folders kept (<c>a/b</c>), characters a file name cannot hold replaced by <c>_</c>, empty and
    /// <c>.</c> / <c>..</c> parts dropped, a typed <c>.json</c> / <c>.app.json</c> replaced by the right suffix. Null when nothing is left.
    /// </summary>
    public static string? PathFor(string name, bool app)
    {
        var trimmed = DisplayName(name.Trim().Replace('\\', '/'));
        var parts = trimmed.Split('/')
            .Select(p => new string(p.Trim().Select(c => c < ' ' || "<>:\"|?*".Contains(c) ? '_' : c).ToArray()).TrimEnd('.', ' '))
            .Where(p => p.Length > 0 && p != "." && p != "..")
            .ToArray();
        return parts.Length == 0 ? null : string.Join('/', parts) + (app ? AppSuffix : Suffix);
    }

    /// <summary>A file renamed to <paramref name="name"/>, kept in its kind (an app stays an app); null when the name is empty.</summary>
    public static string? Renamed(string path, string name) => PathFor(name, IsApp(path));

    /// <summary>A file matches a search when its shown name holds every word of it (any case).</summary>
    public static bool Matches(WorkflowEntry entry, string? search) =>
        string.IsNullOrWhiteSpace(search)
        || search.Split(' ', StringSplitOptions.RemoveEmptyEntries).All(w => entry.Name.Contains(w, StringComparison.OrdinalIgnoreCase));

    /// <summary>The library in the order it is listed: by shown name, any case.</summary>
    public static IEnumerable<WorkflowEntry> Sorted(IEnumerable<WorkflowEntry> entries) =>
        entries.OrderBy(e => e.Name, StringComparer.OrdinalIgnoreCase).ThenBy(e => e.Path, StringComparer.Ordinal);
}
