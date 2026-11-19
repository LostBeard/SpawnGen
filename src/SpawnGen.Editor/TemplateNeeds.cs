using System.Text.Json;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor;

/// <summary>What one of ComfyUI's templates needs: its node types and the model files it names (wwwroot/template-needs.json, from
/// tools/templates-manifest/make_manifest.py).</summary>
public sealed record TemplateNeed(string[] Nodes, string[] Models);

/// <summary>
/// How a template stands on an engine: the nodes it lacks, the model files its loaders do not list, and the model files SpawnGen has not run
/// yet (no measured run of that model: verified-models.json). A template whose models are listed but not verified opens and may fail on Queue.
/// </summary>
public sealed record TemplateStatus(string[] MissingNodes, string[] MissingModels, string[] UnverifiedModels)
{
    public TemplateStatus(string[] missingNodes, string[] missingModels) : this(missingNodes, missingModels, []) { }

    /// <summary>Every node is there: the template opens without "(missing)" nodes.</summary>
    public bool Opens => MissingNodes.Length == 0;

    /// <summary>Every node is there and every model file is listed and has run in SpawnGen: Queue runs it (it may still be too large for the device).</summary>
    public bool Ready => Opens && MissingModels.Length == 0 && UnverifiedModels.Length == 0;
}

/// <summary>
/// The template browser's check: each template's needs (see <see cref="TemplateNeed"/>) against the engine's node registry and the files its
/// loaders list (the combo options of every node; a file in a sub-folder counts by its name).
/// </summary>
public sealed class TemplateNeeds
{
    /// <summary>The templates' repository commit the needs were read from (the Studio and the server open the templates of the same commit).</summary>
    public string Commit { get; }

    readonly Dictionary<string, TemplateNeed> _needs;

    /// <summary>The model files SpawnGen has run or measured (null: the list predates them, every listed file counts).</summary>
    readonly HashSet<string>? _verified;

    TemplateNeeds(string commit, Dictionary<string, TemplateNeed> needs, HashSet<string>? verified) { Commit = commit; _needs = needs; _verified = verified; }

    /// <summary>Reads template-needs.json: {"commit": ..., "templates": {name: {"nodes": [...], "models": [...]}}}.</summary>
    public static TemplateNeeds Parse(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        var needs = new Dictionary<string, TemplateNeed>(StringComparer.Ordinal);
        foreach (var t in root.GetProperty("templates").EnumerateObject())
            needs[t.Name] = new TemplateNeed(Strings(t.Value, "nodes"), Strings(t.Value, "models"));
        HashSet<string>? verified = root.TryGetProperty("verified_models", out var v)
            ? new HashSet<string>(v.EnumerateArray().Select(x => x.GetString() ?? ""), StringComparer.OrdinalIgnoreCase) : null;
        return new TemplateNeeds(root.GetProperty("commit").GetString() ?? "", needs, verified);

        static string[] Strings(JsonElement e, string name) =>
            e.TryGetProperty(name, out var a) ? [.. a.EnumerateArray().Select(x => x.GetString() ?? "").Where(x => x.Length > 0)] : [];
    }

    public bool TryGet(string template, out TemplateNeed need) => _needs.TryGetValue(template, out need!);

    /// <summary>The file names (sub-folders dropped) every combo of the registry offers: the model files this engine can load.</summary>
    public static HashSet<string> OfferedFiles(NodeRegistry registry)
    {
        var files = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var schema in registry.Schemas)
            foreach (var input in schema.Inputs)
                if (input.Widget is { Kind: WidgetKind.Combo } w)
                    foreach (var option in w.ResolveOptions())
                        files.Add(FileName(option));
        return files;
    }

    /// <summary>The template's status on the engine (null for a template the list does not know: a newer one than its commit).</summary>
    public TemplateStatus? Check(string template, NodeRegistry registry, IReadOnlySet<string> offeredFiles)
    {
        if (!_needs.TryGetValue(template, out var need)) return null;
        var nodes = need.Nodes.Where(n => registry.GetSchema(n) is null).ToArray();
        var models = need.Models.Where(m => !offeredFiles.Contains(FileName(m))).ToArray();
        var unverified = _verified is null ? [] : need.Models.Where(m => offeredFiles.Contains(FileName(m)) && !_verified.Contains(FileName(m))).ToArray();
        return new TemplateStatus(nodes, models, unverified);
    }

    static string FileName(string path)
    {
        int cut = Math.Max(path.LastIndexOf('/'), path.LastIndexOf('\\'));
        return cut < 0 ? path : path[(cut + 1)..];
    }
}
