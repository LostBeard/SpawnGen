using System.Text.Json;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// The checkpoints Load Checkpoint offers: the built-in ones, plus whatever the user adds on the machine that runs the model:
/// <list type="bullet">
/// <item>single-file checkpoints (<c>.safetensors</c>: SD 1.x, SD 2.x, SDXL and their fine-tunes) in <c>&lt;models&gt;/checkpoints</c>, sub-folders included,
/// listed by their path there as ComfyUI lists its models/checkpoints;</item>
/// <item>a <c>checkpoints.json</c> in the models folder: <c>{ "my_sd15": { "repo": "owner/name", "textWidth": 768, "unetData": "unet/model.onnx_data" } }</c>
/// (any Optimum-style ONNX export of Stable Diffusion on Hugging Face: <c>text_encoder</c>, <c>unet</c>, <c>vae_decoder</c>, <c>tokenizer</c> folders; it downloads on first use);</item>
/// <item>a folder in the models folder with that layout (and an optional <c>checkpoint.json</c> with the same two settings).</item>
/// </list>
/// </summary>
public static class CheckpointCatalog
{
    /// <summary><paramref name="Folder"/>: the models folder's sub-folder that holds the repo's files when they are local (null = the checkpoint's name).</summary>
    sealed record Entry(string Repo, MlRuntime.CheckpointInfo Info, string? Folder = null);

    /// <summary>The repo id the files of <c>&lt;models&gt;/checkpoints</c> are opened under.</summary>
    public const string LocalCheckpointsRepo = "local/checkpoints";
    /// <summary>The repo id the files of <c>&lt;models&gt;/loras</c> are opened under.</summary>
    public const string LocalLorasRepo = "local/loras";

    static readonly SortedSet<string> LoraFiles = new(StringComparer.Ordinal);

    /// <summary>The LoRA files Load LoRA offers (paths under <c>&lt;models&gt;/loras</c>, as ComfyUI lists them).</summary>
    public static IReadOnlyList<string> Loras { get { lock (Gate) return [.. LoraFiles]; } }

    /// <summary>Adds every <c>.safetensors</c> file under <c>&lt;models&gt;/loras</c> (sub-folders too). Returns how many were added.</summary>
    public static int ScanLoraFiles(string modelsFolder)
    {
        var dir = Path.Combine(modelsFolder, "loras");
        if (!Directory.Exists(dir)) return 0;
        int added = 0;
        foreach (var path in Directory.EnumerateFiles(dir, "*.safetensors", SearchOption.AllDirectories))
            lock (Gate) if (LoraFiles.Add(Path.GetRelativePath(dir, path))) added++;
        return added;
    }

    static readonly object Gate = new();
    static readonly Dictionary<string, Entry> Entries = new(StringComparer.Ordinal)
    {
        ["sd_turbo"] = new("schmuell/sd-turbo-ort-web", new(1024)),
        ["sd_v1-5"] = new("a420/stable-diffusion-v1-5_onnx", new(768, "unet/model.onnx_data")),
    };

    /// <summary>Adds a checkpoint (or replaces one of the same name). Returns false for a name or repo that is empty.</summary>
    public static bool Register(string name, string repo, int textWidth = 1024, string? unetExternalData = null)
    {
        if (string.IsNullOrWhiteSpace(name) || string.IsNullOrWhiteSpace(repo) || textWidth is < 64 or > 8192) return false;
        lock (Gate) Entries[name] = new Entry(repo, new MlRuntime.CheckpointInfo(textWidth, string.IsNullOrWhiteSpace(unetExternalData) ? null : unetExternalData));
        return true;
    }

    public static IReadOnlyDictionary<string, string> Repos { get { lock (Gate) return Entries.ToDictionary(e => e.Key, e => e.Value.Repo); } }
    public static IReadOnlyDictionary<string, MlRuntime.CheckpointInfo> Details { get { lock (Gate) return Entries.ToDictionary(e => e.Key, e => e.Value.Info); } }

    /// <summary>Repo id to checkpoint name: a models folder with a sub-folder of that name is read from disk instead of downloaded.</summary>
    public static IReadOnlyDictionary<string, string> RepoFolders
    {
        get
        {
            lock (Gate)
            {
                var folders = Entries.GroupBy(e => e.Value.Repo).ToDictionary(g => g.Key, g => g.First().Value.Folder ?? g.First().Key);
                folders[LocalLorasRepo] = "loras";
                return folders;
            }
        }
    }

    /// <summary>
    /// Adds every <c>.safetensors</c> file under <c>&lt;models&gt;/checkpoints</c> (sub-folders too), named by its path there with the platform's separator, as
    /// ComfyUI names them. Returns how many were added.
    /// </summary>
    public static int ScanCheckpointFiles(string modelsFolder)
    {
        var dir = Path.Combine(modelsFolder, "checkpoints");
        if (!Directory.Exists(dir)) return 0;
        int added = 0;
        foreach (var path in Directory.EnumerateFiles(dir, "*.safetensors", SearchOption.AllDirectories))
        {
            var name = Path.GetRelativePath(dir, path);
            if (Repos.ContainsKey(name)) continue;
            lock (Gate) Entries[name] = new Entry(LocalCheckpointsRepo, new MlRuntime.CheckpointInfo(0, File: name), "checkpoints");
            added++;
        }
        return added;
    }

    /// <summary>Reads a <c>checkpoints.json</c>; entries that are not understood are skipped. Returns how many were added.</summary>
    public static int LoadJson(string path)
    {
        if (!File.Exists(path)) return 0;
        int added = 0;
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            if (doc.RootElement.ValueKind != JsonValueKind.Object) return 0;
            foreach (var item in doc.RootElement.EnumerateObject())
            {
                if (item.Value.ValueKind != JsonValueKind.Object || !item.Value.TryGetProperty("repo", out var repo) || repo.ValueKind != JsonValueKind.String) continue;
                int width = item.Value.TryGetProperty("textWidth", out var w) && w.TryGetInt32(out var wi) ? wi : 1024;
                string? data = item.Value.TryGetProperty("unetData", out var d) && d.ValueKind == JsonValueKind.String ? d.GetString() : null;
                if (Register(item.Name, repo.GetString()!, width, data)) added++;
            }
        }
        catch (JsonException) { }
        return added;
    }

    /// <summary>Adds every folder of the models folder that has the ONNX Stable Diffusion layout and is not known yet. Returns how many were added.</summary>
    public static int ScanFolder(string modelsFolder)
    {
        if (!Directory.Exists(modelsFolder)) return 0;
        int added = 0;
        foreach (var dir in Directory.EnumerateDirectories(modelsFolder))
        {
            var name = Path.GetFileName(dir);
            if (name is "hub" || Repos.ContainsKey(name)) continue;
            bool layout = File.Exists(Path.Combine(dir, "unet", "model.onnx")) && File.Exists(Path.Combine(dir, "text_encoder", "model.onnx"))
                && File.Exists(Path.Combine(dir, "vae_decoder", "model.onnx")) && File.Exists(Path.Combine(dir, "tokenizer", "vocab.json"));
            if (!layout) continue;
            int width = 1024;
            string? data = File.Exists(Path.Combine(dir, "unet", "model.onnx_data")) ? "unet/model.onnx_data" : null;
            var settings = Path.Combine(dir, "checkpoint.json");
            if (File.Exists(settings))
            {
                try
                {
                    using var doc = JsonDocument.Parse(File.ReadAllText(settings));
                    if (doc.RootElement.TryGetProperty("textWidth", out var w) && w.TryGetInt32(out var wi)) width = wi;
                    if (doc.RootElement.TryGetProperty("unetData", out var d) && d.ValueKind == JsonValueKind.String) data = d.GetString();
                }
                catch (JsonException) { }
            }
            if (Register(name, "local/" + name, width, data)) added++;
        }
        return added;
    }
}
