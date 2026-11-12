using System.Net;
using System.Text;
using System.Text.Json.Nodes;
using Microsoft.JSInterop;
using SpawnGen.Editor;

namespace SpawnGen.Studio.Run;

/// <summary>What a write or a move did: done, refused because the target exists (and overwrite was off), the source was missing, or it failed.</summary>
public enum StoreResult { Done, Exists, Missing, Failed }

/// <summary>
/// The workflow library (ComfyUI's: the user's workflows/ folder, apps as .app.json). Paths are relative to workflows/ (<c>sub/name.json</c>).
/// On a SpawnGen server it is the server's user folder (ComfyUI's userdata API, shared with ComfyUI clients); a Studio running alone keeps it in
/// this browser (IndexedDB).
/// </summary>
public interface IWorkflowStore
{
    /// <summary>Where the files are, for messages: "on the server", "in this browser".</summary>
    string Where { get; }
    Task<IReadOnlyList<WorkflowEntry>> ListAsync();
    Task<string?> ReadAsync(string path);
    Task<StoreResult> WriteAsync(string path, string text, bool overwrite);
    Task<StoreResult> MoveAsync(string from, string to, bool overwrite);
    Task<StoreResult> DeleteAsync(string path);
    /// <summary>A file of the user's beside the workflows folder (comfy.templates.json ...), or null when there is none.</summary>
    Task<string?> ReadUserFileAsync(string name);
    Task<bool> WriteUserFileAsync(string name, string text);
}

/// <summary>The library on the server that serves the Studio, through ComfyUI's userdata API.</summary>
public sealed class ServerWorkflowStore(HttpClient http) : IWorkflowStore
{
    const string Folder = "workflows/";
    public string Where => "on the server";

    static string FileUrl(string path) => "api/userdata/" + Uri.EscapeDataString(Folder + path);

    public async Task<IReadOnlyList<WorkflowEntry>> ListAsync()
    {
        try
        {
            using var response = await http.GetAsync("api/userdata?dir=workflows&recurse=true&split=false&full_info=true");
            if (response.StatusCode == HttpStatusCode.NotFound || !response.IsSuccessStatusCode) return [];
            var list = JsonNode.Parse(await response.Content.ReadAsStringAsync()) as JsonArray ?? [];
            return list.OfType<JsonObject>()
                .Where(o => ((string?)o["path"])?.EndsWith(WorkflowLibrary.Suffix, StringComparison.OrdinalIgnoreCase) == true)
                .Select(o => new WorkflowEntry((string)o["path"]!, (long?)o["size"] ?? 0, (long?)o["modified"] ?? 0)).ToList();
        }
        catch (Exception ex) when (ex is HttpRequestException or System.Text.Json.JsonException or TaskCanceledException) { return []; }
    }

    public async Task<string?> ReadAsync(string path)
    {
        try
        {
            using var response = await http.GetAsync(FileUrl(path));
            return response.IsSuccessStatusCode ? await response.Content.ReadAsStringAsync() : null;
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException) { return null; }
    }

    public Task<StoreResult> WriteAsync(string path, string text, bool overwrite) =>
        Send(() => http.PostAsync(FileUrl(path) + $"?overwrite={(overwrite ? "true" : "false")}&full_info=true", new StringContent(text, Encoding.UTF8, "application/json")));

    public Task<StoreResult> MoveAsync(string from, string to, bool overwrite) =>
        from == to ? Task.FromResult(StoreResult.Done)
        : Send(() => http.PostAsync(FileUrl(from) + "/move/" + Uri.EscapeDataString(Folder + to) + $"?overwrite={(overwrite ? "true" : "false")}&full_info=true", null));

    public Task<StoreResult> DeleteAsync(string path) => Send(() => http.DeleteAsync(FileUrl(path)));

    public async Task<string?> ReadUserFileAsync(string name)
    {
        try
        {
            // Only a file that is there is read: asking for a missing one would put a 404 in the browser's console.
            using var listing = await http.GetAsync("api/userdata?dir=.");
            if (!listing.IsSuccessStatusCode || JsonNode.Parse(await listing.Content.ReadAsStringAsync()) is not JsonArray files
                || !files.Any(f => (string?)f == name)) return null;
            using var response = await http.GetAsync("api/userdata/" + Uri.EscapeDataString(name));
            return response.IsSuccessStatusCode ? await response.Content.ReadAsStringAsync() : null;
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or System.Text.Json.JsonException) { return null; }
    }

    public async Task<bool> WriteUserFileAsync(string name, string text) =>
        await Send(() => http.PostAsync("api/userdata/" + Uri.EscapeDataString(name) + "?overwrite=true", new StringContent(text, Encoding.UTF8, "application/json"))) == StoreResult.Done;

    static async Task<StoreResult> Send(Func<Task<HttpResponseMessage>> request)
    {
        try
        {
            using var response = await request();
            return response.StatusCode switch
            {
                HttpStatusCode.Conflict => StoreResult.Exists,
                HttpStatusCode.NotFound => StoreResult.Missing,
                _ => response.IsSuccessStatusCode ? StoreResult.Done : StoreResult.Failed,
            };
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException) { return StoreResult.Failed; }
    }
}

/// <summary>The library in this browser (IndexedDB, see <c>spawngenFiles</c> in index.html), for a Studio no server serves.</summary>
public sealed class BrowserWorkflowStore(IJSRuntime js) : IWorkflowStore
{
    public string Where => "in this browser";

    public async Task<IReadOnlyList<WorkflowEntry>> ListAsync()
    {
        try
        {
            var list = JsonNode.Parse(await js.InvokeAsync<string>("spawngenFiles.list")) as JsonArray ?? [];
            return list.OfType<JsonObject>()
                .Select(o => new WorkflowEntry((string)o["path"]!, (long?)o["size"] ?? 0, (long?)(double?)o["modified"] ?? 0)).ToList();
        }
        catch (JSException) { return []; }
    }

    public async Task<string?> ReadAsync(string path)
    {
        try { return await js.InvokeAsync<string?>("spawngenFiles.read", path); }
        catch (JSException) { return null; }
    }

    public Task<StoreResult> WriteAsync(string path, string text, bool overwrite) => Call("spawngenFiles.write", path, text, overwrite);
    public Task<StoreResult> MoveAsync(string from, string to, bool overwrite) => Call("spawngenFiles.move", from, to, overwrite);
    public Task<StoreResult> DeleteAsync(string path) => Call("spawngenFiles.remove", path);

    // The user's other files live in the browser's storage (they are small: node templates).
    public async Task<string?> ReadUserFileAsync(string name)
    {
        try { return await js.InvokeAsync<string?>("spawngenStoreGet", "spawngen.user/" + name); }
        catch (JSException) { return null; }
    }

    public async Task<bool> WriteUserFileAsync(string name, string text)
    {
        try { return await js.InvokeAsync<bool>("spawngenStoreSet", "spawngen.user/" + name, text); }
        catch (JSException) { return false; }
    }

    async Task<StoreResult> Call(string function, params object?[] args)
    {
        try
        {
            return await js.InvokeAsync<string>(function, args) switch
            {
                "done" => StoreResult.Done, "exists" => StoreResult.Exists, "missing" => StoreResult.Missing, _ => StoreResult.Failed,
            };
        }
        catch (JSException) { return StoreResult.Failed; }
    }
}
