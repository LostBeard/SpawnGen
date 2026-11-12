using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.AspNetCore.StaticFiles;

namespace SpawnGen.Server;

/// <summary>
/// The user's files (saved workflows and apps, settings), with ComfyUI's userdata API so ComfyUI clients and the Studio share them:
/// <c>GET /userdata?dir=</c> lists, <c>GET/POST/DELETE /userdata/{file}</c> reads / writes / deletes, <c>POST /userdata/{file}/move/{dest}</c>
/// moves, <c>GET /v2/userdata?path=</c> lists files and folders, <c>/settings</c> and <c>/settings/{id}</c> keep settings in
/// <c>comfy.settings.json</c>, <c>/users</c> reports one user. A file name is one URL segment with its slashes escaped (<c>workflows%2Fa.json</c>).
/// Status codes and JSON shapes follow what ComfyUI's server answers (measured against it): 400 no directory / a folder, 403 a path outside the
/// user's folder, 404 missing, 409 a file that exists when overwrite=false.
/// </summary>
public static class UserData
{
    /// <summary>One user: the files live in <c>{UserPath}/default</c>, as in ComfyUI's single-user layout.</summary>
    public static string Root(ServerOptions options) => Path.Combine(options.UserPath, "default");

    const string SettingsFile = "comfy.settings.json";
    static readonly FileExtensionContentTypeProvider Types = new();

    public static void Map(WebApplication app, ServerOptions options)
    {
        string root = Path.GetFullPath(Root(options));
        Directory.CreateDirectory(Path.Combine(root, "workflows")); // the library's folder exists from the start, so listing it is never a 404

        // The user's files and settings change under the same names (a workflow saved again): a browser must never answer from its cache.
        app.Use((context, next) =>
        {
            var path = context.Request.Path;
            if (path.StartsWithSegments("/userdata") || path.StartsWithSegments("/v2/userdata") || path.StartsWithSegments("/settings") || path.StartsWithSegments("/users"))
                context.Response.Headers.CacheControl = "no-store";
            return next(context);
        });

        app.MapGet("/users", () => Results.Json(new JsonObject { ["storage"] = "server", ["migrated"] = true }));

        app.MapGet("/userdata", (HttpRequest request) =>
        {
            var dir = request.Query["dir"].ToString();
            if (dir.Length == 0) return Results.Text("Directory not provided", statusCode: 400);
            var full = Resolve(root, dir, allowRoot: true);
            if (full is null) return Results.Text("Invalid directory", statusCode: 403);
            if (!Directory.Exists(full)) return Results.Text("Directory not found", statusCode: 404);
            bool recurse = Flag(request, "recurse", false), fullInfo = Flag(request, "full_info", false), split = Flag(request, "split", false);
            var files = Directory.EnumerateFiles(full, "*", recurse ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly)
                .Select(f => (Full: f, Rel: Relative(full, f))).OrderBy(f => f.Rel, StringComparer.Ordinal);
            var list = new JsonArray();
            foreach (var (file, rel) in files)
            {
                if (fullInfo) list.Add(Info(rel, file));
                else if (split) list.Add(new JsonArray([JsonValue.Create(rel), .. rel.Split('/').Select(s => (JsonNode?)JsonValue.Create(s))]));
                else list.Add(rel);
            }
            return Json(list);
        });

        app.MapGet("/v2/userdata", (HttpRequest request) =>
        {
            var path = request.Query["path"].ToString();
            var full = Resolve(root, path, allowRoot: true);
            if (full is null) return Results.Text("Invalid path", statusCode: 403);
            if (!Directory.Exists(full)) return Results.Text("Requested path not found", statusCode: 404);
            var list = new JsonArray();
            foreach (var d in Directory.EnumerateDirectories(full, "*", SearchOption.AllDirectories).OrderBy(Path.GetFileName, StringComparer.Ordinal))
                list.Add(new JsonObject { ["name"] = Path.GetFileName(d), ["path"] = Relative(root, d), ["type"] = "directory" });
            foreach (var f in Directory.EnumerateFiles(full, "*", SearchOption.AllDirectories).OrderBy(Path.GetFileName, StringComparer.Ordinal))
            {
                var info = new FileInfo(f);
                list.Add(new JsonObject
                {
                    ["name"] = info.Name, ["path"] = Relative(root, f), ["type"] = "file", ["size"] = info.Length,
                    ["modified"] = (info.LastWriteTimeUtc - DateTime.UnixEpoch).TotalSeconds,
                });
            }
            return Json(list);
        });

        // One file: GET reads, POST writes, DELETE deletes; POST {file}/move/{dest} moves. The segments are parsed from the raw request target so
        // that an escaped slash stays part of the name (an unescaped one, workflows/a.json, is not a file name: 404).
        app.MapMethods("/userdata/{**rest}", ["GET", "POST", "DELETE"], async (HttpContext context) =>
        {
            var segments = Segments(context);
            var request = context.Request;
            if (segments is [var name])
            {
                var file = Resolve(root, name, allowRoot: false);
                if (file is null) return Results.StatusCode(403);
                switch (request.Method)
                {
                    case "GET":
                        if (!File.Exists(file)) return Results.NotFound();
                        return Results.File(file, Types.TryGetContentType(file, out var type) ? type : "application/octet-stream");
                    case "DELETE":
                        if (!File.Exists(file)) return Results.NotFound();
                        File.Delete(file);
                        return Results.NoContent();
                    default:
                        if (Directory.Exists(file)) return Results.StatusCode(400);
                        if (File.Exists(file) && !Flag(request, "overwrite", true)) return Results.Text("File already exists", statusCode: 409);
                        Directory.CreateDirectory(Path.GetDirectoryName(file)!);
                        await using (var output = File.Create(file)) await request.Body.CopyToAsync(output);
                        return Written(root, file, Flag(request, "full_info", false));
                }
            }
            if (segments is [var from, "move", var to] && request.Method == "POST")
            {
                var source = Resolve(root, from, allowRoot: false);
                if (source is null) return Results.StatusCode(403);
                if (!File.Exists(source)) return Results.NotFound();
                var dest = Resolve(root, to, allowRoot: false);
                if (dest is null) return Results.NotFound();
                if (Directory.Exists(dest)) return Results.StatusCode(400);
                if (File.Exists(dest) && !Flag(request, "overwrite", true)) return Results.Text("File already exists", statusCode: 409);
                Directory.CreateDirectory(Path.GetDirectoryName(dest)!);
                File.Move(source, dest, overwrite: true);
                return Written(root, dest, Flag(request, "full_info", false));
            }
            return Results.NotFound();
        });

        string settingsPath = Path.Combine(root, SettingsFile);
        app.MapGet("/settings", () => Json(ReadSettings(settingsPath)));
        app.MapGet("/settings/{id}", (string id) => Json(ReadSettings(settingsPath)[id]?.DeepClone()));
        app.MapPost("/settings", async (HttpRequest request) =>
        {
            JsonNode? body;
            try { body = await ReadJson(request); }
            catch (System.Text.Json.JsonException) { body = null; }
            if (body is not JsonObject changes) return Results.BadRequest("a JSON object of settings expected");
            var settings = ReadSettings(settingsPath);
            foreach (var (key, value) in changes) settings[key] = value?.DeepClone();
            await File.WriteAllTextAsync(settingsPath, settings.ToJsonString(Indented));
            return Results.Ok();
        });
        app.MapPost("/settings/{id}", async (string id, HttpRequest request) =>
        {
            JsonNode? value;
            try { value = await ReadJson(request); }
            catch (System.Text.Json.JsonException) { return Results.BadRequest("a JSON value expected"); }
            var settings = ReadSettings(settingsPath);
            settings[id] = value;
            await File.WriteAllTextAsync(settingsPath, settings.ToJsonString(Indented));
            return Results.Ok();
        });
    }

    static readonly System.Text.Json.JsonSerializerOptions Indented = new() { WriteIndented = true };

    /// <summary>A path below the user's folder, or null when it leaves it (or names the folder itself, unless <paramref name="allowRoot"/>).</summary>
    internal static string? Resolve(string root, string relative, bool allowRoot)
    {
        if (relative.Length > 0 && (relative[0] is '/' or '\\' || Path.IsPathRooted(relative) || relative.Contains('\0'))) return null;
        string full;
        try { full = Path.GetFullPath(Path.Combine(root, relative)); }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException) { return null; }
        full = full.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        if (string.Equals(full, root.TrimEnd(Path.DirectorySeparatorChar), StringComparison.OrdinalIgnoreCase)) return allowRoot ? full : null;
        return full.StartsWith(root.TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase) ? full : null;
    }

    /// <summary>The unescaped segments after <c>/userdata/</c> (with or without <c>/api</c> in front), from the raw request target.</summary>
    static string[]? Segments(HttpContext context)
    {
        var raw = context.Features.Get<IHttpRequestFeature>()?.RawTarget ?? context.Request.Path.Value ?? "";
        int query = raw.IndexOf('?');
        if (query >= 0) raw = raw[..query];
        if (raw.StartsWith("/api/", StringComparison.Ordinal)) raw = raw[4..];
        const string prefix = "/userdata/";
        if (!raw.StartsWith(prefix, StringComparison.Ordinal)) return null;
        try { return raw[prefix.Length..].Split('/').Select(Uri.UnescapeDataString).ToArray(); }
        catch (UriFormatException) { return null; }
    }

    static string Relative(string from, string path) => Path.GetRelativePath(from, path).Replace('\\', '/');

    static JsonObject Info(string path, string file)
    {
        var info = new FileInfo(file);
        return new JsonObject
        {
            ["path"] = path, ["size"] = info.Length,
            ["modified"] = new DateTimeOffset(info.LastWriteTimeUtc).ToUnixTimeMilliseconds(),
            ["created"] = new DateTimeOffset(info.CreationTimeUtc).ToUnixTimeMilliseconds(),
        };
    }

    /// <summary>What a write or move answers: the file's path in the user's folder, or its full info.</summary>
    static IResult Written(string root, string file, bool fullInfo) =>
        fullInfo ? Json(Info(Relative(root, file), file)) : Json(JsonValue.Create(Relative(root, file)));

    static bool Flag(HttpRequest request, string name, bool otherwise) =>
        request.Query.TryGetValue(name, out var v) ? !string.Equals(v.ToString(), "false", StringComparison.OrdinalIgnoreCase) : otherwise;

    static IResult Json(JsonNode? node) => Results.Content(node?.ToJsonString() ?? "null", "application/json; charset=utf-8");

    static async Task<JsonNode?> ReadJson(HttpRequest request)
    {
        using var reader = new StreamReader(request.Body);
        var text = await reader.ReadToEndAsync();
        return JsonNode.Parse(text);
    }

    static JsonObject ReadSettings(string path)
    {
        try { return File.Exists(path) && JsonNode.Parse(File.ReadAllText(path)) is JsonObject o ? o : []; }
        catch (System.Text.Json.JsonException) { return []; }
    }
}
