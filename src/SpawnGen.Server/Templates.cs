using System.Text.RegularExpressions;

namespace SpawnGen.Server;

/// <summary>
/// ComfyUI's workflow templates at <c>/templates/&lt;file&gt;</c>, where ComfyUI's own server serves them (measured 2026-11-14: index.json, each
/// template's .json and its -1.webp thumbnail). The templates are Comfy-Org's (MIT, github.com/Comfy-Org/workflow_templates): a file is fetched
/// once from <see cref="ServerOptions.TemplatesSource"/> and kept in <see cref="ServerOptions.TemplatesPath"/>; the index is fetched again when it
/// is a day old, so new templates appear. Only plain file names are served.
/// </summary>
public static partial class Templates
{
    [GeneratedRegex("^[A-Za-z0-9][A-Za-z0-9._-]*$")]
    private static partial Regex FileName();

    static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(60) };

    /// <summary>A name the endpoint serves: letters, digits, '.', '_' and '-', not starting with a dot (no folders, no "..").</summary>
    public static bool IsServable(string file) => FileName().IsMatch(file) && !file.Contains("..", StringComparison.Ordinal);

    public static string MediaType(string file) => Path.GetExtension(file).ToLowerInvariant() switch
    {
        ".json" => "application/json",
        ".webp" => "image/webp",
        ".png" => "image/png",
        ".jpg" or ".jpeg" => "image/jpeg",
        ".mp4" => "video/mp4",
        ".webm" => "video/webm",
        ".mp3" => "audio/mpeg",
        ".flac" => "audio/flac",
        _ => "application/octet-stream",
    };

    public static void Map(WebApplication app, ServerOptions options)
    {
        app.MapGet("/templates/{file}", async (string file, CancellationToken ct) =>
        {
            if (!IsServable(file)) return Results.NotFound();
            var path = Path.Combine(options.TemplatesPath, file);
            bool stale = !File.Exists(path) || (file.StartsWith("index", StringComparison.Ordinal) && File.GetLastWriteTimeUtc(path) < DateTime.UtcNow.AddDays(-1));
            if (stale && !string.IsNullOrEmpty(options.TemplatesSource))
            {
                try
                {
                    using var response = await Http.GetAsync(options.TemplatesSource.TrimEnd('/') + "/" + file, ct);
                    if (response.IsSuccessStatusCode)
                    {
                        Directory.CreateDirectory(options.TemplatesPath);
                        var bytes = await response.Content.ReadAsByteArrayAsync(ct);
                        var temp = path + ".part";
                        await File.WriteAllBytesAsync(temp, bytes, ct);
                        File.Move(temp, path, overwrite: true);
                    }
                    else if (!File.Exists(path)) return Results.NotFound();
                }
                catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or IOException)
                {
                    // Offline: a copy kept earlier still serves.
                    if (!File.Exists(path)) return Results.Problem($"The template file could not be fetched: {ex.Message}", statusCode: 502);
                }
            }
            return File.Exists(path) ? Results.File(path, MediaType(file)) : Results.NotFound();
        });
    }
}
