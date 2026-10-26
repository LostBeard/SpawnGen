using System.Text.Json;
using System.Text.Json.Nodes;
using SpawnGen.Engine.Execution;
using SpawnGen.Runtime;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Server;

/// <summary>Builds the server: ComfyUI-shaped endpoints (<c>/prompt</c>, <c>/queue</c>, <c>/history</c>, <c>/object_info</c>, <c>/view</c>, <c>/upload/image</c>, <c>/ws</c>) plus the Studio's static files.</summary>
public static class ServerApp
{
    public static WebApplication Build(string[] args, Action<WebApplicationBuilder>? configure = null)
    {
        var builder = WebApplication.CreateBuilder(args);
        // Serve the Studio's files straight from the build output when running from a checkout (`dotnet run`). A published server carries them in
        // wwwroot instead and there is no manifest, so this does nothing there.
        builder.WebHost.UseStaticWebAssets();
        var options = ServerOptions.From(builder.Configuration);
        foreach (var dir in new[] { options.ModelsPath, options.OutputPath, options.InputPath, options.TempPath, options.CustomNodesPath }) Directory.CreateDirectory(dir);

        CheckpointCatalog.LoadJson(Path.Combine(options.ModelsPath, "checkpoints.json"));
        CheckpointCatalog.ScanFolder(options.ModelsPath);
        CheckpointCatalog.ScanCheckpointFiles(options.ModelsPath);
        CheckpointCatalog.ScanLoraFiles(options.ModelsPath);
        ModelFolders.Scan(options.ModelsPath);
        builder.Services.AddSingleton(options);
        builder.Services.AddSingleton(new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly));
        builder.Services.AddSingleton(new OutputOptions { Directory = options.OutputPath, TempDirectory = options.TempPath });
        InputFiles.Directory = options.InputPath;
        // The GPU is only touched when the first ML node runs (resolving this singleton opens the accelerator).
        builder.Services.AddSingleton(_ => new DesktopModelSource(options.ModelsPath, MlRuntime.RepoFolders));
        builder.Services.AddSingleton(sp => new MlRuntime(sp.GetRequiredService<DesktopModelSource>()));
        builder.Services.AddSingleton<EventHub>();
        builder.Services.AddSingleton(sp => new PromptQueue(sp.GetRequiredService<NodeRegistry>(), sp, sp.GetRequiredService<EventHub>()));
        configure?.Invoke(builder);

        var app = builder.Build();
        app.UseWebSockets();
        var customNodes = CustomNodes.Load(app.Services.GetRequiredService<NodeRegistry>(), options.CustomNodesPath);
        var log = app.Services.GetRequiredService<ILoggerFactory>().CreateLogger("SpawnGen.CustomNodes");
        foreach (var pack in customNodes)
        {
            if (pack.Error is not null) log.LogWarning("Custom node pack {File} was not loaded: {Error}", pack.File, pack.Error);
            else log.LogInformation("Custom node pack {File}: {Nodes}", pack.File, string.Join(", ", pack.Nodes));
        }
        app.Services.GetRequiredService<ServerOptions>().LoadedCustomNodes = customNodes;
        MapApi(app);

        // The Studio (Blazor WebAssembly) is served from the same origin so it needs no CORS and no configuration.
        app.UseBlazorFrameworkFiles();
        app.UseStaticFiles();
        app.MapFallbackToFile("index.html");
        return app;
    }

    static void MapApi(WebApplication app)
    {
        var queue = app.Services.GetRequiredService<PromptQueue>();
        var events = app.Services.GetRequiredService<EventHub>();
        var registry = app.Services.GetRequiredService<NodeRegistry>();
        var options = app.Services.GetRequiredService<ServerOptions>();

        app.MapGet("/object_info", () => Results.Content(ObjectInfo(registry).ToJsonString(), "application/json"));

        app.MapGet("/object_info/{node}", (string node) =>
            registry.GetSchema(node) is not null && ObjectInfo(registry)[node] is { } entry ? Results.Json(new JsonObject { [node] = entry.DeepClone() }) : Results.Json(new JsonObject()));

        // Small endpoints other ComfyUI clients poll: what models exist (by folder) and that there are no embeddings or server-side extensions.
        app.MapGet("/models", () => Results.Json(new[] { "checkpoints", "upscale_models", "vae", "loras", "controlnet", "embeddings" }));
        app.MapGet("/models/{folder}", (string folder) => Results.Json(folder switch
        {
            "checkpoints" => SpawnGen.Nodes.ML.CheckpointCatalog.Repos.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray(),
            "upscale_models" => SpawnGen.Nodes.ML.UpscaleModelLoader.Models(),
            _ => Array.Empty<string>(),
        }));
        // The model manager: what downloaded models take on disk, what is downloading now, and removing files.
        app.MapGet("/models/cache", async () =>
        {
            var source = app.Services.GetRequiredService<DesktopModelSource>();
            var files = (await source.Remote.Store.ListAsync()).Select(e => new SpawnGen.Runtime.CachedModelFile(e.Key, e.SizeBytes, e.Complete)).OrderByDescending(f => f.Size).ToList();
            var downloads = source.Remote.ActiveDownloads.Select(d => new SpawnGen.Runtime.ModelDownload(d.Key, d.BytesReceived, d.TotalBytes, d.BytesPerSecond)).ToList();
            return Results.Json(new SpawnGen.Runtime.ModelCacheInfo(files, files.Sum(f => f.Size), downloads));
        });
        app.MapDelete("/models/cache/{key}", async (string key) =>
        {
            await app.Services.GetRequiredService<DesktopModelSource>().Remote.Store.RemoveAsync(key);
            return Results.Ok();
        });
        app.MapPost("/models/cache/clear", async () =>
        {
            await app.Services.GetRequiredService<DesktopModelSource>().Remote.Store.ClearAsync();
            return Results.Ok();
        });
        app.MapGet("/embeddings", () => Results.Json(Array.Empty<string>()));
        app.MapGet("/extensions", () => Results.Json(Array.Empty<string>()));
        app.MapPost("/free", () => Results.Ok()); // models stay loaded: this server frees them when it stops
        app.MapGet("/system_stats", () => Results.Json(new
        {
            system = new { os = Environment.OSVersion.Platform.ToString(), spawngen_version = typeof(ServerApp).Assembly.GetName().Version?.ToString() ?? "0" },
            devices = Array.Empty<object>(),
            custom_nodes = options.LoadedCustomNodes.Select(p => new { file = Path.GetFileName(p.File), nodes = p.Nodes, error = p.Error }),
        }));

        app.MapPost("/prompt", async (HttpRequest request) =>
        {
            JsonObject? body;
            try { body = await JsonNode.ParseAsync(request.Body) as JsonObject; }
            catch (JsonException ex) { return Results.Json(Error("invalid_json", ex.Message), statusCode: 400); }
            if (body?["prompt"] is not JsonObject promptJson)
                return Results.Json(Error("no_prompt", "No prompt provided"), statusCode: 400);

            PromptGraph graph;
            try { graph = PromptJson.Parse(promptJson.ToJsonString()); }
            catch (Exception ex) when (ex is JsonException or InvalidDataException or FormatException)
            {
                return Results.Json(Error("invalid_prompt", ex.Message), statusCode: 400);
            }

            try
            {
                // ComfyUI clients send the editor's workflow as extra_data.extra_pnginfo.workflow; Save Image embeds it in the PNG.
                Dictionary<string, string>? extra = null;
                if (body["extra_data"]?["extra_pnginfo"] is JsonObject pnginfo)
                    extra = pnginfo.Where(kv => kv.Value is not null).ToDictionary(kv => kv.Key, kv => kv.Value is JsonValue v && v.TryGetValue<string>(out var s) ? s : kv.Value!.ToJsonString());
                var item = queue.Enqueue(graph, (string?)body["client_id"], extra, front: (bool?)body["front"] == true);
                return Results.Json(new JsonObject { ["prompt_id"] = item.Id, ["number"] = item.Number, ["node_errors"] = new JsonObject() });
            }
            catch (PromptValidationException ex)
            {
                return Results.Content(ValidationFailure(ex.Result, graph).ToJsonString(), "application/json", statusCode: 400);
            }
        });

        app.MapGet("/prompt", () => Results.Json(new { exec_info = new { queue_remaining = queue.Remaining } }));

        app.MapGet("/queue", () =>
        {
            var (running, pending) = queue.Snapshot();
            return Results.Content(new JsonObject
            {
                ["queue_running"] = new JsonArray(running is null ? [] : [QueueItem(running)]),
                ["queue_pending"] = new JsonArray(pending.Select(p => (JsonNode)QueueItem(p)).ToArray()),
            }.ToJsonString(), "application/json");
        });

        app.MapPost("/queue", async (HttpRequest request) =>
        {
            var body = await JsonNode.ParseAsync(request.Body) as JsonObject;
            if ((bool?)body?["clear"] == true) queue.Remove(_ => true);
            if (body?["delete"] is JsonArray ids)
            {
                var set = ids.Select(i => (string?)i).Where(i => i is not null).ToHashSet();
                queue.Remove(p => set.Contains(p.Id));
            }
            return Results.Ok();
        });

        app.MapPost("/interrupt", () => { queue.Interrupt(); return Results.Ok(); });

        app.MapGet("/history", () => Results.Content(HistoryJson(queue.History).ToJsonString(), "application/json"));
        app.MapGet("/history/{id}", (string id) =>
            queue.FindHistory(id) is { } h ? Results.Content(HistoryJson([h]).ToJsonString(), "application/json") : Results.Json(new JsonObject()));
        app.MapPost("/history", async (HttpRequest request) =>
        {
            if (await JsonNode.ParseAsync(request.Body) is JsonObject body && (bool?)body["clear"] == true) queue.ClearHistory();
            return Results.Ok();
        });

        app.MapGet("/view", (string filename, string? subfolder, string? type) =>
        {
            var root = (type ?? "output") switch { "output" => options.OutputPath, "input" => options.InputPath, "temp" => options.TempPath, _ => null };
            if (root is null) return Results.BadRequest("type must be output, input or temp");
            var path = SafePath(root, subfolder, filename);
            if (path is null || !File.Exists(path)) return Results.NotFound();
            return Results.File(path, ContentType(path));
        });

        app.MapPost("/upload/image", async (HttpRequest request) =>
        {
            if (!request.HasFormContentType) return Results.BadRequest("multipart form data expected");
            var form = await request.ReadFormAsync();
            if (form.Files["image"] is not { } file) return Results.BadRequest("no 'image' file");
            var name = Path.GetFileName(file.FileName);
            if (!ImageDecoder.IsSupportedName(name) && !name.EndsWith(".wav", StringComparison.OrdinalIgnoreCase) && !name.EndsWith(".latent", StringComparison.OrdinalIgnoreCase)) return Results.BadRequest("only PNG and JPEG pictures, WAV sound and .latent files can be loaded for now");
            var target = Path.Combine(options.InputPath, name);
            if (File.Exists(target) && (string?)form["overwrite"] != "true")
                name = $"{Path.GetFileNameWithoutExtension(name)}_{Guid.NewGuid().ToString("N")[..6]}{Path.GetExtension(name)}";
            await using (var output = File.Create(Path.Combine(options.InputPath, name))) await file.CopyToAsync(output);
            return Results.Json(new { name, subfolder = "", type = "input" });
        });

        app.Map("/ws", async (HttpContext context) =>
        {
            if (!context.WebSockets.IsWebSocketRequest) return Results.BadRequest("websocket expected");
            using var socket = await context.WebSockets.AcceptWebSocketAsync();
            var clientId = context.Request.Query["clientId"].ToString();
            await events.ServeAsync(socket, clientId, queue.StatusEvent, context.RequestAborted);
            return Results.Empty;
        });
    }

    // ---- JSON shapes (ComfyUI's) ----

    static JsonObject Error(string type, string message) =>
        new() { ["error"] = new JsonObject { ["type"] = type, ["message"] = message, ["details"] = "", ["extra_info"] = new JsonObject() }, ["node_errors"] = new JsonObject() };

    static JsonObject ValidationFailure(ValidationResult result, PromptGraph graph)
    {
        var nodeErrors = new JsonObject();
        foreach (var group in result.Errors.GroupBy(e => e.NodeId ?? ""))
        {
            var errors = new JsonArray();
            foreach (var e in group)
                errors.Add(new JsonObject { ["type"] = e.Code, ["message"] = e.Message, ["details"] = e.Input ?? "", ["extra_info"] = new JsonObject { ["input_name"] = e.Input } });
            nodeErrors[group.Key] = new JsonObject
            {
                ["errors"] = errors, ["dependent_outputs"] = new JsonArray(),
                ["class_type"] = group.Key.Length > 0 && graph.Nodes.TryGetValue(group.Key, out var n) ? n.ClassType : null,
            };
        }
        var first = result.Errors.FirstOrDefault();
        return new JsonObject
        {
            ["error"] = new JsonObject
            {
                ["type"] = "prompt_outputs_failed_validation", ["message"] = "Prompt outputs failed validation",
                ["details"] = first is null ? "" : first.Message, ["extra_info"] = new JsonObject(),
            },
            ["node_errors"] = nodeErrors,
        };
    }

    static JsonArray QueueItem(QueuedPrompt p) =>
        new(p.Number, p.Id, JsonNode.Parse(PromptJson.ToJson(p.Graph)), new JsonObject { ["client_id"] = p.ClientId },
            new JsonArray(p.Graph.Nodes.Where(n => IsOutput(n.Value.ClassType)).Select(n => (JsonNode)n.Key).ToArray()));

    static bool IsOutput(string classType) => classType is "SaveImage" or "PreviewImage";

    static JsonObject HistoryJson(IEnumerable<HistoryEntry> entries)
    {
        var root = new JsonObject();
        foreach (var h in entries)
        {
            var outputs = new JsonObject();
            foreach (var (node, output) in h.Outputs) outputs[node] = output.DeepClone();
            root[h.Id] = new JsonObject
            {
                ["prompt"] = new JsonArray(h.Number, h.Id, JsonNode.Parse(PromptJson.ToJson(h.Graph)), new JsonObject(), new JsonArray()),
                ["outputs"] = outputs,
                ["status"] = new JsonObject
                {
                    ["status_str"] = h.Status == "success" ? "success" : "error", ["completed"] = h.Status == "success",
                    ["messages"] = new JsonArray(),
                    ["error"] = h.ErrorMessage, ["error_node"] = h.ErrorNodeId, ["interrupted"] = h.Status == "interrupted",
                    ["seconds"] = Math.Round(h.Seconds, 2),
                },
            };
        }
        return root;
    }

    /// <summary>The node catalogue in ComfyUI's <c>object_info</c> format, so tools that read it (and the Studio) know every node's sockets and widgets.</summary>
    public static JsonObject ObjectInfo(NodeRegistry registry)
    {
        var root = new JsonObject();
        foreach (var s in registry.Schemas.OrderBy(s => s.Id))
        {
            JsonObject required = [], optional = [];
            JsonArray requiredOrder = [], optionalOrder = [];
            foreach (var i in s.Inputs)
            {
                (i.ListedAsRequired ? requiredOrder : optionalOrder).Add(i.Name);
                (i.ListedAsRequired ? required : optional)[i.Name] = InputEntry(i);
            }
            var input = new JsonObject { ["required"] = required };
            var inputOrder = new JsonObject { ["required"] = requiredOrder };
            if (optional.Count > 0) { input["optional"] = optional; inputOrder["optional"] = optionalOrder; }
            root[s.Id] = new JsonObject
            {
                ["input"] = input,
                ["input_order"] = inputOrder,
                ["input_is_list"] = s.InputIsList,
                ["output"] = new JsonArray(s.Outputs.Select(o => (JsonNode)(o.MatchTemplate is null ? o.Type.Name : MatchTypeName)).ToArray()),
                ["output_is_list"] = new JsonArray(s.Outputs.Select(o => (JsonNode)o.IsList).ToArray()),
                ["output_name"] = new JsonArray(s.Outputs.Select(o => (JsonNode)o.Name).ToArray()),
                ["output_matchtypes"] = s.Outputs.Any(o => o.MatchTemplate is not null) ? new JsonArray(s.Outputs.Select(o => (JsonNode?)o.MatchTemplate).ToArray()) : null,
                ["name"] = s.Id, ["display_name"] = s.DisplayName, ["description"] = s.Description ?? "",
                ["category"] = s.Category, ["output_node"] = s.IsOutputNode, ["dev_only"] = s.DevOnly,
            };
        }
        return root;
    }

    /// <summary>ComfyUI's type name for a MatchType socket; its config carries the template.</summary>
    const string MatchTypeName = "COMFY_MATCHTYPE_V3";

    static JsonArray InputEntry(InputSpec i)
    {
        var config = new JsonObject();
        if (i.Lazy) config["lazy"] = true;
        if (i.ForceInput) config["forceInput"] = true;
        if (i.Group is { } group) { config["autogrow_group"] = group; config["autogrow_min"] = i.GroupMin; }
        if (i.DynamicParent is { } parent) { config["dynamic_parent"] = parent; config["dynamic_options"] = new JsonArray(i.DynamicOptions.Select(o => (JsonNode)o).ToArray()); }
        if (i.MatchTemplate is { } template)
        {
            config["template"] = new JsonObject { ["template_id"] = template, ["allowed_types"] = "*" };
            return new JsonArray(MatchTypeName, config);
        }
        if (i.Widget is not { } w) return config.Count > 0 ? new JsonArray(i.Type.Name, config) : new JsonArray(i.Type.Name);
        if (w.Default is { } d) config["default"] = JsonSerializer.SerializeToNode(d);
        if (w.Min is { } min) config["min"] = min;
        if (w.Max is { } max) config["max"] = max;
        if (w.Step is { } step) config["step"] = step;
        if (w.Multiline) config["multiline"] = true;
        if (w.Kind == WidgetKind.Color) config["socketless"] = true;
        return w.Kind switch
        {
            WidgetKind.Combo => new JsonArray(new JsonArray(w.ResolveOptions().Select(o => (JsonNode)o).ToArray()), config),
            _ => new JsonArray(i.Type.Name, config),
        };
    }

    // ---- files ----

    /// <summary>The file under <paramref name="root"/>, or null when the request tries to leave it (a path-traversal attempt).</summary>
    internal static string? SafePath(string root, string? subfolder, string filename)
    {
        if (string.IsNullOrEmpty(filename) || filename != Path.GetFileName(filename)) return null;
        var rootFull = Path.GetFullPath(root);
        var full = Path.GetFullPath(Path.Combine(rootFull, subfolder ?? "", filename));
        return full.StartsWith(rootFull + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase) ? full : null;
    }

    static string ContentType(string path) => Path.GetExtension(path).ToLowerInvariant() switch
    {
        ".png" => "image/png", ".jpg" or ".jpeg" => "image/jpeg", ".webp" => "image/webp", ".wav" => "audio/wav", ".glb" => "model/gltf-binary", _ => "application/octet-stream",
    };
}
