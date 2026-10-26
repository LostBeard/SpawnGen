using System.Net.Http.Json;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json.Nodes;

namespace SpawnGen.Studio.Run;


/// <summary>Talks to the SpawnGen server that served this page: queues prompts over HTTP and receives progress over a WebSocket (the ComfyUI protocol).</summary>
public sealed class ServerClient(HttpClient http) : IEngineClient, IAsyncDisposable
{
    readonly CancellationTokenSource _stop = new();
    ClientWebSocket? _socket;
    Task? _loop;

    public string Name => "Server (CUDA / OpenCL)";

    public string ClientId { get; } = Guid.NewGuid().ToString("N");
    public bool Connected => _socket?.State == WebSocketState.Open;

    /// <summary>Raised for every websocket message: <c>{"type": ..., "data": {...}}</c>.</summary>
    public event Action<JsonObject>? EventReceived;
    public event Action? ConnectionChanged;

    /// <summary>True when the page was served by a SpawnGen server (the Studio can also run alone, editing only).</summary>
    public async Task<bool> ProbeAsync()
    {
        // A static host (GitHub Pages) has no server to ask: skip the request, which would only put a red 404 in the console.
        if (http.BaseAddress?.Host.EndsWith(".github.io", StringComparison.OrdinalIgnoreCase) == true) return false;
        try
        {
            using var response = await http.GetAsync("prompt", _stop.Token);
            return response.IsSuccessStatusCode && response.Content.Headers.ContentType?.MediaType == "application/json";
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException) { return false; }
    }

    /// <summary>The server's node catalogue as schemas (its custom nodes included), or empty when it cannot be read.</summary>
    public async Task<IReadOnlyList<SpawnGen.Engine.Nodes.NodeSchema>> FetchSchemasAsync()
    {
        try
        {
            var info = await http.GetFromJsonAsync<JsonObject>("object_info", _stop.Token);
            return info is null ? [] : SpawnGen.Engine.Nodes.NodeSchemaJson.FromObjectInfo(info);
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or System.Text.Json.JsonException) { return []; }
    }

    public void Start() => _loop ??= Task.Run(ConnectLoopAsync);

    async Task ConnectLoopAsync()
    {
        var delay = TimeSpan.FromSeconds(1);
        while (!_stop.IsCancellationRequested)
        {
            try
            {
                _socket = new ClientWebSocket();
                var uri = new UriBuilder(new Uri(http.BaseAddress!, "ws")) { Query = "clientId=" + ClientId };
                uri.Scheme = uri.Scheme == "https" ? "wss" : "ws";
                await _socket.ConnectAsync(uri.Uri, _stop.Token);
                // As ComfyUI's frontend does: previews then come with the node they belong to.
                await _socket.SendAsync(Encoding.UTF8.GetBytes("""{"type":"feature_flags","data":{"supports_preview_metadata":true}}"""),
                    WebSocketMessageType.Text, true, _stop.Token);
                ConnectionChanged?.Invoke();
                delay = TimeSpan.FromSeconds(1);
                await ReceiveAsync(_socket);
            }
            catch (Exception ex) when (ex is WebSocketException or HttpRequestException or OperationCanceledException or ObjectDisposedException) { }
            ConnectionChanged?.Invoke();
            if (_stop.IsCancellationRequested) break;
            try { await Task.Delay(delay, _stop.Token); } catch (OperationCanceledException) { break; }
            delay = TimeSpan.FromSeconds(Math.Min(delay.TotalSeconds * 2, 10)); // back off while the server is down
        }
    }

    async Task ReceiveAsync(ClientWebSocket socket)
    {
        var buffer = new byte[64 * 1024];
        var message = new MemoryStream();
        while (socket.State == WebSocketState.Open && !_stop.IsCancellationRequested)
        {
            var result = await socket.ReceiveAsync(buffer, _stop.Token);
            if (result.MessageType == WebSocketMessageType.Close) return;
            message.Write(buffer, 0, result.Count);
            if (!result.EndOfMessage) continue;
            if (result.MessageType == WebSocketMessageType.Binary)
            {
                if (PreviewEvent(message.GetBuffer().AsSpan(0, (int)message.Length)) is { } preview) EventReceived?.Invoke(preview);
                message.SetLength(0);
                continue;
            }
            var text = Encoding.UTF8.GetString(message.GetBuffer(), 0, (int)message.Length);
            message.SetLength(0);
            if (JsonNode.Parse(text) is JsonObject json) EventReceived?.Invoke(json);
        }
    }

    /// <summary>ComfyUI's PREVIEW_IMAGE_WITH_METADATA frame (event 4: header length, JSON header with the node, the image) as the Studio's preview event.</summary>
    public static JsonObject? PreviewEvent(ReadOnlySpan<byte> frame)
    {
        if (frame.Length < 8 || System.Buffers.Binary.BinaryPrimitives.ReadUInt32BigEndian(frame) != 4) return null;
        int length = (int)System.Buffers.Binary.BinaryPrimitives.ReadUInt32BigEndian(frame[4..]);
        if (length < 0 || 8 + length > frame.Length) return null;
        if (JsonNode.Parse(Encoding.UTF8.GetString(frame.Slice(8, length))) is not JsonObject meta) return null;
        var image = frame[(8 + length)..];
        return new JsonObject
        {
            ["type"] = "preview",
            ["data"] = new JsonObject
            {
                ["prompt_id"] = (string?)meta["prompt_id"], ["node"] = (string?)meta["display_node_id"] ?? (string?)meta["node_id"],
                ["image"] = $"data:{(string?)meta["image_type"] ?? "image/png"};base64,{Convert.ToBase64String(image)}",
            },
        };
    }

    public async Task<QueueResult> QueueAsync(string promptJson, string? workflowJson = null, bool front = false, string? previewMethod = null)
    {
        var body = new JsonObject { ["prompt"] = JsonNode.Parse(promptJson), ["client_id"] = ClientId };
        if (front) body["front"] = true;
        if (workflowJson is not null)
            body["extra_data"] = new JsonObject { ["extra_pnginfo"] = new JsonObject { ["workflow"] = JsonNode.Parse(workflowJson) } };
        if (previewMethod is not null)
        {
            if (body["extra_data"] is not JsonObject extra) body["extra_data"] = extra = new JsonObject();
            extra["preview_method"] = previewMethod; // ComfyUI's frontend sends its preview setting here
        }
        try
        {
            using var response = await http.PostAsJsonAsync("prompt", body, _stop.Token);
            var json = await response.Content.ReadFromJsonAsync<JsonObject>(_stop.Token);
            if (response.IsSuccessStatusCode) return new QueueResult(true, (string?)json?["prompt_id"], null, new Dictionary<int, string>());

            var errors = new Dictionary<int, string>();
            if (json?["node_errors"] is JsonObject perNode)
                foreach (var (id, detail) in perNode)
                    if (int.TryParse(id, out var nodeId) && detail?["errors"] is JsonArray list && list.Count > 0)
                        errors[nodeId] = string.Join("; ", list.Select(e => (string?)e?["message"]));
            return new QueueResult(false, null, (string?)json?["error"]?["message"] ?? $"Server said {(int)response.StatusCode}", errors);
        }
        catch (HttpRequestException ex)
        {
            return new QueueResult(false, null, "Could not reach the server: " + ex.Message, new Dictionary<int, string>());
        }
    }

    public Task InterruptAsync() => http.PostAsync("interrupt", null, _stop.Token);

    public Task CancelQueuedAsync(string promptId) =>
        http.PostAsJsonAsync("queue", new JsonObject { ["delete"] = new JsonArray(promptId) }, _stop.Token);

    public async Task<string?> UploadImageAsync(string name, byte[] png)
    {
        using var form = new MultipartFormDataContent { { new ByteArrayContent(png), "image", name } };
        using var response = await http.PostAsync("upload/image", form, _stop.Token);
        if (!response.IsSuccessStatusCode) return null;
        return (string?)(await response.Content.ReadFromJsonAsync<JsonObject>(_stop.Token))?["name"];
    }

    public async Task<SpawnGen.Runtime.ModelCacheInfo> GetModelCacheAsync()
    {
        try { return await http.GetFromJsonAsync<SpawnGen.Runtime.ModelCacheInfo>("models/cache", _stop.Token) ?? new([], 0, []); }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException or System.Text.Json.JsonException) { return new([], 0, []); }
    }

    public Task RemoveCachedModelAsync(string key) => http.DeleteAsync("models/cache/" + Uri.EscapeDataString(key), _stop.Token);

    public Task ClearModelCacheAsync() => http.PostAsync("models/cache/clear", null, _stop.Token);

    /// <summary>The Load Image choices the server offers, read from its node catalogue (<c>/object_info</c>).</summary>
    public async Task<IReadOnlyList<string>> ListInputImagesAsync()
    {
        try
        {
            var info = await http.GetFromJsonAsync<JsonObject>("object_info", _stop.Token);
            return info?["LoadImage"]?["input"]?["required"]?["image"]?[0] is JsonArray options
                ? options.Select(o => (string?)o ?? "").Where(o => o.Length > 0).ToList()
                : [];
        }
        catch (HttpRequestException) { return []; }
    }

    /// <summary>The URL of a result file the server holds (a SaveImage or PreviewImage output).</summary>
    public string ViewUrl(string filename, string? subfolder, string? type) =>
        $"view?filename={Uri.EscapeDataString(filename)}&subfolder={Uri.EscapeDataString(subfolder ?? "")}&type={Uri.EscapeDataString(type ?? "output")}";

    public async ValueTask DisposeAsync()
    {
        _stop.Cancel();
        try { _socket?.Abort(); _socket?.Dispose(); } catch (ObjectDisposedException) { }
        if (_loop is not null) await Task.WhenAny(_loop, Task.Delay(500));
        _stop.Dispose();
    }
}
