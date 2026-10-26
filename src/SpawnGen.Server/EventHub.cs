using System.Buffers.Binary;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json.Nodes;
using SpawnGen.Runtime;

namespace SpawnGen.Server;



/// <summary>
/// Fans events out to connected websocket clients and in-process subscribers (tests, logging), in ComfyUI's protocol: JSON events, and previews as binary
/// frames. A client whose first message is <c>feature_flags</c> with <c>supports_preview_metadata</c> gets PREVIEW_IMAGE_WITH_METADATA (event 4: the length of a
/// JSON header naming the node and prompt, the header, the image); any other client gets PREVIEW_IMAGE (event 1: the image type, 2 for PNG, then the image).
/// </summary>
public sealed class EventHub : IRunEvents
{
    const int PreviewImage = 1, PreviewImageWithMetadata = 4, PngType = 2;
    readonly List<Client> _clients = [];
    readonly object _gate = new();

    sealed class Client(WebSocket socket, string clientId)
    {
        public WebSocket Socket { get; } = socket;
        public string ClientId { get; } = clientId;
        public SemaphoreSlim SendLock { get; } = new(1, 1);
        public bool PreviewMetadata { get; set; }
    }

    public event Action<RunEvent>? Published;

    /// <summary>What this server tells a client that sends its feature flags (ComfyUI's server feature flags, the ones SpawnGen has).</summary>
    static JsonObject ServerFeatures() => new() { ["supports_preview_metadata"] = true };

    /// <summary>Serves one websocket until it closes. Everything published meanwhile is sent to it.</summary>
    public async Task ServeAsync(WebSocket socket, string clientId, Func<RunEvent> hello, CancellationToken ct)
    {
        var client = new Client(socket, clientId);
        lock (_gate) _clients.Add(client);
        try
        {
            await SendAsync(client, hello().ToJson(), ct);
            var buffer = new byte[4096];
            var message = new MemoryStream();
            bool first = true;
            while (socket.State == WebSocketState.Open)
            {
                var result = await socket.ReceiveAsync(buffer, ct);
                if (result.MessageType == WebSocketMessageType.Close)
                {
                    await socket.CloseAsync(WebSocketCloseStatus.NormalClosure, null, CancellationToken.None);
                    break;
                }
                if (message.Length < 1 << 20) message.Write(buffer, 0, result.Count);
                if (!result.EndOfMessage) continue;
                // Only the first message means anything: ComfyUI's feature flags.
                if (first && result.MessageType == WebSocketMessageType.Text)
                {
                    try
                    {
                        if (JsonNode.Parse(Encoding.UTF8.GetString(message.GetBuffer(), 0, (int)message.Length)) is JsonObject json && (string?)json["type"] == "feature_flags")
                        {
                            client.PreviewMetadata = json["data"]?["supports_preview_metadata"] is JsonValue v && v.TryGetValue<bool>(out var yes) && yes;
                            await SendAsync(client, new RunEvent("feature_flags", ServerFeatures()).ToJson(), ct);
                        }
                    }
                    catch (System.Text.Json.JsonException) { }
                }
                first = false;
                message.SetLength(0);
            }
        }
        catch (Exception ex) when (ex is WebSocketException or OperationCanceledException) { }
        finally
        {
            lock (_gate) _clients.Remove(client);
        }
    }

    public void Publish(RunEvent e)
    {
        Published?.Invoke(e);
        Client[] clients;
        lock (_gate) clients = [.. _clients];
        if (e.Image is { } image)
        {
            byte[]? plain = null, withMetadata = null;
            foreach (var c in clients)
                _ = SendAsync(c, c.PreviewMetadata ? withMetadata ??= MetadataFrame(e, image) : plain ??= PlainFrame(image), WebSocketMessageType.Binary, CancellationToken.None);
            return;
        }
        var json = Encoding.UTF8.GetBytes(e.ToJson());
        foreach (var c in clients)
            _ = SendAsync(c, json, WebSocketMessageType.Text, CancellationToken.None); // a slow or dead client must not hold up a run
    }

    static byte[] PlainFrame(byte[] image)
    {
        var frame = new byte[8 + image.Length];
        BinaryPrimitives.WriteUInt32BigEndian(frame, PreviewImage);
        BinaryPrimitives.WriteUInt32BigEndian(frame.AsSpan(4), PngType);
        image.CopyTo(frame, 8);
        return frame;
    }

    static byte[] MetadataFrame(RunEvent e, byte[] image)
    {
        var node = (string?)e.Data["node"];
        var metadata = Encoding.UTF8.GetBytes(new JsonObject
        {
            ["node_id"] = node, ["prompt_id"] = (string?)e.Data["prompt_id"], ["display_node_id"] = node, ["parent_node_id"] = null, ["real_node_id"] = node,
            ["image_type"] = "image/png",
        }.ToJsonString());
        var frame = new byte[8 + metadata.Length + image.Length];
        BinaryPrimitives.WriteUInt32BigEndian(frame, PreviewImageWithMetadata);
        BinaryPrimitives.WriteUInt32BigEndian(frame.AsSpan(4), (uint)metadata.Length);
        metadata.CopyTo(frame, 8);
        image.CopyTo(frame, 8 + metadata.Length);
        return frame;
    }

    static Task SendAsync(Client c, string json, CancellationToken ct) => SendAsync(c, Encoding.UTF8.GetBytes(json), WebSocketMessageType.Text, ct);

    static async Task SendAsync(Client c, byte[] payload, WebSocketMessageType type, CancellationToken ct)
    {
        await c.SendLock.WaitAsync(ct);
        try
        {
            if (c.Socket.State == WebSocketState.Open)
                await c.Socket.SendAsync(payload, type, true, ct);
        }
        catch (Exception ex) when (ex is WebSocketException or ObjectDisposedException or OperationCanceledException) { }
        finally { c.SendLock.Release(); }
    }
}
