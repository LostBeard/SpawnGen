using System.Net.WebSockets;
using System.Text;
using System.Text.Json.Nodes;
using SpawnGen.Runtime;

namespace SpawnGen.Server;



/// <summary>Fans events out to connected websocket clients and in-process subscribers (tests, logging).</summary>
public sealed class EventHub : IRunEvents
{
    readonly List<Client> _clients = [];
    readonly object _gate = new();

    sealed class Client(WebSocket socket, string clientId)
    {
        public WebSocket Socket { get; } = socket;
        public string ClientId { get; } = clientId;
        public SemaphoreSlim SendLock { get; } = new(1, 1);
    }

    public event Action<RunEvent>? Published;

    /// <summary>Serves one websocket until it closes. Everything published meanwhile is sent to it.</summary>
    public async Task ServeAsync(WebSocket socket, string clientId, Func<RunEvent> hello, CancellationToken ct)
    {
        var client = new Client(socket, clientId);
        lock (_gate) _clients.Add(client);
        try
        {
            await SendAsync(client, hello().ToJson(), ct);
            var buffer = new byte[1024];
            while (socket.State == WebSocketState.Open)
            {
                var result = await socket.ReceiveAsync(buffer, ct); // the client sends nothing we act on; reading keeps the close handshake working
                if (result.MessageType == WebSocketMessageType.Close)
                {
                    await socket.CloseAsync(WebSocketCloseStatus.NormalClosure, null, CancellationToken.None);
                    break;
                }
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
        var json = e.ToJson();
        foreach (var c in clients)
            _ = SendAsync(c, json, CancellationToken.None); // a slow or dead client must not hold up a run
    }

    static async Task SendAsync(Client c, string json, CancellationToken ct)
    {
        await c.SendLock.WaitAsync(ct);
        try
        {
            if (c.Socket.State == WebSocketState.Open)
                await c.Socket.SendAsync(Encoding.UTF8.GetBytes(json), WebSocketMessageType.Text, true, ct);
        }
        catch (Exception ex) when (ex is WebSocketException or ObjectDisposedException or OperationCanceledException) { }
        finally { c.SendLock.Release(); }
    }
}
