using System.Threading.Channels;
using System.Text.Json;
using System.Text.Json.Nodes;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Runtime;

public sealed record QueuedPrompt(int Number, string Id, PromptGraph Graph, string? ClientId, IReadOnlyDictionary<string, string>? ExtraPngInfo = null);

/// <summary>One finished (or failed) run, as <c>/history</c> reports it.</summary>
public sealed class HistoryEntry
{
    public required int Number { get; init; }
    public required string Id { get; init; }
    public required PromptGraph Graph { get; init; }
    /// <summary>"success", "error" or "interrupted".</summary>
    public required string Status { get; init; }
    public string? ErrorMessage { get; init; }
    public string? ErrorNodeId { get; init; }
    public IReadOnlyDictionary<string, JsonObject> Outputs { get; init; } = new Dictionary<string, JsonObject>();
    public IReadOnlyList<string> Cached { get; init; } = [];
    public double Seconds { get; init; }
}

/// <summary>
/// Runs submitted prompts one at a time through the engine (a GPU can only do one job at once) and reports progress as <see cref="RunEvent"/>s
/// shaped like ComfyUI's. The executor is shared, so a re-run with a changed seed only recomputes what depends on it.
/// </summary>
public sealed class PromptQueue : IDisposable
{
    const int HistoryLimit = 200;

    readonly PromptExecutor _executor;
    readonly IRunEvents _events;
    readonly Channel<QueuedPrompt> _channel = Channel.CreateUnbounded<QueuedPrompt>();
    readonly List<QueuedPrompt> _pending = [];
    readonly List<HistoryEntry> _history = [];
    readonly object _gate = new();
    readonly CancellationTokenSource _stop = new();
    readonly Task _worker;
    QueuedPrompt? _running;
    CancellationTokenSource? _runCancel;
    int _number;

    public PromptQueue(NodeRegistry registry, IServiceProvider services, IRunEvents events)
    {
        Registry = registry;
        _events = events;
        _executor = new PromptExecutor(registry, services);
        _worker = Task.Run(RunLoopAsync);
    }

    public NodeRegistry Registry { get; }

    /// <summary>Validates and queues the prompt. Throws <see cref="PromptValidationException"/> (nothing is queued) when it cannot run.</summary>
    public QueuedPrompt Enqueue(PromptGraph graph, string? clientId, IReadOnlyDictionary<string, string>? extraPngInfo = null)
    {
        var validation = _executor.Validate(graph);
        if (!validation.IsValid) throw new PromptValidationException(validation);

        QueuedPrompt item;
        lock (_gate)
        {
            item = new QueuedPrompt(++_number, Guid.NewGuid().ToString("D"), graph, clientId, extraPngInfo);
            _pending.Add(item);
        }
        _channel.Writer.TryWrite(item);
        PublishStatus();
        return item;
    }

    public (QueuedPrompt? Running, IReadOnlyList<QueuedPrompt> Pending) Snapshot()
    {
        lock (_gate) return (_running, [.. _pending]);
    }

    public int Remaining
    {
        get { lock (_gate) return _pending.Count + (_running is null ? 0 : 1); }
    }

    public IReadOnlyList<HistoryEntry> History
    {
        get { lock (_gate) return [.. _history]; }
    }

    public HistoryEntry? FindHistory(string id)
    {
        lock (_gate) return _history.FirstOrDefault(h => h.Id == id);
    }

    public void ClearHistory()
    {
        lock (_gate) _history.Clear();
    }

    /// <summary>Stops the run in progress (the next queued prompt then starts).</summary>
    public bool Interrupt()
    {
        lock (_gate)
        {
            if (_running is null) return false;
            _runCancel?.Cancel();
            return true;
        }
    }

    /// <summary>Removes waiting prompts (not the running one). Returns how many were removed.</summary>
    public int Remove(Func<QueuedPrompt, bool> predicate)
    {
        int removed;
        lock (_gate) removed = _pending.RemoveAll(p => predicate(p));
        if (removed > 0) PublishStatus();
        return removed;
    }

    public void Dispose()
    {
        _stop.Cancel();
        _channel.Writer.TryComplete();
        try { _worker.Wait(TimeSpan.FromSeconds(10)); } catch (AggregateException) { }
        _stop.Dispose();
    }

    public RunEvent StatusEvent() =>
        new("status", new JsonObject { ["status"] = new JsonObject { ["exec_info"] = new JsonObject { ["queue_remaining"] = Remaining } } });

    void PublishStatus() => _events.Publish(StatusEvent());

    async Task RunLoopAsync()
    {
        try
        {
            await foreach (var item in _channel.Reader.ReadAllAsync(_stop.Token))
            {
                lock (_gate)
                {
                    if (!_pending.Remove(item)) continue; // removed from the queue while waiting
                    _running = item;
                    _runCancel = CancellationTokenSource.CreateLinkedTokenSource(_stop.Token);
                }
                PublishStatus();
                await RunOneAsync(item, _runCancel.Token);
                lock (_gate)
                {
                    _running = null;
                    _runCancel.Dispose();
                    _runCancel = null;
                }
                PublishStatus();
            }
        }
        catch (OperationCanceledException) { }
    }

    async Task RunOneAsync(QueuedPrompt item, CancellationToken ct)
    {
        var observer = new RunObserver(item, _events);
        var started = DateTime.UtcNow;
        string status = "success";
        string? errorMessage = null, errorNode = null;
        ExecutionResult? result = null;
        try
        {
            result = await _executor.ExecuteAsync(item.Graph, new ExecutionOptions { PromptId = item.Id, ExtraPngInfo = item.ExtraPngInfo }, observer, ct);
        }
        catch (OperationCanceledException)
        {
            status = "interrupted";
            _events.Publish(new RunEvent("execution_interrupted", new JsonObject { ["prompt_id"] = item.Id }));
        }
        catch (Exception ex)
        {
            status = "error";
            errorMessage = ex is NodeExecutionException { InnerException: { } inner } ? inner.Message : ex.Message;
            errorNode = (ex as NodeExecutionException)?.NodeId;
            // The engine's observer already published execution_error for failures inside a run.
        }

        var entry = new HistoryEntry
        {
            Number = item.Number, Id = item.Id, Graph = item.Graph, Status = status, ErrorMessage = errorMessage, ErrorNodeId = errorNode,
            Outputs = observer.Outputs, Cached = result?.Cached ?? observer.CachedNodes, Seconds = (DateTime.UtcNow - started).TotalSeconds,
        };
        lock (_gate)
        {
            _history.Add(entry);
            if (_history.Count > HistoryLimit) _history.RemoveAt(0);
        }
        if (status == "success")
            _events.Publish(new RunEvent("execution_success", new JsonObject { ["prompt_id"] = item.Id }));
    }

    /// <summary>Turns the engine's run events into the websocket events a ComfyUI client expects, and keeps the UI outputs for the history.</summary>
    sealed class RunObserver(QueuedPrompt item, IRunEvents events) : IExecutionObserver
    {
        public Dictionary<string, JsonObject> Outputs { get; } = [];
        public List<string> CachedNodes { get; } = [];

        JsonObject Data(string? node = null)
        {
            var d = new JsonObject { ["prompt_id"] = item.Id };
            if (node is not null) d["node"] = node;
            return d;
        }

        public void OnStart(string promptId, IReadOnlyList<string> plan) => events.Publish(new("execution_start", Data()));

        public void OnNodeCached(string nodeId)
        {
            CachedNodes.Add(nodeId);
            var d = Data();
            d["nodes"] = new JsonArray(nodeId);
            events.Publish(new("execution_cached", d));
        }

        public void OnNodeExecuting(string nodeId) => events.Publish(new("executing", Data(nodeId)));

        public void OnNodeProgress(string nodeId, double value, double max, string? message)
        {
            var d = Data(nodeId);
            if (max > 0) { d["value"] = value; d["max"] = max; events.Publish(new("progress", d)); }
            else if (message is not null) { d["message"] = message; events.Publish(new("node_status", d)); }
        }

        public void OnNodePreview(string nodeId, byte[] png)
        {
            var d = Data(nodeId);
            d["image"] = "data:image/png;base64," + Convert.ToBase64String(png);
            events.Publish(new("preview", d));
        }

        public void OnNodeExecuted(string nodeId, NodeOutputs outputs)
        {
            if (outputs.Ui is not { Count: > 0 } ui) return;
            var output = new JsonObject();
            foreach (var (key, value) in ui)
                output[key] = JsonSerializer.SerializeToNode(value);
            Outputs[nodeId] = output;
            var d = Data(nodeId);
            d["output"] = output.DeepClone();
            events.Publish(new("executed", d));
        }

        public void OnError(string? nodeId, Exception error)
        {
            if (error is OperationCanceledException) return;
            var d = Data();
            if (nodeId is not null) d["node_id"] = nodeId;
            var inner = error is NodeExecutionException { InnerException: { } i } ? i : error;
            d["exception_message"] = inner.Message;
            d["exception_type"] = inner.GetType().Name;
            if (error is NodeExecutionException nee) d["node_type"] = nee.ClassType;
            events.Publish(new("execution_error", d));
        }
    }
}
