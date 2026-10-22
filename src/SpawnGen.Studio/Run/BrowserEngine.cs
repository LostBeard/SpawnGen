using System.Text.Json.Nodes;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;
using SpawnGen.Runtime;

namespace SpawnGen.Studio.Run;

/// <summary>
/// Runs graphs inside this page, on the browser's GPU (WebGPU). The same run queue, nodes and events as the server; models stream from the model
/// host into the browser's origin-private storage once and stay there, and results are kept in memory.
/// </summary>
public sealed class BrowserEngine : IEngineClient, IMlRuntimeProvider, IServiceProvider
{
    readonly SpawnJSRuntime _js;
    readonly HttpClient _http;
    readonly RunEventBus _bus = new();
    readonly MemoryResultStore _results = new();
    readonly MemoryInputStore _inputs = new();
    readonly PromptQueue _queue;
    Task<MlRuntime>? _runtime;

    public BrowserEngine(SpawnJSRuntime js, HttpClient http, NodeRegistry registry)
    {
        _js = js;
        _http = http;
        InputFiles.Store = _inputs;
        _bus.Published += e =>
        {
            if (_perOp) ReportPerOpTimings(e.Type);
            EventReceived?.Invoke(new JsonObject { ["type"] = e.Type, ["data"] = e.Data.DeepClone() });
        };
        _queue = new PromptQueue(registry, this, _bus);
        // Debug switch: open the page with ?perop=1 to make the library wait for every GPU op and record how long each took (slow), and print the
        // slowest ones to the console when a run ends. For finding the single op that overruns a GPU watchdog.
        try { _perOp = _js.Get<string>("location.search")?.Contains("perop=1") == true; } catch (Exception) { }
    }

    readonly bool _perOp;

    void ReportPerOpTimings(string eventType)
    {
        if (eventType == "execution_start")
        {
            GraphExecutor.PerOpSync = true;
            GraphExecutor.CapturedNodeTimingsMs = new Dictionary<string, double>();
        }
        else if (eventType is "execution_success" or "execution_error" && GraphExecutor.CapturedNodeTimingsMs is { } timings)
        {
            var last = timings.Keys.LastOrDefault();
            Console.WriteLine($"[perop] {timings.Count} ops timed; last op recorded: {last}");
            foreach (var (op, ms) in timings.OrderByDescending(t => t.Value).Take(12))
                Console.WriteLine($"[perop] {ms,9:F0} ms  {op}");
        }
    }

    public string Name => "This browser (WebGPU)";

    public event Action<JsonObject>? EventReceived;

    /// <summary>True when the browser exposes WebGPU (it needs a secure context: https or localhost).</summary>
    public bool WebGpuAvailable
    {
        get
        {
            try
            {
                using var navigator = _js.Get<Navigator>("navigator");
                return navigator.Gpu is not null;
            }
            catch (Exception) { return false; }
        }
    }

    // ---- services the nodes ask for ----

    public object? GetService(Type serviceType)
    {
        if (serviceType == typeof(IResultStore)) return _results;
        if (serviceType == typeof(IMlRuntimeProvider)) return this;
        return null;
    }

    /// <summary>Opens the GPU on first use (the first ML node to run asks for it) and keeps it.</summary>
    public Task<MlRuntime> GetAsync() => _runtime ??= CreateRuntimeAsync();

    async Task<MlRuntime> CreateRuntimeAsync()
    {
        var builder = MLContext.Create();
        await builder.AllAcceleratorsAsync();
        var context = builder.ToContext();
        var accelerator = await context.CreatePreferredAcceleratorAsync()
            ?? throw new InvalidOperationException("This browser has no GPU the engine can use (it needs WebGPU).");
        // Model files stream into the origin-private file system and are reused on the next visit, never held whole in memory.
        return new MlRuntime(new HubModelSource(_js, _http), accelerator);
    }

    // ---- IEngineClient ----

    public Task<QueueResult> QueueAsync(string promptJson, string? workflowJson = null)
    {
        PromptGraph graph;
        try { graph = PromptJson.Parse(promptJson); }
        catch (Exception ex) when (ex is System.Text.Json.JsonException or InvalidDataException or FormatException)
        {
            return Task.FromResult(new QueueResult(false, null, "That prompt cannot be read: " + ex.Message, new Dictionary<int, string>()));
        }
        try
        {
            var item = _queue.Enqueue(graph, "browser", workflowJson is null ? null : new Dictionary<string, string> { ["workflow"] = workflowJson });
            return Task.FromResult(new QueueResult(true, item.Id, null, new Dictionary<int, string>()));
        }
        catch (PromptValidationException ex)
        {
            var perNode = new Dictionary<int, string>();
            foreach (var group in ex.Result.Errors.Where(e => int.TryParse(e.NodeId, out _)).GroupBy(e => int.Parse(e.NodeId!)))
                perNode[group.Key] = string.Join("; ", group.Select(e => e.Message));
            return Task.FromResult(new QueueResult(false, null, "Prompt outputs failed validation", perNode));
        }
    }

    public Task InterruptAsync()
    {
        _queue.Interrupt();
        return Task.CompletedTask;
    }

    public Task CancelQueuedAsync(string promptId)
    {
        _queue.Remove(p => p.Id == promptId);
        return Task.CompletedTask;
    }

    /// <summary>Results live in memory here, so the picture itself is the URL (a data URL).</summary>
    public string ViewUrl(string filename, string? subfolder, string? type) =>
        _results.Get(type ?? "output", filename) is { } png ? "data:image/png;base64," + Convert.ToBase64String(png) : "";

    public async Task<string?> UploadImageAsync(string name, byte[] png) => await _inputs.SaveAsync(Path.GetFileName(name), png);

    public Task<IReadOnlyList<string>> ListInputImagesAsync() => Task.FromResult(_inputs.List());
}
