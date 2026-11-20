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
    // Results stay in JS (AGENTS.md rule 4): a GPU encoder's file goes from the device into a Uint8Array, a host node's file is copied in once
    // and its .NET array dropped; .NET reads one back only when a node loads it.
    readonly MemoryResultStore _results = new(
        data => new JsBytes((Uint8Array)data),
        async view => new JsBytes(await view.CopyToHostUint8ArrayAsync())); // ILGPU 5.4.41's view overload (Data, 2026-11-20): exactly the view, into JS

    /// <summary>A result held in a JS Uint8Array.</summary>
    sealed class JsBytes(Uint8Array data) : StoredBytes
    {
        public Uint8Array Data { get; } = data;
        public override long Length => Data.Length;
        public override byte[] Read() => (byte[])Data;
    }
    readonly MemoryInputStore _inputs = new();

    /// <summary>The pictures uploaded to this engine (Load Image reads them when this engine runs).</summary>
    public IInputStore Inputs => _inputs;
    readonly PromptQueue _queue;
    Task<MlRuntime>? _runtime;

    public BrowserEngine(SpawnJSRuntime js, HttpClient http, NodeRegistry registry)
    {
        _js = js;
        _http = http;
        InputFiles.Store = _inputs;
        _example = AddExamplePictureAsync();
        _hosted = RegisterHostedModelsAsync();
        // WebP, GIF and BMP for Load Image decode on the page's GPU (SpawnDev.Codecs over WebGPU), opened on the first such picture.
        CodecsImages.Install(async () => (await GetAsync()).Accelerator);
        CodecsAudio.Install(async () => (await GetAsync()).Accelerator);
        _bus.Published += e =>
        {
            if (_perOp) ReportPerOpTimings(e.Type);
            if (_nodeTimes) ReportNodeTimes(e.Type, e.Data);
            EventReceived?.Invoke(new JsonObject { ["type"] = e.Type, ["data"] = e.Data.DeepClone() });
        };
        _queue = new PromptQueue(registry, this, _bus) { BeforeRun = ReplaceLostDeviceAsync, RunFailed = NoteLostDevice };
        // Debug switch: open the page with ?perop=1 to make the library wait for every GPU op and record how long each took (slow), and print the
        // slowest ones to the console when a run ends. For finding the single op that overruns a GPU watchdog.
        try { _perOp = _js.Get<string>("location.search")?.Contains("perop=1") == true; } catch (Exception) { }
        try { _nodeTimes = _js.Get<string>("location.search")?.Contains("nodetimes=1") == true; } catch (Exception) { }
        // Test switch: ?testhooks=1 adds window.spawngenLoseGpu(), which destroys the engine's WebGPU device (a real device loss, as a GPU reset
        // gives), for tools/studio-browser-test/drive-device-lost.js.
        try
        {
            if (_js.Get<string>("location.search")?.Contains("testhooks=1") == true)
                _js.Set("spawngenLoseGpu", new ActionCallback(LoseDeviceForTest));
        }
        catch (Exception) { }
    }

    bool _deviceLost;

    /// <summary>
    /// A run that failed because the device was lost (the browser's own words: "device was lost" / "device has been lost") marks it, whether or
    /// not the accelerator's lost flag has caught up (measured 2026-11-19: after GPUDevice.destroy() a run failed on a mapAsync with "The Device
    /// was lost" while IsDeviceLost was still false).
    /// </summary>
    void NoteLostDevice(Exception error)
    {
        for (Exception? e = error; e is not null; e = e.InnerException)
            if (System.Text.RegularExpressions.Regex.IsMatch(e.Message, @"device (was|has been|is) lost", System.Text.RegularExpressions.RegexOptions.IgnoreCase))
            {
                _deviceLost = true;
                Console.WriteLine("[gpu] a run failed on a lost WebGPU device: the next run opens a new one");
                return;
            }
    }

    void LoseDeviceForTest()
    {
        if (_runtime is { IsCompletedSuccessfully: true } task && task.Result.Accelerator is SpawnDev.ILGPU.WebGPU.WebGPUAccelerator gpu)
        {
            Console.WriteLine("[gpu] test hook: destroying the WebGPU device");
            gpu.NativeAccelerator.NativeDevice.Destroy();
        }
    }

    readonly bool _perOp;

    // Debug switch: ?nodetimes=1 prints each node's time when it hands over to the next one, and the run's total. A node that leaves its work on
    // the GPU returns before the GPU is done: that time shows in the next node that waits (the total is exact).
    readonly bool _nodeTimes;
    readonly System.Diagnostics.Stopwatch _runClock = new();
    string? _timedNode;
    double _timedFrom;

    void ReportNodeTimes(string eventType, JsonNode? data)
    {
        double now = _runClock.Elapsed.TotalMilliseconds;
        if (eventType == "execution_start") { _runClock.Restart(); _timedNode = null; return; }
        if (eventType is not ("executing" or "execution_success" or "execution_error")) return;
        if (_timedNode is { } done) Console.WriteLine($"[nodetimes] node {done}: {now - _timedFrom:F0} ms");
        _timedNode = eventType == "executing" ? (string?)data?["node"] : null;
        _timedFrom = now;
        if (eventType != "executing") Console.WriteLine($"[nodetimes] run: {now:F0} ms ({eventType})");
    }

    /// <summary>Done when the examples' Load Image picture is in this page's inputs (or could not be fetched); the input listing waits for it.</summary>
    readonly Task _example;
    readonly Task _hosted;

    /// <summary>
    /// The model files ComfyUI's templates name (model-catalog.json next to the page): the loaders list them, and they come from Hugging Face on
    /// first use (StreamingModelSource: stored in the browser when there is room, streamed when not). Offline, the loaders list what is cached.
    /// </summary>
    async Task RegisterHostedModelsAsync()
    {
        try { Console.WriteLine($"[models] {HostedModels.LoadJson(await _http.GetStringAsync("model-catalog.json"))} hosted model files listed"); }
        catch (Exception ex) when (ex is HttpRequestException or System.Text.Json.JsonException) { Console.WriteLine("[models] model-catalog.json: " + ex.Message); }
    }

    /// <summary>The examples' Load Image pictures, put in the inputs when missing (see <see cref="AddExamplePictureAsync"/>).</summary>
    internal static readonly string[] ExamplePictures = ["example.png", "example_inpaint.png"];

    /// <summary>
    /// The examples' Load Image pictures (ComfyUI ships input/example.png the same way): <see cref="ExamplePictures"/> from the page's own
    /// examples folder, made with FLUX.2 Klein 4B (Apache-2.0; docs/EXAMPLE-PICTURE.md), as this page's inputs. example_inpaint.png is example.png
    /// with the coat transparent, so the Inpainting example has a mask out of the box. Offline or missing, the inputs are simply without them.
    /// </summary>
    async Task AddExamplePictureAsync()
    {
        foreach (var name in ExamplePictures)
        {
            try
            {
                if (_inputs.Exists(name)) continue;
                var bytes = await _http.GetByteArrayAsync("examples/" + name);
                if (!_inputs.Exists(name)) await _inputs.SaveAsync(name, bytes);
            }
            catch (HttpRequestException) { }
        }
    }

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
        if (serviceType == typeof(IPictureEncoder)) return _pictureEncoder ??= new GpuPictureEncoder(this);
        return null;
    }

    GpuPictureEncoder? _pictureEncoder;

    /// <summary>Opens the GPU on first use (the first ML node to run asks for it) and keeps it.</summary>
    public Task<MlRuntime> GetAsync() => _runtime ??= CreateRuntimeAsync();

    /// <summary>
    /// Before each prompt: when the WebGPU device was lost (a GPU reset, a driver timeout, the GPU out of memory), every later dispatch throws, so
    /// the runtime and its models are dropped and the next ML node opens a new device; cached node outputs go too (they may hold models on the
    /// lost device). Found 2026-11-19 on the live site: after one loss every later example failed until the page was reloaded.
    /// </summary>
    async Task ReplaceLostDeviceAsync()
    {
        if (_runtime is not { IsCompletedSuccessfully: true } task) return;
        bool lost = _deviceLost || task.Result.Accelerator is SpawnDev.ILGPU.WebGPU.WebGPUAccelerator { IsDeviceLost: true };
        if (!lost) return;
        _deviceLost = false;
        _runtime = null;
        Console.WriteLine("[gpu] the WebGPU device was lost: opening a new one for this run");
        try { task.Result.Dispose(); } catch (Exception ex) { Console.WriteLine("[gpu] releasing the lost device: " + ex.Message); }
        _queue.ClearResults();
        await Task.CompletedTask;
    }

    async Task<MlRuntime> CreateRuntimeAsync()
    {
        var builder = MLContext.Create();
        await builder.AllAcceleratorsAsync();
        var context = builder.ToContext();
        var accelerator = await context.CreatePreferredAcceleratorAsync()
            ?? throw new InvalidOperationException("This browser has no GPU the engine can use (it needs WebGPU).");
        // Model files stream into the origin-private file system and are reused on the next visit, never held whole in memory.
        // Every model, whatever its size: cached in the browser when there is room, streamed from the model host when not (StreamingModelSource).
        var source = new StreamingModelSource(Hub);
        // Which files stream (no room in the browser's storage) and how far: the run's status shows downloads, not streams.
        source.Streaming += (file, bytes) => Console.WriteLine($"[models] streaming {file} ({bytes / 1e9:F2} GB): no room to store it in this browser");
        source.CacheFallback += file => Console.WriteLine($"[models] the browser refused to store {file}: it streams from now on");
        long lastLogged = 0;
        source.OnProgress += (read, length) =>
        {
            long step = Math.Max(length / 10, 1);
            if (read / step != lastLogged / step || read >= length) { lastLogged = read; Console.WriteLine($"[models] streamed {read / 1e9:F2} of {length / 1e9:F2} GB"); }
        };
        var runtime = new MlRuntime(source, accelerator);
        if (accelerator is SpawnDev.ILGPU.WebGPU.WebGPUAccelerator gpu) _ = WatchDeviceAsync(gpu.NativeAccelerator.NativeDevice, runtime);
        return runtime;
    }

    /// <summary>The device's lost promise: once it resolves, the next run opens a new device (<see cref="ReplaceLostDeviceAsync"/>).</summary>
    async Task WatchDeviceAsync(SpawnDev.SpawnJS.JSObjects.GPUDevice device, MlRuntime runtime)
    {
        try
        {
            var info = await device.Lost;
            if (_runtime is { IsCompletedSuccessfully: true } current && ReferenceEquals(current.Result, runtime)) _deviceLost = true;
            Console.WriteLine($"[gpu] the WebGPU device was lost ({info?.Reason}: {info?.Message}); the next run opens a new one");
        }
        catch (Exception ex) { Console.WriteLine("[gpu] watching the device: " + ex.Message); }
    }

    HubModelSource? _hub;
    /// <summary>The browser's model source, shared by the runtime and the model manager (it can list and delete the cache without opening the GPU).</summary>
    HubModelSource Hub => _hub ??= new HubModelSource(_js, _http);

    public async Task<SpawnGen.Runtime.ModelCacheInfo> GetModelCacheAsync()
    {
        try
        {
            var files = (await Hub.Store.ListAsync()).Select(e => new SpawnGen.Runtime.CachedModelFile(e.Key, e.SizeBytes, e.Complete)).OrderByDescending(f => f.Size).ToList();
            var downloads = Hub.ActiveDownloads.Select(d => new SpawnGen.Runtime.ModelDownload(d.Key, d.BytesReceived, d.TotalBytes, d.BytesPerSecond)).ToList();
            return new(files, files.Sum(f => f.Size), downloads);
        }
        catch (Exception ex) when (ex is InvalidOperationException or Microsoft.JSInterop.JSException or NotSupportedException) { return new([], 0, []); }
    }

    public Task RemoveCachedModelAsync(string key) => Hub.Store.RemoveAsync(key);

    public Task ClearModelCacheAsync() => Hub.Store.ClearAsync();

    // ---- IEngineClient ----

    public Task<QueueResult> QueueAsync(string promptJson, string? workflowJson = null, bool front = false, string? previewMethod = null, IReadOnlyList<string>? targets = null,
        IReadOnlyDictionary<string, string>? settings = null)
    {
        PromptGraph graph;
        try { graph = PromptJson.Parse(promptJson); }
        catch (Exception ex) when (ex is System.Text.Json.JsonException or InvalidDataException or FormatException)
        {
            return Task.FromResult(new QueueResult(false, null, "That prompt cannot be read: " + ex.Message, new Dictionary<int, string>()));
        }
        try
        {
            var item = _queue.Enqueue(graph, "browser", workflowJson is null ? null : new Dictionary<string, string> { ["workflow"] = workflowJson }, front, previewMethod, targets,
                settings is null ? null : new JsonObject { ["spawngen_settings"] = SettingsJson(settings) });
            // Output nodes left out (what they need does not validate) are reported while the rest runs, as on the server.
            return Task.FromResult(new QueueResult(true, item.Id, null, PerNode(item.Skipped.SelectMany(f => f.Result.Errors))));
        }
        catch (PromptValidationException ex)
        {
            var message = ex.Result.Errors is [{ Code: PromptAdmission.NoOutputsCode }] ? "Prompt has no outputs" : "Prompt outputs failed validation";
            return Task.FromResult(new QueueResult(false, null, message, PerNode(ex.Result.Errors)));
        }
    }

    static Dictionary<int, string> PerNode(IEnumerable<ValidationError> errors)
    {
        var perNode = new Dictionary<int, string>();
        foreach (var group in errors.Distinct().Where(e => int.TryParse(e.NodeId, out _)).GroupBy(e => int.Parse(e.NodeId!)))
            perNode[group.Key] = string.Join("; ", group.Select(e => e.Message));
        return perNode;
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

    /// <summary>Run settings as the spawngen_settings object of extra_data.</summary>
    internal static JsonObject SettingsJson(IReadOnlyDictionary<string, string> settings) =>
        new(settings.Select(kv => KeyValuePair.Create(kv.Key, (JsonNode?)JsonValue.Create(kv.Value))));

    readonly System.Threading.Lock _urlGate = new();
    readonly Dictionary<(string Type, string Folder, string Name), (object Data, string Url)> _urls = [];

    /// <summary>
    /// Results live in memory here: the URL is a blob URL of the file's bytes, made once per file (TJ, 2026-11-19: a base64 data URL built in .NET
    /// for every view made the LTX example's WEBP slow to show). A file written again under the same name gets a new URL and the old one is revoked.
    /// "" when there is no such file.
    /// </summary>
    public string ViewUrl(string filename, string? subfolder, string? type)
    {
        var key = (type ?? "output", subfolder ?? "", filename);
        // A result's blob is made from its JS bytes (nothing crosses into .NET); an input file is still a .NET array (the input store is next).
        object? bytes = key.Item1 == "input" ? ReadInput(string.IsNullOrEmpty(subfolder) ? filename : $"{subfolder}/{filename}") : _results.GetStored(key.Item1, filename, key.Item2);
        if (bytes is null) return "";
        lock (_urlGate)
        {
            if (_urls.TryGetValue(key, out var known) && ReferenceEquals(known.Data, bytes)) return known.Url;
            using var blob = bytes switch
            {
                JsBytes js => new Blob(new TypedArray[] { js.Data }, new BlobOptions { Type = MediaType(filename) }),
                StoredBytes stored => new Blob(new[] { stored.Read() }, new BlobOptions { Type = MediaType(filename) }),
                _ => new Blob(new[] { (byte[])bytes }, new BlobOptions { Type = MediaType(filename) }),
            };
            var url = URL.CreateObjectURL(blob);
            if (known.Url is { } stale) URL.RevokeObjectURL(stale);
            _urls[key] = (bytes, url);
            return url;
        }
    }

    /// <summary>
    /// An input file's bytes, or null when there is none: a workflow's example.png this page never had must not throw out of a render (the App
    /// Builder's preview threw FileNotFoundException for it, TJ 2026-11-18).
    /// </summary>
    byte[]? ReadInput(string path)
    {
        if (path.Length == 0 || !_inputs.Exists(path)) return null;
        try { return _inputs.ReadAsync(path).GetAwaiter().GetResult(); }
        catch (Exception ex) when (ex is FileNotFoundException or UnauthorizedAccessException) { return null; } // a folder of that name
    }

    static string MediaType(string filename) => Path.GetExtension(filename).ToLowerInvariant() switch
    {
        ".wav" => "audio/wav", ".flac" => "audio/flac", ".mp3" => "audio/mpeg", ".opus" => "audio/ogg", ".glb" => "model/gltf-binary", ".jpg" or ".jpeg" => "image/jpeg", ".webp" => "image/webp", ".gif" => "image/gif", ".bmp" => "image/bmp",
        _ => "image/png",
    };

    public async Task<string?> UploadImageAsync(string name, byte[] png, string type = "input") => type switch
    {
        "input" => await _inputs.SaveAsync(Path.GetFileName(name), png),
        "output" or "temp" => (await _results.WriteAsync(type, "", Path.GetFileName(name), png)).Filename,
        _ => null,
    };

    public async Task<IReadOnlyList<string>> ListInputImagesAsync()
    {
        await _example;
        return _inputs.List();
    }

    public Task<IReadOnlyList<string>> ListInputFoldersAsync() => Task.FromResult(_inputs.Folders());

    public Task<IReadOnlyList<string>> ListOutputFilesAsync() =>
        Task.FromResult<IReadOnlyList<string>>([.. _results.FilesNewestFirst("output").Select(n => n + " [output]")]);
}
