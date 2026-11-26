using SpawnGen.Engine.Execution;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.Pipelines;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// Owns the GPU accelerator and the model source for the ML nodes, and caches loaded checkpoints by name so
/// the engine's node cache can hand the same handles to several runs. Register one instance as a service.
/// </summary>
public sealed class MlRuntime : IDisposable
{
    /// <summary>Checkpoint names the loader offers, mapped to the repo id on the model source.</summary>
    public static IReadOnlyDictionary<string, string> KnownCheckpoints => CheckpointCatalog.Repos;

    /// <summary>
    /// What differs between checkpoints. An ONNX export: the width of the text encoder's output (the UNet's cross-attention input) and whether the UNet's
    /// weights live in a separate file. A single-file checkpoint: <see cref="File"/>, the <c>.safetensors</c> file in the repo (its layout is read from the tensors).
    /// </summary>
    public sealed record CheckpointInfo(int TextWidth, string? UnetExternalData = null, string? File = null);

    public static IReadOnlyDictionary<string, CheckpointInfo> CheckpointDetails => CheckpointCatalog.Details;

    /// <summary>Repo id to folder name, for <see cref="FolderModelSource"/>: a models folder holds one sub-folder per checkpoint name.</summary>
    public static IReadOnlyDictionary<string, string> RepoFolders => CheckpointCatalog.RepoFolders;

    readonly Context _context;
    readonly Dictionary<string, Task<Checkpoint>> _checkpoints = new();
    readonly Dictionary<string, Loaded> _loaded = new();

    /// <summary>A model of <see cref="GetOrLoadAsync"/>: whether it may be freed between prompts, and the prompt that last used it.</summary>
    sealed class Loaded(Task<IDisposable> task, bool pipeline)
    {
        public Task<IDisposable> Task { get; } = task;
        public bool Pipeline { get; } = pipeline;
        public long Prompt { get; set; } = PromptScope.Current;
    }
    readonly object _gate = new();

    public MlRuntime(IModelSource modelSource, Accelerator? accelerator = null)
    {
        _reportsFreeMemory = new Lazy<bool>(() => Accelerator!.TryGetFreeMemory(out _));
        ModelSource = modelSource;
        if (accelerator != null)
        {
            Accelerator = accelerator;
            _context = accelerator.Context;
            OwnsAccelerator = false;
        }
        else
        {
            _context = MLContext.Create().ToContext();
            Accelerator = CreateBestAccelerator(_context);
            OwnsAccelerator = true;
        }
    }

    public IModelSource ModelSource { get; }
    public Accelerator Accelerator { get; }
    bool OwnsAccelerator { get; }

    // ---- models in device memory: kept while they fit, dropped on need (SpawnDev.ILGPU.ML's ModelResidency) ----

    SpawnDev.ILGPU.ML.ModelResidency? _residency;
    readonly Lazy<bool> _reportsFreeMemory;
    readonly List<IAsyncDisposable> _leases = [];
    readonly HashSet<string> _leased = new(StringComparer.Ordinal);
    /// <summary>The running prompt's leased models (by reference) and how each asks the manager for room before a lazy load.</summary>
    readonly Dictionary<object, (string Key, Func<long, Task> Ensure)> _room = new(ReferenceEqualityComparer.Instance);
    long? _modelMemoryBytes;

    /// <summary>
    /// The library's model memory manager for this runtime's accelerator: the models SpawnGen loads are kept on the device while they fit
    /// the budget and the least recently used idle one is dropped when the next would not fit. Created on first use.
    /// </summary>
    public SpawnDev.ILGPU.ML.ModelResidency Residency
    {
        get
        {
            lock (_gate)
            {
                if (_residency is not null) return _residency;
                var residency = new SpawnDev.ILGPU.ML.ModelResidency(Accelerator) { BudgetBytes = _modelMemoryBytes, Log = line => Console.WriteLine("[models] " + line) };
                residency.Evicted += e =>
                {
                    Console.WriteLine($"[models] dropped {e.Key} ({e.Bytes / 1e9:0.##} GB): {e.Reason}{(e.ForKey is { } f ? " for " + f : "")}");
                    ModelDropped?.Invoke(e.Key);
                };
                return _residency = residency;
            }
        }
    }

    /// <summary>
    /// The most device memory the kept models may hold (the user's setting); null = none set. Where the device reports its free memory
    /// (CUDA, AMD OpenCL) that limits too. In a browser nothing reports memory, so without this number the runtime keeps its old rule
    /// (<see cref="KeepsModelsByBudget"/> false: one workflow's models at a time).
    /// </summary>
    public long? ModelMemoryBytes
    {
        get => _modelMemoryBytes;
        set
        {
            lock (_gate) { _modelMemoryBytes = value; if (_residency is not null) _residency.BudgetBytes = value; }
        }
    }

    /// <summary>
    /// True when models are kept by the manager (a budget is set, or the device reports its free memory): every model is then acquired
    /// with a lease that lasts until <see cref="EndPromptAsync"/>, and the engine must let loader nodes run again each prompt
    /// (PromptExecutor.ReacquireModels). False: models are kept by the older rules (freed when an unrelated workflow starts).
    /// </summary>
    public bool KeepsModelsByBudget => _modelMemoryBytes is not null || _reportsFreeMemory.Value;

    /// <summary>
    /// The sizes the manager has measured (key -> the largest bytes seen), as JSON, for an engine to keep across sessions: a model whose
    /// size is known keeps the others loaded on its first load in a new session, where an unknown one is taken to be as large as the
    /// largest measured (or, with nothing measured, makes every idle model go first).
    /// </summary>
    public string ModelSizesJson() =>
        System.Text.Json.JsonSerializer.Serialize(Residency.KnownSizes.OrderBy(kv => kv.Key, StringComparer.Ordinal).ToDictionary(kv => kv.Key, kv => kv.Value));

    /// <summary>Gives the manager sizes kept from an earlier session (<see cref="ModelSizesJson"/>); text that is not such JSON is ignored.</summary>
    public int SeedModelSizes(string? json)
    {
        if (string.IsNullOrWhiteSpace(json)) return 0;
        try
        {
            var sizes = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, long>>(json)?.Where(kv => kv.Value > 0).ToList() ?? [];
            if (sizes.Count > 0) Residency.SeedKnownSizes(sizes);
            return sizes.Count;
        }
        catch (System.Text.Json.JsonException) { return 0; }
    }

    /// <summary>
    /// Whose the running prompt's models are (the Studio's workflow tab): each model acquired while it is set carries this tag, and
    /// <see cref="ReleaseWorkflowAsync"/> drops the idle models no other workflow uses. The engine sets it at the start of each prompt.
    /// </summary>
    public string? PromptTag { get; set; }

    /// <summary>
    /// A workflow was closed: its idle models that no other workflow used are dropped now (one still in use goes when its prompt
    /// ends). Without a budget nothing is tagged and nothing happens here: the older rules free models. Returns how many were dropped.
    /// </summary>
    public async Task<int> ReleaseWorkflowAsync(string workflow) =>
        KeepsModelsByBudget && _residency is { } residency && !string.IsNullOrEmpty(workflow) ? await residency.ReleaseAsync(workflow) : 0;

    /// <summary>A model the manager dropped (its key): node outputs that hold it must not be reused (the engine drops its cached model results).</summary>
    public event Action<string>? ModelDropped;

    /// <summary>
    /// The model for <paramref name="key"/> from the manager, leased for the running prompt (one lease per key and prompt). The manager
    /// measures what a model holds on the device by itself (its own size where the model states one, else what was allocated over its
    /// load). <paramref name="growsInUse"/>: the model loads most of what it holds during its first run, not in <paramref name="load"/>
    /// (every pipeline a node asks for when it runs: FLUX.2 and FLUX 3 Klein, LTX-Video, TRELLIS.2, ...): room is asked for right
    /// after the acquire, which is also the point the manager measures such a model from, to the end of its lease. Found 2026-11-25 on
    /// the RX 590: measured over their load only, the FLUX pipelines counted as 0.2 GB each, FLUX 3 was loaded beside a FLUX.2 that
    /// really held several GB, and the device was lost.
    /// </summary>
    async Task<T> AcquireAsync<T>(string key, Func<Task<T>> load, long expectedBytes = 0, bool growsInUse = false) where T : class, IDisposable
    {
        var residency = Residency;
        bool kept = residency.Resident.Any(m => m.Key == key);
        var lease = await residency.AcquireAsync(key, load, expectedBytes, PromptTag);
        var model = lease.Model;
        bool extra;
        lock (_gate)
        {
            extra = !_leased.Add(key);
            if (!extra) { _leases.Add(lease); _room[model] = (key, lease.EnsureRoomAsync); }
            if (!kept) _loadedSinceWarmUp = true;
        }
        if (extra) { await lease.DisposeAsync(); return model; }   // this prompt already holds a lease on it
        // no size given = room for what this key was measured at before, less what the model holds now: a kept, loaded pipeline asks
        // for nothing, a never-measured one gets the manager's unknown-size rule
        if (growsInUse) await lease.EnsureRoomAsync(expectedBytes);
        return model;
    }

    /// <summary>
    /// Called right before a model loads weights the loader node did not load (the lazy MODEL / CLIP / VAE objects, and nodes that open a
    /// model for one call). Without a memory budget this is the old fixed rule: the big transformers take turns (one on the device at a
    /// time, sized for an 8 GB card). With a budget, room is made by need: the manager drops idle scratch and the least recently used
    /// idle models for <paramref name="bytes"/> (unknown = as large as the largest model measured so far), and only if the device then
    /// still lacks that room do the transformers give way as before. So a device with room keeps every transformer loaded.
    /// </summary>
    /// <param name="requester">The model object that is about to load (as handed out by <see cref="GetOrLoadAsync{T}(string, Func{Task{T}}, long)"/>), or null.</param>
    public async Task MakeRoomAsync(object? requester = null, long bytes = 0)
    {
        if (!KeepsModelsByBudget) { TransformerTurns.ReleaseTransformers(); return; }
        var residency = Residency;
        (string Key, Func<long, Task> Ensure) room = default;
        if (requester is not null) lock (_gate) _room.TryGetValue(requester, out room);
        if (room.Ensure is not null) await room.Ensure(bytes);   // 0 = the manager's own figure for this model
        // what is about to load, as far as anyone knows: the caller's figure, the size this model measured before, else what the
        // manager takes an unknown model to be
        var known = residency.KnownSizes;
        long need = bytes > 0 ? bytes
            : room.Key is not null && known.TryGetValue(room.Key, out long measured) && measured > 0 ? measured
            : residency.UnknownSizeBytes ?? known.Values.DefaultIfEmpty(0).Max();
        if (need == 0 || !Accelerator.TryGetFreeMemory(out long free) || free < need)
        {
            if (TransformerTurns.ReleaseTransformers() > 0)
                Console.WriteLine($"[models] the loaded transformer gave way: {(need == 0 ? "the size of what loads next is unknown" : $"{need / 1e9:0.##} GB is needed")}");
        }
    }

    /// <summary>Ends the running prompt's leases: its models stay on the device, and from now on may be dropped when another needs the room.</summary>
    public async Task EndPromptAsync()
    {
        IAsyncDisposable[] leases;
        lock (_gate) { leases = [.. _leases]; _leases.Clear(); _leased.Clear(); _room.Clear(); }
        foreach (var lease in leases)
        {
            try { await lease.DisposeAsync(); }
            catch (Exception ex) { Console.WriteLine("[models] ending a lease: " + ex.Message); }
        }
    }

    /// <summary>The device is gone (a lost WebGPU device): the manager forgets its models without touching them.</summary>
    public void DeviceLost()
    {
        lock (_gate) { _leases.Clear(); _leased.Clear(); _room.Clear(); }
        _residency?.Invalidate();
    }

    /// <summary>Loads (or returns the already loading/loaded) checkpoint registered under <paramref name="name"/>.</summary>
    public Task<Checkpoint> GetCheckpointAsync(string name, Action<string, int>? onProgress = null)
    {
        if (!KnownCheckpoints.TryGetValue(name, out var repoId))
            throw new ArgumentException($"Unknown checkpoint '{name}'. Known: {string.Join(", ", KnownCheckpoints.Keys)}.", nameof(name));
        if (KeepsModelsByBudget) return AcquireAsync("checkpoint:" + name, () => Checkpoint.LoadAsync(this, name, repoId, CheckpointDetails[name], onProgress));
        lock (_gate)
        {
            // A failed load (a missing file, the GPU out of memory) is not kept: the next run tries again.
            if (!_checkpoints.TryGetValue(name, out var task) || task.IsFaulted || task.IsCanceled)
            {
                _checkpoints[name] = task = Checkpoint.LoadAsync(this, name, repoId, CheckpointDetails[name], onProgress);
                _loadedSinceWarmUp = true;
            }
            return task;
        }
    }

    /// <summary>
    /// A loaded model (a pipeline wrapping an inference session) shared by every node that asks for the same <paramref name="key"/>: it is created once,
    /// on first use, and disposed with the runtime. A failed load is forgotten so the next run tries again.
    /// </summary>
    /// <param name="expectedBytes">What the model will hold on the device, when the caller knows (its weight files' size is a fair figure for
    /// safetensors): with a memory budget the manager makes exactly that much room before the load. 0 = unknown: the size measured the
    /// last time this key was loaded is used, and with none every idle model is dropped first (on a small card the load itself is the
    /// over-allocation), so a known size is what lets other workflows' models stay.</param>
    public Task<T> GetOrLoadAsync<T>(string key, Func<Task<T>> load, long expectedBytes = 0) where T : class, IDisposable => GetOrLoadAsync(key, load, pipeline: false, expectedBytes);

    /// <summary>
    /// <see cref="GetOrLoadAsync{T}(string, Func{Task{T}})"/> for a model only its own node uses during its run (a pipeline the node asks for every
    /// time it runs; never one handed to other nodes as MODEL / CLIP / VAE, which the executor's cache may hold). Before anything new loads, the
    /// pipelines the current prompt has not used are freed: a page that ran FLUX.2 Klein, then opens FLUX 3 Klein, does not hold both on an 8 GB
    /// GPU (found 2026-11-19: running every example in a row lost the WebGPU device at FLUX 3 Klein, which runs alone).
    /// </summary>
    public Task<T> GetOrLoadPipelineAsync<T>(string key, Func<Task<T>> load, long expectedBytes = 0) where T : class, IDisposable => GetOrLoadAsync(key, load, pipeline: true, expectedBytes);

    async Task<T> GetOrLoadAsync<T>(string key, Func<Task<T>> load, bool pipeline, long expectedBytes) where T : class, IDisposable
    {
        if (KeepsModelsByBudget) return await AcquireAsync(key, load, expectedBytes, growsInUse: pipeline);
        Task<IDisposable> task;
        List<(string Key, IDisposable Model)> freed = [];
        lock (_gate)
        {
            long prompt = PromptScope.Current;
            if (_loaded.TryGetValue(key, out var entry)) { entry.Prompt = prompt; task = entry.Task; }
            else
            {
                if (prompt != 0)
                    foreach (var (k, e) in _loaded.ToArray())
                        if (e.Pipeline && e.Prompt != prompt && e.Task.IsCompletedSuccessfully) { _loaded.Remove(k); freed.Add((k, e.Task.Result)); }
                _loaded[key] = new Loaded(task = Task.Run<IDisposable>(async () => await load()), pipeline);
                _loadedSinceWarmUp = true;
            }
        }
        foreach (var (k, model) in freed)
        {
            Console.WriteLine($"[models] freed {k} (not used by this prompt) before loading {key}");
            model.Dispose();
            await ReportHeapAsync($"after unloading {k}");
        }
        try { return (T)await task; }
        catch
        {
            lock (_gate) { if (_loaded.TryGetValue(key, out var current) && ReferenceEquals(current.Task, task)) _loaded.Remove(key); }
            throw;
        }
    }

    /// <summary>
    /// Frees every model handed out as MODEL / CLIP / VAE and the like (checkpoints, split files, the models of <see cref="GetOrLoadAsync{T}(string, Func{Task{T}})"/>);
    /// the engine calls it at the start of a prompt that reuses none of their node outputs (PromptExecutor.UnrelatedPromptStarting), so
    /// nothing refers to them any more. Each loads again on its next use. Per-node pipelines keep their own rule (freed before something
    /// new loads once a prompt does not use them): a pipeline node's output is plain data, so the same workflow queued again must keep it.
    /// Models still loading are left alone. Returns how many were freed.
    /// </summary>
    public int FreeHandedOutModels()
    {
        List<(string Key, IDisposable Model)> freed = [];
        lock (_gate)
        {
            foreach (var (k, t) in _checkpoints.ToArray())
                if (t.IsCompletedSuccessfully) { _checkpoints.Remove(k); freed.Add(("checkpoint:" + k, t.Result)); }
                else if (t.IsFaulted || t.IsCanceled) _checkpoints.Remove(k);
            foreach (var (k, e) in _loaded.ToArray())
                if (!e.Pipeline && e.Task.IsCompletedSuccessfully) { _loaded.Remove(k); freed.Add((k, e.Task.Result)); }
        }
        foreach (var (k, model) in freed)
        {
            Console.WriteLine($"[models] freed {k}: the new workflow uses none of the earlier one's models");
            try { model.Dispose(); } catch (Exception ex) { Console.WriteLine($"[models] freeing {k}: {ex.Message}"); }
        }
        return freed.Count;
    }

    bool _loadedSinceWarmUp;

    /// <summary>Measuring switch (the browser's ?nodetimes=1): print each warm-up release and the .NET heap at the points <see cref="ReportHeapAsync"/> is called.</summary>
    public static bool ReportWarmUps { get; set; }

    /// <summary>
    /// Prints the collected .NET heap and the library's compiler-cache release count (only with <see cref="ReportWarmUps"/>). Measured the way
    /// SpawnDev.ILGPU.ML's tests do: from its own async method after a short wait, so no caller's locals or stack keep freed objects reachable
    /// (the browser runtime scans the stack conservatively: a collect right beside the release showed no change), then collect, run
    /// finalizers, collect.
    /// </summary>
    public static async Task ReportHeapAsync(string label)
    {
        if (!ReportWarmUps) return;
        await Task.Delay(50);
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        Console.WriteLine($"[heap] {label}: {GC.GetTotalMemory(false) / 1048576.0:F1} MiB; library releases {CompilerCachePolicy.ReleaseCount}");
    }

    /// <summary>How many times <see cref="PromptSucceeded"/> has told the library that newly loaded models are warm (the tests read it).</summary>
    public int WarmUps { get; private set; }

    /// <summary>
    /// Called by the engine after every prompt that ran to the end. If the prompt (or a failed one before it) loaded a model, every model it
    /// used has now completed a full run, so SpawnDev.ILGPU.ML is told once (<see cref="CompilerCachePolicy.ModelWarmedUp"/>): in the browser it
    /// frees ILGPU's compiler caches (6.5-16 MiB of .NET heap per distinct model otherwise stays after the model is gone); on the desktop the
    /// policy's default does nothing. The library does this itself for its whole pipelines and ONNX sessions, but the models SpawnGen builds
    /// from components (SD checkpoints and split files, Z-Image, Qwen-Image, FLUX.1, ACE-Step) have no library entry that knows when the
    /// whole model is warm. Not called per run: a kernel compiled for the first time after a release pays ~5-15 ms to re-import shared code,
    /// so a prompt that loaded nothing new does nothing here. Returns whether the library was told.
    /// </summary>
    public bool PromptSucceeded()
    {
        lock (_gate)
        {
            if (!_loadedSinceWarmUp) return false;
            _loadedSinceWarmUp = false;
            WarmUps++;
        }
        bool released = CompilerCachePolicy.ModelWarmedUp(Accelerator);
        if (ReportWarmUps) Console.WriteLine($"[warmup] released {released}; library releases so far {CompilerCachePolicy.ReleaseCount}");
        return true;
    }

    /// <summary>CUDA first, then OpenCL, then the CPU accelerator as a last resort (it is far too slow for diffusion).</summary>
    static Accelerator CreateBestAccelerator(Context context)
    {
        var cuda = context.GetCudaDevices();
        if (cuda.Count > 0) return cuda[0].CreateAccelerator(context);
        var cl = context.GetCLDevices();
        if (cl.Count > 0) return cl[0].CreateAccelerator(context);
        return MLContext.CreateCpuAccelerator(context);
    }

    public void Dispose()
    {
        Task<Checkpoint>[] loaded;
        lock (_gate) { loaded = _checkpoints.Values.ToArray(); _checkpoints.Clear(); }
        foreach (var t in loaded)
            if (t.IsCompletedSuccessfully) t.Result.Dispose();
        Task<IDisposable>[] models;
        lock (_gate) { models = [.. _loaded.Values.Select(e => e.Task)]; _loaded.Clear(); }
        foreach (var t in models)
            if (t.IsCompletedSuccessfully) t.Result.Dispose();
        if (_residency is { } residency)
        {
            lock (_gate) { _leases.Clear(); _leased.Clear(); }
            // The manager drains the device before it frees its models. A browser cannot wait for that here (no blocking on its one
            // thread): there a runtime is only disposed for a lost device (DeviceLost was called) or as the page goes.
            var dropping = residency.DisposeAsync();
            if (!OperatingSystem.IsBrowser()) dropping.AsTask().GetAwaiter().GetResult();
        }
        if (OwnsAccelerator)
        {
            Accelerator.Dispose();
            _context.Dispose();
        }
    }
}
