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

    /// <summary>Loads (or returns the already loading/loaded) checkpoint registered under <paramref name="name"/>.</summary>
    public Task<Checkpoint> GetCheckpointAsync(string name, Action<string, int>? onProgress = null)
    {
        if (!KnownCheckpoints.TryGetValue(name, out var repoId))
            throw new ArgumentException($"Unknown checkpoint '{name}'. Known: {string.Join(", ", KnownCheckpoints.Keys)}.", nameof(name));
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
    public Task<T> GetOrLoadAsync<T>(string key, Func<Task<T>> load) where T : class, IDisposable => GetOrLoadAsync(key, load, pipeline: false);

    /// <summary>
    /// <see cref="GetOrLoadAsync{T}(string, Func{Task{T}})"/> for a model only its own node uses during its run (a pipeline the node asks for every
    /// time it runs; never one handed to other nodes as MODEL / CLIP / VAE, which the executor's cache may hold). Before anything new loads, the
    /// pipelines the current prompt has not used are freed: a page that ran FLUX.2 Klein, then opens FLUX 3 Klein, does not hold both on an 8 GB
    /// GPU (found 2026-11-19: running every example in a row lost the WebGPU device at FLUX 3 Klein, which runs alone).
    /// </summary>
    public Task<T> GetOrLoadPipelineAsync<T>(string key, Func<Task<T>> load) where T : class, IDisposable => GetOrLoadAsync(key, load, pipeline: true);

    async Task<T> GetOrLoadAsync<T>(string key, Func<Task<T>> load, bool pipeline) where T : class, IDisposable
    {
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
        if (OwnsAccelerator)
        {
            Accelerator.Dispose();
            _context.Dispose();
        }
    }
}
