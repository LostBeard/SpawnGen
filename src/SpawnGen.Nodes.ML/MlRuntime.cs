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
    public static readonly IReadOnlyDictionary<string, string> KnownCheckpoints = new Dictionary<string, string>
    {
        ["sd_turbo"] = "schmuell/sd-turbo-ort-web",
    };

    /// <summary>Repo id to folder name, for <see cref="FolderModelSource"/>: a models folder holds one sub-folder per checkpoint name.</summary>
    public static IReadOnlyDictionary<string, string> RepoFolders { get; } = KnownCheckpoints.ToDictionary(kv => kv.Value, kv => kv.Key);

    readonly Context _context;
    readonly Dictionary<string, Task<Checkpoint>> _checkpoints = new();
    readonly Dictionary<string, Task<IDisposable>> _loaded = new();
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
            if (!_checkpoints.TryGetValue(name, out var task))
                _checkpoints[name] = task = Checkpoint.LoadAsync(this, name, repoId, onProgress);
            return task;
        }
    }

    /// <summary>
    /// A loaded model (a pipeline wrapping an inference session) shared by every node that asks for the same <paramref name="key"/>: it is created once,
    /// on first use, and disposed with the runtime. A failed load is forgotten so the next run tries again.
    /// </summary>
    public async Task<T> GetOrLoadAsync<T>(string key, Func<Task<T>> load) where T : class, IDisposable
    {
        Task<IDisposable> task;
        lock (_gate)
        {
            if (!_loaded.TryGetValue(key, out task!))
                _loaded[key] = task = Task.Run<IDisposable>(async () => await load());
        }
        try { return (T)await task; }
        catch
        {
            lock (_gate) { if (_loaded.TryGetValue(key, out var current) && ReferenceEquals(current, task)) _loaded.Remove(key); }
            throw;
        }
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
        lock (_gate) { models = _loaded.Values.ToArray(); _loaded.Clear(); }
        foreach (var t in models)
            if (t.IsCompletedSuccessfully) t.Result.Dispose();
        if (OwnsAccelerator)
        {
            Accelerator.Dispose();
            _context.Dispose();
        }
    }
}
