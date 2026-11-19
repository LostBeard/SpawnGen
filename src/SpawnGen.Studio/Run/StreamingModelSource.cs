using SpawnDev.ILGPU.ML.Hub;

namespace SpawnGen.Studio.Run;

/// <summary>
/// The browser engine's model source: every model runs in the browser, whatever its size. A file already in the browser's cache (OPFS) opens
/// from there; a file the browser has room for downloads into the cache once (as <see cref="HubModelSource"/> alone does); a file it has no room
/// for streams from the model host with HTTP range requests on every load (the bytes go to the GPU without entering the .NET heap). This is the
/// SpawnDev.ILGPU.ML demo's recipe for TRELLIS.2 (Image3DPage: <see cref="RangeStreamModelSource"/> planned against the storage estimate),
/// applied per file to every model, so LTX-2's 27 GB checkpoint or TRELLIS.2's checkpoints need no special case in their nodes.
/// navigator.storage.estimate() can overstate the room (the library measured Chrome writing past its "quota"): a write the browser refuses
/// falls back to streaming that file (<see cref="RangeStreamModelSource.CacheFallback"/>).
/// </summary>
public sealed class StreamingModelSource(HubModelSource hub) : IModelSource
{
    readonly SemaphoreSlim _plan = new(1, 1);

    /// <summary>The hub source underneath (its cache, its downloads list).</summary>
    public HubModelSource Hub => hub;

    /// <summary>(bytes read, file length) while a file streams, for progress.</summary>
    public event Action<long, long>? OnProgress;

    /// <summary>"repo/path" of a file the browser refused to store after the estimate promised room; it streams from now on.</summary>
    public event Action<string>? CacheFallback;

    /// <summary>"repo/path" of each file that streams instead of being cached (no room), as it opens.</summary>
    public event Action<string, long>? Streaming;

    public async Task<Stream> OpenAsync(string repoId, string filePath, CancellationToken cancellationToken = default)
    {
        if (await hub.Store.ExistsAsync(hub.CacheKey(repoId, filePath))) return await hub.OpenAsync(repoId, filePath, cancellationToken);
        var ranged = new RangeStreamModelSource(hub);
        ranged.OnProgress += (read, length) => OnProgress?.Invoke(read, length);
        ranged.CacheFallback += key => CacheFallback?.Invoke(key);
        // One plan at a time (each reads the room left); the download or the stream itself runs outside the lock.
        await _plan.WaitAsync(cancellationToken);
        try
        {
            var (_, streamed) = await ranged.PlanCacheAsync([(repoId, filePath)], await hub.GetStorageEstimateAsync(), cancellationToken: cancellationToken);
            if (streamed > 0) Streaming?.Invoke(repoId + "/" + filePath, streamed);
        }
        finally { _plan.Release(); }
        return await ranged.OpenAsync(repoId, filePath, cancellationToken);
    }

    public Task<byte[]> FetchBytesAsync(string repoId, string filePath, CancellationToken cancellationToken = default) =>
        hub.FetchBytesAsync(repoId, filePath, cancellationToken);
}
