namespace SpawnGen.Runtime;

/// <summary>One model file kept on the engine's machine (a file in the models folder's hub cache, or an entry of the browser's file system).</summary>
public sealed record CachedModelFile(string Key, long Size, bool Complete);

/// <summary>A model file being downloaded right now.</summary>
public sealed record ModelDownload(string Key, long Received, long Total, double BytesPerSecond)
{
    public double? Fraction => Total > 0 ? Math.Clamp((double)Received / Total, 0, 1) : null;
}

/// <summary>What the model manager shows: the cached files, their total size and the downloads in progress.</summary>
public sealed record ModelCacheInfo(IReadOnlyList<CachedModelFile> Files, long TotalBytes, IReadOnlyList<ModelDownload> Downloads);
