using SpawnDev.ILGPU.ML.Hub;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// The desktop's model source: the checkpoints SpawnGen knows live in sub-folders of the models folder (as before); any other model a node asks for
/// (upscalers, depth, background removal...) is downloaded from Hugging Face on first use into <c>&lt;models&gt;/hub</c> and read from there afterwards.
/// </summary>
public sealed class DesktopModelSource : IModelSource
{
    readonly FolderModelSource _local;
    readonly IReadOnlyDictionary<string, string> _localRepos;

    public DesktopModelSource(string modelsFolder, IReadOnlyDictionary<string, string> localRepos, HttpClient? http = null)
    {
        _local = new FolderModelSource(modelsFolder, localRepos);
        _localRepos = localRepos;
        // A model download can take minutes: no overall timeout (the transfer resumes if it is interrupted).
        Remote = new HttpClientModelSource(http ?? new HttpClient { Timeout = Timeout.InfiniteTimeSpan }, new FileModelStore(Path.Combine(modelsFolder, "hub")));
    }

    /// <summary>The caching downloader behind every repo that is not a local checkpoint (progress, cache listing).</summary>
    public HttpClientModelSource Remote { get; }

    IModelSource For(string repoId) => _localRepos.ContainsKey(repoId) ? _local : Remote;

    public Task<Stream> OpenAsync(string repoId, string filePath, CancellationToken cancellationToken = default) =>
        For(repoId).OpenAsync(repoId, filePath, cancellationToken);

    public Task<byte[]> FetchBytesAsync(string repoId, string filePath, CancellationToken cancellationToken = default) =>
        For(repoId).FetchBytesAsync(repoId, filePath, cancellationToken);
}
