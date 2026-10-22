using SpawnDev.ILGPU.ML.Hub;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// Desktop model source over a local folder. Without <paramref name="repoFolders"/> the repo id is ignored and a file path is read relative to
/// the folder (the layout of schmuell/sd-turbo-ort-web). With it, a repo id found in the map selects a sub-folder, so one models folder can hold
/// several checkpoints (<c>models/sd_turbo/unet/model.onnx</c>). Streams are seekable so the ONNX reader can skip weight blobs.
/// SpawnDev.ILGPU.ML master has an equivalent <c>LocalFolderModelSource</c>; swap to it once a NuGet carries it.
/// </summary>
public sealed class FolderModelSource(string root, IReadOnlyDictionary<string, string>? repoFolders = null) : IModelSource
{
    string Resolve(string repoId, string filePath)
    {
        var folder = repoFolders is not null && repoFolders.TryGetValue(repoId, out var f) ? Path.Combine(root, f) : root;
        return Path.Combine(folder, filePath.Replace('/', Path.DirectorySeparatorChar));
    }

    public Task<Stream> OpenAsync(string repoId, string filePath, CancellationToken cancellationToken = default) =>
        Task.FromResult<Stream>(new FileStream(Resolve(repoId, filePath), FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16,
            FileOptions.Asynchronous | FileOptions.RandomAccess));

    public Task<byte[]> FetchBytesAsync(string repoId, string filePath, CancellationToken cancellationToken = default) =>
        File.ReadAllBytesAsync(Resolve(repoId, filePath), cancellationToken);

    public bool Exists(string repoId, string filePath) => File.Exists(Resolve(repoId, filePath));
}
