using SpawnDev.ILGPU.ML.Hub;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// Desktop model source over a local folder: <c>repoId</c> is ignored and <c>filePath</c> is read relative to the folder
/// (the layout of schmuell/sd-turbo-ort-web). Streams are seekable so the ONNX reader can skip weight blobs.
/// SpawnDev.ILGPU.ML master has an equivalent <c>LocalFolderModelSource</c>; swap to it once a NuGet carries it.
/// </summary>
public sealed class FolderModelSource(string root) : IModelSource
{
    string Resolve(string filePath) => Path.Combine(root, filePath.Replace('/', Path.DirectorySeparatorChar));

    public Task<Stream> OpenAsync(string repoId, string filePath, CancellationToken cancellationToken = default) =>
        Task.FromResult<Stream>(new FileStream(Resolve(filePath), FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16,
            FileOptions.Asynchronous | FileOptions.RandomAccess));

    public Task<byte[]> FetchBytesAsync(string repoId, string filePath, CancellationToken cancellationToken = default) =>
        File.ReadAllBytesAsync(Resolve(filePath), cancellationToken);

    public bool Exists(string filePath) => File.Exists(Resolve(filePath));
}
