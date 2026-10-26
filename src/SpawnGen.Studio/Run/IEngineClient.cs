using System.Text.Json.Nodes;

namespace SpawnGen.Studio.Run;

/// <summary>The result of asking an engine to run a prompt: queued, or refused with a message and per-node reasons.</summary>
public sealed record QueueResult(bool Queued, string? PromptId, string? Message, IReadOnlyDictionary<int, string> NodeErrors);

/// <summary>
/// Where graphs run: this browser (WebGPU) or a SpawnGen server (CUDA/OpenCL on the desktop). The Studio drives both the same way and gets the same
/// events (ComfyUI's: execution_start, executing, progress, executed, execution_success ...).
/// </summary>
public interface IEngineClient
{
    string Name { get; }

    /// <summary>Raised for every run event: <c>{"type": ..., "data": {...}}</c>.</summary>
    event Action<JsonObject>? EventReceived;

    /// <param name="workflowJson">The editor workflow (ComfyUI format); Save Image embeds it in each picture so the picture can be dropped back on the canvas.</param>
    Task<QueueResult> QueueAsync(string promptJson, string? workflowJson = null, bool front = false);
    Task InterruptAsync();

    /// <summary>Removes a run that is still waiting in the queue (not the one that is running: that is <see cref="InterruptAsync"/>).</summary>
    Task CancelQueuedAsync(string promptId);

    /// <summary>A URL the canvas can load a result picture from.</summary>
    string ViewUrl(string filename, string? subfolder, string? type);

    /// <summary>Makes a picture available to Load Image nodes. Returns the name it is stored under.</summary>
    Task<string?> UploadImageAsync(string name, byte[] png);

    /// <summary>The model files this engine has downloaded and the downloads in progress (the model manager).</summary>
    Task<SpawnGen.Runtime.ModelCacheInfo> GetModelCacheAsync();
    Task RemoveCachedModelAsync(string key);
    Task ClearModelCacheAsync();

    /// <summary>The picture names a Load Image node can pick from on this engine.</summary>
    Task<IReadOnlyList<string>> ListInputImagesAsync();

    /// <summary>The input folders a Load Image (from Folder) node can pick from on this engine.</summary>
    Task<IReadOnlyList<string>> ListInputFoldersAsync();
}
