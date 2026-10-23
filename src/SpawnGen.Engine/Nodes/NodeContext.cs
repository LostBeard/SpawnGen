namespace SpawnGen.Engine.Nodes;

/// <summary>Injected into <c>Execute</c> methods that declare a <see cref="NodeContext"/> parameter.</summary>
public sealed class NodeContext
{
    readonly Action<double, double, string?>? _progress;
    readonly Action<byte[]>? _preview;

    readonly Lazy<string>? _promptJson;
    readonly Func<string, bool>? _isConnected;

    internal NodeContext(string nodeId, string classType, IServiceProvider services, CancellationToken cancellationToken,
        Action<double, double, string?>? progress, Lazy<string>? promptJson = null, IReadOnlyDictionary<string, string>? extraPngInfo = null, Action<byte[]>? preview = null,
        Func<string, bool>? isConnected = null, int listIndex = 0)
    {
        _isConnected = isConnected;
        ListIndex = listIndex;
        NodeId = nodeId;
        ClassType = classType;
        Services = services;
        CancellationToken = cancellationToken;
        _progress = progress;
        _preview = preview;
        _promptJson = promptJson;
        ExtraPngInfo = extraPngInfo ?? new Dictionary<string, string>();
    }

    /// <summary>The whole prompt being run, as ComfyUI API-format JSON (for nodes that record where a result came from).</summary>
    public string PromptJson => _promptJson?.Value ?? "{}";

    /// <summary>Extra text to embed in saved PNGs by chunk name (the editor's workflow, so a picture can be dropped back on the canvas).</summary>
    public IReadOnlyDictionary<string, string> ExtraPngInfo { get; }

    public string NodeId { get; }

    /// <summary>True when the input is linked or given a value in the prompt. A lazy input that is connected but not evaluated yet is null and still connected (ComfyUI tells MISSING from None the same way).</summary>
    public bool IsConnected(string input) => _isConnected?.Invoke(input) ?? false;

    /// <summary>Which item of its list inputs this call works on (0 for a node that runs once).</summary>
    public int ListIndex { get; }
    public string ClassType { get; }
    public IServiceProvider Services { get; }
    public CancellationToken CancellationToken { get; }

    /// <summary>Reports <paramref name="value"/> of <paramref name="max"/> units done (e.g. sampler steps).</summary>
    public void Progress(double value, double max, string? message = null) => _progress?.Invoke(value, max, message);

    /// <summary>True when something shows previews, so a node can skip making them when nobody looks.</summary>
    public bool WantsPreview => _preview is not null;

    /// <summary>Shows a picture (PNG bytes) in the node while it works, e.g. a sampler's current guess; each call replaces the last.</summary>
    public void Preview(byte[] png) => _preview?.Invoke(png);

    /// <summary>Reports a status message with no fraction (e.g. "Downloading model").</summary>
    public void Status(string message) => _progress?.Invoke(0, 0, message);
}

/// <summary>
/// Preview data an output node sends to the editor (e.g. <c>["images"]</c>). It is not a socket; it travels
/// next to the node's outputs and is cached with them.
/// </summary>
public sealed class UiOutput : Dictionary<string, object?>
{
    public UiOutput() { }

    public UiOutput(string key, object? value) => this[key] = value;
}
