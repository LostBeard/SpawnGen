namespace SpawnGen.Engine.Nodes;

/// <summary>Injected into <c>Execute</c> methods that declare a <see cref="NodeContext"/> parameter.</summary>
public sealed class NodeContext
{
    readonly Action<double, double, string?>? _progress;

    readonly Lazy<string>? _promptJson;

    internal NodeContext(string nodeId, string classType, IServiceProvider services, CancellationToken cancellationToken,
        Action<double, double, string?>? progress, Lazy<string>? promptJson = null, IReadOnlyDictionary<string, string>? extraPngInfo = null)
    {
        NodeId = nodeId;
        ClassType = classType;
        Services = services;
        CancellationToken = cancellationToken;
        _progress = progress;
        _promptJson = promptJson;
        ExtraPngInfo = extraPngInfo ?? new Dictionary<string, string>();
    }

    /// <summary>The whole prompt being run, as ComfyUI API-format JSON (for nodes that record where a result came from).</summary>
    public string PromptJson => _promptJson?.Value ?? "{}";

    /// <summary>Extra text to embed in saved PNGs by chunk name (the editor's workflow, so a picture can be dropped back on the canvas).</summary>
    public IReadOnlyDictionary<string, string> ExtraPngInfo { get; }

    public string NodeId { get; }
    public string ClassType { get; }
    public IServiceProvider Services { get; }
    public CancellationToken CancellationToken { get; }

    /// <summary>Reports <paramref name="value"/> of <paramref name="max"/> units done (e.g. sampler steps).</summary>
    public void Progress(double value, double max, string? message = null) => _progress?.Invoke(value, max, message);

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
