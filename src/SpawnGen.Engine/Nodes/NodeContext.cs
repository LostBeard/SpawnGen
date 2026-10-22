namespace SpawnGen.Engine.Nodes;

/// <summary>Injected into <c>Execute</c> methods that declare a <see cref="NodeContext"/> parameter.</summary>
public sealed class NodeContext
{
    readonly Action<double, double, string?>? _progress;

    internal NodeContext(string nodeId, string classType, IServiceProvider services, CancellationToken cancellationToken,
        Action<double, double, string?>? progress)
    {
        NodeId = nodeId;
        ClassType = classType;
        Services = services;
        CancellationToken = cancellationToken;
        _progress = progress;
    }

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
