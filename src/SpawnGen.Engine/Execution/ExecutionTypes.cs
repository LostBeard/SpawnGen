using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Execution;

public sealed record ValidationError(string? NodeId, string? Input, string Code, string Message);

public sealed class ValidationResult
{
    public ValidationResult(IReadOnlyList<ValidationError> errors, IReadOnlyList<string> outputNodeIds, IReadOnlyList<string> plan)
    {
        Errors = errors;
        OutputNodeIds = outputNodeIds;
        Plan = plan;
    }

    public IReadOnlyList<ValidationError> Errors { get; }
    /// <summary>The output nodes the run is driven from.</summary>
    public IReadOnlyList<string> OutputNodeIds { get; }
    /// <summary>Needed nodes in dependency order (empty when invalid).</summary>
    public IReadOnlyList<string> Plan { get; }
    public bool IsValid => Errors.Count == 0;
}

public sealed class PromptValidationException(ValidationResult result)
    : Exception("Prompt is invalid: " + string.Join("; ", result.Errors.Select(e => e.Message)))
{
    public ValidationResult Result { get; } = result;
}

public sealed class NodeExecutionException(string nodeId, string classType, Exception inner)
    : Exception($"Node {nodeId} ({classType}) failed: {inner.Message}", inner)
{
    public string NodeId { get; } = nodeId;
    public string ClassType { get; } = classType;
}

public sealed class ExecutionOptions
{
    /// <summary>Run only these output nodes. Default: every output node in the graph.</summary>
    public IReadOnlyList<string>? OutputNodeIds { get; init; }
    public string? PromptId { get; init; }
    /// <summary>Text the Save Image node embeds in its PNGs next to the prompt (ComfyUI's <c>extra_pnginfo</c>: the editor's workflow JSON), by chunk name.</summary>
    public IReadOnlyDictionary<string, string>? ExtraPngInfo { get; init; }
}

public sealed class ExecutionResult
{
    public required string PromptId { get; init; }
    /// <summary>Outputs of the output nodes (fresh or cached), by node id.</summary>
    public required IReadOnlyDictionary<string, NodeOutputs> Outputs { get; init; }
    public required IReadOnlyList<string> Executed { get; init; }
    public required IReadOnlyList<string> Cached { get; init; }
    public required TimeSpan Elapsed { get; init; }
}

/// <summary>Run events, shaped like ComfyUI's websocket messages. All members are optional.</summary>
public interface IExecutionObserver
{
    void OnStart(string promptId, IReadOnlyList<string> plan) { }
    void OnNodeCached(string nodeId) { }
    void OnNodeExecuting(string nodeId) { }
    void OnNodeProgress(string nodeId, double value, double max, string? message) { }
    void OnNodeExecuted(string nodeId, NodeOutputs outputs) { }
    /// <summary>A node shows a picture while it works (a sampler's latent preview): a PNG, replaced by the next one.</summary>
    void OnNodePreview(string nodeId, byte[] png) { }
    void OnError(string? nodeId, Exception error) { }
    void OnFinished(ExecutionResult result) { }
}
