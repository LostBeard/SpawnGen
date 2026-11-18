using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Execution;

public sealed record ValidationError(string? NodeId, string? Input, string Code, string Message);

public sealed class ValidationResult
{
    public ValidationResult(IReadOnlyList<ValidationError> errors, IReadOnlyList<string> outputNodeIds, IReadOnlyList<string> plan,
        IReadOnlyDictionary<string, LoopScope>? loops = null)
    {
        Errors = errors;
        OutputNodeIds = outputNodeIds;
        Plan = plan;
        Loops = loops ?? new Dictionary<string, LoopScope>();
    }

    /// <summary>The prompt's loops by Start Loop id: body and End Loop, as validation paired them.</summary>
    public IReadOnlyDictionary<string, LoopScope> Loops { get; }

    public IReadOnlyList<ValidationError> Errors { get; }
    /// <summary>The output nodes the run is driven from.</summary>
    public IReadOnlyList<string> OutputNodeIds { get; }
    /// <summary>Needed nodes in dependency order (empty when invalid).</summary>
    public IReadOnlyList<string> Plan { get; }
    public bool IsValid => Errors.Count == 0;
}

public sealed class PromptValidationException(ValidationResult result, IReadOnlyList<OutputFailure>? failures = null)
    : Exception("Prompt is invalid: " + string.Join("; ", result.Errors.Select(e => e.Message)))
{
    public ValidationResult Result { get; } = result;
    /// <summary>Each output node that failed with its own errors (which outputs an error stops), when the prompt was checked output by output.</summary>
    public IReadOnlyList<OutputFailure> Failures { get; } = failures ?? [];
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
    /// <summary>How samplers preview (ComfyUI's <c>extra_data.preview_method</c>): "auto" / "latent2rgb", "taesd", "none"; null or "default" for the default.</summary>
    public string? PreviewMethod { get; init; }
    /// <summary>
    /// SpawnGen's run settings (<c>extra_data.spawngen_settings</c>): choices that change how a model computes, such as LTX-2's ComfyUI-compatible
    /// fp8 LoRA rounding (<see cref="RunSettings"/> names them). They are part of every node's cache signature, so changing one runs again.
    /// </summary>
    public IReadOnlyDictionary<string, string>? Settings { get; init; }
}

/// <summary>The run settings SpawnGen knows (keys of <see cref="ExecutionOptions.Settings"/>), with what a run uses when one is not given.</summary>
public static class RunSettings
{
    /// <summary>
    /// "true": a LoRA on LTX-2's 8-bit (F8_E4M3) weights re-rounds the patched weight with ComfyUI's seeded rounding, so the result is ComfyUI's.
    /// "false": the exact float W + strength B A (the LoRA's full effect, about 7e-3 from ComfyUI on one block). Default false: exact, with ComfyUI's
    /// rounding opt-in (TJ, 2026-11-18).
    /// </summary>
    public const string Ltx2ComfyFp8Lora = "ltx2.comfy_fp8_lora";

    /// <summary>The value of <paramref name="key"/> as a boolean, <paramref name="fallback"/> when it is missing or not "true" / "false".</summary>
    public static bool Flag(IReadOnlyDictionary<string, string>? settings, string key, bool fallback) =>
        settings is not null && settings.TryGetValue(key, out var text) && bool.TryParse(text, out var value) ? value : fallback;

    /// <summary>The settings as one canonical string (sorted keys), for cache signatures; "" when there are none.</summary>
    public static string Canonical(IReadOnlyDictionary<string, string>? settings) =>
        settings is null || settings.Count == 0 ? "" : string.Join(";", settings.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => kv.Key + "=" + kv.Value));

    /// <summary>
    /// The settings in an <c>extra_data.spawngen_settings</c> object: strings as they are, booleans and numbers as their JSON text; null when there
    /// are none.
    /// </summary>
    public static IReadOnlyDictionary<string, string>? FromJson(System.Text.Json.Nodes.JsonNode? node)
    {
        if (node is not System.Text.Json.Nodes.JsonObject o || o.Count == 0) return null;
        var settings = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var (key, value) in o)
            if (value is System.Text.Json.Nodes.JsonValue v)
                settings[key] = v.TryGetValue<string>(out var text) ? text : v.ToJsonString();
        return settings.Count == 0 ? null : settings;
    }
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
    /// <summary>Right after <see cref="OnStart"/>: the needed nodes whose results are cached, in the prompt's order (ComfyUI's execution_cached).</summary>
    void OnCachedNodes(IReadOnlyList<string> nodeIds) { }
    /// <summary>A node is served from the cache as the run reaches it (or a cached node's cached upstream).</summary>
    void OnNodeCached(string nodeId) { }
    void OnNodeExecuting(string nodeId) { }
    void OnNodeProgress(string nodeId, double value, double max, string? message) { }
    void OnNodeExecuted(string nodeId, NodeOutputs outputs) { }
    /// <summary>
    /// A node served from the cache shows its ui again (ComfyUI's emit_cached_output: an "executed" message with the cached ui): a cached node the run reached,
    /// and after the run any cached node with an intermediate output it did not reach. Only called when the cached result has ui.
    /// </summary>
    void OnNodeCachedUi(string nodeId, NodeOutputs outputs) { }
    /// <summary>A node shows a picture while it works (a sampler's latent preview): a PNG, replaced by the next one.</summary>
    void OnNodePreview(string nodeId, byte[] png) { }
    void OnError(string? nodeId, Exception error) { }
    void OnFinished(ExecutionResult result) { }
}

/// <summary>A node did not run because an <see cref="ExecutionBlocker"/> with a message reached it (ComfyUI reports "Execution Blocked: ..." the same way); the run goes on.</summary>
public sealed class ExecutionBlockedException(string nodeId, string message) : Exception("Execution Blocked: " + message)
{
    public string NodeId { get; } = nodeId;
}
