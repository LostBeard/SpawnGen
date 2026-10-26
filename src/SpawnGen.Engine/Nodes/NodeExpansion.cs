using SpawnGen.Engine.Graph;

namespace SpawnGen.Engine.Nodes;

/// <summary>
/// ComfyUI's node expansion: nodes a node adds to the running prompt (ComfyUI's <c>expand</c>), in the order they were made. The executor runs the output nodes
/// among them, and an output the expanding node returns as a <see cref="PromptInput"/> link is the value of that link. Return it as an element of the result tuple.
/// </summary>
public sealed class NodeExpansion
{
    /// <summary>The added nodes by id, in creation order.</summary>
    public OrderedDictionary<string, PromptNode> Nodes { get; } = new(StringComparer.Ordinal);
    /// <summary>The node an added node shows as in the editor (ComfyUI's override_display_id); unlisted ones show as the expanding node.</summary>
    public Dictionary<string, string> DisplayIds { get; } = new(StringComparer.Ordinal);
    /// <summary>Loops among the added nodes (copies of an inner Start Loop): their bodies and End Loops, as validation found them for the originals.</summary>
    public Dictionary<string, LoopScope> Loops { get; } = new(StringComparer.Ordinal);

    /// <summary>Adds a node; the id must be unique in the running prompt (prefix it with the expanding node's id).</summary>
    public PromptNode Add(string id, string classType, string? displayId = null)
    {
        var node = new PromptNode { ClassType = classType };
        Nodes[id] = node;
        if (displayId is not null) DisplayIds[id] = displayId;
        return node;
    }
}

/// <summary>A loop as validation pairs it (ComfyUI's _loop_body / _loop_end): the nodes between a Start Loop and its End Loop, and that End Loop.</summary>
public sealed record LoopScope(IReadOnlyCollection<string> Body, string End);

/// <summary>What a node that drives execution itself (a loop) can see of the run: ComfyUI's dynprompt and execution_list hidden inputs.</summary>
public interface IExecutionScope
{
    /// <summary>A node of the running prompt, the added ones included.</summary>
    PromptNode Node(string id);
    /// <summary>The body and End Loop of a Start Loop, or null when <paramref name="startId"/> is not one.</summary>
    LoopScope? Loop(string startId);
    /// <summary>Hands <paramref name="value"/> to a node waiting on it (ComfyUI's release_external_block).</summary>
    void ReleaseExternalBlock(string nodeId, object? value);
    /// <summary>What was handed to <paramref name="nodeId"/> (ComfyUI's get_external_block_result), null when nothing was.</summary>
    object? ExternalBlockResult(string nodeId);
}
