namespace SpawnGen.Engine.Nodes;

public enum WidgetKind { Int, Float, String, Boolean, Combo }

/// <summary>Editor/validation metadata for an input that can be set from a literal value.</summary>
public sealed record WidgetSpec(WidgetKind Kind)
{
    public object? Default { get; init; }
    public double? Min { get; init; }
    public double? Max { get; init; }
    public double? Step { get; init; }
    public bool Multiline { get; init; }
    public IReadOnlyList<string>? Options { get; init; }
    /// <summary>Evaluated on demand (e.g. the models folder listing) so the list is never stale.</summary>
    public Func<IReadOnlyList<string>>? OptionsProvider { get; init; }

    public IReadOnlyList<string> ResolveOptions() => OptionsProvider?.Invoke() ?? Options ?? [];
}

public sealed record InputSpec(string Name, SocketType Type, bool Required, WidgetSpec? Widget = null, string? Tooltip = null);

public sealed record OutputSpec(string Name, SocketType Type, string? Tooltip = null);

/// <summary>Everything the editor and validator need to know about a node type (ComfyUI's <c>object_info</c> entry).</summary>
public sealed class NodeSchema
{
    public required string Id { get; init; }
    public required string DisplayName { get; init; }
    public string Category { get; init; } = "utils";
    public string? Description { get; init; }
    public IReadOnlyList<InputSpec> Inputs { get; init; } = [];
    public IReadOnlyList<OutputSpec> Outputs { get; init; } = [];
    /// <summary>Output nodes (preview/save) are what a run is driven from.</summary>
    public bool IsOutputNode { get; init; }
    /// <summary>Never served from cache (clocks, randomness, external state).</summary>
    public bool AlwaysExecute { get; init; }
    /// <summary>Editor-only node (Reroute, Note): not part of the prompt that is run.</summary>
    public bool FrontendOnly { get; init; }

    public InputSpec? FindInput(string name) => Inputs.FirstOrDefault(i => i.Name == name);
}
