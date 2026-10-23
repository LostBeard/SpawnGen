namespace SpawnGen.Engine.Nodes;

/// <summary>Marks a class as a node. The class needs a public <c>Execute</c> or <c>ExecuteAsync</c> method.</summary>
[AttributeUsage(AttributeTargets.Class, Inherited = false)]
public sealed class NodeAttribute(string? id = null) : Attribute
{
    /// <summary>Node type id (ComfyUI's <c>class_type</c>). Defaults to the class name.</summary>
    public string? Id { get; } = id;
    public string? DisplayName { get; init; }
    public string Category { get; init; } = "utils";
    public string? Description { get; init; }
    public bool IsOutput { get; init; }
    public bool AlwaysExecute { get; init; }
    /// <summary>A node that only exists in the editor (Reroute, Note): the compiler resolves or drops it, a run never sends it.</summary>
    public bool FrontendOnly { get; init; }
    /// <summary>Earlier ids of this node type: workflows and prompts that use them still find it.</summary>
    public string[]? Aliases { get; init; }
    /// <summary>Output names, in order, for outputs a tuple cannot name (a single output such as ComfyUI's "patched_model"). Overrides the derived names.</summary>
    public string[]? OutputNames { get; init; }
}

/// <summary>Widget and socket hints for an <c>Execute</c> parameter. Unset numeric hints use <see cref="double.NaN"/>.</summary>
[AttributeUsage(AttributeTargets.Parameter)]
public sealed class InputAttribute : Attribute
{
    /// <summary>Overrides the snake_case name derived from the parameter name.</summary>
    public string? Name { get; init; }
    public double Min { get; init; } = double.NaN;
    public double Max { get; init; } = double.NaN;
    public double Step { get; init; } = double.NaN;
    public bool Multiline { get; init; }
    /// <summary>Fixed combo options for a string parameter.</summary>
    public string[]? Options { get; init; }
    /// <summary>Name of a public static method on the node class returning the combo options (e.g. model files).</summary>
    public string? OptionsProvider { get; init; }
    public string? Tooltip { get; init; }
    /// <summary>Earlier names of this input: a saved workflow that links to one still connects.</summary>
    public string[]? Aliases { get; init; }
}

/// <summary>Names the socket type for a value class (e.g. <c>[Socket("MODEL")]</c>). Without it the type is the upper-cased class name.</summary>
[AttributeUsage(AttributeTargets.Class | AttributeTargets.Struct | AttributeTargets.Interface, Inherited = false)]
public sealed class SocketAttribute(string name) : Attribute
{
    public string Name { get; } = name;
}
