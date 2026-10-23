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
    /// <summary>
    /// ComfyUI's <c>INPUT_IS_LIST</c>: the node runs once and every input parameter receives the whole list that reaches it
    /// (declare them as <c>IReadOnlyList&lt;T&gt;</c>). Without it the node runs once per list item, see <see cref="Execution.PromptExecutor"/>.
    /// </summary>
    public bool InputIsList { get; init; }
    /// <summary>ComfyUI's <c>OUTPUT_IS_LIST</c>, one flag per output: that output returns a list (<c>IReadOnlyList&lt;T&gt;</c>) whose items flow on one by one.</summary>
    public bool[]? OutputIsList { get; init; }
    /// <summary>Output names in order, where ComfyUI names them (a single return value cannot carry a tuple name).</summary>
    public string[]? OutputNames { get; init; }
    /// <summary>Socket types of the outputs in order, where the CLR type does not say it (a string that is a COLOR); null entries keep the derived type.</summary>
    public string?[]? OutputTypes { get; init; }
}

/// <summary>Widget and socket hints for an <c>Execute</c> parameter. Unset numeric hints use <see cref="double.NaN"/>.</summary>
[AttributeUsage(AttributeTargets.Parameter)]
public sealed class InputAttribute : Attribute
{
    /// <summary>Overrides the snake_case name derived from the parameter name.</summary>
    public string? Name { get; init; }
    /// <summary>
    /// Overrides the socket type derived from the parameter type: a union such as <c>"IMAGE,MASK"</c> for an <c>object</c> parameter that takes
    /// either (ComfyUI's MultiType and MatchType inputs).
    /// </summary>
    public string? Type { get; init; }
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
    /// <summary>
    /// ComfyUI's lazy input: what feeds it only runs when the node's <c>CheckLazyStatus</c> method asks for it. Until then the node sees null.
    /// </summary>
    public bool Lazy { get; init; }
    /// <summary>
    /// Listed under "optional" in object_info, as ComfyUI's optional inputs are. A widget is "required" there even though its parameter has a
    /// default (it always has a value); a socket parameter with a default (<c>MaskBatch? mask = null</c>) is optional without saying so.
    /// </summary>
    public bool Optional { get; init; }
    /// <summary>A COLOR widget for a string parameter ("#rrggbb", ComfyUI's colour picker; default "#ffffff"). It has no socket, as in ComfyUI.</summary>
    public bool Color { get; init; }
    /// <summary>The widget's default when the parameter cannot carry one itself (a list parameter of an <see cref="NodeAttribute.InputIsList"/> node).</summary>
    public object? Default { get; init; }
}

/// <summary>
/// ComfyUI's Autogrow input: one parameter that stands for a growing row of sockets. With <see cref="Prefix"/> "image" the inputs are
/// <c>images.image0</c>, <c>images.image1</c>, ... up to <see cref="Max"/> (the group is the parameter's input name); with <see cref="Names"/> they are
/// <c>group.name</c> for each name. The first <see cref="Min"/> are required. The editor shows the connected ones and one free one. The parameter
/// receives the values that were given, in order: declare it <c>IReadOnlyList&lt;T&gt;</c>, or <c>IReadOnlyDictionary&lt;string, T&gt;</c> to know which
/// socket each came from (keyed by the name without the group).
/// </summary>
[AttributeUsage(AttributeTargets.Parameter)]
public sealed class AutogrowAttribute : Attribute
{
    public string? Prefix { get; init; }
    public string[]? Names { get; init; }
    public int Min { get; init; } = 1;
    public int Max { get; init; } = 10;
}

/// <summary>Names the socket type for a value class (e.g. <c>[Socket("MODEL")]</c>). Without it the type is the upper-cased class name.</summary>
[AttributeUsage(AttributeTargets.Class | AttributeTargets.Struct | AttributeTargets.Interface, Inherited = false)]
public sealed class SocketAttribute(string name) : Attribute
{
    public string Name { get; } = name;
}
