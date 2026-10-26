namespace SpawnGen.Engine.Nodes;

/// <summary>The widget an input shows. <see cref="Color"/> is ComfyUI's COLOR: a "#rrggbb" text picked with a colour picker, never a socket.</summary>
public enum WidgetKind { Int, Float, String, Boolean, Combo, Color }

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

public sealed record InputSpec(string Name, SocketType Type, bool Required, WidgetSpec? Widget = null, string? Tooltip = null)
{
    /// <summary>Earlier names of the input (see <see cref="InputAttribute.Aliases"/>).</summary>
    public IReadOnlyList<string> Aliases { get; init; } = [];
    /// <summary>Evaluated only when the node asks for it (see <see cref="InputAttribute.Lazy"/>).</summary>
    public bool Lazy { get; init; }
    /// <summary>Declared optional (see <see cref="InputAttribute.Optional"/>).</summary>
    public bool Optional { get; init; }
    /// <summary>The Autogrow group this input belongs to (<c>images</c> for <c>images.image0</c>), null for an ordinary input.</summary>
    public string? Group { get; init; }
    /// <summary>For a group member: how many members the editor always shows (the group's minimum).</summary>
    public int GroupMin { get; init; }
    /// <summary>For an input of a dynamic combo option: the combo's input name (<c>resize_type</c> for <c>resize_type.width</c>).</summary>
    public string? DynamicParent { get; init; }
    /// <summary>For an input of a dynamic combo option: the options that bring it (shown, saved and sent only while one of them is chosen).</summary>
    public IReadOnlyList<string> DynamicOptions { get; init; } = [];
    /// <summary>ComfyUI's MatchType: the template id this any-type input shares with other inputs and outputs of the node.</summary>
    public string? MatchTemplate { get; init; }
    /// <summary>ComfyUI's forceInput: a primitive that is only a socket.</summary>
    public bool ForceInput { get; init; }

    /// <summary>Where ComfyUI's object_info lists the input: "required" for sockets that must be linked, for widgets and for Autogrow group members
    /// not declared optional (ComfyUI lists the whole group in its declared section).</summary>
    public bool ListedAsRequired => !Optional && (Required || Widget is not null || Group is not null);
}

public sealed record OutputSpec(string Name, SocketType Type, string? Tooltip = null)
{
    /// <summary>ComfyUI's <c>OUTPUT_IS_LIST</c> for this output.</summary>
    public bool IsList { get; init; }
    /// <summary>ComfyUI's MatchType output: the template id it takes its type from.</summary>
    public string? MatchTemplate { get; init; }
}

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
    /// <summary>ComfyUI's <c>INPUT_IS_LIST</c>: one run with whole lists rather than one run per item.</summary>
    public bool InputIsList { get; init; }
    /// <summary>ComfyUI's loop_boundary: "start", "end" or null.</summary>
    public string? LoopBoundary { get; init; }
    /// <summary>ComfyUI's accept_all_inputs: inputs the schema does not list are evaluated and passed on too.</summary>
    public bool AcceptAllInputs { get; init; }
    /// <summary>The boolean input that allows caching (see <see cref="NodeAttribute.CacheWhen"/>), null when caching follows <see cref="AlwaysExecute"/>.</summary>
    public string? CacheWhen { get; init; }
    /// <summary>ComfyUI's dev_only: made by expansions, not offered in the editor.</summary>
    public bool DevOnly { get; init; }
    /// <summary>For each dynamic combo (by input name): each option's inputs (full names) in the order the option declares them.</summary>
    public IReadOnlyDictionary<string, IReadOnlyDictionary<string, IReadOnlyList<string>>> DynamicCombos { get; init; } =
        new Dictionary<string, IReadOnlyDictionary<string, IReadOnlyList<string>>>();

    public InputSpec? FindInput(string name) => Inputs.FirstOrDefault(i => i.Name == name);

    /// <summary>The current name of an input given by its name or an earlier one; the name itself when no input answers to it.</summary>
    public string CanonicalInput(string name) => FindInput(name)?.Name ?? Inputs.FirstOrDefault(i => i.Aliases.Contains(name))?.Name ?? name;
}
