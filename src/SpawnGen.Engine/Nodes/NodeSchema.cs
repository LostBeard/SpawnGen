namespace SpawnGen.Engine.Nodes;

/// <summary>
/// The widget an input shows. <see cref="Color"/> is ComfyUI's COLOR: a "#rrggbb" text picked with a colour picker, never a socket. <see cref="Json"/> holds a
/// JSON-like value (<see cref="Types.DictValue"/>, <see cref="Types.ArrayValue"/>): ComfyUI's object-valued widgets such as BOUNDING_BOX, BOUNDING_BOXES and
/// COMPOSITOR; prompts carry it as a JSON object, or wrapped as <c>{"__value__": [...]}</c> when it is a list.
/// </summary>
public enum WidgetKind { Int, Float, String, Boolean, Combo, Color, Json }

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
    /// <summary>ComfyUI's <c>socketless</c> for a <see cref="WidgetKind.Json"/> widget: true when it cannot be turned into a socket (null = not stated).</summary>
    public bool? Socketless { get; init; }
    /// <summary>A widget the editor keeps, saves and sends but does not draw (ComfyUI's hidden widgets: Custom Combo's index).</summary>
    public bool Hidden { get; init; }

    /// <summary>
    /// ComfyUI lists some combo options as numbers (Create Video's bit_depth: "auto", 8, 10). The editor keeps every option as text; with this set,
    /// an option that reads as an integer is a number in object_info, in prompts and in saved workflows, as ComfyUI's frontend sends it.
    /// </summary>
    public bool NumericOptions { get; init; }

    public IReadOnlyList<string> ResolveOptions() => OptionsProvider?.Invoke() ?? Options ?? [];

    /// <summary>A combo value as ComfyUI has it: the integer for a <see cref="NumericOptions"/> option that reads as one, else the text.</summary>
    public object OptionValue(string text) =>
        NumericOptions && long.TryParse(text, System.Globalization.NumberStyles.AllowLeadingSign, System.Globalization.CultureInfo.InvariantCulture, out var n) ? n : text;
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
    /// <summary>
    /// For an input several options bring with different widgets (Save Image (Advanced)'s bit_depth: "8-bit" / "16-bit" for png, "32-bit float" /
    /// "16-bit float" for exr): the widget and tooltip per option key. <see cref="Widget"/> and <see cref="Tooltip"/> are the first option's.
    /// </summary>
    public IReadOnlyDictionary<string, InputVariant>? Variants { get; init; }

    /// <summary>This input as option <paramref name="option"/> of its dynamic combo brings it (itself when it has no variant for that option).</summary>
    public InputSpec ForOption(string? option) =>
        option is not null && Variants is { } variants && variants.TryGetValue(option, out var v) ? this with { Widget = v.Widget, Tooltip = v.Tooltip } : this;
    /// <summary>ComfyUI's MatchType: the template id this any-type input shares with other inputs and outputs of the node.</summary>
    public string? MatchTemplate { get; init; }
    /// <summary>ComfyUI's forceInput: a primitive that is only a socket.</summary>
    public bool ForceInput { get; init; }
    /// <summary>The default ComfyUI states for a socket-only input that would otherwise be a widget (a forceInput BOUNDING_BOX); the schema shows it, nothing uses it.</summary>
    public object? StatedDefault { get; init; }

    /// <summary>Where ComfyUI's object_info lists the input: "required" for sockets that must be linked, for widgets and for Autogrow group members
    /// not declared optional (ComfyUI lists the whole group in its declared section).</summary>
    public bool ListedAsRequired => !Optional && (Required || Widget is not null || Group is not null);
}

/// <summary>The widget and tooltip one option of a dynamic combo gives an input it shares with other options (see <see cref="InputSpec.Variants"/>).</summary>
public sealed record InputVariant(WidgetSpec? Widget, string? Tooltip);

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
    /// <summary>ComfyUI's <c>has_intermediate_output</c> (see <see cref="NodeAttribute.HasIntermediateOutput"/>).</summary>
    public bool HasIntermediateOutput { get; init; }
    /// <summary>Editor-only node (Reroute, Note): not part of the prompt that is run.</summary>
    public bool FrontendOnly { get; init; }
    /// <summary>ComfyUI's <c>INPUT_IS_LIST</c>: one run with whole lists rather than one run per item.</summary>
    public bool InputIsList { get; init; }
    /// <summary>ComfyUI's loop_boundary: "start", "end" or null.</summary>
    public string? LoopBoundary { get; init; }
    /// <summary>ComfyUI's accept_all_inputs: inputs the schema does not list are evaluated and passed on too.</summary>
    public bool AcceptAllInputs { get; init; }
    /// <summary>See <see cref="NodeAttribute.ValidatesOwnInputs"/>: literal values are not checked against the schema.</summary>
    public bool ValidatesOwnInputs { get; init; }
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
