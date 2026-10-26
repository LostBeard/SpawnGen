namespace SpawnGen.Engine.Nodes;

/// <summary>
/// ComfyUI's DynamicCombo: a combo whose choice brings its own inputs. Mark a <see cref="DynamicChoice"/> parameter with this and name a public
/// static method returning the options; each option's inputs become <c>combo.input</c> (<c>resize_type.width</c>), shown only while it is chosen.
/// </summary>
[AttributeUsage(AttributeTargets.Parameter)]
public sealed class DynamicComboAttribute(string optionsMethod) : Attribute
{
    public string OptionsMethod { get; } = optionsMethod;
}

/// <summary>One choice of a dynamic combo and the inputs it brings (ComfyUI's DynamicCombo.Option).</summary>
public sealed record DynamicOption(string Key, params DynamicInput[] Inputs);

/// <summary>An input inside a dynamic combo option: a widget, or a socket when it has no widget.</summary>
public sealed record DynamicInput(string Name, SocketType Type, WidgetSpec? Widget = null, bool Required = true, string? Tooltip = null)
{
    /// <summary>ComfyUI's MatchType socket: the template id it shares with the node's other matching inputs and outputs.</summary>
    public string? MatchTemplate { get; init; }

    public static DynamicInput MatchType(string name, string template, bool required = true, string? tooltip = null) =>
        new(name, SocketType.Any, null, required, tooltip) { MatchTemplate = template };

    public static DynamicInput Int(string name, long @default = 0, double? min = null, double? max = null, double step = 1, string? tooltip = null) =>
        new(name, SocketType.Int, new WidgetSpec(WidgetKind.Int) { Default = @default, Min = min, Max = max, Step = step }, Tooltip: tooltip);

    public static DynamicInput Float(string name, double @default = 0, double? min = null, double? max = null, double step = 0.01, string? tooltip = null) =>
        new(name, SocketType.Float, new WidgetSpec(WidgetKind.Float) { Default = @default, Min = min, Max = max, Step = step }, Tooltip: tooltip);

    public static DynamicInput Boolean(string name, bool @default = false, string? tooltip = null) =>
        new(name, SocketType.Boolean, new WidgetSpec(WidgetKind.Boolean) { Default = @default }, Tooltip: tooltip);

    public static DynamicInput String(string name, string @default = "", bool multiline = false, string? tooltip = null) =>
        new(name, SocketType.String, new WidgetSpec(WidgetKind.String) { Default = @default, Multiline = multiline }, Tooltip: tooltip);

    public static DynamicInput Combo(string name, IReadOnlyList<string> options, string? @default = null, string? tooltip = null) =>
        new(name, SocketType.String, new WidgetSpec(WidgetKind.Combo) { Options = options, Default = @default ?? options.FirstOrDefault() }, Tooltip: tooltip);

    public static DynamicInput Socket(string name, string type, bool required = true, string? tooltip = null) => new(name, new SocketType(type), null, required, tooltip);
}

/// <summary>What a dynamic combo parameter receives: the chosen option and the values of that option's inputs, by name (without the combo's prefix).</summary>
public sealed class DynamicChoice(string key, IReadOnlyDictionary<string, object?> values)
{
    public string Key { get; } = key;
    public IReadOnlyDictionary<string, object?> Values { get; } = values;

    /// <summary>The value of one of the option's inputs as <typeparamref name="T"/>, or <paramref name="fallback"/> when it was not given.</summary>
    public T Get<T>(string name, T fallback = default!) =>
        Values.TryGetValue(name, out var v) && v is not null && ValueCoercion.TryCoerce(v, typeof(T), out var typed) ? (T)typed! : fallback;
}
