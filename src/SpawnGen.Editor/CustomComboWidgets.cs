using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor;

/// <summary>
/// Custom Combo's widgets, which ComfyUI's frontend defines (its schema has only the <c>choice</c> combo): text widgets <c>option1</c>, <c>option2</c>, ... that
/// the user types the options into, one empty one always last; <c>choice</c> offers the options that are not empty; a hidden <c>index</c> holds the
/// choice's position among the option widgets (-1 when none matches). All are saved in that order (<c>[choice, index, option1, ...]</c>) and sent.
/// </summary>
public static class CustomComboWidgets
{
    public const string NodeType = "CustomCombo";
    public const string Choice = "choice", Index = "index";

    static readonly WidgetSpec IndexSpec = new(WidgetKind.Int) { Default = 0L, Hidden = true };
    static readonly WidgetSpec OptionSpec = new(WidgetKind.String) { Default = "" };

    /// <summary>The option widgets' values in order (<c>option1</c> on), empty ones included.</summary>
    public static List<string> Options(IReadOnlyDictionary<string, object?> widgets)
    {
        var list = new List<string>();
        for (int k = 1; widgets.GetValueOrDefault("option" + k) is { } v; k++) list.Add(v.ToString() ?? "");
        return list;
    }

    /// <summary>The node's inputs: the choice (its options: the option widgets that are not empty), the hidden index, then the option widgets.</summary>
    public static IEnumerable<InputSpec> Inputs(NodeSchema schema, IReadOnlyDictionary<string, object?> widgets)
    {
        var options = Options(widgets);
        var choice = schema.FindInput(Choice)!;
        yield return choice with { Widget = choice.Widget! with { Options = [.. options.Where(o => o.Length > 0)], OptionsProvider = null, Default = "" } };
        yield return new InputSpec(Index, new SocketType("INT"), true, IndexSpec);
        for (int k = 1; k <= options.Count; k++) yield return new InputSpec("option" + k, new SocketType("STRING"), true, OptionSpec);
    }

    /// <summary>
    /// The widget values after a change, as ComfyUI's frontend keeps them: exactly one empty option last, the choice moved to the first option when its
    /// option is gone (unless <paramref name="keepChoice"/>, as when a saved workflow opens), and the index recomputed. Returns the names whose value
    /// changes (null = remove).
    /// </summary>
    public static Dictionary<string, object?> Normalize(IReadOnlyDictionary<string, object?> widgets, bool keepChoice = false)
    {
        var options = Options(widgets);
        int count = options.Count;
        while (count > 1 && options[count - 1].Length == 0 && options[count - 2].Length == 0) count--;
        var kept = options.Take(count).ToList();
        if (kept.Count == 0 || kept[^1].Length > 0) kept.Add("");

        var changes = new Dictionary<string, object?>();
        for (int k = 0; k < Math.Max(kept.Count, options.Count); k++)
        {
            object? want = k < kept.Count ? kept[k] : null;
            if (!Equals(widgets.GetValueOrDefault("option" + (k + 1)), want)) changes["option" + (k + 1)] = want;
        }
        var choice = widgets.GetValueOrDefault(Choice)?.ToString() ?? "";
        // ComfyUI's updateCombo: the choice must be one of the filled options (the empty spare is not one).
        if (!keepChoice && !kept.Any(o => o.Length > 0 && o == choice)) choice = kept.FirstOrDefault(o => o.Length > 0) ?? "";
        if (!Equals(widgets.GetValueOrDefault(Choice), choice)) changes[Choice] = choice;
        long index = kept.IndexOf(choice);
        if (!Equals(widgets.GetValueOrDefault(Index), index)) changes[Index] = index;
        return changes;
    }
}
