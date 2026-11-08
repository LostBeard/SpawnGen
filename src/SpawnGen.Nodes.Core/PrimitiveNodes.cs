using System.Numerics;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// The primitive nodes hand on the value of their one widget. Their interfaces (ranges, steps, multiline) are ComfyUI's object_info entries.

[Node("PrimitiveInt", Category = "utilities/primitive", DisplayName = "Int")]
public sealed class PrimitiveInt
{
    public long Execute([Input(Min = -9223372036854775807d, Max = 9223372036854775807d)] long value = 0) => value;
}

[Node("PrimitiveFloat", Category = "utilities/primitive", DisplayName = "Float")]
public sealed class PrimitiveFloat
{
    public double Execute([Input(Min = -9223372036854775807d, Max = 9223372036854775807d, Step = 0.1)] double value = 0) => value;
}

[Node("PrimitiveString", Category = "utilities/primitive", DisplayName = "Text")]
public sealed class PrimitiveString
{
    public string Execute(string value = "") => value;
}

[Node("PrimitiveStringMultiline", Category = "utilities/primitive", DisplayName = "Text (Multiline)")]
public sealed class PrimitiveStringMultiline
{
    public string Execute([Input(Multiline = true)] string value = "") => value;
}

[Node("PrimitiveBoolean", Category = "utilities/primitive", DisplayName = "Boolean")]
public sealed class PrimitiveBoolean
{
    public bool Execute(bool value = false) => value;
}

/// <summary>
/// Custom Combo. Its schema has only the <c>choice</c> combo with no options: the editor adds the option widgets and a hidden <c>index</c>
/// (see the editor's CustomComboWidgets) and sends them all, so the node takes inputs its schema does not list and does not check the choice
/// against the (empty) option list. Measured with ComfyUI: STRING is the choice as sent (even one that is not among the options), INDEX is the
/// <c>index</c> value as sent (5 stays 5 when the choice is the second option; 2.0 stays 2.0; -1 stays -1), and 0 when no index is sent.
/// </summary>
[Node("CustomCombo", Category = "utilities", DisplayName = "Custom Combo", OutputNames = ["STRING", "INDEX"], OutputTypes = [null, "INT"],
    AcceptAllInputs = true, ValidatesOwnInputs = true)]
public sealed class CustomCombo
{
    public (string Choice, object? Index) Execute([Input(Options = [])] string choice, [ExtraInputs] IReadOnlyDictionary<string, object?> extra)
        => (choice, extra.TryGetValue("index", out var index) && index is not null ? index : 0L);
}

/// <summary>
/// Preview as Text: shows any value as text in the node and passes that text on. Measured with ComfyUI: text is shown as it is; ints, floats,
/// booleans and None as Python's <c>str</c> prints them ("120", "1.0", "1e+16", "1.5e-05", "-0.0", "nan", "True", "None"). Lists and dicts read as
/// <c>json.dumps(value, indent=4, ensure_ascii=False)</c>, and what JSON cannot write falls back to <c>str</c>. A tensor (which ComfyUI prints
/// with all its numbers) reads as its kind and shape, "ImageBatch [1, 512, 512, 3]".
/// </summary>
[Node("PreviewAny", Category = "utilities", DisplayName = "Preview as Text", Description = "Preview any input value as text.", IsOutput = true)]
public sealed class PreviewAny
{
    public (string, UiOutput) Execute(object? source)
    {
        var text = Format(source);
        return (text, new UiOutput("text", new[] { text }));
    }

    /// <summary>The text the node shows for <paramref name="value"/>.</summary>
    public static string Format(object? value)
    {
        switch (value)
        {
            case string s: return s;
            case null or bool or long or int or short or byte or BigInteger or double or float: return PyText.Str(value);
        }
        try
        {
            return PyJson.Dumps(value, 4);
        }
        catch (PyException)
        {
            return PyText.Str(value);
        }
    }
}
