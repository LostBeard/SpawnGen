using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.Core;

/// <summary>The growing value rows of And, Or, Math Expression and Format Text.</summary>
static class ValueRows
{
    public static readonly string[] Letters =
        ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"];

    public static readonly string[] Numbered = ["value0", "value1", "value2", "value3", "value4", "value5", "value6", "value7", "value8", "value9"];

    /// <summary>
    /// The rows that are given, in the schema's order (not the prompt's: measured with ComfyUI, <c>values[0]</c> is <c>a</c> even when the prompt
    /// lists <c>values.b</c> first). A row linked to a node whose result is None is given, with the value None (measured: Format Text writes "None").
    /// </summary>
    public static List<KeyValuePair<string, object?>> Given(string group, string[] names, IReadOnlyDictionary<string, object?>? values, NodeContext? context)
    {
        var rows = new List<KeyValuePair<string, object?>>();
        foreach (var name in names)
        {
            if (values is not null && values.TryGetValue(name, out var value)) rows.Add(new(name, value));
            else if (context?.IsConnected($"{group}.{name}") == true) rows.Add(new(name, null));
        }
        return rows;
    }
}

/// <summary>
/// And: true when every value is true by Python's rules. Measured with ComfyUI: the values are tested in order and testing stops at the first
/// false one (0 then an ambiguous image gives False; the image first fails), and None is false.
/// </summary>
[Node("ComfyAndNode", Category = "utilities/logic", DisplayName = "And",
    Description = "Logical AND operation. Returns true if all of the values are truthy. Uses Python's rules for truthiness.")]
public sealed class ComfyAndNode
{
    public bool Execute([Autogrow(Prefix = "value", Min = 1, Max = 10), Input(Type = "*")] IReadOnlyDictionary<string, object?> values, NodeContext? context = null)
        => ValueRows.Given("values", ValueRows.Numbered, values, context).All(row => Truthiness.Of(row.Value));
}

/// <summary>Or: true when any value is true by Python's rules; testing stops at the first true one (measured with ComfyUI: 1 then an ambiguous image gives True).</summary>
[Node("ComfyOrNode", Category = "utilities/logic", DisplayName = "Or",
    Description = "Logical OR operation. Returns true if any of the values are truthy. Uses Python's rules for truthiness.")]
public sealed class ComfyOrNode
{
    public bool Execute([Autogrow(Prefix = "value", Min = 1, Max = 10), Input(Type = "*")] IReadOnlyDictionary<string, object?> values, NodeContext? context = null)
        => ValueRows.Given("values", ValueRows.Numbered, values, context).Any(row => Truthiness.Of(row.Value));
}

/// <summary>
/// Math Expression: evaluates the expression with each connected value under its letter and all of them, in letter order, as the list
/// <c>values</c>, and puts out the result as a float, an int and a boolean (SpawnGen's <see cref="PyExpression.EvaluateMath"/>). Measured with
/// ComfyUI: "a / 2" with a = 5 gives 2.5, 2 and True; a blank expression fails with "Expression cannot be empty."; a None value fails with
/// "Math Expression 'a' must evaluate to a numeric result, got NoneType: None".
/// </summary>
[Node("ComfyMathExpression", Category = "utilities", DisplayName = "Math Expression", OutputNames = ["FLOAT", "INT", "BOOL"])]
public sealed class ComfyMathExpression
{
    public (double, long, bool) Execute([Input(Multiline = true)] string expression = "a + b",
        [Autogrow(Names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], Min = 1),
         Input(Type = "FLOAT,INT,BOOLEAN")] IReadOnlyDictionary<string, object?>? values = null,
        NodeContext? context = null)
        => PyExpression.EvaluateMath(expression, ValueRows.Given("values", ValueRows.Letters, values, context));
}

/// <summary>
/// Format Text: Python's <c>str.format</c> with the connected values as named fields <c>{a}</c> .. <c>{z}</c> (SpawnGen's <see cref="PyFormat.FormatMap"/>).
/// Measured with ComfyUI: "{a}{b}" with only a fails with "KeyError: 'b'"; "{}" fails with "IndexError: Replacement index 0 out of range for
/// positional args tuple"; "x{{}}" with no values gives "x{}"; 0.1, True and "s" read "0.1", "True" and "s"; a None value reads "None".
/// Attributes: <c>.real</c> and <c>.imag</c> of numbers and <c>.numerator</c> / <c>.denominator</c> of ints ("{a.imag}" of 2.5 gives "0.0",
/// "{b.real}" of True gives "1"); any other fails as "'int' object has no attribute 'foo'" does.
/// </summary>
[Node("StringFormat", Category = "text", DisplayName = "Format Text",
    Description = "Same as Python's string format method. Supports all of Python's format options and features.")]
public sealed class StringFormat
{
    public string Execute(
        [Autogrow(Names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], Min = 0),
         Input(Type = "*")] IReadOnlyDictionary<string, object?>? values = null,
        [Input(Multiline = true)] string fString = "{a}",
        NodeContext? context = null)
    {
        var rows = ValueRows.Given("values", ValueRows.Letters, values, context).ToDictionary(r => r.Key, r => r.Value, StringComparer.Ordinal);
        return PyFormat.FormatMap(fString, name => rows.TryGetValue(name, out var v) ? (true, v) : (false, null), Attribute);
    }

    static object? Attribute(object? value, string name) => (value, name) switch
    {
        (bool b, "real" or "numerator") => b ? 1L : 0L,
        (bool, "imag") => 0L,
        (bool, "denominator") => 1L,
        (long or int or System.Numerics.BigInteger, "real" or "numerator") => value,
        (long or int or System.Numerics.BigInteger, "imag") => 0L,
        (long or int or System.Numerics.BigInteger, "denominator") => 1L,
        (double or float, "real") => Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture),
        (double or float, "imag") => 0.0,
        _ => throw new PyException("AttributeError", $"'{PyValues.TypeName(value)}' object has no attribute '{name}'"),
    };
}
