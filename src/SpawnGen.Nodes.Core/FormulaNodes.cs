using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.Core;

/// <summary>ComfyUI's Format Text: Python's <c>str.format</c> with the connected values under their socket names (a, b, ...).</summary>
[Node("StringFormat", Category = "text", DisplayName = "Format Text", OutputNames = ["STRING"],
    Description = "Same as Python's string format method. Supports all of Python's format options and features.")]
public sealed class StringFormat
{
    public string Execute([Autogrow(Names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], Min = 0)]
        IReadOnlyDictionary<string, object?> values,
        [Input(Multiline = true)] string fString = "{a}") =>
        PyFormat.FormatMap(fString, name => values.TryGetValue(name, out var v) ? (true, PyExpression.Normalize(v)) : (false, null));
}

/// <summary>ComfyUI's Math Expression: a simpleeval expression over the connected numbers (a, b, ... and the list <c>values</c>), as FLOAT, INT and BOOL.</summary>
[Node("ComfyMathExpression", Category = "utilities", DisplayName = "Math Expression", OutputNames = ["FLOAT", "INT", "BOOL"])]
public sealed class ComfyMathExpression
{
    public (double Float, long Int, bool Bool) Execute([Input(Multiline = true)] string expression = "a + b",
        [Autogrow(Names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"], Min = 1), Input(Type = "FLOAT,INT,BOOLEAN")]
        IReadOnlyDictionary<string, object?>? values = null) =>
        PyExpression.EvaluateMath(expression, [.. values ?? new Dictionary<string, object?>()]);
}

/// <summary>ComfyUI's And: true when every connected value is true by Python's rules.</summary>
[Node("ComfyAndNode", Category = "utilities/logic", DisplayName = "And",
    Description = "Logical AND operation. Returns true if all of the values are truthy. Uses Python's rules for truthiness.")]
public sealed class ComfyAndNode
{
    public bool Execute([Autogrow(Prefix = "value", Min = 1)] IReadOnlyList<object?> values) => values.All(Truthiness.Of);
}

/// <summary>ComfyUI's Or: true when any connected value is true by Python's rules.</summary>
[Node("ComfyOrNode", Category = "utilities/logic", DisplayName = "Or",
    Description = "Logical OR operation. Returns true if any of the values are truthy. Uses Python's rules for truthiness.")]
public sealed class ComfyOrNode
{
    public bool Execute([Autogrow(Prefix = "value", Min = 1)] IReadOnlyList<object?> values) => values.Any(Truthiness.Of);
}
