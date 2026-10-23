using System.Text.RegularExpressions;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Text nodes (ComfyUI's comfy_extras/nodes_string.py): small, exact, host-side.

[Node(Category = "utils/string", DisplayName = "Concatenate")]
public sealed class StringConcatenate
{
    public string Execute([Input(Multiline = true)] string stringA = "", [Input(Multiline = true)] string stringB = "", string delimiter = "") =>
        stringA + delimiter + stringB;
}

[Node(Category = "utils/string", DisplayName = "Substring")]
public sealed class StringSubstring
{
    /// <summary>Characters [start, end) of the text; negative values count from the end, and out-of-range values are clamped (Python slicing).</summary>
    public string Execute([Input(Multiline = true)] string @string = "", int start = 0, int end = 0)
    {
        int n = @string.Length;
        int Fix(int i) => Math.Clamp(i < 0 ? n + i : i, 0, n);
        int s = Fix(start), e = Fix(end);
        return e > s ? @string[s..e] : "";
    }
}

[Node(Category = "utils/string", DisplayName = "Length")]
public sealed class StringLength
{
    public long Execute([Input(Multiline = true)] string @string = "") => @string.Length;
}

[Node(Category = "utils/string", DisplayName = "Case Converter")]
public sealed class CaseConverter
{
    public static string[] Modes() => ["UPPERCASE", "lowercase", "Capitalize", "Title Case"];

    public string Execute([Input(Multiline = true)] string @string = "", [Input(OptionsProvider = nameof(Modes))] string mode = "UPPERCASE") => mode switch
    {
        "UPPERCASE" => @string.ToUpperInvariant(),
        "lowercase" => @string.ToLowerInvariant(),
        "Capitalize" => @string.Length == 0 ? "" : char.ToUpperInvariant(@string[0]) + @string[1..].ToLowerInvariant(),
        "Title Case" => Regex.Replace(@string.ToLowerInvariant(), @"\b[\p{L}\p{N}']+", m => char.ToUpperInvariant(m.Value[0]) + m.Value[1..]),
        _ => throw new ArgumentException($"Unknown mode '{mode}'.", nameof(mode)),
    };
}

[Node(Category = "utils/string", DisplayName = "Trim")]
public sealed class StringTrim
{
    public static string[] Modes() => ["Both", "Left", "Right"];

    public string Execute([Input(Multiline = true)] string @string = "", [Input(OptionsProvider = nameof(Modes))] string mode = "Both") => mode switch
    {
        "Both" => @string.Trim(),
        "Left" => @string.TrimStart(),
        "Right" => @string.TrimEnd(),
        _ => throw new ArgumentException($"Unknown mode '{mode}'.", nameof(mode)),
    };
}

[Node(Category = "utils/string", DisplayName = "Replace")]
public sealed class StringReplace
{
    public string Execute([Input(Multiline = true)] string @string = "", [Input(Multiline = true)] string find = "", [Input(Multiline = true)] string replace = "") =>
        find.Length == 0 ? @string : @string.Replace(find, replace, StringComparison.Ordinal);
}

[Node(Category = "utils/string", DisplayName = "Contains")]
public sealed class StringContains
{
    public bool Execute([Input(Multiline = true)] string @string = "", [Input(Multiline = true)] string substring = "", bool caseSensitive = true) =>
        @string.Contains(substring, caseSensitive ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase);
}

[Node(Category = "utils/string", DisplayName = "Compare")]
public sealed class StringCompare
{
    public static string[] Modes() => ["Starts With", "Ends With", "Equal"];

    public bool Execute([Input(Multiline = true)] string stringA = "", [Input(Multiline = true)] string stringB = "",
        [Input(OptionsProvider = nameof(Modes))] string mode = "Starts With", bool caseSensitive = true)
    {
        var c = caseSensitive ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase;
        return mode switch
        {
            "Starts With" => stringA.StartsWith(stringB, c),
            "Ends With" => stringA.EndsWith(stringB, c),
            "Equal" => string.Equals(stringA, stringB, c),
            _ => throw new ArgumentException($"Unknown mode '{mode}'.", nameof(mode)),
        };
    }
}

static class RegexOptionsFor
{
    public static RegexOptions Of(bool caseInsensitive, bool multiline, bool dotAll) =>
        (caseInsensitive ? RegexOptions.IgnoreCase : 0) | (multiline ? RegexOptions.Multiline : 0) | (dotAll ? RegexOptions.Singleline : 0);
}

[Node("RegexMatch", Category = "utils/string", DisplayName = "Regex Match", Aliases = ["RegexMatchNode"])]
public sealed class RegexMatchNode
{
    /// <summary>True when the pattern matches anywhere in the text.</summary>
    public bool Execute([Input(Multiline = true)] string @string = "", [Input(Multiline = true)] string regexPattern = "",
        bool caseInsensitive = true, bool multiline = false, bool dotall = false) =>
        Regex.IsMatch(@string, regexPattern, RegexOptionsFor.Of(caseInsensitive, multiline, dotall), TimeSpan.FromSeconds(2));
}

[Node("RegexReplace", Category = "utils/string", DisplayName = "Regex Replace", Aliases = ["RegexReplaceNode"])]
public sealed class RegexReplaceNode
{
    /// <summary>Replaces matches of the pattern (<c>$1</c> and <c>${name}</c> in the replacement refer to groups); <c>count</c> 0 replaces all of them.</summary>
    public string Execute([Input(Multiline = true)] string @string = "", [Input(Multiline = true)] string regexPattern = "", [Input(Multiline = true)] string replace = "",
        bool caseInsensitive = true, bool multiline = false, bool dotall = false, [Input(Min = 0, Max = 100)] int count = 0)
    {
        var regex = new Regex(regexPattern, RegexOptionsFor.Of(caseInsensitive, multiline, dotall), TimeSpan.FromSeconds(2));
        return count <= 0 ? regex.Replace(@string, replace) : regex.Replace(@string, replace, count);
    }
}

[Node(Category = "utils/math", DisplayName = "Math")]
public sealed class MathNode
{
    public static string[] Operations() => ["add", "subtract", "multiply", "divide", "floor divide", "modulo", "power", "min", "max"];

    /// <summary>One calculation on two numbers; <c>int</c> is the result rounded to the nearest whole number. Dividing by zero is an error, not infinity.</summary>
    public (double Float, long Int) Execute([Input(Min = -1e15, Max = 1e15, Step = 0.01)] double a = 0, [Input(Min = -1e15, Max = 1e15, Step = 0.01)] double b = 0,
        [Input(OptionsProvider = nameof(Operations))] string operation = "add")
    {
        if (operation is "divide" or "floor divide" or "modulo" && b == 0) throw new DivideByZeroException("The second number is 0.");
        double r = operation switch
        {
            "add" => a + b,
            "subtract" => a - b,
            "multiply" => a * b,
            "divide" => a / b,
            "floor divide" => Math.Floor(a / b),
            "modulo" => a - b * Math.Floor(a / b), // the sign follows b, as in Python
            "power" => Math.Pow(a, b),
            "min" => Math.Min(a, b),
            "max" => Math.Max(a, b),
            _ => throw new ArgumentException($"Unknown operation '{operation}'.", nameof(operation)),
        };
        return (r, (long)Math.Round(r, MidpointRounding.AwayFromZero));
    }
}

[Node(Category = "image", DisplayName = "Get Image Size")]
public sealed class GetImageSize
{
    public (long Width, long Height, long BatchSize) Execute(ImageBatch image) => (image.Width, image.Height, image.Batch);
}
