using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Text nodes (ComfyUI's comfy_extras/nodes_string.py): small, exact, host-side. Python's str rules come from PyText (code points, full case mappings,
// Python whitespace) and its re rules from PyRegex. ComfyUI marks case_sensitive / case_insensitive / multiline / dotall / group_index / count as
// "advanced" widgets (folded away in its frontend); here they are ordinary widgets, in the same order.

[Node(Category = "text", DisplayName = "Concatenate Text")]
public sealed class StringConcatenate
{
    public string Execute([Input(Multiline = true)] string stringA = "", [Input(Multiline = true)] string stringB = "", string delimiter = "") =>
        stringA + delimiter + stringB;
}

[Node(Category = "text", DisplayName = "Substring")]
public sealed class StringSubstring
{
    /// <summary>Characters [start, end) of the text (Python's <c>string[start:end]</c>): negative values count from the end, out-of-range values are clamped.</summary>
    public string Execute([Input(Multiline = true)] string @string = "", long start = 0, long end = 0) => PyText.Slice(@string, start, end);
}

[Node(Category = "text", DisplayName = "Text Length", OutputNames = ["length"])]
public sealed class StringLength
{
    /// <summary>The number of characters as Python counts them (an emoji is one).</summary>
    public long Execute([Input(Multiline = true)] string @string = "") => PyText.Length(@string);
}

[Node(Category = "text", DisplayName = "Convert Text Case")]
public sealed class CaseConverter
{
    public static string[] Modes() => ["UPPERCASE", "lowercase", "Capitalize", "Title Case"];

    /// <summary>Python's upper / lower / capitalize / title ("ß" upper-cases to "SS"; Title Case starts a word after any non-letter, so "it's" becomes "It'S").</summary>
    public string Execute([Input(Multiline = true)] string @string = "", [Input(OptionsProvider = nameof(Modes))] string mode = "UPPERCASE") => mode switch
    {
        "UPPERCASE" => PyText.Upper(@string),
        "lowercase" => PyText.Lower(@string),
        "Capitalize" => PyText.Capitalize(@string),
        "Title Case" => PyText.Title(@string),
        _ => @string,
    };
}

[Node(Category = "text", DisplayName = "Trim Text")]
public sealed class StringTrim
{
    public static string[] Modes() => ["Both", "Left", "Right"];

    /// <summary>Removes Python's whitespace (spaces, tabs, line breaks and the other Unicode spaces) from the ends.</summary>
    public string Execute([Input(Multiline = true)] string @string = "", [Input(OptionsProvider = nameof(Modes))] string mode = "Both") => mode switch
    {
        "Both" => PyText.Strip(@string),
        "Left" => PyText.LStrip(@string),
        "Right" => PyText.RStrip(@string),
        _ => @string,
    };
}

[Node(Category = "text", DisplayName = "Replace Text")]
public sealed class StringReplace
{
    /// <summary>Replaces every occurrence. An empty <c>find</c> inserts <c>replace</c> between all characters and at both ends, as Python's <c>str.replace</c> does.</summary>
    public string Execute([Input(Multiline = true)] string @string = "", [Input(Multiline = true)] string find = "", [Input(Multiline = true)] string replace = "") =>
        PyText.Replace(@string, find, replace);
}

[Node(Category = "text", DisplayName = "Contains Text", OutputNames = ["contains"])]
public sealed class StringContains
{
    public bool Execute([Input(Multiline = true)] string @string = "", [Input(Multiline = true)] string substring = "", bool caseSensitive = true) =>
        caseSensitive ? @string.Contains(substring, StringComparison.Ordinal) : PyText.Lower(@string).Contains(PyText.Lower(substring), StringComparison.Ordinal);
}

[Node(Category = "text", DisplayName = "Compare Text")]
public sealed class StringCompare
{
    public static string[] Modes() => ["Starts With", "Ends With", "Equal"];

    public bool Execute([Input(Multiline = true)] string stringA = "", [Input(Multiline = true)] string stringB = "",
        [Input(OptionsProvider = nameof(Modes))] string mode = "Starts With", bool caseSensitive = true)
    {
        string a = caseSensitive ? stringA : PyText.Lower(stringA), b = caseSensitive ? stringB : PyText.Lower(stringB);
        return mode switch
        {
            "Starts With" => a.StartsWith(b, StringComparison.Ordinal),
            "Ends With" => a.EndsWith(b, StringComparison.Ordinal),
            "Equal" => string.Equals(a, b, StringComparison.Ordinal),
            _ => throw new ArgumentException($"Unknown mode '{mode}'.", nameof(mode)),
        };
    }
}

[Node("RegexMatch", Category = "text", DisplayName = "Match Text", Aliases = ["RegexMatchNode"], OutputNames = ["matches"])]
public sealed class RegexMatchNode
{
    /// <summary>True when the Python regex matches anywhere in the text; a pattern Python cannot compile gives false (ComfyUI catches <c>re.error</c>).</summary>
    public bool Execute([Input(Multiline = true)] string @string = "", [Input(Multiline = true)] string regexPattern = "",
        bool caseInsensitive = true, bool multiline = false, bool dotall = false)
    {
        try
        {
            return PyRegex.Compile(regexPattern, caseInsensitive, multiline, dotall).Search(@string) is not null;
        }
        catch (PyException e) when (e.PyType == "error")
        {
            return false;
        }
    }
}

[Node("RegexReplace", Category = "text", DisplayName = "Replace Text (Regex)", Description = "Find and replace text using regex patterns.", Aliases = ["RegexReplaceNode"])]
public sealed class RegexReplaceNode
{
    /// <summary>
    /// Python's <c>re.sub</c>: <c>\1</c>, <c>\g&lt;1&gt;</c> and <c>\g&lt;name&gt;</c> in the replacement refer to groups, <c>\n</c> is a line break, <c>$</c> is a plain dollar sign;
    /// <c>count</c> 0 replaces every match.
    /// </summary>
    public string Execute([Input(Multiline = true)] string @string = "", [Input(Multiline = true)] string regexPattern = "", [Input(Multiline = true)] string replace = "",
        [Input(Optional = true)] bool caseInsensitive = true, [Input(Optional = true)] bool multiline = false,
        [Input(Optional = true, Tooltip = "When enabled, the dot (.) character will match any character including newline characters. When disabled, dots won't match newlines.")] bool dotall = false,
        [Input(Optional = true, Min = 0, Max = 100, Tooltip = "Maximum number of replacements to make. Set to 0 to replace all occurrences (default). Set to 1 to replace only the first match, 2 for the first two matches, etc.")] int count = 0) =>
        PyRegex.Compile(regexPattern, caseInsensitive, multiline, dotall).Sub(replace, @string, count);
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
