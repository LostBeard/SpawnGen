using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// The text nodes work on Python strings: lengths, slices and case changes count code points (an emoji is one character), as the Python
// documentation describes str. The Py* helpers give those semantics; each node here is a thin layer over them. Behavior that the
// documentation leaves open (what a mode outside the list does, how "not case sensitive" compares) was measured by running ComfyUI.

[Node("StringConcatenate", Category = "text", DisplayName = "Concatenate Text")]
public sealed class StringConcatenate
{
    /// <summary>The two texts joined with the delimiter between them.</summary>
    public string Execute([Input(Multiline = true)] string stringA = "", [Input(Multiline = true)] string stringB = "", string delimiter = "")
        => string.Concat(stringA, delimiter, stringB);
}

[Node("StringSubstring", Category = "text", DisplayName = "Substring")]
public sealed class StringSubstring
{
    /// <summary>
    /// The characters from <paramref name="start"/> up to (not including) <paramref name="end"/>, as a Python slice <c>s[start:end]</c>: a negative
    /// index counts from the end, an index past either end is clamped, and an end before the start gives "".
    /// </summary>
    public string Execute([Input(Multiline = true)] string @string = "", long start = 0, long end = 0) => PyText.Slice(@string, start, end);
}

[Node("StringLength", Category = "text", DisplayName = "Text Length", OutputNames = ["length"])]
public sealed class StringLength
{
    /// <summary>The number of characters (Unicode code points, as Python's <c>len</c>).</summary>
    public long Execute([Input(Multiline = true)] string @string = "") => PyText.Length(@string);
}

[Node("CaseConverter", Category = "text", DisplayName = "Convert Text Case")]
public sealed class CaseConverter
{
    /// <summary>
    /// Python's <c>str.upper</c>, <c>str.lower</c>, <c>str.capitalize</c> or <c>str.title</c>. Measured with ComfyUI: a mode outside the list (fed
    /// through a link, which validation does not check) gives the text back unchanged.
    /// </summary>
    public string Execute([Input(Multiline = true)] string @string = "",
        [Input(Options = ["UPPERCASE", "lowercase", "Capitalize", "Title Case"])] string mode = "UPPERCASE") => mode switch
    {
        "UPPERCASE" => PyText.Upper(@string),
        "lowercase" => PyText.Lower(@string),
        "Capitalize" => PyText.Capitalize(@string),
        "Title Case" => PyText.Title(@string),
        _ => @string,
    };
}

[Node("StringTrim", Category = "text", DisplayName = "Trim Text")]
public sealed class StringTrim
{
    /// <summary>
    /// Removes whitespace (Python's definition, as <c>str.strip</c>) from both ends, the start or the end. Measured with ComfyUI: a mode outside the
    /// list gives the text back unchanged.
    /// </summary>
    public string Execute([Input(Multiline = true)] string @string = "", [Input(Options = ["Both", "Left", "Right"])] string mode = "Both") => mode switch
    {
        "Both" => PyText.Strip(@string),
        "Left" => PyText.LStrip(@string),
        "Right" => PyText.RStrip(@string),
        _ => @string,
    };
}

[Node("StringReplace", Category = "text", DisplayName = "Replace Text")]
public sealed class StringReplace
{
    /// <summary>Every occurrence of <paramref name="find"/> replaced, as Python's <c>str.replace</c> (an empty find inserts between every character).</summary>
    public string Execute([Input(Multiline = true)] string @string = "", [Input(Multiline = true)] string find = "", [Input(Multiline = true)] string replace = "")
        => PyText.Replace(@string, find, replace);
}

/// <summary>
/// Case-insensitive comparison for the text nodes. Measured with ComfyUI: both texts are lowercased with Python's <c>str.lower</c> (not casefolded):
/// "STRASSE" does not contain "ß", "ﬁ" is not "fi", "ſ" is not "s", while "İ" (lowercased to "i" plus a combining dot) does contain "i".
/// </summary>
static class TextCase
{
    public static (string A, string B) Fold(string a, string b, bool caseSensitive) => caseSensitive ? (a, b) : (PyText.Lower(a), PyText.Lower(b));
}

[Node("StringContains", Category = "text", DisplayName = "Contains Text", OutputNames = ["contains"])]
public sealed class StringContains
{
    /// <summary>True when <paramref name="substring"/> occurs in the text (an empty substring always does).</summary>
    public bool Execute([Input(Multiline = true)] string @string = "", [Input(Multiline = true)] string substring = "", bool caseSensitive = true)
    {
        var (text, part) = TextCase.Fold(@string, substring, caseSensitive);
        return text.Contains(part, StringComparison.Ordinal);
    }
}

[Node("StringCompare", Category = "text", DisplayName = "Compare Text")]
public sealed class StringCompare
{
    /// <summary>
    /// Whether text A starts with, ends with, or equals text B. Measured with ComfyUI: a mode outside the list (fed through a link) fails the run
    /// with "IndexError: list index out of range".
    /// </summary>
    public bool Execute([Input(Multiline = true)] string stringA = "", [Input(Multiline = true)] string stringB = "",
        [Input(Options = ["Starts With", "Ends With", "Equal"])] string mode = "Starts With", bool caseSensitive = true)
    {
        var (a, b) = TextCase.Fold(stringA, stringB, caseSensitive);
        return mode switch
        {
            "Starts With" => a.StartsWith(b, StringComparison.Ordinal),
            "Ends With" => a.EndsWith(b, StringComparison.Ordinal),
            "Equal" => string.Equals(a, b, StringComparison.Ordinal),
            _ => throw new PyException("IndexError", "list index out of range"),
        };
    }
}

[Node("RegexMatch", Category = "text", DisplayName = "Match Text", OutputNames = ["matches"], Aliases = ["RegexMatchNode"])]
public sealed class RegexMatchNode
{
    /// <summary>
    /// True when the Python regular expression matches anywhere in the text (<c>re.search</c>). Measured with ComfyUI: a pattern that does not
    /// compile ("(") gives false rather than an error; an empty pattern matches.
    /// </summary>
    public bool Execute([Input(Multiline = true)] string @string = "", [Input(Multiline = true)] string regexPattern = "",
        bool caseInsensitive = true, bool multiline = false, bool dotall = false)
    {
        PyRegex regex;
        try
        {
            regex = PyRegex.Compile(regexPattern, caseInsensitive, multiline, dotall);
        }
        catch (PyException)
        {
            return false;
        }
        return regex.Search(@string) is not null;
    }
}

[Node("RegexReplace", Category = "text", DisplayName = "Replace Text (Regex)", Description = "Find and replace text using regex patterns.",
    Aliases = ["RegexReplaceNode"])]
public sealed class RegexReplaceNode
{
    /// <summary>
    /// Python's <c>re.sub</c>: <c>\1</c> and <c>\g&lt;name&gt;</c> in the replacement insert groups, a <c>$</c> is plain text. Measured with ComfyUI: a
    /// pattern that does not compile, or a reference to a group the pattern does not have, fails the run with <c>re.error</c>
    /// ("missing ), unterminated subpattern at position 0", "invalid group reference 2 at position 1").
    /// </summary>
    public string Execute([Input(Multiline = true)] string @string = "", [Input(Multiline = true)] string regexPattern = "",
        [Input(Multiline = true)] string replace = "",
        [Input(Optional = true)] bool caseInsensitive = true,
        [Input(Optional = true)] bool multiline = false,
        [Input(Optional = true, Tooltip = "When enabled, the dot (.) character will match any character including newline characters. When disabled, dots won't match newlines.")]
        bool dotall = false,
        [Input(Optional = true, Min = 0, Max = 100,
            Tooltip = "Maximum number of replacements to make. Set to 0 to replace all occurrences (default). Set to 1 to replace only the first match, 2 for the first two matches, etc.")]
        long count = 0)
        => PyRegex.Compile(regexPattern, caseInsensitive, multiline, dotall).Sub(replace, @string, (int)Math.Clamp(count, 0, int.MaxValue));
}

/// <summary>
/// SpawnGen's own arithmetic node (ComfyUI has no node of this id; its Math Expression is <c>ComfyMathExpression</c>). The operations follow
/// Python's float arithmetic as the Python documentation describes it: floor division rounds toward negative infinity, the remainder takes the
/// sign of the divisor, and dividing by zero is an error rather than infinity.
/// </summary>
[Node("MathNode", Category = "utils/math", DisplayName = "Math", Description = "Applies an arithmetic operation to two numbers; the result as a float and rounded to an int.")]
public sealed class MathNode
{
    public static readonly string[] Operations = ["add", "subtract", "multiply", "divide", "floor divide", "modulo", "power", "min", "max"];

    /// <summary>The result as a float, and as an int rounded half away from zero (3.5 gives 4, -3.5 gives -4).</summary>
    public (double Float, long Int) Execute([Input(Min = -1e15, Max = 1e15)] double a = 0, [Input(Min = -1e15, Max = 1e15)] double b = 0,
        [Input(OptionsProvider = nameof(OperationOptions))] string operation = "add")
    {
        double r = operation switch
        {
            "add" => a + b,
            "subtract" => a - b,
            "multiply" => a * b,
            "divide" => b == 0 ? throw new DivideByZeroException("float division by zero") : a / b,
            "floor divide" => FloorDivide(a, b),
            "modulo" => Modulo(a, b),
            "power" => Power(a, b),
            "min" => Math.Min(a, b),
            "max" => Math.Max(a, b),
            _ => throw new ArgumentException($"Unknown operation '{operation}'. Expected one of: {string.Join(", ", Operations)}."),
        };
        return (r, ToInt(r));
    }

    public static IEnumerable<string> OperationOptions() => Operations;

    /// <summary>Python's <c>a % b</c> for floats: zero or the sign of <paramref name="b"/>, so that <c>a == (a // b) * b + a % b</c>.</summary>
    static double Modulo(double a, double b)
    {
        if (b == 0) throw new DivideByZeroException("float modulo");
        double r = a % b; // C#'s % on doubles truncates toward zero (the sign of a)
        if (r != 0)
        {
            if (r < 0 != b < 0) r += b;
        }
        else r = Math.CopySign(0.0, b);
        return r;
    }

    /// <summary>Python's <c>a // b</c> for floats: the quotient rounded toward negative infinity, consistent with <see cref="Modulo"/>.</summary>
    static double FloorDivide(double a, double b)
    {
        if (b == 0) throw new DivideByZeroException("float floor division by zero");
        double truncRem = a % b;
        double q = (a - truncRem) / b; // exact multiple of b, so this is (nearly) an integer
        if (truncRem != 0 && truncRem < 0 != b < 0) q -= 1;
        if (q == 0) return Math.CopySign(0.0, a / b);
        double f = Math.Floor(q);
        return q - f > 0.5 ? f + 1 : f;
    }

    /// <summary>Python's <c>a ** b</c> for floats, refusing the cases Python refuses (or answers with a complex number).</summary>
    static double Power(double a, double b)
    {
        if (a == 0 && b < 0) throw new DivideByZeroException("0.0 cannot be raised to a negative power");
        if (a < 0 && double.IsFinite(b) && b != Math.Floor(b))
            throw new ArgumentException("A negative number raised to a fractional power has no real result.");
        double r = Math.Pow(a, b);
        if (double.IsInfinity(r) && double.IsFinite(a) && double.IsFinite(b)) throw new OverflowException("(34, 'Result too large')");
        return r;
    }

    static long ToInt(double r)
    {
        if (double.IsNaN(r)) throw new ArgumentException("cannot convert float NaN to integer");
        if (double.IsInfinity(r)) throw new OverflowException("cannot convert float infinity to integer");
        double rounded = Math.Round(r, MidpointRounding.AwayFromZero);
        // 2^63 is the first double past long's range.
        if (rounded >= 9223372036854775808.0 || rounded < -9223372036854775808.0) throw new OverflowException($"{PyNumber.FloatRepr(r)} does not fit in an INT.");
        return (long)rounded;
    }
}

[Node("GetImageSize", Category = "image", DisplayName = "Get Image Size", Description = "Returns the width, height, and batch size of the image.")]
public sealed class GetImageSize
{
    /// <summary>
    /// The image's width, height and number of pictures. Measured with ComfyUI: the node also shows the numbers as text,
    /// "width: 37, height: 21\n batch size: 2".
    /// </summary>
    public (long Width, long Height, long BatchSize) Execute(ImageBatch image, NodeContext? context = null)
    {
        long width = image.Width, height = image.Height, batchSize = image.Batch;
        context?.Status($"width: {width}, height: {height}\n batch size: {batchSize}");
        return (width, height, batchSize);
    }
}
