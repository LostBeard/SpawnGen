using System.Globalization;
using System.Numerics;
using System.Text;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// Python's <c>json</c> module as the text nodes use it. Parsing gives Python values: <see cref="DictValue"/> (keys in first-seen order, the last value wins),
/// <see cref="ArrayValue"/>, string, <see cref="BigInteger"/> for integers, double, bool, null; NaN / Infinity / -Infinity are accepted.
/// <see cref="Dumps"/> writes what <c>json.dumps(value, ensure_ascii=False, indent=...)</c> writes.
/// </summary>
public static class PyJson
{
    /// <summary>Python's JSONDecodeError: the message and where in the text (a UTF-16 index; the message gives it in code points, as Python counts).</summary>
    sealed class JsonDecodeError(string message, int pos) : Exception(message)
    {
        public int Pos { get; } = pos;

        /// <summary><c>str(JSONDecodeError)</c>: "msg: line L column C (char P)".</summary>
        public string Describe(string doc)
        {
            var before = doc[..Math.Min(Pos, doc.Length)];
            int pos = PyText.Length(before);
            int line = before.Count(ch => ch == '\n') + 1;
            int lastNewline = before.LastIndexOf('\n');
            int column = lastNewline < 0 ? pos + 1 : pos - PyText.Length(before[..lastNewline]);
            return $"{Message}: line {line} column {column} (char {pos})";
        }
    }

    /// <summary><c>json.JSONDecoder().raw_decode(text, start)</c>: the value that starts exactly at <paramref name="start"/> and where it ends; false when it is not valid JSON.</summary>
    public static bool TryRawDecode(string text, int start, out object? value, out int end)
    {
        try
        {
            int i = start;
            value = ParseValue(text, ref i, 0);
            end = i;
            return true;
        }
        catch (JsonDecodeError)
        {
            value = null;
            end = start;
            return false;
        }
    }

    /// <summary><c>json.loads(text)</c>: one value, optional whitespace around it. Throws <see cref="PyException"/> (JSONDecodeError) when the text is not JSON.</summary>
    public static object? Loads(string text)
    {
        int i = SkipWs(text, 0);
        try
        {
            var v = ParseValue(text, ref i, 0);
            i = SkipWs(text, i);
            if (i != text.Length) throw new JsonDecodeError("Extra data", i);
            return v;
        }
        catch (JsonDecodeError e) { throw new PyException("JSONDecodeError", e.Describe(text)); }
    }

    static int SkipWs(string s, int i)
    {
        while (i < s.Length && s[i] is ' ' or '\t' or '\n' or '\r') i++;
        return i;
    }

    static object? ParseValue(string s, ref int i, int depth)
    {
        if (depth > 900) throw new JsonDecodeError("too deeply nested", i); // Python stops at its recursion limit
        if (i >= s.Length) throw new JsonDecodeError("Expecting value", i);
        char c = s[i];
        switch (c)
        {
            case '"': i++; return ParseString(s, ref i);
            case '{': i++; return ParseObject(s, ref i, depth);
            case '[': i++; return ParseArray(s, ref i, depth);
            case 'n' when Match(s, i, "null"): i += 4; return null;
            case 't' when Match(s, i, "true"): i += 4; return true;
            case 'f' when Match(s, i, "false"): i += 5; return false;
            case 'N' when Match(s, i, "NaN"): i += 3; return double.NaN;
            case 'I' when Match(s, i, "Infinity"): i += 8; return double.PositiveInfinity;
            case '-' when Match(s, i, "-Infinity"): i += 9; return double.NegativeInfinity;
        }
        if (c == '-' || char.IsAsciiDigit(c)) return ParseNumber(s, ref i);
        throw new JsonDecodeError("Expecting value", i);
    }

    static bool Match(string s, int i, string word) => string.CompareOrdinal(s, i, word, 0, word.Length) == 0 && i + word.Length <= s.Length;

    /// <summary>Python's NUMBER_RE: <c>-?(0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?</c>; an integer unless it has a fraction or exponent.</summary>
    static object ParseNumber(string s, ref int i)
    {
        int start = i;
        if (s[i] == '-') i++;
        if (i >= s.Length || !char.IsAsciiDigit(s[i])) throw new JsonDecodeError("Expecting value", start);
        if (s[i] == '0') i++;
        else while (i < s.Length && char.IsAsciiDigit(s[i])) i++;
        int intEnd = i;
        bool isFloat = false;
        if (i + 1 < s.Length && s[i] == '.' && char.IsAsciiDigit(s[i + 1]))
        {
            i += 2;
            while (i < s.Length && char.IsAsciiDigit(s[i])) i++;
            isFloat = true;
        }
        if (i < s.Length && s[i] is 'e' or 'E')
        {
            int j = i + 1;
            if (j < s.Length && s[j] is '+' or '-') j++;
            if (j < s.Length && char.IsAsciiDigit(s[j]))
            {
                while (j < s.Length && char.IsAsciiDigit(s[j])) j++;
                i = j;
                isFloat = true;
            }
        }
        var text = s[start..i];
        if (!isFloat) return BigInteger.Parse(text, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
        return double.Parse(text, NumberStyles.Float, CultureInfo.InvariantCulture);
    }

    static string ParseString(string s, ref int i)
    {
        int begin = i - 1; // the opening quote
        var sb = new StringBuilder();
        while (true)
        {
            if (i >= s.Length) throw new JsonDecodeError("Unterminated string starting at", begin);
            char c = s[i++];
            if (c == '"') return sb.ToString();
            if (c < 0x20) throw new JsonDecodeError("Invalid control character at", i - 1); // strict mode
            if (c != '\\') { sb.Append(c); continue; }
            if (i >= s.Length) throw new JsonDecodeError("Unterminated string starting at", begin);
            char e = s[i++];
            switch (e)
            {
                case '"': sb.Append('"'); break;
                case '\\': sb.Append('\\'); break;
                case '/': sb.Append('/'); break;
                case 'b': sb.Append('\b'); break;
                case 'f': sb.Append('\f'); break;
                case 'n': sb.Append('\n'); break;
                case 'r': sb.Append('\r'); break;
                case 't': sb.Append('\t'); break;
                case 'u':
                    // CPython's scanner wants a character after the four digits (the closing quote at least), else it reports the escape.
                    if (i + 4 >= s.Length || !int.TryParse(s.AsSpan(i, 4), NumberStyles.AllowHexSpecifier, CultureInfo.InvariantCulture, out var code))
                        throw new JsonDecodeError("Invalid \\uXXXX escape", i - 1);
                    sb.Append((char)code); // a pair of escaped surrogates makes one character, as in Python
                    i += 4;
                    break;
                default: throw new JsonDecodeError("Invalid \\escape", i - 2);
            }
        }
    }

    static DictValue ParseObject(string s, ref int i, int depth)
    {
        var d = new DictValue();
        i = SkipWs(s, i);
        if (i < s.Length && s[i] == '}') { i++; return d; }
        while (true)
        {
            if (i >= s.Length || s[i] != '"') throw new JsonDecodeError("Expecting property name enclosed in double quotes", i);
            i++;
            var key = ParseString(s, ref i);
            i = SkipWs(s, i);
            if (i >= s.Length || s[i] != ':') throw new JsonDecodeError("Expecting ':' delimiter", i);
            i = SkipWs(s, i + 1);
            d[key] = ParseValue(s, ref i, depth + 1);
            i = SkipWs(s, i);
            if (i < s.Length && s[i] == '}') { i++; return d; }
            if (i >= s.Length || s[i] != ',') throw new JsonDecodeError("Expecting ',' delimiter", i);
            i = SkipWs(s, i + 1);
        }
    }

    static ArrayValue ParseArray(string s, ref int i, int depth)
    {
        var a = new ArrayValue();
        i = SkipWs(s, i);
        if (i < s.Length && s[i] == ']') { i++; return a; }
        while (true)
        {
            a.Add(ParseValue(s, ref i, depth + 1));
            i = SkipWs(s, i);
            if (i < s.Length && s[i] == ']') { i++; return a; }
            if (i >= s.Length || s[i] != ',') throw new JsonDecodeError("Expecting ',' delimiter", i);
            i = SkipWs(s, i + 1);
        }
    }

    // ---- dumps -----------------------------------------------------------------------------------------------------

    /// <summary>
    /// <c>json.dumps(value, ensure_ascii=False, indent=indent or None)</c>: indent 0 (or less) is one line with ", " and ": " separators;
    /// otherwise one item per line, indented, with "," and ": ".
    /// </summary>
    public static string Dumps(object? value, int indent)
    {
        var sb = new StringBuilder();
        Write(sb, value, indent > 0 ? indent : null, 0);
        return sb.ToString();
    }

    static void Write(StringBuilder sb, object? value, int? indent, int level)
    {
        switch (value)
        {
            case null: sb.Append("null"); return;
            case bool b: sb.Append(b ? "true" : "false"); return;
            case string s: WriteString(sb, s); return;
            case double d: sb.Append(FloatJson(d)); return;
            case float f: sb.Append(FloatJson(f)); return;
            case BigInteger or long or int or short or byte or uint or ulong: sb.Append(Convert.ToString(value, CultureInfo.InvariantCulture)); return;
            case IEnumerable<KeyValuePair<string, object?>> dict:
            {
                var items = dict.ToList();
                if (items.Count == 0) { sb.Append("{}"); return; }
                sb.Append('{');
                for (int k = 0; k < items.Count; k++)
                {
                    if (k > 0) sb.Append(indent is null ? ", " : ",");
                    NewLine(sb, indent, level + 1);
                    WriteString(sb, items[k].Key);
                    sb.Append(": ");
                    Write(sb, items[k].Value, indent, level + 1);
                }
                NewLine(sb, indent, level);
                sb.Append('}');
                return;
            }
            case PyTuple t: WriteList(sb, t.Items, indent, level); return;
            case System.Collections.IList list: WriteList(sb, list.Cast<object?>().ToList(), indent, level); return;
            default: throw new PyException("TypeError", $"Object of type {value.GetType().Name} is not JSON serializable");
        }
    }

    static void WriteList(StringBuilder sb, IReadOnlyList<object?> items, int? indent, int level)
    {
        if (items.Count == 0) { sb.Append("[]"); return; }
        sb.Append('[');
        for (int k = 0; k < items.Count; k++)
        {
            if (k > 0) sb.Append(indent is null ? ", " : ",");
            NewLine(sb, indent, level + 1);
            Write(sb, items[k], indent, level + 1);
        }
        NewLine(sb, indent, level);
        sb.Append(']');
    }

    static void NewLine(StringBuilder sb, int? indent, int level)
    {
        if (indent is { } n) sb.Append('\n').Append(' ', n * level);
    }

    static string FloatJson(double d) =>
        double.IsNaN(d) ? "NaN" : double.IsPositiveInfinity(d) ? "Infinity" : double.IsNegativeInfinity(d) ? "-Infinity" : PyNumber.FloatRepr(d);

    /// <summary>Python's <c>encode_basestring</c>: quotes, backslash and the controls below 0x20 are escaped, everything else is written as is.</summary>
    static void WriteString(StringBuilder sb, string s)
    {
        sb.Append('"');
        foreach (var c in s)
        {
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                case '\b': sb.Append("\\b"); break;
                case '\f': sb.Append("\\f"); break;
                default:
                    if (c < 0x20) sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                    else sb.Append(c);
                    break;
            }
        }
        sb.Append('"');
    }
}
