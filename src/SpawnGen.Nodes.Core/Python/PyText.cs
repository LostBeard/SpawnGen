using System.Globalization;
using System.Numerics;
using System.Text;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// Python's <c>str</c> rules for the text nodes: lengths and slices count code points (an emoji is one character), full Unicode case mappings
/// (<c>"ß".upper()</c> is "SS", a final sigma lowers to "ς"), Python's whitespace for <c>strip</c>, and <c>str()</c> / <c>repr()</c> of values.
/// </summary>
public static class PyText
{
    // ---- code points -----------------------------------------------------------------------------------------------

    /// <summary>The text as code points (a surrogate pair is one, a lone surrogate stays itself), as Python sees a <c>str</c>.</summary>
    public static int[] CodePoints(string s)
    {
        var list = new List<int>(s.Length);
        for (int i = 0; i < s.Length; i++)
        {
            if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1])) { list.Add(char.ConvertToUtf32(s[i], s[i + 1])); i++; }
            else list.Add(s[i]);
        }
        return [.. list];
    }

    static void AppendCodePoint(StringBuilder sb, int cp)
    {
        if (cp < 0x10000) sb.Append((char)cp);
        else sb.Append(char.ConvertFromUtf32(cp));
    }

    public static string FromCodePoints(IEnumerable<int> cps)
    {
        var sb = new StringBuilder();
        foreach (var cp in cps) AppendCodePoint(sb, cp);
        return sb.ToString();
    }

    static bool HasSurrogates(string s)
    {
        foreach (var c in s) if (char.IsSurrogate(c)) return true;
        return false;
    }

    /// <summary><c>len(s)</c>: code points.</summary>
    public static int Length(string s) => HasSurrogates(s) ? CodePoints(s).Length : s.Length;

    /// <summary><c>s[start:end:step]</c> with Python's slice rules (None = <c>null</c>, negative indices count from the end, out of range is clamped).</summary>
    public static string Slice(string s, long? start, long? end, long? step = null)
    {
        if (!HasSurrogates(s) && (step ?? 1) == 1)
        {
            var (a, b, _, n) = SliceIndices(s.Length, start, end, 1);
            return n <= 0 ? "" : s.Substring((int)a, (int)(b - a));
        }
        var cps = CodePoints(s);
        return FromCodePoints(SliceItems(cps, start, end, step));
    }

    /// <summary>The items <c>seq[start:end:step]</c> picks.</summary>
    public static IEnumerable<T> SliceItems<T>(IReadOnlyList<T> seq, long? start, long? end, long? step)
    {
        var (a, _, st, n) = SliceIndices(seq.Count, start, end, step ?? 1);
        for (long i = 0, j = a; i < n; i++, j += st) yield return seq[(int)j];
    }

    /// <summary>Python's <c>slice.indices(length)</c> plus the number of items.</summary>
    public static (long Start, long Stop, long Step, long Count) SliceIndices(long length, long? start, long? stop, long step)
    {
        if (step == 0) throw new ArgumentException("slice step cannot be zero");
        long lower = step < 0 ? -1 : 0, upper = step < 0 ? length - 1 : length;
        long Fix(long? v, long dflt)
        {
            if (v is not { } x) return dflt;
            if (x < 0) { x += length; return x < lower ? lower : x; }
            return x > upper ? upper : x;
        }
        long a = Fix(start, step < 0 ? upper : lower), b = Fix(stop, step < 0 ? lower : upper);
        long count = step < 0 ? (b < a ? (a - b - 1) / -step + 1 : 0) : (a < b ? (b - a - 1) / step + 1 : 0);
        return (a, b, step, count);
    }

    /// <summary><c>s.replace(old, new)</c>: an empty <paramref name="find"/> puts <paramref name="replace"/> before every character and at the end.</summary>
    public static string Replace(string s, string find, string replace)
    {
        if (find.Length > 0) return s.Replace(find, replace, StringComparison.Ordinal);
        var sb = new StringBuilder(replace);
        foreach (var cp in CodePoints(s)) { AppendCodePoint(sb, cp); sb.Append(replace); }
        return sb.ToString();
    }

    // ---- whitespace ------------------------------------------------------------------------------------------------

    public static string Strip(string s) => RStrip(LStrip(s));

    public static string LStrip(string s)
    {
        int i = 0;
        while (i < s.Length && PyNumber.IsSpace(s[i])) i++;
        return s[i..];
    }

    public static string RStrip(string s)
    {
        int i = s.Length;
        while (i > 0 && PyNumber.IsSpace(s[i - 1])) i--;
        return s[..i];
    }

    // ---- case ------------------------------------------------------------------------------------------------------

    static readonly Dictionary<int, string> UpperFull = BuildUpper();
    static readonly Dictionary<int, string> TitleFull = BuildTitle();

    static Dictionary<int, string> BuildUpper()
    {
        var d = new Dictionary<int, string>
        {
            [0x00DF] = "SS", [0x0149] = "\u02BCN", [0x01F0] = "J\u030C", [0x0390] = "\u0399\u0308\u0301", [0x03B0] = "\u03A5\u0308\u0301",
            [0x0587] = "\u0535\u0552", [0x1E96] = "H\u0331", [0x1E97] = "T\u0308", [0x1E98] = "W\u030A", [0x1E99] = "Y\u030A", [0x1E9A] = "A\u02BE",
            [0x1F50] = "\u03A5\u0313", [0x1F52] = "\u03A5\u0313\u0300", [0x1F54] = "\u03A5\u0313\u0301", [0x1F56] = "\u03A5\u0313\u0342",
            [0x1FB2] = "\u1FBA\u0399", [0x1FB3] = "\u0391\u0399", [0x1FB4] = "\u0386\u0399", [0x1FB6] = "\u0391\u0342", [0x1FB7] = "\u0391\u0342\u0399",
            [0x1FBC] = "\u0391\u0399", [0x1FC2] = "\u1FCA\u0399", [0x1FC3] = "\u0397\u0399", [0x1FC4] = "\u0389\u0399", [0x1FC6] = "\u0397\u0342",
            [0x1FC7] = "\u0397\u0342\u0399", [0x1FCC] = "\u0397\u0399", [0x1FD2] = "\u0399\u0308\u0300", [0x1FD3] = "\u0399\u0308\u0301",
            [0x1FD6] = "\u0399\u0342", [0x1FD7] = "\u0399\u0308\u0342", [0x1FE2] = "\u03A5\u0308\u0300", [0x1FE3] = "\u03A5\u0308\u0301",
            [0x1FE4] = "\u03A1\u0313", [0x1FE6] = "\u03A5\u0342", [0x1FE7] = "\u03A5\u0308\u0342", [0x1FF2] = "\u1FFA\u0399", [0x1FF3] = "\u03A9\u0399",
            [0x1FF4] = "\u038F\u0399", [0x1FF6] = "\u03A9\u0342", [0x1FF7] = "\u03A9\u0342\u0399", [0x1FFC] = "\u03A9\u0399",
            [0xFB00] = "FF", [0xFB01] = "FI", [0xFB02] = "FL", [0xFB03] = "FFI", [0xFB04] = "FFL", [0xFB05] = "ST", [0xFB06] = "ST",
            [0xFB13] = "\u0544\u0546", [0xFB14] = "\u0544\u0535", [0xFB15] = "\u0544\u053B", [0xFB16] = "\u054E\u0546", [0xFB17] = "\u0544\u053D",
        };
        // Greek letters with ypogegrammeni / prosgegrammeni: capital letter + capital iota.
        foreach (var (from, to) in new[] { (0x1F80, 0x1F08), (0x1F90, 0x1F28), (0x1FA0, 0x1F68) })
            for (int i = 0; i < 16; i++) d[from + i] = ((char)(to + i % 8)).ToString() + "\u0399";
        return d;
    }

    static Dictionary<int, string> BuildTitle()
    {
        var d = new Dictionary<int, string>
        {
            [0x00DF] = "Ss", [0x01C4] = "\u01C5", [0x01C5] = "\u01C5", [0x01C6] = "\u01C5", [0x01C7] = "\u01C8", [0x01C8] = "\u01C8", [0x01C9] = "\u01C8",
            [0x01CA] = "\u01CB", [0x01CB] = "\u01CB", [0x01CC] = "\u01CB", [0x01F1] = "\u01F2", [0x01F2] = "\u01F2", [0x01F3] = "\u01F2", [0x0587] = "\u0535\u0582",
            [0x1FB2] = "\u1FBA\u0345", [0x1FB3] = "\u1FBC", [0x1FB4] = "\u0386\u0345", [0x1FB7] = "\u0391\u0342\u0345", [0x1FBC] = "\u1FBC",
            [0x1FC2] = "\u1FCA\u0345", [0x1FC3] = "\u1FCC", [0x1FC4] = "\u0389\u0345", [0x1FC7] = "\u0397\u0342\u0345", [0x1FCC] = "\u1FCC",
            [0x1FF2] = "\u1FFA\u0345", [0x1FF3] = "\u1FFC", [0x1FF4] = "\u038F\u0345", [0x1FF7] = "\u03A9\u0342\u0345", [0x1FFC] = "\u1FFC",
            [0xFB00] = "Ff", [0xFB01] = "Fi", [0xFB02] = "Fl", [0xFB03] = "Ffi", [0xFB04] = "Ffl", [0xFB05] = "St", [0xFB06] = "St",
            [0xFB13] = "\u0544\u0576", [0xFB14] = "\u0544\u0565", [0xFB15] = "\u0544\u056B", [0xFB16] = "\u054E\u0576", [0xFB17] = "\u0544\u056D",
        };
        // Georgian Mkhedruli letters have no title case form: they stay as they are (their upper case is Mtavruli).
        for (int c = 0x10D0; c <= 0x10FF; c++) if (c is not (0x10FB or 0x10FC)) d[c] = ((char)c).ToString();
        foreach (var from in new[] { 0x1F80, 0x1F90, 0x1FA0 })
            for (int i = 0; i < 16; i++) d[from + i] = ((char)(from + 8 + i % 8)).ToString();
        return d;
    }

    // Characters Python counts as cased (Lowercase / Uppercase properties) that have no case mapping and are not Lu / Ll / Lt (ª, º, modifier letters, ...).
    static readonly (int From, int To)[] OtherCased =
    [
        (0x00AA, 0x00AA), (0x00BA, 0x00BA), (0x02B0, 0x02B8), (0x02C0, 0x02C1), (0x02E0, 0x02E4), (0x037A, 0x037A), (0x1D2C, 0x1D6A), (0x1D78, 0x1D78),
        (0x1D9B, 0x1DBF), (0x2071, 0x2071), (0x207F, 0x207F), (0x2090, 0x209C), (0x2C7C, 0x2C7D), (0xA69C, 0xA69D), (0xA770, 0xA770), (0xA7F8, 0xA7F9),
        (0xAB5C, 0xAB5F), (0x10780, 0x10780), (0x10783, 0x10785), (0x10787, 0x107B0), (0x107B2, 0x107BA), (0x1F130, 0x1F149), (0x1F150, 0x1F169), (0x1F170, 0x1F189),
    ];

    static string SimpleUpper(int cp) => cp < 0x10000 ? char.ToUpperInvariant((char)cp).ToString() : char.ConvertFromUtf32(cp).ToUpperInvariant();
    static string SimpleLower(int cp) => cp < 0x10000 ? char.ToLowerInvariant((char)cp).ToString() : char.ConvertFromUtf32(cp).ToLowerInvariant();

    static string UpperOf(int cp) => UpperFull.TryGetValue(cp, out var s) ? s : SimpleUpper(cp);
    static string TitleOf(int cp) => TitleFull.TryGetValue(cp, out var s) ? s : UpperOf(cp);

    /// <summary>Python's <c>_PyUnicode_IsCased</c>.</summary>
    static bool IsCased(int cp)
    {
        if (cp is >= 0xD800 and <= 0xDFFF) return false;
        var cat = CharUnicodeInfo.GetUnicodeCategory(cp);
        if (cat is UnicodeCategory.UppercaseLetter or UnicodeCategory.LowercaseLetter or UnicodeCategory.TitlecaseLetter) return true;
        foreach (var (a, b) in OtherCased) if (cp >= a && cp <= b) return true;
        var s = cp < 0x10000 ? ((char)cp).ToString() : char.ConvertFromUtf32(cp);
        return SimpleUpper(cp) != s || SimpleLower(cp) != s;
    }

    /// <summary>Unicode Case_Ignorable (marks, format characters, modifier letters and symbols, and the apostrophe / period / colon family).</summary>
    static bool IsCaseIgnorable(int cp)
    {
        if (cp is 0x27 or 0x2E or 0x3A or 0x5E or 0x60 or 0xA8 or 0xAD or 0xAF or 0xB4 or 0xB7 or 0xB8 or 0x0387 or 0x05F4 or 0x2018 or 0x2019 or 0x2024 or 0x2027
            or 0xFE13 or 0xFE52 or 0xFE55 or 0xFF07 or 0xFF0E or 0xFF1A or 0xFF3E or 0xFF40) return true;
        if (cp is >= 0xD800 and <= 0xDFFF) return false;
        return CharUnicodeInfo.GetUnicodeCategory(cp) is UnicodeCategory.NonSpacingMark or UnicodeCategory.EnclosingMark or UnicodeCategory.Format
            or UnicodeCategory.ModifierLetter or UnicodeCategory.ModifierSymbol;
    }

    /// <summary>The lower case of the code point at <paramref name="i"/>, with Python's final-sigma rule and the dotted capital I.</summary>
    static string LowerAt(int[] cps, int i)
    {
        int cp = cps[i];
        if (cp == 0x0130) return "i\u0307";
        if (cp == 0x03A3)
        {
            int j = i - 1;
            while (j >= 0 && IsCaseIgnorable(cps[j])) j--;
            bool final = j >= 0 && IsCased(cps[j]);
            if (final)
            {
                j = i + 1;
                while (j < cps.Length && IsCaseIgnorable(cps[j])) j++;
                final = j == cps.Length || !IsCased(cps[j]);
            }
            return final ? "\u03C2" : "\u03C3";
        }
        return SimpleLower(cp);
    }

    public static string Lower(string s)
    {
        var cps = CodePoints(s);
        var sb = new StringBuilder(s.Length);
        for (int i = 0; i < cps.Length; i++) sb.Append(LowerAt(cps, i));
        return sb.ToString();
    }

    public static string Upper(string s)
    {
        var sb = new StringBuilder(s.Length);
        foreach (var cp in CodePoints(s)) sb.Append(UpperOf(cp));
        return sb.ToString();
    }

    /// <summary><c>s.capitalize()</c>: the first character in title case, the rest in lower case.</summary>
    public static string Capitalize(string s)
    {
        var cps = CodePoints(s);
        var sb = new StringBuilder(s.Length);
        for (int i = 0; i < cps.Length; i++) sb.Append(i == 0 ? TitleOf(cps[0]) : LowerAt(cps, i));
        return sb.ToString();
    }

    /// <summary><c>s.title()</c>: a character after a cased one is lowered, any other is title-cased (so "fox's" becomes "Fox'S", "3rd" becomes "3Rd").</summary>
    public static string Title(string s)
    {
        var cps = CodePoints(s);
        var sb = new StringBuilder(s.Length);
        bool previousCased = false;
        for (int i = 0; i < cps.Length; i++)
        {
            sb.Append(previousCased ? LowerAt(cps, i) : TitleOf(cps[i]));
            previousCased = IsCased(cps[i]);
        }
        return sb.ToString();
    }

    // ---- str() and repr() ------------------------------------------------------------------------------------------

    /// <summary>Python's <c>str(value)</c> for the values these nodes pass around (None, bool, int, float, str, list, tuple, dict).</summary>
    public static string Str(object? value) => value switch
    {
        string s => s,
        _ => Repr(value),
    };

    /// <summary>Python's <c>repr(value)</c>.</summary>
    public static string Repr(object? value)
    {
        switch (value)
        {
            case null: return "None";
            case bool b: return b ? "True" : "False";
            case string s: return ReprString(s);
            case double d: return PyNumber.FloatRepr(d);
            case float f: return PyNumber.FloatRepr(f);
            case BigInteger bi: return bi.ToString(CultureInfo.InvariantCulture);
            case long or int or short or byte or uint or ulong: return Convert.ToString(value, CultureInfo.InvariantCulture)!;
            case PyTuple t: return t.Items.Length == 1 ? "(" + Repr(t.Items[0]) + ",)" : "(" + string.Join(", ", t.Items.Select(Repr)) + ")";
            case IEnumerable<KeyValuePair<string, object?>> dict: return "{" + string.Join(", ", dict.Select(kv => ReprString(kv.Key) + ": " + Repr(kv.Value))) + "}";
            case System.Collections.IList list: return "[" + string.Join(", ", list.Cast<object?>().Select(Repr)) + "]";
            default: return value.ToString() ?? "";
        }
    }

    /// <summary>A string's <c>repr</c>: single quotes unless the text has a single quote and no double quote; non-printable characters escaped.</summary>
    public static string ReprString(string s, bool asciiOnly = false)
    {
        char quote = s.Contains('\'') && !s.Contains('"') ? '"' : '\'';
        var sb = new StringBuilder(s.Length + 2).Append(quote);
        foreach (var cp in CodePoints(s))
        {
            if (cp == quote || cp == '\\') sb.Append('\\').Append((char)cp);
            else if (cp == '\t') sb.Append("\\t");
            else if (cp == '\n') sb.Append("\\n");
            else if (cp == '\r') sb.Append("\\r");
            else if (cp < 0x20 || cp == 0x7F) sb.Append("\\x").Append(cp.ToString("x2"));
            else if (cp < 0x7F) sb.Append((char)cp);
            else if (!asciiOnly && IsPrintable(cp)) AppendCodePoint(sb, cp);
            else if (cp <= 0xFF) sb.Append("\\x").Append(cp.ToString("x2"));
            else if (cp <= 0xFFFF) sb.Append("\\u").Append(cp.ToString("x4"));
            else sb.Append("\\U").Append(cp.ToString("x8"));
        }
        return sb.Append(quote).ToString();
    }

    /// <summary>Python's <c>ascii(value)</c>: the repr with every non-ASCII character escaped.</summary>
    public static string Ascii(object? value)
    {
        var r = Repr(value);
        var sb = new StringBuilder(r.Length);
        foreach (var cp in CodePoints(r))
        {
            if (cp < 0x80) sb.Append((char)cp);
            else if (cp <= 0xFF) sb.Append("\\x").Append(cp.ToString("x2"));
            else if (cp <= 0xFFFF) sb.Append("\\u").Append(cp.ToString("x4"));
            else sb.Append("\\U").Append(cp.ToString("x8"));
        }
        return sb.ToString();
    }

    /// <summary>Python's <c>str.isprintable</c> for one code point: not a control, format, surrogate, private-use, unassigned or separator character (space is printable).</summary>
    public static bool IsPrintable(int cp)
    {
        if (cp == ' ') return true;
        if (cp is >= 0xD800 and <= 0xDFFF) return false;
        return CharUnicodeInfo.GetUnicodeCategory(cp) is not (UnicodeCategory.Control or UnicodeCategory.Format or UnicodeCategory.Surrogate
            or UnicodeCategory.PrivateUse or UnicodeCategory.OtherNotAssigned or UnicodeCategory.LineSeparator or UnicodeCategory.ParagraphSeparator
            or UnicodeCategory.SpaceSeparator);
    }
}

/// <summary>A Python tuple (what <c>as_integer_ratio()</c> returns in a math expression).</summary>
public sealed record PyTuple(object?[] Items);
