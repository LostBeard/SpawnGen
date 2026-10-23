using System.Globalization;
using System.Numerics;
using System.Text;

namespace SpawnGen.Nodes.Core;

/// <summary>Python's <c>ValueError</c> / <c>TypeError</c> / <c>KeyError</c> / <c>IndexError</c> raised by the ported Python code; the message is Python's.</summary>
public sealed class PyException(string type, string message) : Exception($"{type}: {message}")
{
    public string PyType { get; } = type;
    public string PyMessage { get; } = message;
}

/// <summary>
/// Python's format mini-language: <c>format(value, spec)</c> for str, int, bool, float and None (fill, align, sign, z, #, 0, width, grouping, precision, type),
/// and <c>str.format(**values)</c> with field names, attributes, indices, conversions and nested specs.
/// </summary>
public static class PyFormat
{
    static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

    sealed record Spec(char? Fill, char? Align, char Sign, bool SignGiven, bool NoNegZero, bool Alt, int? Width, char? Grouping, int? Precision, char? Type);

    /// <param name="numeric">Numbers align right by default, so a leading '0' also means '=' alignment for them (Python 3.10+).</param>
    static Spec ParseSpec(string spec, string typeName, bool numeric)
    {
        var cps = PyText.CodePoints(spec);
        int i = 0, n = cps.Length;
        char? fill = null, align = null;
        static bool IsAlign(int c) => c is '<' or '>' or '=' or '^';
        if (n >= 2 && IsAlign(cps[1])) { fill = (char)cps[0]; if (cps[0] > 0xFFFF) throw new PyException("ValueError", "fill character outside the BMP is not supported"); align = (char)cps[1]; i = 2; }
        else if (n >= 1 && IsAlign(cps[0])) { align = (char)cps[0]; i = 1; }
        char sign = '-';
        bool signGiven = false;
        if (i < n && cps[i] is '+' or '-' or ' ') { sign = (char)cps[i++]; signGiven = true; }
        bool z = false;
        if (i < n && cps[i] == 'z') { z = true; i++; }
        bool alt = false;
        if (i < n && cps[i] == '#') { alt = true; i++; }
        if (fill is null && i < n && cps[i] == '0')
        {
            fill = '0';
            if (align is null && numeric) align = '=';
            i++;
        }
        int? width = ReadInt(cps, ref i);
        char? grouping = null;
        if (i < n && cps[i] is ',' or '_') grouping = (char)cps[i++];
        if (i < n && cps[i] is ',' or '_') throw new PyException("ValueError", "Cannot specify both ',' and '_'.");
        int? precision = null;
        if (i < n && cps[i] == '.')
        {
            i++;
            precision = ReadInt(cps, ref i) ?? throw new PyException("ValueError", "Format specifier missing precision");
        }
        if (n - i > 1) throw new PyException("ValueError", $"Invalid format specifier '{spec}' for object of type '{typeName}'");
        char? type = i < n ? (char)cps[i] : null;
        if (grouping is { } g && type is { } t && !(t is 'e' or 'E' or 'f' or 'F' or 'g' or 'G' or '%' or 'd' || (g == '_' && t is 'b' or 'o' or 'x' or 'X')))
            throw new PyException("ValueError", $"Cannot specify '{g}' with '{t}'.");
        return new Spec(fill, align, sign, signGiven, z, alt, width, grouping, precision, type);
    }

    static int? ReadInt(int[] cps, ref int i)
    {
        int start = i;
        long v = 0;
        while (i < cps.Length && cps[i] is >= '0' and <= '9')
        {
            v = v * 10 + (cps[i] - '0');
            if (v > int.MaxValue) throw new PyException("ValueError", "Too many decimal digits in format string");
            i++;
        }
        return i > start ? (int)v : null;
    }

    /// <summary>Python's <c>format(value, spec)</c>.</summary>
    public static string Format(object? value, string spec)
    {
        switch (value)
        {
            case string s: return FormatString(s, spec);
            case bool b: return spec.Length == 0 ? (b ? "True" : "False") : FormatInt(b ? 1 : 0, spec, "bool");
            case BigInteger bi: return FormatInt(bi, spec, "int");
            case long or int or short or byte: return FormatInt(new BigInteger(Convert.ToInt64(value, Inv)), spec, "int");
            case double d: return FormatFloat(d, spec);
            case float f: return FormatFloat(f, spec);
            default:
                if (spec.Length == 0) return PyText.Str(value);
                throw new PyException("TypeError", $"unsupported format string passed to {TypeName(value)}.__format__");
        }
    }

    internal static string TypeName(object? v) => v switch
    {
        null => "NoneType",
        bool => "bool",
        string => "str",
        BigInteger or long or int => "int",
        double or float => "float",
        PyTuple => "tuple",
        IEnumerable<KeyValuePair<string, object?>> => "dict",
        System.Collections.IList => "list",
        _ => v.GetType().Name,
    };

    static string FormatString(string s, string specText)
    {
        var spec = ParseSpec(specText, "str", numeric: false);
        if (spec.Type is { } t && t != 's') throw new PyException("ValueError", $"Unknown format code '{t}' for object of type 'str'");
        if (spec.Grouping is { } g) throw new PyException("ValueError", $"Cannot specify '{g}' with 's'.");
        if (spec.SignGiven) throw new PyException("ValueError", "Sign not allowed in string format specifier");
        if (spec.NoNegZero) throw new PyException("ValueError", "Negative zero coercion (z) not allowed in format specifier");
        if (spec.Alt) throw new PyException("ValueError", "Alternate form (#) not allowed in string format specifier");
        if (spec.Align == '=') throw new PyException("ValueError", "'=' alignment not allowed in string format specifier");
        if (spec.Precision is { } p && PyText.Length(s) > p) s = PyText.Slice(s, 0, p);
        return Pad(s, "", spec.Width, spec.Fill ?? ' ', spec.Align ?? '<');
    }

    /// <summary>Pads <paramref name="body"/> (after <paramref name="signPrefix"/>) to the width; '=' puts the padding between them.</summary>
    static string Pad(string body, string signPrefix, int? width, char fill, char align)
    {
        int len = PyText.Length(body) + signPrefix.Length;
        int pad = width is { } w && w > len ? w - len : 0;
        if (pad == 0) return signPrefix + body;
        return align switch
        {
            '<' => signPrefix + body + new string(fill, pad),
            '>' => new string(fill, pad) + signPrefix + body,
            '^' => new string(fill, pad / 2) + signPrefix + body + new string(fill, pad - pad / 2),
            _ => signPrefix + new string(fill, pad) + body, // '='
        };
    }

    /// <summary>Digits with a separator every <paramref name="every"/> from the right.</summary>
    static string Group(string digits, char sep, int every)
    {
        var sb = new StringBuilder();
        int first = digits.Length % every;
        if (first == 0) first = every;
        sb.Append(digits, 0, Math.Min(first, digits.Length));
        for (int i = first; i < digits.Length; i += every) sb.Append(sep).Append(digits, i, every);
        return sb.ToString();
    }

    /// <summary>
    /// Assembles sign + prefix + integer digits (grouped) + rest, then pads. Zero padding ('0' fill with '=') is grouped too, as Python does
    /// (<c>format(1234, '010,')</c> is "00,001,234").
    /// </summary>
    static string Assemble(Spec spec, bool negative, string prefix, string intDigits, string rest, char? sep, int every)
    {
        string sign = negative ? "-" : spec.Sign == '+' ? "+" : spec.Sign == ' ' ? " " : "";
        var align = spec.Align ?? '>';
        var fill = spec.Fill ?? ' ';
        if (sep is not { } s) return Pad(intDigits + rest, sign + prefix, spec.Width, fill, align);
        var grouped = Group(intDigits, s, every);
        if (align == '=' && fill == '0' && spec.Width is { } w)
        {
            int need = w - sign.Length - prefix.Length - PyText.Length(rest);
            var raw = intDigits;
            while (grouped.Length < need) { raw = "0" + raw; grouped = Group(raw, s, every); }
            return sign + prefix + grouped + rest;
        }
        return Pad(grouped + rest, sign + prefix, spec.Width, fill, align);
    }

    static string FormatInt(BigInteger v, string specText, string typeName)
    {
        var spec = ParseSpec(specText, typeName, numeric: true);
        char type = spec.Type ?? 'd';
        if (type is 'e' or 'E' or 'f' or 'F' or 'g' or 'G' or '%')
            return FormatFloat(PyNumber.ToDouble(v), specText);
        if (spec.Precision is not null) throw new PyException("ValueError", "Precision not allowed in integer format specifier");
        if (spec.NoNegZero) throw new PyException("ValueError", "Negative zero coercion (z) not allowed in integer format specifier");
        bool neg = v.Sign < 0;
        var a = BigInteger.Abs(v);
        string digits, prefix = "";
        int every = 3;
        switch (type)
        {
            case 'd': case 'n': digits = a.ToString(Inv); break;
            case 'x': case 'X':
                digits = ToBase(a, 16, type == 'X'); prefix = spec.Alt ? (type == 'X' ? "0X" : "0x") : ""; every = 4; break;
            case 'o': digits = ToBase(a, 8, false); prefix = spec.Alt ? "0o" : ""; every = 4; break;
            case 'b': digits = ToBase(a, 2, false); prefix = spec.Alt ? "0b" : ""; every = 4; break;
            case 'c':
                if (spec.Sign != '-') throw new PyException("ValueError", "Sign not allowed with integer format specifier 'c'");
                if (spec.Alt) throw new PyException("ValueError", "Alternate form (#) not allowed with integer format specifier 'c'");
                if (v < 0 || v > 0x10FFFF) throw new PyException("OverflowError", "%c arg not in range(0x110000)");
                return Pad(char.ConvertFromUtf32((int)v), "", spec.Width, spec.Fill ?? ' ', spec.Align ?? '>');
            default:
                throw new PyException("ValueError", $"Unknown format code '{type}' for object of type '{typeName}'");
        }
        return Assemble(spec, neg, prefix, digits, "", spec.Grouping, every);
    }

    static string ToBase(BigInteger a, int b, bool upper)
    {
        if (a.IsZero) return "0";
        var sb = new StringBuilder();
        const string lower = "0123456789abcdef", up = "0123456789ABCDEF";
        while (!a.IsZero) { a = BigInteger.DivRem(a, b, out var r); sb.Insert(0, (upper ? up : lower)[(int)r]); }
        return sb.ToString();
    }

    static string FormatFloat(double x, string specText)
    {
        var spec = ParseSpec(specText, "float", numeric: true);
        char? type = spec.Type;
        if (type is { } t0 && t0 is not ('e' or 'E' or 'f' or 'F' or 'g' or 'G' or 'n' or '%'))
            throw new PyException("ValueError", $"Unknown format code '{t0}' for object of type 'float'");
        bool upper = type is 'E' or 'F' or 'G';
        bool pct = type == '%';
        if (pct) x *= 100;
        bool neg = double.IsNegative(x) && !double.IsNaN(x);
        string body;
        if (double.IsNaN(x) || double.IsInfinity(x))
        {
            body = double.IsNaN(x) ? "nan" : "inf";
            if (upper) body = body.ToUpperInvariant();
            if (pct) body += "%";
            return Assemble(spec, neg, "", body, "", null, 3);
        }
        double ax = Math.Abs(x);
        string mode; // 'e', 'f', 'g' or 'r'
        int prec;
        bool addDot0 = false;
        switch (type)
        {
            case null:
                addDot0 = true;
                if (spec.Precision is { } pp) { mode = "g"; prec = pp; } else { mode = "r"; prec = 0; }
                break;
            case 'n': mode = "g"; prec = spec.Precision ?? 6; break;
            case 'e' or 'E': mode = "e"; prec = spec.Precision ?? 6; break;
            case 'f' or 'F' or '%': mode = "f"; prec = spec.Precision ?? 6; break;
            default: mode = "g"; prec = spec.Precision ?? 6; break;
        }
        string intDigits, rest;
        switch (mode)
        {
            case "f":
            {
                var digits = PyNumber.FixedDigits(ax, prec).ToString(Inv).PadLeft(prec + 1, '0');
                intDigits = digits[..^prec];
                if (prec == 0) intDigits = digits;
                rest = prec > 0 ? "." + digits[^prec..] : spec.Alt ? "." : "";
                if (spec.NoNegZero && digits.All(c => c == '0')) neg = false;
                break;
            }
            case "e":
            {
                var (d, exp) = PyNumber.SignificantDigits(ax, prec + 1);
                intDigits = d[..1];
                rest = (prec > 0 ? "." + d[1..] : spec.Alt ? "." : "") + (upper ? "E" : "e") + (exp < 0 ? "-" : "+") + Math.Abs(exp).ToString("00", Inv);
                if (spec.NoNegZero && d.All(c => c == '0')) neg = false;
                break;
            }
            default: // "g" and "r"
            {
                string d;
                int decpt;
                if (mode == "r") (d, decpt) = PyNumber.ShortestDigits(ax);
                else
                {
                    if (prec == 0) prec = 1;
                    (d, int e) = PyNumber.SignificantDigits(ax, prec);
                    decpt = ax == 0 ? 1 : e + 1;
                    if (!spec.Alt) { d = d.TrimEnd('0'); if (d.Length == 0) d = "0"; }
                }
                if (spec.NoNegZero && d.All(c => c == '0')) neg = false;
                bool useExp = mode == "r" ? decpt <= -4 || decpt > 16 : decpt <= -4 || decpt > (addDot0 ? prec - 1 : prec);
                if (useExp)
                {
                    int exp = decpt - 1;
                    intDigits = d[..1];
                    var frac = d[1..];
                    rest = (frac.Length > 0 ? "." + frac : spec.Alt ? "." : "") + (upper ? "E" : "e") + (exp < 0 ? "-" : "+") + Math.Abs(exp).ToString("00", Inv);
                }
                else
                {
                    // With '#' the digits were not trimmed, so all the significant digits stay and the point is always written.
                    var fixedText = PyNumber.FixedFromDigits(d, decpt, 0);
                    int dot = fixedText.IndexOf('.');
                    intDigits = dot < 0 ? fixedText : fixedText[..dot];
                    var frac = dot < 0 ? "" : fixedText[(dot + 1)..];
                    rest = frac.Length > 0 ? "." + frac : addDot0 ? ".0" : spec.Alt ? "." : "";
                }
                if (upper) rest = rest.ToUpperInvariant();
                break;
            }
        }
        if (pct) rest += "%";
        return Assemble(spec, neg, "", intDigits, rest, spec.Grouping, 3);
    }

    // ---- str.format ------------------------------------------------------------------------------------------------

    /// <summary>
    /// Python's <c>template.format(**values)</c>: <c>{name}</c>, <c>{name.attr}</c>, <c>{name[0]}</c>, <c>{name!r}</c>, <c>{name:spec}</c> (the spec may hold fields),
    /// <c>{{</c> / <c>}}</c> for braces. There are no positional values, so <c>{}</c> and <c>{0}</c> raise IndexError like Python's.
    /// </summary>
    public static string FormatMap(string template, Func<string, (bool Found, object? Value)> lookup, Func<object?, string, object?>? getAttr = null) =>
        Render(template, lookup, getAttr ?? DefaultAttr, 2);

    static object? DefaultAttr(object? value, string name) =>
        throw new PyException("AttributeError", $"'{TypeName(value)}' object has no attribute '{name}'");

    static string Render(string s, Func<string, (bool, object?)> lookup, Func<object?, string, object?> getAttr, int depth)
    {
        if (depth < 0) throw new PyException("ValueError", "Max string recursion exceeded");
        var sb = new StringBuilder();
        int i = 0, n = s.Length;
        int autoIndex = 0;
        while (i < n)
        {
            char c = s[i];
            if (c == '}')
            {
                if (i + 1 < n && s[i + 1] == '}') { sb.Append('}'); i += 2; continue; }
                throw new PyException("ValueError", "Single '}' encountered in format string");
            }
            if (c != '{') { sb.Append(c); i++; continue; }
            if (i + 1 < n && s[i + 1] == '{') { sb.Append('{'); i += 2; continue; }
            // A field: up to the matching '}' (braces nest inside the spec).
            int start = ++i, level = 1;
            while (i < n)
            {
                if (s[i] == '[' ) { while (i < n && s[i] != ']') i++; if (i >= n) break; }
                else if (s[i] == '{') level++;
                else if (s[i] == '}') { if (--level == 0) break; }
                i++;
            }
            if (i >= n) throw new PyException("ValueError", level > 0 ? "expected '}' before end of string" : "unmatched '{' in format spec");
            var field = s[start..i];
            i++; // past '}'
            sb.Append(RenderField(field, lookup, getAttr, depth, ref autoIndex));
        }
        return sb.ToString();
    }

    static string RenderField(string field, Func<string, (bool, object?)> lookup, Func<object?, string, object?> getAttr, int depth, ref int autoIndex)
    {
        // field_name [! conversion] [: spec]; '[' ... ']' in the name may hold ':' and '!'.
        int j = 0;
        while (j < field.Length)
        {
            char c = field[j];
            if (c == '[') { while (j < field.Length && field[j] != ']') j++; if (j < field.Length) j++; continue; }
            if (c == '{') throw new PyException("ValueError", "unexpected '{' in field name");
            if (c is ':' or '!') break;
            j++;
        }
        var name = field[..j];
        char? conversion = null;
        string spec = "";
        if (j < field.Length && field[j] == '!')
        {
            if (j + 1 >= field.Length) throw new PyException("ValueError", "end of string while looking for conversion specifier");
            conversion = field[j + 1];
            j += 2;
            if (j < field.Length && field[j] != ':') throw new PyException("ValueError", "expected ':' after conversion specifier");
        }
        if (j < field.Length && field[j] == ':') spec = field[(j + 1)..];

        object? value = ResolveField(name, lookup, getAttr, ref autoIndex);
        value = conversion switch
        {
            null => value,
            's' => PyText.Str(value),
            'r' => PyText.Repr(value),
            'a' => PyText.Ascii(value),
            { } other => throw new PyException("ValueError", $"Unknown conversion specifier {other}"),
        };
        if (spec.Contains('{')) spec = Render(spec, lookup, getAttr, depth - 1);
        return Format(value, spec);
    }

    static object? ResolveField(string name, Func<string, (bool, object?)> lookup, Func<object?, string, object?> getAttr, ref int autoIndex)
    {
        int k = 0;
        while (k < name.Length && name[k] is not ('.' or '[')) k++;
        var first = name[..k];
        if (first.Length == 0 || first.All(char.IsAsciiDigit))
        {
            int index = first.Length == 0 ? autoIndex++ : int.Parse(first, Inv);
            throw new PyException("IndexError", $"Replacement index {index} out of range for positional args tuple");
        }
        var (found, value) = lookup(first);
        if (!found) throw new PyException("KeyError", PyText.ReprString(first));
        while (k < name.Length)
        {
            if (name[k] == '.')
            {
                int e = k + 1;
                while (e < name.Length && name[e] is not ('.' or '[')) e++;
                var attr = name[(k + 1)..e];
                if (attr.Length == 0) throw new PyException("ValueError", "Empty attribute in format string");
                value = getAttr(value, attr);
                k = e;
            }
            else // '['
            {
                int e = name.IndexOf(']', k);
                if (e < 0) throw new PyException("ValueError", "Missing ']' in format string");
                var key = name[(k + 1)..e];
                if (key.Length == 0) throw new PyException("ValueError", "Empty attribute in format string");
                value = Index(value, key);
                k = e + 1;
                if (k < name.Length && name[k] is not ('.' or '[')) throw new PyException("ValueError", "Only '.' or '[' may follow ']' in format field specifier");
            }
        }
        return value;
    }

    static object? Index(object? value, string key)
    {
        bool numeric = key.All(char.IsAsciiDigit);
        switch (value)
        {
            case string s when numeric:
            {
                var cps = PyText.CodePoints(s);
                int i = int.Parse(key, Inv);
                if (i >= cps.Length) throw new PyException("IndexError", "string index out of range");
                return PyText.FromCodePoints([cps[i]]);
            }
            case IEnumerable<KeyValuePair<string, object?>> dict:
                foreach (var kv in dict) if (kv.Key == key) return kv.Value;
                throw new PyException("KeyError", numeric ? key : PyText.ReprString(key));
            case System.Collections.IList list when numeric:
            {
                int i = int.Parse(key, Inv);
                if (i >= list.Count) throw new PyException("IndexError", "list index out of range");
                return list[i];
            }
            case PyTuple t when numeric:
            {
                int i = int.Parse(key, Inv);
                if (i >= t.Items.Length) throw new PyException("IndexError", "tuple index out of range");
                return t.Items[i];
            }
            case string or System.Collections.IList or PyTuple:
                throw new PyException("TypeError", $"{TypeName(value)} indices must be integers or slices, not str");
            default:
                throw new PyException("TypeError", $"'{TypeName(value)}' object is not subscriptable");
        }
    }
}
