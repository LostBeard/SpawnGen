using System.Numerics;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// Python semantics for the JSON-like values nodes pass around (<see cref="DictValue"/>, <see cref="ArrayValue"/>, str, int as long or BigInteger, float,
/// bool, None): truthiness, <c>==</c>, iteration, <c>float()</c>, <c>int()</c>, <c>min</c>, <c>round</c> and the type names Python puts in its messages.
/// Errors are <see cref="PyException"/> with Python's exception type.
/// </summary>
public static class PyValues
{
    /// <summary><c>d.get(key, fallback)</c>; <paramref name="dict"/> must be a dict (AttributeError otherwise, as <c>.get</c> on a list).</summary>
    public static object? Get(object? dict, string key, object? fallback = null) => dict is DictValue d
        ? d.TryGetValue(key, out var v) ? v : fallback
        : throw new PyException("AttributeError", $"'{TypeName(dict)}' object has no attribute 'get'");

    /// <summary><c>d[key]</c>: KeyError when it is missing, TypeError when <paramref name="dict"/> is not a dict.</summary>
    public static object? Item(object? dict, string key) => dict switch
    {
        DictValue d => d.TryGetValue(key, out var v) ? v : throw new PyException("KeyError", PyText.Repr(key)),
        ArrayValue => throw new PyException("TypeError", "list indices must be integers or slices, not str"),
        string => throw new PyException("TypeError", "string indices must be integers, not 'str'"),
        _ => throw new PyException("TypeError", $"'{TypeName(dict)}' object is not subscriptable"),
    };

    public static string TypeName(object? v) => v switch
    {
        null => "NoneType",
        bool => "bool",
        long or int or BigInteger => "int",
        double or float => "float",
        string => "str",
        ArrayValue => "list",
        DictValue => "dict",
        _ => v.GetType().Name,
    };

    public static bool Truthy(object? v) => v switch
    {
        null => false,
        bool b => b,
        long l => l != 0,
        int i => i != 0,
        BigInteger b => !b.IsZero,
        double d => d != 0,
        float f => f != 0,
        string s => s.Length > 0,
        ArrayValue a => a.Count > 0,
        DictValue d => d.Count > 0,
        _ => true,
    };

    /// <summary>An int (bool counts, as in Python).</summary>
    public static bool IsInt(object? v) => v is long or int or BigInteger or bool;

    public static bool IsNumber(object? v) => IsInt(v) || v is double or float;

    public static BigInteger Int(object? v) => v switch
    {
        long l => l,
        int i => i,
        BigInteger b => b,
        bool b => b ? 1 : 0,
        _ => throw new PyException("TypeError", $"'{TypeName(v)}' object cannot be interpreted as an integer"),
    };

    /// <summary>A number as a double (Python's <c>float(n)</c> for an int or float); TypeError for anything else.</summary>
    public static double Float(object? v) => v switch
    {
        double d => d,
        float f => f,
        long l => l,
        int i => i,
        bool b => b ? 1 : 0,
        BigInteger b => PyNumber.ToDouble(b),
        _ => throw new PyException("TypeError", $"unsupported operand type(s): '{TypeName(v)}'"),
    };

    /// <summary>Python's <c>float(v)</c>: numbers, and text float() accepts (ValueError otherwise); TypeError for other values.</summary>
    public static double ToFloat(object? v) => v switch
    {
        string s => PyNumber.TryParseFloat(s) ?? throw new PyException("ValueError", $"could not convert string to float: {PyText.Repr(s)}"),
        _ when IsNumber(v) => Float(v),
        _ => throw new PyException("TypeError", $"float() argument must be a string or a real number, not '{TypeName(v)}'"),
    };

    /// <summary>Python's <c>int(v)</c>: an int as it is, a float truncated, text parsed as a decimal integer.</summary>
    public static BigInteger ToInt(object? v) => v switch
    {
        double d => PyNumber.ToBigInteger(d),
        float f => PyNumber.ToBigInteger(f),
        string s => PyNumber.TryParseInt(s) ?? throw new PyException("ValueError", $"invalid literal for int() with base 10: {PyText.Repr(s)}"),
        _ when IsInt(v) => Int(v),
        _ => throw new PyException("TypeError", $"int() argument must be a string, a bytes-like object or a real number, not '{TypeName(v)}'"),
    };

    /// <summary>Python's <c>a + b</c>: int + int stays an int, a float makes a float, two strs join; TypeError for other mixes.</summary>
    public static object Add(object? a, object? b)
    {
        if (IsInt(a) && IsInt(b)) return Normal(Int(a) + Int(b));
        if (IsNumber(a) && IsNumber(b)) return Float(a) + Float(b);
        if (a is string s && b is string t) return s + t;
        if (a is string) throw new PyException("TypeError", $"can only concatenate str (not \"{TypeName(b)}\") to str");
        throw new PyException("TypeError", $"unsupported operand type(s) for +: '{TypeName(a)}' and '{TypeName(b)}'");
    }

    /// <summary>Python's <c>a / b</c> (true division) for numbers.</summary>
    public static double TrueDivide(object? a, object? b)
    {
        if (IsInt(a) && IsInt(b)) return PyNumber.TrueDivide(Int(a), Int(b));
        double d = Float(b);
        if (d == 0) throw new PyException("ZeroDivisionError", "division by zero");
        return Float(a) / d;
    }

    /// <summary>Python's <c>round(x)</c> of a float: ties to even, ValueError for NaN, OverflowError for infinity.</summary>
    public static long Round(double x)
    {
        if (double.IsNaN(x)) throw new PyException("ValueError", "cannot convert float NaN to integer");
        if (double.IsInfinity(x)) throw new PyException("OverflowError", "cannot convert float infinity to integer");
        return (long)PyNumber.RoundToInt(x);
    }

    /// <summary>An int that fits in a long as a long (how JSON integers are held).</summary>
    public static object Normal(BigInteger v) => v >= long.MinValue && v <= long.MaxValue ? (long)v : v;

    /// <summary>Python's <c>a &lt; b</c> for numbers.</summary>
    public static bool Less(object? a, object? b)
    {
        if (IsInt(a) && IsInt(b)) return Int(a) < Int(b);
        if (IsInt(a)) return PyNumber.CompareIntFloat(Int(a), Float(b)) is < 0;
        if (IsInt(b)) return PyNumber.CompareIntFloat(Int(b), Float(a)) is > 0;
        return Float(a) < Float(b);
    }

    /// <summary>Python's two-argument <c>min(a, b)</c>: <paramref name="a"/> unless <paramref name="b"/> is smaller.</summary>
    public static object? Min(object? a, object? b) => Less(b, a) ? b : a;

    /// <summary>Python's <c>a == b</c> for JSON-like values: 1 == 1.0 == True, dicts by their items in any order, lists item by item.</summary>
    public static bool Equal(object? a, object? b)
    {
        if (a is null || b is null) return a is null && b is null;
        if (IsNumber(a) && IsNumber(b))
        {
            if (IsInt(a) && IsInt(b)) return Int(a) == Int(b);
            if (IsInt(a)) return PyNumber.CompareIntFloat(Int(a), Float(b)) == 0;
            if (IsInt(b)) return PyNumber.CompareIntFloat(Int(b), Float(a)) == 0;
            return Float(a) == Float(b);
        }
        switch (a)
        {
            case string s: return b is string t && s == t;
            case ArrayValue xs:
                if (b is not ArrayValue ys || xs.Count != ys.Count) return false;
                for (int i = 0; i < xs.Count; i++) if (!Equal(xs[i], ys[i])) return false;
                return true;
            case DictValue xd:
                if (b is not DictValue yd || xd.Count != yd.Count) return false;
                foreach (var (k, v) in xd) if (!yd.TryGetValue(k, out var w) || !Equal(v, w)) return false;
                return true;
            default: return ReferenceEquals(a, b) || a.Equals(b);
        }
    }

    /// <summary>Python's <c>iter(v)</c>: a list's items, a str's characters, a dict's keys; TypeError for anything else.</summary>
    public static IEnumerable<object?> Iterate(object? v) => v switch
    {
        ArrayValue a => a,
        string s => PyText.CodePoints(s).Select(cp => (object?)char.ConvertFromUtf32(cp)),
        DictValue d => d.Keys,
        _ => throw new PyException("TypeError", $"'{TypeName(v)}' object is not iterable"),
    };

    /// <summary>Python's <c>len(v)</c> for str, list and dict.</summary>
    public static int Len(object? v) => v switch
    {
        ArrayValue a => a.Count,
        string s => PyText.Length(s),
        DictValue d => d.Count,
        _ => throw new PyException("TypeError", $"object of type '{TypeName(v)}' has no len()"),
    };
}
