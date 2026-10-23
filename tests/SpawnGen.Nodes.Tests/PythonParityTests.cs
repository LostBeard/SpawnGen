using System.Globalization;
using System.Numerics;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// The Python ports (PyText, PyFormat, PyRegex, PyJson, PyExpression) and the text / utility nodes built on them, checked against values CPython and
/// simpleeval computed (<see cref="PythonVectors"/>). An expected value starting with "ERR:" means Python raised; the port must throw too.
/// </summary>
public class PythonParityTests
{
    static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

    static object? Value(string kind, string text) => kind switch
    {
        "i" => BigInteger.Parse(text, Inv),
        "f" => ParseRepr(text),
        "b" => text == "True",
        "n" => null,
        _ => text,
    };

    static double ParseRepr(string r) => r switch
    {
        "inf" => double.PositiveInfinity,
        "-inf" => double.NegativeInfinity,
        "nan" => double.NaN,
        _ => double.Parse(r, NumberStyles.Float, Inv),
    };

    /// <summary>Python's <c>float.hex()</c> text back to the double.</summary>
    static double FromHex(string h)
    {
        bool neg = h.StartsWith('-');
        if (neg) h = h[1..];
        h = h[2..]; // 0x
        int p = h.IndexOf('p');
        var mant = h[..p];
        int exp = int.Parse(h[(p + 1)..], Inv);
        var parts = mant.Split('.');
        var digits = parts[0] + (parts.Length > 1 ? parts[1] : "");
        var m = BigInteger.Parse("0" + digits, NumberStyles.AllowHexSpecifier, Inv);
        int fracBits = parts.Length > 1 ? 4 * parts[1].Length : 0;
        double v = Math.ScaleB((double)m, exp - fracBits);
        return neg ? -v : v;
    }

    static void Expect(string expected, Func<string> actual, string because)
    {
        if (expected.StartsWith("ERR:")) Assert.That(() => actual(), Throws.Exception, because);
        else Assert.That(actual(), Is.EqualTo(expected), because);
    }

    [Test]
    public void Format_spec_matches_cpython()
    {
        Assert.That(PythonVectors.Format, Has.Length.GreaterThan(2000));
        Assert.Multiple(() =>
        {
            foreach (var (kind, text, spec, expected) in PythonVectors.Format)
                Expect(expected, () => PyFormat.Format(Value(kind, text), spec), $"format({kind}:{text}, '{spec}')");
        });
    }

    [Test]
    public void Str_format_matches_cpython()
    {
        var values = new Dictionary<string, object?>
        {
            ["a"] = new BigInteger(3), ["b"] = 2.5, ["c"] = "xyz", ["d"] = true, ["e"] = null, ["f"] = -0.25, ["g"] = "h" + (char)0xE9,
        };
        object? Attr(object? v, string name) => name == "real" && v is BigInteger i ? i : throw new PyException("AttributeError", name);
        Assert.Multiple(() =>
        {
            foreach (var (template, expected) in PythonVectors.StrFormat)
                Expect(expected, () => PyFormat.FormatMap(template, k => (values.ContainsKey(k), values.GetValueOrDefault(k)), Attr), template);
        });
    }

    [Test]
    public void Float_repr_matches_cpython()
    {
        Assert.Multiple(() =>
        {
            foreach (var (bits, repr) in PythonVectors.FloatRepr)
                Assert.That(PyNumber.FloatRepr(FromHex(bits)), Is.EqualTo(repr), bits);
        });
    }

    [Test]
    public void Str_methods_match_cpython()
    {
        Assert.Multiple(() =>
        {
            foreach (var c in PythonVectors.StrMethods)
            {
                Assert.That(PyText.Upper(c.Text), Is.EqualTo(c.Upper), "upper: " + c.Text);
                Assert.That(PyText.Lower(c.Text), Is.EqualTo(c.Lower), "lower: " + c.Text);
                Assert.That(PyText.Capitalize(c.Text), Is.EqualTo(c.Capitalize), "capitalize: " + c.Text);
                Assert.That(PyText.Title(c.Text), Is.EqualTo(c.Title), "title: " + c.Text);
                Assert.That(PyText.Strip(c.Text), Is.EqualTo(c.Strip), "strip: " + c.Text);
                Assert.That(PyText.LStrip(c.Text), Is.EqualTo(c.LStrip), "lstrip: " + c.Text);
                Assert.That(PyText.RStrip(c.Text), Is.EqualTo(c.RStrip), "rstrip: " + c.Text);
                Assert.That(PyText.Length(c.Text), Is.EqualTo(c.Length), "len: " + c.Text);
                // The nodes are thin wrappers over these.
                Assert.That(new CaseConverter().Execute(c.Text, "Title Case"), Is.EqualTo(c.Title));
                Assert.That(new StringTrim().Execute(c.Text, "Both"), Is.EqualTo(c.Strip));
                Assert.That(new StringLength().Execute(c.Text), Is.EqualTo(c.Length));
            }
        });
    }

    [Test]
    public void Substring_and_replace_match_cpython()
    {
        Assert.Multiple(() =>
        {
            foreach (var (text, start, end, expected) in PythonVectors.Substring)
                Assert.That(new StringSubstring().Execute(text, start, end), Is.EqualTo(expected), $"[{start}:{end}] of {text}");
            foreach (var (text, find, replace, expected) in PythonVectors.Replace)
                Assert.That(new StringReplace().Execute(text, find, replace), Is.EqualTo(expected), $"{text}.replace({find}, {replace})");
        });
    }

    [Test]
    public void Regex_match_matches_python_re()
    {
        Assert.Multiple(() =>
        {
            foreach (var (text, pattern, ic, ml, da, expected) in PythonVectors.RegexMatch)
                Assert.That(new RegexMatchNode().Execute(text, pattern, ic, ml, da), Is.EqualTo(expected), $"re.search({pattern}, {text})");
        });
    }

    [Test]
    public void Regex_extract_matches_comfyui()
    {
        Assert.Multiple(() =>
        {
            foreach (var (text, pattern, mode, group, expected) in PythonVectors.RegexExtract)
            {
                // ComfyUI returns None for a group that did not take part; the node gives "".
                var want = expected == "NONE" ? "" : expected;
                Expect(want, () => new RegexExtract().Execute(text, pattern, mode, true, text.Contains('\n'), false, group), $"{mode} {group}: {pattern} on {text}");
            }
        });
    }

    [Test]
    public void Regex_replace_matches_python_re_sub()
    {
        Assert.Multiple(() =>
        {
            foreach (var (text, pattern, replace, count, expected) in PythonVectors.RegexSub)
                Expect(expected, () => new RegexReplaceNode().Execute(text, pattern, replace, true, false, false, count), $"re.sub({pattern}, {replace}, {text}, {count})");
        });
    }

    [Test]
    public void Json_extract_matches_comfyui()
    {
        Assert.Multiple(() =>
        {
            foreach (var (json, key, expected) in PythonVectors.JsonExtract)
                Assert.That(new JsonExtractString().Execute(json, key), Is.EqualTo(expected), $"{key} in {json}");
        });
    }

    [Test]
    public void Json_dumps_matches_python()
    {
        Assert.Multiple(() =>
        {
            foreach (var (json, indent, expected) in PythonVectors.JsonDump)
            {
                var value = PyJson.Loads(json);
                Assert.That(PyJson.Dumps(value, indent), Is.EqualTo(expected), $"indent {indent}: {json}");
                if (value is Engine.Types.DictValue d) Assert.That(new ConvertDictionaryToString().Execute(d, indent), Is.EqualTo(expected));
                if (value is Engine.Types.ArrayValue a) Assert.That(new ConvertArrayToString().Execute(a, indent), Is.EqualTo(expected));
            }
        });
    }

    [Test]
    public void Number_convert_matches_comfyui()
    {
        Assert.Multiple(() =>
        {
            foreach (var (kind, text, f, i) in PythonVectors.NumberConvert)
            {
                object value = kind switch { "i" => long.Parse(text, Inv), "f" => ParseRepr(text), "b" => text == "True", _ => text };
                Expect(f, () => PyNumber.FloatRepr(new ComfyNumberConvert().Execute(value).Item1), $"float of {kind}:{text}");
                Expect(i, () => new ComfyNumberConvert().Execute(value).Item2.ToString(Inv), $"int of {kind}:{text}");
            }
        });
    }

    [Test]
    public void Math_expression_matches_simpleeval()
    {
        var values = new List<KeyValuePair<string, object?>>
        {
            new("a", 7L), new("b", 2L), new("c", 2.5), new("d", true), new("e", 0L), new("f", -3.75),
        };
        Assert.Multiple(() =>
        {
            foreach (var (expr, f, i, b) in PythonVectors.MathExpression)
            {
                Expect(f, () => PyNumber.FloatRepr(PyExpression.EvaluateMath(expr, values).Float), "FLOAT of " + expr);
                Expect(i, () => PyExpression.EvaluateMath(expr, values).Int.ToString(Inv), "INT of " + expr);
                Expect(b, () => PyExpression.EvaluateMath(expr, values).Bool ? "True" : "False", "BOOL of " + expr);
            }
        });
    }

    [Test]
    public void Math_expression_sum_uses_python_312_compensated_float_sum()
    {
        // Python 3.12+ sums floats with Neumaier compensation: sum([0.1] * 10) is exactly 1.0 (3.11 gave 0.9999999999999999).
        var tenths = Enumerable.Range(0, 10).Select(k => new KeyValuePair<string, object?>(((char)('a' + k)).ToString(), 0.1)).ToList();
        Assert.That(PyExpression.EvaluateMath("sum(values)", tenths).Float, Is.EqualTo(1.0));
        Assert.That(PyExpression.EvaluateMath("a+b+c+d+e+f+g+h+i+j", tenths).Float, Is.EqualTo(0.9999999999999999), "plain + is not compensated");
    }

    [Test]
    public void Math_expression_refuses_what_simpleeval_refuses()
    {
        var none = new List<KeyValuePair<string, object?>> { new("a", 1L) };
        Assert.That(() => PyExpression.EvaluateMath("a.__class__", none), Throws.TypeOf<PyException>().With.Property(nameof(PyException.PyType)).EqualTo("FeatureNotAvailable"));
        Assert.That(() => PyExpression.EvaluateMath("import os", none), Throws.TypeOf<PyException>().With.Property(nameof(PyException.PyType)).EqualTo("FeatureNotAvailable"));
        Assert.That(() => PyExpression.EvaluateMath("a.format()", none), Throws.TypeOf<PyException>().With.Property(nameof(PyException.PyType)).EqualTo("FeatureNotAvailable"));
        Assert.That(() => PyExpression.EvaluateMath("open('x')", none), Throws.TypeOf<PyException>().With.Property(nameof(PyException.PyType)).EqualTo("FunctionNotDefined"));
        Assert.That(PyExpression.EvaluateMath("True or (1, 2)", none).Int, Is.EqualTo(1), "a refused node is only refused when it is evaluated");
    }
}
