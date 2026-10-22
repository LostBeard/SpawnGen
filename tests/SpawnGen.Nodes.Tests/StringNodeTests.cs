using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

public class StringNodeTests
{
    [Test]
    public void Concatenate_substring_and_length()
    {
        Assert.That(new StringConcatenate().Execute("a", "b", ", "), Is.EqualTo("a, b"));
        var sub = new StringSubstring();
        Assert.That(sub.Execute("hello world", 0, 5), Is.EqualTo("hello"));
        Assert.That(sub.Execute("hello world", -5, 11), Is.EqualTo("world"), "negative start counts from the end");
        Assert.That(sub.Execute("hello", 3, 100), Is.EqualTo("lo"), "out of range is clamped");
        Assert.That(sub.Execute("hello", 4, 2), Is.EqualTo(""));
        Assert.That(new StringLength().Execute("héllo"), Is.EqualTo(5));
    }

    [Test]
    public void Case_and_trim()
    {
        var c = new CaseConverter();
        Assert.That(c.Execute("hello World", "UPPERCASE"), Is.EqualTo("HELLO WORLD"));
        Assert.That(c.Execute("hello World", "lowercase"), Is.EqualTo("hello world"));
        Assert.That(c.Execute("hello World", "Capitalize"), Is.EqualTo("Hello world"));
        Assert.That(c.Execute("the quick BROWN fox's", "Title Case"), Is.EqualTo("The Quick Brown Fox's"));
        var t = new StringTrim();
        Assert.That(t.Execute("  x  ", "Both"), Is.EqualTo("x"));
        Assert.That(t.Execute("  x  ", "Left"), Is.EqualTo("x  "));
        Assert.That(t.Execute("  x  ", "Right"), Is.EqualTo("  x"));
        Assert.That(() => t.Execute("x", "Middle"), Throws.ArgumentException);
    }

    [Test]
    public void Replace_contains_and_compare()
    {
        Assert.That(new StringReplace().Execute("a-b-c", "-", "+"), Is.EqualTo("a+b+c"));
        Assert.That(new StringReplace().Execute("abc", "", "x"), Is.EqualTo("abc"), "an empty search leaves the text alone");
        Assert.That(new StringContains().Execute("Hello", "ell", true), Is.True);
        Assert.That(new StringContains().Execute("Hello", "ELL", true), Is.False);
        Assert.That(new StringContains().Execute("Hello", "ELL", false), Is.True);
        var cmp = new StringCompare();
        Assert.That(cmp.Execute("foobar", "foo", "Starts With", true), Is.True);
        Assert.That(cmp.Execute("foobar", "BAR", "Ends With", false), Is.True);
        Assert.That(cmp.Execute("Foo", "foo", "Equal", true), Is.False);
        Assert.That(cmp.Execute("Foo", "foo", "Equal", false), Is.True);
    }

    [Test]
    public void Regex_match_and_replace()
    {
        Assert.That(new RegexMatchNode().Execute("abc123", @"\d+", true, false, false), Is.True);
        Assert.That(new RegexMatchNode().Execute("ABC", "abc", false, false, false), Is.False);
        Assert.That(new RegexMatchNode().Execute("a\nb", "^b$", false, true, false), Is.True, "multiline");
        Assert.That(new RegexMatchNode().Execute("a\nb", "a.b", false, false, true), Is.True, "dotall");
        var r = new RegexReplaceNode();
        Assert.That(r.Execute("a1b22", @"\d+", "#", true, false, false, 0), Is.EqualTo("a#b#"));
        Assert.That(r.Execute("a1b22", @"\d+", "#", true, false, false, 1), Is.EqualTo("a#b22"), "count limits the replacements");
        Assert.That(r.Execute("John Smith", @"(\w+) (\w+)", "$2, $1", true, false, false, 0), Is.EqualTo("Smith, John"));
    }

    [Test]
    public void Math_follows_python_rules_and_refuses_division_by_zero()
    {
        var m = new MathNode();
        Assert.That(m.Execute(7, 2, "add"), Is.EqualTo((9.0, 9L)));
        Assert.That(m.Execute(7, 2, "divide"), Is.EqualTo((3.5, 4L)), "int rounds half away from zero");
        Assert.That(m.Execute(7, 2, "floor divide").Float, Is.EqualTo(3));
        Assert.That(m.Execute(-7, 2, "floor divide").Float, Is.EqualTo(-4));
        Assert.That(m.Execute(-7, 3, "modulo").Float, Is.EqualTo(2), "the sign follows the divisor");
        Assert.That(m.Execute(2, 10, "power").Int, Is.EqualTo(1024));
        Assert.That(m.Execute(3, 5, "min").Float, Is.EqualTo(3));
        Assert.That(m.Execute(3, 5, "max").Float, Is.EqualTo(5));
        Assert.That(() => m.Execute(1, 0, "divide"), Throws.TypeOf<DivideByZeroException>());
        Assert.That(m.Execute(1, 0, "add").Float, Is.EqualTo(1));
    }

    [Test]
    public void Get_image_size_reports_width_height_and_batch()
    {
        var size = new GetImageSize().Execute(new ImageBatch(3, 5, 7, 3));
        Assert.That(size, Is.EqualTo((7L, 5L, 3L)));
    }
}
