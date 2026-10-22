using SpawnGen.Editor;

namespace SpawnGen.Editor.Tests;

public class TextWrapTests
{
    static double Ten(string s) => s.Length * 10.0; // every character is 10 wide

    [Test]
    public void Short_lines_stay_and_newlines_break()
    {
        Assert.That(TextWrap.Wrap("one\ntwo\n\nfour", 100, Ten), Is.EqualTo(new[] { "one", "two", "", "four" }));
        Assert.That(TextWrap.Wrap("", 100, Ten), Is.EqualTo(new[] { "" }));
        Assert.That(TextWrap.Wrap("a\r\nb", 100, Ten), Is.EqualTo(new[] { "a", "b" }));
    }

    [Test]
    public void Long_lines_break_between_words_to_the_width()
    {
        Assert.That(TextWrap.Wrap("aaa bbb ccc ddd", 70, Ten), Is.EqualTo(new[] { "aaa bbb", "ccc ddd" }), "7 characters fit");
        Assert.That(TextWrap.Wrap("aaa bbb", 30, Ten), Is.EqualTo(new[] { "aaa", "bbb" }));
    }

    [Test]
    public void A_word_wider_than_the_line_is_cut_into_pieces()
    {
        Assert.That(TextWrap.Wrap("abcdefghij", 40, Ten), Is.EqualTo(new[] { "abcd", "efgh", "ij" }));
        Assert.That(TextWrap.Wrap("hi abcdefghij", 40, Ten), Is.EqualTo(new[] { "hi", "abcd", "efgh", "ij" }));
        Assert.That(TextWrap.Wrap("abcdef", 5, Ten), Is.EqualTo(new[] { "a", "b", "c", "d", "e", "f" }), "a width below one character still makes progress");
    }

    [Test]
    public void Every_line_fits_whatever_the_text()
    {
        var text = string.Join(" ", Enumerable.Range(1, 60).Select(i => new string('x', i % 13 + 1)));
        foreach (var line in TextWrap.Wrap(text, 95, Ten)) Assert.That(Ten(line), Is.LessThanOrEqualTo(95).Or.EqualTo(10), line);
    }
}
