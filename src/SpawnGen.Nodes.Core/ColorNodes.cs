using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// Python's <c>int(text, base)</c> for code that reads numbers the way Python does (Python documentation, "int"): surrounding whitespace, a sign,
/// single underscores between digits and, for base 16, an optional "0x" prefix are accepted; anything else is not a number.
/// </summary>
public static class PyInt
{
    /// <summary>The value, or an <see cref="ArgumentException"/> with Python's ValueError text when <paramref name="text"/> is not an integer.</summary>
    public static long Parse(string text, int @base = 10) =>
        TryParse(text, @base, out var value) ? value : throw new ArgumentException($"invalid literal for int() with base {@base}: {PyText.ReprString(text)}");

    /// <summary>False when <paramref name="text"/> is not an integer in <paramref name="base"/> (or does not fit a long).</summary>
    public static bool TryParse(string text, int @base, out long value)
    {
        value = 0;
        if (PyNumber.TryParseInt(text, @base) is not { } big || big < long.MinValue || big > long.MaxValue) return false;
        value = (long)big;
        return true;
    }
}

/// <summary>
/// The colour picker: a "#RRGGBB" or "#RRGGBBAA" string as an RGB integer, the "#RRGGBB" part, and the alpha as a fraction.
/// Behavior measured with ComfyUI (ColorToRGBInt, 2026-11-08):
/// <list type="bullet">
/// <item>The text must be 7 or 9 characters starting with '#', and everything after the '#' must read as one Python base-16 integer; otherwise
/// "Color must be in format #RRGGBB or #RRGGBBAA". So "#ff_fff", "#+fffff", "#0x1234" and "# fffff" pass that check.</item>
/// <item>The RGB integer comes from the first six characters after '#' read as three two-character Python base-16 integers, r * 65536 + g * 256 + b:
/// "#+fffff" gives 0x0fffff (r = int("+f") = 15), "#-fffff" gives -917505. When any pair is not an integer ("#ff_fff", "#0x1234") the colour is
/// white, 0xffffff.</item>
/// <item>The hex output is the first seven characters as given (case kept, "#12345_" possible).</item>
/// <item>With nine characters the last two are the alpha, int(pair, 16) / 255; a pair Python cannot read ("#123456_8", "#123456  ") fails with
/// "invalid literal for int() with base 16: '_8'". Seven characters: alpha 1.0.</item>
/// </list>
/// </summary>
[Node(Category = "utilities", DisplayName = "Color Picker", Description = "Return a color RGB integer value and hexadecimal representation.",
    OutputTypes = [null, "COLOR", null])]
public sealed class ColorToRGBInt
{
    const string FormatError = "Color must be in format #RRGGBB or #RRGGBBAA";

    public (int RgbInt, string Hex, double Alpha) Execute([Input(Color = true)] string color = "#ffffff")
    {
        if ((color.Length != 7 && color.Length != 9) || color[0] != '#' || !PyInt.TryParse(color[1..], 16, out _))
            throw new ArgumentException(FormatError);

        var hex = color[..7];
        int rgb = PyInt.TryParse(hex[1..3], 16, out var r) && PyInt.TryParse(hex[3..5], 16, out var g) && PyInt.TryParse(hex[5..7], 16, out var b)
            ? (int)(r * 65536 + g * 256 + b)
            : 0xFFFFFF;
        double alpha = color.Length == 9 ? PyInt.Parse(color[7..9], 16) / 255.0 : 1.0;
        return (rgb, hex, alpha);
    }
}
