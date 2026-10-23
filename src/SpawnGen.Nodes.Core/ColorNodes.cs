using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.Core;

/// <summary>Python's <c>int(text, base)</c> for the colour codes ComfyUI parses: spaces around, a sign, a 0x prefix (base 16) and single underscores between digits are allowed.</summary>
internal static class PyInt
{
    public static bool TryParse(string text, int numberBase, out long value)
    {
        value = 0;
        var s = text.Trim();
        int i = 0;
        bool negative = false;
        if (i < s.Length && s[i] is '+' or '-') negative = s[i++] == '-';
        bool prefixed = false;
        if (numberBase == 16 && i + 1 < s.Length && s[i] == '0' && s[i + 1] is 'x' or 'X') { i += 2; prefixed = true; }
        if (prefixed && i < s.Length && s[i] == '_') i++; // "0x_ff" is valid
        int digits = 0;
        bool lastUnderscore = false;
        for (; i < s.Length; i++)
        {
            char ch = s[i];
            if (ch == '_')
            {
                if (digits == 0 || lastUnderscore) return false;
                lastUnderscore = true;
                continue;
            }
            int d = ch is >= '0' and <= '9' ? ch - '0' : ch is >= 'a' and <= 'z' ? ch - 'a' + 10 : ch is >= 'A' and <= 'Z' ? ch - 'A' + 10 : 99;
            if (d >= numberBase) return false;
            value = checked(value * numberBase + d);
            digits++;
            lastUnderscore = false;
        }
        if (digits == 0 || lastUnderscore) return false;
        if (negative) value = -value;
        return true;
    }

    public static long Parse(string text, int numberBase) =>
        TryParse(text, numberBase, out var v) ? v : throw new FormatException($"invalid literal for int() with base {numberBase}: '{text}'");
}

[Node(Category = "utilities", DisplayName = "Color Picker", Description = "Return a color RGB integer value and hexadecimal representation.", OutputTypes = [null, "COLOR", null])]
public sealed class ColorToRGBInt
{
    /// <summary>
    /// A colour code <c>#RRGGBB</c> or <c>#RRGGBBAA</c> as the number 0xRRGGBB (what Empty Image and Image Color to Mask take), the <c>#RRGGBB</c> part, and the
    /// alpha (AA / 255, 1 without it).
    /// </summary>
    public (int RgbInt, string Hex, double Alpha) Execute([Input(Color = true)] string color = "#ffffff")
    {
        const string message = "Color must be in format #RRGGBB or #RRGGBBAA";
        if (color.Length is not (7 or 9) || color[0] != '#' || !PyInt.TryParse(color[1..], 16, out _)) throw new ArgumentException(message, nameof(color));
        double alpha = 1.0;
        if (color.Length == 9)
        {
            alpha = PyInt.Parse(color[7..9], 16) / 255.0;
            color = color[..7];
        }
        // comfy_extras/color_util.hex_to_rgb: white when a pair does not parse.
        var h = color.TrimStart('#');
        long r = 255, g = 255, b = 255;
        if (h.Length == 6 && PyInt.TryParse(h[0..2], 16, out var pr) && PyInt.TryParse(h[2..4], 16, out var pg) && PyInt.TryParse(h[4..6], 16, out var pb))
            (r, g, b) = (pr, pg, pb);
        return ((int)(r * 256 * 256 + g * 256 + b), color, alpha);
    }
}
