namespace SpawnGen.Editor;

/// <summary>Breaks a text into lines that fit a width: at newlines, then between words, and inside a word only when the word alone is wider than the line.</summary>
public static class TextWrap
{
    public static List<string> Wrap(string text, double width, Func<string, double> measure)
    {
        var lines = new List<string>();
        foreach (var raw in text.Replace("\r", "").Split('\n'))
        {
            if (raw.Length == 0 || measure(raw) <= width) { lines.Add(raw); continue; }
            int before = lines.Count;
            var line = "";
            foreach (var word in raw.Split(' '))
            {
                var candidate = line.Length == 0 ? word : line + " " + word;
                if (measure(candidate) <= width) { line = candidate; continue; }
                if (line.Length > 0) { lines.Add(line); line = ""; }
                // A word wider than the line is cut into pieces that fit.
                var rest = word;
                while (rest.Length > 0 && measure(rest) > width)
                {
                    int cut = Math.Max(1, rest.Length);
                    while (cut > 1 && measure(rest[..cut]) > width) cut--;
                    lines.Add(rest[..cut]);
                    rest = rest[cut..];
                }
                line = rest;
            }
            if (line.Length > 0) lines.Add(line);
            if (lines.Count == before) lines.Add("");
        }
        return lines;
    }
}
