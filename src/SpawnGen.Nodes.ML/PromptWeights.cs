using System.Globalization;
using System.Text;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's documented prompt syntax ("(text)" emphasis, "(text:1.2)" weights, a backslash before a
// bracket for a literal one), SpawnGen's PromptWeightsTests (ComfyUI's results for the same strings) and CLIP's token layout (start 49406,
// end and padding 49407).

/// <summary>Prompt emphasis: splitting a prompt into weighted pieces, laying their tokens out in CLIP chunks, and weighting the encoder's rows.</summary>
public static class PromptWeights
{
    public const int StartToken = 49406, EndToken = 49407;

    /// <summary>Pieces of 8 tokens or more are split at a chunk's end; shorter ones move whole to the next chunk.</summary>
    public const int MaxWordLength = 8;

    public readonly record struct Piece(string Text, float Weight);

    public sealed record Chunk(int[] Ids, float[] Weights);

    static bool IsEscape(string s, int i, int end) => s[i] == '\\' && i + 1 < end && s[i + 1] is '(' or ')';

    /// <summary>
    /// The prompt's pieces in order. "(text)" multiplies the weight around it by 1.1; "(text:w)" gives its text the weight w (the text after the
    /// group's last colon, read as a number; otherwise the colon is text and the group is a bare one); a backslash makes the bracket after it
    /// literal; a bracket without a partner is text. Empty pieces are left out.
    /// </summary>
    public static List<Piece> Parse(string prompt)
    {
        var partner = new Dictionary<int, int>();
        var open = new Stack<int>();
        for (int i = 0; i < prompt.Length; i++)
        {
            if (IsEscape(prompt, i, prompt.Length)) { i++; continue; }
            if (prompt[i] == '(') open.Push(i);
            else if (prompt[i] == ')' && open.Count > 0) partner[open.Pop()] = i;
        }
        var pieces = new List<Piece>();
        Walk(prompt, 0, prompt.Length, 1f, partner, pieces);
        return pieces;
    }

    static void Walk(string s, int start, int end, float weight, Dictionary<int, int> partner, List<Piece> pieces)
    {
        var text = new StringBuilder();
        void Flush()
        {
            if (text.Length > 0) pieces.Add(new Piece(text.ToString(), weight));
            text.Clear();
        }
        for (int i = start; i < end; i++)
        {
            if (IsEscape(s, i, end)) { text.Append(s[++i]); continue; }
            if (s[i] == '(' && partner.TryGetValue(i, out int close) && close < end)
            {
                Flush();
                int colon = LastTopLevelColon(s, i + 1, close, partner);
                if (colon >= 0 && ParseFloat(s[(colon + 1)..close]) is float w) Walk(s, i + 1, colon, w, partner, pieces);
                else Walk(s, i + 1, close, weight * 1.1f, partner, pieces);
                i = close;
                continue;
            }
            text.Append(s[i]);
        }
        Flush();
    }

    static int LastTopLevelColon(string s, int start, int end, Dictionary<int, int> partner)
    {
        int last = -1;
        for (int i = start; i < end; i++)
        {
            if (IsEscape(s, i, end)) { i++; continue; }
            if (s[i] == '(' && partner.TryGetValue(i, out int close) && close < end) { i = close; continue; }
            if (s[i] == ':') last = i;
        }
        return last;
    }

    /// <summary>Python's float(text) for what a weight can be: surrounding whitespace, a sign, digits with a point and an exponent, inf and nan.</summary>
    static float? ParseFloat(string text)
    {
        var t = text.Trim().ToLowerInvariant();
        if (t.Length == 0) return null;
        string body = t.TrimStart('+', '-');
        if (body is "inf" or "infinity") return t.StartsWith('-') ? float.NegativeInfinity : float.PositiveInfinity;
        if (body == "nan") return float.NaN;
        if (body.Length == 0 || !char.IsAsciiDigit(body[0]) && body[0] != '.') return null;
        if (body.Any(ch => !(char.IsAsciiDigit(ch) || ch is '.' or 'e' or '+' or '-'))) return null;
        return double.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out var v) ? (float)v : null;
    }

    /// <summary>Whether any piece has a weight other than 1.</summary>
    public static bool HasEmphasis(string prompt) => Parse(prompt).Any(p => p.Weight != 1f);

    /// <summary>
    /// The prompt as CLIP chunks of <paramref name="length"/> ids: start token, up to length - 2 prompt tokens, end token, then padding, each id
    /// with its piece's weight (1 for the special tokens). A piece that does not fit the current chunk starts the next one, unless it has
    /// <see cref="MaxWordLength"/> tokens or more: then it fills the chunk and goes on in the next.
    /// </summary>
    public static List<Chunk> Chunks(string prompt, Func<string, int[]> tokenize, int length = 77, int padToken = EndToken)
    {
        int room = length - 2;
        var chunks = new List<Chunk>();
        var ids = new List<int>();
        var weights = new List<float>();
        void Close()
        {
            var i = new List<int> { StartToken };
            var w = new List<float> { 1f };
            i.AddRange(ids);
            w.AddRange(weights);
            i.Add(EndToken);
            w.Add(1f);
            while (i.Count < length) { i.Add(padToken); w.Add(1f); }
            chunks.Add(new Chunk([.. i], [.. w]));
            ids.Clear();
            weights.Clear();
        }
        void Take(int[] tokens, int from, int count, float weight)
        {
            ids.AddRange(tokens[from..(from + count)]);
            weights.AddRange(Enumerable.Repeat(weight, count));
        }
        foreach (var piece in Parse(prompt))
        {
            var tokens = tokenize(piece.Text);
            int at = 0;
            while (at < tokens.Length)
            {
                int left = tokens.Length - at, free = room - ids.Count;
                if (left <= free) { Take(tokens, at, left, piece.Weight); at = tokens.Length; }
                else if (tokens.Length >= MaxWordLength || ids.Count == 0) { Take(tokens, at, free, piece.Weight); at += free; Close(); }
                else Close();
            }
        }
        if (ids.Count > 0 || chunks.Count == 0) Close();
        return chunks;
    }

    /// <summary>
    /// Weights the encoder's output rows (one per token position, <paramref name="width"/> values each): a row with weight w becomes
    /// (row - emptyRow) * w + emptyRow, where emptyRow is the empty prompt's row at that position; weight 1 leaves it as it is.
    /// </summary>
    public static void Apply(float[] z, float[] empty, float[] weights, int width) => Apply(z, empty, weights, width, 0, width);

    /// <summary>
    /// <see cref="Apply(float[], float[], float[], int)"/> on the columns <paramref name="column"/> to column + <paramref name="count"/> of each row only
    /// (one text tower's part of a joined output, such as SDXL's CLIP-L and CLIP-G halves); rows are <paramref name="width"/> values apart.
    /// </summary>
    public static void Apply(float[] z, float[] empty, float[] weights, int width, int column, int count)
    {
        for (int r = 0; r < weights.Length; r++)
        {
            float w = weights[r];
            if (w == 1f) continue;
            int start = r * width + column;
            for (int k = start; k < start + count; k++) z[k] = (z[k] - empty[k]) * w + empty[k];
        }
    }
}
