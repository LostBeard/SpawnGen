namespace SpawnGen.Nodes.ML;

/// <summary>
/// ComfyUI's prompt emphasis: <c>(text:1.3)</c> weights the text, <c>(text)</c> is 1.1, each extra level of parentheses multiplies by 1.1 again, and
/// <c>\(</c> <c>\)</c> are literal brackets. Parsing follows ComfyUI's <c>token_weights</c> so a prompt means the same thing here and there.
/// </summary>
public static class PromptWeights
{
    /// <summary>Splits the prompt into pieces of text with their weight (unescaped; brackets that do not pair up stay text).</summary>
    public static List<(string Text, float Weight)> Parse(string prompt)
    {
        var escaped = prompt.Replace("\\)", "\0\u0001").Replace("\\(", "\0\u0002");
        var result = new List<(string, float)>();
        foreach (var (text, weight) in Weigh(escaped, 1f))
        {
            var plain = text.Replace("\0\u0001", ")").Replace("\0\u0002", "(");
            if (plain.Length > 0) result.Add((plain, weight));
        }
        return result;
    }

    /// <summary>True when the prompt has any emphasis, so the plain (cheaper, exact) path can be used when it has not.</summary>
    public static bool HasEmphasis(string prompt) => Parse(prompt).Any(p => p.Weight != 1f);

    static List<string> SplitParentheses(string text)
    {
        var result = new List<string>();
        var current = new System.Text.StringBuilder();
        int level = 0;
        foreach (char c in text)
        {
            if (c == '(')
            {
                if (level == 0)
                {
                    if (current.Length > 0) { result.Add(current.ToString()); current.Clear(); }
                    current.Append('(');
                }
                else current.Append(c);
                level++;
            }
            else if (c == ')')
            {
                level--;
                if (level == 0) { result.Add(current.ToString() + ")"); current.Clear(); }
                else current.Append(c);
            }
            else current.Append(c);
        }
        if (current.Length > 0) result.Add(current.ToString());
        return result;
    }

    static IEnumerable<(string, float)> Weigh(string text, float current)
    {
        foreach (var piece in SplitParentheses(text))
        {
            if (piece.Length >= 2 && piece[0] == '(' && piece[^1] == ')')
            {
                var inner = piece[1..^1];
                float weight = current * 1.1f;
                int colon = inner.LastIndexOf(':');
                if (colon > 0 && float.TryParse(inner[(colon + 1)..], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var explicitWeight))
                {
                    weight = explicitWeight; // an explicit weight replaces the one inherited from outside, as in ComfyUI
                    inner = inner[..colon];
                }
                foreach (var part in Weigh(inner, weight)) yield return part;
            }
            else yield return (piece, current);
        }
    }

    /// <summary>
    /// The token ids of a whole prompt in CLIP's layout (start, tokens, end, padding with end) and a weight per position (1 for the special tokens).
    /// <paramref name="tokenize"/> gives the plain tokens of a piece of text (no start/end).
    /// </summary>
    public static (int[] Ids, float[] Weights) Tokens(string prompt, Func<string, int[]> tokenize, int length, int start = 49406, int end = 49407)
    {
        var ids = new List<int> { start };
        var weights = new List<float> { 1f };
        foreach (var (text, weight) in Parse(prompt.Replace('\n', ' ')))
            foreach (int token in tokenize(text))
            {
                if (ids.Count >= length - 1) break; // room for the end token
                ids.Add(token);
                weights.Add(weight);
            }
        while (ids.Count < length) { ids.Add(end); weights.Add(1f); }
        return ([.. ids.Take(length)], [.. weights.Take(length)]);
    }

    /// <summary>
    /// Applies the weights to the encoder's output rows <paramref name="z"/> (tokens x width): each weighted row is moved away from the same row of the
    /// empty prompt's encoding by its weight, <c>(z - empty) * w + empty</c> (ComfyUI's rule), so weight 1 leaves it alone and weight 0 gives the empty prompt's row.
    /// </summary>
    public static void Apply(float[] z, float[] empty, float[] weights, int width) => Apply(z, empty, weights, width, 0, width);

    /// <summary>Like <see cref="Apply(float[], float[], float[], int)"/> on columns <paramref name="start"/> .. start + count of each row (one tower of SDXL's two).</summary>
    public static void Apply(float[] z, float[] empty, float[] weights, int width, int start, int count)
    {
        for (int t = 0; t < weights.Length; t++)
        {
            float w = weights[t];
            if (w == 1f) continue;
            for (int c = start; c < start + count; c++)
            {
                int i = t * width + c;
                z[i] = (z[i] - empty[i]) * w + empty[i];
            }
        }
    }
}
