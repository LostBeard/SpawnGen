using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>Prompt emphasis against what ComfyUI's token_weights gives for the same strings.</summary>
public class PromptWeightsTests
{
    static string Show(string prompt) => string.Join("|", PromptWeights.Parse(prompt).Select(p => $"{p.Text}@{p.Weight:0.###}"));

    [Test]
    public void Plain_text_has_weight_one_and_no_emphasis()
    {
        Assert.That(Show("a red apple"), Is.EqualTo("a red apple@1"));
        Assert.That(PromptWeights.HasEmphasis("a red apple"), Is.False);
        Assert.That(PromptWeights.HasEmphasis(""), Is.False);
    }

    [Test]
    public void An_explicit_weight_and_bare_brackets()
    {
        Assert.That(Show("a (red:1.3) apple"), Is.EqualTo("a @1|red@1.3| apple@1"));
        Assert.That(Show("(x)"), Is.EqualTo("x@1.1"));
        Assert.That(Show("((x))"), Is.EqualTo("x@1.21"));
        Assert.That(Show("(((x)))"), Is.EqualTo("x@1.331"));
        Assert.That(Show("(x:0.5)"), Is.EqualTo("x@0.5"));
        Assert.That(PromptWeights.HasEmphasis("(x)"), Is.True);
    }

    [Test]
    public void Nested_weights_an_explicit_weight_replaces_the_outer_one_and_a_bare_bracket_multiplies()
    {
        Assert.That(Show("(a (b:0.5):1.2)"), Is.EqualTo("a @1.2|b@0.5"));
        Assert.That(Show("(a (b)):1.5"), Is.EqualTo("a @1.1|b@1.21|:1.5@1"), "a weight outside the brackets is just text");
        Assert.That(Show("((a:1.5) b)"), Is.EqualTo("a@1.5| b@1.1"));
    }

    [Test]
    public void Escaped_brackets_are_literal_and_unbalanced_ones_stay_text()
    {
        Assert.That(Show(@"a \(b\) c"), Is.EqualTo("a (b) c@1"));
        Assert.That(Show("(x"), Is.EqualTo("(x@1"));
        Assert.That(Show("x)"), Is.EqualTo("x)@1"));
        Assert.That(Show("(x:abc)"), Is.EqualTo("x:abc@1.1"), "a colon that is not followed by a number is part of the text");
    }

    // Pretend every word is one token whose id is its length.
    static int[] Tokenize(string text) => text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(w => w.Length).ToArray();

    [Test]
    public void Tokens_are_laid_out_like_CLIP_with_weights_per_position()
    {
        var chunks = PromptWeights.Chunks("ab (cde:1.5) f", Tokenize, 8);
        Assert.That(chunks, Has.Count.EqualTo(1));
        Assert.That(chunks[0].Ids, Is.EqualTo(new[] { 49406, 2, 3, 1, 49407, 49407, 49407, 49407 }));
        Assert.That(chunks[0].Weights, Is.EqualTo(new[] { 1f, 1f, 1.5f, 1f, 1f, 1f, 1f, 1f }));
        Assert.That(PromptWeights.Chunks("", Tokenize, 5).Single().Ids, Is.EqualTo(new[] { 49406, 49407, 49407, 49407, 49407 }), "the empty prompt");
    }

    [Test]
    public void A_long_piece_is_split_at_the_chunk_end_and_a_short_one_moves_to_the_next_chunk_as_in_ComfyUI()
    {
        // 8 tokens or more (max_word_length): split exactly where the chunk is full (room for 3 tokens in a chunk of 5).
        var large = PromptWeights.Chunks("a b c d e f g h", Tokenize, 5);
        Assert.That(large.Select(c => c.Ids), Is.EqualTo(new[]
        {
            new[] { 49406, 1, 1, 1, 49407 },
            new[] { 49406, 1, 1, 1, 49407 },
            new[] { 49406, 1, 1, 49407, 49407 },
        }));
        // A short weighted piece that does not fit is not split: the chunk is closed and padded, the piece starts the next one.
        var small = PromptWeights.Chunks("aa bb (c d:1.2)", Tokenize, 5);
        Assert.That(small.Select(c => c.Ids), Is.EqualTo(new[]
        {
            new[] { 49406, 2, 2, 49407, 49407 },
            new[] { 49406, 1, 1, 49407, 49407 },
        }));
        Assert.That(small[1].Weights, Is.EqualTo(new[] { 1f, 1.2f, 1.2f, 1f, 1f }));
    }

    [Test]
    public void Applying_weights_moves_each_row_away_from_the_empty_prompts_row()
    {
        float[] z = [2f, 4f, /* row 1 */ 10f, 20f];
        float[] empty = [1f, 1f, 5f, 5f];
        PromptWeights.Apply(z, empty, [1f, 2f], width: 2);
        Assert.That(z, Is.EqualTo(new[] { 2f, 4f, 15f, 35f }), "row 0 untouched; row 1 = (z - empty) * 2 + empty");

        float[] zero = [3f, 3f];
        PromptWeights.Apply(zero, [1f, 1f], [0f], 2);
        Assert.That(zero, Is.EqualTo(new[] { 1f, 1f }), "weight 0 gives the empty prompt's row");
    }
}
