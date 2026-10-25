using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

/// <summary>An area of the latent: in latent cells (1 cell = 8 pixels), or as fractions of the latent when <see cref="Fractional"/>.</summary>
public readonly record struct CondArea(double X, double Y, double Width, double Height, bool Fractional = false)
{
    /// <summary>The rectangle in cells of a latent of the given size, clamped to it (at least 1 cell wide and tall).</summary>
    public (int X, int Y, int W, int H) Resolve(int latentWidth, int latentHeight)
    {
        double sx = Fractional ? latentWidth : 1, sy = Fractional ? latentHeight : 1;
        int x = Math.Clamp((int)Math.Round(X * sx), 0, latentWidth - 1), y = Math.Clamp((int)Math.Round(Y * sy), 0, latentHeight - 1);
        int w = Math.Clamp((int)Math.Round(Width * sx), 1, latentWidth - x), h = Math.Clamp((int)Math.Round(Height * sy), 1, latentHeight - y);
        return (x, y, w, h);
    }
}

/// <summary>SDXL's size conditioning (ComfyUI's width, height, crop_w, crop_h, target_width, target_height), in pixels.</summary>
public readonly record struct SdxlSize(int Width, int Height, int CropW, int CropH, int TargetWidth, int TargetHeight);

/// <summary>
/// One piece of conditioning: text embeddings <c>[1, tokens, width]</c> plus where and when they apply. A plain prompt is one entry that applies
/// everywhere, always; <see cref="Area"/>, <see cref="Mask"/> and the percent range narrow it.
/// </summary>
public sealed record CondEntry(float[] Embeddings, int[] Shape)
{
    /// <summary>How strongly this entry counts where it applies (multiplies its area or mask weight).</summary>
    public float Strength { get; init; } = 1f;
    public CondArea? Area { get; init; }
    /// <summary>A mask (any size: it is resized to the latent) that weights where the entry applies.</summary>
    public MaskBatch? Mask { get; init; }
    /// <summary>How much of the sampling the entry is active for, as fractions from the start (0) to the end (1) of the run.</summary>
    public float StartPercent { get; init; }
    public float EndPercent { get; init; } = 1f;
    /// <summary>SDXL's pooled text embedding (ComfyUI's pooled_output); null for SD 1.x / 2.x.</summary>
    public float[]? Pooled { get; init; }
    /// <summary>SDXL's size numbers; null = the latent's size, no crop (ComfyUI's defaults).</summary>
    public SdxlSize? Size { get; init; }
}

/// <summary>What the model is asked about: one or more <see cref="CondEntry"/>. Several entries are blended by their area weights, like ComfyUI's Combine.</summary>
[Socket("CONDITIONING")]
public sealed class Conditioning(IReadOnlyList<CondEntry> entries)
{
    public Conditioning(float[] embeddings, int[] shape) : this([new CondEntry(embeddings, shape)]) { }

    public IReadOnlyList<CondEntry> Entries { get; } = entries;

    /// <summary>The embeddings of the first entry (a plain prompt has exactly one).</summary>
    public float[] Embeddings => Entries[0].Embeddings;
    public int[] Shape => Entries[0].Shape;

    public Conditioning Map(Func<CondEntry, CondEntry> change) => new([.. Entries.Select(change)]);
}

[Node(Category = "model/conditioning/transform", DisplayName = "Conditioning (Combine)")]
public sealed class ConditioningCombine
{
    /// <summary>Both conditionings at once: the sampler blends their predictions by area, so two prompts can steer different regions or share one.</summary>
    public Conditioning Execute([Input(Name = "conditioning_1", Aliases = ["conditioning1"])] Conditioning conditioning1, [Input(Name = "conditioning_2", Aliases = ["conditioning2"])] Conditioning conditioning2) =>
        new([.. conditioning1.Entries, .. conditioning2.Entries]);
}

[Node(Category = "model/conditioning/transform", DisplayName = "Conditioning (Concat)")]
public sealed class ConditioningConcat
{
    /// <summary>Appends the first entry of <c>conditioning_from</c> to every entry of <c>conditioning_to</c> along the token axis (a longer prompt).</summary>
    public Conditioning Execute(Conditioning conditioningTo, Conditioning conditioningFrom)
    {
        var from = conditioningFrom.Entries[0];
        return conditioningTo.Map(to =>
        {
            if (to.Shape.Length != 3 || from.Shape.Length != 3 || to.Shape[2] != from.Shape[2])
                throw new ArgumentException("The two conditionings have different embedding sizes and cannot be concatenated.");
            return to with { Embeddings = [.. to.Embeddings, .. from.Embeddings], Shape = [to.Shape[0], to.Shape[1] + from.Shape[1], to.Shape[2]] };
        });
    }
}

[Node(Category = "model/conditioning/transform", DisplayName = "Conditioning (Average)")]
public sealed class ConditioningAverage
{
    /// <summary>Blends two prompts' embeddings: <c>to * strength + from * (1 - strength)</c> (the first token set of <c>from</c> is padded or cut to the length of <c>to</c>).</summary>
    public Conditioning Execute(Conditioning conditioningTo, Conditioning conditioningFrom, [Input(Min = 0, Max = 1, Step = 0.01)] float conditioningToStrength = 1f)
    {
        var from = conditioningFrom.Entries[0];
        return conditioningTo.Map(to =>
        {
            var blended = new float[to.Embeddings.Length];
            for (int i = 0; i < blended.Length; i++)
            {
                float f = i < from.Embeddings.Length ? from.Embeddings[i] : 0f; // a shorter prompt is padded with zeros
                blended[i] = to.Embeddings[i] * conditioningToStrength + f * (1f - conditioningToStrength);
            }
            return to with { Embeddings = blended };
        });
    }
}

[Node(Category = "model/conditioning/transform", DisplayName = "Conditioning Zero Out")]
public sealed class ConditioningZeroOut
{
    public Conditioning Execute(Conditioning conditioning) => conditioning.Map(e => e with { Embeddings = new float[e.Embeddings.Length] });
}

[Node(Category = "model/conditioning/transform", DisplayName = "Conditioning (Set Area)")]
public sealed class ConditioningSetArea
{
    /// <summary>Restricts the prompt to a rectangle (in pixels; the latent works in 8-pixel cells).</summary>
    public Conditioning Execute(Conditioning conditioning, [Input(Min = 64, Max = 16384, Step = 8)] int width = 64, [Input(Min = 64, Max = 16384, Step = 8)] int height = 64,
        [Input(Min = 0, Max = 16384, Step = 8)] int x = 0, [Input(Min = 0, Max = 16384, Step = 8)] int y = 0, [Input(Min = 0, Max = 10, Step = 0.01)] float strength = 1f) =>
        conditioning.Map(e => e with { Area = new CondArea(x / 8, y / 8, Math.Max(1, width / 8), Math.Max(1, height / 8)), Strength = strength });
}

[Node(Category = "model/conditioning/transform", DisplayName = "Conditioning (Set Area with Percentage)")]
public sealed class ConditioningSetAreaPercentage
{
    /// <summary>Like Set Area, but the rectangle is given as fractions of the picture (resolved against the latent when sampling).</summary>
    public Conditioning Execute(Conditioning conditioning, [Input(Min = 0, Max = 1, Step = 0.01)] float width = 1f, [Input(Min = 0, Max = 1, Step = 0.01)] float height = 1f,
        [Input(Min = 0, Max = 1, Step = 0.01)] float x = 0f, [Input(Min = 0, Max = 1, Step = 0.01)] float y = 0f, [Input(Min = 0, Max = 10, Step = 0.01)] float strength = 1f) =>
        conditioning.Map(e => e with { Area = new CondArea(x, y, width, height, Fractional: true), Strength = strength });
}

[Node(Category = "model/conditioning/transform", DisplayName = "Conditioning (Set Mask)")]
public sealed class ConditioningSetMask
{
    public static string[] AreaModes() => ["default", "mask bounds"];

    public Conditioning Execute(Conditioning conditioning, MaskBatch mask, [Input(Min = 0, Max = 10, Step = 0.01)] float strength = 1f,
        [Input(OptionsProvider = nameof(AreaModes))] string setCondArea = "default") =>
        conditioning.Map(e => e with { Mask = mask, Strength = strength });
}

[Node(Category = "model/conditioning/transform", DisplayName = "ConditioningSetTimestepRange")]
public sealed class ConditioningSetTimestepRange
{
    public Conditioning Execute(Conditioning conditioning, [Input(Min = 0, Max = 1, Step = 0.001)] float start = 0f, [Input(Min = 0, Max = 1, Step = 0.001)] float end = 1f) =>
        conditioning.Map(e => e with { StartPercent = start, EndPercent = end });
}

[Node(Category = "model/conditioning/transform", DisplayName = "Conditioning (Set Area Strength)")]
public sealed class ConditioningSetAreaStrength
{
    /// <summary>Sets how strongly every entry of the conditioning counts where it applies (its area or mask weight is multiplied by it).</summary>
    public Conditioning Execute(Conditioning conditioning, [Input(Min = 0, Max = 10, Step = 0.01)] float strength = 1f) =>
        conditioning.Map(e => e with { Strength = strength });
}
