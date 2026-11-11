using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries and SamplingSetupTests (combine, concat, average, zero out, areas,
// masks, timestep ranges); SDXL's size numbers are its published micro-conditioning.

/// <summary>An area of the picture a conditioning entry applies to: pixels (whole latent cells of 8) or, when fractional, fractions of the latent.</summary>
public readonly record struct CondArea(double X, double Y, double Width, double Height, bool Fractional = false)
{
    /// <summary>The area in latent cells: (x, y, width, height).</summary>
    public (int X, int Y, int Width, int Height) Resolve(int latentWidth, int latentHeight) => Fractional
        ? (Round(X * latentWidth), Round(Y * latentHeight), Round(Width * latentWidth), Round(Height * latentHeight))
        : ((int)X / 8, (int)Y / 8, (int)Width / 8, (int)Height / 8);

    static int Round(double v) => (int)Math.Round(v, MidpointRounding.ToEven);
}

/// <summary>SDXL's size conditioning in pixels: the original size, the crop's top-left corner and the target size.</summary>
public readonly record struct SdxlSize(int Width, int Height, int CropW, int CropH, int TargetWidth, int TargetHeight);

/// <summary>
/// One conditioning entry: the text context (rows x width, shape [1, rows, width]) and what limits or adds to it: strength, area, mask, the part
/// of the run it is active in, SDXL's pooled output and size, a ControlNet, and an inpainting model's extra inputs.
/// </summary>
public sealed record CondEntry(float[] Embeddings, int[] Shape)
{
    public float Strength { get; init; } = 1f;
    public CondArea? Area { get; init; }
    public MaskBatch? Mask { get; init; }
    public float StartPercent { get; init; }
    public float EndPercent { get; init; } = 1f;
    public float[]? Pooled { get; init; }
    public SdxlSize? Size { get; init; }
    public ControlUse? Control { get; init; }
    public bool ControlApplyToUncond { get; init; }
    public LatentBatch? ConcatLatent { get; init; }
    public MaskBatch? ConcatMask { get; init; }
}

/// <summary>CONDITIONING: a list of entries, each guiding the sampler on its own.</summary>
[Socket("CONDITIONING")]
public sealed class Conditioning
{
    public Conditioning(IReadOnlyList<CondEntry> entries) => Entries = entries;

    public Conditioning(float[] embeddings, int[] shape) : this([new CondEntry(embeddings, shape)]) { }

    public IReadOnlyList<CondEntry> Entries { get; }

    /// <summary>The first entry's context.</summary>
    public float[] Embeddings => Entries[0].Embeddings;

    /// <summary>The first entry's shape.</summary>
    public int[] Shape => Entries[0].Shape;

    /// <summary>A new conditioning with every entry changed.</summary>
    public Conditioning Map(Func<CondEntry, CondEntry> change) => new([.. Entries.Select(change)]);
}

[Node("ConditioningAverage", Category = "model/conditioning/transform", DisplayName = "Conditioning (Average)")]
public sealed class ConditioningAverage
{
    /// <summary>
    /// Each entry of <paramref name="conditioningTo"/> blended with the first entry of <paramref name="conditioningFrom"/>: to * strength + from *
    /// (1 - strength), the "from" rows cut or padded with zeros to the "to" rows (and the pooled outputs blended alike).
    /// </summary>
    public Conditioning Execute(Conditioning conditioningTo, Conditioning conditioningFrom,
        [Input(Min = 0.0, Max = 1.0, Step = 0.01)] float conditioningToStrength = 1f)
    {
        var from = conditioningFrom.Entries[0];
        float s = conditioningToStrength;
        return conditioningTo.Map(t =>
        {
            var blended = new float[t.Embeddings.Length];
            for (int i = 0; i < blended.Length; i++) blended[i] = t.Embeddings[i] * s + (i < from.Embeddings.Length ? from.Embeddings[i] : 0f) * (1 - s);
            float[]? pooled = t.Pooled;
            if (t.Pooled is not null && from.Pooled is not null)
            {
                pooled = new float[t.Pooled.Length];
                for (int i = 0; i < pooled.Length; i++) pooled[i] = t.Pooled[i] * s + from.Pooled[i] * (1 - s);
            }
            return t with { Embeddings = blended, Pooled = pooled };
        });
    }
}

[Node("ConditioningCombine", Category = "model/conditioning/transform", DisplayName = "Conditioning (Combine)")]
public sealed class ConditioningCombine
{
    /// <summary>Both conditionings' entries, each guiding on its own.</summary>
    public Conditioning Execute([Input(Name = "conditioning_1", Aliases = ["conditioning1"])] Conditioning conditioning1,
        [Input(Name = "conditioning_2", Aliases = ["conditioning2"])] Conditioning conditioning2) =>
        new([.. conditioning1.Entries, .. conditioning2.Entries]);
}

[Node("ConditioningConcat", Category = "model/conditioning/transform", DisplayName = "Conditioning (Concat)")]
public sealed class ConditioningConcat
{
    /// <summary>The rows of the first entry of <paramref name="conditioningFrom"/> appended to each entry of <paramref name="conditioningTo"/>.</summary>
    public Conditioning Execute(Conditioning conditioningTo, Conditioning conditioningFrom)
    {
        var from = conditioningFrom.Entries[0];
        return conditioningTo.Map(t =>
        {
            int width = t.Shape[^1];
            if (from.Shape[^1] != width)
                throw new ArgumentException($"Cannot concatenate conditionings of width {width} and {from.Shape[^1]}: they come from different text encoders.");
            return t with { Embeddings = [.. t.Embeddings, .. from.Embeddings], Shape = [1, (t.Embeddings.Length + from.Embeddings.Length) / width, width] };
        });
    }
}

[Node("ConditioningMultiply", Category = "model/conditioning/transform", DisplayName = "Conditioning (Multiply)")]
public sealed class ConditioningMultiply
{
    /// <summary>Every entry's context (and pooled output) times the multiplier.</summary>
    public Conditioning Execute(Conditioning conditioning, [Input(Min = -100.0, Max = 100.0, Step = 0.01, Default = 1.0)] double multiplier) =>
        conditioning.Map(e => e with
        {
            Embeddings = [.. e.Embeddings.Select(v => v * (float)multiplier)],
            Pooled = e.Pooled is null ? null : [.. e.Pooled.Select(v => v * (float)multiplier)],
        });
}

[Node("ConditioningSetArea", Category = "model/conditioning/transform", DisplayName = "Conditioning (Set Area)")]
public sealed class ConditioningSetArea
{
    /// <summary>Each entry limited to a rectangle in pixels (whole latent cells of 8), at that strength.</summary>
    public Conditioning Execute(Conditioning conditioning,
        [Input(Min = 64, Max = 16384, Step = 8)] int width = 64, [Input(Min = 64, Max = 16384, Step = 8)] int height = 64,
        [Input(Min = 0, Max = 16384, Step = 8)] int x = 0, [Input(Min = 0, Max = 16384, Step = 8)] int y = 0,
        [Input(Min = 0.0, Max = 10.0, Step = 0.01)] float strength = 1f) =>
        conditioning.Map(e => e with { Area = new CondArea(x, y, width, height), Strength = strength });
}

[Node("ConditioningSetAreaPercentage", Category = "model/conditioning/transform", DisplayName = "Conditioning (Set Area with Percentage)")]
public sealed class ConditioningSetAreaPercentage
{
    /// <summary>Each entry limited to a rectangle given in fractions of the picture, at that strength.</summary>
    public Conditioning Execute(Conditioning conditioning,
        [Input(Min = 0.0, Max = 1.0, Step = 0.01)] float width = 1f, [Input(Min = 0.0, Max = 1.0, Step = 0.01)] float height = 1f,
        [Input(Min = 0.0, Max = 1.0, Step = 0.01)] float x = 0f, [Input(Min = 0.0, Max = 1.0, Step = 0.01)] float y = 0f,
        [Input(Min = 0.0, Max = 10.0, Step = 0.01)] float strength = 1f) =>
        conditioning.Map(e => e with { Area = new CondArea(x, y, width, height, Fractional: true), Strength = strength });
}

[Node("ConditioningSetAreaStrength", Category = "model/conditioning/transform", DisplayName = "Conditioning (Set Area Strength)")]
public sealed class ConditioningSetAreaStrength
{
    public Conditioning Execute(Conditioning conditioning, [Input(Min = 0.0, Max = 10.0, Step = 0.01)] float strength = 1f) =>
        conditioning.Map(e => e with { Strength = strength });
}

[Node("ConditioningSetMask", Category = "model/conditioning/transform", DisplayName = "Conditioning (Set Mask)")]
public sealed class ConditioningSetMask
{
    public static string[] AreaModes() => ["default", "mask bounds"];

    /// <summary>
    /// Each entry weighted by the mask (resized to the latent when sampling), at that strength; "mask bounds" also limits it to the rectangle around
    /// the mask's non-zero values (as fractions of the mask, so it fits any latent size).
    /// </summary>
    public Conditioning Execute(Conditioning conditioning, MaskBatch mask, [Input(Min = 0.0, Max = 10.0, Step = 0.01)] float strength = 1f,
        [Input(OptionsProvider = nameof(AreaModes))] string setCondArea = "default")
    {
        CondArea? bounds = setCondArea == "mask bounds" ? Bounds(mask) : null;
        return conditioning.Map(e => e with { Mask = mask, Strength = strength, Area = bounds ?? e.Area });
    }

    static CondArea? Bounds(MaskBatch mask)
    {
        int w = mask.Width, h = mask.Height, x0 = w, y0 = h, x1 = -1, y1 = -1;
        for (int b = 0; b < mask.Batch; b++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    if (mask.Data[(b * h + y) * w + x] != 0)
                    {
                        x0 = Math.Min(x0, x); y0 = Math.Min(y0, y); x1 = Math.Max(x1, x); y1 = Math.Max(y1, y);
                    }
        if (x1 < 0) return null;
        return new CondArea((double)x0 / w, (double)y0 / h, (double)(x1 - x0 + 1) / w, (double)(y1 - y0 + 1) / h, Fractional: true);
    }
}

[Node("ConditioningSetTimestepRange", Category = "model/conditioning/transform", DisplayName = "ConditioningSetTimestepRange")]
public sealed class ConditioningSetTimestepRange
{
    /// <summary>Each entry active only from <paramref name="start"/> to <paramref name="end"/> of the run (fractions; both ends included).</summary>
    public Conditioning Execute(Conditioning conditioning, [Input(Min = 0.0, Max = 1.0, Step = 0.001)] float start = 0f,
        [Input(Min = 0.0, Max = 1.0, Step = 0.001)] float end = 1f) =>
        conditioning.Map(e => e with { StartPercent = start, EndPercent = end });
}

[Node("ConditioningZeroOut", Category = "model/conditioning/transform", DisplayName = "Conditioning Zero Out")]
public sealed class ConditioningZeroOut
{
    /// <summary>Each entry's context (and pooled output) set to zeros, keeping its shape.</summary>
    public Conditioning Execute(Conditioning conditioning) =>
        conditioning.Map(e => e with { Embeddings = new float[e.Embeddings.Length], Pooled = e.Pooled is null ? null : new float[e.Pooled.Length] });
}
