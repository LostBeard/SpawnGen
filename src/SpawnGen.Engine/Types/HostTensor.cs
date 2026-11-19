using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Types;

/// <summary>Dense float32 tensor on the host.</summary>
public class HostTensor
{
    public HostTensor(int[] shape, float[]? data = null)
    {
        Shape = shape;
        var length = 1L;
        foreach (var d in shape) length *= d;
        if (length > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(shape), "Tensor is too large.");
        Data = data ?? new float[length];
        if (Data.Length != length)
            throw new ArgumentException($"Data length {Data.Length} does not match shape [{string.Join(",", shape)}].", nameof(data));
    }

    public int[] Shape { get; }
    public float[] Data { get; }

    /// <summary>The kind and shape ("ImageBatch [1, 512, 512, 3]"): how Preview as Text shows a tensor, also inside a list.</summary>
    public override string ToString() => $"{GetType().Name} [{string.Join(", ", Shape)}]";
}

/// <summary>A batch of images, NHWC, RGB(A), values in 0..1 (same layout as ComfyUI's IMAGE).</summary>
[Socket("IMAGE")]
public sealed class ImageBatch : HostTensor
{
    public ImageBatch(int batch, int height, int width, int channels = 3, float[]? data = null)
        : base([batch, height, width, channels], data) { }

    public int Batch => Shape[0];
    public int Height => Shape[1];
    public int Width => Shape[2];
    public int Channels => Shape[3];
}

/// <summary>
/// A batch of latents, NCHW, or NCFHW for a video model's latents (frames between channels and height), or NCL for an audio model's (ACE-Step:
/// [batch, 64, frames]; LTX-2's audio: [batch, 8, frames, 16] typed "audio"), the same layouts as ComfyUI's LATENT "samples", or LTX-2's audio +
/// video latent (ComfyUI's NestedTensor), held packed as [batch, 1, values] with its parts' shapes. Only nodes marked
/// <see cref="VideoLatentsAttribute"/> are given a video latent, only nodes marked <see cref="AudioLatentsAttribute"/> an audio latent and only
/// nodes marked <see cref="NestedLatentsAttribute"/> an audio + video one; the others refuse them before they run.
/// </summary>
[Socket("LATENT")]
public sealed class LatentBatch : HostTensor
{
    public LatentBatch(int batch, int channels, int height, int width, float[]? data = null)
        : base([batch, channels, height, width], data) { }

    /// <summary>A video latent [batch, channels, frames, height, width].</summary>
    public LatentBatch(int batch, int channels, int frames, int height, int width, float[]? data = null)
        : base([batch, channels, frames, height, width], data) { }

    /// <summary>A latent of any layout above (the keys, mask and parts are set with an object initializer).</summary>
    public LatentBatch(int[] shape, float[]? data = null) : base(shape, data) { }

    /// <summary>An audio latent [batch, channels, length] (ComfyUI's 3-D LATENT; ACE-Step 1.5: 64 channels, 25 frames per second).</summary>
    public static LatentBatch Audio(int batch, int channels, int length, float[]? data = null) => new([batch, channels, length], data);

    public int Batch => Shape[0];
    public int Channels => Shape[1];
    /// <summary>The rows: a 3-D latent (ACE-Step's audio, a packed audio + video one) has one (its length is the <see cref="Width"/>).</summary>
    public int Height => Shape.Length == 3 ? 1 : Shape[^2];
    public int Width => Shape[^1];

    /// <summary>Whether this is an audio latent: 3-D (batch, channels, length: ACE-Step) or 4-D typed "audio" (LTX-2: batch, channels, frames, bins).</summary>
    public bool IsAudio => !IsNested && (Shape.Length == 3 || (Shape.Length == 4 && Type == "audio"));

    /// <summary>Whether this is a video latent (5-D, with a frames dimension).</summary>
    public bool IsVideo => !IsNested && Shape.Length == 5;

    /// <summary>Whether this is an audio + video latent (LTX-2's; ComfyUI's NestedTensor), packed as [batch, 1, values]; <see cref="Parts"/> has the parts' shapes.</summary>
    public bool IsNested => Parts is not null;

    /// <summary>The parts' shapes of a packed audio + video latent, in order (each with this latent's batch); null for any other latent.</summary>
    public IReadOnlyList<int[]>? Parts { get; init; }

    /// <summary>
    /// A packed audio + video latent's noise mask, per part in <see cref="Parts"/>' order, each as that part's own <see cref="NoiseMask"/> (the video
    /// part's frames stacked); a null entry is all ones (ComfyUI's audio part of the mask is all ones, measured). Null: no noise mask.
    /// </summary>
    public IReadOnlyList<MaskBatch?>? PartMasks { get; init; }

    /// <summary>ComfyUI's LATENT "type" key ("audio" on ACE-Step's and LTX-2's audio latents and on what LTX-2's audio + video nodes pass on).</summary>
    public string? Type { get; init; }

    /// <summary>ComfyUI's LATENT "downscale_ratio_spacial" key (LTX-2's video latents: 32).</summary>
    public int? DownscaleRatioSpacial { get; init; }

    /// <summary>ComfyUI's LATENT "downscale_ratio_temporal" key (ACE-Step 1.5's audio latent: 1764).</summary>
    public int? DownscaleRatioTemporal { get; init; }

    /// <summary>
    /// LTX-2's audio + video latent from its parts (same batch), packed per batch item: the video part's values, then the audio part's, the order
    /// ComfyUI's model input has (measured: [batch, 1, video + audio values]). The keys are the parts' merged, the audio part's winning.
    /// </summary>
    public static LatentBatch Nested(LatentBatch video, LatentBatch audio)
    {
        if (video.Batch != audio.Batch) throw new ArgumentException($"The video latent's batch ({video.Batch}) and the audio latent's ({audio.Batch}) differ.");
        int vn = video.ItemSize, an = audio.ItemSize, b = video.Batch;
        var data = new float[b * (vn + an)];
        for (int i = 0; i < b; i++)
        {
            Array.Copy(video.Data, i * vn, data, i * (vn + an), vn);
            Array.Copy(audio.Data, i * an, data, i * (vn + an) + vn, an);
        }
        // ComfyUI's NestedTensor noise mask (measured with a keyframe guide, 2026-11-18): the video part's per-frame mask and the audio part's all ones.
        if (audio.NoiseMask is not null) throw new NotSupportedException("An audio latent with a noise mask in an audio + video latent is not measured yet in SpawnGen.");
        return new LatentBatch([b, 1, vn + an], data)
        {
            Parts = [[.. video.Shape], [.. audio.Shape]],
            PartMasks = video.NoiseMask is null ? null : [video.NoiseMask, null],
            Type = audio.Type ?? video.Type,
            DownscaleRatioSpacial = audio.DownscaleRatioSpacial ?? video.DownscaleRatioSpacial,
            DownscaleRatioTemporal = audio.DownscaleRatioTemporal ?? video.DownscaleRatioTemporal,
        };
    }

    /// <summary>The parts of a packed audio + video latent, each carrying this latent's keys (as ComfyUI's Separate AV Latent gives them, measured).</summary>
    public IReadOnlyList<LatentBatch> SplitParts()
    {
        if (Parts is null) throw new InvalidOperationException("This latent is not an audio + video latent.");
        int b = Batch, total = Width, offset = 0;
        var parts = new List<LatentBatch>();
        foreach (var shape in Parts)
        {
            int n = 1;
            for (int d = 1; d < shape.Length; d++) n *= shape[d];
            var data = new float[b * n];
            for (int i = 0; i < b; i++) Array.Copy(Data, i * total + offset, data, i * n, n);
            parts.Add(new LatentBatch([b, .. shape[1..]], data)
            {
                Type = Type, DownscaleRatioSpacial = DownscaleRatioSpacial, DownscaleRatioTemporal = DownscaleRatioTemporal,
                NoiseMask = PartMasks is { } masks ? masks[parts.Count] : null,
            });
            offset += n;
        }
        return parts;
    }

    /// <summary>The latent frames of a video latent; 1 for a picture latent.</summary>
    public int Frames => IsVideo ? Shape[2] : 1;

    /// <summary>One batch item's values: channels x frames x height x width.</summary>
    public int ItemSize => Channels * Frames * Height * Width;

    /// <summary>An empty latent shaped like <paramref name="template"/> (picture or video) with another batch size and these values.</summary>
    public static LatentBatch Shaped(LatentBatch template, int batch, float[]? data = null) =>
        new([batch, .. template.Shape[1..]], data)
        {
            Parts = template.Parts?.Select(p => (int[])[batch, .. p[1..]]).ToArray(),
            PartMasks = template.PartMasks,
            Type = template.Type, DownscaleRatioSpacial = template.DownscaleRatioSpacial, DownscaleRatioTemporal = template.DownscaleRatioTemporal,
        };

    /// <summary>A copy of these values with <paramref name="source"/>'s noise mask, batch indexes and keys (ComfyUI's nodes copy the LATENT dict).</summary>
    public LatentBatch With(LatentBatch source) =>
        new LatentBatch(Shape, Data)
        {
            NoiseMask = source.NoiseMask, BatchIndex = source.BatchIndex, Parts = Parts ?? source.Parts, PartMasks = source.PartMasks,
            Type = source.Type, DownscaleRatioSpacial = source.DownscaleRatioSpacial, DownscaleRatioTemporal = source.DownscaleRatioTemporal,
        };

    /// <summary>
    /// Optional inpainting mask at latent size (batch, h, w): 1 = this area is regenerated, 0 = it is kept. Set by Set Latent Noise Mask. A video
    /// latent's has its frames stacked (batch, frames x h, w), as ComfyUI's [batch, 1, frames, h, w].
    /// </summary>
    public MaskBatch? NoiseMask { get; init; }

    /// <summary>
    /// ComfyUI's <c>batch_index</c>: which noise of the seed each picture starts from (null = picture i gets noise i). Get Latent From Batch keeps a picture's
    /// original noise this way, and Latent Batch Seed Behavior "fixed" gives the whole batch one noise.
    /// </summary>
    public int[]? BatchIndex { get; init; }
}

/// <summary>A batch of masks, NHW (batch, height, width), values 0..1 (same layout as ComfyUI's MASK). 1 = selected.</summary>
[Socket("MASK")]
public sealed class MaskBatch : HostTensor
{
    public MaskBatch(int batch, int height, int width, float[]? data = null) : base([batch, height, width], data) { }

    public int Batch => Shape[0];
    public int Height => Shape[1];
    public int Width => Shape[2];

    public static MaskBatch Filled(int batch, int height, int width, float value)
    {
        var m = new MaskBatch(batch, height, width);
        Array.Fill(m.Data, value);
        return m;
    }
}
