using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// Joining pictures, masks and latents into one batch, as measured with ComfyUI's Batch Images / Batch Masks / Batch Latents (and the deprecated
/// two-input Image Batch / Latent Batch, which behave the same):
/// <list type="bullet">
/// <item>Every item is brought to the first one's height and width: cut to the first one's aspect ratio around its middle, then resized bilinearly
/// (PyTorch's bilinear without corner alignment).</item>
/// <item>Pictures: when the channel counts differ, each picture with fewer channels than the most gets one channel of 1.0 added (an RGB picture next
/// to an RGBA one gets full alpha). Counts that still differ (a 1-channel picture next to an RGB one) fail as PyTorch's <c>torch.cat</c> does.</item>
/// <item>Latents: the result keeps the first latent's noise mask; its batch indexes are each latent's own (or 0, 1, ... when it has none), one after
/// another. Different channel counts fail as <c>torch.cat</c> does.</item>
/// </list>
/// </summary>
public static class Batching
{
    /// <summary>PyTorch's <c>torch.cat</c> message for a tensor whose size differs from the first one's outside the joined dimension.</summary>
    internal static ArgumentException SizeMismatch(long expected, long got, int tensorNumber) =>
        new($"Sizes of tensors must match except in dimension 0. Expected size {expected} but got size {got} for tensor number {tensorNumber} in the list.");

    public static ImageBatch Images(IReadOnlyList<ImageBatch> images)
    {
        if (images.Count == 0) throw new ArgumentException("There are no images to batch.", nameof(images));
        var first = images[0];
        int channels = images.Max(i => i.Channels);
        var parts = new ImageBatch[images.Count];
        for (int i = 0; i < images.Count; i++)
        {
            var image = images[i].Channels < channels ? AddOpaqueChannel(images[i]) : images[i];
            if (image.Height != first.Height || image.Width != first.Width)
                image = CommonUpscale.Resize(image, first.Width, first.Height, "bilinear", "center");
            parts[i] = image;
        }
        for (int i = 1; i < parts.Length; i++)
            if (parts[i].Channels != parts[0].Channels) throw SizeMismatch(parts[0].Channels, parts[i].Channels, i);
        return new ImageBatch(parts.Sum(p => p.Batch), first.Height, first.Width, parts[0].Channels, Join(parts.Select(p => p.Data)));
    }

    public static MaskBatch Masks(IReadOnlyList<MaskBatch> masks)
    {
        if (masks.Count == 0) throw new ArgumentException("There are no masks to batch.", nameof(masks));
        var first = masks[0];
        var parts = masks.Select(m => m.Height != first.Height || m.Width != first.Width
            ? CommonUpscale.Resize(m, first.Width, first.Height, "bilinear", "center") : m).ToArray();
        return new MaskBatch(parts.Sum(p => p.Batch), first.Height, first.Width, Join(parts.Select(p => p.Data)));
    }

    public static LatentBatch Latents(IReadOnlyList<LatentBatch> latents)
    {
        if (latents.Count == 0) throw new ArgumentException("There are no latents to batch.", nameof(latents));
        var first = latents[0];
        for (int i = 1; i < latents.Count; i++)
            if (latents[i].Channels != first.Channels) throw SizeMismatch(first.Channels, latents[i].Channels, i);
        var data = new List<float[]>();
        var batchIndex = new List<int>();
        foreach (var latent in latents)
        {
            data.Add(latent.Height != first.Height || latent.Width != first.Width ? ResizePlanes(latent, first.Width, first.Height) : latent.Data);
            batchIndex.AddRange(latent.BatchIndex ?? Enumerable.Range(0, latent.Batch));
        }
        return new LatentBatch(latents.Sum(l => l.Batch), first.Channels, first.Height, first.Width, Join(data))
        {
            NoiseMask = first.NoiseMask,
            BatchIndex = [.. batchIndex],
        };
    }

    /// <summary>The picture with one more channel, filled with 1.0.</summary>
    static ImageBatch AddOpaqueChannel(ImageBatch image)
    {
        int c = image.Channels, pixels = image.Batch * image.Height * image.Width;
        var r = new ImageBatch(image.Batch, image.Height, image.Width, c + 1);
        for (int p = 0; p < pixels; p++)
        {
            Array.Copy(image.Data, p * c, r.Data, p * (c + 1), c);
            r.Data[p * (c + 1) + c] = 1f;
        }
        return r;
    }

    /// <summary>Each channel plane of each latent resized to (w, h) with a centre crop, bilinear.</summary>
    static float[] ResizePlanes(LatentBatch latent, int w, int h)
    {
        int plane = latent.Height * latent.Width, planes = latent.Batch * latent.Channels;
        var r = new float[planes * w * h];
        for (int p = 0; p < planes; p++)
        {
            var one = new float[plane];
            Array.Copy(latent.Data, p * plane, one, 0, plane);
            Array.Copy(CommonUpscale.Resize(one, latent.Width, latent.Height, 1, w, h, "bilinear", "center"), 0, r, p * w * h, w * h);
        }
        return r;
    }

    static float[] Join(IEnumerable<float[]> parts)
    {
        var list = parts.ToList();
        var r = new float[list.Sum(p => (long)p.Length)];
        long at = 0;
        foreach (var p in list)
        {
            Array.Copy(p, 0, r, at, p.Length);
            at += p.Length;
        }
        return r;
    }
}

[Node("BatchImagesNode", Category = "image/batch", DisplayName = "Batch Images")]
public sealed class BatchImagesNode
{
    /// <summary>The pictures as one batch, in input order, at the first picture's size (see <see cref="Batching"/>).</summary>
    public ImageBatch Execute([Autogrow(Prefix = "image", Min = 1, Max = 50)] IReadOnlyList<ImageBatch> images) => Batching.Images(images);
}

[Node("BatchMasksNode", Category = "image/mask", DisplayName = "Batch Masks")]
public sealed class BatchMasksNode
{
    /// <summary>The masks as one batch, in input order, at the first mask's size (see <see cref="Batching"/>).</summary>
    public MaskBatch Execute([Autogrow(Prefix = "mask", Min = 1, Max = 50)] IReadOnlyList<MaskBatch> masks) => Batching.Masks(masks);
}

[Node("BatchLatentsNode", Category = "model/latent/batch", DisplayName = "Batch Latents")]
public sealed class BatchLatentsNode
{
    /// <summary>The latents as one batch, in input order, at the first latent's size, with the first latent's noise mask (see <see cref="Batching"/>).</summary>
    public LatentBatch Execute([Autogrow(Prefix = "latent", Min = 1, Max = 50)] IReadOnlyList<LatentBatch> latents) => Batching.Latents(latents);
}

[Node("CreateList", Category = "utilities", DisplayName = "Create List", InputIsList = true, OutputIsList = [true], OutputNames = ["list"])]
public sealed class CreateList
{
    /// <summary>
    /// One list of everything connected, in input order. Measured with ComfyUI: an input that is itself a list (another Create List) adds its items,
    /// so the result is flat; items of any type can be mixed.
    /// </summary>
    public IReadOnlyList<object?> Execute([Autogrow(Prefix = "input", Min = 1, Max = 10), Input(Type = "*")] IReadOnlyList<IReadOnlyList<object?>> inputs) =>
        [.. inputs.SelectMany(items => items)];
}
