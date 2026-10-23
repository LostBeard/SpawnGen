using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>ComfyUI's batching rules (nodes_post_processing.batch_images / batch_masks / batch_latents), shared by the batch nodes.</summary>
public static class Batching
{
    /// <summary>
    /// One batch of every picture. A picture with fewer channels gets an opaque alpha channel (RGB next to RGBA), and every picture is scaled to the
    /// first one's size (bilinear, centre crop).
    /// </summary>
    public static ImageBatch Images(IReadOnlyList<ImageBatch> images)
    {
        if (images.Count == 0) throw new ArgumentException("There are no pictures to batch.", nameof(images));
        int channels = images.Max(i => i.Channels);
        var first = WithChannels(images[0], channels);
        var frames = new List<float[]>();
        foreach (var image in images)
        {
            var padded = WithChannels(image, channels);
            if (padded.Width != first.Width || padded.Height != first.Height || padded.Channels != first.Channels)
                padded = new ImageScale().Execute(padded, "bilinear", first.Width, first.Height, "center");
            for (int b = 0; b < padded.Batch; b++) frames.Add(Pixels.Frame(padded, b));
        }
        return Pixels.From(frames, first.Height, first.Width, first.Channels);
    }

    /// <summary>One batch of every mask, each scaled to the first one's size (bilinear, centre crop).</summary>
    public static MaskBatch Masks(IReadOnlyList<MaskBatch> masks)
    {
        if (masks.Count == 0) throw new ArgumentException("There are no masks to batch.", nameof(masks));
        int h = masks[0].Height, w = masks[0].Width;
        var data = new List<float>();
        int batch = 0;
        foreach (var mask in masks)
        {
            for (int b = 0; b < mask.Batch; b++)
            {
                var frame = mask.Data.AsSpan(b * mask.Height * mask.Width, mask.Height * mask.Width).ToArray();
                data.AddRange(mask.Height == h && mask.Width == w ? frame : Resample.CommonUpscale(frame, mask.Width, mask.Height, 1, w, h, ResampleMethod.Bilinear, true));
            }
            batch += mask.Batch;
        }
        return new MaskBatch(batch, h, w, [.. data]);
    }

    /// <summary>One batch of every latent, each scaled to the first one's size (bilinear, centre crop); the first one's noise mask is kept.</summary>
    public static LatentBatch Latents(IReadOnlyList<LatentBatch> latents)
    {
        if (latents.Count == 0) throw new ArgumentException("There are no latents to batch.", nameof(latents));
        var first = latents[0];
        var frames = new List<float[]>();
        foreach (var latent in latents)
        {
            if (latent.Channels != first.Channels) throw new ArgumentException($"The latents have {first.Channels} and {latent.Channels} channels and cannot be batched.");
            var same = latent.Height == first.Height && latent.Width == first.Width
                ? latent : new LatentUpscale().Execute(latent, "bilinear", first.Width * 8, first.Height * 8, "center");
            for (int b = 0; b < same.Batch; b++) frames.Add(Core.Latents.Frame(same, b));
        }
        var batched = Core.Latents.From(frames, first.Channels, first.Height, first.Width);
        return new LatentBatch(batched.Batch, batched.Channels, batched.Height, batched.Width, batched.Data) { NoiseMask = first.NoiseMask };
    }

    /// <summary>ComfyUI pads a narrower picture with one channel of 1.0 (torch pad (0, 1)): RGB becomes RGBA with full alpha.</summary>
    static ImageBatch WithChannels(ImageBatch image, int channels)
    {
        if (image.Channels >= channels) return image;
        int c = image.Channels, pixels = image.Batch * image.Height * image.Width;
        var padded = new ImageBatch(image.Batch, image.Height, image.Width, c + 1);
        for (int p = 0; p < pixels; p++)
        {
            Array.Copy(image.Data, p * c, padded.Data, p * (c + 1), c);
            padded.Data[p * (c + 1) + c] = 1f;
        }
        return padded;
    }
}

[Node("BatchImagesNode", Category = "image/batch", DisplayName = "Batch Images")]
public sealed class BatchImagesNode
{
    public ImageBatch Execute([Autogrow(Prefix = "image", Min = 1, Max = 50)] IReadOnlyList<ImageBatch> images) => Batching.Images(images);
}

[Node("BatchMasksNode", Category = "image/mask", DisplayName = "Batch Masks")]
public sealed class BatchMasksNode
{
    public MaskBatch Execute([Autogrow(Prefix = "mask", Min = 1, Max = 50)] IReadOnlyList<MaskBatch> masks) => Batching.Masks(masks);
}

[Node("BatchLatentsNode", Category = "model/latent/batch", DisplayName = "Batch Latents")]
public sealed class BatchLatentsNode
{
    public LatentBatch Execute([Autogrow(Prefix = "latent", Min = 1, Max = 50)] IReadOnlyList<LatentBatch> latents) => Batching.Latents(latents);
}

/// <summary>ComfyUI's Create List: every connected input, in order, made into one list (each input may itself bring a list).</summary>
[Node("CreateList", Category = "utilities", DisplayName = "Create List", InputIsList = true, OutputIsList = [true], OutputNames = ["list"])]
public sealed class CreateList
{
    public IReadOnlyList<object?> Execute([Autogrow(Prefix = "input")] IReadOnlyList<IReadOnlyList<object?>> inputs) => [.. inputs.SelectMany(i => i)];
}
