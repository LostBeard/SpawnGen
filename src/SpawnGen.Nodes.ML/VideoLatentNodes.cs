using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries and black-box runs of its Wan 2.2 latent nodes with its Wan 2.2 VAE
// (Fixtures/wan22, 2026-11-12): the latent's shape, the start picture's fit (bilinear, centre crop: exact), the frames after it (the latent mean, i.e.
// a zero latent in the model's units), the noise mask (0 on the start frames, 1 after), and VAE Encode's crop to multiples of 16 (centred).

/// <summary>
/// A video VAE (Wan 2.2's): pictures to a video latent [1, channels, frames, h, w] and back, both in ComfyUI's units (the VAE's raw latent). A video of
/// n pictures takes (n - 1) / 4 + 1 latent frames; the first latent frame holds one picture, each later one four.
/// </summary>
public interface IVideoVaeModel : IVaeModel
{
    /// <summary>Pixels per latent position, each side (16 for Wan 2.2).</summary>
    int SpatialScale { get; }

    /// <summary>Video frames per latent frame after the first (4 for Wan 2.2).</summary>
    int TemporalScale { get; }

    int LatentChannels { get; }

    /// <summary>The pictures (sides multiples of <see cref="SpatialScale"/>) as one video latent, batch 1, in the VAE's raw units.</summary>
    Task<LatentBatch> EncodeVideoAsync(ImageBatch frames, CancellationToken cancellationToken = default);

    /// <summary>Every item of a video latent decoded to its frames, one item after another, values in 0..1.</summary>
    Task<ImageBatch> DecodeVideoAsync(LatentBatch latent, CancellationToken cancellationToken = default);
}

/// <summary>The video side of VAE Encode / VAE Decode.</summary>
internal static class VideoVae
{
    /// <summary>The latent frames a video of <paramref name="length"/> frames takes: (length - 1) / 4 + 1.</summary>
    public static int LatentFrames(int length, int temporalScale = 4) => (length - 1) / temporalScale + 1;

    /// <summary>VAE Encode with a video VAE: the pictures cut to multiples of its scale (the cut shared between both sides, as measured), as one video.</summary>
    public static Task<LatentBatch> EncodeAsync(IVideoVaeModel vae, ImageBatch pixels, CancellationToken ct)
    {
        int s = vae.SpatialScale;
        return vae.EncodeVideoAsync(Inpaint.Crop(pixels, pixels.Height / s * s, pixels.Width / s * s), ct);
    }

    public static Task<ImageBatch> DecodeAsync(IVaeModel vae, LatentBatch samples, CancellationToken ct) =>
        vae is IVideoVaeModel video ? video.DecodeVideoAsync(samples, ct)
            : throw new NotSupportedException($"{vae.Name} is a picture VAE; a video latent needs its video model's VAE (Wan 2.2's for Wan 2.2).");
}

/// <summary>
/// Wan22ImageToVideoLatent: Wan 2.2 5B's latent for a video of <c>length</c> frames, optionally starting from pictures. Without them: zeros
/// [1, 48, (length - 1) / 4 + 1, height / 16, width / 16] (batch_size is not applied, as measured). With them: the first <c>length</c> pictures fitted
/// to width x height (bilinear, centre crop), encoded, and put first; the later frames hold the latent mean (zero in the model's units); a noise mask
/// keeps the encoded frames (0) and regenerates the rest (1); the whole repeated batch_size times.
/// </summary>
[Node("Wan22ImageToVideoLatent", Category = "model/conditioning/wan", DisplayName = "Wan22ImageToVideoLatent")]
public sealed class Wan22ImageToVideoLatent
{
    public async Task<LatentBatch> ExecuteAsync(VaeHandle vae, NodeContext context,
        [Input(Min = 32, Max = 16384, Step = 32)] int width = 1280,
        [Input(Min = 32, Max = 16384, Step = 32)] int height = 704,
        [Input(Min = 1, Max = 16384, Step = 4)] int length = 49,
        [Input(Min = 1, Max = 4096)] int batchSize = 1,
        ImageBatch? startImage = null)
    {
        const int Channels = 48, Scale = 16;
        int frames = VideoVae.LatentFrames(length), h = height / Scale, w = width / Scale, plane = h * w;
        if (startImage is null) return new LatentBatch(1, Channels, frames, h, w);

        var video = vae.Vae as IVideoVaeModel
            ?? throw new NotSupportedException($"{vae.Vae.Name} is a picture VAE; Wan22ImageToVideoLatent needs Wan 2.2's VAE.");
        var pictures = startImage.Batch > length ? Take(startImage, length) : startImage;
        var fitted = CommonUpscale.Resize(pictures, width, height, "bilinear", "center");
        var encoded = await video.EncodeVideoAsync(fitted, context.CancellationToken);
        int kept = Math.Min(encoded.Frames, frames);

        var item = LatentFormat.Wan22.ProcessOut(new float[Channels * frames * plane]);
        for (int c = 0; c < Channels; c++)
            Array.Copy(encoded.Data, c * encoded.Frames * plane, item, c * frames * plane, kept * plane);
        var mask = new float[frames * plane];
        Array.Fill(mask, 1f, kept * plane, (frames - kept) * plane);

        var data = new float[batchSize * item.Length];
        var masks = new float[batchSize * mask.Length];
        for (int b = 0; b < batchSize; b++)
        {
            item.CopyTo(data, b * item.Length);
            mask.CopyTo(masks, b * mask.Length);
        }
        return new LatentBatch(batchSize, Channels, frames, h, w, data) { NoiseMask = new MaskBatch(batchSize, frames * h, w, masks) };
    }

    static ImageBatch Take(ImageBatch images, int count)
    {
        int per = images.Height * images.Width * images.Channels;
        return new ImageBatch(count, images.Height, images.Width, images.Channels, images.Data.AsSpan(0, count * per).ToArray());
    }
}
