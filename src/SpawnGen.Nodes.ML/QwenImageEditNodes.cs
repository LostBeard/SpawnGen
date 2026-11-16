using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// Text Encode Qwen Image Edit Plus (ComfyUI's TextEncodeQwenImageEditPlus, Qwen-Image Edit 2509): the prompt and up to three pictures through
/// Qwen2.5-VL (each picture's vision rows inside the text, the edit template), and with a VAE each picture's reference latent. Measured with
/// ComfyUI (2026-11-15, black box: Fixtures/qwenedit):
/// <list type="bullet">
/// <item>the vision picture: area-resized to each side times sqrt(384^2 / (w h)), rounded, then bilinear to the nearest multiples of 28
/// (256 x 192 -> 443 x 333 -> 448 x 336; 200 x 120 -> 496 x 297 -> 504 x 308), matched to 1.3e-6;</item>
/// <item>the reference latent: the picture resized to each side times sqrt(1024^2 / (w h)), rounded to a multiple of 8 (1184 x 888, 1320 x 792),
/// VAE-encoded; one per picture, on the conditioning in the pictures' order;</item>
/// <item>"Picture k: " before each picture's vision span, numbered over the pictures given (image1 alone, image1 + image2 measured).</item>
/// </list>
/// A picture input holding several pictures uses its first, as one picture per input is what was measured.
/// </summary>
[Node("TextEncodeQwenImageEditPlus", Category = "model/conditioning/qwen image", DisplayName = "Text Encode Qwen Image Edit Plus")]
public sealed class TextEncodeQwenImageEditPlus
{
    /// <summary>The vision tower's picture area (384 x 384 pixels) and the reference latent's (1024 x 1024).</summary>
    public const int VisionArea = 384 * 384, ReferenceArea = 1024 * 1024;

    /// <summary>The reference picture's resampler (CommonUpscale method), measured by encoding candidates against ComfyUI's reference latents.</summary>
    public const string ReferenceMethod = "area";

    /// <summary>Python's round (half to even), as the measured sizes follow it.</summary>
    static int Round(double v) => (int)Math.Round(v, MidpointRounding.ToEven);

    /// <summary>The vision picture's first size (each side times sqrt(384^2 / (w h)), rounded) and its final one (the nearest multiples of 28).</summary>
    public static ((int H, int W) First, (int H, int W) Final) VisionSize(int height, int width)
    {
        double s = Math.Sqrt((double)VisionArea / ((double)width * height));
        int h = Round(height * s), w = Round(width * s);
        return ((h, w), (Math.Max(28, Round(h / 28.0) * 28), Math.Max(28, Round(w / 28.0) * 28)));
    }

    /// <summary>The reference picture's size: each side times sqrt(1024^2 / (w h)), rounded to a multiple of 8.</summary>
    public static (int H, int W) ReferenceSize(int height, int width)
    {
        double s = Math.Sqrt((double)ReferenceArea / ((double)width * height));
        return (Math.Max(8, Round(height * s / 8.0) * 8), Math.Max(8, Round(width * s / 8.0) * 8));
    }

    /// <summary>The picture the vision tower sees: area to the first size, bilinear to the final one (values in 0..1, not rounded to 8 bits).</summary>
    public static ImageBatch VisionPicture(ImageBatch picture)
    {
        var ((h1, w1), (h2, w2)) = VisionSize(picture.Height, picture.Width);
        return CommonUpscale.Resize(CommonUpscale.Resize(picture, w1, h1, "area"), w2, h2, "bilinear");
    }

    public async Task<Conditioning> ExecuteAsync(ClipHandle clip, NodeContext context, [Input(Multiline = true)] string prompt = "",
        [Input(Optional = true)] VaeHandle? vae = null, [Input(Optional = true)] ImageBatch? image1 = null,
        [Input(Optional = true)] ImageBatch? image2 = null, [Input(Optional = true)] ImageBatch? image3 = null)
    {
        if (clip.Encoder is not QwenImageTextEncoderModel encoder)
            throw new NotSupportedException($"Text Encode Qwen Image Edit Plus takes Qwen-Image's text encoder (Load CLIP type qwen_image); '{clip.Encoder.Name}' is not one.");
        var pictures = new[] { image1, image2, image3 }.Where(p => p is not null).Select(p => First(p!)).ToList();
        var encoding = await encoder.EncodeEditAsync(prompt, [.. pictures.Select(VisionPicture)], context.CancellationToken);
        var entry = new CondEntry(encoding.Context, encoding.Shape);
        if (vae is not null && pictures.Count > 0)
        {
            if (vae.Vae is not IVideoVaeModel video)
                throw new NotSupportedException($"{vae.Vae.Name} is not Qwen-Image's VAE (qwen_image_vae, a Wan 2.1 video VAE).");
            var references = new List<LatentBatch>();
            foreach (var picture in pictures)
            {
                var (h, w) = ReferenceSize(picture.Height, picture.Width);
                references.Add(await VideoVae.EncodeAsync(video, CommonUpscale.Resize(picture, w, h, ReferenceMethod), context.CancellationToken));
            }
            entry = entry with { ReferenceLatents = references };
        }
        return new Conditioning([entry]);
    }

    static ImageBatch First(ImageBatch batch)
    {
        if (batch.Batch == 1) return batch;
        var one = new ImageBatch(1, batch.Height, batch.Width, batch.Channels);
        Array.Copy(batch.Data, one.Data, one.Data.Length);
        return one;
    }
}
