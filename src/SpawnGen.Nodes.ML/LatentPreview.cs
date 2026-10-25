using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// A quick look at a latent without running the VAE: each pixel is a fixed linear mix of the latent channels (ComfyUI's "latent2rgb" previewer, with the
/// model family's factors and bias). Blurry and colour-shifted, but it shows composition and colour while a sampler works, for almost no cost.
/// </summary>
public static class LatentPreview
{
    /// <summary>The preview of a model-space latent (what the sampler holds) as an RGB picture the size of the latent (values 0..1).</summary>
    public static ImageBatch Image(float[] latent, int height, int width, LatentFormat? format = null)
    {
        format ??= LatentFormat.SD15;
        var factors = format.RgbFactors;
        int plane = height * width, channels = format.Channels;
        var image = new ImageBatch(1, height, width, 3);
        for (int i = 0; i < plane; i++)
            for (int ch = 0; ch < 3; ch++)
            {
                float v = format.RgbBias?[ch] ?? 0f;
                for (int c = 0; c < channels; c++) v += latent[c * plane + i] * factors[c, ch];
                image.Data[i * 3 + ch] = Math.Clamp((v + 1f) / 2f, 0f, 1f);
            }
        return image;
    }

    public static byte[] Png(float[] latent, int height, int width, LatentFormat? format = null) => PngWriter.Encode(Image(latent, height, width, format));
}
