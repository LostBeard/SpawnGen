using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// A quick look at a latent without running the VAE: each pixel is a fixed linear mix of the four latent channels (ComfyUI's "latent2rgb" previewer for
/// SD 1.x / 2.x). Blurry and colour-shifted, but it shows composition and colour while a sampler works, for almost no cost.
/// </summary>
public static class LatentPreview
{
    // Rows are latent channels, columns are R, G, B.
    static readonly float[,] Factors =
    {
        {  0.3512f,  0.2297f,  0.3227f },
        {  0.3250f,  0.4974f,  0.2350f },
        { -0.2829f,  0.1762f,  0.2721f },
        { -0.2120f, -0.2616f, -0.7177f },
    };

    /// <summary>The preview as an RGB picture the size of the latent (values 0..1).</summary>
    public static ImageBatch Image(float[] latent, int height, int width)
    {
        int plane = height * width;
        var image = new ImageBatch(1, height, width, 3);
        for (int i = 0; i < plane; i++)
            for (int ch = 0; ch < 3; ch++)
            {
                float v = 0;
                for (int c = 0; c < 4; c++) v += latent[c * plane + i] * Factors[c, ch];
                image.Data[i * 3 + ch] = Math.Clamp((v + 1f) / 2f, 0f, 1f);
            }
        return image;
    }

    public static byte[] Png(float[] latent, int height, int width) => PngWriter.Encode(Image(latent, height, width));
}
