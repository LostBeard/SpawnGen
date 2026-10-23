using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// ComfyUI's LATENT_OPERATION: something to do to a latent, made by one node and applied by another (Latent Apply Operation, or Latent Apply Operation CFG
/// during sampling). <c>first</c> is where the latent's first picture sits in the whole batch: a sampler hands pictures over one at a time, and an operation
/// that matches a reference batch picture by picture needs to know which one it has.
/// </summary>
[Socket("LATENT_OPERATION")]
public sealed class LatentOperation(Func<LatentBatch, int, LatentBatch> apply)
{
    public LatentBatch Apply(LatentBatch latent, int first = 0) => apply(latent, first);
}

/// <summary>Shared by the latent operations: the per-position length of the channel vector (<c>torch.linalg.vector_norm(latent, dim=1)</c>).</summary>
internal static class ChannelNorm
{
    public static float[] Of(float[] data, int b, int c, int plane)
    {
        var m = new float[b * plane];
        for (int n = 0; n < b; n++)
            for (int p = 0; p < plane; p++)
            {
                double sum = 0;
                for (int ch = 0; ch < c; ch++) { double v = data[(n * c + ch) * plane + p]; sum += v * v; }
                m[n * plane + p] = (float)Math.Sqrt(sum);
            }
        return m;
    }
}

[Node(Category = "model/latent/advanced/operations", DisplayName = "LatentApplyOperation")]
public sealed class LatentApplyOperation
{
    public LatentBatch Execute(LatentBatch samples, LatentOperation operation)
    {
        var r = operation.Apply(samples);
        return Latents.Like(samples, r.Batch, r.Channels, r.Height, r.Width, r.Data);
    }
}

[Node(Category = "model/latent/advanced/operations", DisplayName = "Latent Operation Blend",
    Description = "Blend the entire latent toward the reference latent, resizing the reference with nearest-neighbor interpolation when needed.")]
public sealed class LatentOperationBlend
{
    /// <summary><c>torch.lerp(latent, reference, strength)</c>, the reference resized (nearest) to the latent and repeated or cut to its batch.</summary>
    public LatentOperation Execute(LatentBatch reference,
        [Input(Min = 0, Max = 1, Step = 0.0001, Tooltip = "0 leaves the latent unchanged; 1 matches the resized reference latent.")] double strength = 1.0) =>
        new((latent, first) =>
        {
            if (strength == 0) return latent;
            var target = reference;
            if (target.Height != latent.Height || target.Width != latent.Width)
                target = new LatentBatch(target.Batch, target.Channels, latent.Height, latent.Width,
                    TorchInterpolate.Resize(target.Data, target.Batch * target.Channels, target.Height, target.Width, latent.Height, latent.Width, "nearest"));
            if (target.Channels != latent.Channels && target.Channels != 1)
                throw new ArgumentException($"The size of tensor a ({latent.Channels}) must match the size of tensor b ({target.Channels}) at non-singleton dimension 1.");
            int c = latent.Channels, plane = latent.Height * latent.Width;
            float weight = (float)strength;
            var r = new float[latent.Data.Length];
            for (int n = 0; n < latent.Batch; n++)
            {
                int tb = (first + n) % target.Batch; // repeat_to_batch_size over the whole batch
                for (int ch = 0; ch < c; ch++)
                {
                    int tc = target.Channels == 1 ? 0 : ch;
                    for (int p = 0; p < plane; p++)
                    {
                        float a = latent.Data[(n * c + ch) * plane + p], e = target.Data[(tb * target.Channels + tc) * plane + p];
                        // PyTorch's lerp: from the nearer end, for precision.
                        r[(n * c + ch) * plane + p] = weight < 0.5f ? a + weight * (e - a) : e - (e - a) * (1f - weight);
                    }
                }
            }
            return new LatentBatch(latent.Batch, c, latent.Height, latent.Width, r);
        });
}

[Node(Category = "model/latent/advanced/operations", DisplayName = "LatentOperationTonemapReinhard")]
public sealed class LatentOperationTonemapReinhard
{
    /// <summary>
    /// Reinhard tone mapping of the latent's per-position strength: lengths are compressed towards a ceiling of (5 standard deviations + mean) x <c>multiplier</c>
    /// of the picture's lengths, keeping every direction. Tames over-strong areas (what high cfg produces).
    /// </summary>
    public LatentOperation Execute([Input(Min = 0, Max = 100, Step = 0.01)] double multiplier = 1.0) =>
        new((latent, _) =>
        {
            int b = latent.Batch, c = latent.Channels, plane = latent.Height * latent.Width;
            var magnitude = ChannelNorm.Of(latent.Data, b, c, plane);
            for (int i = 0; i < magnitude.Length; i++) magnitude[i] += 1e-10f;
            var r = new float[latent.Data.Length];
            for (int n = 0; n < b; n++)
            {
                // Mean and unbiased standard deviation of this picture's lengths.
                double sum = 0;
                for (int p = 0; p < plane; p++) sum += magnitude[n * plane + p];
                double mean = sum / plane, sq = 0;
                for (int p = 0; p < plane; p++) { double d = magnitude[n * plane + p] - mean; sq += d * d; }
                float std = (float)Math.Sqrt(sq / (plane - 1)), m = (float)mean;
                float top = (std * 5 + m) * (float)multiplier, inverse = 1f / top;
                for (int p = 0; p < plane; p++)
                {
                    float length = magnitude[n * plane + p], scaled = length * inverse;
                    float mapped = scaled / (scaled + 1f) * top;
                    for (int ch = 0; ch < c; ch++)
                    {
                        int i = (n * c + ch) * plane + p;
                        r[i] = latent.Data[i] / length * mapped;
                    }
                }
            }
            return new LatentBatch(b, c, latent.Height, latent.Width, r);
        });
}

[Node(Category = "model/latent/advanced/operations", DisplayName = "LatentOperationSharpen")]
public sealed class LatentOperationSharpen
{
    /// <summary>
    /// Sharpens the direction of the latent's channel vectors (an unsharp kernel built from ComfyUI's Gaussian over a (2 x radius + 1) square, edges reflected) and puts
    /// the original lengths back.
    /// </summary>
    public LatentOperation Execute([Input(Min = 1, Max = 31, Step = 1)] int sharpenRadius = 9, [Input(Min = 0.1, Max = 10, Step = 0.1)] double sigma = 1.0,
        [Input(Min = 0, Max = 5, Step = 0.01)] double alpha = 0.1) =>
        new((latent, _) =>
        {
            int b = latent.Batch, c = latent.Channels, h = latent.Height, w = latent.Width, plane = h * w, r = sharpenRadius;
            if (r >= h || r >= w)
                throw new ArgumentException($"Padding size should be less than the corresponding input dimension, but got padding {r} for a {w}x{h} latent (sharpen_radius must be smaller than the latent).");
            var kernel = Kernel(sharpenRadius, sigma, alpha);
            int size = 2 * r + 1;
            var luminance = ChannelNorm.Of(latent.Data, b, c, plane);
            for (int i = 0; i < luminance.Length; i++) luminance[i] += 1e-6f;
            static int Reflect(int i, int n) => i < 0 ? -i : i >= n ? 2 * (n - 1) - i : i;

            var normalized = new float[latent.Data.Length];
            for (int n = 0; n < b; n++)
                for (int ch = 0; ch < c; ch++)
                    for (int p = 0; p < plane; p++) normalized[(n * c + ch) * plane + p] = latent.Data[(n * c + ch) * plane + p] / luminance[n * plane + p];
            var result = new float[latent.Data.Length];
            for (int n = 0; n < b; n++)
                for (int ch = 0; ch < c; ch++)
                {
                    int planeBase = (n * c + ch) * plane;
                    for (int y = 0; y < h; y++)
                        for (int x = 0; x < w; x++)
                        {
                            double acc = 0;
                            for (int i = 0; i < size; i++)
                            {
                                int sy = Reflect(y + i - r, h);
                                for (int j = 0; j < size; j++) acc += kernel[i * size + j] * normalized[planeBase + sy * w + Reflect(x + j - r, w)];
                            }
                            result[planeBase + y * w + x] = luminance[n * plane + y * w + x] * (float)acc;
                        }
                }
            return new LatentBatch(b, c, h, w, result);
        });

    /// <summary>
    /// ComfyUI's <c>gaussian_kernel(2r + 1, sigma)</c> (sampled over [-1, 1] with PyTorch's linspace, normalised), scaled by -10 x alpha, with the centre raised so the
    /// kernel sums to 1.
    /// </summary>
    public static float[] Kernel(int radius, double sigma, double alpha)
    {
        int size = 2 * radius + 1;
        var lin = new float[size];
        float step = 2f / (size - 1);
        for (int i = 0; i < size; i++) lin[i] = i < size / 2 ? -1f + step * i : 1f - step * (size - i - 1);
        float twoSigmaSq = (float)(2.0 * sigma * sigma), scale = (float)(alpha * -10);
        var k = new float[size * size];
        double total = 0;
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
            {
                float d = MathF.Sqrt(lin[i] * lin[i] + lin[j] * lin[j]);
                k[i * size + j] = MathF.Exp(-(d * d) / twoSigmaSq);
                total += k[i * size + j];
            }
        double sum = 0;
        for (int i = 0; i < k.Length; i++) { k[i] = k[i] / (float)total * scale; sum += k[i]; }
        int centre = radius * size + radius;
        k[centre] = k[centre] - (float)sum + 1f;
        return k;
    }
}
