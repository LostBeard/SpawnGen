using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// The LATENT_OPERATION nodes. Behavior measured with ComfyUI (2026-11-08) by applying each operation through Latent Apply Operation to latents
// loaded from files and saving the result with Save Latent.

/// <summary>
/// A change to a latent that a node hands on to be applied later (ComfyUI's LATENT_OPERATION): by Latent Apply Operation directly, or inside sampling by
/// Latent Apply Operation CFG. The function gets the latent and the batch position of its first picture (0 outside sampling; inside sampling the
/// sampler may hand over one picture of a larger batch at a time).
/// </summary>
[Socket("LATENT_OPERATION")]
public sealed class LatentOperation(Func<LatentBatch, int, LatentBatch> apply)
{
    /// <summary>Runs the operation on <paramref name="latent"/>, whose first picture is picture <paramref name="first"/> of the whole batch.</summary>
    public LatentBatch Apply(LatentBatch latent, int first = 0) => apply(latent, first);
}

/// <summary>The length of the channel vector at each position of a latent: one value per picture, row and column.</summary>
public static class ChannelNorm
{
    /// <summary>
    /// For each picture and position, the Euclidean length over the channels (accumulated in double, rounded to float32), laid out (batch, height,
    /// width).
    /// </summary>
    public static float[] Lengths(LatentBatch latent)
    {
        int n = latent.Batch, c = latent.Channels, plane = latent.Height * latent.Width;
        var r = new float[n * plane];
        for (int b = 0; b < n; b++)
            for (int p = 0; p < plane; p++)
            {
                double sum = 0;
                for (int k = 0; k < c; k++)
                {
                    double v = latent.Data[(b * c + k) * plane + p];
                    sum += v * v;
                }
                r[b * plane + p] = (float)Math.Sqrt(sum);
            }
        return r;
    }
}

[Node(Category = "model/latent/advanced/operations", DisplayName = "LatentApplyOperation")]
public sealed class LatentApplyOperation
{
    /// <summary>Runs the operation on the whole latent (its first picture is picture 0). The noise mask and batch index are kept.</summary>
    public LatentBatch Execute(LatentBatch samples, LatentOperation operation)
    {
        var changed = operation.Apply(samples, 0);
        return new LatentBatch(changed.Batch, changed.Channels, changed.Height, changed.Width, changed.Data)
        {
            NoiseMask = samples.NoiseMask, BatchIndex = samples.BatchIndex,
        };
    }
}

[Node(Category = "model/latent/advanced/operations", DisplayName = "Latent Operation Blend",
    Description = "Blend the entire latent toward the reference latent, resizing the reference with nearest-neighbor interpolation when needed.")]
public sealed class LatentOperationBlend
{
    /// <summary>
    /// Moves the latent toward the reference by <paramref name="strength"/> (torch.lerp, its vectorized multiply-add form: 502 of 504 random values bit-equal to ComfyUI, the rest within 6e-8). Measured with ComfyUI:
    /// <list type="bullet">
    /// <item>A reference of another height or width is resized to the latent's with PyTorch's legacy "nearest" (3x3 to 2x2 takes cells 0 and 1 of each
    /// axis, not "nearest-exact"'s 0 and 2).</item>
    /// <item>The reference batch is read around the latent's batch (latent picture i gets reference picture i mod n); a one-channel reference is
    /// spread over the channels.</item>
    /// </list>
    /// Strength 0 hands the latent back unchanged.
    /// </summary>
    public LatentOperation Execute(LatentBatch reference,
        [Input(Min = 0, Max = 1, Step = 0.0001, Tooltip = "0 leaves the latent unchanged; 1 matches the resized reference latent.")] double strength = 1.0) =>
        new((latent, first) => Blend(latent, reference, (float)strength, first));

    static LatentBatch Blend(LatentBatch latent, LatentBatch reference, float strength, int first)
    {
        if (strength == 0) return latent;
        int h = latent.Height, w = latent.Width, plane = h * w;
        var refData = reference.Height == h && reference.Width == w
            ? reference.Data
            : TorchInterpolate.Resize(reference.Data, reference.Batch * reference.Channels, reference.Height, reference.Width, h, w, "nearest");
        if (reference.Batch == 0) throw new ArgumentException("The reference latent has no pictures.", nameof(reference));
        int c = Latents.Broadcast(latent.Channels, reference.Channels, 1), n = latent.Batch;
        var r = new float[n * c * plane];
        for (int i = 0; i < n; i++)
        {
            int rb = (first + i) % reference.Batch;
            for (int k = 0; k < c; k++)
            {
                int ia = (i * latent.Channels + (latent.Channels == 1 ? 0 : k)) * plane;
                int ib = (rb * reference.Channels + (reference.Channels == 1 ? 0 : k)) * plane;
                int o = (i * c + k) * plane;
                for (int p = 0; p < plane; p++) r[o + p] = TorchMath.LerpFma(latent.Data[ia + p], refData[ib + p], strength);
            }
        }
        return new LatentBatch(n, c, h, w, r);
    }
}

[Node(Category = "model/latent/advanced/operations", DisplayName = "LatentOperationTonemapReinhard")]
public sealed class LatentOperationTonemapReinhard
{
    /// <summary>
    /// Reinhard tone mapping of the channel-vector lengths, keeping each vector's direction. Per picture, the lengths' mean and unbiased standard
    /// deviation over the positions give a ceiling <c>top = (5 std + mean) * multiplier</c>; each length L becomes <c>x / (x + 1) * top</c> with
    /// <c>x = L / top</c>. Multiplier 0 makes every value NaN (0 / 0), as in ComfyUI. A position whose channels are all 0 stays 0 (measured).
    /// </summary>
    public LatentOperation Execute([Input(Min = 0, Max = 100, Step = 0.01)] double multiplier = 1.0) =>
        new((latent, _) => Tonemap(latent, (float)multiplier));

    static LatentBatch Tonemap(LatentBatch latent, float multiplier)
    {
        int n = latent.Batch, c = latent.Channels, plane = latent.Height * latent.Width;
        var lengths = ChannelNorm.Lengths(latent);
        var r = new float[latent.Data.Length];
        for (int b = 0; b < n; b++)
        {
            double mean = 0;
            for (int p = 0; p < plane; p++) mean += lengths[b * plane + p];
            mean /= plane;
            double sq = 0;
            for (int p = 0; p < plane; p++)
            {
                double d = lengths[b * plane + p] - mean;
                sq += d * d;
            }
            float std = (float)Math.Sqrt(sq / (plane - 1));
            float top = (std * 5f + (float)mean) * multiplier;
            for (int p = 0; p < plane; p++)
            {
                float length = lengths[b * plane + p];
                float x = length / top;
                float mapped = x / (x + 1f) * top;
                for (int k = 0; k < c; k++)
                {
                    int i = (b * c + k) * plane + p;
                    r[i] = length == 0 ? 0 : latent.Data[i] / length * mapped;
                }
            }
        }
        return new LatentBatch(n, c, latent.Height, latent.Width, r);
    }
}

[Node(Category = "model/latent/advanced/operations", DisplayName = "LatentOperationSharpen")]
public sealed class LatentOperationSharpen
{
    /// <summary>
    /// An unsharp mask on the directions of the channel vectors: each channel is divided by the vector length (plus 1e-6), convolved with
    /// <see cref="Kernel"/> over reflected edges, and multiplied by the same length again.
    /// </summary>
    public LatentOperation Execute([Input(Min = 1, Max = 31)] int sharpenRadius = 9,
        [Input(Min = 0.1, Max = 10, Step = 0.1)] double sigma = 1.0,
        [Input(Min = 0, Max = 5, Step = 0.01)] double alpha = 0.1)
    {
        var kernel = Kernel(sharpenRadius, sigma, alpha);
        return new((latent, _) => Sharpen(latent, sharpenRadius, kernel));
    }

    /// <summary>
    /// The (2r+1)² sharpening kernel: a Gaussian over a grid that spans -1..1 on both axes (whatever the radius), <c>exp(-d² / (2 sigma²))</c>
    /// normalised to sum 1, scaled by <c>-10 alpha</c>, with <c>1 + 10 alpha</c> added to the centre so the kernel sums to 1.
    /// </summary>
    public static float[] Kernel(int radius, double sigma, double alpha)
    {
        int size = 2 * radius + 1;
        var g = new double[size * size];
        double sum = 0;
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
            {
                double gy = size == 1 ? -1 : -1 + 2.0 * y / (size - 1), gx = size == 1 ? -1 : -1 + 2.0 * x / (size - 1);
                sum += g[y * size + x] = Math.Exp(-(gx * gx + gy * gy) / (2 * sigma * sigma));
            }
        var k = new float[size * size];
        for (int i = 0; i < k.Length; i++) k[i] = (float)(-10 * alpha * g[i] / sum);
        int centre = radius * size + radius;
        k[centre] = (float)(-10 * alpha * g[centre] / sum + 1 + 10 * alpha);
        return k;
    }

    static LatentBatch Sharpen(LatentBatch latent, int radius, float[] kernel)
    {
        int n = latent.Batch, c = latent.Channels, h = latent.Height, w = latent.Width, plane = h * w, size = 2 * radius + 1;
        // Reflect padding needs the radius below each side (PyTorch's message, measured).
        if (radius >= w) throw new ArgumentException($"Argument #4: Padding size should be less than the corresponding input dimension, but got: padding ({radius}, {radius}) at dimension 3 of input [{n}, {c}, {h}, {w}]");
        if (radius >= h) throw new ArgumentException($"Argument #6: Padding size should be less than the corresponding input dimension, but got: padding ({radius}, {radius}) at dimension 2 of input [{n}, {c}, {h}, {w}]");
        var lengths = ChannelNorm.Lengths(latent);
        var unit = new float[latent.Data.Length];
        for (int b = 0; b < n; b++)
            for (int k = 0; k < c; k++)
                for (int p = 0; p < plane; p++)
                    unit[(b * c + k) * plane + p] = latent.Data[(b * c + k) * plane + p] / (lengths[b * plane + p] + 1e-6f);
        static int Reflect(int i, int n) => i < 0 ? -i : i >= n ? 2 * (n - 1) - i : i;
        var r = new float[latent.Data.Length];
        for (int b = 0; b < n; b++)
            for (int k = 0; k < c; k++)
            {
                int o = (b * c + k) * plane;
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        double acc = 0;
                        for (int ky = 0; ky < size; ky++)
                        {
                            int row = o + Reflect(y + ky - radius, h) * w;
                            for (int kx = 0; kx < size; kx++) acc += (double)kernel[ky * size + kx] * unit[row + Reflect(x + kx - radius, w)];
                        }
                        r[o + y * w + x] = (float)acc * (lengths[b * plane + y * w + x] + 1e-6f);
                    }
            }
        return new LatentBatch(n, c, h, w, r);
    }
}
