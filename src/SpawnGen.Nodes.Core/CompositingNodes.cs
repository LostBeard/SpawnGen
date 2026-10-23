using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>ComfyUI's Porter-Duff operators (nodes_compositing.py), in the order of its PorterDuffMode enum.</summary>
public enum PorterDuffMode { ADD, CLEAR, DARKEN, DST, DST_ATOP, DST_IN, DST_OUT, DST_OVER, LIGHTEN, MULTIPLY, OVERLAY, SCREEN, SRC, SRC_ATOP, SRC_IN, SRC_OUT, SRC_OVER, XOR }

[Node(Category = "image/compositing", DisplayName = "Porter-Duff Image Composite", OutputNames = ["IMAGE", "MASK"])]
public sealed class PorterDuffImageComposite
{
    public static string[] Modes() => Enum.GetNames<PorterDuffMode>();

    /// <summary>
    /// Composites <paramref name="source"/> over (or into, out of, ...) <paramref name="destination"/>. The alphas are MASKs in ComfyUI's sense
    /// (1 = transparent), the result's mask too. A destination alpha of another size, a source of another size and then a source alpha of another size
    /// are scaled to fit (bicubic, centred crop), picture by picture; the batch is the smallest of the four.
    /// </summary>
    public (ImageBatch Image, MaskBatch Mask) Execute(ImageBatch source, MaskBatch sourceAlpha, ImageBatch destination, MaskBatch destinationAlpha,
        [Input(OptionsProvider = nameof(Modes))] string mode = "DST")
    {
        if (!Enum.TryParse<PorterDuffMode>(mode, out var op) || !Enum.IsDefined(op) || mode != op.ToString())
            throw new ArgumentException($"Unknown mode '{mode}'.", nameof(mode));
        if (source.Channels != destination.Channels) throw new ArgumentException("The source and the destination need the same number of channels.");
        int batch = Math.Min(Math.Min(source.Batch, sourceAlpha.Batch), Math.Min(destination.Batch, destinationAlpha.Batch));
        int h = destination.Height, w = destination.Width, c = destination.Channels;
        var outImage = new ImageBatch(batch, h, w, c);
        var outMask = new MaskBatch(batch, h, w);
        for (int b = 0; b < batch; b++)
        {
            var src = Pixels.Frame(source, b);
            var dst = Pixels.Frame(destination, b);
            var srcA = Plane(sourceAlpha, b);
            var dstA = Plane(destinationAlpha, b);
            int saw = sourceAlpha.Width, sah = sourceAlpha.Height, daw = destinationAlpha.Width, dah = destinationAlpha.Height;
            if (dah != h || daw != w) { dstA = CommonUpscale.Resize(dstA, daw, dah, 1, w, h, "bicubic", "center"); daw = w; dah = h; }
            if (source.Height != h || source.Width != w) src = CommonUpscale.Resize(src, source.Width, source.Height, c, w, h, "bicubic", "center");
            if (sah != dah || saw != daw) srcA = CommonUpscale.Resize(srcA, saw, sah, 1, daw, dah, "bicubic", "center");

            for (int p = 0; p < h * w; p++)
            {
                float sa = 1 - srcA[p], da = 1 - dstA[p]; // mask to alpha
                float outA = Alpha(op, sa, da);
                for (int ch = 0; ch < c; ch++)
                {
                    float si = src[p * c + ch] * sa, di = dst[p * c + ch] * da; // premultiplied
                    float o = Colour(op, si, sa, di, da);
                    o = outA > 1e-5f ? o / outA : 0f;
                    outImage.Data[(b * h * w + p) * c + ch] = Math.Clamp(o, 0f, 1f);
                }
                outMask.Data[b * h * w + p] = 1 - outA;
            }
        }
        return (outImage, outMask);
    }

    static float[] Plane(MaskBatch m, int b)
    {
        var one = new float[m.Height * m.Width];
        Array.Copy(m.Data, b * one.Length, one, 0, one.Length);
        return one;
    }

    static float Alpha(PorterDuffMode op, float sa, float da) => op switch
    {
        PorterDuffMode.ADD => Math.Clamp(sa + da, 0f, 1f),
        PorterDuffMode.CLEAR => 0f,
        PorterDuffMode.DST => da,
        PorterDuffMode.DST_ATOP => sa,
        PorterDuffMode.DST_IN or PorterDuffMode.SRC_IN => sa * da,
        PorterDuffMode.DST_OUT => (1 - sa) * da,
        PorterDuffMode.DST_OVER => da + (1 - da) * sa,
        PorterDuffMode.SRC => sa,
        PorterDuffMode.SRC_ATOP => da,
        PorterDuffMode.SRC_OUT => (1 - da) * sa,
        PorterDuffMode.SRC_OVER => sa + (1 - sa) * da,
        PorterDuffMode.XOR => (1 - da) * sa + (1 - sa) * da,
        _ => sa + da - sa * da, // DARKEN, LIGHTEN, MULTIPLY, OVERLAY, SCREEN
    };

    /// <summary>The premultiplied colour of one channel, ComfyUI's formulas term for term.</summary>
    static float Colour(PorterDuffMode op, float si, float sa, float di, float da) => op switch
    {
        PorterDuffMode.ADD => Math.Clamp(si + di, 0f, 1f),
        PorterDuffMode.CLEAR => 0f,
        PorterDuffMode.DARKEN => (1 - da) * si + (1 - sa) * di + Math.Min(si * da, di * sa),
        PorterDuffMode.DST => di,
        PorterDuffMode.DST_ATOP => sa * di + (1 - da) * si,
        PorterDuffMode.DST_IN => di * sa,
        PorterDuffMode.DST_OUT => (1 - sa) * di,
        PorterDuffMode.DST_OVER => di + (1 - da) * si,
        PorterDuffMode.LIGHTEN => (1 - da) * si + (1 - sa) * di + Math.Max(si * da, di * sa),
        PorterDuffMode.MULTIPLY => (1 - da) * si + (1 - sa) * di + si * di,
        PorterDuffMode.OVERLAY => (1 - da) * si + (1 - sa) * di + (2 * di < da ? 2 * si * di : sa * da - 2 * (sa - si) * (da - di)),
        PorterDuffMode.SCREEN => si + di - si * di,
        PorterDuffMode.SRC => si,
        PorterDuffMode.SRC_ATOP => da * si + (1 - sa) * di,
        PorterDuffMode.SRC_IN => si * da,
        PorterDuffMode.SRC_OUT => (1 - da) * si,
        PorterDuffMode.SRC_OVER => si + (1 - sa) * di,
        _ => (1 - da) * si + (1 - sa) * di, // XOR
    };
}

[Node(Category = "image/compositing", DisplayName = "Split Image with Alpha", OutputNames = ["IMAGE", "MASK"])]
public sealed class SplitImageWithAlpha
{
    /// <summary>The colour channels (the first three) and the alpha channel as a MASK (1 - alpha; all 0 for a picture without alpha).</summary>
    public (ImageBatch Image, MaskBatch Mask) Execute(ImageBatch image)
    {
        int c = image.Channels, keep = Math.Min(c, 3), n = image.Height * image.Width;
        var rgb = new ImageBatch(image.Batch, image.Height, image.Width, keep);
        var mask = new MaskBatch(image.Batch, image.Height, image.Width);
        for (int p = 0; p < image.Batch * n; p++)
        {
            for (int ch = 0; ch < keep; ch++) rgb.Data[p * keep + ch] = image.Data[p * c + ch];
            mask.Data[p] = 1f - (c > 3 ? image.Data[p * c + 3] : 1f);
        }
        return (rgb, mask);
    }
}

[Node(Category = "image/compositing", DisplayName = "Join Image with Alpha")]
public sealed class JoinImageWithAlpha
{
    /// <summary>
    /// The first three channels plus 1 - <paramref name="alpha"/> as a fourth (alpha) channel. The mask is scaled to the picture (bilinear) and
    /// both batches are repeated to the larger one.
    /// </summary>
    public ImageBatch Execute(ImageBatch image, MaskBatch alpha)
    {
        int batch = Math.Max(image.Batch, alpha.Batch), h = image.Height, w = image.Width, c = image.Channels, keep = Math.Min(c, 3);
        var resized = alpha.Height == h && alpha.Width == w ? alpha : CommonUpscale.Resize(alpha, w, h, "bilinear");
        var result = new ImageBatch(batch, h, w, keep + 1);
        for (int b = 0; b < batch; b++)
        {
            int ib = b % image.Batch, ab = b % alpha.Batch;
            for (int p = 0; p < h * w; p++)
            {
                int o = (b * h * w + p) * (keep + 1);
                for (int ch = 0; ch < keep; ch++) result.Data[o + ch] = image.Data[(ib * h * w + p) * c + ch];
                result.Data[o + keep] = 1f - resized.Data[ab * h * w + p];
            }
        }
        return result;
    }
}
