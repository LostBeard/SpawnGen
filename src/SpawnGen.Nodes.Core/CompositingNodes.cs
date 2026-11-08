using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// The alpha compositing nodes. A MASK here is transparency (1 - alpha), as everywhere in ComfyUI's image nodes: alpha = 1 - mask going in and
// mask = 1 - alpha coming out. Behavior measured with ComfyUI (2026-11-08) by running each node on chosen tensors; the arithmetic is single
// precision in the order the measurements require (bit-identical on random inputs).

/// <summary>
/// Composites <paramref name="source"/> onto <paramref name="destination"/> with one of the 18 Porter-Duff / blend modes. Measured with ComfyUI:
/// <list type="bullet">
/// <item>The output has the destination's size. A source picture, source alpha or destination alpha of another size is first cut to the
/// destination's aspect ratio around its centre (the trimmed amount per side rounded half to even), then resized bicubically (PyTorch's
/// interpolate, align_corners false); the resized values are not clamped.</item>
/// <item>The output batch is the smallest of the four batches; picture i of each input is used for picture i of the output.</item>
/// <item>Source and destination must have the same number of channels; every channel is treated as a colour.</item>
/// <item>The colours are premultiplied by their alpha, combined, and divided by the result's alpha where that alpha is above 1e-5 (0 elsewhere);
/// the colour is then clamped to 0..1 (NaN stays NaN). The output mask is 1 - alpha, not clamped (except ADD, whose alpha is clamped).</item>
/// </list>
/// </summary>
[Node(Category = "image/compositing", DisplayName = "Porter-Duff Image Composite", OutputNames = ["IMAGE", "MASK"])]
public sealed class PorterDuffImageComposite
{
    /// <summary>The modes, in the order the combo lists them.</summary>
    public static readonly string[] Modes =
    [
        "ADD", "CLEAR", "DARKEN", "DST", "DST_ATOP", "DST_IN", "DST_OUT", "DST_OVER", "LIGHTEN", "MULTIPLY", "OVERLAY", "SCREEN",
        "SRC", "SRC_ATOP", "SRC_IN", "SRC_OUT", "SRC_OVER", "XOR",
    ];

    /// <summary>The modes in the order of <see cref="Modes"/>.</summary>
    enum Op { Add, Clear, Darken, Dst, DstAtop, DstIn, DstOut, DstOver, Lighten, Multiply, Overlay, Screen, Src, SrcAtop, SrcIn, SrcOut, SrcOver, Xor }

    public (ImageBatch Image, MaskBatch Mask) Execute(ImageBatch source, MaskBatch sourceAlpha, ImageBatch destination, MaskBatch destinationAlpha,
        [Input(Options = ["ADD", "CLEAR", "DARKEN", "DST", "DST_ATOP", "DST_IN", "DST_OUT", "DST_OVER", "LIGHTEN", "MULTIPLY", "OVERLAY", "SCREEN",
            "SRC", "SRC_ATOP", "SRC_IN", "SRC_OUT", "SRC_OVER", "XOR"])] string mode = "DST")
    {
        int index = Array.IndexOf(Modes, mode);
        if (index < 0) throw new ArgumentException($"Unknown Porter-Duff mode '{mode}'.", nameof(mode));
        var m = (Op)index;
        // Measured: ComfyUI fails on differing channel counts too, with an assertion that carries no message.
        if (source.Channels != destination.Channels)
            throw new ArgumentException($"The source has {source.Channels} channels and the destination {destination.Channels}: they must be the same.");
        int n = Math.Min(Math.Min(source.Batch, sourceAlpha.Batch), Math.Min(destination.Batch, destinationAlpha.Batch));
        // Measured: a batch of none fails in ComfyUI with this PyTorch message.
        if (n == 0) throw new ArgumentException("stack expects a non-empty TensorList");

        int h = destination.Height, w = destination.Width, c = destination.Channels, area = h * w;
        var image = new ImageBatch(n, h, w, c);
        var mask = new MaskBatch(n, h, w);
        for (int b = 0; b < n; b++)
        {
            // Fitted in this order so that, when several cannot be cut (see Fit), the same one is reported as in ComfyUI (measured).
            var dstMask = Fit(Item(destinationAlpha.Data, b, destinationAlpha.Height * destinationAlpha.Width), destinationAlpha.Height, destinationAlpha.Width, 1, h, w);
            var src = Fit(Item(source.Data, b, source.Height * source.Width * c), source.Height, source.Width, c, h, w);
            var srcMask = Fit(Item(sourceAlpha.Data, b, sourceAlpha.Height * sourceAlpha.Width), sourceAlpha.Height, sourceAlpha.Width, 1, h, w);
            var dst = Item(destination.Data, b, area * c);
            for (int p = 0; p < area; p++)
            {
                float aS = 1f - srcMask[p], aD = 1f - dstMask[p];
                float alpha = Alpha(m, aS, aD);
                int o = (b * area + p) * c;
                for (int k = 0; k < c; k++)
                {
                    float colour = Colour(m, aS, aD, src[p * c + k] * aS, dst[p * c + k] * aD);
                    float straight = alpha > 1e-5f ? colour / alpha : 0f;
                    image.Data[o + k] = Math.Clamp(straight, 0f, 1f);
                }
                mask.Data[b * area + p] = 1f - alpha;
            }
        }
        return (image, mask);
    }

    /// <summary>
    /// The result's alpha. Porter &amp; Duff 1984 for the operators (alpha = Fa * aS + Fb * aD); the blend modes (DARKEN, LIGHTEN, MULTIPLY, OVERLAY,
    /// SCREEN) take the union aS + aD - aS * aD as in W3C Compositing and Blending Level 1. Each expression is written in the order that reproduces
    /// ComfyUI's float32 results exactly (measured: SRC_OVER as aS + (1 - aS) * aD, DST_OVER as aD + (1 - aD) * aS).
    /// </summary>
    static float Alpha(Op mode, float aS, float aD) => mode switch
    {
        Op.Add => Math.Clamp(aS + aD, 0f, 1f),
        Op.Clear => 0f,
        Op.Darken or Op.Lighten or Op.Multiply or Op.Overlay or Op.Screen => aS + aD - aS * aD,
        Op.Dst => aD,
        Op.DstAtop => aS,
        Op.DstIn or Op.SrcIn => aS * aD,
        Op.DstOut => (1f - aS) * aD,
        Op.DstOver => aD + (1f - aD) * aS,
        Op.Src => aS,
        Op.SrcAtop => aD,
        Op.SrcOut => (1f - aD) * aS,
        Op.SrcOver => aS + (1f - aS) * aD,
        Op.Xor => (1f - aD) * aS + (1f - aS) * aD,
        _ => throw new ArgumentOutOfRangeException(nameof(mode)),
    };

    /// <summary>
    /// The result's premultiplied colour from the premultiplied colours <paramref name="cS"/> and <paramref name="cD"/>. The operators follow
    /// Porter &amp; Duff 1984 (Android's PorterDuff.Mode lists the same formulas); ADD clamps to 1 (and 0) like its alpha. The blend modes are the
    /// W3C separable blend in premultiplied form, cS (1 - aD) + cD (1 - aS) + aS aD B(straight colours), with B:
    /// darken min, lighten max, multiply the product, overlay Android's form (2 cS cD where 2 cD &lt; aD, else aS aD - 2 (aD - cD)(aS - cS));
    /// SCREEN is cS + cD - cS cD.
    /// </summary>
    static float Colour(Op mode, float aS, float aD, float cS, float cD) => mode switch
    {
        Op.Add => Math.Clamp(cS + cD, 0f, 1f),
        Op.Clear => 0f,
        Op.Darken => (1f - aD) * cS + (1f - aS) * cD + Math.Min(cS * aD, cD * aS),
        Op.Dst => cD,
        Op.DstAtop => aS * cD + (1f - aD) * cS,
        Op.DstIn => cD * aS,
        Op.DstOut => (1f - aS) * cD,
        Op.DstOver => cD + (1f - aD) * cS,
        Op.Lighten => (1f - aD) * cS + (1f - aS) * cD + Math.Max(cS * aD, cD * aS),
        Op.Multiply => (1f - aD) * cS + (1f - aS) * cD + cS * cD,
        Op.Overlay => (1f - aD) * cS + (1f - aS) * cD + (2f * cD < aD ? 2f * cS * cD : aS * aD - 2f * (aD - cD) * (aS - cS)),
        Op.Screen => cS + cD - cS * cD,
        Op.Src => cS,
        Op.SrcAtop => aD * cS + (1f - aS) * cD,
        Op.SrcIn => cS * aD,
        Op.SrcOut => (1f - aD) * cS,
        Op.SrcOver => cS + (1f - aS) * cD,
        Op.Xor => (1f - aD) * cS + (1f - aS) * cD,
        _ => throw new ArgumentOutOfRangeException(nameof(mode)),
    };

    static float[] Item(float[] data, int index, int length)
    {
        var item = new float[length];
        Array.Copy(data, (long)index * length, item, 0, length);
        return item;
    }

    /// <summary>
    /// One picture (height x width x channels, channels last) brought to <paramref name="height"/> x <paramref name="width"/>: measured with
    /// ComfyUI, the side that is too long for the target's aspect ratio loses round((size - size * (target aspect / own aspect)) / 2) pixels at
    /// each end (Python's round, half to even, in double), and the rest is resized bicubically without clamping.
    /// </summary>
    static float[] Fit(float[] item, int h, int w, int channels, int height, int width)
    {
        if (h == height && w == width) return item;
        int top = 0, left = 0, ch = h, cw = w;
        double own = (double)w / h, target = (double)width / height;
        if (own > target)
        {
            left = (int)Math.Round((w - w * (target / own)) / 2);
            cw = Math.Max(0, w - 2 * left);
        }
        else if (own < target)
        {
            top = (int)Math.Round((h - h * (own / target)) / 2);
            ch = Math.Max(0, h - 2 * top);
        }
        // Measured: a cut that leaves nothing fails with PyTorch's interpolate message.
        if (ch == 0 || cw == 0)
            throw new ArgumentException($"Input and output sizes should be greater than 0, but got input (H: {ch}, W: {cw}) output (H: {height}, W: {width})");

        var planes = new float[channels * ch * cw];
        for (int k = 0; k < channels; k++)
            for (int y = 0; y < ch; y++)
                for (int x = 0; x < cw; x++)
                    planes[(k * ch + y) * cw + x] = item[((top + y) * w + left + x) * channels + k];
        var resized = TorchInterpolate.Resize(planes, channels, ch, cw, height, width, "bicubic");
        var result = new float[height * width * channels];
        for (int k = 0; k < channels; k++)
            for (int p = 0; p < height * width; p++)
                result[p * channels + k] = resized[k * height * width + p];
        return result;
    }
}

/// <summary>
/// The colour channels of each picture and its transparency as a mask. Measured with ComfyUI: the picture keeps its first three channels (all
/// of them when it has fewer); the mask is 1 - the fourth channel, unclamped, or all zeros (opaque) when there is no fourth channel.
/// </summary>
[Node(Category = "image/compositing", DisplayName = "Split Image with Alpha", OutputNames = ["IMAGE", "MASK"])]
public sealed class SplitImageWithAlpha
{
    public (ImageBatch Image, MaskBatch Mask) Execute(ImageBatch image)
    {
        // Measured: a batch of none fails in ComfyUI with this PyTorch message.
        if (image.Batch == 0) throw new ArgumentException("stack expects a non-empty TensorList");
        int c = image.Channels, kept = Math.Min(c, 3), pixels = image.Batch * image.Height * image.Width;
        var rgb = new ImageBatch(image.Batch, image.Height, image.Width, kept);
        var mask = new MaskBatch(image.Batch, image.Height, image.Width);
        for (int p = 0; p < pixels; p++)
        {
            Array.Copy(image.Data, (long)p * c, rgb.Data, (long)p * kept, kept);
            if (c > 3) mask.Data[p] = 1f - image.Data[p * c + 3];
        }
        return (rgb, mask);
    }
}

/// <summary>
/// The picture with the mask as its alpha channel (alpha = 1 - mask, unclamped). Measured with ComfyUI:
/// <list type="bullet">
/// <item>The output batch is the larger of the two; the shorter one repeats from its start (picture i uses item i mod batch).</item>
/// <item>A mask of another size is resized to the picture's size bilinearly (PyTorch's interpolate, align_corners false, no anti-aliasing), the
/// alpha taken from the resized mask.</item>
/// <item>The picture keeps its first three channels (all of them when it has fewer) and gets the alpha after them; an alpha channel it had is
/// replaced.</item>
/// </list>
/// </summary>
[Node(Category = "image/compositing", DisplayName = "Join Image with Alpha")]
public sealed class JoinImageWithAlpha
{
    public ImageBatch Execute(ImageBatch image, MaskBatch alpha)
    {
        // Measured: an empty batch on either side fails in ComfyUI with "division by zero".
        if (image.Batch == 0 || alpha.Batch == 0) throw new ArgumentException("division by zero");
        int h = image.Height, w = image.Width, area = h * w, c = image.Channels, kept = Math.Min(c, 3);
        var mask = alpha.Height == h && alpha.Width == w ? alpha.Data : Bilinear(alpha.Data, alpha.Batch, alpha.Height, alpha.Width, h, w);
        int n = Math.Max(image.Batch, alpha.Batch);
        var result = new ImageBatch(n, h, w, kept + 1);
        for (int b = 0; b < n; b++)
        {
            int ib = b % image.Batch, ab = b % alpha.Batch;
            for (int p = 0; p < area; p++)
            {
                int o = (b * area + p) * (kept + 1);
                Array.Copy(image.Data, ((long)ib * area + p) * c, result.Data, o, kept);
                result.Data[o + kept] = 1f - mask[ab * area + p];
            }
        }
        return result;
    }

    /// <summary>
    /// PyTorch's <c>interpolate(mode="bilinear", align_corners=False)</c> (documented: source coordinate (x + 0.5) * in / out - 0.5, clamped at 0)
    /// on <paramref name="planes"/> planes of h x w. The float32 arithmetic order reproduces PyTorch's CPU results bit for bit (measured on 210
    /// random sizes): the scale in / out and the coordinates in float, the four weights multiplied out and summed left to right.
    /// </summary>
    static float[] Bilinear(float[] src, int planes, int h, int w, int newH, int newW)
    {
        var (y0, y1, ly0, ly1) = Taps(h, newH);
        var (x0, x1, lx0, lx1) = Taps(w, newW);
        var dst = new float[planes * newH * newW];
        for (int p = 0; p < planes; p++)
        {
            int s = p * h * w, d = p * newH * newW;
            for (int i = 0; i < newH; i++)
            {
                int r0 = s + y0[i] * w, r1 = s + y1[i] * w;
                float a0 = ly0[i], a1 = ly1[i];
                for (int j = 0; j < newW; j++)
                {
                    float b0 = lx0[j], b1 = lx1[j];
                    dst[d + i * newW + j] = a0 * b0 * src[r0 + x0[j]] + a0 * b1 * src[r0 + x1[j]] + a1 * b0 * src[r1 + x0[j]] + a1 * b1 * src[r1 + x1[j]];
                }
            }
        }
        return dst;
    }

    static (int[] I0, int[] I1, float[] L0, float[] L1) Taps(int size, int newSize)
    {
        var i0 = new int[newSize];
        var i1 = new int[newSize];
        var l0 = new float[newSize];
        var l1 = new float[newSize];
        float scale = (float)size / newSize;
        for (int d = 0; d < newSize; d++)
        {
            float real = Math.Max(scale * (d + 0.5f) - 0.5f, 0f);
            int index = (int)real;
            i0[d] = index;
            i1[d] = index < size - 1 ? index + 1 : index;
            l1[d] = real - index;
            l0[d] = 1f - l1[d];
        }
        return (i0, i1, l0, l1);
    }
}
