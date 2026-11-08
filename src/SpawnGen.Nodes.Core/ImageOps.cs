using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// The basic image nodes. Pictures are (batch, height, width, channels) float32, values nominally 0..1. Behavior measured with ComfyUI (2026-11-08)
// by running each node on chosen tensors; the measurements are quoted next to the code they decided.

/// <summary>Frame-by-frame helpers for <see cref="ImageBatch"/>: one frame is its interleaved (height, width, channels) floats.</summary>
public static class Pixels
{
    /// <summary>A copy of frame <paramref name="b"/>.</summary>
    public static float[] Frame(ImageBatch images, int b)
    {
        int n = images.Height * images.Width * images.Channels;
        var frame = new float[n];
        Array.Copy(images.Data, (long)b * n, frame, 0, n);
        return frame;
    }

    /// <summary>The frames, each <paramref name="height"/> x <paramref name="width"/> x <paramref name="channels"/>, joined into one batch.</summary>
    public static ImageBatch From(IReadOnlyList<float[]> frames, int height, int width, int channels)
    {
        var result = new ImageBatch(frames.Count, height, width, channels);
        int n = height * width * channels;
        for (int b = 0; b < frames.Count; b++)
        {
            if (frames[b].Length != n) throw new ArgumentException($"Frame {b} holds {frames[b].Length} values, expected {n} ({width}x{height}x{channels}).", nameof(frames));
            Array.Copy(frames[b], 0, result.Data, (long)b * n, n);
        }
        return result;
    }

    /// <summary>
    /// Runs <paramref name="step"/> on a copy of every frame and joins the results. The result keeps the channel count; its size is
    /// <paramref name="height"/> x <paramref name="width"/> when given, else the input's.
    /// </summary>
    public static ImageBatch Map(ImageBatch image, Func<float[], float[]> step, int? height = null, int? width = null)
    {
        var frames = new float[image.Batch][];
        for (int b = 0; b < image.Batch; b++) frames[b] = step(Frame(image, b));
        return From(frames, height ?? image.Height, width ?? image.Width, image.Channels);
    }

    /// <summary>
    /// Brings <paramref name="image"/> to <paramref name="channels"/> channels the way the measured nodes do: one channel of 1.0 is appended when it has
    /// fewer (only one, so 1 next to 3 still differs), the extra channels are dropped when it has more.
    /// </summary>
    internal static ImageBatch MatchChannels(ImageBatch image, int channels)
    {
        int c = image.Channels;
        if (c == channels) return image;
        int outC = c < channels ? c + 1 : channels, pixels = image.Batch * image.Height * image.Width;
        var result = new ImageBatch(image.Batch, image.Height, image.Width, outC);
        for (int p = 0; p < pixels; p++)
        {
            Array.Copy(image.Data, (long)p * c, result.Data, (long)p * outC, Math.Min(c, outC));
            if (outC > c) result.Data[(long)p * outC + c] = 1f;
        }
        return result;
    }

    /// <summary>
    /// Resizes every picture with <see cref="CommonUpscale"/>. A size of 0 fails as in ComfyUI (measured): "area" gives an empty picture, "lanczos"
    /// Pillow's error, the other modes PyTorch's.
    /// </summary>
    internal static ImageBatch Resize(ImageBatch image, int width, int height, string method, string crop = "disabled")
    {
        if (width <= 0 || height <= 0)
        {
            if (!Resample.MethodNames.Contains(method)) throw new ArgumentException($"Unknown upscale method '{method}'.", nameof(method));
            return method switch
            {
                "area" => new ImageBatch(image.Batch, Math.Max(height, 0), Math.Max(width, 0), image.Channels),
                "lanczos" => throw new ArgumentException("height and width must be > 0"),
                _ => throw new ArgumentException(
                    $"Input and output sizes should be greater than 0, but got input (H: {image.Height}, W: {image.Width}) output (H: {height}, W: {width})"),
            };
        }
        return CommonUpscale.Resize(image, width, height, method, crop);
    }

    /// <summary>The message PyTorch gives when two sizes cannot be broadcast together.</summary>
    internal static ArgumentException BroadcastMismatch(int a, int b, int dimension) =>
        new($"The size of tensor a ({a}) must match the size of tensor b ({b}) at non-singleton dimension {dimension}");
}

/// <summary>A batch of pictures of one colour: <c>color</c> is 0xRRGGBB, each channel stored as the float32 of value / 255 (measured, bit for bit).</summary>
[Node(Category = "image", DisplayName = "Empty Image")]
public sealed class EmptyImage
{
    public ImageBatch Execute(
        [Input(Min = 1, Max = 16384, Step = 1)] int width = 512,
        [Input(Min = 1, Max = 16384, Step = 1)] int height = 512,
        [Input(Min = 1, Max = 4096)] int batchSize = 1,
        [Input(Min = 0, Max = 16777215, Step = 1)] int color = 0)
    {
        float r = ((color >> 16) & 0xFF) / 255f, g = ((color >> 8) & 0xFF) / 255f, b = (color & 0xFF) / 255f;
        var image = new ImageBatch(batchSize, height, width, 3);
        var data = image.Data;
        for (long i = 0; i < data.Length; i += 3) { data[i] = r; data[i + 1] = g; data[i + 2] = b; }
        return image;
    }
}

/// <summary>The deprecated two-input batch: measured to give the same pictures as Batch Images with the two inputs (see <see cref="Batching"/>).</summary>
[Node("ImageBatch", Category = "image/batch", DisplayName = "Batch Images (DEPRECATED)")]
public sealed class ImageBatchNode
{
    public ImageBatch Execute(ImageBatch image1, ImageBatch image2) => Batching.Images([image1, image2]);
}

/// <summary>
/// Blends <paramref name="image2"/> over <paramref name="image1"/>. Measured with ComfyUI:
/// <list type="bullet">
/// <item>The modes, with a = image1 and b = image2: normal b; multiply a*b; screen 1-(1-a)(1-b); overlay 2ab where a &lt;= 0.5, else 1-2(1-a)(1-b);
/// soft_light the W3C compositing formula (b &lt;= 0.5: a-(1-2b)a(1-a); else a+(2b-1)(D(a)-a) with D(a) = ((16a-12)a+4)a for a &lt;= 0.25, else
/// sqrt(a)); difference a-b (not its absolute value). All in float32, in that order, bit for bit.</item>
/// <item>The result is a*(1-f) + mode*f, clamped to 0..1.</item>
/// <item>image2 of another size is resized to image1's with bicubic and a centre crop. Channels: image2 with fewer channels gets one channel of 1.0,
/// with more it is cut to image1's count. With 4 channels only the first 3 are blended and image1's 4th is kept as it is (not clamped).</item>
/// <item>Batches broadcast when one of them is 1; other mismatches fail with PyTorch's message.</item>
/// </list>
/// </summary>
[Node(Category = "image/filters", DisplayName = "Blend Images")]
public sealed class ImageBlend
{
    public static readonly string[] Modes = ["normal", "multiply", "screen", "overlay", "soft_light", "difference"];

    public ImageBatch Execute(ImageBatch image1, ImageBatch image2,
        [Input(Min = 0.0, Max = 1.0, Step = 0.01)] double blendFactor = 0.5,
        [Input(Options = ["normal", "multiply", "screen", "overlay", "soft_light", "difference"])] string blendMode = "normal")
    {
        Func<float, float, float> mode = blendMode switch
        {
            "normal" => (_, b) => b,
            "multiply" => (a, b) => a * b,
            "screen" => (a, b) => 1f - (1f - a) * (1f - b),
            "overlay" => (a, b) => a <= 0.5f ? 2f * a * b : 1f - 2f * (1f - a) * (1f - b),
            "soft_light" => SoftLight,
            "difference" => (a, b) => a - b,
            _ => throw new ArgumentException($"Unsupported blend mode: {blendMode}", nameof(blendMode)),
        };
        if (image2.Height != image1.Height || image2.Width != image1.Width)
            image2 = Pixels.Resize(image2, image1.Width, image1.Height, "bicubic", "center");
        int c = image1.Channels;
        image2 = Pixels.MatchChannels(image2, c);
        int b1 = image1.Batch, b2 = image2.Batch;
        // PyTorch names the operands in the order the mode's formula meets them: image2 first for soft_light (measured).
        bool secondFirst = blendMode == "soft_light";
        if (image2.Channels != c) throw secondFirst ? Pixels.BroadcastMismatch(image2.Channels, c, 3) : Pixels.BroadcastMismatch(c, image2.Channels, 3);
        if (b1 != b2 && b1 != 1 && b2 != 1) throw secondFirst ? Pixels.BroadcastMismatch(b2, b1, 0) : Pixels.BroadcastMismatch(b1, b2, 0);

        int batch = Math.Max(b1, b2), pixels = image1.Height * image1.Width, frame = pixels * c;
        int blended = c == 4 ? 3 : c;
        float keep = (float)(1.0 - blendFactor), take = (float)blendFactor;
        var result = new ImageBatch(batch, image1.Height, image1.Width, c);
        for (int b = 0; b < batch; b++)
        {
            long o1 = (long)(b % b1) * frame, o2 = (long)(b % b2) * frame, o = (long)b * frame;
            for (int p = 0; p < pixels; p++)
                for (int ch = 0; ch < c; ch++)
                {
                    int i = p * c + ch;
                    float a = image1.Data[o1 + i];
                    if (ch >= blended) { result.Data[o + i] = a; continue; }
                    float v = a * keep + mode(a, image2.Data[o2 + i]) * take;
                    result.Data[o + i] = Math.Clamp(v, 0f, 1f);
                }
        }
        return result;
    }

    static float SoftLight(float a, float b)
    {
        if (b <= 0.5f) return a - (1f - 2f * b) * a * (1f - a);
        float d = a <= 0.25f ? ((16f * a - 12f) * a + 4f) * a : MathF.Sqrt(a);
        return a + (2f * b - 1f) * (d - a);
    }
}

/// <summary>
/// Gaussian filters of radius r. Measured with ComfyUI: the kernel is (2r+1) x (2r+1) samples of exp(-(x² + y²) / (2 sigma²)) on an even grid from -1 to
/// 1 (not one sample per pixel), normalised to sum 1, applied to every channel (alpha too) with mirrored edges (the edge pixel is not repeated).
/// </summary>
static class Gaussian
{
    /// <summary>The normalised one-dimensional kernel; the two-dimensional one is its outer product with itself.</summary>
    static double[] Kernel(int radius, double sigma)
    {
        int size = 2 * radius + 1;
        var k = new double[size];
        double sum = 0;
        for (int i = 0; i < size; i++)
        {
            double x = -1.0 + 2.0 * i / (size - 1);
            k[i] = Math.Exp(-(x * x) / (2 * sigma * sigma));
            sum += k[i];
        }
        for (int i = 0; i < size; i++) k[i] /= sum;
        return k;
    }

    /// <summary>
    /// Mirror index for any distance outside 0..n-1. ComfyUI refuses a radius that is not smaller than the picture's side (PyTorch's reflect padding
    /// error); here the mirroring continues instead, so tiny pictures (one row, for example) still filter.
    /// </summary>
    static int Reflect(int i, int n)
    {
        if (n == 1) return 0;
        int period = 2 * (n - 1);
        i %= period;
        if (i < 0) i += period;
        return i < n ? i : period - i;
    }

    /// <summary>The blurred frame (interleaved h x w x c) in double precision.</summary>
    public static double[] Blur(float[] src, int w, int h, int c, int radius, double sigma)
    {
        var k = Kernel(radius, sigma);
        var across = new double[src.Length];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                for (int ch = 0; ch < c; ch++)
                {
                    double s = 0;
                    for (int j = -radius; j <= radius; j++) s += k[j + radius] * src[(y * w + Reflect(x + j, w)) * c + ch];
                    across[(y * w + x) * c + ch] = s;
                }
        var result = new double[src.Length];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                for (int ch = 0; ch < c; ch++)
                {
                    double s = 0;
                    for (int j = -radius; j <= radius; j++) s += k[j + radius] * across[(Reflect(y + j, h) * w + x) * c + ch];
                    result[(y * w + x) * c + ch] = s;
                }
        return result;
    }
}

/// <summary>Gaussian blur (see <see cref="Gaussian"/>); the values are not clamped (measured: 1.5 stays 1.5).</summary>
[Node(Category = "image/filters", DisplayName = "Blur Image")]
public sealed class ImageBlur
{
    public ImageBatch Execute(ImageBatch image,
        [Input(Min = 1, Max = 31, Step = 1)] int blurRadius = 1,
        [Input(Min = 0.1, Max = 10.0, Step = 0.1)] double sigma = 1.0)
    {
        if (blurRadius <= 0) return image;
        int w = image.Width, h = image.Height, c = image.Channels;
        return Pixels.Map(image, f => Array.ConvertAll(Gaussian.Blur(f, w, h, c, blurRadius, sigma), v => (float)v));
    }
}

/// <summary>
/// Unsharp filter. Measured with ComfyUI: the kernel is -10 * alpha times the Gaussian of <see cref="Gaussian"/> with its centre raised so that it sums
/// to 1, i.e. (1 + 10 alpha) * picture - 10 alpha * blurred picture; every channel is filtered and the result is clamped to 0..1.
/// </summary>
[Node(Category = "image/filters", DisplayName = "Sharpen Image")]
public sealed class ImageSharpen
{
    public ImageBatch Execute(ImageBatch image,
        [Input(Min = 1, Max = 31, Step = 1)] int sharpenRadius = 1,
        [Input(Min = 0.1, Max = 10.0, Step = 0.01)] double sigma = 1.0,
        [Input(Min = 0.0, Max = 5.0, Step = 0.01)] double alpha = 1.0)
    {
        if (sharpenRadius <= 0) return image;
        int w = image.Width, h = image.Height, c = image.Channels;
        double gain = 10 * alpha;
        return Pixels.Map(image, f =>
        {
            var blurred = Gaussian.Blur(f, w, h, c, sharpenRadius, sigma);
            var o = new float[f.Length];
            for (int i = 0; i < o.Length; i++) o[i] = (float)Math.Clamp((1 + gain) * f[i] - gain * blurred[i], 0.0, 1.0);
            return o;
        });
    }
}

/// <summary>
/// Pastes <paramref name="source"/> onto <paramref name="destination"/> with its top-left corner at (x, y). Measured with ComfyUI:
/// <list type="bullet">
/// <item>resize_source stretches the source to the destination's size first (bilinear, no crop).</item>
/// <item>The mask (all ones without one) is resized to the source's size (PyTorch bilinear) when it differs; each pixel is
/// mask * source + (1 - mask) * destination in float32, the mask used as given (not clamped).</item>
/// <item>Only the part inside the destination is pasted, from the source's (and mask's) top-left; a corner at or past the edge pastes nothing.</item>
/// <item>Source and mask pictures repeat in turn to the destination's batch size (or are cut to it).</item>
/// <item>Channels: a source with fewer channels gets one channel of 1.0 (an RGB source over RGBA is opaque), with more it is cut to the destination's
/// count; counts that still differ fail with PyTorch's message.</item>
/// </list>
/// </summary>
[Node(Category = "image/compositing", DisplayName = "Image Composite Masked")]
public sealed class ImageCompositeMasked
{
    public ImageBatch Execute(ImageBatch destination, ImageBatch source,
        [Input(Min = 0, Max = 16384, Step = 1)] int x = 0,
        [Input(Min = 0, Max = 16384, Step = 1)] int y = 0,
        bool resizeSource = false,
        MaskBatch? mask = null)
    {
        if (resizeSource) source = Pixels.Resize(source, destination.Width, destination.Height, "bilinear");
        int c = destination.Channels;
        source = Pixels.MatchChannels(source, c);
        if (source.Channels != c) throw Pixels.BroadcastMismatch(source.Channels, c, 1);
        int sw = source.Width, sh = source.Height;
        if (mask is not null && (mask.Width != sw || mask.Height != sh)) mask = CommonUpscale.Resize(mask, sw, sh, "bilinear");

        var result = new ImageBatch(destination.Batch, destination.Height, destination.Width, c, (float[])destination.Data.Clone());
        int W = destination.Width, H = destination.Height;
        int visibleW = Math.Min(sw, W - x), visibleH = Math.Min(sh, H - y);
        if (visibleW <= 0 || visibleH <= 0 || source.Batch == 0) return result;
        for (int b = 0; b < destination.Batch; b++)
        {
            long srcFrame = (long)(b % source.Batch) * sh * sw * c, dstFrame = (long)b * H * W * c;
            long maskFrame = mask is null || mask.Batch == 0 ? -1 : (long)(b % mask.Batch) * sh * sw;
            for (int row = 0; row < visibleH; row++)
                for (int col = 0; col < visibleW; col++)
                {
                    float m = maskFrame < 0 ? 1f : mask!.Data[maskFrame + row * sw + col];
                    long s = srcFrame + ((long)row * sw + col) * c, d = dstFrame + ((long)(y + row) * W + x + col) * c;
                    for (int ch = 0; ch < c; ch++)
                        result.Data[d + ch] = m * source.Data[s + ch] + (1f - m) * result.Data[d + ch];
                }
        }
        return result;
    }
}

/// <summary>
/// The width x height region whose top-left corner is (x, y). Measured with ComfyUI: x and y are first moved inside the picture (to at most the last
/// column / row), and the region is cut at the picture's edge, so it is never empty.
/// </summary>
[Node(Category = "image/transform", DisplayName = "Crop Image (DEPRECATED)")]
public sealed class ImageCrop
{
    public ImageBatch Execute(ImageBatch image,
        [Input(Min = 1, Max = 16384, Step = 1)] int width = 512,
        [Input(Min = 1, Max = 16384, Step = 1)] int height = 512,
        [Input(Min = 0, Max = 16384, Step = 1)] int x = 0,
        [Input(Min = 0, Max = 16384, Step = 1)] int y = 0)
    {
        int W = image.Width, H = image.Height, c = image.Channels;
        int x0 = Math.Max(0, Math.Min(x, W - 1)), y0 = Math.Max(0, Math.Min(y, H - 1));
        int w = (int)Math.Min((long)x0 + width, W) - x0, h = (int)Math.Min((long)y0 + height, H) - y0;
        w = Math.Max(w, 0); h = Math.Max(h, 0);
        var result = new ImageBatch(image.Batch, h, w, c);
        for (int b = 0; b < image.Batch; b++)
            for (int row = 0; row < h; row++)
                Array.Copy(image.Data, (((long)b * H + y0 + row) * W + x0) * c, result.Data, (((long)b * h + row) * w) * c, (long)w * c);
        return result;
    }
}

/// <summary>
/// <paramref name="length"/> pictures starting at <paramref name="batchIndex"/>. Measured with ComfyUI: a negative index counts from the end, the
/// start is then held to the batch (below 0 gives 0, past the end gives the last picture), and the length is cut at the end of the batch.
/// </summary>
[Node(Category = "image/batch", DisplayName = "Get Image from Batch")]
public sealed class ImageFromBatch
{
    public ImageBatch Execute(ImageBatch image,
        [Input(Min = -16384, Max = 16384)] int batchIndex = 0,
        [Input(Min = 1, Max = 4096)] int length = 1)
    {
        int n = image.Batch;
        if (n == 0) return image;
        int start = Math.Clamp(batchIndex < 0 ? batchIndex + n : batchIndex, 0, n - 1);
        int count = Math.Max(0, Math.Min(length, n - start));
        long frame = (long)image.Height * image.Width * image.Channels;
        var result = new ImageBatch(count, image.Height, image.Width, image.Channels);
        Array.Copy(image.Data, start * frame, result.Data, 0, count * frame);
        return result;
    }
}

/// <summary>
/// 1 - value, unclamped. Measured with ComfyUI: a 4-channel picture keeps its alpha as it is; with any other channel count every channel is inverted
/// (1, 2, 3 and 5 channels alike).
/// </summary>
[Node(Category = "image/color", DisplayName = "Invert Image Colors")]
public sealed class ImageInvert
{
    public ImageBatch Execute(ImageBatch image)
    {
        int c = image.Channels;
        var result = new ImageBatch(image.Batch, image.Height, image.Width, c);
        for (long i = 0; i < result.Data.Length; i++) result.Data[i] = c == 4 && i % 4 == 3 ? image.Data[i] : 1f - image.Data[i];
        return result;
    }
}

/// <summary>
/// Adds borders for outpainting. Measured with ComfyUI:
/// <list type="bullet">
/// <item>The new area is 0.5 in every channel (alpha too); the mask (a batch of one) is 1 there.</item>
/// <item>Inside the old picture the mask is 0, except that with feathering f &gt; 0 and 2f smaller than both the old width and height, each padded side
/// fades in: at distance d from that side the value is ((f - d) / f)² for d &lt; f, where d counts from 0 at the first column / row on the left and top
/// and from 1 at the last one on the right and bottom; where sides meet the larger value wins.</item>
/// </list>
/// </summary>
[Node(Category = "image/transform", DisplayName = "Pad Image for Outpainting", OutputNames = ["IMAGE", "MASK"])]
public sealed class ImagePadForOutpaint
{
    public (ImageBatch Image, MaskBatch Mask) Execute(ImageBatch image,
        [Input(Min = 0, Max = 16384, Step = 8)] int left = 0,
        [Input(Min = 0, Max = 16384, Step = 8)] int top = 0,
        [Input(Min = 0, Max = 16384, Step = 8)] int right = 0,
        [Input(Min = 0, Max = 16384, Step = 8)] int bottom = 0,
        [Input(Min = 0, Max = 16384, Step = 1)] int feathering = 40)
    {
        int W = image.Width, H = image.Height, c = image.Channels;
        int newW = W + left + right, newH = H + top + bottom;
        var padded = new ImageBatch(image.Batch, newH, newW, c);
        Array.Fill(padded.Data, 0.5f);
        for (int b = 0; b < image.Batch; b++)
            for (int row = 0; row < H; row++)
                Array.Copy(image.Data, (((long)b * H + row) * W) * c, padded.Data, (((long)b * newH + top + row) * newW + left) * c, (long)W * c);

        var mask = MaskBatch.Filled(1, newH, newW, 1f);
        bool feather = feathering > 0 && 2L * feathering < W && 2L * feathering < H;
        for (int row = 0; row < H; row++)
            for (int col = 0; col < W; col++)
            {
                float v = 0f;
                if (feather)
                {
                    if (left > 0) v = Math.Max(v, Fade(col, feathering));
                    if (top > 0) v = Math.Max(v, Fade(row, feathering));
                    if (right > 0) v = Math.Max(v, Fade(W - col, feathering));
                    if (bottom > 0) v = Math.Max(v, Fade(H - row, feathering));
                }
                mask.Data[(long)(top + row) * newW + left + col] = v;
            }
        return (padded, mask);
    }

    static float Fade(int distance, int feathering)
    {
        if (distance >= feathering) return 0f;
        double t = (double)(feathering - distance) / feathering;
        return (float)(t * t);
    }
}

/// <summary>
/// Resizes to width x height. Measured with ComfyUI: both 0 returns the picture unchanged; one of them 0 keeps the aspect ratio, that side being
/// max(1, round(other / other side * side)) with Python's rounding (half to even).
/// </summary>
[Node(Category = "image/upscaling", DisplayName = "Upscale Image")]
public sealed class ImageScale
{
    public ImageBatch Execute(ImageBatch image,
        [Input(Options = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"])] string upscaleMethod = "nearest-exact",
        [Input(Min = 0, Max = 16384, Step = 1)] int width = 512,
        [Input(Min = 0, Max = 16384, Step = 1)] int height = 512,
        [Input(Options = ["disabled", "center"])] string crop = "disabled")
    {
        if (width == 0 && height == 0) return image;
        if (width == 0) width = Math.Max(1, (int)Math.Round((double)height / image.Height * image.Width, MidpointRounding.ToEven));
        else if (height == 0) height = Math.Max(1, (int)Math.Round((double)width / image.Width * image.Height, MidpointRounding.ToEven));
        return Pixels.Resize(image, width, height, upscaleMethod, crop);
    }
}

/// <summary>Resizes by a factor: each side becomes round(side * factor) with Python's rounding (measured: 2.5 gives 2, 3.5 gives 4); 0 fails.</summary>
[Node(Category = "image/upscaling", DisplayName = "Upscale Image By")]
public sealed class ImageScaleBy
{
    public ImageBatch Execute(ImageBatch image,
        [Input(Options = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"])] string upscaleMethod = "nearest-exact",
        [Input(Min = 0.01, Max = 8.0, Step = 0.01)] double scaleBy = 1.0)
    {
        int width = (int)Math.Round(image.Width * scaleBy, MidpointRounding.ToEven);
        int height = (int)Math.Round(image.Height * scaleBy, MidpointRounding.ToEven);
        return Pixels.Resize(image, width, height, upscaleMethod);
    }
}

/// <summary>
/// Resizes to about megapixels * 1024 * 1024 pixels keeping the aspect ratio. Measured with ComfyUI: with s = sqrt(total / (width * height)) each side
/// becomes round(side * s / steps) * steps (Python's rounding), which can be 0 and then fails like Upscale Image By.
/// </summary>
[Node(Category = "image/upscaling", DisplayName = "Scale Image to Total Pixels")]
public sealed class ImageScaleToTotalPixels
{
    public ImageBatch Execute(ImageBatch image,
        [Input(Options = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"])] string upscaleMethod = "nearest-exact",
        [Input(Min = 0.01, Max = 16.0, Step = 0.01)] double megapixels = 1.0,
        [Input(Min = 1, Max = 256)] int resolutionSteps = 1)
    {
        double scale = Math.Sqrt(megapixels * 1024 * 1024 / ((double)image.Width * image.Height));
        int width = (int)Math.Round(image.Width * scale / resolutionSteps, MidpointRounding.ToEven) * resolutionSteps;
        int height = (int)Math.Round(image.Height * scale / resolutionSteps, MidpointRounding.ToEven) * resolutionSteps;
        return Pixels.Resize(image, width, height, upscaleMethod);
    }
}

/// <summary>
/// One channel of the picture as a mask. ComfyUI fails with an IndexError when the channel is missing (measured); here a missing alpha counts as opaque
/// (all ones) and any other missing channel fails with the same message.
/// </summary>
[Node(Category = "image/mask", DisplayName = "Convert Image to Mask")]
public sealed class ImageToMask
{
    public MaskBatch Execute(ImageBatch image, [Input(Options = ["red", "green", "blue", "alpha"])] string channel = "red")
    {
        int index = channel switch
        {
            "red" => 0, "green" => 1, "blue" => 2, "alpha" => 3,
            _ => throw new ArgumentException($"Unknown channel '{channel}'.", nameof(channel)),
        };
        int c = image.Channels, pixels = image.Batch * image.Height * image.Width;
        if (index >= c)
        {
            if (index == 3) return MaskBatch.Filled(image.Batch, image.Height, image.Width, 1f);
            throw new ArgumentException($"index {index} is out of bounds for dimension 3 with size {c}");
        }
        var mask = new MaskBatch(image.Batch, image.Height, image.Width);
        for (long p = 0; p < pixels; p++) mask.Data[p] = image.Data[p * c + index];
        return mask;
    }
}

/// <summary>The mask as a grey RGB picture (the value copied to all three channels).</summary>
[Node(Category = "image/mask", DisplayName = "Convert Mask to Image")]
public sealed class MaskToImage
{
    public ImageBatch Execute(MaskBatch mask)
    {
        var image = new ImageBatch(mask.Batch, mask.Height, mask.Width, 3);
        for (long p = 0; p < mask.Data.Length; p++)
        {
            float v = mask.Data[p];
            image.Data[p * 3] = v; image.Data[p * 3 + 1] = v; image.Data[p * 3 + 2] = v;
        }
        return image;
    }
}

/// <summary>The whole batch <paramref name="amount"/> times over, one copy after another (measured: [a, b] x 2 gives a, b, a, b).</summary>
[Node(Category = "image/batch", DisplayName = "Repeat Image Batch")]
public sealed class RepeatImageBatch
{
    public ImageBatch Execute(ImageBatch image, [Input(Min = 1, Max = 4096)] int amount = 1)
    {
        var result = new ImageBatch(image.Batch * amount, image.Height, image.Width, image.Channels);
        for (int i = 0; i < amount; i++) Array.Copy(image.Data, 0, result.Data, (long)i * image.Data.Length, image.Data.Length);
        return result;
    }
}
