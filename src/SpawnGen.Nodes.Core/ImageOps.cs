using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>Shared helpers for the image nodes: pictures are NHWC floats in 0..1 (<see cref="ImageBatch"/>).</summary>
internal static class Pixels
{
    public static float[] Frame(ImageBatch img, int index)
    {
        int n = img.Height * img.Width * img.Channels;
        var f = new float[n];
        Array.Copy(img.Data, index * n, f, 0, n);
        return f;
    }

    public static ImageBatch From(IReadOnlyList<float[]> frames, int h, int w, int c)
    {
        var batch = new ImageBatch(frames.Count, h, w, c);
        for (int i = 0; i < frames.Count; i++) Array.Copy(frames[i], 0, batch.Data, i * h * w * c, h * w * c);
        return batch;
    }

    public static ImageBatch Map(ImageBatch img, Func<float[], float[]> perFrame, int? newH = null, int? newW = null)
    {
        var frames = Enumerable.Range(0, img.Batch).Select(i => perFrame(Frame(img, i))).ToList();
        return From(frames, newH ?? img.Height, newW ?? img.Width, img.Channels);
    }

    /// <summary>Gaussian blur with ComfyUI's kernel: a (2r+1) square sampled over [-1, 1], so sigma is relative to the radius. Edges reflect.</summary>
    public static float[] Gaussian(float[] src, int w, int h, int c, int radius, float sigma)
    {
        if (radius < 1) return (float[])src.Clone();
        int size = 2 * radius + 1;
        var k1 = new double[size];
        double sum = 0;
        for (int i = 0; i < size; i++)
        {
            double x = size == 1 ? 0 : -1 + 2.0 * i / (size - 1);
            k1[i] = Math.Exp(-(x * x) / (2.0 * sigma * sigma));
            sum += k1[i];
        }
        for (int i = 0; i < size; i++) k1[i] /= sum; // a Gaussian is separable, so two 1D passes equal the 2D kernel

        int Reflect(int i, int n) { if (n == 1) return 0; while (i < 0 || i >= n) i = i < 0 ? -i : 2 * (n - 1) - i; return i; }
        var tmp = new float[src.Length];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                for (int ch = 0; ch < c; ch++)
                {
                    double acc = 0;
                    for (int k = -radius; k <= radius; k++) acc += k1[k + radius] * src[(y * w + Reflect(x + k, w)) * c + ch];
                    tmp[(y * w + x) * c + ch] = (float)acc;
                }
        var dst = new float[src.Length];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                for (int ch = 0; ch < c; ch++)
                {
                    double acc = 0;
                    for (int k = -radius; k <= radius; k++) acc += k1[k + radius] * tmp[(Reflect(y + k, h) * w + x) * c + ch];
                    dst[(y * w + x) * c + ch] = (float)acc;
                }
        return dst;
    }

    /// <summary>The size ComfyUI's scale nodes pick: a 0 for one side keeps the aspect ratio; both 0 keeps the size.</summary>
    public static (int W, int H) TargetSize(int w, int h, int width, int height)
    {
        if (width == 0 && height == 0) return (w, h);
        if (width == 0) width = Math.Max(1, (int)Math.Round((double)w * height / h));
        else if (height == 0) height = Math.Max(1, (int)Math.Round((double)h * width / w));
        return (width, height);
    }
}

[Node(Category = "image/upscaling", DisplayName = "Upscale Image")]
public sealed class ImageScale
{
    public static string[] Methods() => Resample.MethodNames;
    public static string[] Crops() => ["disabled", "center"];

    public ImageBatch Execute(ImageBatch image,
        [Input(OptionsProvider = nameof(Methods))] string upscaleMethod = "nearest-exact",
        [Input(Min = 0, Max = 16384, Step = 1)] int width = 512,
        [Input(Min = 0, Max = 16384, Step = 1)] int height = 512,
        [Input(OptionsProvider = nameof(Crops))] string crop = "disabled")
    {
        var method = Resample.Parse(upscaleMethod);
        var (nw, nh) = Pixels.TargetSize(image.Width, image.Height, width, height);
        if (nw == image.Width && nh == image.Height) return image;

        if (crop == "center")
        {
            // Scale to cover the target, then cut the middle out (the aspect ratio of the picture is kept).
            double scale = Math.Max((double)nw / image.Width, (double)nh / image.Height);
            int sw = Math.Max(nw, (int)Math.Round(image.Width * scale)), sh = Math.Max(nh, (int)Math.Round(image.Height * scale));
            int x0 = (sw - nw) / 2, y0 = (sh - nh) / 2;
            return Pixels.Map(image, f =>
            {
                var big = Resample.Resize(f, image.Width, image.Height, image.Channels, sw, sh, method);
                var cut = new float[nw * nh * image.Channels];
                for (int y = 0; y < nh; y++) Array.Copy(big, ((y0 + y) * sw + x0) * image.Channels, cut, y * nw * image.Channels, nw * image.Channels);
                return cut;
            }, nh, nw);
        }
        return Pixels.Map(image, f => Resample.Resize(f, image.Width, image.Height, image.Channels, nw, nh, method), nh, nw);
    }
}

[Node(Category = "image/upscaling", DisplayName = "Upscale Image By")]
public sealed class ImageScaleBy
{
    public static string[] Methods() => Resample.MethodNames;

    public ImageBatch Execute(ImageBatch image,
        [Input(OptionsProvider = "Methods")] string upscaleMethod = "nearest-exact",
        [Input(Min = 0.01, Max = 8, Step = 0.01)] float scaleBy = 1f)
    {
        int nw = Math.Max(1, (int)Math.Round(image.Width * scaleBy)), nh = Math.Max(1, (int)Math.Round(image.Height * scaleBy));
        return new ImageScale().Execute(image, upscaleMethod, nw, nh);
    }
}

[Node(Category = "image/upscaling", DisplayName = "Scale Image to Total Pixels")]
public sealed class ImageScaleToTotalPixels
{
    public static string[] Methods() => Resample.MethodNames;

    public ImageBatch Execute(ImageBatch image,
        [Input(OptionsProvider = "Methods")] string upscaleMethod = "nearest-exact",
        [Input(Min = 0.01, Max = 16, Step = 0.01)] float megapixels = 1f,
        [Input(Min = 1, Max = 256, Tooltip = "Width and height are rounded to a multiple of this")] int resolutionSteps = 1)
    {
        double total = megapixels * 1024 * 1024, scale = Math.Sqrt(total / (image.Width * image.Height));
        // Python's round() is round-half-to-even, as Math.Round is by default.
        int Size(int side) => Math.Max(resolutionSteps, (int)Math.Round(side * scale / resolutionSteps) * resolutionSteps);
        return new ImageScale().Execute(image, upscaleMethod, Size(image.Width), Size(image.Height));
    }
}

[Node(Category = "image", DisplayName = "Invert Image")]
public sealed class ImageInvert
{
    public ImageBatch Execute(ImageBatch image) => Pixels.Map(image, f => f.Select(v => 1f - v).ToArray());
}

[Node("ImageBatch", Category = "image", DisplayName = "Batch Images")]
public sealed class ImageBatchNode
{
    /// <summary>Stacks two batches; a picture of another size is scaled to the first's size (bicubic, centred crop), like ComfyUI.</summary>
    public ImageBatch Execute(ImageBatch image1, ImageBatch image2)
    {
        if (image2.Height != image1.Height || image2.Width != image1.Width)
            image2 = new ImageScale().Execute(image2, "bicubic", image1.Width, image1.Height, "center");
        if (image2.Channels != image1.Channels) throw new ArgumentException("The two batches have different channel counts.");
        var frames = Enumerable.Range(0, image1.Batch).Select(i => Pixels.Frame(image1, i))
            .Concat(Enumerable.Range(0, image2.Batch).Select(i => Pixels.Frame(image2, i))).ToList();
        return Pixels.From(frames, image1.Height, image1.Width, image1.Channels);
    }
}

[Node(Category = "image/transform", DisplayName = "Image Crop")]
public sealed class ImageCrop
{
    public ImageBatch Execute(ImageBatch image,
        [Input(Min = 1, Max = 16384)] int width = 512, [Input(Min = 1, Max = 16384)] int height = 512,
        [Input(Min = 0, Max = 16384)] int x = 0, [Input(Min = 0, Max = 16384)] int y = 0)
    {
        x = Math.Min(x, image.Width - 1);
        y = Math.Min(y, image.Height - 1);
        int w = Math.Min(width, image.Width - x), h = Math.Min(height, image.Height - y);
        return Pixels.Map(image, f =>
        {
            var cut = new float[w * h * image.Channels];
            for (int row = 0; row < h; row++) Array.Copy(f, ((y + row) * image.Width + x) * image.Channels, cut, row * w * image.Channels, w * image.Channels);
            return cut;
        }, h, w);
    }
}

[Node(Category = "image/batch", DisplayName = "Get Image from Batch")]
public sealed class ImageFromBatch
{
    public ImageBatch Execute(ImageBatch image, [Input(Min = 0, Max = 4095)] int batchIndex = 0, [Input(Min = 1, Max = 4096)] int length = 1)
    {
        int start = Math.Min(batchIndex, image.Batch - 1), count = Math.Min(length, image.Batch - start);
        return Pixels.From(Enumerable.Range(start, count).Select(i => Pixels.Frame(image, i)).ToList(), image.Height, image.Width, image.Channels);
    }
}

[Node(Category = "image/batch", DisplayName = "Repeat Image Batch")]
public sealed class RepeatImageBatch
{
    public ImageBatch Execute(ImageBatch image, [Input(Min = 1, Max = 4096)] int amount = 1) =>
        Pixels.From(Enumerable.Range(0, amount).SelectMany(_ => Enumerable.Range(0, image.Batch).Select(i => Pixels.Frame(image, i))).ToList(),
            image.Height, image.Width, image.Channels);
}

[Node(Category = "image", DisplayName = "Empty Image")]
public sealed class EmptyImage
{
    public ImageBatch Execute([Input(Min = 1, Max = 16384)] int width = 512, [Input(Min = 1, Max = 16384)] int height = 512,
        [Input(Min = 1, Max = 4096)] int batchSize = 1, [Input(Min = 0, Max = 0xFFFFFF)] int color = 0)
    {
        var img = new ImageBatch(batchSize, height, width, 3);
        float r = ((color >> 16) & 0xFF) / 255f, g = ((color >> 8) & 0xFF) / 255f, b = (color & 0xFF) / 255f;
        for (int i = 0; i < img.Data.Length; i += 3) { img.Data[i] = r; img.Data[i + 1] = g; img.Data[i + 2] = b; }
        return img;
    }
}

[Node(Category = "image/postprocessing", DisplayName = "Image Blend")]
public sealed class ImageBlend
{
    public static string[] Modes() => ["normal", "multiply", "screen", "overlay", "soft_light", "difference"];

    public ImageBatch Execute(ImageBatch image1, ImageBatch image2,
        [Input(Min = 0, Max = 1, Step = 0.01)] float blendFactor = 0.5f,
        [Input(OptionsProvider = nameof(Modes))] string blendMode = "normal")
    {
        if (!Modes().Contains(blendMode)) throw new ArgumentException($"Unknown blend mode '{blendMode}'.", nameof(blendMode));
        if (image2.Width != image1.Width || image2.Height != image1.Height)
            image2 = new ImageScale().Execute(image2, "bicubic", image1.Width, image1.Height, "center");
        var result = new ImageBatch(image1.Batch, image1.Height, image1.Width, image1.Channels);
        for (int i = 0; i < result.Data.Length; i++)
        {
            float a = image1.Data[i], b = image2.Data[i % image2.Data.Length];
            float blended = blendMode switch
            {
                "multiply" => a * b,
                "screen" => 1 - (1 - a) * (1 - b),
                "overlay" => a < 0.5f ? 2 * a * b : 1 - 2 * (1 - a) * (1 - b),
                "soft_light" => a > 0.5f ? b + (1 - b) * ((a - 0.5f) / 0.5f) * b : b - (1 - b) * (1 - a / 0.5f) * b,
                "difference" => a - b,
                _ => b,
            };
            result.Data[i] = Math.Clamp(a * (1 - blendFactor) + blended * blendFactor, 0f, 1f);
        }
        return result;
    }
}

[Node(Category = "image/postprocessing", DisplayName = "Image Blur")]
public sealed class ImageBlur
{
    public ImageBatch Execute(ImageBatch image, [Input(Min = 1, Max = 31)] int blurRadius = 1, [Input(Min = 0.1, Max = 10, Step = 0.1)] float sigma = 1f) =>
        Pixels.Map(image, f => Pixels.Gaussian(f, image.Width, image.Height, image.Channels, blurRadius, sigma));
}

[Node(Category = "image/postprocessing", DisplayName = "Image Sharpen")]
public sealed class ImageSharpen
{
    /// <summary>ComfyUI's unsharp mask: <c>image + 10 * alpha * (image - blur)</c>, clamped.</summary>
    public ImageBatch Execute(ImageBatch image, [Input(Min = 1, Max = 31)] int sharpenRadius = 1,
        [Input(Min = 0.1, Max = 10, Step = 0.01)] float sigma = 1f, [Input(Min = 0, Max = 5, Step = 0.01)] float alpha = 1f) =>
        sharpenRadius == 0 ? image : Pixels.Map(image, f =>
        {
            var blur = Pixels.Gaussian(f, image.Width, image.Height, image.Channels, sharpenRadius, sigma);
            return f.Select((v, i) => Math.Clamp(v + 10f * alpha * (v - blur[i]), 0f, 1f)).ToArray();
        });
}

[Node(Category = "image", DisplayName = "Composite Image (masked)")]
public sealed class ImageCompositeMasked
{
    /// <summary>Pastes <paramref name="source"/> onto <paramref name="destination"/> at (x, y), where the mask is 1 (or everywhere with no mask). Parts outside are cut off.</summary>
    public ImageBatch Execute(ImageBatch destination, ImageBatch source, [Input(Min = 0, Max = 16384)] int x = 0, [Input(Min = 0, Max = 16384)] int y = 0,
        bool resizeSource = false, MaskBatch? mask = null)
    {
        if (resizeSource) source = new ImageScale().Execute(source, "bilinear", destination.Width, destination.Height);
        var result = new ImageBatch(destination.Batch, destination.Height, destination.Width, destination.Channels);
        Array.Copy(destination.Data, result.Data, destination.Data.Length);
        int c = destination.Channels;
        for (int b = 0; b < destination.Batch; b++)
        {
            int sb = b % source.Batch;
            float[]? m = null;
            if (mask is not null)
            {
                int mb = b % mask.Batch;
                var one = new float[mask.Height * mask.Width];
                Array.Copy(mask.Data, mb * one.Length, one, 0, one.Length);
                m = mask.Width == source.Width && mask.Height == source.Height ? one : Resample.Resize(one, mask.Width, mask.Height, 1, source.Width, source.Height, ResampleMethod.Bilinear);
            }
            for (int sy = 0; sy < source.Height; sy++)
            {
                int dy = y + sy;
                if (dy >= destination.Height) break;
                for (int sx = 0; sx < source.Width; sx++)
                {
                    int dx = x + sx;
                    if (dx >= destination.Width) break;
                    float k = m is null ? 1f : Math.Clamp(m[sy * source.Width + sx], 0f, 1f);
                    for (int ch = 0; ch < c; ch++)
                    {
                        int di = ((b * destination.Height + dy) * destination.Width + dx) * c + ch;
                        float s = source.Data[((sb * source.Height + sy) * source.Width + sx) * source.Channels + Math.Min(ch, source.Channels - 1)];
                        result.Data[di] = s * k + result.Data[di] * (1 - k);
                    }
                }
            }
        }
        return result;
    }
}

[Node(Category = "mask", DisplayName = "Convert Image to Mask")]
public sealed class ImageToMask
{
    public static string[] Channels() => ["red", "green", "blue", "alpha"];

    /// <summary>One channel of a picture as a mask. Pictures here have no alpha channel unless they came with one: then alpha is fully opaque (1).</summary>
    public MaskBatch Execute(ImageBatch image, [Input(OptionsProvider = nameof(Channels))] string channel = "red")
    {
        int idx = Array.IndexOf(Channels(), channel);
        if (idx < 0) throw new ArgumentException($"Unknown channel '{channel}'.", nameof(channel));
        var mask = new MaskBatch(image.Batch, image.Height, image.Width);
        for (int i = 0; i < mask.Data.Length; i++)
            mask.Data[i] = idx < image.Channels ? image.Data[i * image.Channels + idx] : 1f;
        return mask;
    }
}

[Node(Category = "mask", DisplayName = "Convert Mask to Image")]
public sealed class MaskToImage
{
    public ImageBatch Execute(MaskBatch mask)
    {
        var img = new ImageBatch(mask.Batch, mask.Height, mask.Width, 3);
        for (int i = 0; i < mask.Data.Length; i++) { img.Data[i * 3] = img.Data[i * 3 + 1] = img.Data[i * 3 + 2] = mask.Data[i]; }
        return img;
    }
}

[Node(Category = "image/transform", DisplayName = "Pad Image for Outpainting")]
public sealed class ImagePadForOutpaint
{
    /// <summary>Adds grey (0.5) borders and returns the picture plus a mask that is 1 on the new area, softened towards the old picture by <c>feathering</c> pixels.</summary>
    public (ImageBatch Image, MaskBatch Mask) Execute(ImageBatch image,
        [Input(Min = 0, Max = 16384, Step = 8)] int left = 0, [Input(Min = 0, Max = 16384, Step = 8)] int top = 0,
        [Input(Min = 0, Max = 16384, Step = 8)] int right = 0, [Input(Min = 0, Max = 16384, Step = 8)] int bottom = 0,
        [Input(Min = 0, Max = 16384)] int feathering = 40)
    {
        int nw = image.Width + left + right, nh = image.Height + top + bottom, c = image.Channels;
        var padded = new ImageBatch(image.Batch, nh, nw, c);
        Array.Fill(padded.Data, 0.5f);
        var mask = MaskBatch.Filled(image.Batch, nh, nw, 1f);
        for (int b = 0; b < image.Batch; b++)
            for (int y = 0; y < image.Height; y++)
            {
                Array.Copy(image.Data, ((b * image.Height + y) * image.Width) * c, padded.Data, ((b * nh + top + y) * nw + left) * c, image.Width * c);
                for (int x = 0; x < image.Width; x++)
                {
                    // Distance (in pixels) to the nearest edge that was padded; inside the old picture the mask is 0, ramping up to 1 over `feathering`.
                    float d = float.MaxValue;
                    if (left > 0) d = Math.Min(d, x);
                    if (right > 0) d = Math.Min(d, image.Width - 1 - x);
                    if (top > 0) d = Math.Min(d, y);
                    if (bottom > 0) d = Math.Min(d, image.Height - 1 - y);
                    float m = 0;
                    if (d < feathering && feathering > 0) { float t = (feathering - d) / feathering; m = t * t; }
                    mask.Data[(b * nh + top + y) * nw + left + x] = m;
                }
            }
        return (padded, mask);
    }
}
