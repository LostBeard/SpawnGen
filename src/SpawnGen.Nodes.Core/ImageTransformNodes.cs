using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries, SpawnGen's tests and the fixtures made by running ComfyUI
// (Fixtures/comfy/comfy_image_nodes_main.json.gz: 30 Scale to Max Dimension, 40 Resize And Pad, 96 Stitch cases, all reproduced), plus probes of
// ComfyUI (2026-11-08) for RGBA padding and zero sizes.

/// <summary>
/// Scales every picture so its longer side is <c>largest_size</c>: a taller picture gets height = largest and width = round(w / h * largest), otherwise
/// width = largest and height = round(h / w * largest) (Python round). A size of 0 fails as the resize method does (area gives an empty picture).
/// </summary>
[Node(Category = "image/upscaling", DisplayName = "Scale Image to Max Dimension")]
public sealed class ImageScaleToMaxDimension
{
    public ImageBatch Execute(ImageBatch image,
        [Input(Options = ["area", "lanczos", "bilinear", "nearest-exact", "bilinear", "bicubic"])] string upscaleMethod = "area",
        [Input(Min = 0, Max = 16384, Step = 1)] int largestSize = 512)
    {
        int h = image.Height, w = image.Width, nw, nh;
        if (h > w)
        {
            nh = largestSize;
            nw = (int)Math.Round((double)w / h * largestSize, MidpointRounding.ToEven);
        }
        else
        {
            nw = largestSize;
            nh = (int)Math.Round((double)h / w * largestSize, MidpointRounding.ToEven);
        }
        return Pixels.Resize(image, nw, nh, upscaleMethod);
    }
}

/// <summary>
/// Fits every picture inside target_width x target_height without changing its aspect: scale s = min(tw / w, th / h), size int(w s) x int(h s), placed at
/// ((tw - nw) / 2, (th - nh) / 2) (integer halves) on a canvas of padding_color, 1.0 or 0.0 in every channel (alpha too).
/// </summary>
[Node(Category = "image/transform", DisplayName = "Resize And Pad Image")]
public sealed class ResizeAndPadImage
{
    public ImageBatch Execute(ImageBatch image,
        [Input(Min = 1, Max = 16384, Step = 1)] int targetWidth = 512,
        [Input(Min = 1, Max = 16384, Step = 1)] int targetHeight = 512,
        [Input(Options = ["white", "black"])] string paddingColor = "white",
        [Input(Options = ["area", "bicubic", "nearest-exact", "bilinear", "lanczos"])] string interpolation = "area")
    {
        float fill = paddingColor switch
        {
            "white" => 1f,
            "black" => 0f,
            _ => throw new ArgumentException($"Unknown padding color '{paddingColor}'.", nameof(paddingColor)),
        };
        double scale = Math.Min((double)targetWidth / image.Width, (double)targetHeight / image.Height);
        int nw = (int)(image.Width * scale), nh = (int)(image.Height * scale);
        var resized = Pixels.Resize(image, nw, nh, interpolation);
        int c = image.Channels, x0 = (targetWidth - nw) / 2, y0 = (targetHeight - nh) / 2;
        var result = new ImageBatch(image.Batch, targetHeight, targetWidth, c);
        Array.Fill(result.Data, fill);
        for (int b = 0; b < image.Batch; b++)
            for (int y = 0; y < resized.Height; y++)
                Array.Copy(resized.Data, (((long)b * resized.Height + y) * resized.Width) * c, result.Data,
                    (((long)b * targetHeight + y + y0) * targetWidth + x0) * c, (long)resized.Width * c);
        return result;
    }
}

/// <summary>
/// Joins image2 to image1 on the given side, as measured with ComfyUI:
/// <list type="number">
/// <item>The shorter batch repeats its last picture.</item>
/// <item>match_image_size: image2 is scaled with lanczos to image1's height (left / right; width int(w2 h1 / h2)) or width (up / down; height
/// int(h2 w1 / w2)). Otherwise the smaller picture is padded across the joining direction, half before (rounded down) and the rest after, with 1.0
/// in every channel when spacing_color is white and 0.0 otherwise.</item>
/// <item>Then both get the larger channel count, missing channels set to 1.0.</item>
/// <item>A spacing strip of spacing_width rounded up to even pixels, in spacing_color (alpha 1), goes between them.</item>
/// </list>
/// Without image2, image1 is returned as it is.
/// </summary>
[Node(Category = "image/transform", DisplayName = "Stitch Images",
    Description = "Stitches image2 to image1 in the specified direction.\nIf image2 is not provided, returns image1 unchanged.\nOptional spacing can be added between images.")]
public sealed class ImageStitch
{
    public ImageBatch Execute(ImageBatch image1,
        [Input(Options = ["right", "down", "left", "up"])] string direction = "right",
        bool matchImageSize = true,
        [Input(Min = 0, Max = 1024, Step = 2)] int spacingWidth = 0,
        [Input(Options = ["white", "black", "red", "green", "blue"])] string spacingColor = "white",
        [Input(Optional = true)] ImageBatch? image2 = null)
    {
        if (image2 is null) return image1;
        bool sideways = direction switch
        {
            "left" or "right" => true,
            "up" or "down" => false,
            _ => throw new ArgumentException($"Unknown direction '{direction}'.", nameof(direction)),
        };
        float[] rgb = spacingColor switch
        {
            "white" => [1f, 1f, 1f],
            "black" => [0f, 0f, 0f],
            "red" => [1f, 0f, 0f],
            "green" => [0f, 1f, 0f],
            "blue" => [0f, 0f, 1f],
            _ => throw new ArgumentException($"Unknown spacing color '{spacingColor}'.", nameof(spacingColor)),
        };
        int batch = Math.Max(image1.Batch, image2.Batch);
        var a = RepeatLast(image1, batch);
        var b = RepeatLast(image2, batch);
        if (matchImageSize)
        {
            if (sideways) b = CommonUpscale.Resize(b, (int)((double)b.Width * a.Height / b.Height), a.Height, "lanczos");
            else b = CommonUpscale.Resize(b, a.Width, (int)((double)b.Height * a.Width / b.Width), "lanczos");
        }
        else
        {
            float fill = spacingColor == "white" ? 1f : 0f;
            if (sideways)
            {
                int height = Math.Max(a.Height, b.Height);
                a = Pad(a, height, a.Width, fill);
                b = Pad(b, height, b.Width, fill);
            }
            else
            {
                int width = Math.Max(a.Width, b.Width);
                a = Pad(a, a.Height, width, fill);
                b = Pad(b, b.Height, width, fill);
            }
        }
        int channels = Math.Max(a.Channels, b.Channels);
        a = WithChannels(a, channels);
        b = WithChannels(b, channels);
        var parts = new List<ImageBatch> { a };
        if (spacingWidth > 0)
        {
            int sp = spacingWidth + spacingWidth % 2;
            var strip = sideways ? new ImageBatch(batch, a.Height, sp, channels) : new ImageBatch(batch, sp, a.Width, channels);
            for (long i = 0; i < strip.Data.Length; i++)
            {
                int ch = (int)(i % channels);
                strip.Data[i] = ch < 3 ? rgb[ch] : 1f;
            }
            parts.Add(strip);
        }
        parts.Add(b);
        if (direction is "left" or "up") parts.Reverse();
        return sideways ? JoinColumns(parts) : JoinRows(parts);
    }

    static ImageBatch RepeatLast(ImageBatch image, int batch)
    {
        if (image.Batch >= batch) return image;
        int n = image.Height * image.Width * image.Channels;
        var result = new ImageBatch(batch, image.Height, image.Width, image.Channels);
        for (int i = 0; i < batch; i++) Array.Copy(image.Data, (long)Math.Min(i, image.Batch - 1) * n, result.Data, (long)i * n, n);
        return result;
    }

    /// <summary>Pads to <paramref name="height"/> x <paramref name="width"/>: half the difference before (rounded down), the rest after.</summary>
    static ImageBatch Pad(ImageBatch image, int height, int width, float fill)
    {
        if (image.Height == height && image.Width == width) return image;
        int c = image.Channels, top = (height - image.Height) / 2, left = (width - image.Width) / 2;
        var result = new ImageBatch(image.Batch, height, width, c);
        Array.Fill(result.Data, fill);
        for (int b = 0; b < image.Batch; b++)
            for (int y = 0; y < image.Height; y++)
                Array.Copy(image.Data, (((long)b * image.Height + y) * image.Width) * c, result.Data, (((long)b * height + y + top) * width + left) * c,
                    (long)image.Width * c);
        return result;
    }

    /// <summary>Adds channels of 1.0 up to <paramref name="channels"/>.</summary>
    static ImageBatch WithChannels(ImageBatch image, int channels)
    {
        int c = image.Channels;
        if (c == channels) return image;
        long pixels = (long)image.Batch * image.Height * image.Width;
        var result = new ImageBatch(image.Batch, image.Height, image.Width, channels);
        for (long p = 0; p < pixels; p++)
        {
            Array.Copy(image.Data, p * c, result.Data, p * channels, c);
            for (int k = c; k < channels; k++) result.Data[p * channels + k] = 1f;
        }
        return result;
    }

    static ImageBatch JoinColumns(List<ImageBatch> parts)
    {
        int batch = parts[0].Batch, h = parts[0].Height, c = parts[0].Channels, width = parts.Sum(p => p.Width);
        var result = new ImageBatch(batch, h, width, c);
        for (int b = 0; b < batch; b++)
            for (int y = 0; y < h; y++)
            {
                long at = (((long)b * h + y) * width) * c;
                foreach (var p in parts)
                {
                    Array.Copy(p.Data, (((long)b * h + y) * p.Width) * c, result.Data, at, (long)p.Width * c);
                    at += (long)p.Width * c;
                }
            }
        return result;
    }

    static ImageBatch JoinRows(List<ImageBatch> parts)
    {
        int batch = parts[0].Batch, w = parts[0].Width, c = parts[0].Channels, height = parts.Sum(p => p.Height);
        var result = new ImageBatch(batch, height, w, c);
        for (int b = 0; b < batch; b++)
        {
            long at = (long)b * height * w * c;
            foreach (var p in parts)
            {
                long n = (long)p.Height * w * c;
                Array.Copy(p.Data, b * n, result.Data, at, n);
                at += n;
            }
        }
        return result;
    }
}
