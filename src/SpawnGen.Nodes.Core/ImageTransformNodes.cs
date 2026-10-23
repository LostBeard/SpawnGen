using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

[Node(Category = "image/upscaling", DisplayName = "Scale Image to Max Dimension")]
public sealed class ImageScaleToMaxDimension
{
    // ComfyUI's list, "bilinear" twice included: saved workflows store the name, the editor shows the list.
    public static string[] Methods() => ["area", "lanczos", "bilinear", "nearest-exact", "bilinear", "bicubic"];

    /// <summary>Scales so the longer side is <paramref name="largestSize"/> (the shorter one keeps the aspect ratio, Python <c>round</c>); a square becomes largest x largest.</summary>
    public ImageBatch Execute(ImageBatch image, [Input(OptionsProvider = nameof(Methods))] string upscaleMethod = "area",
        [Input(Min = 0, Max = 16384, Step = 1)] int largestSize = 512)
    {
        int height = image.Height, width = image.Width;
        if (height > width) { width = (int)Math.Round((double)width / height * largestSize); height = largestSize; }
        else if (width > height) { height = (int)Math.Round((double)height / width * largestSize); width = largestSize; }
        else height = width = largestSize;
        return CommonUpscale.Resize(image, width, height, upscaleMethod);
    }
}

[Node(Category = "image/transform", DisplayName = "Resize And Pad Image")]
public sealed class ResizeAndPadImage
{
    public static string[] PaddingColors() => ["white", "black"];
    public static string[] Interpolations() => ["area", "bicubic", "nearest-exact", "bilinear", "lanczos"];

    /// <summary>Scales the picture to fit inside the target keeping its aspect ratio (sizes truncated), centred on a white or black canvas of the target size.</summary>
    public ImageBatch Execute(ImageBatch image, [Input(Min = 1, Max = 16384, Step = 1)] int targetWidth = 512, [Input(Min = 1, Max = 16384, Step = 1)] int targetHeight = 512,
        [Input(OptionsProvider = nameof(PaddingColors))] string paddingColor = "white",
        [Input(OptionsProvider = nameof(Interpolations))] string interpolation = "area")
    {
        double scale = Math.Min((double)targetWidth / image.Width, (double)targetHeight / image.Height);
        int newW = (int)(image.Width * scale), newH = (int)(image.Height * scale), c = image.Channels;
        var resized = CommonUpscale.Resize(image, newW, newH, interpolation);
        var result = new ImageBatch(image.Batch, targetHeight, targetWidth, c);
        Array.Fill(result.Data, paddingColor == "black" ? 0f : 1f);
        int y0 = (targetHeight - newH) / 2, x0 = (targetWidth - newW) / 2;
        for (int b = 0; b < image.Batch; b++)
            for (int y = 0; y < newH; y++)
                Array.Copy(resized.Data, ((b * newH + y) * newW) * c, result.Data, ((b * targetHeight + y0 + y) * targetWidth + x0) * c, newW * c);
        return result;
    }
}

[Node(Category = "image/transform", DisplayName = "Stitch Images",
    Description = "Stitches image2 to image1 in the specified direction.\nIf image2 is not provided, returns image1 unchanged.\nOptional spacing can be added between images.")]
public sealed class ImageStitch
{
    public static string[] Directions() => ["right", "down", "left", "up"];
    public static string[] SpacingColors() => ["white", "black", "red", "green", "blue"];

    /// <summary>
    /// ComfyUI's Stitch Images: the shorter batch repeats its last picture; with <paramref name="matchImageSize"/> image2 is scaled (lanczos) to image1's
    /// height (left/right) or width (up/down), else the smaller side is padded (centred); missing channels are filled with 1; an even
    /// <paramref name="spacingWidth"/> strip of the colour goes between them.
    /// </summary>
    public ImageBatch Execute(ImageBatch image1,
        [Input(OptionsProvider = nameof(Directions))] string direction = "right",
        bool matchImageSize = true,
        [Input(Min = 0, Max = 1024, Step = 2)] int spacingWidth = 0,
        [Input(OptionsProvider = nameof(SpacingColors))] string spacingColor = "white",
        ImageBatch? image2 = null)
    {
        if (image2 is null) return image1;
        if (!Directions().Contains(direction)) throw new ArgumentException($"Unknown direction '{direction}'.", nameof(direction));
        float[] colour = spacingColor switch
        {
            "white" => [1f], "black" => [0f], "red" => [1f, 0f, 0f], "green" => [0f, 1f, 0f], "blue" => [0f, 0f, 1f],
            _ => throw new ArgumentException($"Unknown spacing color '{spacingColor}'.", nameof(spacingColor)),
        };
        bool horizontal = direction is "left" or "right";

        int batch = Math.Max(image1.Batch, image2.Batch);
        image1 = RepeatLast(image1, batch);
        image2 = RepeatLast(image2, batch);

        if (matchImageSize)
        {
            double aspect = (double)image2.Width / image2.Height;
            int tw, th;
            if (horizontal) { th = image1.Height; tw = (int)(image1.Height * aspect); }
            else { tw = image1.Width; th = (int)(image1.Width / aspect); }
            image2 = CommonUpscale.Resize(image2, tw, th, "lanczos");
        }
        else
        {
            // Pad the other side to match, half before and the rest after, with white/black (0 for the coloured choices).
            float pad = colour.Length == 1 ? colour[0] : 0f;
            if (horizontal && image1.Height != image2.Height)
            {
                int target = Math.Max(image1.Height, image2.Height);
                image1 = Pad(image1, 0, target - image1.Height, pad);
                image2 = Pad(image2, 0, target - image2.Height, pad);
            }
            else if (!horizontal && image1.Width != image2.Width)
            {
                int target = Math.Max(image1.Width, image2.Width);
                image1 = Pad(image1, target - image1.Width, 0, pad);
                image2 = Pad(image2, target - image2.Width, 0, pad);
            }
        }

        int channels = Math.Max(image1.Channels, image2.Channels);
        image1 = WithChannels(image1, channels);
        image2 = WithChannels(image2, channels);

        var parts = direction is "left" or "up" ? new List<ImageBatch> { image2, image1 } : [image1, image2];
        if (spacingWidth > 0)
        {
            spacingWidth += spacingWidth % 2;
            var strip = horizontal
                ? new ImageBatch(batch, Math.Max(image1.Height, image2.Height), spacingWidth, channels)
                : new ImageBatch(batch, spacingWidth, Math.Max(image1.Width, image2.Width), channels);
            for (int p = 0; p < strip.Data.Length; p += channels)
            {
                if (colour.Length == 3) for (int k = 0; k < Math.Min(3, channels); k++) strip.Data[p + k] = colour[k];
                else for (int k = 0; k < Math.Min(3, channels); k++) strip.Data[p + k] = colour[0];
                if (channels == 4) strip.Data[p + 3] = 1f;
            }
            parts.Insert(1, strip);
        }
        return horizontal ? ConcatWidth(parts) : ConcatHeight(parts);
    }

    static ImageBatch RepeatLast(ImageBatch img, int batch)
    {
        if (img.Batch >= batch) return img;
        int n = img.Height * img.Width * img.Channels;
        var r = new ImageBatch(batch, img.Height, img.Width, img.Channels);
        Array.Copy(img.Data, r.Data, img.Data.Length);
        for (int b = img.Batch; b < batch; b++) Array.Copy(img.Data, (img.Batch - 1) * n, r.Data, b * n, n);
        return r;
    }

    /// <summary>Adds <paramref name="padW"/> columns (half left, the rest right) and <paramref name="padH"/> rows (half on top) of <paramref name="value"/> in every channel.</summary>
    static ImageBatch Pad(ImageBatch img, int padW, int padH, float value)
    {
        if (padW == 0 && padH == 0) return img;
        int c = img.Channels, nw = img.Width + padW, nh = img.Height + padH, left = padW / 2, top = padH / 2;
        var r = new ImageBatch(img.Batch, nh, nw, c);
        Array.Fill(r.Data, value);
        for (int b = 0; b < img.Batch; b++)
            for (int y = 0; y < img.Height; y++)
                Array.Copy(img.Data, ((b * img.Height + y) * img.Width) * c, r.Data, ((b * nh + top + y) * nw + left) * c, img.Width * c);
        return r;
    }

    static ImageBatch WithChannels(ImageBatch img, int channels)
    {
        if (img.Channels == channels) return img;
        var r = new ImageBatch(img.Batch, img.Height, img.Width, channels);
        for (int p = 0; p < img.Batch * img.Height * img.Width; p++)
            for (int k = 0; k < channels; k++) r.Data[p * channels + k] = k < img.Channels ? img.Data[p * img.Channels + k] : 1f;
        return r;
    }

    static ImageBatch ConcatWidth(List<ImageBatch> parts)
    {
        int h = parts[0].Height, c = parts[0].Channels, batch = parts[0].Batch;
        if (parts.Any(p => p.Height != h)) throw new ArgumentException("The pictures have different heights; turn on match_image_size or use pictures of the same height.");
        int w = parts.Sum(p => p.Width);
        var r = new ImageBatch(batch, h, w, c);
        for (int b = 0; b < batch; b++)
            for (int y = 0; y < h; y++)
            {
                int x0 = 0;
                foreach (var p in parts)
                {
                    Array.Copy(p.Data, ((b * h + y) * p.Width) * c, r.Data, ((b * h + y) * w + x0) * c, p.Width * c);
                    x0 += p.Width;
                }
            }
        return r;
    }

    static ImageBatch ConcatHeight(List<ImageBatch> parts)
    {
        int w = parts[0].Width, c = parts[0].Channels, batch = parts[0].Batch;
        if (parts.Any(p => p.Width != w)) throw new ArgumentException("The pictures have different widths; turn on match_image_size or use pictures of the same width.");
        int h = parts.Sum(p => p.Height), row = w * c;
        var r = new ImageBatch(batch, h, w, c);
        for (int b = 0; b < batch; b++)
        {
            int y0 = 0;
            foreach (var p in parts)
            {
                Array.Copy(p.Data, b * p.Height * row, r.Data, (b * h + y0) * row, p.Height * row);
                y0 += p.Height;
            }
        }
        return r;
    }
}

[Node(Category = "image/transform", DisplayName = "Crop Image to Mask", OutputNames = ["images"])]
public sealed class ImageCropToMask
{
    /// <summary>
    /// ComfyUI's Crop Image to Mask (the TRELLIS.2 preprocessing): the box around the mask (alpha above 0.8) grown to the output aspect ratio
    /// and by <paramref name="padFactor"/>, cut out (zero outside the picture), the subject composited over <paramref name="background"/> and
    /// scaled (lanczos) to width x height. An inverted mask (border set, centre clear) is turned around first; <paramref name="growMask"/> dilates
    /// (or erodes, negative) the mask, values under 0.05 are dropped, and a picture larger than the output is scaled down first.
    /// </summary>
    public ImageBatch Execute(ImageBatch images, MaskBatch masks,
        [Input(Min = 64, Max = 4096, Step = 8, Tooltip = "Output width in pixels.")] int width = 1024,
        [Input(Min = 64, Max = 4096, Step = 8, Tooltip = "Output height in pixels.")] int height = 1024,
        [Input(Min = 1.0, Max = 2.0, Step = 0.01, Tooltip = "Extra margin around the mask bounding box as a multiplier.")] double padFactor = 1.0,
        [Input(Min = -32, Max = 32, Step = 1, Tooltip = "Grow or shrink the mask by this many pixels before cropping.")] int growMask = 0,
        [Input(Color = true, Tooltip = "Background color behind the masked subject.")] string background = "#000000")
    {
        var hex = background.TrimStart('#');
        float[] bg = hex.Length == 6
            ? [(float)(PyInt.Parse(hex[0..2], 16) / 255.0), (float)(PyInt.Parse(hex[2..4], 16) / 255.0), (float)(PyInt.Parse(hex[4..6], 16) / 255.0)]
            : [0f, 0f, 0f];
        int batch = images.Batch, h = images.Height, w = images.Width;
        if (masks.Batch != 1 && masks.Batch != batch) throw new ArgumentException($"Mask batch {masks.Batch} does not match image batch {batch}");
        if (masks.Height != h || masks.Width != w) masks = CommonUpscale.Resize(masks, w, h, "bilinear");

        var result = new ImageBatch(batch, height, width, 3);
        for (int b = 0; b < batch; b++)
        {
            var img = new float[h * w * 3];
            for (int p = 0; p < h * w; p++)
                for (int k = 0; k < 3; k++) img[p * 3 + k] = Math.Clamp(images.Data[(b * h * w + p) * images.Channels + Math.Min(k, images.Channels - 1)], 0f, 1f);
            var mask = new float[h * w];
            int mb = masks.Batch == 1 ? 0 : b;
            for (int p = 0; p < h * w; p++) mask[p] = Math.Clamp(masks.Data[mb * h * w + p], 0f, 1f);

            var (composite, cw, ch) = Crop(img, mask, w, h, Math.Max(width, height), padFactor, growMask, bg, (double)width / height);
            var scaled = PilLanczos.Resize(composite, cw, ch, 3, width, height);
            Array.Copy(scaled, 0, result.Data, b * width * height * 3, scaled.Length);
        }
        return result;
    }

    /// <summary>ComfyUI's <c>_crop_image_with_mask</c> with its default threshold 0.05: the composited crop and its size.</summary>
    static (float[] Image, int W, int H) Crop(float[] img, float[] mask, int W, int H, int maxSize, double padFactor, int offset, float[] bg, double aspectRatio)
    {
        // An inverted mask is detected only when border and centre disagree (border pixels as ComfyUI lists them: rows then columns, corners twice).
        double border = 0;
        for (int x = 0; x < W; x++) border += mask[x] + mask[(H - 1) * W + x];
        for (int y = 0; y < H; y++) border += mask[y * W] + mask[y * W + W - 1];
        border /= 2 * W + 2 * H;
        double centre = 0;
        int cy0 = H / 4, cy1 = H - H / 4, cx0 = W / 4, cx1 = W - W / 4, count = 0;
        for (int y = cy0; y < cy1; y++) for (int x = cx0; x < cx1; x++) { centre += mask[y * W + x]; count++; }
        centre = count == 0 ? double.NaN : centre / count;
        if (border > 0.5 && centre < 0.5) for (int i = 0; i < mask.Length; i++) mask[i] = 1f - mask[i];

        if (offset != 0) mask = Grow(mask, W, H, offset);
        for (int i = 0; i < mask.Length; i++) if (mask[i] < 0.05f) mask[i] = 0f;

        if (Math.Max(H, W) > maxSize)
        {
            double scale = (double)maxSize / Math.Max(H, W);
            int nw = (int)(W * scale), nh = (int)(H * scale);
            img = PilLanczos.Resize(img, W, H, 3, nw, nh);
            mask = PilLanczos.Resize(mask, W, H, 1, nw, nh);
            (W, H) = (nw, nh);
        }

        // Foreground: alpha as 8 bits (truncated) above 204; if there is none, the inverted alpha is tried (and the mask turned around).
        var alpha = new int[W * H];
        for (int i = 0; i < alpha.Length; i++) alpha[i] = (byte)(Math.Clamp(mask[i], 0f, 1f) * 255f);
        Func<int, bool> fg = i => alpha[i] > 204;
        bool any = Enumerable.Range(0, W * H).Any(fg);
        if (!any && Enumerable.Range(0, W * H).Any(i => 255 - alpha[i] > 204))
        {
            for (int i = 0; i < mask.Length; i++) mask[i] = 1f - mask[i];
            fg = i => 255 - alpha[i] > 204;
            any = true;
        }

        int x1, y1, x2, y2;
        if (any)
        {
            int xMin = int.MaxValue, yMin = int.MaxValue, xMax = int.MinValue, yMax = int.MinValue;
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    if (fg(y * W + x)) { xMin = Math.Min(xMin, x); xMax = Math.Max(xMax, x); yMin = Math.Min(yMin, y); yMax = Math.Max(yMax, y); }
            double centerY = (yMin + yMax) / 2.0, centerX = (xMin + xMax) / 2.0;
            int bw = xMax - xMin + 1, bh = yMax - yMin + 1, cropW, cropH;
            if (bw / (double)Math.Max(bh, 1) >= aspectRatio) { cropW = (int)(bw * padFactor); cropH = (int)(bw / aspectRatio * padFactor); }
            else { cropH = (int)(bh * padFactor); cropW = (int)(bh * aspectRatio * padFactor); }
            int halfW = (int)Math.Ceiling(cropW / 2.0), halfH = (int)Math.Ceiling(cropH / 2.0);
            x1 = (int)(centerX - halfW); y1 = (int)(centerY - halfH);
            x2 = x1 + 2 * halfW; y2 = y1 + 2 * halfH;
        }
        else (x1, y1, x2, y2) = (0, 0, W, H);

        // Cut the box out; outside the picture the image and the mask are 0 (Pillow crop semantics), so the background shows there.
        int cw = x2 - x1, ch = y2 - y1;
        var outImg = new float[cw * ch * 3];
        for (int y = 0; y < ch; y++)
            for (int x = 0; x < cw; x++)
            {
                int sx = x1 + x, sy = y1 + y;
                bool inside = sx >= 0 && sx < W && sy >= 0 && sy < H;
                float m = inside ? mask[sy * W + sx] : 0f;
                for (int k = 0; k < 3; k++)
                {
                    float v = inside ? img[(sy * W + sx) * 3 + k] : 0f;
                    outImg[(y * cw + x) * 3 + k] = Math.Clamp(v * m + bg[k] * (1f - m), 0f, 1f);
                }
            }
        return (outImg, cw, ch);
    }

    /// <summary>max_pool2d with a (2r+1) window (outside ignored): grows the mask; a negative offset shrinks it (the same on 1 - mask).</summary>
    static float[] Grow(float[] mask, int w, int h, int offset)
    {
        int r = Math.Abs(offset);
        bool shrink = offset < 0;
        float V(int i) => shrink ? 1f - mask[i] : mask[i];
        var tmp = new float[mask.Length];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                float v = float.NegativeInfinity;
                for (int k = Math.Max(0, x - r); k <= Math.Min(w - 1, x + r); k++) v = Math.Max(v, V(y * w + k));
                tmp[y * w + x] = v;
            }
        var dst = new float[mask.Length];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                float v = float.NegativeInfinity;
                for (int k = Math.Max(0, y - r); k <= Math.Min(h - 1, y + r); k++) v = Math.Max(v, tmp[k * w + x]);
                dst[y * w + x] = shrink ? 1f - v : v;
            }
        return dst;
    }
}
