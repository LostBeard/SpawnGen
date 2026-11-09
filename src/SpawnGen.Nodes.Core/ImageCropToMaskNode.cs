using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// NOT YET REWRITTEN CLEAN-ROOM (docs/CLEANROOM.md): moved unchanged from the old ImageTransformNodes.cs during batch J; still on the rewrite list.

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
