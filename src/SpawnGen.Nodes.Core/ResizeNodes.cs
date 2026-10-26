using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// ComfyUI's Resize Image/Mask (nodes_post_processing.ResizeImageMaskNode): one node, nine ways to choose the new size, for a picture or a mask
/// (the output is the same kind as the input). Every size rule is ComfyUI's, rounding included (Python's round is half to even).
/// </summary>
[Node("ResizeImageMaskNode", Category = "image/transform", DisplayName = "Resize Image/Mask",
    Description = "Resize an image or mask using various scaling methods.", OutputNames = ["resized"], OutputMatchTypes = ["input_type"])]
public sealed class ResizeImageMaskNode
{
    const int MaxResolution = 16384;
    static readonly string[] Crops = ["disabled", "center"];

    public static string[] ScaleMethods() => Resample.MethodNames;

    public static DynamicOption[] ResizeTypes()
    {
        var crop = DynamicInput.Combo("crop", Crops, "center", "How to handle aspect ratio mismatch: 'disabled' stretches to fit, 'center' crops to maintain aspect ratio.");
        return
        [
            new("scale dimensions",
                DynamicInput.Int("width", 512, 0, MaxResolution, tooltip: "Target width in pixels. Set to 0 to auto-calculate from height while preserving aspect ratio."),
                DynamicInput.Int("height", 512, 0, MaxResolution, tooltip: "Target height in pixels. Set to 0 to auto-calculate from width while preserving aspect ratio."),
                crop),
            new("scale by multiplier", DynamicInput.Float("multiplier", 1.0, 0.01, 8.0, tooltip: "Scale factor (e.g., 2.0 doubles size, 0.5 halves size).")),
            new("scale longer dimension", DynamicInput.Int("longer_size", 512, 0, MaxResolution, tooltip: "The longer edge will be resized to this value. Aspect ratio is preserved.")),
            new("scale shorter dimension", DynamicInput.Int("shorter_size", 512, 0, MaxResolution, tooltip: "The shorter edge will be resized to this value. Aspect ratio is preserved.")),
            new("scale width", DynamicInput.Int("width", 512, 0, MaxResolution, tooltip: "Target width in pixels. Height auto-adjusts to preserve aspect ratio.")),
            new("scale height", DynamicInput.Int("height", 512, 0, MaxResolution, tooltip: "Target height in pixels. Width auto-adjusts to preserve aspect ratio.")),
            new("scale total pixels", DynamicInput.Float("megapixels", 1.0, 0.01, 16.0, tooltip: "Target total megapixels (e.g., 1.0 ≈ 1024×1024). Aspect ratio is preserved.")),
            new("match size", DynamicInput.Socket("match", "IMAGE,MASK", tooltip: "Resize input to match the dimensions of this reference image or mask."), crop),
            new("scale to multiple", DynamicInput.Int("multiple", 8, 1, MaxResolution, tooltip: "Resize so width and height are divisible by this number. Useful for latent alignment (e.g., 8 or 64).")),
        ];
    }

    public object Execute([Input(Type = "IMAGE,MASK", MatchType = "input_type")] object input,
        [DynamicCombo(nameof(ResizeTypes)), Input(Tooltip = "Select how to resize: by exact dimensions, scale factor, matching another image, etc.")] DynamicChoice resizeType,
        [Input(OptionsProvider = nameof(ScaleMethods), Tooltip = "Interpolation algorithm. 'area' is best for downscaling, 'lanczos' for upscaling, 'nearest-exact' for pixel art.")] string scaleMethod = "area")
    {
        var (w, h) = Size(input);
        var method = Resample.Parse(scaleMethod);
        string crop = resizeType.Get("crop", "center");
        switch (resizeType.Key)
        {
            case "scale by multiplier":
            {
                double m = resizeType.Get("multiplier", 1.0);
                return Scale(input, (int)Math.Round(w * m), (int)Math.Round(h * m), method, false);
            }
            case "scale dimensions": return ScaleDimensions(input, resizeType.Get("width", 512), resizeType.Get("height", 512), method, crop);
            case "scale width": return ScaleDimensions(input, resizeType.Get("width", 512), 0, method, "disabled");
            case "scale height": return ScaleDimensions(input, 0, resizeType.Get("height", 512), method, "disabled");
            case "scale longer dimension":
            {
                int size = resizeType.Get("longer_size", 512);
                var (nw, nh) = h > w ? ((int)Math.Round((double)w / h * size), size) : w > h ? (size, (int)Math.Round((double)h / w * size)) : (size, size);
                return Scale(input, nw, nh, method, false);
            }
            case "scale shorter dimension":
            {
                int size = resizeType.Get("shorter_size", 512);
                var (nw, nh) = h < w ? ((int)Math.Round((double)w / h * size), size) : w < h ? (size, (int)Math.Round((double)h / w * size)) : (size, size);
                return Scale(input, nw, nh, method, false);
            }
            case "scale total pixels":
            {
                long total = (long)(resizeType.Get("megapixels", 1.0) * 1024 * 1024);
                double s = Math.Sqrt((double)total / (w * h));
                return Scale(input, (int)Math.Round(w * s), (int)Math.Round(h * s), method, false);
            }
            case "match size":
            {
                var match = resizeType.Values.GetValueOrDefault("match") ?? throw new ArgumentException("'match size' needs a picture or mask on its match input.");
                var (mw, mh) = Size(match);
                return Scale(input, mw, mh, method, crop == "center");
            }
            case "scale to multiple": return ScaleToMultipleCover(input, resizeType.Get("multiple", 8), method);
            default: throw new ArgumentException($"Unsupported resize type: {resizeType.Key}");
        }
    }

    static object ScaleDimensions(object input, int width, int height, ResampleMethod method, string crop)
    {
        if (width == 0 && height == 0) return input;
        var (w, h) = Size(input);
        if (width == 0) width = Math.Max(1, (int)Math.Round((double)w * height / h));
        else if (height == 0) height = Math.Max(1, (int)Math.Round((double)h * width / w));
        return Scale(input, width, height, method, crop == "center");
    }

    /// <summary>Scales to cover a size whose sides are multiples of <paramref name="multiple"/>, then cuts the middle out (ComfyUI's scale_to_multiple_cover).</summary>
    static object ScaleToMultipleCover(object input, int multiple, ResampleMethod method)
    {
        if (multiple <= 1) return input;
        var (w, h) = Size(input);
        int tw = w / multiple * multiple, th = h / multiple * multiple;
        if (tw == 0 || th == 0 || (tw == w && th == h)) return input;
        double sw = (double)tw / w, sh = (double)th / h;
        int scaledW, scaledH;
        if (sw >= sh) { scaledW = tw; scaledH = Math.Max(th, (int)Math.Ceiling(h * sw)); }
        else { scaledH = th; scaledW = Math.Max(tw, (int)Math.Ceiling(w * sh)); }
        var scaled = Scale(input, scaledW, scaledH, method, false);
        int x0 = (scaledW - tw) / 2, y0 = (scaledH - th) / 2;
        return Cut(scaled, x0, y0, tw, th);
    }

    static (int W, int H) Size(object value) => value switch
    {
        ImageBatch i => (i.Width, i.Height),
        MaskBatch m => (m.Width, m.Height),
        _ => throw new ArgumentException($"Expected a picture or a mask, got {value.GetType().Name}."),
    };

    static object Scale(object input, int width, int height, ResampleMethod method, bool centerCrop)
    {
        if (width <= 0 || height <= 0) throw new ArgumentException($"The new size {width}x{height} is empty.");
        switch (input)
        {
            case ImageBatch image:
                return Pixels.Map(image, f => Resample.CommonUpscale(f, image.Width, image.Height, image.Channels, width, height, method, centerCrop), height, width);
            case MaskBatch mask:
            {
                var result = new MaskBatch(mask.Batch, height, width);
                int n = mask.Width * mask.Height;
                for (int b = 0; b < mask.Batch; b++)
                {
                    var frame = Resample.CommonUpscale(mask.Data.AsSpan(b * n, n).ToArray(), mask.Width, mask.Height, 1, width, height, method, centerCrop);
                    Array.Copy(frame, 0, result.Data, b * width * height, width * height);
                }
                return result;
            }
            default: throw new ArgumentException($"Expected a picture or a mask, got {input.GetType().Name}.");
        }
    }

    static object Cut(object input, int x0, int y0, int w, int h)
    {
        int c = input is ImageBatch img ? img.Channels : 1;
        var (sw, sh) = Size(input);
        int batch = input is ImageBatch ib ? ib.Batch : ((MaskBatch)input).Batch;
        var src = ((HostTensor)input).Data;
        var dst = new float[batch * h * w * c];
        for (int b = 0; b < batch; b++)
            for (int y = 0; y < h; y++)
                Array.Copy(src, ((b * sh + y0 + y) * sw + x0) * c, dst, ((b * h + y) * w) * c, w * c);
        return input is ImageBatch ? new ImageBatch(batch, h, w, c, dst) : new MaskBatch(batch, h, w, dst);
    }
}
