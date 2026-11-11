using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entry, SpawnGen's tests and black-box measurements of ComfyUI (2026-11-08:
// output sizes of 360 random cases over every resize type, pixel comparisons for crops, multiples, matching and masks, zero-size probes).

/// <summary>
/// Resizes an image or a mask (it stays what it was). Sizes, as measured with ComfyUI (round = Python's, half to even):
/// <list type="bullet">
/// <item>scale dimensions: width x height; a 0 side follows the aspect, round(h * width / w) or round(w * height / h), at least 1; both 0 = unchanged.
/// crop "center" cuts to the target aspect first.</item>
/// <item>scale by multiplier: round(w m) x round(h m). scale longer / shorter dimension: that side becomes the size, the other round(side * size /
/// this side) (no minimum: a 0 fails).</item>
/// <item>scale width / scale height: the other side round(...), at least 1. scale total pixels: s = sqrt(int(mp * 1024²) / (w h)), round(w s) x
/// round(h s).</item>
/// <item>match size: the size of the given image or mask, with its crop.</item>
/// <item>scale to multiple: both sides rounded down to the multiple (unchanged when either is smaller than it); the picture is scaled by
/// max(W / w, H / h) to ceil(w s) x ceil(h s) and the middle W x H is kept.</item>
/// </list>
/// </summary>
[Node("ResizeImageMaskNode", Category = "image/transform", DisplayName = "Resize Image/Mask", OutputNames = ["resized"], OutputMatchTypes = ["input_type"],
    Description = "Resize an image or mask using various scaling methods.")]
public sealed class ResizeImageMaskNode
{
    const string CropTip = "How to handle aspect ratio mismatch: 'disabled' stretches to fit, 'center' crops to maintain aspect ratio.";

    public static IEnumerable<DynamicOption> ResizeTypes() =>
    [
        new("scale dimensions",
            DynamicInput.Int("width", 512, 0, 16384, tooltip: "Target width in pixels. Set to 0 to auto-calculate from height while preserving aspect ratio."),
            DynamicInput.Int("height", 512, 0, 16384, tooltip: "Target height in pixels. Set to 0 to auto-calculate from width while preserving aspect ratio."),
            DynamicInput.Combo("crop", ["disabled", "center"], "center", CropTip)),
        new("scale by multiplier",
            DynamicInput.Float("multiplier", 1.0, 0.01, 8.0, 0.01, "Scale factor (e.g., 2.0 doubles size, 0.5 halves size).")),
        new("scale longer dimension",
            DynamicInput.Int("longer_size", 512, 0, 16384, tooltip: "The longer edge will be resized to this value. Aspect ratio is preserved.")),
        new("scale shorter dimension",
            DynamicInput.Int("shorter_size", 512, 0, 16384, tooltip: "The shorter edge will be resized to this value. Aspect ratio is preserved.")),
        new("scale width",
            DynamicInput.Int("width", 512, 0, 16384, tooltip: "Target width in pixels. Height auto-adjusts to preserve aspect ratio.")),
        new("scale height",
            DynamicInput.Int("height", 512, 0, 16384, tooltip: "Target height in pixels. Width auto-adjusts to preserve aspect ratio.")),
        new("scale total pixels",
            DynamicInput.Float("megapixels", 1.0, 0.01, 16.0, 0.01, "Target total megapixels (e.g., 1.0 ≈ 1024×1024). Aspect ratio is preserved.")),
        new("match size",
            DynamicInput.Socket("match", "IMAGE,MASK", tooltip: "Resize input to match the dimensions of this reference image or mask."),
            DynamicInput.Combo("crop", ["disabled", "center"], "center", CropTip)),
        new("scale to multiple",
            DynamicInput.Int("multiple", 8, 1, 16384, tooltip: "Resize so width and height are divisible by this number. Useful for latent alignment (e.g., 8 or 64).")),
    ];

    public object Execute(
        [Input(MatchType = "input_type", Type = "IMAGE,MASK")] object input,
        [DynamicCombo(nameof(ResizeTypes)), Input(Tooltip = "Select how to resize: by exact dimensions, scale factor, matching another image, etc.")]
        DynamicChoice resizeType,
        [Input(Options = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
            Tooltip = "Interpolation algorithm. 'area' is best for downscaling, 'lanczos' for upscaling, 'nearest-exact' for pixel art.")]
        string scaleMethod = "area")
    {
        var (w, h) = input switch
        {
            ImageBatch i => (i.Width, i.Height),
            MaskBatch m => (m.Width, m.Height),
            _ => throw new ArgumentException("Resize Image/Mask takes an IMAGE or a MASK.", nameof(input)),
        };
        static int Round(double v) => (int)Math.Round(v, MidpointRounding.ToEven);
        string crop = "disabled";
        int nw, nh;
        switch (resizeType.Key)
        {
            case "scale dimensions":
            {
                nw = (int)resizeType.Get<long>("width", 512);
                nh = (int)resizeType.Get<long>("height", 512);
                crop = resizeType.Get("crop", "center");
                if (nw == 0 && nh == 0) return input;
                if (nw == 0) nw = Math.Max(1, Round((double)w * nh / h));
                else if (nh == 0) nh = Math.Max(1, Round((double)h * nw / w));
                break;
            }
            case "scale by multiplier":
            {
                double m = resizeType.Get("multiplier", 1.0);
                (nw, nh) = (Round(w * m), Round(h * m));
                break;
            }
            case "scale longer dimension":
            case "scale shorter dimension":
            {
                bool longer = resizeType.Key == "scale longer dimension";
                int size = (int)resizeType.Get<long>(longer ? "longer_size" : "shorter_size", 512);
                if (longer == w >= h) (nw, nh) = (size, Round((double)h * size / w));
                else (nw, nh) = (Round((double)w * size / h), size);
                break;
            }
            case "scale width":
                nw = (int)resizeType.Get<long>("width", 512);
                nh = Math.Max(1, Round((double)h * nw / w));
                break;
            case "scale height":
                nh = (int)resizeType.Get<long>("height", 512);
                nw = Math.Max(1, Round((double)w * nh / h));
                break;
            case "scale total pixels":
            {
                double s = Math.Sqrt((long)(resizeType.Get("megapixels", 1.0) * 1024 * 1024) / ((double)w * h));
                (nw, nh) = (Round(w * s), Round(h * s));
                break;
            }
            case "match size":
            {
                (nw, nh) = resizeType.Get<object?>("match", null) switch
                {
                    ImageBatch i => (i.Width, i.Height),
                    MaskBatch m => (m.Width, m.Height),
                    _ => throw new ArgumentException("match size needs an IMAGE or MASK to match."),
                };
                crop = resizeType.Get("crop", "center");
                break;
            }
            case "scale to multiple":
            {
                int multiple = (int)resizeType.Get<long>("multiple", 8);
                if (multiple < 1) throw new ArgumentOutOfRangeException(nameof(resizeType), "multiple must be at least 1.");
                if (w < multiple || h < multiple) return input;
                int tw = w / multiple * multiple, th = h / multiple * multiple;
                double s = Math.Max((double)tw / w, (double)th / h);
                int cw = (int)Math.Ceiling(w * s), ch = (int)Math.Ceiling(h * s);
                return Apply(input, f => CutMiddle(CommonUpscale.Resize(f.Data, w, h, f.Channels, cw, ch, scaleMethod), cw, ch, f.Channels, tw, th), tw, th);
            }
            default:
                throw new ArgumentException($"Unknown resize type '{resizeType.Key}'.", nameof(resizeType));
        }
        return Apply(input, f => CommonUpscale.Resize(f.Data, w, h, f.Channels, nw, nh, scaleMethod, crop), nw, nh);
    }

    /// <summary>One frame: its values and channel count.</summary>
    readonly record struct Frame(float[] Data, int Channels);

    /// <summary>Runs <paramref name="resize"/> on every picture or mask (a mask as one channel) and joins the results.</summary>
    static object Apply(object input, Func<Frame, float[]> resize, int width, int height)
    {
        if (input is ImageBatch image) return Pixels.Map(image, f => resize(new Frame(f, image.Channels)), height, width);
        var mask = (MaskBatch)input;
        int n = mask.Width * mask.Height;
        var result = new MaskBatch(mask.Batch, height, width);
        for (int b = 0; b < mask.Batch; b++)
            Array.Copy(resize(new Frame(mask.Data.AsSpan(b * n, n).ToArray(), 1)), 0, result.Data, (long)b * width * height, width * height);
        return result;
    }

    /// <summary>The middle <paramref name="tw"/> x <paramref name="th"/> of a picture, offsets rounded down.</summary>
    static float[] CutMiddle(float[] src, int w, int h, int c, int tw, int th)
    {
        int x0 = (w - tw) / 2, y0 = (h - th) / 2;
        var cut = new float[tw * th * c];
        for (int y = 0; y < th; y++) Array.Copy(src, ((y + y0) * w + x0) * c, cut, y * tw * c, tw * c);
        return cut;
    }
}
