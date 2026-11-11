using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Colour-model nodes. Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries and node descriptions, SpawnGen's tests and fixtures
// made by running ComfyUI, black-box measurements, and the published standards cited next to each formula.

/// <summary>
/// Splits RGB into Y, Cb (U) and Cr (V), each returned as a grey 3-channel picture: Y = 0.299 R + 0.587 G + 0.114 B, U = (B - Y) * 0.564 + 0.5,
/// V = (R - Y) * 0.713 + 0.5 (the JPEG / BT.601 full-range coefficients, values measured with ComfyUI).
/// </summary>
[Node(Category = "image/color", DisplayName = "Image RGB to YUV", OutputNames = ["Y", "U", "V"])]
public sealed class ImageRGBToYUV
{
    public (ImageBatch Y, ImageBatch U, ImageBatch V) Execute(ImageBatch image)
    {
        int c = image.Channels, n = image.Batch * image.Height * image.Width;
        var y = new ImageBatch(image.Batch, image.Height, image.Width, 3);
        var u = new ImageBatch(image.Batch, image.Height, image.Width, 3);
        var v = new ImageBatch(image.Batch, image.Height, image.Width, 3);
        for (int i = 0; i < n; i++)
        {
            long p = (long)i * c;
            float r = image.Data[p], g = image.Data[p + Math.Min(1, c - 1)], b = image.Data[p + Math.Min(2, c - 1)];
            float luma = 0.299f * r + 0.587f * g + 0.114f * b;
            float cb = (b - luma) * 0.564f + 0.5f;
            float cr = (r - luma) * 0.713f + 0.5f;
            for (int ch = 0; ch < 3; ch++)
            {
                y.Data[i * 3 + ch] = luma;
                u.Data[i * 3 + ch] = cb;
                v.Data[i * 3 + ch] = cr;
            }
        }
        return (y, u, v);
    }
}

/// <summary>
/// Joins Y, U and V pictures (each taken as the mean of its channels) back into RGB: R = Y + 1.403 (V - 0.5), G = Y - 0.714 (V - 0.5) - 0.344 (U - 0.5),
/// B = Y + 1.773 (U - 0.5), each clamped to 0..1 (measured with ComfyUI).
/// </summary>
[Node(Category = "image/color", DisplayName = "Image YUV to RGB")]
public sealed class ImageYUVToRGB
{
    public ImageBatch Execute([Input(Name = "Y")] ImageBatch y, [Input(Name = "U")] ImageBatch u, [Input(Name = "V")] ImageBatch v)
    {
        if (u.Batch != y.Batch || v.Batch != y.Batch || u.Height != y.Height || v.Height != y.Height || u.Width != y.Width || v.Width != y.Width)
            throw new ArgumentException("Y, U and V must have the same size.");
        int n = y.Batch * y.Height * y.Width;
        var result = new ImageBatch(y.Batch, y.Height, y.Width, 3);
        for (int i = 0; i < n; i++)
        {
            float luma = Mean(y, i), cb = Mean(u, i) - 0.5f, cr = Mean(v, i) - 0.5f;
            result.Data[i * 3] = Math.Clamp(luma + 1.403f * cr, 0f, 1f);
            result.Data[i * 3 + 1] = Math.Clamp(luma - 0.714f * cr - 0.344f * cb, 0f, 1f);
            result.Data[i * 3 + 2] = Math.Clamp(luma + 1.773f * cb, 0f, 1f);
        }
        return result;
    }

    /// <summary>The mean of pixel <paramref name="i"/>'s channels: each input counts as grey (measured: non-grey Y, U and V pictures).</summary>
    static float Mean(ImageBatch image, int i)
    {
        int c = image.Channels;
        float sum = 0;
        for (int k = 0; k < c; k++) sum += image.Data[(long)i * c + k];
        return sum / c;
    }
}

/// <summary>
/// Converts pictures between sRGB, linear Rec.709, HDR (Rec.2020 HLG), HDR PQ (Rec.2020 PQ), ARRI LogC3 (EI 800) and ACEScct. A fourth (alpha) channel
/// passes through unchanged. See <see cref="ColorSpaces"/> for the formulas.
/// </summary>
[Node(Category = "image/color", DisplayName = "Convert Image Color Space",
    Description = "Convert sRGB, linear Rec.709, HDR (Rec.2020 HLG), HDR PQ (Rec.2020 PQ), HDR LogC3, and HDR ACEScct. LogC3 uses the EI 800 curve with Rec.709 primaries and codes clamped to [0, 1]. ACEScct uses AP1 primaries and D60 white, with Bradford adaptation to D65. Convert LogC3 or ACEScct to linear for EXR saving. Linear 1.0 uses the same 203-nit reference white as sRGB; HLG uses a 1000-nit reference display. Linear and ACEScct outputs preserve extended values. Linear-to-HDR conversions preserve highlights without tone mapping; HLG and PQ outputs clip negative channels. SDR output and PQ-to-HLG conversion tone-map excess luminance across the batch and compress out-of-gamut colors. Conversions compute in float32 and return the intermediate device and dtype. Straight alpha is not color-transformed.")]
public sealed class ImageColorSpace
{
    public ImageBatch Execute(ImageBatch image,
        [Input(Options = ["sRGB", "HDR", "HDR PQ", "linear", "HDR LogC3", "HDR ACEScct"], Tooltip = "Color space of the input pixels.")]
        string source = "sRGB",
        [Input(Options = ["sRGB", "HDR", "HDR PQ", "linear", "HDR LogC3", "HDR ACEScct"],
            Tooltip = "Color space of the output pixels. Set the save node to this same color space. Convert LogC3 or ACEScct to linear before saving EXR, or to sRGB/HDR/HDR PQ before saving video.")]
        string destination = "sRGB")
    {
        var from = ColorSpaces.Parse(source, nameof(source));
        var to = ColorSpaces.Parse(destination, nameof(destination));
        if (from == to) return image;
        var converted = ColorSpaces.Convert(image.Data, image.Batch * image.Height * image.Width, image.Channels, from, to);
        return new ImageBatch(image.Batch, image.Height, image.Width, image.Channels, converted.Data);
    }
}
