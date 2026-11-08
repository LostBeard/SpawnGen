using System.Collections.ObjectModel;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// A histogram (the HISTOGRAM socket): one count per level 0..255. It is a list of numbers, so Preview as Text shows it as JSON and a page receives it as
/// an array, as ComfyUI's (a plain list) does.
/// </summary>
[Socket("HISTOGRAM")]
public sealed class Histogram : ReadOnlyCollection<long>
{
    public Histogram(long[] bins) : base(bins) => Bins = bins;

    /// <summary>The counts by level.</summary>
    public long[] Bins { get; }

    /// <summary>Python's repr of the list: <c>[1, 0, 0, ...]</c>.</summary>
    public override string ToString() => "[" + string.Join(", ", Bins) + "]";
}

/// <summary>
/// Image Histogram: the levels of the first picture of the batch. Each channel's value is scaled to 0..255 in single precision, clipped and truncated to a
/// level; luminance is 0.2126 R + 0.7152 G + 0.0722 B of the unclipped values, scaled the same way; rgb is the mean of the three channel histograms per level
/// (integer division). Measured with ComfyUI: the outputs are plain lists of 256 counts.
/// </summary>
[Node("ImageHistogram", Category = "utilities", DisplayName = "Image Histogram")]
public sealed class ImageHistogram
{
    public (Histogram Rgb, Histogram Luminance, Histogram Red, Histogram Green, Histogram Blue) Execute(ImageBatch image)
    {
        if (image.Batch == 0) throw new ArgumentException("The image batch is empty.", nameof(image));
        if (image.Channels < 3) throw new ArgumentException($"Image Histogram needs a picture with three colour channels, not {image.Channels}.", nameof(image));
        var red = new long[256];
        var green = new long[256];
        var blue = new long[256];
        var lum = new long[256];
        int c = image.Channels, pixels = image.Height * image.Width;
        var data = image.Data;
        for (int p = 0; p < pixels; p++)
        {
            int o = p * c;
            float r = data[o], g = data[o + 1], b = data[o + 2];
            red[Level(r)]++;
            green[Level(g)]++;
            blue[Level(b)]++;
            lum[Level(0.2126f * r + 0.7152f * g + 0.0722f * b)]++;
        }
        var rgb = new long[256];
        for (int i = 0; i < 256; i++) rgb[i] = (red[i] + green[i] + blue[i]) / 3;
        return (new Histogram(rgb), new Histogram(lum), new Histogram(red), new Histogram(green), new Histogram(blue));
    }

    /// <summary>The level of a value: value * 255 in single precision, clipped to 0..255 and truncated (NaN counts as 0).</summary>
    static int Level(float v)
    {
        float scaled = v * 255f;
        if (!(scaled > 0f)) return 0;
        if (scaled >= 255f) return 255;
        return (int)scaled;
    }
}

/// <summary>Preview Mask: shows the masks as grey pictures (each stored as a temporary preview) and passes them on.</summary>
[Node("MaskPreview", Category = "image/mask", DisplayName = "Preview Mask", IsOutput = true,
    Description = "Preview the masks without saving them to the ComfyUI output directory.")]
public sealed class MaskPreview
{
    public async Task<(MaskBatch Mask, UiOutput Ui)> ExecuteAsync(MaskBatch mask, NodeContext context, IServiceProvider services)
    {
        int pixels = mask.Height * mask.Width;
        var grey = new ImageBatch(mask.Batch, mask.Height, mask.Width, 3);
        for (int i = 0; i < mask.Batch * pixels; i++)
        {
            float v = mask.Data[i];
            grey.Data[i * 3] = v;
            grey.Data[i * 3 + 1] = v;
            grey.Data[i * 3 + 2] = v;
        }
        var saved = await Results.PreviewAsync(grey, context, services);
        return (mask, new UiOutput("images", saved));
    }
}

/// <summary>
/// Resolution Selector: a width and height with the chosen aspect ratio and about the chosen number of megapixels (1 MP = 1024 x 1024 pixels), each a
/// multiple of <c>multiple</c>. Measured with ComfyUI over every ratio, 0.1..16 MP and every multiple: width = round(sqrt(total * w / h) / multiple) * multiple
/// and height = round(sqrt(total * h / w) / multiple) * multiple, rounding halves to even.
/// </summary>
[Node("ResolutionSelector", Category = "utilities", DisplayName = "Resolution Selector",
    Description = "Calculate width and height from aspect ratio and megapixel target. Useful for setting up Empty Latent Image dimensions.")]
public sealed class ResolutionSelector
{
    static readonly (string Name, int W, int H)[] Ratios =
    [
        ("1:1 (Square)", 1, 1), ("2:3 (Portrait Photo)", 2, 3), ("3:2 (Photo)", 3, 2), ("3:4 (Portrait Standard)", 3, 4), ("4:3 (Standard)", 4, 3),
        ("9:16 (Portrait Widescreen)", 9, 16), ("16:9 (Widescreen)", 16, 9), ("21:9 (Ultrawide)", 21, 9),
    ];

    public (int Width, int Height) Execute(
        [Input(Options = ["1:1 (Square)", "2:3 (Portrait Photo)", "3:2 (Photo)", "3:4 (Portrait Standard)", "4:3 (Standard)", "9:16 (Portrait Widescreen)",
            "16:9 (Widescreen)", "21:9 (Ultrawide)"], Tooltip = "The aspect ratio for the output dimensions.")] string aspectRatio,
        [Input(Min = 0.1, Max = 16.0, Step = 0.1, Tooltip = "Target total megapixels. 1.0 MP ≈ 1024x1024 for square.")] double megapixels = 1.0,
        [Input(Min = 8, Max = 128, Step = 4, Tooltip = "Nearest multiple of the result to set the selected resolution to.")] int multiple = 8)
    {
        var ratio = Ratios.FirstOrDefault(r => r.Name == aspectRatio);
        if (ratio.Name is null) throw new ArgumentException($"Unknown aspect ratio '{aspectRatio}'.", nameof(aspectRatio));
        double total = megapixels * 1024 * 1024;
        double width = Math.Sqrt(total * ratio.W / ratio.H);
        double height = Math.Sqrt(total * ratio.H / ratio.W);
        return ((int)Math.Round(width / multiple, MidpointRounding.ToEven) * multiple, (int)Math.Round(height / multiple, MidpointRounding.ToEven) * multiple);
    }
}
