using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>A histogram moving between nodes: one count per level 0..255 (ComfyUI's HISTOGRAM, a list of bin counts).</summary>
[Socket("HISTOGRAM")]
public sealed record Histogram(long[] Bins)
{
    /// <summary>Python's list text, <c>[0, 3, ...]</c>, as ComfyUI shows it.</summary>
    public override string ToString() => "[" + string.Join(", ", Bins) + "]";
}

/// <summary>Shows masks in the editor as grey pictures (ComfyUI's MaskPreview) and passes the mask on.</summary>
[Node(Category = "image/mask", DisplayName = "Preview Mask", IsOutput = true, Description = "Preview the masks without saving them to the ComfyUI output directory.")]
public sealed class MaskPreview
{
    public async Task<(MaskBatch Mask, UiOutput Ui)> ExecuteAsync(MaskBatch mask, NodeContext context, IServiceProvider services)
    {
        var images = new MaskToImage().Execute(mask);
        var store = Results.For(services);
        var saved = new List<object>();
        for (int i = 0; i < images.Batch; i++)
            saved.Add(await store.SaveAsync("temp", "preview", PngWriter.Encode(images, i, Results.Metadata(context))));
        return (mask, new UiOutput("images", saved));
    }
}

[Node(Category = "utilities", DisplayName = "Resolution Selector",
    Description = "Calculate width and height from aspect ratio and megapixel target. Useful for setting up Empty Latent Image dimensions.")]
public sealed class ResolutionSelector
{
    // ComfyUI's AspectRatio values, in its order. Its "preview" input is a frontend-only widget (not saved, not sent), so it is not an input here.
    static readonly (string Name, int W, int H)[] Ratios =
    [
        ("1:1 (Square)", 1, 1), ("2:3 (Portrait Photo)", 2, 3), ("3:2 (Photo)", 3, 2), ("3:4 (Portrait Standard)", 3, 4),
        ("4:3 (Standard)", 4, 3), ("9:16 (Portrait Widescreen)", 9, 16), ("16:9 (Widescreen)", 16, 9), ("21:9 (Ultrawide)", 21, 9),
    ];

    public static string[] AspectRatios() => Ratios.Select(r => r.Name).ToArray();

    /// <summary>The sides with the ratio whose area is the megapixel target, each rounded (half to even, Python's round) to a multiple of <c>multiple</c>.</summary>
    public (int Width, int Height) Execute(
        [Input(OptionsProvider = nameof(AspectRatios), Tooltip = "The aspect ratio for the output dimensions.")] string aspectRatio = "1:1 (Square)",
        [Input(Min = 0.1, Max = 16, Step = 0.1, Tooltip = "Target total megapixels. 1.0 MP ≈ 1024x1024 for square.")] double megapixels = 1.0,
        [Input(Min = 8, Max = 128, Step = 4, Tooltip = "Nearest multiple of the result to set the selected resolution to.")] int multiple = 8)
    {
        int idx = Array.FindIndex(Ratios, r => r.Name == aspectRatio);
        if (idx < 0) throw new ArgumentException($"Unknown aspect ratio '{aspectRatio}'.", nameof(aspectRatio));
        var (_, w, h) = Ratios[idx];
        double total = megapixels * 1024 * 1024, scale = Math.Sqrt(total / (w * h));
        return ((int)Math.Round(w * scale / multiple) * multiple, (int)Math.Round(h * scale / multiple) * multiple);
    }
}

[Node(Category = "utilities", DisplayName = "Image Histogram")]
public sealed class ImageHistogram
{
    /// <summary>
    /// Histograms of the first picture, 256 levels: each channel's value times 255 clamped and truncated to a level; rgb is the mean of the three (integer division);
    /// luminance uses BT.709 weights (computed in single precision like ComfyUI's numpy code).
    /// </summary>
    public (Histogram Rgb, Histogram Luminance, Histogram Red, Histogram Green, Histogram Blue) Execute(ImageBatch image)
    {
        if (image.Channels < 3) throw new ArgumentException("Image Histogram needs a colour picture.");
        int n = image.Height * image.Width, c = image.Channels;
        long[] r = new long[256], g = new long[256], b = new long[256], lum = new long[256];
        static int Level(float v) => (int)Math.Clamp(v * 255f, 0f, 255f); // np.clip then astype(uint8) truncates
        for (int i = 0; i < n; i++)
        {
            float vr = image.Data[i * c], vg = image.Data[i * c + 1], vb = image.Data[i * c + 2];
            r[Level(vr)]++; g[Level(vg)]++; b[Level(vb)]++;
            float l = 0.2126f * vr + 0.7152f * vg;
            l += 0.0722f * vb;
            lum[Level(l)]++;
        }
        var rgb = new long[256];
        for (int k = 0; k < 256; k++) rgb[k] = (r[k] + g[k] + b[k]) / 3;
        return (new(rgb), new(lum), new(r), new(g), new(b));
    }
}
