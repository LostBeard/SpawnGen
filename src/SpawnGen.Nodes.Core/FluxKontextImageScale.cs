using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// FluxKontextImageScale: the picture resized to the FLUX Kontext resolution whose aspect ratio is nearest its own (smallest |w/h - W/H|), with
/// Lanczos after a centre crop to that aspect. Measured with ComfyUI (2026-11-13): 239 aspect ratios from 1/4 to 4 give these 17 sizes, the
/// boundaries fall where |w/h - W/H| ties (1.083 stays 1024 x 1024, 1.086 goes to 1104 x 944; 0.9294 stays, 0.9259 goes to 944 x 1104), and the
/// pixels equal Lanczos with a centre crop exactly (Fixtures/flux_kontext_scale).
/// </summary>
[Node("FluxKontextImageScale", Category = "model/conditioning/flux", DisplayName = "FluxKontextImageScale",
    Description = "This node resizes the image to one that is more optimal for flux kontext.")]
public sealed class FluxKontextImageScale
{
    /// <summary>The 17 sizes (width, height), from tallest to widest.</summary>
    public static readonly (int Width, int Height)[] Resolutions =
    [
        (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328), (832, 1248), (880, 1184), (944, 1104), (1024, 1024),
        (1104, 944), (1184, 880), (1248, 832), (1328, 800), (1392, 752), (1456, 720), (1504, 688), (1568, 672),
    ];

    /// <summary>The size a <paramref name="width"/> x <paramref name="height"/> picture is scaled to (the first of equally near ones).</summary>
    public static (int Width, int Height) Target(int width, int height)
    {
        double aspect = (double)width / height;
        var best = Resolutions[0];
        double bestDistance = double.MaxValue;
        foreach (var r in Resolutions)
        {
            double d = Math.Abs(aspect - (double)r.Width / r.Height);
            if (d < bestDistance) { bestDistance = d; best = r; }
        }
        return best;
    }

    public ImageBatch Execute(ImageBatch image)
    {
        var (w, h) = Target(image.Width, image.Height);
        return CommonUpscale.Resize(image, w, h, "lanczos", "center");
    }
}
