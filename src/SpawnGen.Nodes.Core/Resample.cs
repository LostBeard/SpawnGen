namespace SpawnGen.Nodes.Core;

public enum ResampleMethod { NearestExact, Bilinear, Area, Bicubic, Lanczos }

/// <summary>
/// Image resizing by method name or enum, on interleaved frames. Everything goes through <see cref="Core.CommonUpscale"/>, the port of ComfyUI's
/// common_upscale checked against PyTorch and Pillow output: PyTorch's interpolate for nearest-exact, bilinear, bicubic and area, Pillow's 8-bit
/// LANCZOS for lanczos.
/// </summary>
public static class Resample
{
    public static readonly string[] MethodNames = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"];

    public static ResampleMethod Parse(string name) => name switch
    {
        "nearest-exact" or "nearest" => ResampleMethod.NearestExact,
        "bilinear" => ResampleMethod.Bilinear,
        "area" => ResampleMethod.Area,
        "bicubic" => ResampleMethod.Bicubic,
        "lanczos" => ResampleMethod.Lanczos,
        _ => throw new ArgumentException($"Unknown upscale method '{name}'. Known: {string.Join(", ", MethodNames)}.", nameof(name)),
    };

    /// <summary>ComfyUI's <c>comfy.utils.common_upscale</c>: with <paramref name="centerCrop"/> the source is first cut to the target's aspect ratio around its middle, then resized.</summary>
    public static float[] CommonUpscale(float[] src, int w, int h, int channels, int newW, int newH, ResampleMethod method, bool centerCrop) =>
        Core.CommonUpscale.Resize(src, w, h, channels, newW, newH, Name(method), centerCrop ? "center" : "disabled");

    /// <summary>Resizes one picture of <paramref name="channels"/> interleaved channels from (w, h) to (newW, newH).</summary>
    public static float[] Resize(float[] src, int w, int h, int channels, int newW, int newH, ResampleMethod method)
    {
        if (w <= 0 || h <= 0 || newW <= 0 || newH <= 0) throw new ArgumentOutOfRangeException(nameof(newW), "Sizes must be positive.");
        return Core.CommonUpscale.Resize(src, w, h, channels, newW, newH, Name(method));
    }

    public static string Name(ResampleMethod method) => method switch
    {
        ResampleMethod.NearestExact => "nearest-exact",
        ResampleMethod.Bilinear => "bilinear",
        ResampleMethod.Area => "area",
        ResampleMethod.Bicubic => "bicubic",
        _ => "lanczos",
    };
}
