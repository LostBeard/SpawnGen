namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md): a typed front for CommonUpscale.

/// <summary>The resize methods of the image nodes.</summary>
public enum ResampleMethod { NearestExact, Bilinear, Area, Bicubic, Lanczos }

/// <summary>Resizing by <see cref="ResampleMethod"/>, over <see cref="CommonUpscale"/>.</summary>
public static class Resample
{
    /// <summary>The method names, in the order the nodes list them.</summary>
    public static readonly string[] MethodNames = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"];

    public static string Name(ResampleMethod method) => MethodNames[(int)method];

    public static ResampleMethod Parse(string name)
    {
        int i = Array.IndexOf(MethodNames, name);
        if (i < 0) throw new ArgumentException($"Unknown resize method '{name}'; known: {string.Join(", ", MethodNames)}.", nameof(name));
        return (ResampleMethod)i;
    }

    /// <summary>Resizes one picture of <paramref name="channels"/> interleaved floats per pixel; sizes must be positive.</summary>
    public static float[] Resize(float[] src, int w, int h, int channels, int newW, int newH, ResampleMethod method)
    {
        if (newW <= 0) throw new ArgumentOutOfRangeException(nameof(newW), "The new width must be positive.");
        if (newH <= 0) throw new ArgumentOutOfRangeException(nameof(newH), "The new height must be positive.");
        return global::SpawnGen.Nodes.Core.CommonUpscale.Resize(src, w, h, channels, newW, newH, Name(method));
    }

    /// <summary><see cref="CommonUpscale.Resize(float[], int, int, int, int, int, string, string)"/>, cropping to the target aspect first when asked.</summary>
    public static float[] CommonUpscale(float[] src, int w, int h, int channels, int newW, int newH, ResampleMethod method, bool centerCrop) =>
        global::SpawnGen.Nodes.Core.CommonUpscale.Resize(src, w, h, channels, newW, newH, Name(method), centerCrop ? "center" : "disabled");
}
