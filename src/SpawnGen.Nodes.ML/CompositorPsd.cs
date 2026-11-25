using ILGPU.Runtime;
using SpawnDev.Codecs.Images.Psd;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// Export PSD of the compositor editor: a composition exported by <see cref="ImageCompositor.Export"/> written as a layered PSD with
/// SpawnDev.Codecs' PsdEncoderGpu (PSD version 1, RGB, 8 bit, RLE; every layer with its place, name, blend key, opacity and visibility, and the
/// flattened picture). The file's name is "spawngen-layers-YYYYMMDD-HHMMSS.psd" (ComfyUI's is comfyui-layers-...).
/// </summary>
public static class CompositorPsd
{
    /// <summary>The PSD blend mode of a compositor blend id (the 26 of <see cref="CompositorBlend.ModeNames"/>, in the same order as the enum).</summary>
    public static PsdBlendMode BlendMode(string id)
    {
        int index = Array.IndexOf(CompositorBlend.ModeNames, id);
        if (index < 0) throw new ArgumentException($"Unknown blend mode '{id}'.", nameof(id));
        return Enum.Parse<PsdBlendMode>(id.Replace("-", ""), ignoreCase: true);
    }

    /// <summary>A layer opacity 0..1 as the file's byte: round(opacity x 255), as ComfyUI's export stores it (0.8 -> 204, 0.5 -> 128).</summary>
    public static byte OpacityByte(double opacity) => (byte)Math.Clamp(Math.Floor(opacity * 255 + 0.5), 0, 255);

    public static string FileName(DateTime now) => $"spawngen-layers-{now:yyyyMMdd-HHmmss}.psd";

    /// <summary>The document as PsdEncoderGpu takes it: the layers bottom first, managed pixels.</summary>
    public static PsdDocument Document(ImageCompositor.ExportDocument export) => new()
    {
        Width = export.Width, Height = export.Height,
        Layers = [.. export.Layers.Select(l => new PsdLayer
        {
            Name = l.Name, Left = l.Left, Top = l.Top, Width = l.Width, Height = l.Height, Rgba = l.Rgba,
            Blend = BlendMode(l.Blend), Opacity = OpacityByte(l.Opacity), Visible = l.Visible,
        })],
        Composite = new PsdComposite { Rgba = export.Composite, HasAlpha = export.HasAlpha },
    };

    /// <summary>The PSD file's bytes.</summary>
    public static async Task<byte[]> EncodeAsync(Accelerator accelerator, ImageCompositor.ExportDocument export)
    {
        using var encoder = new PsdEncoderGpu(accelerator);
        return await encoder.EncodeAsync(Document(export));
    }
}
