using SpawnGen.Engine.Types;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md): a latent shown as a picture through its format's latent-to-RGB factors (LatentFormat), mapped from
// -1..1 to 0..1 and clamped, as SpawnGen's LatentPreviewTests and LatentFormatTests pin.

/// <summary>Cheap previews of a latent while it is being sampled.</summary>
public static class LatentPreview
{
    /// <summary>One latent (channels x height x width) as an RGB picture of the same size.</summary>
    public static ImageBatch Image(float[] latent, int height, int width, LatentFormat? format = null)
    {
        format ??= LatentFormat.SD15;
        int pixels = height * width, channels = Math.Min(format.RgbFactors.Length, latent.Length / Math.Max(pixels, 1));
        var image = new ImageBatch(1, height, width, 3);
        for (int p = 0; p < pixels; p++)
            for (int k = 0; k < 3; k++)
            {
                float v = format.RgbBias?[k] ?? 0f;
                for (int c = 0; c < channels; c++) v += latent[c * pixels + p] * format.RgbFactors[c][k];
                image.Data[p * 3 + k] = Math.Clamp((v + 1f) / 2f, 0f, 1f);
            }
        return image;
    }

    /// <summary><see cref="Image"/> as a PNG.</summary>
    public static byte[] Png(float[] latent, int height, int width, LatentFormat? format = null) => PngWriter.Encode(Image(latent, height, width, format));
}

/// <summary>
/// A sampler's live preview for one run, chosen by ComfyUI's preview method: "none" gives none; "latent2rgb" the colour guess of
/// <see cref="LatentPreview"/>; "taesd" and "auto" (also null / "default") the tiny VAE decoder from models/vae_approx when one is there for the
/// model family, otherwise the colour guess.
/// </summary>
public sealed class LatentPreviewer
{
    readonly LatentFormat _format;
    readonly SdTaesd? _taesd;

    LatentPreviewer(LatentFormat format, SdTaesd? taesd)
    {
        _format = format;
        _taesd = taesd;
    }

    /// <summary>Whether previews go through the tiny VAE decoder.</summary>
    public bool UsesTaesd => _taesd is not null;

    public static async Task<LatentPreviewer?> CreateAsync(string? method, LatentFormat format, IServiceProvider? services)
    {
        method = method is null or "" or "default" ? "auto" : method;
        if (method == "none") return null;
        if (method is "taesd" or "auto" && services is not null && format.TaesdDecoderName is { } decoder
            && ModelFolders.Names(ModelFolders.VaeApprox).FirstOrDefault(n => Path.GetFileNameWithoutExtension(n) == decoder) is { } name)
        {
            var runtime = await GpuHelpers.RuntimeAsync(services);
            var (repo, file) = ModelFolders.Locate(ModelFolders.VaeApprox, name);
            var taesd = await runtime.GetOrLoadAsync($"preview:{repo}/{file}",
                () => SdTaesd.LoadAsync(runtime.Accelerator, runtime.ModelSource, repo, file, null!, format.ScaleFactor));
            return new LatentPreviewer(format, taesd);
        }
        return new LatentPreviewer(format, null);
    }

    /// <summary>The preview of one latent (channels x height x width, sampling space) as a PNG: 8 times its size through the tiny VAE, else its own size.</summary>
    public async Task<byte[]> PngAsync(float[] latent, int height, int width, CancellationToken cancellationToken)
    {
        if (_taesd is null) return LatentPreview.Png(latent, height, width, _format);
        var rgb = await _taesd.DecodeAsync(_format.ProcessOut(latent), height, width, cancellationToken); // [3, 8h, 8w] in -1..1
        int h = height * 8, w = width * 8, plane = h * w;
        var image = new ImageBatch(1, h, w, 3);
        for (int p = 0; p < plane; p++)
            for (int k = 0; k < 3; k++) image.Data[p * 3 + k] = Math.Clamp((rgb[k * plane + p] + 1f) / 2f, 0f, 1f);
        return PngWriter.Encode(image);
    }
}
