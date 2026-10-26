using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// A quick look at a latent without running the VAE: each pixel is a fixed linear mix of the latent channels (ComfyUI's "latent2rgb" previewer, with the
/// model family's factors and bias). Blurry and colour-shifted, but it shows composition and colour while a sampler works, for almost no cost.
/// </summary>
public static class LatentPreview
{
    /// <summary>The preview of a model-space latent (what the sampler holds) as an RGB picture the size of the latent (values 0..1).</summary>
    public static ImageBatch Image(float[] latent, int height, int width, LatentFormat? format = null)
    {
        format ??= LatentFormat.SD15;
        var factors = format.RgbFactors;
        int plane = height * width, channels = format.Channels;
        var image = new ImageBatch(1, height, width, 3);
        for (int i = 0; i < plane; i++)
            for (int ch = 0; ch < 3; ch++)
            {
                float v = format.RgbBias?[ch] ?? 0f;
                for (int c = 0; c < channels; c++) v += latent[c * plane + i] * factors[c, ch];
                image.Data[i * 3 + ch] = Math.Clamp((v + 1f) / 2f, 0f, 1f);
            }
        return image;
    }

    public static byte[] Png(float[] latent, int height, int width, LatentFormat? format = null) => PngWriter.Encode(Image(latent, height, width, format));
}

/// <summary>
/// The sampler's live preview, chosen like ComfyUI's get_previewer: "taesd" decodes the estimate with the family's tiny VAE decoder from vae_approx
/// (taesd_decoder / taesdxl_decoder; without one it falls back to latent2rgb), "none" shows nothing, anything else (auto, latent2rgb, default, unset) the
/// latent2rgb approximation. Pictures larger than 512 are shrunk to fit (ComfyUI's --preview-size).
/// </summary>
internal sealed class LatentPreviewer
{
    const int MaxSize = 512;
    readonly LatentFormat _format;
    readonly SdTaesd? _taesd;

    LatentPreviewer(LatentFormat format, SdTaesd? taesd) { _format = format; _taesd = taesd; }

    /// <summary>Whether previews come from the tiny VAE (else latent2rgb).</summary>
    public bool UsesTaesd => _taesd is not null;

    public static async Task<LatentPreviewer?> CreateAsync(string? method, LatentFormat format, IServiceProvider services)
    {
        if (method == "none") return null;
        if (method != "taesd") return new(format, null);
        var file = ModelFolders.Names(ModelFolders.VaeApprox).Order(StringComparer.Ordinal).FirstOrDefault(n => n.StartsWith(format.TaesdDecoderName, StringComparison.Ordinal));
        if (file is null) return new(format, null);
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var (repo, path) = ModelFolders.Locate(ModelFolders.VaeApprox, file);
        var taesd = await runtime.GetOrLoadAsync($"tae-preview:{repo}/{path}", async () =>
        {
            await using var stream = await runtime.ModelSource.OpenAsync(repo, path);
            return await SdTaesd.LoadAsync(runtime.Accelerator, stream, path, null, null, format.ScaleFactor);
        });
        return new(format, taesd);
    }

    /// <summary>The preview of a model-space estimate (what the sampler holds), as PNG bytes.</summary>
    public async Task<byte[]> PngAsync(float[] latent, int height, int width, CancellationToken cancellationToken)
    {
        if (_taesd is null) return LatentPreview.Png(latent, height, width, _format);
        // The tiny VAE takes the VAE's values and gives [3, 8h, 8w] in [-1, 1].
        var chw = await _taesd.DecodeAsync(_format.ProcessOut(latent), height, width, cancellationToken);
        int h = height * 8, w = width * 8, plane = h * w;
        var image = new ImageBatch(1, h, w, 3);
        for (int i = 0; i < plane; i++)
            for (int c = 0; c < 3; c++) image.Data[i * 3 + c] = Math.Clamp((chw[c * plane + i] + 1f) / 2f, 0f, 1f);
        if (Math.Max(h, w) > MaxSize)
        {
            double scale = (double)MaxSize / Math.Max(h, w);
            int nh = Math.Max(1, (int)(h * scale)), nw = Math.Max(1, (int)(w * scale));
            image = new ImageBatch(1, nh, nw, 3, Resample.Resize(image.Data, w, h, 3, nw, nh, ResampleMethod.Area));
        }
        return PngWriter.Encode(image);
    }
}
