using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.ILGPU.ML.Flux2;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.StableDiffusion;

namespace SpawnGen.Nodes.ML;

// SpawnGen's plumbing over SpawnDev.ILGPU.ML's FLUX.2 building blocks as its docs describe them (SdVae for ComfyUI's flux2-vae.safetensors,
// Flux2VaeNorm, Flux2Kernels.VaeToLatent / LatentToVae), so ComfyUI's split FLUX.2 graph keeps ComfyUI's LATENT. Measured with ComfyUI
// (2026-11-13, _DevComms 2026-11-13-2230 dumps, gated by the library against them): FLUX.2's LATENT is [b, 128, H/16, W/16], the VAE encoder's
// mean patchified 2 x 2 (channel c * 4 + dy * 2 + dx) and batch-normalised with the statistics the VAE file carries; a picture is cut to multiples
// of 16 around its middle first.

/// <summary>FLUX.2's VAE (ComfyUI's flux2-vae.safetensors, Load VAE) giving and taking ComfyUI's FLUX.2 LATENT.</summary>
internal sealed class Flux2VaeModel(string name, SdVae vae, Flux2VaeNorm norm, Accelerator accelerator) : IVaeModel, IDisposable
{
    /// <summary>Latent channels of FLUX.2's LATENT (the VAE's 32 times the 2 x 2 patch).</summary>
    public const int LatentChannels = 128;

    readonly Flux2Kernels _kernels = new(accelerator);
    readonly MemoryBuffer1D<float, Stride1D.Dense> _mean = accelerator.Allocate1D(norm.Mean);
    readonly MemoryBuffer1D<float, Stride1D.Dense> _std = accelerator.Allocate1D(norm.Std);

    public string Name { get; } = name;
    public SdVae SdVae { get; } = vae;

    /// <summary>One LATENT cell is 16 x 16 pixels.</summary>
    public int Downscale => 16;

    public static async Task<Flux2VaeModel> CreateAsync(string name, SdVae vae, Accelerator accelerator, IModelSource source, string repo, string file,
        CancellationToken ct = default) => new(name, vae, await Flux2VaeNorm.FromSafeTensorsAsync(source, repo, file, ct), accelerator);

    /// <summary>[128, h, w] LATENT to the picture [3, 16h, 16w] in [-1, 1].</summary>
    public async Task<float[]> DecodeAsync(float[] latent, int h, int w, CancellationToken cancellationToken = default)
    {
        if (latent.Length != LatentChannels * h * w) throw new ArgumentException($"A FLUX.2 latent is {LatentChannels} channels; this one has {latent.Length / (h * w)}.");
        using var lat = accelerator.Allocate1D(latent);
        using var z = accelerator.Allocate1D<float>(32 * 4 * h * w);
        using var picture = accelerator.Allocate1D<float>(3 * 256 * h * w);
        _kernels.LatentToVae(lat.View, _mean.View, _std.View, z.View, h, w);
        await SdVae.DecodeAsync(z.View, 2 * h, 2 * w, picture.View, cancellationToken);
        await accelerator.SynchronizeAsync();
        return await picture.CopyToHostAsync(0, picture.Length);
    }

    /// <summary>The picture [3, H, W] in [-1, 1] (H, W multiples of 16) to the LATENT [128, H/16, W/16].</summary>
    public async Task<float[]> EncodeImageAsync(float[] image, int H, int W, CancellationToken cancellationToken = default)
    {
        if (H % 16 != 0 || W % 16 != 0) throw new ArgumentException($"FLUX.2's VAE encodes sides that are multiples of 16, got {W}x{H}.");
        using var pixels = accelerator.Allocate1D(image);
        using var z = accelerator.Allocate1D<float>(32 * (H / 8) * (W / 8));
        using var latent = accelerator.Allocate1D<float>(LatentChannels * (H / 16) * (W / 16));
        await SdVae.EncodeAsync(pixels.View, H, W, z.View, cancellationToken);
        _kernels.VaeToLatent(z.View, _mean.View, _std.View, latent.View, H / 8, W / 8);
        await accelerator.SynchronizeAsync();
        return await latent.CopyToHostAsync(0, latent.Length);
    }

    public void Dispose()
    {
        _mean.Dispose();
        _std.Dispose();
        SdVae.Dispose();
    }
}
