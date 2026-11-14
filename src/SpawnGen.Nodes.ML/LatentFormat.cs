namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md): the scale factors are the published VAE scaling factors of SD 1.x (0.18215) and SDXL (0.13025); the
// latent-to-RGB factors were read from a running ComfyUI (2026-11-11); the empty-latent rule is measured (LatentFormatTests). Wan 2.2's per-channel
// mean and std are the published constants its VAE file carries (latents_mean / latents_std, Apache-2.0); the mean is also what ComfyUI's
// Wan22ImageToVideoLatent fills the frames after a start picture with (Fixtures/wan22, measured 2026-11-12).

/// <summary>A model family's latent space: how sampling-space latents relate to the VAE's, and a cheap RGB preview of a latent.</summary>
public sealed class LatentFormat(string name, float scaleFactor, float[][] rgbFactors, float[]? rgbBias = null, string? taesdDecoderName = null,
    int channels = 4, float[]? latentsMean = null, float[]? latentsStd = null)
{
    /// <summary>The latent's channels (4 for SD 1.x / SDXL, 48 for Wan 2.2).</summary>
    public int Channels { get; } = channels;

    /// <summary>Per channel, the mean taken off on the way into the model (null: none).</summary>
    public float[]? LatentsMean { get; } = latentsMean;

    /// <summary>Per channel, the spread divided out on the way into the model (null: none).</summary>
    public float[]? LatentsStd { get; } = latentsStd;

    public string Name { get; } = name;

    /// <summary>The tiny VAE decoder for previews, as named in models/vae_approx (taesd_decoder, taesdxl_decoder).</summary>
    public string? TaesdDecoderName { get; } = taesdDecoderName;
    public float ScaleFactor { get; } = scaleFactor;

    /// <summary>Per latent channel, its contribution to red, green and blue in the preview.</summary>
    public float[][] RgbFactors { get; } = rgbFactors;

    /// <summary>Added to red, green and blue in the preview (null: none).</summary>
    public float[]? RgbBias { get; } = rgbBias;

    public static LatentFormat SD15 { get; } = new("SD15", 0.18215f,
        [[0.3512f, 0.2297f, 0.3227f], [0.3250f, 0.4974f, 0.2350f], [-0.2829f, 0.1762f, 0.2721f], [-0.2120f, -0.2616f, -0.7177f]], null, "taesd_decoder");

    public static LatentFormat SDXL { get; } = new("SDXL", 0.13025f,
        [[0.3651f, 0.4232f, 0.4341f], [-0.2533f, -0.0042f, 0.1068f], [0.1076f, 0.1111f, -0.0362f], [-0.3165f, -0.2492f, -0.2188f]],
        [0.1084f, -0.0175f, -0.0011f], "taesdxl_decoder");

    /// <summary>
    /// FLUX.2 (Klein): ComfyUI's LATENT is already the transformer's space (the VAE mean patchified 2 x 2 and batch-normalised by the VAE file's
    /// statistics, measured), so nothing is taken off or scaled on the way in or out. 128 channels; no cheap RGB preview yet.
    /// </summary>
    public static LatentFormat Flux2 { get; } = new("Flux2", 1f, [], null, null, 128);

    /// <summary>
    /// FLUX.1 / Z-Image (ComfyUI's ae.safetensors): LATENT is the VAE's raw mean, 16 channels; on the way into the model 0.1159 is taken off and the
    /// rest scaled by 0.3611 (measured with ComfyUI's model calls, 2026-11-14). No cheap RGB preview yet.
    /// </summary>
    public static LatentFormat Flux1 { get; } = new("Flux1", 0.3611f, [], null, null, 16, [.. Enumerable.Repeat(0.1159f, 16)], [.. Enumerable.Repeat(1f, 16)]);

    /// <summary>
    /// Wan 2.2 (TI2V 5B): 48 channels, frames x h x w per channel, each channel normalised by its own mean and spread (scale factor 1).
    /// No cheap RGB preview.
    /// </summary>
    public static LatentFormat Wan22 { get; } = new("Wan22", 1f, [], null, null, 48,
        [-0.2289f, -0.0052f, -0.1323f, -0.2339f, -0.2799f, 0.0174f, 0.1838f, 0.1557f, -0.1382f, 0.0542f, 0.2813f, 0.0891f, 0.1570f, -0.0098f, 0.0375f, -0.1825f,
         -0.2246f, -0.1207f, -0.0698f, 0.5109f, 0.2665f, -0.2108f, -0.2158f, 0.2502f, -0.2055f, -0.0322f, 0.1109f, 0.1567f, -0.0729f, 0.0899f, -0.2799f, -0.1230f,
         -0.0313f, -0.1649f, 0.0117f, 0.0723f, -0.2839f, -0.2083f, -0.0520f, 0.3748f, 0.0152f, 0.1957f, 0.1433f, -0.2944f, 0.3573f, -0.0548f, -0.1681f, -0.0667f],
        [0.4765f, 1.0364f, 0.4514f, 1.1677f, 0.5313f, 0.4990f, 0.4818f, 0.5013f, 0.8158f, 1.0344f, 0.5894f, 1.0901f, 0.6885f, 0.6165f, 0.8454f, 0.4978f,
         0.5759f, 0.3523f, 0.7135f, 0.6804f, 0.5833f, 1.4146f, 0.8986f, 0.5659f, 0.7069f, 0.5338f, 0.4889f, 0.4917f, 0.4069f, 0.4999f, 0.6866f, 0.4093f,
         0.5709f, 0.6065f, 0.6415f, 0.4944f, 0.5726f, 1.2042f, 0.5458f, 1.6887f, 0.3971f, 1.0600f, 0.3943f, 0.5537f, 0.5444f, 0.4089f, 0.7468f, 0.7744f]);

    /// <summary>
    /// VAE latent to sampling space: (value - mean) x scale factor / std per channel (plain x scale factor without them). An all-zero latent (an empty
    /// latent) is left as it is (measured), always as a copy. <paramref name="latent"/> is one item, channel-major.
    /// </summary>
    public float[] ProcessIn(float[] latent)
    {
        var r = (float[])latent.Clone();
        if (Array.TrueForAll(latent, v => v == 0f)) return r;
        int plane = LatentsMean is null ? 0 : latent.Length / Channels;
        for (int i = 0; i < r.Length; i++)
        {
            if (plane == 0) { r[i] *= ScaleFactor; continue; }
            int c = i / plane;
            r[i] = (r[i] - LatentsMean![c]) * ScaleFactor / LatentsStd![c];
        }
        return r;
    }

    /// <summary>Sampling space back to the VAE's latent: value x std / scale factor + mean per channel (divided by the scale factor without them).</summary>
    public float[] ProcessOut(float[] latent)
    {
        var r = new float[latent.Length];
        int plane = LatentsMean is null ? 0 : latent.Length / Channels;
        for (int i = 0; i < r.Length; i++)
        {
            if (plane == 0) { r[i] = latent[i] / ScaleFactor; continue; }
            int c = i / plane;
            r[i] = latent[i] * LatentsStd![c] / ScaleFactor + LatentsMean![c];
        }
        return r;
    }
}
