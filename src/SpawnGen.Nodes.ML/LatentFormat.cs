namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md): the scale factors are the published VAE scaling factors of SD 1.x (0.18215) and SDXL (0.13025); the
// latent-to-RGB factors were read from a running ComfyUI (2026-11-11); the empty-latent rule is measured (LatentFormatTests).

/// <summary>A model family's latent space: how sampling-space latents relate to the VAE's, and a cheap RGB preview of a latent.</summary>
public sealed class LatentFormat(string name, float scaleFactor, float[][] rgbFactors, float[]? rgbBias = null, string? taesdDecoderName = null)
{
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
    /// VAE latent to sampling space: times the scale factor. An all-zero latent (an empty latent) is left as it is (measured), always as a copy.
    /// </summary>
    public float[] ProcessIn(float[] latent)
    {
        var r = (float[])latent.Clone();
        if (Array.TrueForAll(latent, v => v == 0f)) return r;
        for (int i = 0; i < r.Length; i++) r[i] *= ScaleFactor;
        return r;
    }

    /// <summary>Sampling space back to the VAE's latent: divided by the scale factor.</summary>
    public float[] ProcessOut(float[] latent)
    {
        var r = new float[latent.Length];
        for (int i = 0; i < r.Length; i++) r[i] = latent[i] / ScaleFactor;
        return r;
    }
}
