namespace SpawnGen.Nodes.ML;

/// <summary>
/// ComfyUI's <c>latent_format</c>: how a model family's latents relate to its VAE's. A LATENT holds the VAE's own values (as in ComfyUI); the sampler brings them
/// into the model's space with <see cref="ProcessIn"/> and back with <see cref="ProcessOut"/>, and the latent preview mixes the model-space channels to RGB.
/// </summary>
public sealed class LatentFormat
{
    LatentFormat(string name, float scaleFactor, float[,] rgbFactors, float[]? rgbBias)
    {
        Name = name;
        ScaleFactor = scaleFactor;
        RgbFactors = rgbFactors;
        RgbBias = rgbBias;
    }

    /// <summary>SD 1.x, SD 2.x and SD-Turbo.</summary>
    public static readonly LatentFormat SD15 = new("SD15", 0.18215f, new float[,]
    {
        {  0.3512f,  0.2297f,  0.3227f },
        {  0.3250f,  0.4974f,  0.2350f },
        { -0.2829f,  0.1762f,  0.2721f },
        { -0.2120f, -0.2616f, -0.7177f },
    }, null);

    /// <summary>SDXL (base and fine-tunes).</summary>
    public static readonly LatentFormat SDXL = new("SDXL", 0.13025f, new float[,]
    {
        {  0.3651f,  0.4232f,  0.4341f },
        { -0.2533f, -0.0042f,  0.1068f },
        {  0.1076f,  0.1111f, -0.0362f },
        { -0.3165f, -0.2492f, -0.2188f },
    }, [0.1084f, -0.0175f, -0.0011f]);

    public string Name { get; }
    /// <summary>The start of the tiny VAE decoder's file name in vae_approx used for previews (ComfyUI's taesd_decoder_name).</summary>
    public string TaesdDecoderName => Name == "SDXL" ? "taesdxl_decoder" : "taesd_decoder";
    public float ScaleFactor { get; }
    public int Channels => RgbFactors.GetLength(0);
    /// <summary>The latent-to-RGB preview mix: rows are latent channels, columns R, G, B.</summary>
    public float[,] RgbFactors { get; }
    public float[]? RgbBias { get; }

    /// <summary>VAE values to the model's (<c>latent * scale_factor</c> in float32). An all-zero latent is left alone, as ComfyUI does ("don't shift the empty latent").</summary>
    public float[] ProcessIn(float[] latent)
    {
        if (Array.TrueForAll(latent, v => v == 0f)) return (float[])latent.Clone();
        var r = new float[latent.Length];
        for (int i = 0; i < r.Length; i++) r[i] = latent[i] * ScaleFactor;
        return r;
    }

    /// <summary>The model's values back to the VAE's (<c>latent / scale_factor</c>).</summary>
    public float[] ProcessOut(float[] latent)
    {
        var r = new float[latent.Length];
        for (int i = 0; i < r.Length; i++) r[i] = latent[i] / ScaleFactor;
        return r;
    }
}
