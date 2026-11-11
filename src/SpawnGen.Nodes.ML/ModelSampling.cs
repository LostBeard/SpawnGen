namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md). The formulas come from Karras et al. 2022 (EDM: input scaling c_in, the eps / v / x0 / EDM estimates with
// sigma_data), k-diffusion (MIT), the LCM paper (boundary conditions c_skip / c_out with timestep scaling 10 and sigma_data 0.5) and rectified flow
// (Liu et al. 2022: x_sigma = sigma * noise + (1 - sigma) * x0). ComfyUI's choices (no input scaling for the img_to_img kinds, the noise on the
// highest sigma, the inverse scalings) were measured: Fixtures/sampling/reference.json, Fixtures/model_sampling/continuous.json, and in-process
// calls of its model sampling classes (2026-11-11).

/// <summary>What a model's output means.</summary>
public enum PredictionType
{
    /// <summary>The noise (Stable Diffusion 1.x / 2.x base, SDXL).</summary>
    Eps,
    /// <summary>The velocity v (SD 2.x 768, some fine-tunes).</summary>
    VPrediction,
    /// <summary>The EDM parameterisation (Karras et al.) with the model's sigma_data.</summary>
    Edm,
    /// <summary>The clean picture itself.</summary>
    X0,
    /// <summary>Image-to-image models: the input is not scaled and the output is the result.</summary>
    ImgToImg,
    /// <summary>Image-to-image flow models: as <see cref="ImgToImg"/>, the latent is not noised.</summary>
    ImgToImgFlow,
    /// <summary>Latent consistency models: an eps estimate mixed with the input by the LCM boundary conditions.</summary>
    Lcm,
    /// <summary>Rectified flow (SD3, Flux): the velocity from noise to data, with x_sigma = sigma * noise + (1 - sigma) * x0.</summary>
    Const,
}

/// <summary>A model's sampling: its noise levels and how its input and output relate to the noisy latent at a sigma.</summary>
public sealed class ModelSampling(ModelSigmas sigmas, PredictionType prediction = PredictionType.Eps, double sigmaData = 1.0)
{
    public ModelSigmas Sigmas { get; } = sigmas;
    public PredictionType Prediction { get; } = prediction;
    public double SigmaData { get; } = sigmaData;

    public float SigmaMin => Sigmas.SigmaMin;
    public float SigmaMax => Sigmas.SigmaMax;

    /// <summary>Flow models, whose latent at a sigma is a blend of noise and picture rather than picture plus noise.</summary>
    public bool IsConst => Prediction is PredictionType.Const or PredictionType.ImgToImgFlow;

    public float Timestep(double sigma) => Sigmas.Timestep(sigma);

    public double PercentToSigma(double percent) => Sigmas.PercentToSigma(percent);

    /// <summary>Whether a run starting at <paramref name="sigma"/> starts from pure noise: at (within 1e-5, relative) or above the model's highest sigma.</summary>
    public bool IsMaxDenoise(double sigma) => sigma > SigmaMax || Math.Abs(sigma - SigmaMax) <= 1e-5 * Math.Max(Math.Abs(sigma), Math.Abs(SigmaMax));

    /// <summary>The model's input at a sigma: x / sqrt(sigma^2 + sigma_data^2), or x itself for flow and image-to-image models.</summary>
    public float[] CalculateInput(double sigma, float[] x)
    {
        if (Prediction is PredictionType.ImgToImg or PredictionType.ImgToImgFlow or PredictionType.Const) return (float[])x.Clone();
        float s = (float)sigma, d = (float)SigmaData, scale = MathF.Sqrt(s * s + d * d);
        var r = new float[x.Length];
        for (int i = 0; i < r.Length; i++) r[i] = x[i] / scale;
        return r;
    }

    /// <summary>The estimate of the clean latent from the model's output at a sigma.</summary>
    public float[] CalculateDenoised(double sigma, float[] output, float[] x)
    {
        float s = (float)sigma, d = (float)SigmaData;
        var r = new float[x.Length];
        switch (Prediction)
        {
            case PredictionType.Eps:
            case PredictionType.Const:
                for (int i = 0; i < r.Length; i++) r[i] = x[i] - output[i] * s;
                break;
            case PredictionType.VPrediction:
            case PredictionType.Edm:
            {
                float skip = d * d / (s * s + d * d), outScale = s * d / MathF.Sqrt(s * s + d * d);
                float sign = Prediction == PredictionType.Edm ? 1f : -1f;
                for (int i = 0; i < r.Length; i++) r[i] = x[i] * skip + sign * output[i] * outScale;
                break;
            }
            case PredictionType.X0:
            case PredictionType.ImgToImg:
            case PredictionType.ImgToImgFlow:
                Array.Copy(output, r, r.Length);
                break;
            case PredictionType.Lcm:
            {
                float t = (float)Timestep(sigma) * 10f, cSkip = 0.25f / (t * t + 0.25f), cOut = t / MathF.Sqrt(t * t + 0.25f);
                for (int i = 0; i < r.Length; i++) r[i] = (x[i] - output[i] * s) * cOut + x[i] * cSkip;
                break;
            }
            default: throw new NotSupportedException($"Prediction type {Prediction}.");
        }
        return r;
    }

    /// <summary>
    /// The noisy latent a run starts from: latent + noise * sigma (from the top: noise * sqrt(1 + sigma^2)); flow models blend
    /// sigma * noise + (1 - sigma) * latent; image-to-image flow models start from the latent itself (measured).
    /// </summary>
    public float[] NoiseScaling(double sigma, float[] noise, float[] latent, bool maxDenoise = false)
    {
        float s = (float)sigma;
        var r = new float[latent.Length];
        switch (Prediction)
        {
            case PredictionType.ImgToImgFlow:
                Array.Copy(latent, r, r.Length);
                break;
            case PredictionType.Const:
                for (int i = 0; i < r.Length; i++) r[i] = s * noise[i] + (1f - s) * latent[i];
                break;
            default:
            {
                float scale = maxDenoise ? MathF.Sqrt(1f + s * s) : s;
                for (int i = 0; i < r.Length; i++) r[i] = noise[i] * scale + latent[i];
                break;
            }
        }
        return r;
    }

    /// <summary>A factor on the noise the samplers add between steps (1 for every model SpawnGen runs).</summary>
    public double NoiseScale { get; init; } = 1.0;

    /// <summary>
    /// The result of a run turned back into a latent (measured): unchanged for most models, latent / (1 - sigma) for flow models, 1 - latent for
    /// image-to-image flow models.
    /// </summary>
    public float[] InverseNoiseScaling(double sigma, float[] latent)
    {
        var r = new float[latent.Length];
        switch (Prediction)
        {
            case PredictionType.Const:
                for (int i = 0; i < r.Length; i++) r[i] = latent[i] / (1f - (float)sigma);
                break;
            case PredictionType.ImgToImgFlow:
                for (int i = 0; i < r.Length; i++) r[i] = 1f - latent[i];
                break;
            default:
                Array.Copy(latent, r, r.Length);
                break;
        }
        return r;
    }
}
