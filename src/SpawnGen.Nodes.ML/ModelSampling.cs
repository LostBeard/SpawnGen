namespace SpawnGen.Nodes.ML;

/// <summary>What the denoiser's output means (ComfyUI's model_sampling types that ModelSamplingDiscrete can pick).</summary>
public enum PredictionType
{
    /// <summary>The noise in the input (SD 1.x, SD-Turbo).</summary>
    Eps,
    /// <summary>The velocity v = alpha * noise - sigma * x0 (SD 2.x 768).</summary>
    VPrediction,
    /// <summary>EDM's preconditioned output (Karras et al.): denoised = c_skip input + c_out output, with sigma_data (0.5 for "edm").</summary>
    Edm,
    /// <summary>Latent consistency: an eps model whose output is blended with the input by the LCM boundary condition, on 50 distilled timesteps.</summary>
    Lcm,
    /// <summary>The clean latent itself.</summary>
    X0,
    /// <summary>The clean latent, from an unscaled input.</summary>
    ImgToImg,
    /// <summary>A flow-style image to image model: the clean latent from an unscaled input, no noise added (ComfyUI's IMG_TO_IMG_FLOW, a CONST type).</summary>
    ImgToImgFlow,
}

/// <summary>
/// How a discrete-timestep denoiser is driven, ComfyUI's <c>model_sampling</c> object: its noise levels (and which timestep each is told as), how the input is
/// scaled for it, how its output becomes a denoised estimate, and how noise is added at the start of a run. The default is SD's eps prediction; ModelSamplingDiscrete
/// swaps it.
/// </summary>
public sealed class ModelSampling
{
    public ModelSampling(ModelSigmas sigmas, PredictionType prediction = PredictionType.Eps, double sigmaData = 1.0)
    {
        Sigmas = sigmas;
        Prediction = prediction;
        SigmaData = sigmaData;
    }

    public ModelSigmas Sigmas { get; }
    public PredictionType Prediction { get; }
    /// <summary>The data's standard deviation the model was trained with (ComfyUI's sigma_data): 1, or 0.5 for EDM models.</summary>
    public double SigmaData { get; }

    /// <summary>A CONST (rectified-flow) sampling: x = (1 - sigma) x0 + sigma noise. The samplers switch to their flow forms for it.</summary>
    public bool IsConst => Prediction == PredictionType.ImgToImgFlow;

    /// <summary>The training noise scale of a flow model sampling (ComfyUI's <c>noise_scale</c>); the discrete samplings here have none, so it is 1.</summary>
    public double NoiseScale => 1.0;

    public double SigmaMin => Sigmas.SigmaMin;
    public double SigmaMax => Sigmas.SigmaMax;

    /// <summary>The sigma at which a run is the given fraction done (0 = start, 1 = end), ComfyUI's <c>percent_to_sigma</c>.</summary>
    public double PercentToSigma(double percent) => Sigmas.PercentToSigma(percent);

    /// <summary>The timestep the model is told for a noise level.</summary>
    public float Timestep(double sigma) => Sigmas.Timestep(sigma);

    /// <summary>The model's input for the sample <paramref name="x"/> at <paramref name="sigma"/> (ComfyUI's <c>calculate_input</c>: x / sqrt(sigma^2 + sigma_data^2)).</summary>
    public float[] CalculateInput(double sigma, float[] x)
    {
        if (Prediction is PredictionType.ImgToImg or PredictionType.ImgToImgFlow) return x;
        double scale = 1.0 / Math.Sqrt(sigma * sigma + SigmaData * SigmaData);
        var r = new float[x.Length];
        for (int i = 0; i < r.Length; i++) r[i] = (float)(x[i] * scale);
        return r;
    }

    /// <summary>The denoised estimate from the model's raw output (ComfyUI's <c>calculate_denoised</c>).</summary>
    public float[] CalculateDenoised(double sigma, float[] output, float[] input)
    {
        var r = new float[output.Length];
        switch (Prediction)
        {
            case PredictionType.Eps:
                for (int i = 0; i < r.Length; i++) r[i] = (float)(input[i] - output[i] * sigma);
                break;
            case PredictionType.VPrediction:
            case PredictionType.Edm:
            {
                // V: input sd^2 / (sigma^2 + sd^2) - output sigma sd / sqrt(sigma^2 + sd^2); EDM adds the output term instead of subtracting it.
                double sd = SigmaData, a = sd * sd / (sigma * sigma + sd * sd), b = sigma * sd / Math.Sqrt(sigma * sigma + sd * sd);
                if (Prediction == PredictionType.Edm) b = -b;
                for (int i = 0; i < r.Length; i++) r[i] = (float)(input[i] * a - output[i] * b);
                break;
            }
            case PredictionType.Lcm:
            {
                // LCM's boundary condition: c_skip and c_out from the (distilled) timestep scaled by 10, sigma_data 0.5.
                double t = Timestep(sigma) * 10.0, sd = 0.5;
                double cSkip = sd * sd / (t * t + sd * sd), cOut = t / Math.Sqrt(t * t + sd * sd);
                for (int i = 0; i < r.Length; i++) r[i] = (float)(cOut * (input[i] - output[i] * sigma) + cSkip * input[i]);
                break;
            }
            default:
                Array.Copy(output, r, r.Length);
                break;
        }
        return r;
    }

    /// <summary>
    /// The start of a run (ComfyUI's <c>noise_scaling</c>): the latent plus noise at <paramref name="sigma"/>; from the top of the schedule (<paramref name="maxDenoise"/>)
    /// the noise is scaled by sqrt(1 + sigma^2) so the model's scaled input has unit variance. The flow image-to-image type adds no noise.
    /// </summary>
    public float[] NoiseScaling(double sigma, float[] noise, float[] latent, bool maxDenoise = false)
    {
        if (Prediction == PredictionType.ImgToImgFlow) return (float[])latent.Clone();
        double scale = maxDenoise ? Math.Sqrt(1.0 + sigma * sigma) : sigma;
        var r = new float[noise.Length];
        for (int i = 0; i < r.Length; i++) r[i] = (float)(noise[i] * scale) + latent[i];
        return r;
    }

    /// <summary>What a run's result is turned into at the end (ComfyUI's <c>inverse_noise_scaling</c>): unchanged, except 1 - x for the flow image-to-image type.</summary>
    public float[] InverseNoiseScaling(double sigma, float[] latent)
    {
        if (Prediction != PredictionType.ImgToImgFlow) return latent;
        var r = new float[latent.Length];
        for (int i = 0; i < r.Length; i++) r[i] = 1f - latent[i];
        return r;
    }

    /// <summary>Whether a run starting at <paramref name="firstSigma"/> starts from pure noise (ComfyUI's <c>max_denoise</c>: within 1e-5 of sigma_max, or above it).</summary>
    public bool IsMaxDenoise(double firstSigma)
    {
        double max = SigmaMax;
        return Math.Abs(max - firstSigma) <= 1e-5 * Math.Max(Math.Abs(max), Math.Abs(firstSigma)) || firstSigma > max;
    }
}
