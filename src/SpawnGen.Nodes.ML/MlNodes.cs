using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

[Node(Category = "loaders", DisplayName = "Load Checkpoint")]
public sealed class CheckpointLoader
{
    public static string[] CheckpointNames() => MlRuntime.KnownCheckpoints.Keys.ToArray();

    public async Task<(ModelHandle Model, ClipHandle Clip, VaeHandle Vae)> ExecuteAsync(
        [Input(OptionsProvider = nameof(CheckpointNames))] string ckptName,
        NodeContext context, IServiceProvider services)
    {
        var runtime = GpuHelpers.Runtime(services);
        context.Status($"Loading {ckptName}");
        var ckpt = await runtime.GetCheckpointAsync(ckptName, (stage, pct) => context.Status($"Loading {ckptName}: {stage} {pct}%"));
        return (new ModelHandle(ckpt), new ClipHandle(ckpt), new VaeHandle(ckpt));
    }
}

[Node(Category = "conditioning", DisplayName = "CLIP Text Encode (Prompt)")]
public sealed class CLIPTextEncode
{
    public async Task<Conditioning> ExecuteAsync(
        [Input(Multiline = true)] string text, ClipHandle clip, IServiceProvider services)
    {
        var acc = GpuHelpers.Runtime(services).Accelerator;
        var ckpt = clip.Checkpoint;
        var ids = ckpt.Tokenizer.EncodeCLIP(text, Checkpoint.TokenCount);
        var (buf, tokens) = GpuHelpers.Upload(acc, ids.Select(t => (float)t).ToArray(), [1, Checkpoint.TokenCount]);
        using (buf)
        {
            var outputs = await ckpt.TextEncoder.RunAsync(new Dictionary<string, Tensor>
            {
                [ckpt.TextEncoder.InputNames[0]] = tokens,
            });
            var embeddings = outputs[ckpt.TextEncoder.OutputNames[0]];
            int count = embeddings.Shape.Aggregate(1, (a, b) => a * b);
            return new Conditioning(await GpuHelpers.ReadAsync(acc, embeddings, count), embeddings.Shape.ToArray());
        }
    }
}

[Node(Category = "sampling", DisplayName = "KSampler")]
public sealed class KSampler
{
    public static string[] SamplerNames() => ["euler"];
    public static string[] SchedulerNames() => ["normal"];

    public async Task<LatentBatch> ExecuteAsync(
        ModelHandle model,
        long seed,
        [Input(Min = 1, Max = 10000)] int steps,
        [Input(Min = 0, Max = 100, Step = 0.1)] float cfg,
        [Input(OptionsProvider = nameof(SamplerNames))] string samplerName,
        [Input(OptionsProvider = nameof(SchedulerNames))] string scheduler,
        Conditioning positive, Conditioning negative, LatentBatch latentImage,
        [Input(Min = 0, Max = 1, Step = 0.01)] float denoise,
        NodeContext context, IServiceProvider services)
    {
        if (denoise < 1f) throw new NotSupportedException("KSampler denoise < 1 (img2img) is not implemented yet.");
        if (latentImage.Channels != Checkpoint.LatentChannels)
            throw new ArgumentException($"Latent has {latentImage.Channels} channels; this checkpoint expects {Checkpoint.LatentChannels}.", nameof(latentImage));

        var acc = GpuHelpers.Runtime(services).Accelerator;
        var ckpt = model.Checkpoint;
        var unet = ckpt.Unet;
        int h = latentImage.Height, w = latentImage.Width, perImage = Checkpoint.LatentChannels * h * w;

        // Trailing timesteps and their sigmas; sigmas[^1] = 0. SD-Turbo's single step starts from full noise.
        var timesteps = DiffusionScheduler.GetTimesteps(steps);
        var sigmas = DiffusionScheduler.TimestepsToSigmas(timesteps, ckpt.AlphasCumprod);
        bool useCfg = cfg > 1f;

        var result = new LatentBatch(latentImage.Batch, latentImage.Channels, h, w);
        var shape = new[] { 1, Checkpoint.LatentChannels, h, w };
        var (condBuf, cond) = GpuHelpers.Upload(acc, positive.Embeddings, positive.Shape);
        var (uncondBuf, uncond) = GpuHelpers.Upload(acc, negative.Embeddings, negative.Shape);
        using (condBuf)
        using (uncondBuf)
        {
            for (int b = 0; b < latentImage.Batch; b++)
            {
                // One seed offset per batch item keeps runs reproducible.
                var noise = DiffusionScheduler.GenerateNoise(Checkpoint.LatentChannels, h, w, unchecked((int)(seed + b)));
                var x = new float[perImage];
                for (int i = 0; i < perImage; i++)
                    x[i] = latentImage.Data[b * perImage + i] + noise[i] * sigmas[0];

                for (int step = 0; step < steps; step++)
                {
                    context.CancellationToken.ThrowIfCancellationRequested();
                    float sigma = sigmas[step];
                    float scale = 1f / MathF.Sqrt(sigma * sigma + 1f);
                    var scaled = new float[perImage];
                    for (int i = 0; i < perImage; i++) scaled[i] = x[i] * scale;

                    var eps = await PredictAsync(acc, unet, scaled, shape, timesteps[step], cond);
                    if (useCfg)
                    {
                        var epsUncond = await PredictAsync(acc, unet, scaled, shape, timesteps[step], uncond);
                        for (int i = 0; i < perImage; i++) eps[i] = epsUncond[i] + cfg * (eps[i] - epsUncond[i]);
                    }

                    float dt = sigmas[step + 1] - sigma;
                    for (int i = 0; i < perImage; i++) x[i] += eps[i] * dt;
                    context.Progress(b * steps + step + 1, latentImage.Batch * steps);
                }
                Array.Copy(x, 0, result.Data, b * perImage, perImage);
            }
        }
        return result;
    }

    static async Task<float[]> PredictAsync(Accelerator acc, InferenceSession unet, float[] sample, int[] shape,
        int timestep, Tensor conditioning)
    {
        var (sampleBuf, sampleTensor) = GpuHelpers.Upload(acc, sample, shape);
        var (tBuf, tTensor) = GpuHelpers.Upload(acc, [(float)timestep], [1]);
        using (sampleBuf)
        using (tBuf)
        {
            var outputs = await unet.RunAsync(new Dictionary<string, Tensor>
            {
                [unet.InputNames[0]] = sampleTensor,
                [unet.InputNames[1]] = tTensor,
                [unet.InputNames[2]] = conditioning,
            });
            return await GpuHelpers.ReadAsync(acc, outputs[unet.OutputNames[0]], sample.Length);
        }
    }
}

[Node(Category = "latent", DisplayName = "VAE Decode")]
public sealed class VAEDecode
{
    public async Task<ImageBatch> ExecuteAsync(LatentBatch samples, VaeHandle vae, NodeContext context, IServiceProvider services)
    {
        var acc = GpuHelpers.Runtime(services).Accelerator;
        var session = vae.Checkpoint.Vae;
        int h = samples.Height, w = samples.Width, perLatent = samples.Channels * h * w;
        int imgH = h * 8, imgW = w * 8, perImage = 3 * imgH * imgW;
        var images = new ImageBatch(samples.Batch, imgH, imgW, 3);

        for (int b = 0; b < samples.Batch; b++)
        {
            context.CancellationToken.ThrowIfCancellationRequested();
            var latent = new float[perLatent];
            for (int i = 0; i < perLatent; i++) latent[i] = samples.Data[b * perLatent + i] / Checkpoint.VaeScale;

            var (buf, tensor) = GpuHelpers.Upload(acc, latent, [1, samples.Channels, h, w]);
            using (buf)
            {
                var outputs = await session.RunAsync(new Dictionary<string, Tensor> { [session.InputNames[0]] = tensor });
                var chw = await GpuHelpers.ReadAsync(acc, outputs[session.OutputNames[0]], perImage);

                // NCHW in [-1, 1] -> NHWC in [0, 1].
                int plane = imgH * imgW;
                for (int p = 0; p < plane; p++)
                    for (int c = 0; c < 3; c++)
                        images.Data[(b * plane + p) * 3 + c] = Math.Clamp((chw[c * plane + p] + 1f) * 0.5f, 0f, 1f);
            }
            context.Progress(b + 1, samples.Batch);
        }
        return images;
    }
}
