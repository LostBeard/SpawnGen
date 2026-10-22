using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

[Node(Category = "loaders", DisplayName = "Load Checkpoint")]
public sealed class CheckpointLoaderSimple
{
    public static string[] CheckpointNames() => MlRuntime.KnownCheckpoints.Keys.ToArray();

    public async Task<(ModelHandle Model, ClipHandle Clip, VaeHandle Vae)> ExecuteAsync(
        [Input(OptionsProvider = nameof(CheckpointNames))] string ckptName,
        NodeContext context, IServiceProvider services)
    {
        var runtime = await GpuHelpers.RuntimeAsync(services);
        context.Status($"Loading {ckptName}");
        // The loader reports every percent of every sub-model; one message per 10% is plenty for a status line.
        string? lastStage = null;
        int lastBucket = -1;
        var ckpt = await runtime.GetCheckpointAsync(ckptName, (stage, pct) =>
        {
            int bucket = pct / 10;
            if (stage == lastStage && bucket == lastBucket) return;
            (lastStage, lastBucket) = (stage, bucket);
            context.Status($"Loading {ckptName}: {stage} {pct}%");
        });
        return (new ModelHandle(ckpt), new ClipHandle(ckpt), new VaeHandle(ckpt));
    }
}

[Node(Category = "conditioning", DisplayName = "CLIP Text Encode (Prompt)")]
public sealed class CLIPTextEncode
{
    public async Task<Conditioning> ExecuteAsync(
        [Input(Multiline = true)] string text, ClipHandle clip, IServiceProvider services)
    {
        var acc = (await GpuHelpers.RuntimeAsync(services)).Accelerator;
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
    public static string[] SamplerNames() => Samplers.Names;
    public static string[] SchedulerNames() => ["normal"];

    // Socket inputs first, then the widgets in ComfyUI's order with its defaults (so a new node is runnable as dropped on the canvas).
    public async Task<LatentBatch> ExecuteAsync(
        ModelHandle model, Conditioning positive, Conditioning negative, LatentBatch latentImage,
        NodeContext context, IServiceProvider services,
        long seed = 0,
        [Input(Min = 1, Max = 10000)] int steps = 20,
        [Input(Min = 0, Max = 100, Step = 0.1)] float cfg = 8f,
        [Input(OptionsProvider = nameof(SamplerNames))] string samplerName = "euler",
        [Input(OptionsProvider = nameof(SchedulerNames))] string scheduler = "normal",
        [Input(Min = 0, Max = 1, Step = 0.01)] float denoise = 1f)
    {
        if (denoise <= 0f) return latentImage; // nothing to denoise: the input passes through
        if (latentImage.Channels != Checkpoint.LatentChannels)
            throw new ArgumentException($"Latent has {latentImage.Channels} channels; this checkpoint expects {Checkpoint.LatentChannels}.", nameof(latentImage));

        var acc = (await GpuHelpers.RuntimeAsync(services)).Accelerator;
        var ckpt = model.Checkpoint;
        var unet = ckpt.Unet;
        int h = latentImage.Height, w = latentImage.Width, perImage = Checkpoint.LatentChannels * h * w;

        // Trailing timesteps and their sigmas; sigmas[^1] = 0. SD-Turbo's single step starts from full noise.
        var (timesteps, sigmas) = SigmaSchedule.Build(steps, denoise, ckpt.AlphasCumprod);
        steps = timesteps.Length;
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

                // Epsilon for the sample at a step: scale_model_input (x / sqrt(sigma^2 + 1), the UNet is trained on ~unit variance), then the
                // UNet, plus classifier-free guidance when cfg > 1.
                async Task<float[]> Epsilon(float[] sample, int step)
                {
                    float scale = 1f / MathF.Sqrt(sigmas[step] * sigmas[step] + 1f);
                    var scaled = new float[perImage];
                    for (int i = 0; i < perImage; i++) scaled[i] = sample[i] * scale;

                    var eps = await PredictAsync(acc, unet, scaled, shape, timesteps[step], cond);
                    if (useCfg)
                    {
                        var epsUncond = await PredictAsync(acc, unet, scaled, shape, timesteps[step], uncond);
                        for (int i = 0; i < perImage; i++) eps[i] = epsUncond[i] + cfg * (eps[i] - epsUncond[i]);
                    }
                    return eps;
                }

                int item = b;
                x = await Samplers.RunAsync(samplerName, x, sigmas, Epsilon, unchecked((int)(seed + b) ^ 0x5EED),
                    Checkpoint.LatentChannels, h, w, done => context.Progress(item * steps + done, latentImage.Batch * steps),
                    context.CancellationToken);
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
        var acc = (await GpuHelpers.RuntimeAsync(services)).Accelerator;
        var session = vae.Checkpoint.Vae;
        int h = samples.Height, w = samples.Width, perLatent = samples.Channels * h * w;
        int imgH = h * 8, imgW = w * 8, perImage = 3 * imgH * imgW;
        var images = new ImageBatch(samples.Batch, imgH, imgW, 3);

        // The browser GPU process has a memory budget the desktop does not: the library reclaims dead buffers during this decode there so its
        // peak follows the working set (ImageGenerationPipeline does the same). Desktop backends have room to spare and are left alone.
        bool browser = acc.AcceleratorType is AcceleratorType.WebGPU or AcceleratorType.WebGL or AcceleratorType.Wasm;
        long savedCap = GraphExecutor.MaxPendingReleaseBytes;
        int savedReclaim = BufferPool.ForceReclaimEveryNRents;
        if (browser)
        {
            GraphExecutor.MaxPendingReleaseBytes = 64L * 1024 * 1024;
            BufferPool.ForceReclaimEveryNRents = 8;
        }
        try
        {
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
        }
        finally
        {
            GraphExecutor.MaxPendingReleaseBytes = savedCap;
            BufferPool.ForceReclaimEveryNRents = savedReclaim;
        }
        return images;
    }
}

[Node(Category = "latent", DisplayName = "VAE Encode")]
public sealed class VAEEncode
{
    public async Task<LatentBatch> ExecuteAsync(ImageBatch pixels, VaeHandle vae, NodeContext context, IServiceProvider services)
    {
        var acc = (await GpuHelpers.RuntimeAsync(services)).Accelerator;
        var session = await vae.Checkpoint.GetVaeEncoderAsync();

        // The VAE downsamples by 8: crop to a multiple of 8 like ComfyUI does.
        int h = pixels.Height / 8 * 8, w = pixels.Width / 8 * 8;
        if (h == 0 || w == 0) throw new ArgumentException($"Image {pixels.Width}x{pixels.Height} is smaller than 8 pixels.", nameof(pixels));
        int lh = h / 8, lw = w / 8, perLatent = Checkpoint.LatentChannels * lh * lw;
        var latents = new LatentBatch(pixels.Batch, Checkpoint.LatentChannels, lh, lw);

        for (int b = 0; b < pixels.Batch; b++)
        {
            context.CancellationToken.ThrowIfCancellationRequested();
            // NHWC in [0, 1] -> NCHW in [-1, 1]
            var chw = new float[3 * h * w];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    for (int c = 0; c < 3; c++)
                        chw[(c * h + y) * w + x] = pixels.Data[(((b * pixels.Height) + y) * pixels.Width + x) * pixels.Channels + c] * 2f - 1f;

            var (buf, tensor) = GpuHelpers.Upload(acc, chw, [1, 3, h, w]);
            using (buf)
            {
                var outputs = await session.RunAsync(new Dictionary<string, Tensor> { [session.InputNames[0]] = tensor });
                var z = await GpuHelpers.ReadAsync(acc, outputs[session.OutputNames[0]], perLatent);
                for (int i = 0; i < perLatent; i++) latents.Data[b * perLatent + i] = z[i] * Checkpoint.VaeScale;
            }
            context.Progress(b + 1, pixels.Batch);
        }
        return latents;
    }
}
