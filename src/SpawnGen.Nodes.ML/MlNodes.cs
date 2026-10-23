using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

[Node(Category = "model/loaders", DisplayName = "Load Checkpoint")]
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

[Node(Category = "model/conditioning", DisplayName = "CLIP Text Encode (Prompt)")]
public sealed class CLIPTextEncode
{
    /// <summary>
    /// Encodes the prompt. Emphasis works as in ComfyUI: <c>(text:1.3)</c>, <c>(text)</c> = 1.1, nested brackets multiply, <c>(</c> and <c>)</c> are literal brackets.
    /// </summary>
    public async Task<Conditioning> ExecuteAsync(
        [Input(Multiline = true)] string text, ClipHandle clip, IServiceProvider services)
    {
        var acc = (await GpuHelpers.RuntimeAsync(services)).Accelerator;
        var ckpt = clip.Checkpoint;
        if (!PromptWeights.HasEmphasis(text))
        {
            var ids = ckpt.Tokenizer.EncodeCLIP(text, Checkpoint.TokenCount);
            var (plain, shape) = await EncodeAsync(acc, ckpt, ids);
            return new Conditioning(plain, shape);
        }

        // Weighted prompt: tokenise each piece, encode the lot, then move each weighted token's row away from the empty prompt's by its weight.
        var (weightedIds, weights) = PromptWeights.Tokens(text, piece => PlainTokens(ckpt, piece), Checkpoint.TokenCount);
        var (z, zShape) = await EncodeAsync(acc, ckpt, weightedIds);
        var empty = await ckpt.EmptyPromptEmbeddingsAsync(async () => (await EncodeAsync(acc, ckpt, ckpt.Tokenizer.EncodeCLIP("", Checkpoint.TokenCount))).Embeddings);
        PromptWeights.Apply(z, empty, weights, zShape[^1]);
        return new Conditioning(z, zShape);
    }

    /// <summary>The tokens of a piece of text without the start and end markers or padding.</summary>
    static int[] PlainTokens(Checkpoint ckpt, string piece)
    {
        var ids = ckpt.Tokenizer.EncodeCLIP(piece, Math.Max(8, piece.Length + 3));
        var tokens = new List<int>();
        for (int i = 1; i < ids.Length && ids[i] != 49407; i++) tokens.Add(ids[i]);
        return [.. tokens];
    }

    static async Task<(float[] Embeddings, int[] Shape)> EncodeAsync(Accelerator acc, Checkpoint ckpt, int[] ids)
    {
        var (buf, tokens) = GpuHelpers.Upload(acc, ids.Select(t => (float)t).ToArray(), [1, Checkpoint.TokenCount]);
        using (buf)
        {
            var outputs = await ckpt.TextEncoder.RunAsync(new Dictionary<string, Tensor>
            {
                [ckpt.TextEncoder.InputNames[0]] = tokens,
            });
            var embeddings = outputs[ckpt.TextEncoder.OutputNames[0]];
            int count = embeddings.Shape.Aggregate(1, (x, y) => x * y);
            return (await GpuHelpers.ReadAsync(acc, embeddings, count), embeddings.Shape.ToArray());
        }
    }
}

[Node(Category = "model/latent", DisplayName = "VAE Decode")]
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

[Node(Category = "model/latent", DisplayName = "VAE Encode")]
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

/// <summary>
/// ComfyUI's tiled decode. The library's VAE already splits large pictures into overlapping tiles whose result matches the untiled decode exactly, so the
/// tile settings are accepted for workflow compatibility and the decode is the same one VAE Decode does.
/// </summary>
[Node(Category = "model/latent", DisplayName = "VAE Decode (Tiled)")]
public sealed class VAEDecodeTiled
{
    public Task<ImageBatch> ExecuteAsync(LatentBatch samples, VaeHandle vae, NodeContext context, IServiceProvider services,
        [Input(Min = 64, Max = 4096, Step = 32)] int tileSize = 512, [Input(Min = 0, Max = 4096, Step = 32)] int overlap = 64,
        [Input(Min = 8, Max = 4096, Step = 4)] int temporalSize = 64, [Input(Min = 4, Max = 4096, Step = 4)] int temporalOverlap = 8) =>
        new VAEDecode().ExecuteAsync(samples, vae, context, services);
}

/// <summary>ComfyUI's tiled encode; see <see cref="VAEDecodeTiled"/> for why the tile settings do not change the work.</summary>
[Node(Category = "model/latent", DisplayName = "VAE Encode (Tiled)")]
public sealed class VAEEncodeTiled
{
    public Task<LatentBatch> ExecuteAsync(ImageBatch pixels, VaeHandle vae, NodeContext context, IServiceProvider services,
        [Input(Min = 64, Max = 4096, Step = 64)] int tileSize = 512, [Input(Min = 0, Max = 4096, Step = 32)] int overlap = 64,
        [Input(Min = 8, Max = 4096, Step = 4)] int temporalSize = 64, [Input(Min = 4, Max = 4096, Step = 4)] int temporalOverlap = 8) =>
        new VAEEncode().ExecuteAsync(pixels, vae, context, services);
}

[Node(Category = "model/latent", DisplayName = "VAE Encode (for Inpainting)")]
public sealed class VAEEncodeForInpaint
{
    /// <summary>
    /// Encodes a picture for inpainting: the area to repaint (<c>mask</c> = 1, grown by <c>grow_mask_by</c> pixels so the seam is repainted too) is blanked to
    /// grey before encoding, and the latent carries that area as its noise mask so the sampler only changes it. Use denoise 1.0 on the sampler.
    /// </summary>
    public async Task<LatentBatch> ExecuteAsync(ImageBatch pixels, VaeHandle vae, MaskBatch mask, NodeContext context, IServiceProvider services,
        [Input(Min = 0, Max = 64)] int growMaskBy = 6)
    {
        int h = pixels.Height / 8 * 8, w = pixels.Width / 8 * 8;
        if (h == 0 || w == 0) throw new ArgumentException($"Image {pixels.Width}x{pixels.Height} is smaller than 8 pixels.", nameof(pixels));
        var grown = GrowAndCrop(mask, pixels.Batch, pixels.Height, pixels.Width, h, w, growMaskBy);

        var blanked = new ImageBatch(pixels.Batch, h, w, pixels.Channels);
        for (int b = 0; b < pixels.Batch; b++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    float keep = 1f - grown.Data[(b * h + y) * w + x];
                    for (int c = 0; c < pixels.Channels; c++)
                    {
                        // Where the mask is set the picture becomes grey (0.5); in between it fades, like ComfyUI.
                        float v = pixels.Data[((b * pixels.Height + y) * pixels.Width + x) * pixels.Channels + c];
                        blanked.Data[((b * h + y) * w + x) * pixels.Channels + c] = (v - 0.5f) * keep + 0.5f;
                    }
                }

        var latent = await new VAEEncode().ExecuteAsync(blanked, vae, context, services);
        for (int i = 0; i < grown.Data.Length; i++) grown.Data[i] = MathF.Round(grown.Data[i]);
        return new LatentBatch(latent.Batch, latent.Channels, latent.Height, latent.Width, latent.Data) { NoiseMask = grown };
    }

    /// <summary>The mask resized to the picture, grown by a square window of <paramref name="grow"/> pixels (sum clamped to 1, ComfyUI's convolution), cropped to the encoded size.</summary>
    internal static MaskBatch GrowAndCrop(MaskBatch mask, int batch, int imageH, int imageW, int h, int w, int grow)
    {
        var result = new MaskBatch(batch, h, w);
        for (int b = 0; b < batch; b++)
        {
            var one = new float[mask.Height * mask.Width];
            Array.Copy(mask.Data, (b % mask.Batch) * one.Length, one, 0, one.Length);
            if (mask.Width != imageW || mask.Height != imageH) one = Resample.Resize(one, mask.Width, mask.Height, 1, imageW, imageH, ResampleMethod.Bilinear);
            int pad = grow <= 1 ? 0 : (grow - 1 + 1) / 2;   // ceil((grow - 1) / 2)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    float sum = 0;
                    if (grow <= 1) sum = one[y * imageW + x];
                    else
                        for (int dy = 0; dy < grow; dy++)
                        {
                            int yy = y - pad + dy;
                            if (yy < 0 || yy >= imageH) continue;
                            for (int dx = 0; dx < grow; dx++)
                            {
                                int xx = x - pad + dx;
                                if (xx >= 0 && xx < imageW) sum += one[yy * imageW + xx];
                            }
                        }
                    result.Data[(b * h + y) * w + x] = Math.Clamp(sum, 0f, 1f);
                }
        }
        return result;
    }
}
