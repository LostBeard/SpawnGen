using ILGPU.Runtime;
using SpawnDev.ILGPU.ML;
using SpawnDev.ILGPU.ML.Graph;
using SpawnDev.ILGPU.ML.Preprocessing;
using SpawnDev.ILGPU.ML.Tensors;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

[Node(Category = "model/loaders", DisplayName = "Load Checkpoint", OutputNames = ["MODEL", "CLIP", "VAE"])]
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
        [Input(Multiline = true)] string text, ClipHandle clip, NodeContext context)
    {
        var e = await EncodeAsync(clip, text, text, context.CancellationToken);
        return new Conditioning([new CondEntry(e.Context, e.Shape) { Pooled = e.Pooled }]);
    }

    /// <summary>
    /// Encodes a prompt with the checkpoint's CLIP at the handle's layer. <paramref name="textG"/> is SDXL's second tower's prompt (the same text for
    /// CLIP Text Encode). Weighted prompts move each weighted token's row away from the empty prompt's by its weight, in each tower's columns.
    /// </summary>
    internal static async Task<TextEncoding> EncodeAsync(ClipHandle clip, string textL, string textG, CancellationToken cancellationToken)
    {
        var ckpt = clip.Checkpoint;
        bool two = ckpt.Family == SpawnDev.ILGPU.ML.StableDiffusion.SdFamily.SDXL;
        var (idsL, weightsL) = Tokens(ckpt, textL);
        var (idsG, weightsG) = two ? Tokens(ckpt, textG) : (idsL, weightsL);
        var z = await ckpt.EncodeAsync(idsL, two ? idsG : null, clip.Layer, cancellationToken);
        bool weighted = weightsL.Any(w => w != 1f) || weightsG.Any(w => w != 1f);
        if (!weighted) return z;
        var empty = (await ckpt.EmptyPromptAsync(clip.Layer)).Context;
        var context = (float[])z.Context.Clone();
        int width = z.Shape[^1], first = ckpt.FirstTowerWidth;
        PromptWeights.Apply(context, empty, weightsL, width, 0, first);
        if (width > first) PromptWeights.Apply(context, empty, weightsG, width, first, width - first);
        return z with { Context = context };
    }

    /// <summary>A prompt's token ids in CLIP's layout and a weight per position (all 1 for a prompt without emphasis).</summary>
    static (int[] Ids, float[] Weights) Tokens(Checkpoint ckpt, string text)
    {
        if (!PromptWeights.HasEmphasis(text))
        {
            var ids = ckpt.Tokenizer.EncodeCLIP(text, Checkpoint.TokenCount);
            var ones = new float[ids.Length];
            Array.Fill(ones, 1f);
            return (ids, ones);
        }
        return PromptWeights.Tokens(text, piece => PlainTokens(ckpt, piece), Checkpoint.TokenCount);
    }

    /// <summary>The tokens of a piece of text without the start and end markers or padding.</summary>
    static int[] PlainTokens(Checkpoint ckpt, string piece)
    {
        var ids = ckpt.Tokenizer.EncodeCLIP(piece, Math.Max(8, piece.Length + 3));
        var tokens = new List<int>();
        for (int i = 1; i < ids.Length && ids[i] != 49407; i++) tokens.Add(ids[i]);
        return [.. tokens];
    }
}

/// <summary>
/// What fills a CLIP prompt after its end token. ComfyUI pads SD 1.x's CLIP-L with the end token itself (pad_with_end) but SD 2.x's OpenCLIP-H (SD 2.1, SD-Turbo)
/// with token 0 ("!"); the UNet attends to all 77 rows, so the padding changes every prediction, most of all the empty prompt's (the negative of a CFG++ sampler).
/// </summary>
public static class ClipPadding
{
    public const int End = 49407;

    /// <summary>The pad token of a text encoder by its width: 1024 (OpenCLIP-H) pads with 0, 768 (CLIP-L) with the end token.</summary>
    public static int ForTextWidth(int textWidth) => textWidth == 1024 ? 0 : End;

    /// <summary>The ids with everything after the first end token set to <paramref name="pad"/> (the tokenizer pads with the end token).</summary>
    public static int[] Pad(int[] ids, int pad)
    {
        int first = Array.IndexOf(ids, End);
        if (pad == End || first < 0) return ids;
        var r = (int[])ids.Clone();
        for (int i = first + 1; i < r.Length; i++) r[i] = pad;
        return r;
    }
}

[Node(Category = "model/latent", DisplayName = "VAE Decode")]
public sealed class VAEDecode
{
    public async Task<ImageBatch> ExecuteAsync(LatentBatch samples, VaeHandle vae, NodeContext context, IServiceProvider services)
    {
        var acc = (await GpuHelpers.RuntimeAsync(services)).Accelerator;
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
                var latent = samples.Data[(b * perLatent)..((b + 1) * perLatent)]; // the VAE's own values, as a LATENT holds them
                var chw = await vae.Checkpoint.DecodeAsync(latent, h, w, context.CancellationToken);

                // NCHW in [-1, 1] -> NHWC in [0, 1].
                int plane = imgH * imgW;
                for (int p = 0; p < plane; p++)
                    for (int c = 0; c < 3; c++)
                        images.Data[(b * plane + p) * 3 + c] = Math.Clamp((chw[c * plane + p] + 1f) * 0.5f, 0f, 1f);
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

            var z = await vae.Checkpoint.EncodeImageAsync(chw, h, w, context.CancellationToken);
            Array.Copy(z, 0, latents.Data, b * perLatent, perLatent); // a LATENT holds the VAE's values; the sampler scales them
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
