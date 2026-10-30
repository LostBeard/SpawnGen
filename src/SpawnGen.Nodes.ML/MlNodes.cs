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
    /// Encodes a prompt with the CLIP at the handle's layer. <paramref name="textG"/> is SDXL's second tower's prompt (the same text for
    /// CLIP Text Encode). Weighted prompts move each weighted token's row away from the empty prompt's by its weight, in each tower's columns.
    /// </summary>
    internal static async Task<TextEncoding> EncodeAsync(ClipHandle clip, string textL, string textG, CancellationToken cancellationToken)
    {
        var ckpt = clip.Encoder;
        bool two = ckpt.Family == SpawnDev.ILGPU.ML.StableDiffusion.SdFamily.SDXL;
        var chunksL = PromptWeights.Chunks(textL, piece => PlainTokens(ckpt, piece), Checkpoint.TokenCount);
        var chunksG = two ? PromptWeights.Chunks(textG, piece => PlainTokens(ckpt, piece), Checkpoint.TokenCount) : chunksL;
        // SDXL's two prompts may need different numbers of chunks: the shorter one is filled up with empty chunks, as ComfyUI does.
        var emptyChunk = PromptWeights.Chunks("", piece => PlainTokens(ckpt, piece), Checkpoint.TokenCount)[0];
        while (chunksL.Count < chunksG.Count) chunksL = [.. chunksL, emptyChunk];
        while (chunksG.Count < chunksL.Count) chunksG = [.. chunksG, emptyChunk];

        await ckpt.UseWeightsAsync(clip.Weights, cancellationToken);
        bool weighted = chunksL.Concat(chunksG).Any(c => c.Weights.Any(w => w != 1f));
        float[]? empty = weighted ? (await ckpt.EmptyPromptAsync(clip.Layer, clip.Weights)).Context : null;
        var contexts = new List<float[]>();
        float[]? pooled = null;
        int width = 0;
        for (int i = 0; i < chunksL.Count; i++)
        {
            var z = await ckpt.EncodeAsync(chunksL[i].Ids, two ? chunksG[i].Ids : null, clip.Layer, cancellationToken);
            width = z.Shape[^1];
            pooled ??= z.Pooled; // SDXL's pooled embedding is the first chunk's, as in ComfyUI
            var context = z.Context;
            if (empty is not null)
            {
                // Each weighted token's row moves away from the empty prompt's row by its weight, in each tower's columns.
                context = (float[])context.Clone();
                int first = ckpt.FirstTowerWidth;
                PromptWeights.Apply(context, empty, chunksL[i].Weights, width, 0, first);
                if (width > first) PromptWeights.Apply(context, empty, chunksG[i].Weights, width, first, width - first);
            }
            contexts.Add(context);
        }
        var all = contexts.Count == 1 ? contexts[0] : [.. contexts.SelectMany(c => c)];
        return new TextEncoding(all, [1, all.Length / width, width], pooled);
    }

    /// <summary>The tokens of a piece of text without the start and end markers or padding.</summary>
    internal static int[] PlainTokens(ITextEncoderModel ckpt, string piece)
    {
        // A UTF-16 character is at most 3 bytes of UTF-8 and CLIP's BPE gives at most one token per byte: room for every token plus start and end.
        var ids = ckpt.Tokenizer.EncodeCLIP(piece, 3 * piece.Length + 3);
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
                var chw = await vae.Vae.DecodeAsync(latent, h, w, context.CancellationToken);

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
        pixels = CropForVae(pixels);
        int h = pixels.Height, w = pixels.Width;
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

            var z = await vae.Vae.EncodeImageAsync(chw, h, w, context.CancellationToken);
            Array.Copy(z, 0, latents.Data, b * perLatent, perLatent); // a LATENT holds the VAE's values; the sampler scales them
            context.Progress(b + 1, pixels.Batch);
        }
        return latents;
    }

    /// <summary>ComfyUI's vae_encode_crop_pixels: the VAE downsamples by 8, so each side is cut to a multiple of 8, the cut split between both ends (the odd
    /// pixel at the end).</summary>
    internal static ImageBatch CropForVae(ImageBatch pixels)
    {
        int h = pixels.Height / 8 * 8, w = pixels.Width / 8 * 8;
        if (h == 0 || w == 0) throw new ArgumentException($"Image {pixels.Width}x{pixels.Height} is smaller than 8 pixels.", nameof(pixels));
        if (h == pixels.Height && w == pixels.Width) return pixels;
        int top = pixels.Height % 8 / 2, left = pixels.Width % 8 / 2, c = pixels.Channels;
        var cropped = new ImageBatch(pixels.Batch, h, w, c);
        for (int b = 0; b < pixels.Batch; b++)
            for (int y = 0; y < h; y++)
                Array.Copy(pixels.Data, ((b * pixels.Height + top + y) * pixels.Width + left) * c, cropped.Data, (b * h + y) * w * c, w * c);
        return cropped;
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
    /// Encodes a picture for inpainting: the area to repaint (<c>mask</c> rounded to 0 / 1) is blanked to grey before encoding, and the latent carries that
    /// area, grown by <c>grow_mask_by</c> pixels so the seam is repainted too, as its noise mask so the sampler only changes it. Use denoise 1.0 on the sampler.
    /// </summary>
    public async Task<LatentBatch> ExecuteAsync(ImageBatch pixels, VaeHandle vae, MaskBatch mask, NodeContext context, IServiceProvider services,
        [Input(Min = 0, Max = 64)] int growMaskBy = 6)
    {
        var (greyed, noiseMask) = Prepare(pixels, mask, growMaskBy);
        var latent = await new VAEEncode().ExecuteAsync(greyed, vae, context, services);
        return new LatentBatch(latent.Batch, latent.Channels, latent.Height, latent.Width, latent.Data) { NoiseMask = noiseMask };
    }

    /// <summary>ComfyUI's steps before the VAE: the greyed picture to encode and the noise mask (the rounded mask grown by a square window of
    /// <paramref name="grow"/> pixels inside the cropped picture, clamped to 1, then rounded; with no growth the mask is only rounded).</summary>
    internal static (ImageBatch Greyed, MaskBatch NoiseMask) Prepare(ImageBatch pixels, MaskBatch mask, int grow)
    {
        var (_, cropped, greyed) = InpaintPixels.Prepare(pixels, mask);
        int h = cropped.Height, w = cropped.Width, plane = h * w;
        var noise = new MaskBatch(cropped.Batch, h, w);
        int pad = (grow - 1 + 1) / 2; // ceil((grow - 1) / 2), the convolution's padding
        for (int b = 0; b < cropped.Batch; b++)
        {
            var one = cropped.Data.AsSpan(b * plane, plane);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    float sum = 0;
                    if (grow == 0) sum = one[y * w + x];
                    else
                        for (int dy = 0; dy < grow; dy++)
                        {
                            int yy = y - pad + dy;
                            if (yy < 0 || yy >= h) continue;
                            for (int dx = 0; dx < grow; dx++)
                            {
                                int xx = x - pad + dx;
                                if (xx >= 0 && xx < w) sum += MathF.Round(one[yy * w + xx]);
                            }
                        }
                    noise.Data[b * plane + y * w + x] = MathF.Round(Math.Clamp(sum, 0f, 1f));
                }
        }
        return (greyed, noise);
    }
}

/// <summary>
/// The steps ComfyUI's inpainting encoders share: the mask resized to the picture (bilinear), picture and mask cropped as the VAE crops (to a multiple of 8,
/// the cut split between both ends), and the picture greyed (0.5) where the rounded mask is set.
/// </summary>
internal static class InpaintPixels
{
    public static (ImageBatch Pixels, MaskBatch Mask, ImageBatch Greyed) Prepare(ImageBatch pixels, MaskBatch mask)
    {
        var picture = VAEEncode.CropForVae(pixels);
        int h = picture.Height, w = picture.Width, top = pixels.Height % 8 / 2, left = pixels.Width % 8 / 2, plane = pixels.Height * pixels.Width;
        var cropped = new MaskBatch(mask.Batch, h, w);
        for (int b = 0; b < mask.Batch; b++)
        {
            var one = mask.Data.AsSpan(b * mask.Height * mask.Width, mask.Height * mask.Width).ToArray();
            if (mask.Width != pixels.Width || mask.Height != pixels.Height)
                one = Resample.Resize(one, mask.Width, mask.Height, 1, pixels.Width, pixels.Height, ResampleMethod.Bilinear);
            for (int y = 0; y < h; y++) Array.Copy(one, (top + y) * pixels.Width + left, cropped.Data, (b * h + y) * w, w);
        }

        var greyed = new ImageBatch(picture.Batch, h, w, picture.Channels);
        int c = picture.Channels;
        for (int b = 0; b < picture.Batch; b++)
            for (int i = 0; i < h * w; i++)
            {
                float keep = 1f - MathF.Round(cropped.Data[(b % mask.Batch) * h * w + i]);
                for (int k = 0; k < c; k++)
                {
                    float v = picture.Data[(b * h * w + i) * c + k];
                    // ComfyUI greys the three colour channels: (v - 0.5) * keep + 0.5 in float32; an alpha channel is left as it is.
                    greyed.Data[(b * h * w + i) * c + k] = k < 3 ? (v - 0.5f) * keep + 0.5f : v;
                }
            }
        return (picture, cropped, greyed);
    }
}

[Node(Category = "model/conditioning", DisplayName = "InpaintModelConditioning", OutputNames = ["positive", "negative", "latent"])]
public sealed class InpaintModelConditioning
{
    /// <summary>
    /// Prepares an inpainting run: the latent is the whole picture encoded, with the mask as its noise mask (when <c>noise_mask</c> is on), and both
    /// conditionings carry the picture with the masked area greyed (encoded) plus the mask, which inpainting models take as extra input channels. Other
    /// models ignore those two, as in ComfyUI, and the run is then an img2img confined to the mask.
    /// </summary>
    public async Task<(Conditioning Positive, Conditioning Negative, LatentBatch Latent)> ExecuteAsync(Conditioning positive, Conditioning negative, VaeHandle vae,
        ImageBatch pixels, MaskBatch mask, NodeContext context, IServiceProvider services,
        [Input(Default = true, Tooltip = "Add a noise mask to the latent so sampling will only happen within the mask. Might improve results or completely break things depending on the model.")]
        bool noiseMask)
    {
        var (_, cropped, greyed) = InpaintPixels.Prepare(pixels, mask);
        var concat = await new VAEEncode().ExecuteAsync(greyed, vae, context, services);
        var latent = await new VAEEncode().ExecuteAsync(pixels, vae, context, services);
        if (noiseMask) latent = new LatentBatch(latent.Batch, latent.Channels, latent.Height, latent.Width, latent.Data) { NoiseMask = cropped };
        Conditioning Set(Conditioning c) => c.Map(e => e with { ConcatLatent = concat, ConcatMask = cropped });
        return (Set(positive), Set(negative), latent);
    }
}
