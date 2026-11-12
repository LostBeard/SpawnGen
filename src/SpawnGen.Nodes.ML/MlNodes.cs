using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries and SpawnGen's tests: ClipComfyParityTests (ComfyUI's token chunks
// and conditioning), ComfyUIRenderParityTests (padding), InpaintParityTests (ComfyUI's cropping, greying, mask resizing and growing, measured with
// a stand-in VAE). The models run on SpawnDev.ILGPU.ML; CLIP's token layout is OpenAI's (start 49406, end 49407).

/// <summary>How a CLIP tower pads a chunk after its end token: CLIP-L (768 wide) with more end tokens, the OpenCLIP towers with 0.</summary>
public static class ClipPadding
{
    public const int End = PromptWeights.EndToken;

    public static int ForTextWidth(int textWidth) => textWidth == 768 ? End : 0;

    /// <summary>A copy of the ids with everything after the first end token replaced by <paramref name="pad"/>.</summary>
    public static int[] Pad(int[] ids, int pad)
    {
        var r = (int[])ids.Clone();
        int end = Array.IndexOf(r, End);
        if (end >= 0) for (int i = end + 1; i < r.Length; i++) r[i] = pad;
        return r;
    }
}

[Node("CheckpointLoaderSimple", Category = "model/loaders", DisplayName = "Load Checkpoint", OutputNames = ["MODEL", "CLIP", "VAE"])]
public sealed class CheckpointLoaderSimple
{
    public static string[] CheckpointNames() => [.. MlRuntime.KnownCheckpoints.Keys];

    public async Task<(ModelHandle Model, ClipHandle Clip, VaeHandle Vae)> ExecuteAsync([Input(OptionsProvider = nameof(CheckpointNames))] string ckptName,
        NodeContext context, IServiceProvider services)
    {
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var checkpoint = await runtime.GetCheckpointAsync(ckptName, (stage, percent) => context.Progress(percent, 100, $"Loading {ckptName}: {stage}"));
        return (new ModelHandle(checkpoint), new ClipHandle(checkpoint), new VaeHandle(checkpoint));
    }
}

[Node("CLIPTextEncode", Category = "model/conditioning", DisplayName = "CLIP Text Encode (Prompt)")]
public sealed class CLIPTextEncode
{
    public async Task<Conditioning> ExecuteAsync([Input(Multiline = true)] string text, ClipHandle clip, NodeContext context)
    {
        var e = await EncodeAsync(clip, text, text, context.CancellationToken);
        return new Conditioning([new CondEntry(e.Context, e.Shape) { Pooled = e.Pooled }]);
    }

    /// <summary>A piece of prompt text as CLIP BPE ids (CLIP's own text cleaning and BPE), without the start, end and padding tokens.</summary>
    public static int[] PlainTokens(ITextEncoderModel ckpt, string piece)
    {
        var ids = ckpt.Tokenizer.EncodeCLIP(piece, piece.Length * 4 + 8, PromptWeights.EndToken);
        int start = Array.IndexOf(ids, PromptWeights.StartToken) + 1, end = Array.IndexOf(ids, PromptWeights.EndToken, start);
        return ids[start..(end < 0 ? ids.Length : end)];
    }

    /// <summary>
    /// The prompt encoded in 77-token chunks (<see cref="PromptWeights.Chunks"/>), each chunk's rows weighted against the empty prompt's
    /// (<see cref="PromptWeights.Apply(float[], float[], float[], int, int, int)"/>, per tower), the chunks' rows one after another; the pooled output is
    /// the first chunk's. SDXL encodes <paramref name="textL"/> with CLIP-L and <paramref name="textG"/> with OpenCLIP-bigG (the shorter padded with empty
    /// chunks); the other families use <paramref name="textL"/>.
    /// </summary>
    public static async Task<TextEncoding> EncodeAsync(ClipHandle clip, string textL, string textG, CancellationToken cancellationToken)
    {
        var encoder = clip.Encoder;
        await encoder.UseWeightsAsync(clip.Weights, cancellationToken);
        bool xl = encoder.Family == SpawnDev.ILGPU.ML.StableDiffusion.SdFamily.SDXL;
        int padL = ClipPadding.ForTextWidth(encoder.FirstTowerWidth);
        var chunksL = PromptWeights.Chunks(textL, piece => PlainTokens(encoder, piece), Checkpoint.TokenCount);
        var chunksG = xl ? PromptWeights.Chunks(textG, piece => PlainTokens(encoder, piece), Checkpoint.TokenCount) : null;
        int count = Math.Max(chunksL.Count, chunksG?.Count ?? 0);
        var empty = PromptWeights.Chunks("", _ => [], Checkpoint.TokenCount)[0];

        var context = new List<float>();
        float[]? pooled = null;
        int width = 0, rows = 0;
        TextEncoding? emptyEncoding = null;
        for (int c = 0; c < count; c++)
        {
            var l = c < chunksL.Count ? chunksL[c] : empty;
            var g = chunksG is null ? null : c < chunksG.Count ? chunksG[c] : empty;
            var e = await encoder.EncodeAsync(ClipPadding.Pad(l.Ids, padL), g is null ? null : ClipPadding.Pad(g.Ids, 0), clip.Layer, cancellationToken);
            width = e.Shape[^1];
            var z = (float[])e.Context.Clone();
            bool weighted = l.Weights.Any(w => w != 1f) || (g?.Weights.Any(w => w != 1f) ?? false);
            if (weighted)
            {
                emptyEncoding ??= await encoder.EmptyPromptAsync(clip.Layer, clip.Weights);
                if (g is null) PromptWeights.Apply(z, emptyEncoding.Context, l.Weights, width);
                else
                {
                    int first = encoder.FirstTowerWidth;
                    PromptWeights.Apply(z, emptyEncoding.Context, l.Weights, width, 0, first);
                    PromptWeights.Apply(z, emptyEncoding.Context, g.Weights, width, first, width - first);
                }
            }
            context.AddRange(z);
            rows += z.Length / width;
            pooled ??= e.Pooled;
        }
        return new TextEncoding([.. context], [1, rows, width], pooled);
    }
}

/// <summary>Moving pictures between ComfyUI's IMAGE layout (batch, height, width, channels in 0..1) and the VAE's (channels, height, width in -1..1).</summary>
static class VaePictures
{
    public static float[] ToVae(ImageBatch image, int b)
    {
        int h = image.Height, w = image.Width, c = image.Channels, plane = h * w;
        var r = new float[3 * plane];
        for (int p = 0; p < plane; p++)
            for (int k = 0; k < 3; k++) r[k * plane + p] = image.Data[(b * plane + p) * c + Math.Min(k, c - 1)] * 2f - 1f;
        return r;
    }

    public static void FromVae(float[] chw, ImageBatch into, int b)
    {
        int plane = into.Height * into.Width;
        for (int p = 0; p < plane; p++)
            for (int k = 0; k < 3; k++) into.Data[(b * plane + p) * 3 + k] = Math.Clamp((chw[k * plane + p] + 1f) / 2f, 0f, 1f);
    }

    public static float[] Frame(LatentBatch latent, int b)
    {
        int size = latent.Channels * latent.Height * latent.Width;
        return latent.Data.AsSpan(b * size, size).ToArray();
    }

    public static async Task<LatentBatch> EncodeAsync(IVaeModel vae, ImageBatch pixels, CancellationToken ct)
    {
        var cropped = VAEEncode.CropForVae(pixels);
        int H = cropped.Height, W = cropped.Width;
        float[][] frames = new float[cropped.Batch][];
        for (int b = 0; b < cropped.Batch; b++) frames[b] = await vae.EncodeImageAsync(ToVae(cropped, b), H, W, ct);
        int size = frames[0].Length, channels = size / ((H / 8) * (W / 8));
        var latent = new LatentBatch(cropped.Batch, channels, H / 8, W / 8);
        for (int b = 0; b < frames.Length; b++) Array.Copy(frames[b], 0, latent.Data, b * size, size);
        return latent;
    }
}

[VideoLatents]
[Node("VAEDecode", Category = "model/latent", DisplayName = "VAE Decode")]
public sealed class VAEDecode
{
    /// <summary>Each latent decoded to a picture 8 times its size, (v + 1) / 2 clamped to 0..1; a video latent to its frames (every item's in turn).</summary>
    public async Task<ImageBatch> ExecuteAsync(LatentBatch samples, VaeHandle vae, NodeContext context, IServiceProvider services)
    {
        if (samples.IsVideo) return await VideoVae.DecodeAsync(vae.Vae, samples, context.CancellationToken);
        if (vae.Vae is IVideoVaeModel) throw new NotSupportedException($"{vae.Vae.Name} is a video VAE; it decodes video latents.");
        int h = samples.Height, w = samples.Width;
        var image = new ImageBatch(samples.Batch, h * 8, w * 8, 3);
        for (int b = 0; b < samples.Batch; b++)
            VaePictures.FromVae(await vae.Vae.DecodeAsync(VaePictures.Frame(samples, b), h, w, context.CancellationToken), image, b);
        return image;
    }
}

[Node("VAEEncode", Category = "model/latent", DisplayName = "VAE Encode")]
public sealed class VAEEncode
{
    /// <summary>The pictures as latents; with a video VAE (Wan 2.2's) the batch is one video, cut to multiples of 16.</summary>
    public async Task<LatentBatch> ExecuteAsync(ImageBatch pixels, VaeHandle vae, NodeContext context, IServiceProvider services) =>
        vae.Vae is IVideoVaeModel video ? await VideoVae.EncodeAsync(video, pixels, context.CancellationToken)
            : await VaePictures.EncodeAsync(vae.Vae, pixels, context.CancellationToken);

    /// <summary>The picture cut down to whole multiples of 8, the cut shared between both sides (the extra pixel at the far side), as measured.</summary>
    public static ImageBatch CropForVae(ImageBatch pixels) => Inpaint.Crop(pixels, (pixels.Height / 8) * 8, (pixels.Width / 8) * 8);
}

/// <summary>The picture and mask preparation of the inpainting nodes (measured with ComfyUI, InpaintParityTests).</summary>
static class Inpaint
{
    public static int Offset(int size, int kept) => (size - kept) / 2;

    public static ImageBatch Crop(ImageBatch image, int h, int w)
    {
        if (h == image.Height && w == image.Width) return image;
        int y0 = Offset(image.Height, h), x0 = Offset(image.Width, w), c = image.Channels;
        var r = new ImageBatch(image.Batch, h, w, c);
        for (int b = 0; b < image.Batch; b++)
            for (int y = 0; y < h; y++)
                Array.Copy(image.Data, ((b * image.Height + y0 + y) * image.Width + x0) * c, r.Data, ((b * h + y) * w) * c, w * c);
        return r;
    }

    public static MaskBatch Crop(MaskBatch mask, int h, int w)
    {
        if (h == mask.Height && w == mask.Width) return mask;
        int y0 = Offset(mask.Height, h), x0 = Offset(mask.Width, w);
        var r = new MaskBatch(mask.Batch, h, w);
        for (int b = 0; b < mask.Batch; b++)
            for (int y = 0; y < h; y++)
                Array.Copy(mask.Data, (b * mask.Height + y0 + y) * mask.Width + x0, r.Data, (b * h + y) * w, w);
        return r;
    }

    /// <summary>The mask resized (bilinear, as torch's interpolate without corner alignment) to the picture's size.</summary>
    public static MaskBatch ToSize(MaskBatch mask, int h, int w)
    {
        if (mask.Height == h && mask.Width == w) return mask;
        var data = TorchInterpolate.Resize(mask.Data, mask.Batch, mask.Height, mask.Width, h, w, "bilinear");
        return new MaskBatch(mask.Batch, h, w, data);
    }

    /// <summary>The picture grey (0.5) where the mask rounds to 1 (each picture taking the mask of its index, the last one when there are fewer).</summary>
    public static ImageBatch Grey(ImageBatch image, MaskBatch mask)
    {
        var r = new ImageBatch(image.Batch, image.Height, image.Width, image.Channels, (float[])image.Data.Clone());
        int plane = image.Height * image.Width, c = image.Channels;
        for (int b = 0; b < image.Batch; b++)
        {
            int mb = Math.Min(b, mask.Batch - 1);
            for (int p = 0; p < plane; p++)
            {
                float keep = 1f - MathF.Round(mask.Data[mb * plane + p], MidpointRounding.ToEven);
                for (int k = 0; k < Math.Min(c, 3); k++)
                {
                    int i = (b * plane + p) * c + k;
                    r.Data[i] = (r.Data[i] - 0.5f) * keep + 0.5f;
                }
            }
        }
        return r;
    }
}

[Node("VAEEncodeForInpaint", Category = "model/latent", DisplayName = "VAE Encode (for Inpainting)")]
public sealed class VAEEncodeForInpaint
{
    public async Task<LatentBatch> ExecuteAsync(ImageBatch pixels, VaeHandle vae, MaskBatch mask, NodeContext context, IServiceProvider services,
        [Input(Min = 0, Max = 64)] int growMaskBy = 6)
    {
        var (greyed, noiseMask) = Prepare(pixels, mask, growMaskBy);
        var latent = await VaePictures.EncodeAsync(vae.Vae, greyed, context.CancellationToken);
        return new LatentBatch(latent.Batch, latent.Channels, latent.Height, latent.Width, latent.Data) { NoiseMask = noiseMask };
    }

    /// <summary>
    /// The picture greyed where the mask rounds to 1 and the noise mask: the mask resized to the picture, both cut to multiples of 8, then the rounded
    /// mask grown (measured): a pixel is set when any set pixel lies in the grow x grow window starting grow / 2 before it (none for grow 0).
    /// </summary>
    public static (ImageBatch Greyed, MaskBatch NoiseMask) Prepare(ImageBatch pixels, MaskBatch mask, int grow)
    {
        int H = (pixels.Height / 8) * 8, W = (pixels.Width / 8) * 8;
        var full = Inpaint.ToSize(mask, pixels.Height, pixels.Width);
        var picture = Inpaint.Crop(pixels, H, W);
        var m = Inpaint.Crop(full, H, W);
        var grown = Grow(m, grow);
        return (Inpaint.Grey(picture, m), grown);
    }

    static MaskBatch Grow(MaskBatch mask, int grow)
    {
        if (grow <= 0) return new MaskBatch(mask.Batch, mask.Height, mask.Width, [.. mask.Data.Select(v => MathF.Round(v, MidpointRounding.ToEven))]);
        int h = mask.Height, w = mask.Width;
        var r = new MaskBatch(mask.Batch, h, w);
        for (int b = 0; b < mask.Batch; b++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    float v = 0;
                    for (int dy = -(grow / 2); dy < grow - grow / 2 && v < 1; dy++)
                        for (int dx = -(grow / 2); dx < grow - grow / 2; dx++)
                        {
                            int yy = y + dy, xx = x + dx;
                            if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
                            if (MathF.Round(mask.Data[(b * h + yy) * w + xx], MidpointRounding.ToEven) >= 1) { v = 1; break; }
                        }
                    r.Data[(b * h + y) * w + x] = v;
                }
        return r;
    }
}

/// <summary>The picture preparation of InpaintModelConditioning.</summary>
internal static class InpaintPixels
{
    /// <summary>
    /// The picture and the mask (resized to the picture) cut to multiples of 8, and the picture greyed where the mask rounds to 1: what is encoded for
    /// the latent, the masks, and what is encoded for the concat latent.
    /// </summary>
    public static (ImageBatch Picture, MaskBatch Mask, ImageBatch Greyed) Prepare(ImageBatch pixels, MaskBatch mask)
    {
        int H = (pixels.Height / 8) * 8, W = (pixels.Width / 8) * 8;
        var m = Inpaint.Crop(Inpaint.ToSize(mask, pixels.Height, pixels.Width), H, W);
        var picture = Inpaint.Crop(pixels, H, W);
        return (picture, m, Inpaint.Grey(picture, m));
    }
}

[Node("InpaintModelConditioning", Category = "model/conditioning", DisplayName = "InpaintModelConditioning", OutputNames = ["positive", "negative", "latent"])]
public sealed class InpaintModelConditioning
{
    /// <summary>
    /// For inpainting models: both conditionings carry the greyed picture's latent and the mask (the UNet's extra input channels); the latent is the
    /// whole picture encoded, with the mask as its noise mask when asked.
    /// </summary>
    public async Task<(Conditioning Positive, Conditioning Negative, LatentBatch Latent)> ExecuteAsync(Conditioning positive, Conditioning negative,
        VaeHandle vae, ImageBatch pixels, MaskBatch mask, NodeContext context, IServiceProvider services,
        [Input(Default = true, Tooltip = "Add a noise mask to the latent so sampling will only happen within the mask. Might improve results or completely break things depending on the model.")]
        bool noiseMask)
    {
        var (picture, m, greyed) = InpaintPixels.Prepare(pixels, mask);
        var concat = await VaePictures.EncodeAsync(vae.Vae, greyed, context.CancellationToken);
        var latent = await VaePictures.EncodeAsync(vae.Vae, picture, context.CancellationToken);
        if (noiseMask) latent = new LatentBatch(latent.Batch, latent.Channels, latent.Height, latent.Width, latent.Data) { NoiseMask = m };
        return (positive.Map(e => e with { ConcatLatent = concat, ConcatMask = m }), negative.Map(e => e with { ConcatLatent = concat, ConcatMask = m }), latent);
    }
}

/// <summary>Decoding and encoding in overlapping tiles, blended with linear ramps over the overlaps (SpawnGen's own tiling; seams can differ from ComfyUI's).</summary>
static class Tiles
{
    /// <summary>Tile starts covering 0..size with tiles of <paramref name="tile"/> overlapping by <paramref name="overlap"/>.</summary>
    public static List<int> Starts(int size, int tile, int overlap)
    {
        if (tile >= size) return [0];
        int step = Math.Max(1, tile - overlap);
        var s = new List<int>();
        for (int p = 0; ; p += step)
        {
            if (p + tile >= size) { s.Add(size - tile); break; }
            s.Add(p);
        }
        return s;
    }

    public static float Ramp(int i, int length, int overlap, bool first, bool last)
    {
        if (overlap <= 0) return 1f;
        float a = first ? 1f : Math.Min(1f, (i + 1) / (float)(overlap + 1));
        float b = last ? 1f : Math.Min(1f, (length - i) / (float)(overlap + 1));
        return Math.Min(a, b);
    }
}

[Node("VAEDecodeTiled", Category = "model/latent", DisplayName = "VAE Decode (Tiled)")]
public sealed class VAEDecodeTiled
{
    public async Task<ImageBatch> ExecuteAsync(LatentBatch samples, VaeHandle vae, NodeContext context, IServiceProvider services,
        [Input(Min = 64, Max = 4096, Step = 32)] int tileSize = 512, [Input(Min = 0, Max = 4096, Step = 32)] int overlap = 64,
        [Input(Min = 8, Max = 4096, Step = 4)] int temporalSize = 64, [Input(Min = 4, Max = 4096, Step = 4)] int temporalOverlap = 8)
    {
        int lt = Math.Max(1, tileSize / 8), lo = Math.Min(overlap / 8, lt / 4), h = samples.Height, w = samples.Width, c = samples.Channels;
        var image = new ImageBatch(samples.Batch, h * 8, w * 8, 3);
        var ys = Tiles.Starts(h, Math.Min(lt, h), lo);
        var xs = Tiles.Starts(w, Math.Min(lt, w), lo);
        for (int b = 0; b < samples.Batch; b++)
        {
            var frame = VaePictures.Frame(samples, b);
            var sum = new float[3 * h * 8 * w * 8];
            var weight = new float[h * 8 * w * 8];
            int th = Math.Min(lt, h), tw = Math.Min(lt, w);
            foreach (var y0 in ys)
                foreach (var x0 in xs)
                {
                    var tile = new float[c * th * tw];
                    for (int k = 0; k < c; k++)
                        for (int y = 0; y < th; y++)
                            Array.Copy(frame, (k * h + y0 + y) * w + x0, tile, (k * th + y) * tw, tw);
                    var rgb = await vae.Vae.DecodeAsync(tile, th, tw, context.CancellationToken);
                    int H = th * 8, W = tw * 8, plane = H * W;
                    for (int y = 0; y < H; y++)
                        for (int x = 0; x < W; x++)
                        {
                            float wgt = Tiles.Ramp(y, H, lo * 8, y0 == 0, y0 + th == h) * Tiles.Ramp(x, W, lo * 8, x0 == 0, x0 + tw == w);
                            int at = (y0 * 8 + y) * w * 8 + x0 * 8 + x;
                            weight[at] += wgt;
                            for (int k = 0; k < 3; k++) sum[k * h * 8 * w * 8 + at] += rgb[k * plane + y * W + x] * wgt;
                        }
                }
            int full = h * 8 * w * 8;
            for (int i = 0; i < full; i++)
                for (int k = 0; k < 3; k++) sum[k * full + i] /= Math.Max(weight[i], 1e-8f);
            VaePictures.FromVae(sum, image, b);
        }
        return image;
    }
}

[Node("VAEEncodeTiled", Category = "model/latent", DisplayName = "VAE Encode (Tiled)")]
public sealed class VAEEncodeTiled
{
    public async Task<LatentBatch> ExecuteAsync(ImageBatch pixels, VaeHandle vae, NodeContext context, IServiceProvider services,
        [Input(Min = 64, Max = 4096, Step = 64)] int tileSize = 512, [Input(Min = 0, Max = 4096, Step = 32)] int overlap = 64,
        [Input(Min = 8, Max = 4096, Step = 4)] int temporalSize = 64, [Input(Min = 4, Max = 4096, Step = 4)] int temporalOverlap = 8)
    {
        var cropped = VAEEncode.CropForVae(pixels);
        int H = cropped.Height, W = cropped.Width, h = H / 8, w = W / 8;
        int lt = Math.Max(1, tileSize / 8), lo = Math.Min(overlap / 8, lt / 4);
        int th = Math.Min(lt, h), tw = Math.Min(lt, w);
        var ys = Tiles.Starts(h, th, lo);
        var xs = Tiles.Starts(w, tw, lo);
        LatentBatch? latent = null;
        for (int b = 0; b < cropped.Batch; b++)
        {
            var picture = VaePictures.ToVae(cropped, b);
            float[]? sum = null;
            var weight = new float[h * w];
            int channels = 0;
            foreach (var y0 in ys)
                foreach (var x0 in xs)
                {
                    int TH = th * 8, TW = tw * 8;
                    var tile = new float[3 * TH * TW];
                    for (int k = 0; k < 3; k++)
                        for (int y = 0; y < TH; y++)
                            Array.Copy(picture, (k * H + y0 * 8 + y) * W + x0 * 8, tile, (k * TH + y) * TW, TW);
                    var lat = await vae.Vae.EncodeImageAsync(tile, TH, TW, context.CancellationToken);
                    channels = lat.Length / (th * tw);
                    sum ??= new float[channels * h * w];
                    for (int y = 0; y < th; y++)
                        for (int x = 0; x < tw; x++)
                        {
                            float wgt = Tiles.Ramp(y, th, lo, y0 == 0, y0 + th == h) * Tiles.Ramp(x, tw, lo, x0 == 0, x0 + tw == w);
                            int at = (y0 + y) * w + x0 + x;
                            weight[at] += wgt;
                            for (int k = 0; k < channels; k++) sum[k * h * w + at] += lat[(k * th + y) * tw + x] * wgt;
                        }
                }
            latent ??= new LatentBatch(cropped.Batch, channels, h, w);
            for (int k = 0; k < channels; k++)
                for (int i = 0; i < h * w; i++) latent.Data[(b * channels + k) * h * w + i] = sum![k * h * w + i] / Math.Max(weight[i], 1e-8f);
        }
        return latent!;
    }
}
