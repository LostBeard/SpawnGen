using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md). Behavior measured with ComfyUI (2026-11-08) against the 84 common_upscale cases of
// Fixtures/comfy/comfy_image_nodes_main.json.gz and probes of its Upscale Image node; the resampling itself follows PyTorch's documented
// torch.nn.functional.interpolate (BSD-3-Clause) and Pillow's LANCZOS resize (PilLanczos).

/// <summary>
/// The picture resize the image nodes share. Measured with ComfyUI:
/// <list type="bullet">
/// <item>crop "center" first cuts the picture to the target aspect: with aspects a = w / h and A = width / height, x = round((w - w * A / a) / 2)
/// columns go from each side when a &gt; A, else y = round((h - h * a / A) / 2) rows from top and bottom (Python round, half to even).</item>
/// <item>"nearest-exact", "bilinear", "area" and "bicubic" are PyTorch's interpolate modes (align_corners false, no antialiasing; area is adaptive
/// average pooling); "lanczos" is Pillow's LANCZOS on 8-bit L / LA / RGB / RGBA pictures made by truncating 255 * v.</item>
/// <item>A crop that leaves nothing fails with PyTorch's errors; lanczos gives a black picture.</item>
/// </list>
/// </summary>
public static class CommonUpscale
{
    /// <summary>Resizes every picture of the batch.</summary>
    public static ImageBatch Resize(ImageBatch image, int width, int height, string method, string crop = "disabled") =>
        Pixels.Map(image, f => Resize(f, image.Width, image.Height, image.Channels, width, height, method, crop), height, width);

    /// <summary>Resizes every mask of the batch (one channel).</summary>
    public static MaskBatch Resize(MaskBatch mask, int width, int height, string method, string crop = "disabled")
    {
        int n = mask.Width * mask.Height;
        var result = new MaskBatch(mask.Batch, height, width);
        for (int b = 0; b < mask.Batch; b++)
        {
            var frame = Resize(mask.Data.AsSpan(b * n, n).ToArray(), mask.Width, mask.Height, 1, width, height, method, crop);
            Array.Copy(frame, 0, result.Data, (long)b * width * height, width * height);
        }
        return result;
    }

    /// <summary>Resizes one picture, <paramref name="w"/> x <paramref name="h"/> x <paramref name="c"/> interleaved floats.</summary>
    public static float[] Resize(float[] src, int w, int h, int c, int newW, int newH, string method, string crop = "disabled")
    {
        if (crop == "center") (src, w, h) = CenterCrop(src, w, h, c, newW, newH);
        else if (crop != "disabled") throw new ArgumentException($"Unknown crop '{crop}'.", nameof(crop));
        return method switch
        {
            "lanczos" => PilLanczos.FromBytes(w == 0 || h == 0 ? new byte[newW * newH * c] : PilLanczos.Resize(PilLanczos.ToBytes(src), w, h, c, newW, newH)),
            "nearest-exact" => Torch(src, w, h, c, newW, newH, NearestExact, 1),
            "bilinear" => Torch(src, w, h, c, newW, newH, Bilinear, 2),
            "bicubic" => Torch(src, w, h, c, newW, newH, Bicubic, 4),
            "area" => Area(src, w, h, c, newW, newH),
            _ => throw new ArgumentException($"Unknown upscale method '{method}'.", nameof(method)),
        };
    }

    static (float[] Data, int W, int H) CenterCrop(float[] src, int w, int h, int c, int newW, int newH)
    {
        double oldAspect = (double)w / h, newAspect = (double)newW / newH;
        int x = 0, y = 0;
        if (oldAspect > newAspect) x = (int)Math.Round((w - w * (newAspect / oldAspect)) / 2, MidpointRounding.ToEven);
        else if (oldAspect < newAspect) y = (int)Math.Round((h - h * (oldAspect / newAspect)) / 2, MidpointRounding.ToEven);
        if (x == 0 && y == 0) return (src, w, h);
        int cw = Math.Max(w - 2 * x, 0), ch = Math.Max(h - 2 * y, 0);
        var cut = new float[cw * ch * c];
        for (int row = 0; row < ch; row++) Array.Copy(src, ((row + y) * w + x) * c, cut, row * cw * c, cw * c);
        return (cut, cw, ch);
    }

    static void CheckSizes(int w, int h, int newW, int newH)
    {
        if (w <= 0 || h <= 0 || newW <= 0 || newH <= 0)
            throw new ArgumentException($"Input and output sizes should be greater than 0, but got input (H: {h}, W: {w}) output (H: {newH}, W: {newW})");
    }

    // ---- PyTorch interpolate (CPU, float32, align_corners false) ----

    /// <summary>Indices and weights of one output position along one axis.</summary>
    delegate void Axis(int dst, int inSize, int outSize, Span<int> index, Span<float> weight);

    /// <summary>Separable interpolation: every output pixel is Σ_j wy_j Σ_i wx_i x[iy_j, ix_i], clamped indices supplied by the axis rule.</summary>
    static float[] Torch(float[] src, int w, int h, int c, int newW, int newH, Axis axis, int taps)
    {
        CheckSizes(w, h, newW, newH);
        var result = new float[newW * newH * c];
        Span<int> ix = stackalloc int[4], iy = stackalloc int[4];
        Span<float> wx = stackalloc float[4], wy = stackalloc float[4];
        var xIndex = new int[newW * taps];
        var xWeight = new float[newW * taps];
        for (int x = 0; x < newW; x++)
        {
            axis(x, w, newW, ix, wx);
            for (int t = 0; t < taps; t++) { xIndex[x * taps + t] = ix[t]; xWeight[x * taps + t] = wx[t]; }
        }
        for (int y = 0; y < newH; y++)
        {
            axis(y, h, newH, iy, wy);
            for (int x = 0; x < newW; x++)
                for (int k = 0; k < c; k++)
                {
                    float sum = 0;
                    for (int j = 0; j < taps; j++)
                    {
                        float row = 0;
                        for (int i = 0; i < taps; i++) row += xWeight[x * taps + i] * src[(iy[j] * w + xIndex[x * taps + i]) * c + k];
                        sum += wy[j] * row;
                    }
                    result[(y * newW + x) * c + k] = sum;
                }
        }
        return result;
    }

    /// <summary>PyTorch's scale for a given output size: input / output in float32.</summary>
    static float Scale(int inSize, int outSize) => (float)inSize / outSize;

    /// <summary>PyTorch's source coordinate (align_corners false): scale * (dst + 0.5) - 0.5, in double, stored as float32.</summary>
    static float SourceCoordinate(float scale, int dst) => (float)(scale * (dst + 0.5) - 0.5);

    /// <summary>nearest-exact: the input pixel min(floor((dst + 0.5) * scale), in - 1).</summary>
    static void NearestExact(int dst, int inSize, int outSize, Span<int> index, Span<float> weight)
    {
        float scale = Scale(inSize, outSize);
        index[0] = Math.Min((int)MathF.Floor((float)((dst + 0.5) * scale)), inSize - 1);
        weight[0] = 1f;
    }

    /// <summary>bilinear: the source coordinate clamped at 0, its floor and the next pixel (the last pixel repeats), weights 1 - λ and λ.</summary>
    static void Bilinear(int dst, int inSize, int outSize, Span<int> index, Span<float> weight)
    {
        float real = SourceCoordinate(Scale(inSize, outSize), dst);
        if (real < 0) real = 0;
        int i0 = Math.Min((int)MathF.Floor(real), inSize - 1);
        float lambda = Math.Clamp(real - i0, 0f, 1f);
        index[0] = i0;
        index[1] = i0 + (i0 < inSize - 1 ? 1 : 0);
        weight[0] = 1f - lambda;
        weight[1] = lambda;
    }

    /// <summary>bicubic: Keys' cubic convolution with A = -0.75 over the four pixels around the source coordinate, indices clamped to the picture.</summary>
    static void Bicubic(int dst, int inSize, int outSize, Span<int> index, Span<float> weight)
    {
        const float A = -0.75f;
        float real = SourceCoordinate(Scale(inSize, outSize), dst);
        int i = (int)MathF.Floor(real);
        float t = real - i;
        static float Conv1(float x) => ((A + 2) * x - (A + 3)) * x * x + 1;
        static float Conv2(float x) => ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
        weight[0] = Conv2(t + 1f);
        weight[1] = Conv1(t);
        float u = 1f - t;
        weight[2] = Conv1(u);
        weight[3] = Conv2(u + 1f);
        for (int k = 0; k < 4; k++) index[k] = Math.Clamp(i - 1 + k, 0, inSize - 1);
    }

    /// <summary>area: adaptive average pooling, each output pixel the mean of input rows floor(y h / H) .. ceil((y + 1) h / H) and the same for columns.</summary>
    static float[] Area(float[] src, int w, int h, int c, int newW, int newH)
    {
        if (w <= 0 || h <= 0)
            throw new ArgumentException($"adaptive_avg_pool2d(): Expected input to have non-zero size for non-batch dimensions, but input has sizes [1, {c}, {h}, {w}] with dimension {(h <= 0 ? 2 : 3)} being empty");
        var result = new float[Math.Max(newW, 0) * Math.Max(newH, 0) * c];
        for (int y = 0; y < newH; y++)
        {
            int y0 = (int)((long)y * h / newH), y1 = (int)(((long)(y + 1) * h + newH - 1) / newH);
            for (int x = 0; x < newW; x++)
            {
                int x0 = (int)((long)x * w / newW), x1 = (int)(((long)(x + 1) * w + newW - 1) / newW);
                for (int k = 0; k < c; k++)
                {
                    float sum = 0;
                    for (int yy = y0; yy < y1; yy++)
                        for (int xx = x0; xx < x1; xx++) sum += src[(yy * w + xx) * c + k];
                    result[(y * newW + x) * c + k] = sum / (y1 - y0) / (x1 - x0);
                }
            }
        }
        return result;
    }
}
