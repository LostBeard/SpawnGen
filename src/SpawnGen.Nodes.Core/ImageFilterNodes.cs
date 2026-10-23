using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

[Node(Category = "image/filters", DisplayName = "Add Noise to Image")]
public sealed class ImageAddNoise
{
    /// <summary>
    /// Adds <c>strength</c> times Gaussian noise and clamps to 0..1. The noise is PyTorch's CPU <c>randn</c> for the seed over the whole batch,
    /// so the same seed gives ComfyUI's grain. An alpha channel (the fourth) is kept as it was.
    /// </summary>
    public ImageBatch Execute(ImageBatch image,
        [Input(Min = 0, Max = 18446744073709551615d, Tooltip = "The random seed used for creating the noise.")] long seed = 0,
        [Input(Min = 0, Max = 1, Step = 0.01)] float strength = 0.5f)
    {
        var noise = TorchRandom.Randn(seed, image.Data.Length);
        var result = new ImageBatch(image.Batch, image.Height, image.Width, image.Channels);
        for (int i = 0; i < noise.Length; i++) result.Data[i] = Math.Clamp(image.Data[i] + strength * noise[i], 0f, 1f);
        if (image.Channels == 4)
            for (int i = 3; i < noise.Length; i += 4) result.Data[i] = image.Data[i];
        return result;
    }
}

[Node(Category = "image/filters", DisplayName = "Detect Edges (Canny)")]
public sealed class Canny
{
    /// <summary>
    /// kornia's <c>canny</c> as ComfyUI calls it (5x5 Gaussian, sigma 1; Sobel; non-maximum suppression in 8 directions; hysteresis): white edges on
    /// black, the same in all three channels. A gradient above <c>high_threshold</c> is an edge; one above <c>low_threshold</c> is an edge when it
    /// touches (8 neighbours, through other such pixels) a strong one.
    /// </summary>
    public ImageBatch Execute(ImageBatch image,
        [Input(Min = 0.01, Max = 0.99, Step = 0.01)] float lowThreshold = 0.4f,
        [Input(Min = 0.01, Max = 0.99, Step = 0.01)] float highThreshold = 0.8f)
    {
        if (lowThreshold > highThreshold) throw new ArgumentException($"low_threshold ({lowThreshold}) must not be greater than high_threshold ({highThreshold}).");
        if (lowThreshold is <= 0 or >= 1 || highThreshold is <= 0 or >= 1) throw new ArgumentException("The thresholds must be between 0 and 1.");
        int w = image.Width, h = image.Height, c = image.Channels;
        if (c == 2) throw new ArgumentException("Canny needs a colour (3 channel) or grey (1 channel) picture.");
        if (w < 3 || h < 3) throw new ArgumentException("Canny needs a picture of at least 3x3 pixels (the blur reflects 2 pixels at the borders).");
        var result = new ImageBatch(image.Batch, h, w, 3);
        for (int b = 0; b < image.Batch; b++)
        {
            var grey = new float[w * h];
            for (int p = 0; p < w * h; p++)
            {
                int i = (b * w * h + p) * c;
                // kornia rgb_to_grayscale: 0.299 r + 0.587 g + 0.114 b (one channel stays as it is).
                grey[p] = c == 1 ? image.Data[i] : 0.299f * image.Data[i] + 0.587f * image.Data[i + 1] + 0.114f * image.Data[i + 2];
            }
            var edges = Edges(grey, w, h, lowThreshold, highThreshold);
            for (int p = 0; p < w * h; p++) result.Data[(b * w * h + p) * 3] = result.Data[(b * w * h + p) * 3 + 1] = result.Data[(b * w * h + p) * 3 + 2] = edges[p];
        }
        return result;
    }

    // The neighbour each of kornia's 8 NMS / hysteresis kernels looks at, by direction index (0 = +x, then clockwise in image coordinates).
    static readonly (int Dy, int Dx)[] Neighbours = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)];

    static float[] Edges(float[] grey, int w, int h, float low, float high)
    {
        // Gaussian blur 5x5, sigma 1, separable (rows then columns), reflected borders.
        var k = new float[5];
        float ksum = 0;
        for (int i = 0; i < 5; i++) { float x = i - 2; k[i] = MathF.Exp(-(x * x) / 2f); ksum += k[i]; }
        for (int i = 0; i < 5; i++) k[i] /= ksum;
        static int Reflect(int i, int n) => i < 0 ? -i : i >= n ? 2 * (n - 1) - i : i;
        var tmp = new float[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                float acc = 0;
                for (int j = 0; j < 5; j++) acc += grey[y * w + Reflect(x + j - 2, w)] * k[j];
                tmp[y * w + x] = acc;
            }
        var blur = new float[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                float acc = 0;
                for (int j = 0; j < 5; j++) acc += tmp[Reflect(y + j - 2, h) * w + x] * k[j];
                blur[y * w + x] = acc;
            }

        // Sobel (unnormalised) with replicated borders; magnitude with kornia's eps; the angle rounded to 45 degrees (half to even, like torch.round).
        float B(int y, int x) => blur[Math.Clamp(y, 0, h - 1) * w + Math.Clamp(x, 0, w - 1)];
        var mag = new float[w * h];
        var dir = new int[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                float gx = -B(y - 1, x - 1) + B(y - 1, x + 1) - 2 * B(y, x - 1) + 2 * B(y, x + 1) - B(y + 1, x - 1) + B(y + 1, x + 1);
                float gy = -B(y - 1, x - 1) - 2 * B(y - 1, x) - B(y - 1, x + 1) + B(y + 1, x - 1) + 2 * B(y + 1, x) + B(y + 1, x + 1);
                mag[y * w + x] = MathF.Sqrt(gx * gx + gy * gy + 1e-6f);
                float angle = 180f * MathF.Atan2(gy, gx) / MathF.PI;
                float steps = MathF.Round(angle / 45f, MidpointRounding.ToEven);
                dir[y * w + x] = (((int)steps % 8) + 8) % 8;
            }

        // Non-maximum suppression: keep a pixel only when it is strictly larger than both neighbours along its gradient (outside counts as 0).
        float M(int y, int x) => y < 0 || y >= h || x < 0 || x >= w ? 0f : mag[y * w + x];
        var edges = new float[w * h];
        var strong = new Queue<int>();
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int p = y * w + x;
                var (py, px) = Neighbours[dir[p]];
                var (ny, nx) = Neighbours[(dir[p] + 4) % 8];
                float m = mag[p];
                bool isMax = Math.Min(m - M(y + py, x + px), m - M(y + ny, x + nx)) > 0f;
                if (!isMax) m = 0f;
                edges[p] = (m > low ? 0.5f : 0f) + (m > high ? 0.5f : 0f);
                if (edges[p] == 1f) strong.Enqueue(p);
            }

        // Hysteresis: kornia repeats "a weak pixel next to a strong one becomes strong" until nothing changes, then drops the weak ones.
        // Growing from the strong pixels through weak 8-neighbours reaches the same fixed point.
        while (strong.Count > 0)
        {
            int p = strong.Dequeue(), y = p / w, x = p % w;
            foreach (var (dy, dx) in Neighbours)
            {
                int qy = y + dy, qx = x + dx;
                if (qy < 0 || qy >= h || qx < 0 || qx >= w || edges[qy * w + qx] != 0.5f) continue;
                edges[qy * w + qx] = 1f;
                strong.Enqueue(qy * w + qx);
            }
        }
        for (int p = 0; p < edges.Length; p++) if (edges[p] != 1f) edges[p] = 0f;
        return edges;
    }
}
