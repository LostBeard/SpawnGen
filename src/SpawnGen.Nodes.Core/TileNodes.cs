using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// Where tiles go along one axis, as measured with ComfyUI's tile nodes: the step is the tile size minus the overlap but at least a quarter of the tile
/// (rounded half to even, so a 10-pixel tile steps at least 2); tiles start at 0, step, 2 x step, ... and the first one that would reach the far edge is
/// moved back to end exactly there, which is the last. A tile larger than the picture starts at 0.
/// </summary>
static class TileGrid
{
    public static List<int> Starts(int size, int tile, int overlap)
    {
        int quarter = (int)Math.Round(tile / 4.0, MidpointRounding.ToEven);
        int step = Math.Max(1, Math.Max(tile - overlap, quarter)); // ComfyUI never ends for a step of 0 (tiles of 1 or 2 pixels); SpawnGen steps 1
        var starts = new List<int>();
        for (long x = 0; ; x += step)
        {
            if (x + tile >= size)
            {
                starts.Add(Math.Max(size - tile, 0));
                return starts;
            }
            starts.Add((int)x);
        }
    }
}

[Node("SplitImageToTileList", Category = "image/batch", DisplayName = "Split Image into List of Tiles",
    Description = "Splits an image into a batched list of tiles with a specified overlap.", OutputIsList = [true])]
public sealed class SplitImageToTileList
{
    /// <summary>The tiles row by row, left to right; each keeps the whole batch and is cut to the picture where the tile is larger.</summary>
    public IReadOnlyList<ImageBatch> Execute(ImageBatch image,
        [Input(Min = 64, Max = 16384)] long tileWidth = 1024,
        [Input(Min = 64, Max = 16384)] long tileHeight = 1024,
        [Input(Min = 0, Max = 4096)] long overlap = 128)
    {
        int b = image.Batch, h = image.Height, w = image.Width, c = image.Channels;
        int tw = (int)Math.Clamp(tileWidth, 1, w), th = (int)Math.Clamp(tileHeight, 1, h), ov = (int)Math.Clamp(overlap, 0, int.MaxValue);
        var xs = TileGrid.Starts(w, (int)Math.Min(tileWidth, int.MaxValue), ov);
        var ys = TileGrid.Starts(h, (int)Math.Min(tileHeight, int.MaxValue), ov);
        var tiles = new List<ImageBatch>(xs.Count * ys.Count);
        foreach (int y in ys)
            foreach (int x in xs)
            {
                var tile = new ImageBatch(b, th, tw, c);
                for (int f = 0; f < b; f++)
                    for (int r = 0; r < th; r++)
                        Array.Copy(image.Data, ((f * h + y + r) * w + x) * c, tile.Data, (f * th + r) * tw * c, tw * c);
                tiles.Add(tile);
            }
        return tiles;
    }
}

/// <summary>
/// Merge List of Tiles to Image, as measured: the tiles go on the grid that Split Image into List of Tiles makes for the final size, the first tile's
/// size and the overlap, row by row (extra tiles are left out, missing ones leave black). Each tile is weighted by sin(linspace(0, pi, n)) along each axis,
/// at least 1e-5, the two multiplied; every pixel is the weighted sum over the weight sum (0 where no tile reaches). The canvas takes the first tile's
/// batch and channels; a later tile may have one picture or one channel (it is broadcast), anything else is an error, as in torch.
/// </summary>
[Node("ImageMergeTileList", Category = "image/batch", DisplayName = "Merge List of Tiles to Image", InputIsList = true)]
public sealed class ImageMergeTileList
{
    public ImageBatch Execute(IReadOnlyList<ImageBatch> imageList,
        [Input(Min = 64, Max = 32768, Default = 1024L)] IReadOnlyList<long> finalWidth,
        [Input(Min = 64, Max = 32768, Default = 1024L)] IReadOnlyList<long> finalHeight,
        [Input(Min = 0, Max = 4096, Default = 128L)] IReadOnlyList<long> overlap)
    {
        if (imageList.Count == 0) throw new PyException("IndexError", "list index out of range");
        int W = (int)finalWidth[0], H = (int)finalHeight[0], ov = (int)overlap[0];
        var first = imageList[0];
        int B = first.Batch, C = first.Channels, th0 = first.Height, tw0 = first.Width;
        var xs = TileGrid.Starts(W, tw0, ov);
        var ys = TileGrid.Starts(H, th0, ov);
        var acc = new float[B * H * W * C];
        var weightSum = new float[H * W];
        int k = 0;
        foreach (int y in ys)
            foreach (int x in xs)
            {
                if (k >= imageList.Count) break;
                var tile = imageList[k++];
                if (tile.Batch != B && tile.Batch != 1)
                    throw new InvalidOperationException($"The size of tensor a ({B}) must match the size of tensor b ({tile.Batch}) at non-singleton dimension 0");
                if (tile.Channels != C && tile.Channels != 1)
                    throw new InvalidOperationException($"The size of tensor a ({C}) must match the size of tensor b ({tile.Channels}) at non-singleton dimension 3");
                // Measured: a tile larger than the first is cut to the first tile's size (and to the canvas); the weights span what is left.
                int th = Math.Min(Math.Min(tile.Height, th0), H - y), tw = Math.Min(Math.Min(tile.Width, tw0), W - x);
                if (th <= 0 || tw <= 0) continue;
                var wy = Window(th);
                var wx = Window(tw);
                for (int r = 0; r < th; r++)
                    for (int q = 0; q < tw; q++)
                    {
                        float wgt = wy[r] * wx[q];
                        int p = (y + r) * W + x + q;
                        weightSum[p] += wgt;
                        for (int f = 0; f < B; f++)
                        {
                            int src = (((tile.Batch == 1 ? 0 : f) * tile.Height + r) * tile.Width + q) * tile.Channels;
                            int dst = (f * H * W + p) * C;
                            for (int ch = 0; ch < C; ch++)
                                acc[dst + ch] += tile.Data[src + (tile.Channels == 1 ? 0 : ch)] * wgt;
                        }
                    }
            }
        var result = new ImageBatch(B, H, W, C);
        for (int f = 0; f < B; f++)
            for (int p = 0; p < H * W; p++)
            {
                float s = weightSum[p];
                if (s <= 0f) continue;
                int i = (f * H * W + p) * C;
                for (int ch = 0; ch < C; ch++) result.Data[i + ch] = acc[i + ch] / s;
            }
        return result;
    }

    /// <summary>sin over torch.linspace(0, pi, n) in float32 (the first half counted up from 0, the second down from pi), at least 1e-5.</summary>
    static float[] Window(int n)
    {
        var w = new float[n];
        float end = MathF.PI;
        float step = n > 1 ? end / (n - 1) : 0f;
        for (int i = 0; i < n; i++)
        {
            float v = i < n / 2 ? i * step : end - (n - 1 - i) * step;
            w[i] = Math.Max(MathF.Sin(v), 1e-5f);
        }
        return w;
    }
}
