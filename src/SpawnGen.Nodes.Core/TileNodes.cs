using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

[Node(Category = "image/batch", DisplayName = "Split Image into List of Tiles", OutputIsList = [true],
    Description = "Splits an image into a batched list of tiles with a specified overlap.")]
public sealed class SplitImageToTileList
{
    /// <summary>
    /// ComfyUI's tile grid: rows and columns step by <c>round(max(tile * 0.25, tile - overlap))</c>; the last tile of a row or column is moved back so it ends
    /// at the edge (a tile larger than the picture is the whole picture). Tiles in row order, as (x_start, y_start, x_end, y_end).
    /// </summary>
    public static List<(long X0, long Y0, long X1, long Y1)> GridCoords(long width, long height, long tileWidth, long tileHeight, long overlap)
    {
        // Python's max(float, int) keeps the first unless the second is larger; round() of either gives the same int.
        static long Stride(long tile, long overlap)
        {
            double quarter = tile * 0.25;
            long rest = tile - overlap;
            return rest > quarter ? rest : PyValues.Round(quarter);
        }
        long strideX = Stride(tileWidth, overlap), strideY = Stride(tileHeight, overlap);
        // ComfyUI loops forever on a zero stride (tiles under 2 px, which the widgets' minimum of 64 rules out).
        if (strideX <= 0 || strideY <= 0) throw new ArgumentException($"Tiles of {tileWidth}x{tileHeight} with overlap {overlap} do not move across the picture.");
        var coords = new List<(long, long, long, long)>();
        for (long y = 0; y < height; y += strideY)
        {
            long yEnd = Math.Min(y + tileHeight, height), yStart = Math.Max(0, yEnd - tileHeight);
            for (long x = 0; x < width; x += strideX)
            {
                long xEnd = Math.Min(x + tileWidth, width), xStart = Math.Max(0, xEnd - tileWidth);
                coords.Add((xStart, yStart, xEnd, yEnd));
                if (xEnd >= width) break;
            }
            if (yEnd >= height) break;
        }
        return coords;
    }

    public IReadOnlyList<ImageBatch> Execute(ImageBatch image, [Input(Min = 64, Max = 16384)] long tileWidth = 1024, [Input(Min = 64, Max = 16384)] long tileHeight = 1024,
        [Input(Min = 0, Max = 4096)] long overlap = 128)
    {
        int c = image.Channels;
        var tiles = new List<ImageBatch>();
        foreach (var (x0, y0, x1, y1) in GridCoords(image.Width, image.Height, tileWidth, tileHeight, overlap))
        {
            int tw = (int)(x1 - x0), th = (int)(y1 - y0);
            var tile = new ImageBatch(image.Batch, th, tw, c);
            for (int b = 0; b < image.Batch; b++)
                for (int r = 0; r < th; r++)
                    Array.Copy(image.Data, (((long)b * image.Height + y0 + r) * image.Width + x0) * c, tile.Data, ((long)b * th + r) * tw * c, tw * c);
            tiles.Add(tile);
        }
        return tiles;
    }
}

[Node(Category = "image/batch", DisplayName = "Merge List of Tiles to Image", InputIsList = true)]
public sealed class ImageMergeTileList
{
    /// <summary>
    /// ComfyUI's tile merge: the tiles go back on the grid Split Image into List of Tiles makes for the final size (with the first tile's size), each
    /// weighted by sin(pi t) across its height and width (at least 1e-5) when they overlap, else evenly, and the weighted sum is divided by the weights.
    /// Extra tiles are ignored, missing ones leave black. A tile of batch 1 (or 1 channel) is used for every picture (channel) of the first tile's batch.
    /// </summary>
    public ImageBatch Execute(IReadOnlyList<ImageBatch> imageList,
        [Input(Min = 64, Max = 32768, Default = 1024L)] IReadOnlyList<long> finalWidth,
        [Input(Min = 64, Max = 32768, Default = 1024L)] IReadOnlyList<long> finalHeight,
        [Input(Min = 0, Max = 4096, Default = 128L)] IReadOnlyList<long> overlap)
    {
        int w = (int)finalWidth[0], h = (int)finalHeight[0];
        long ovlp = overlap[0];
        var first = imageList[0];
        int b = first.Batch, th = first.Height, tw = first.Width, c = first.Channels;
        var coords = SplitImageToTileList.GridCoords(w, h, tw, th, ovlp);

        var canvas = new float[(long)b * h * w * c];
        var weights = new float[(long)h * w]; // the same for every picture of the batch
        var mask = new float[th * tw];
        if (ovlp > 0)
        {
            var yw = SineRamp(th);
            var xw = SineRamp(tw);
            // torch.lerp(ones, sine, 1.0) is the sine mask itself.
            for (int r = 0; r < th; r++)
                for (int q = 0; q < tw; q++) mask[r * tw + q] = yw[r] * xw[q];
        }
        else Array.Fill(mask, 1f);

        for (int i = 0; i < coords.Count && i < imageList.Count; i++)
        {
            var (x0, y0, x1, y1) = coords[i];
            var tile = imageList[i];
            if (tile.Batch != b && tile.Batch != 1)
                throw new ArgumentException($"Tile {i} has a batch of {tile.Batch}; the merge's batch is {b} (the first tile's), so only {b} or 1 can be added.", nameof(imageList));
            if (tile.Channels != c && tile.Channels != 1)
                throw new ArgumentException($"Tile {i} has {tile.Channels} channels; the merge has {c} (the first tile's), so only {c} or 1 can be added.", nameof(imageList));
            int realH = (int)Math.Min(y1 - y0, tile.Height), realW = (int)Math.Min(x1 - x0, tile.Width), tc = tile.Channels;
            for (int p = 0; p < b; p++)
            {
                int tp = tile.Batch == 1 ? 0 : p;
                for (int r = 0; r < realH; r++)
                    for (int q = 0; q < realW; q++)
                    {
                        float m = mask[r * tw + q];
                        long dst = (((long)p * h + y0 + r) * w + x0 + q) * c, src = (((long)tp * tile.Height + r) * tile.Width + q) * tc;
                        for (int k = 0; k < c; k++) canvas[dst + k] += tile.Data[src + (tc == 1 ? 0 : k)] * m;
                    }
            }
            for (int r = 0; r < realH; r++)
                for (int q = 0; q < realW; q++) weights[(y0 + r) * w + x0 + q] += mask[r * tw + q];
        }

        var result = new ImageBatch(b, h, w, c);
        for (int p = 0; p < b; p++)
            for (int i = 0; i < h * w; i++)
            {
                float wt = weights[i] == 0 ? 1f : weights[i];
                for (int k = 0; k < c; k++) result.Data[((long)p * h * w + i) * c + k] = canvas[((long)p * h * w + i) * c + k] / wt;
            }
        return result;
    }

    /// <summary><c>clamp(sin(pi * torch.linspace(0, 1, n)), min=1e-5)</c> in float32, with torch's linspace (from both ends towards the middle).</summary>
    static float[] SineRamp(int n)
    {
        var v = new float[n];
        if (n == 1) v[0] = 0f;
        else
        {
            float step = 1f / (n - 1);
            int half = n / 2;
            for (int i = 0; i < n; i++) v[i] = i < half ? step * i : 1f - step * (n - i - 1);
        }
        for (int i = 0; i < n; i++) v[i] = Math.Max(MathF.Sin(MathF.PI * v[i]), 1e-5f);
        return v;
    }
}
