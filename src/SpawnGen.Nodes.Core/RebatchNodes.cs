using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

[Node("RebatchImages", Category = "image/batch", DisplayName = "Rebatch Images", InputIsList = true, OutputIsList = [true])]
public sealed class RebatchImages
{
    /// <summary>
    /// The pictures of every incoming batch, in order, regrouped into batches of <paramref name="batchSize"/> (the last one may be smaller). An
    /// incoming batch is split where a group fills up. Measured with ComfyUI: pictures are not resized, so a group that would hold pictures of
    /// different sizes fails as <c>torch.cat</c> does, naming the first piece of the group that differs from the group's first piece; an empty list
    /// gives an empty list.
    /// </summary>
    public IReadOnlyList<ImageBatch> Execute(IReadOnlyList<ImageBatch> images,
        [Input(Default = 1L, Min = 1, Max = 4096)] IReadOnlyList<long> batchSize)
    {
        int size = (int)Math.Clamp(batchSize[0], 1, int.MaxValue);
        var groups = new List<ImageBatch>();
        var pieces = new List<(ImageBatch Source, int Start, int Count)>();
        int filled = 0;
        foreach (var image in images)
        {
            for (int start = 0; start < image.Batch;)
            {
                int take = Math.Min(size - filled, image.Batch - start);
                pieces.Add((image, start, take));
                filled += take;
                start += take;
                if (filled == size)
                {
                    groups.Add(Join(pieces));
                    pieces.Clear();
                    filled = 0;
                }
            }
        }
        if (pieces.Count > 0) groups.Add(Join(pieces));
        return groups;
    }

    static ImageBatch Join(List<(ImageBatch Source, int Start, int Count)> pieces)
    {
        var first = pieces[0].Source;
        for (int i = 1; i < pieces.Count; i++)
        {
            var s = pieces[i].Source;
            if (s.Height != first.Height) throw Batching.SizeMismatch(first.Height, s.Height, i);
            if (s.Width != first.Width) throw Batching.SizeMismatch(first.Width, s.Width, i);
            if (s.Channels != first.Channels) throw Batching.SizeMismatch(first.Channels, s.Channels, i);
        }
        int per = first.Height * first.Width * first.Channels;
        var r = new ImageBatch(pieces.Sum(p => p.Count), first.Height, first.Width, first.Channels);
        int at = 0;
        foreach (var (source, start, count) in pieces)
        {
            Array.Copy(source.Data, start * per, r.Data, at * per, count * per);
            at += count;
        }
        return r;
    }
}

[Node("RebatchLatents", Category = "model/latent/batch", DisplayName = "Rebatch Latents", InputIsList = true, OutputIsList = [true])]
public sealed class RebatchLatents
{
    /// <summary>
    /// The latents of every incoming batch, in order, regrouped into batches of <paramref name="batchSize"/>. Measured with ComfyUI:
    /// <list type="bullet">
    /// <item>A latent of another height or width closes the group being filled (that group may be short); latents are not resized. Different channel
    /// counts in one group fail as <c>torch.cat</c> does.</item>
    /// <item>Batch indexes: a latent's own, or else a running count continued from the number of batch indexes of the latents before it (so
    /// [0, 1, 2], then [3, 4] for the next latent of two). Each group takes the next <paramref name="batchSize"/> of them.</item>
    /// <item>Noise masks: a latent without one counts as all ones at 8x its height and width; a mask with fewer pictures than its latent is repeated
    /// (and cut) to the latent's count, one with more is kept whole. Each group takes the next <paramref name="batchSize"/> masks, so the masks
    /// follow their pictures; masks of different sizes in one group fail as <c>torch.cat</c> does. A group whose masks sum to their number of values
    /// (all ones) gets no mask.</item>
    /// </list>
    /// </summary>
    public IReadOnlyList<LatentBatch> Execute(IReadOnlyList<LatentBatch> latents,
        [Input(Default = 1L, Min = 1, Max = 4096)] IReadOnlyList<long> batchSize)
    {
        int size = (int)Math.Clamp(batchSize[0], 1, int.MaxValue);
        var output = new List<LatentBatch>();
        Pending? pending = null;
        int counted = 0;
        foreach (var latent in latents)
        {
            var batchIndex = latent.BatchIndex ?? [.. Enumerable.Range(counted, latent.Batch)];
            counted += batchIndex.Length;
            var next = new Pending(latent, MaskRows(latent), batchIndex);
            if (pending is not null && (pending.Height != latent.Height || pending.Width != latent.Width))
            {
                pending.EmitInto(output, size);
                pending = null;
            }
            if (pending is null) pending = next;
            else pending.Append(next);
            while (pending.Samples.Count >= size)
            {
                pending.EmitInto(output, size);
                pending.DropFirst(size);
            }
        }
        pending?.EmitInto(output, size);
        return output;
    }

    /// <summary>The latent's noise mask, one row per mask picture: all ones at 8x the latent's size when it has none, repeated up to the latent's count.</summary>
    static (List<float[]> Rows, int Height, int Width) MaskRows(LatentBatch latent)
    {
        var mask = latent.NoiseMask;
        if (mask is null)
        {
            int h = latent.Height * 8, w = latent.Width * 8;
            return ([.. Enumerable.Range(0, latent.Batch).Select(_ => Enumerable.Repeat(1f, h * w).ToArray())], h, w);
        }
        int plane = mask.Height * mask.Width;
        var rows = Enumerable.Range(0, mask.Batch).Select(b => mask.Data.AsSpan(b * plane, plane).ToArray()).ToList();
        if (mask.Batch > 0 && mask.Batch < latent.Batch)
            rows = [.. Enumerable.Range(0, latent.Batch).Select(i => rows[i % mask.Batch])];
        return (rows, mask.Height, mask.Width);
    }

    /// <summary>The group being filled: sample rows, mask rows and batch indexes, each joined in order and cut from the front as groups are emitted.</summary>
    sealed class Pending
    {
        public List<float[]> Samples { get; }
        public int Channels { get; }
        public int Height { get; }
        public int Width { get; }
        readonly List<float[]> _masks;
        readonly int _maskHeight, _maskWidth;
        readonly List<int> _batchIndex;

        public Pending(LatentBatch latent, (List<float[]> Rows, int Height, int Width) mask, int[] batchIndex)
        {
            int per = latent.Channels * latent.Height * latent.Width;
            Samples = [.. Enumerable.Range(0, latent.Batch).Select(b => latent.Data.AsSpan(b * per, per).ToArray())];
            (Channels, Height, Width) = (latent.Channels, latent.Height, latent.Width);
            (_masks, _maskHeight, _maskWidth) = mask;
            _batchIndex = [.. batchIndex];
        }

        public void Append(Pending next)
        {
            if (next.Channels != Channels) throw Batching.SizeMismatch(Channels, next.Channels, 1);
            if (next._maskHeight != _maskHeight) throw Batching.SizeMismatch(_maskHeight, next._maskHeight, 1);
            if (next._maskWidth != _maskWidth) throw Batching.SizeMismatch(_maskWidth, next._maskWidth, 1);
            Samples.AddRange(next.Samples);
            _masks.AddRange(next._masks);
            _batchIndex.AddRange(next._batchIndex);
        }

        /// <summary>Adds a latent of the first <paramref name="size"/> rows (nothing when there are no samples left).</summary>
        public void EmitInto(List<LatentBatch> output, int size)
        {
            int n = Math.Min(size, Samples.Count);
            if (n == 0) return;
            int per = Channels * Height * Width;
            var data = new float[n * per];
            for (int i = 0; i < n; i++) Array.Copy(Samples[i], 0, data, i * per, per);

            int maskCount = Math.Min(size, _masks.Count), plane = _maskHeight * _maskWidth;
            var mask = new MaskBatch(maskCount, _maskHeight, _maskWidth);
            double sum = 0;
            for (int i = 0; i < maskCount; i++)
            {
                Array.Copy(_masks[i], 0, mask.Data, i * plane, plane);
                foreach (var v in _masks[i]) sum += v;
            }
            output.Add(new LatentBatch(n, Channels, Height, Width, data)
            {
                NoiseMask = sum == (double)maskCount * plane ? null : mask,
                BatchIndex = [.. _batchIndex.Take(size)],
            });
        }

        public void DropFirst(int size)
        {
            Samples.RemoveRange(0, Math.Min(size, Samples.Count));
            _masks.RemoveRange(0, Math.Min(size, _masks.Count));
            _batchIndex.RemoveRange(0, Math.Min(size, _batchIndex.Count));
        }
    }
}
