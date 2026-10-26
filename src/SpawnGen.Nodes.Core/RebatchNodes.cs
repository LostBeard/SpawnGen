using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

[Node("RebatchImages", Category = "image/batch", DisplayName = "Rebatch Images", InputIsList = true, OutputIsList = [true])]
public sealed class RebatchImages
{
    /// <summary>Every picture of every incoming batch, regrouped into batches of <c>batch_size</c> (the last may be smaller), as ComfyUI's Rebatch Images.</summary>
    public IReadOnlyList<ImageBatch> Execute(IReadOnlyList<ImageBatch> images, [Input(Min = 1, Max = 4096, Default = 1L)] IReadOnlyList<long> batchSize)
    {
        int size = (int)batchSize[0];
        var singles = images.SelectMany(b => Enumerable.Range(0, b.Batch).Select(i => (Batch: b, Index: i))).ToList();
        var result = new List<ImageBatch>();
        for (int start = 0; start < singles.Count; start += size)
        {
            var group = singles.Skip(start).Take(size).ToList();
            var first = group[0].Batch;
            if (group.Any(g => g.Batch.Height != first.Height || g.Batch.Width != first.Width || g.Batch.Channels != first.Channels))
                throw new ArgumentException("Rebatch Images joins pictures of different sizes into one batch; resize them first.", nameof(images));
            var batch = new ImageBatch(group.Count, first.Height, first.Width, first.Channels);
            int per = first.Height * first.Width * first.Channels;
            for (int i = 0; i < group.Count; i++) Array.Copy(group[i].Batch.Data, group[i].Index * per, batch.Data, i * per, per);
            result.Add(batch);
        }
        return result;
    }
}

[Node("RebatchLatents", Category = "model/latent/batch", DisplayName = "Rebatch Latents", InputIsList = true, OutputIsList = [true])]
public sealed class RebatchLatents
{
    /// <summary>
    /// Every latent of every incoming batch, regrouped into batches of <c>batch_size</c>, as ComfyUI's Rebatch Latents: a change of latent size closes the batch
    /// being filled, each latent keeps its noise mask (a missing one counts as all ones at the picture's size, and is dropped again when a whole batch has none)
    /// and its batch_index (which noise of the seed it gets; by default its position in the whole list).
    /// </summary>
    public IReadOnlyList<LatentBatch> Execute(IReadOnlyList<LatentBatch> latents, [Input(Min = 1, Max = 4096, Default = 1L)] IReadOnlyList<long> batchSize)
    {
        int size = (int)batchSize[0];
        var output = new List<LatentBatch>();
        Part? current = null;
        int processed = 0;
        foreach (var latent in latents)
        {
            var next = Part.Of(latent, processed);
            processed += next.Count;
            if (current is null) current = next;
            else if (next.Height != current.Height || next.Width != current.Width)
            {
                output.Add(current.Take(0, size));
                current = next;
            }
            else current = current.Concat(next);
            if (current.Count > size)
            {
                int whole = current.Count / size;
                for (int i = 0; i < whole; i++) output.Add(current.Take(i * size, size));
                current = whole * size < current.Count ? current.Slice(whole * size, current.Count - whole * size) : null;
            }
        }
        if (current is not null) output.Add(current.Take(0, size));
        return output;
    }

    /// <summary>A run of latents being regrouped: samples, one mask per latent (picture size) and the batch indexes, as ComfyUI's (samples, mask, batch_inds).</summary>
    sealed record Part(int Channels, int Height, int Width, List<float[]> Samples, List<float[]> Masks, int MaskHeight, int MaskWidth, List<int> Indexes)
    {
        public int Count => Samples.Count;

        public static Part Of(LatentBatch l, int offset)
        {
            int per = l.Channels * l.Height * l.Width;
            var samples = Enumerable.Range(0, l.Batch).Select(b => l.Data[(b * per)..((b + 1) * per)]).ToList();
            int mh = l.NoiseMask?.Height ?? l.Height * 8, mw = l.NoiseMask?.Width ?? l.Width * 8;
            var masks = new List<float[]>();
            for (int b = 0; b < l.Batch; b++)
            {
                if (l.NoiseMask is { } m)
                {
                    // ComfyUI repeats a shorter mask batch until it covers the latents. (A mask batch longer than its latents is kept whole by ComfyUI and
                    // then sliced out of step with them; here each latent keeps its own mask.)
                    int plane = m.Height * m.Width, at = b % m.Batch;
                    masks.Add(m.Data[(at * plane)..((at + 1) * plane)]);
                }
                else
                {
                    var ones = new float[mh * mw];
                    Array.Fill(ones, 1f);
                    masks.Add(ones);
                }
            }
            var indexes = l.BatchIndex is { } bi ? bi.ToList() : Enumerable.Range(offset, l.Batch).ToList();
            return new Part(l.Channels, l.Height, l.Width, samples, masks, mh, mw, indexes);
        }

        public Part Concat(Part next)
        {
            if (next.Channels != Channels) throw new ArgumentException("Rebatch Latents joins latents with different channel counts.");
            if (next.MaskHeight != MaskHeight || next.MaskWidth != MaskWidth)
                throw new ArgumentException("Rebatch Latents joins latents whose noise masks have different sizes (ComfyUI cannot join them either).");
            return this with { Samples = [.. Samples, .. next.Samples], Masks = [.. Masks, .. next.Masks], Indexes = [.. Indexes, .. next.Indexes] };
        }

        public Part Slice(int start, int count) =>
            this with { Samples = Samples.GetRange(start, count), Masks = Masks.GetRange(start, count), Indexes = Indexes.GetRange(start, count) };

        /// <summary>Latents start .. start + count (fewer at the end) as a LATENT; the mask is dropped when it is all ones, as ComfyUI does.</summary>
        public LatentBatch Take(int start, int count)
        {
            var part = Slice(start, Math.Min(count, Count - start));
            int per = Channels * Height * Width, plane = MaskHeight * MaskWidth;
            var data = new float[part.Count * per];
            for (int i = 0; i < part.Count; i++) part.Samples[i].CopyTo(data, i * per);
            var maskData = new float[part.Count * plane];
            for (int i = 0; i < part.Count; i++) part.Masks[i].CopyTo(maskData, i * plane);
            // ComfyUI: noise_mask.mean() == 1.0 in float32.
            double sum = 0;
            foreach (var v in maskData) sum += v;
            bool allOnes = (float)(sum / maskData.Length) == 1f;
            return new LatentBatch(part.Count, Channels, Height, Width, data)
            {
                NoiseMask = allOnes ? null : new MaskBatch(part.Count, MaskHeight, MaskWidth, maskData),
                BatchIndex = [.. part.Indexes],
            };
        }
    }
}
