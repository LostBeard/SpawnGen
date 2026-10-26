using System.Text.Json;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Rebatch Latents / Rebatch Images against ComfyUI's own code run on the same list (Fixtures/rebatch/make_rebatch.py -> rebatch.json): which item lands in
/// which batch, the batch indexes, and which masks survive.
/// </summary>
public class RebatchTests
{
    static JsonElement Expected() => JsonDocument.Parse(File.ReadAllText(Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "rebatch", "rebatch.json"))).RootElement;

    static LatentBatch Latent(int item, int batch, int h, int w, MaskBatch? mask = null, int[]? batchIndex = null)
    {
        var l = new LatentBatch(batch, 4, h, w) { NoiseMask = mask, BatchIndex = batchIndex };
        int per = 4 * h * w;
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < per; i++) l.Data[b * per + i] = item * 1000 + b * 100 + i % 100;
        return l;
    }

    [Test]
    public void Latents_are_regrouped_as_ComfyUI_regroups_them()
    {
        var half = MaskBatch.Filled(1, 64, 64, 1f);
        for (int y = 0; y < 64; y++) for (int x = 0; x < 32; x++) half.Data[y * 64 + x] = 0f;
        var latents = new[] { Latent(0, 3, 8, 8), Latent(1, 2, 8, 8, half), Latent(2, 1, 16, 16), Latent(3, 2, 16, 16, batchIndex: [7, 9]) };
        var output = new RebatchLatents().Execute(latents, [2L]);

        var expected = Expected().GetProperty("latents").EnumerateArray().ToList();
        Assert.That(output, Has.Count.EqualTo(expected.Count));
        for (int i = 0; i < expected.Count; i++)
        {
            var e = expected[i];
            var o = output[i];
            var shape = e.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            Assert.That(new[] { o.Batch, o.Channels, o.Height, o.Width }, Is.EqualTo(shape), $"output {i}: shape");
            int per = o.Channels * o.Height * o.Width;
            Assert.That(Enumerable.Range(0, o.Batch).Select(b => o.Data[b * per]), Is.EqualTo(e.GetProperty("firsts").EnumerateArray().Select(v => (float)v.GetDouble())), $"output {i}: which latents");
            Assert.That(o.BatchIndex, Is.EqualTo(e.GetProperty("batch_index").EnumerateArray().Select(v => v.GetInt32())), $"output {i}: batch_index");
            if (e.GetProperty("mask").ValueKind == JsonValueKind.Null) Assert.That(o.NoiseMask, Is.Null, $"output {i}: an all-ones mask is dropped");
            else
            {
                var m = e.GetProperty("mask");
                var mshape = m.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
                Assert.That(new[] { o.NoiseMask!.Batch, 1, o.NoiseMask.Height, o.NoiseMask.Width }, Is.EqualTo(mshape), $"output {i}: mask shape");
                Assert.That(o.NoiseMask.Data.Sum(), Is.EqualTo(m.GetProperty("sum").GetDouble()).Within(1e-3), $"output {i}: mask");
            }
        }
    }

    [Test]
    public void Images_are_regrouped_as_ComfyUI_regroups_them()
    {
        static ImageBatch Images(int item, int batch)
        {
            var img = new ImageBatch(batch, 4, 4, 3);
            for (int b = 0; b < batch; b++) for (int i = 0; i < 48; i++) img.Data[b * 48 + i] = (item * 10 + b) / 100f;
            return img;
        }
        var output = new RebatchImages().Execute([Images(0, 3), Images(1, 2)], [2L]);
        var expected = Expected().GetProperty("images").EnumerateArray().ToList();
        Assert.That(output, Has.Count.EqualTo(expected.Count));
        for (int i = 0; i < expected.Count; i++)
        {
            var e = expected[i];
            Assert.That(output[i].Batch, Is.EqualTo(e.GetProperty("shape")[0].GetInt32()));
            Assert.That(Enumerable.Range(0, output[i].Batch).Select(b => (int)MathF.Round(output[i].Data[b * 48] * 100)),
                Is.EqualTo(e.GetProperty("firsts").EnumerateArray().Select(v => v.GetInt32())), $"output {i}");
        }
        Assert.That(() => new RebatchImages().Execute([Images(0, 1), new ImageBatch(1, 8, 8, 3)], [2L]), Throws.ArgumentException, "different sizes cannot share a batch");
    }

    [Test]
    public void Conditioning_multiply_scales_the_embeddings_and_the_pooled_output()
    {
        var c = new Conditioning([new CondEntry([1f, -2f, 0.5f], [1, 1, 3]) { Pooled = [4f], Strength = 0.7f }]);
        var m = new ConditioningMultiply().Execute(c, 1.5);
        Assert.That(m.Entries[0].Embeddings, Is.EqualTo(new[] { 1.5f, -3f, 0.75f }));
        Assert.That(m.Entries[0].Pooled, Is.EqualTo(new[] { 6f }));
        Assert.That(m.Entries[0].Strength, Is.EqualTo(0.7f), "everything else is kept");
        Assert.That(c.Entries[0].Embeddings, Is.EqualTo(new[] { 1f, -2f, 0.5f }), "the input is not changed");
    }
}
