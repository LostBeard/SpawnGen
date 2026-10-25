using System.Text.Json;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// CLIP Text Encode against ComfyUI's own CLIP (Fixtures/clip, comfy_encode.py on the CPU in float32): the token chunks of a long weighted prompt must be ComfyUI's
/// exactly (ids and weights per tower, padding included), and the conditioning (and SDXL's pooled output) must match within fp16-weight rounding.
/// </summary>
public class ClipComfyParityTests
{
    [TestCase("v1-5-pruned-emaonly.safetensors", "sd15")]
    [TestCase("sd_xl_base_1.0.safetensors", "sdxl")]
    [Explicit("Needs SPAWNGEN_MODELS_ROOT (a models folder with checkpoints/<file>) and a GPU.")]
    public async Task Prompts_encode_like_ComfyUI(string file, string fixture)
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        CheckpointCatalog.ScanCheckpointFiles(root!);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var ckpt = await runtime.GetCheckpointAsync(file);
        var dir = Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "clip");
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, fixture + ".json")));

        int n = 0;
        foreach (var prompt in doc.RootElement.GetProperty("prompts").EnumerateArray())
        {
            var text = prompt.GetProperty("text").GetString()!;
            var chunks = PromptWeights.Chunks(text, piece => CLIPTextEncode.PlainTokens(ckpt, piece), Checkpoint.TokenCount);
            foreach (var tower in prompt.GetProperty("tokens").EnumerateObject())
            {
                // ComfyUI's towers: "l" (CLIP-L, padded with the end token), "g" (OpenCLIP bigG, padded with 0).
                int pad = tower.Name == "g" ? 0 : ClipPadding.End;
                var theirs = tower.Value.EnumerateArray().ToList();
                Assert.That(chunks, Has.Count.EqualTo(theirs.Count), $"prompt {n} tower {tower.Name}: chunk count");
                for (int c = 0; c < theirs.Count; c++)
                {
                    var ids = theirs[c].EnumerateArray().Select(t => t[0].GetInt32()).ToArray();
                    var weights = theirs[c].EnumerateArray().Select(t => (float)t[1].GetDouble()).ToArray();
                    Assert.That(ClipPadding.Pad(chunks[c].Ids, pad), Is.EqualTo(ids), $"prompt {n} tower {tower.Name} chunk {c}: ids");
                    Assert.That(chunks[c].Weights, Is.EqualTo(weights).Within(1e-6), $"prompt {n} tower {tower.Name} chunk {c}: weights");
                }
            }

            var shape = prompt.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            int pooledLength = prompt.GetProperty("pooled").GetInt32();
            var blob = File.ReadAllBytes(Path.Combine(dir, $"{fixture}_{n}.bin"));
            var all = new float[blob.Length / 4];
            Buffer.BlockCopy(blob, 0, all, 0, blob.Length);
            var expected = all[..(all.Length - pooledLength)];

            var ours = await CLIPTextEncode.EncodeAsync(new ClipHandle(ckpt), text, text, CancellationToken.None);
            Assert.That(ours.Shape, Is.EqualTo(shape), $"prompt {n}: shape");
            double rel = RelRms(ours.Context, expected);
            TestContext.Out.WriteLine($"{fixture} prompt {n}: shape [{string.Join(",", shape)}] conditioning relRMS {rel:E2}");
            Assert.That(rel, Is.LessThan(2e-3), $"prompt {n}: conditioning");
            if (pooledLength > 0)
            {
                double relPooled = RelRms(ours.Pooled!, all[^pooledLength..]);
                TestContext.Out.WriteLine($"{fixture} prompt {n}: pooled relRMS {relPooled:E2}");
                Assert.That(relPooled, Is.LessThan(2e-3), $"prompt {n}: pooled");
            }
            n++;
        }
    }

    static double RelRms(float[] a, float[] b)
    {
        Assert.That(a, Has.Length.EqualTo(b.Length));
        double d = 0, s = 0;
        for (int i = 0; i < a.Length; i++) { d += (double)(a[i] - b[i]) * (a[i] - b[i]); s += (double)b[i] * b[i]; }
        return Math.Sqrt(d / s);
    }
}
