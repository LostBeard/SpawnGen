using System.Text.Json;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Video Linear / Video Triangle CFG Guidance against ComfyUI's own cfg functions (Fixtures/video_cfg/make_video_cfg.py -> video_cfg.json): batches of 1 to 8
/// frames, cfg above and below min_cfg. ComfyUI sees the whole batch at once; SpawnGen's sampler goes frame by frame, so each frame is computed from its
/// place in the batch and must give that frame's slice of ComfyUI's result.
/// </summary>
public class VideoCfgTests
{
    static readonly JsonElement[] Cases = JsonDocument.Parse(File.ReadAllText(
        Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "video_cfg", "video_cfg.json"))).RootElement.EnumerateArray().ToArray();

    public static IEnumerable<int> CaseIndexes() => Enumerable.Range(0, Cases.Length);

    static float[] Floats(JsonElement c, string name) => c.GetProperty(name).EnumerateArray().Select(v => v.GetSingle()).ToArray();

    [TestCaseSource(nameof(CaseIndexes))]
    public void Each_frame_gets_ComfyUIs_cfg(int index)
    {
        var c = Cases[index];
        int batch = c.GetProperty("batch").GetInt32(), perFrame = 4 * 3 * 2;
        double condScale = c.GetProperty("cond_scale").GetDouble(), minCfg = c.GetProperty("min_cfg").GetDouble();
        float[] cond = Floats(c, "cond"), uncond = Floats(c, "uncond"), expected = Floats(c, "out");
        bool linear = c.GetProperty("node").GetString() == "VideoLinearCFGGuidance";
        for (int b = 0; b < batch; b++)
        {
            var context = new PatchContext
            {
                Channels = 4, Height = 3, Width = 2, Sampling = new ModelSampling(SamplingSetupTests.Sd()), BatchIndex = b, BatchSize = batch,
                EvaluateUnscheduled = _ => throw new NotSupportedException(),
            };
            var x = new float[perFrame];
            var args = new CfgFunctionArgs(cond[(b * perFrame)..((b + 1) * perFrame)], uncond[(b * perFrame)..((b + 1) * perFrame)], condScale, 1.0, x, x, x, true, true, context);
            var ours = linear ? GuidancePatches.VideoLinearCfg(args, minCfg) : GuidancePatches.VideoTriangleCfg(args, minCfg);
            Assert.That(ours, Is.EqualTo(expected[(b * perFrame)..((b + 1) * perFrame)]).Within(1e-5f), $"{c.GetProperty("node")} batch {batch} frame {b}");
        }
    }

    [Test]
    public void Linspace_counts_like_torch()
    {
        // torch.linspace(0, 1, 3) is [0, 0.5, 1]; one step gives the start.
        Assert.That(GuidancePatches.Linspace(0f, 1f, 3), Is.EqualTo(new[] { 0f, 0.5f, 1f }));
        Assert.That(GuidancePatches.Linspace(2.5f, 7f, 1), Is.EqualTo(new[] { 2.5f }));
    }
}
