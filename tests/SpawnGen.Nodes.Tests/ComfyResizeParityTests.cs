using System.IO.Compression;
using System.Text.Json;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Resize Image/Mask, bislerp (Upscale Latent) and Resize And Pad Image against black-box measurements of ComfyUI: Fixtures/comfy/comfy_resize_nodes.json.gz,
/// written by tools/parity/make_resize_fixtures.py, which calls the node classes of a ComfyUI checkout on chosen tensors.
/// </summary>
public class ComfyResizeParityTests
{
    public sealed record Case(int Index, JsonElement Json)
    {
        public override string ToString() => $"{Index:D3} {Json.GetProperty("node").GetString()} {Json.GetProperty("args").GetRawText()}";
    }

    static JsonDocument? doc;

    static IEnumerable<TestCaseData> Cases()
    {
        lock (typeof(ComfyResizeParityTests))
        {
            if (doc is null)
            {
                var path = Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "comfy", "comfy_resize_nodes.json.gz");
                using var gz = new GZipStream(File.OpenRead(path), CompressionMode.Decompress);
                doc = JsonDocument.Parse(gz);
            }
        }
        return doc.RootElement.GetProperty("cases").EnumerateArray().Select((c, i) => new TestCaseData(new Case(i, c)).SetName($"Resize_{i:D3}"));
    }

    static int[] Shape(JsonElement t) => t.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
    static float[] Data(JsonElement t) => t.GetProperty("data").EnumerateArray().Select(v => (float)v.GetDouble()).ToArray();

    static object Input(JsonElement t)
    {
        var s = Shape(t);
        var data = t.TryGetProperty("data", out _) ? Data(t) : null;
        if (s.Length == 3) return new MaskBatch(s[0], s[1], s[2], data);
        return new ImageBatch(s[0], s[1], s[2], s[3], data ?? Enumerable.Repeat(0.5f, s.Aggregate(1, (a, b) => a * b)).ToArray());
    }

    [TestCaseSource(nameof(Cases))]
    public void Matches_ComfyUI(Case test)
    {
        var c = test.Json;
        var args = c.GetProperty("args");
        var expected = c.GetProperty("out");
        string method = c.GetProperty("method").GetString()!;
        Func<HostTensor> run = c.GetProperty("node").GetString() switch
        {
            "ResizeImageMaskNode" => () => (HostTensor)new ResizeImageMaskNode().Execute(Input(c.GetProperty("input")), Choice(args), method),
            "LatentUpscale" => () =>
            {
                var s = Shape(c.GetProperty("input"));
                var latent = new LatentBatch(s[0], s[1], s[2], s[3], Data(c.GetProperty("input")));
                return new LatentUpscale().Execute(latent, method, args.GetProperty("width").GetInt32(), args.GetProperty("height").GetInt32());
            },
            "ResizeAndPadImage" => () => new ResizeAndPadImage().Execute((ImageBatch)Input(c.GetProperty("input")),
                args.GetProperty("target_width").GetInt32(), args.GetProperty("target_height").GetInt32(), args.GetProperty("padding_color").GetString()!, method),
            var other => throw new InvalidOperationException(other),
        };
        if (expected.TryGetProperty("error", out var error))
        {
            var ex = Assert.Throws<ArgumentException>(() => run());
            Assert.That(ex!.Message, Is.EqualTo(error.GetString()!.Split(": ", 2)[1]));
            return;
        }
        var result = run();
        Assert.That(result.Shape, Is.EqualTo(Shape(expected)));
        if (!expected.TryGetProperty("data", out _)) return;
        var want = Data(expected);
        double worst = 0;
        for (int i = 0; i < want.Length; i++) worst = Math.Max(worst, Math.Abs(result.Data[i] - want[i]));
        // Lanczos goes through 8-bit Pillow (exact); PyTorch modes agree to float32 rounding. bislerp agrees to its float32 trigonometry: 1.4e-6 on
        // the random latents, 2.5e-5 for directions just short of opposite (dot -0.9999895, sin of the angle 0.005), where slerp is ill-conditioned.
        Assert.That(worst, Is.LessThanOrEqualTo(method == "lanczos" ? 0 : method == "bislerp" ? 5e-5 : 2e-6));
    }

    static DynamicChoice Choice(JsonElement args)
    {
        var values = new Dictionary<string, object?>();
        foreach (var p in args.EnumerateObject())
        {
            if (p.Name == "resize_type") continue;
            values[p.Name] = p.Name == "match"
                ? Input(JsonDocument.Parse($"{{\"shape\": {p.Value.GetRawText()}}}").RootElement)
                : p.Value.ValueKind switch
                {
                    JsonValueKind.Number when p.Value.TryGetInt64(out var l) && !p.Value.GetRawText().Contains('.') => l,
                    JsonValueKind.Number => p.Value.GetDouble(),
                    _ => p.Value.GetString(),
                };
        }
        return new DynamicChoice(args.GetProperty("resize_type").GetString()!, values);
    }
}
