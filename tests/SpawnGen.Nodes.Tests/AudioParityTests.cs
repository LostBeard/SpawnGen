using System.IO.Compression;
using System.Text.Json;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// The model-free audio nodes against what ComfyUI's own node classes gave for the same waveforms (tools/parity/make_audio_fixtures.py ->
/// Fixtures/comfy/comfy_audio_nodes.json.gz, torch 2.14 / torchaudio 2.11 CPU): samples, lengths, channel counts and sample rates, and the same errors.
/// SpawnGen's AUDIO holds one clip (no batch dimension), so ComfyUI's batch-of-2 cases are not run.
/// </summary>
public class AudioParityTests
{
    public sealed record Tensor(int[] Shape, float[] Data);

    public sealed record Case(int Index, JsonElement Json, Tensor[] Tensors)
    {
        public string Node => Json.GetProperty("node").GetString()!;
        public override string ToString() => $"{Node} {Json.GetProperty("case").GetString()}";
    }

    static List<Case>? _cases;

    static List<Case> Load()
    {
        lock (typeof(AudioParityTests))
        {
            if (_cases is not null) return _cases;
            var path = Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "comfy", "comfy_audio_nodes.json.gz");
            using var gz = new GZipStream(File.OpenRead(path), CompressionMode.Decompress);
            using var doc = JsonDocument.Parse(gz);
            var tensors = doc.RootElement.GetProperty("tensors").EnumerateArray().Select(t =>
            {
                var raw = Convert.FromBase64String(t.GetProperty("f32").GetString()!);
                var data = new float[raw.Length / 4];
                Buffer.BlockCopy(raw, 0, data, 0, raw.Length);
                return new Tensor([.. t.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32())], data);
            }).ToArray();
            return _cases = [.. doc.RootElement.GetProperty("cases").EnumerateArray().Select((c, i) => new Case(i, c.Clone(), tensors))];
        }
    }

    static bool Batched(Case c) =>
        c.Json.GetProperty("inputs").EnumerateObject().Any(i => c.Tensors[i.Value.GetProperty("tensor").GetInt32()].Shape[0] != 1);

    static IEnumerable<TestCaseData> Cases() =>
        Load().Where(c => !Batched(c)).Select(c => new TestCaseData(c).SetName($"{c.Node}_{c.Index:D3}"));

    static AudioClip Clip(Case c, JsonElement a)
    {
        var t = c.Tensors[a.GetProperty("tensor").GetInt32()];
        int channels = t.Shape[1], n = t.Shape[2];
        return new AudioClip([.. Enumerable.Range(0, channels).Select(ch => t.Data.AsSpan(ch * n, n).ToArray())], a.GetProperty("sample_rate").GetInt32());
    }

    static AudioClip In(Case c, string name) => Clip(c, c.Json.GetProperty("inputs").GetProperty(name));
    static JsonElement Arg(Case c, string name) => c.Json.GetProperty("args").GetProperty(name);

    static void Same(AudioClip? actual, Case c, JsonElement expected, double tolerance, string what)
    {
        var t = c.Tensors[expected.GetProperty("tensor").GetInt32()];
        Assert.That(actual, Is.Not.Null, what);
        Assert.That((actual!.ChannelCount, actual.Length, actual.SampleRate), Is.EqualTo((t.Shape[1], t.Shape[2], expected.GetProperty("sample_rate").GetInt32())), what + " shape");
        double worst = 0;
        int at = -1;
        for (int ch = 0; ch < t.Shape[1]; ch++)
            for (int i = 0; i < t.Shape[2]; i++)
            {
                double d = Math.Abs(actual.Channels[ch][i] - t.Data[ch * t.Shape[2] + i]);
                if (!(d <= worst)) { worst = d; at = ch * t.Shape[2] + i; }
            }
        Assert.That(worst, Is.LessThanOrEqualTo(tolerance), $"{what} worst at {at}");
    }

    /// <summary>The same Python error (type and message); an error torch raised, or ComfyUI's own UnboundLocalError for an unknown merge method, only needs to be an error.</summary>
    static void Expect(Case c, Action run)
    {
        if (!c.Json.TryGetProperty("error", out var error))
        {
            run();
            return;
        }
        var ex = Assert.Catch(() => run(), $"ComfyUI raised {error.GetString()}: {c.Json.GetProperty("message").GetString()}");
        if (error.GetString() is "RuntimeError" or "UnboundLocalError") return;
        Assert.That(ex, Is.InstanceOf<PyException>(), ex!.Message);
        Assert.That(((PyException)ex).PyType, Is.EqualTo(error.GetString()));
        Assert.That(((PyException)ex).PyMessage, Is.EqualTo(c.Json.GetProperty("message").GetString()));
    }

    [TestCaseSource(nameof(Cases))]
    public void Matches_ComfyUI(Case c)
    {
        var o = c.Json.TryGetProperty("out", out var outs) ? outs : default;
        Expect(c, () =>
        {
            switch (c.Node)
            {
                case "TrimAudioDuration":
                    Same(new TrimAudioDuration().Execute(In(c, "audio"), Arg(c, "start_index").GetDouble(), Arg(c, "duration").GetDouble()), c, o[0], 0, "trimmed");
                    break;
                case "SplitAudioChannels":
                    var (left, right) = new SplitAudioChannels().Execute(In(c, "audio"));
                    Same(left, c, o[0], 0, "left");
                    Same(right, c, o[1], 0, "right");
                    break;
                case "JoinAudioChannels":
                    // Resampling is torch's float32 sinc kernel arithmetic; SpawnGen's sums can round differently in the last bits.
                    Same(new JoinAudioChannels().Execute(In(c, "audio_left"), In(c, "audio_right")), c, o[0], 2e-6, "joined");
                    break;
                case "AudioConcat":
                    Same(new AudioConcat().Execute(In(c, "audio1"), In(c, "audio2"), Arg(c, "direction").GetString()!), c, o[0], 2e-6, "concatenated");
                    break;
                case "AudioMerge":
                    Same(new AudioMerge().Execute(In(c, "audio1"), In(c, "audio2"), Arg(c, "merge_method").GetString()!), c, o[0], 2e-6, "merged");
                    break;
                case "AudioAdjustVolume":
                    Same(new AudioAdjustVolume().Execute(In(c, "audio"), Arg(c, "volume").GetInt32()), c, o[0], 1e-6, "adjusted");
                    break;
                case "EmptyAudio":
                    Same(new EmptyAudio().Execute(Arg(c, "duration").GetDouble(), Arg(c, "sample_rate").GetInt32(), Arg(c, "channels").GetInt32()), c, o[0], 0, "empty");
                    break;
                case "AudioEqualizer3Band":
                    Same(new AudioEqualizer3Band().Execute(In(c, "audio"), Arg(c, "low_gain_dB").GetDouble(), Arg(c, "low_freq").GetInt32(),
                        Arg(c, "mid_gain_dB").GetDouble(), Arg(c, "mid_freq").GetInt32(), Arg(c, "mid_q").GetDouble(), Arg(c, "high_gain_dB").GetDouble(),
                        Arg(c, "high_freq").GetInt32()), c, o[0], 0, "equalized");
                    break;
                default:
                    Assert.Fail("no runner for " + c.Node);
                    break;
            }
        });
    }

    [Test]
    public void The_fixture_covers_every_node_and_its_errors()
    {
        var cases = Load();
        foreach (var node in new[] { "TrimAudioDuration", "SplitAudioChannels", "JoinAudioChannels", "AudioConcat", "AudioMerge", "AudioAdjustVolume", "EmptyAudio", "AudioEqualizer3Band" })
            Assert.That(cases.Count(c => c.Node == node && !c.Json.TryGetProperty("error", out _) && !Batched(c)), Is.GreaterThanOrEqualTo(1), node);
        Assert.That(cases.Count(c => c.Json.TryGetProperty("error", out _)), Is.GreaterThanOrEqualTo(8));
    }
}
