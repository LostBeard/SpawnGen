using SpawnGen.Engine.Graph;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>Speech models as nodes (explicit: GPU and downloads on first use): text to speech, then the speech back to text.</summary>
public class SpeechTests
{
    [Test]
    public void Speech_nodes_validate_in_a_graph()
    {
        var registry = new SpawnGen.Engine.Nodes.NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(SpawnGen.Nodes.ML.KSampler).Assembly);
        var g = new PromptGraph();
        g.Add("1", "KokoroTextToSpeech", ("text", "Hello"), ("voice", "af_heart"), ("speed", 1.0));
        g.Add("2", "SpeechToText", ("audio", PromptInput.Link("1")));
        g.Add("3", "PreviewAny", ("source", PromptInput.Link("2")));
        g.Add("4", "SaveAudio", ("audio", PromptInput.Link("1")), ("filename_prefix", "tts"));
        var result = new SpawnGen.Engine.Execution.PromptExecutor(registry).Validate(g);
        Assert.That(result.Errors, Is.Empty, string.Join("\n", result.Errors.Select(e => e.Message)));
    }

    [Test, Explicit("Needs SPAWNGEN_MODELS, a GPU with 4+ GB free and downloads on first use (Kokoro 330 MB, Parakeet 2.6 GB).")]
    public async Task Kokoro_speaks_and_Parakeet_hears_it_back()
    {
        const string sentence = "Hello from SpawnGen. This is a test of speech.";
        var files = await VisionTests.RunAsync("speech", () =>
        {
            var g = new PromptGraph();
            g.Add("1", "KokoroTextToSpeech", ("text", sentence), ("voice", "af_heart"), ("speed", 1.0));
            g.Add("2", "SaveAudio", ("audio", PromptInput.Link("1")), ("filename_prefix", "speech"));
            g.Add("3", "SpeechToText", ("audio", PromptInput.Link("1")));
            g.Add("4", "CaptureText", ("text", PromptInput.Link("3")));
            return g;
        }, extension: "wav");
        TestContext.Out.WriteLine("heard: " + VisionTests.CaptureText.Last);
        var clip = WavFile.Decode(File.ReadAllBytes(files.Single()));
        double rms = Math.Sqrt(clip.Channels[0].Average(s => (double)s * s));
        TestContext.Out.WriteLine($"{clip.Seconds:F2} s at {clip.SampleRate} Hz, rms {rms:F3}");
        Assert.That(clip.SampleRate, Is.EqualTo(24000));
        Assert.That(clip.Seconds, Is.InRange(1.5, 8.0), "a sentence of this length is a few seconds");
        Assert.That(rms, Is.GreaterThan(0.01), "not silence");
        var heard = VisionTests.CaptureText.Last.ToLowerInvariant();
        Assert.That(heard, Does.Contain("test").And.Contain("speech"), "the recogniser understands what the synthesiser said");
    }
}
