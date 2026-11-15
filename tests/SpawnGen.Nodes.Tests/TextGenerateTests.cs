using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Generate Text (ComfyUI's TextGenerate) against ComfyUI's runs of its llm_qwen3_text_gen graph (CLIPLoader qwen_3_4b type ltxv, greedy; measured
/// 2026-11-14 with the ids it fed the model: _scratch/probe/textgen, _DevComms 2026-11-14-0600).
/// </summary>
public class TextGenerateTests
{
    [Test]
    public void The_text_splits_into_answer_and_thinking_as_ComfyUIs()
    {
        // Thinking off: the answer alone.
        Assert.That(TextGenerate.Split("A red apple is round.", false), Is.EqualTo(("A red apple is round.", "")));
        // Thinking on, cut off before </think> (ComfyUI's 160-token run): all thinking, no answer, the tag taken off.
        Assert.That(TextGenerate.Split("<think>\nOkay, so I need to figure out", true), Is.EqualTo(("", "Okay, so I need to figure out")));
        // Thinking on and finished.
        Assert.That(TextGenerate.Split("<think>\nSeventeen times twenty-three.\n</think>\n\n391", true), Is.EqualTo(("391", "Seventeen times twenty-three.")));
    }

    [Test]
    public void The_schema_is_ComfyUIs()
    {
        var schema = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly).GetSchema("TextGenerate")!;
        Assert.That(schema.DisplayName, Is.EqualTo("Generate Text"));
        Assert.That(schema.Outputs.Select(o => o.Name), Is.EqualTo(new[] { "generated_text", "thinking" }));
        Assert.That(schema.FindInput("max_length")!.Widget!.Default, Is.EqualTo(512L));
        Assert.That(schema.FindInput("thinking")!.Required, Is.False);
        Assert.That(TextGenerate.SamplingModes().Select(o => o.Key), Is.EqualTo(new[] { "on", "off" }));
    }

    sealed class Capture : IExecutionObserver
    {
        public Dictionary<string, NodeOutputs> Outputs { get; } = [];
        public void OnNodeExecuted(string nodeId, NodeOutputs outputs) => Outputs[nodeId] = outputs;
    }

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }

    /// <summary>
    /// The template's graph, greedy, thinking off, on the GPU. On an 8 GB card the library loads the 16-bit weights 8-bit (Q8_0), and greedy output then
    /// follows ComfyUI's for its first tokens (the library measured 13 of 23); the test checks those and reports the rest.
    /// </summary>
    [Test, Explicit("Needs SPAWNGEN_MODELS_ROOT with text_encoders/qwen_3_4b.safetensors, a GPU, and Qwen3's tokenizer files (downloaded on first use).")]
    public async Task Greedy_text_follows_ComfyUIs()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        ModelFolders.Scan(root!);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var executor = new PromptExecutor(new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly), new Services(runtime));
        var report = new List<string>();
        foreach (var (text, maxLength, thinking, comfyAnswer, comfyPrefix) in new[]
        {
            ("Describe a red apple in one sentence.", 48L, false,
                "A red apple is a round, juicy fruit with a smooth, shiny red skin and a sweet, tart flavor.", "A red apple is a round, juicy fruit with a smooth,"),
            ("What is 17 times 23?", 40L, true, "", "Okay, so I need to figure out what 1"),
        })
        {
            var g = new PromptGraph();
            g.Add("clip", "CLIPLoader", ("clip_name", "qwen_3_4b.safetensors"), ("type", "ltxv"), ("device", "default"));
            g.Add("gen", "TextGenerate", ("clip", PromptInput.Link("clip")), ("prompt", text), ("max_length", maxLength), ("sampling_mode", "off"), ("thinking", thinking),
                ("use_default_template", true), ("mtp", "off"));
            g.Add("show", "PreviewAny", ("source", PromptInput.Link("gen")));
            var capture = new Capture();
            var clock = System.Diagnostics.Stopwatch.StartNew();
            await executor.ExecuteAsync(g, null, capture);
            var answer = (string)capture.Outputs["gen"].Values[0]!;
            var thought = (string)capture.Outputs["gen"].Values[1]!;
            var shown = thinking ? thought : answer;
            report.Add($"thinking {(thinking ? "on" : "off")}: {clock.Elapsed.TotalSeconds:F1} s; answer \"{answer}\"; thinking \"{(thought.Length > 80 ? thought[..80] + "..." : thought)}\"");
            Assert.That(shown, Does.StartWith(comfyPrefix), "ComfyUI's first 13 greedy tokens");
            if (thinking) Assert.That(answer, Is.EqualTo(""), "a thinking run cut off before </think> has no answer (ComfyUI's)");
            report.Add(answer == comfyAnswer || thinking ? "  = ComfyUI's text" : $"  ComfyUI's: \"{comfyAnswer}\"");
        }
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name} ({runtime.Accelerator.MemorySize >> 20} MB):\n" + string.Join("\n", report));
    }
    /// <summary>
    /// Sampling "on" reaches the library's min_p and presence_penalty. min_p 1.0 keeps only the most likely token, so even at temperature 2 with
    /// top_k and top_p open the text is the greedy text; presence_penalty 5 (applied before min-p) pushes repeated tokens down, so the text changes.
    /// </summary>
    [Test, Explicit("Needs SPAWNGEN_MODELS_ROOT with text_encoders/qwen_3_4b.safetensors, a GPU, and Qwen3's tokenizer files (downloaded on first use).")]
    public async Task Sampling_takes_min_p_and_presence_penalty()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        ModelFolders.Scan(root!);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var executor = new PromptExecutor(new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly), new Services(runtime));
        async Task<string> Run(params (string, object?)[] sampling)
        {
            var g = new PromptGraph();
            g.Add("clip", "CLIPLoader", ("clip_name", "qwen_3_4b.safetensors"), ("type", "ltxv"), ("device", "default"));
            var inputs = new List<(string, object?)> { ("clip", PromptInput.Link("clip")), ("prompt", "Describe a red apple in one sentence."), ("max_length", 32L),
                ("thinking", false), ("use_default_template", true), ("mtp", "off") };
            inputs.AddRange(sampling);
            g.Add("gen", "TextGenerate", inputs.ToArray());
            g.Add("show", "PreviewAny", ("source", PromptInput.Link("gen")));
            var capture = new Capture();
            await executor.ExecuteAsync(g, null, capture);
            return (string)capture.Outputs["gen"].Values[0]!;
        }
        (string, object?)[] Hot(double presence) => [("sampling_mode", "on"), ("sampling_mode.temperature", 2.0), ("sampling_mode.top_k", 0L),
            ("sampling_mode.top_p", 1.0), ("sampling_mode.min_p", 1.0), ("sampling_mode.repetition_penalty", 1.0), ("sampling_mode.seed", 7L),
            ("sampling_mode.presence_penalty", presence)];
        var greedy = await Run(("sampling_mode", "off"));
        var minP = await Run(Hot(0.0));
        var presence = await Run(Hot(5.0));
        TestContext.Out.WriteLine($"greedy: \"{greedy}\"\nmin_p 1, T 2: \"{minP}\"\n+ presence 5: \"{presence}\"");
        Assert.That(minP, Is.EqualTo(greedy), "min_p 1.0 keeps only the top token");
        Assert.That(presence, Is.Not.EqualTo(greedy), "presence_penalty 5 changes the text");
    }
}
