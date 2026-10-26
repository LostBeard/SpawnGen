using System.Text.Json;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Start Loop / End Loop against ComfyUI itself: Fixtures/loops/make_loops.py ran seven graphs through a ComfyUI server (simple mode with a carried value and
/// accumulate, For mode, List mode, nested loops, a termination in the body, an End Loop without a Start, a body that escapes) and loops.json holds what
/// came out: the Preview as Text values, or the loop validation errors by node. SpawnGen must give the same.
/// </summary>
public class LoopTests
{
    static readonly JsonElement Cases = JsonDocument.Parse(File.ReadAllText(
        Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "loops", "loops.json"))).RootElement;

    static PromptExecutor Executor() => new(new NodeRegistry().AddAssembly(typeof(StartLoop).Assembly));

    public static IEnumerable<string> Names() => Cases.EnumerateObject().Select(p => p.Name);

    [TestCaseSource(nameof(Names))]
    public async Task Loops_give_ComfyUIs_results(string name)
    {
        var c = Cases.GetProperty(name);
        var graph = PromptJson.Parse(c.GetProperty("graph"));
        var comfy = c.GetProperty("comfy");
        var executor = Executor();
        if (comfy.TryGetProperty("errors", out var errors))
        {
            var ex = Assert.ThrowsAsync<PromptValidationException>(() => executor.ExecuteAsync(graph))!;
            var ours = ex.Result.Errors.Where(e => e.Input == "loop boundary").ToDictionary(e => e.NodeId!, e => e.Code);
            var theirs = errors.EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetString()!);
            Assert.That(ours, Is.EquivalentTo(theirs));
            return;
        }
        var result = await executor.ExecuteAsync(graph);
        foreach (var output in comfy.GetProperty("texts").EnumerateObject())
        {
            var expected = output.Value.EnumerateArray().Select(v => v.GetString()).ToList();
            var ui = result.Outputs[output.Name];
            // A list reaching Preview as Text runs it once per item; the previews are joined in order.
            var shown = ((System.Collections.IEnumerable)ui.Ui!["text"]!).Cast<object?>().Select(v => v?.ToString()).ToList();
            Assert.That(shown, Is.EqualTo(expected), $"node {output.Name}");
        }
    }

    [Test]
    public async Task The_termination_runs_every_iteration_and_shows_on_its_own_node()
    {
        var graph = PromptJson.Parse(Cases.GetProperty("termination").GetProperty("graph"));
        var observer = new Recorder();
        await Executor().ExecuteAsync(graph, observer: observer);
        // Body node 6 (Preview as Text, connected as a termination) ran once per iteration, reported as node 6 (its copies show as the original).
        Assert.That(observer.Executed.Count(id => id == "6"), Is.EqualTo(3));
        Assert.That(observer.Texts["6"], Is.EqualTo(new[] { "100", "101", "102" }));
        // The progress text goes to the Start Loop.
        Assert.That(observer.Progress.Where(p => p.Node == "1").Select(p => p.Message), Is.SupersetOf(new[] { "Iteration 1 / 3", "Iteration 2 / 3", "Iteration 3 / 3" }));
        Assert.That(observer.Executed, Does.Not.Contain("1.0.0.result"), "added nodes report as the node they show as");
    }

    sealed class Recorder : IExecutionObserver
    {
        public List<string> Executed { get; } = [];
        public Dictionary<string, List<string>> Texts { get; } = new();
        public List<(string Node, string? Message)> Progress { get; } = [];
        public void OnNodeExecuted(string nodeId, NodeOutputs outputs)
        {
            Executed.Add(nodeId);
            if (outputs.Ui is { } ui && ui.TryGetValue("text", out var t) && t is System.Collections.IEnumerable items)
            {
                if (!Texts.TryGetValue(nodeId, out var list)) Texts[nodeId] = list = [];
                list.AddRange(items.Cast<object?>().Select(v => v?.ToString() ?? ""));
            }
        }
        public void OnNodeProgress(string nodeId, double value, double max, string? message) => Progress.Add((nodeId, message));
    }
}
