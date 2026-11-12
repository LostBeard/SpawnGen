using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Tests;

/// <summary>
/// What a queued prompt runs, as ComfyUI's /prompt decides it (measured against its server): each output node is checked on its own, broken ones
/// are left out and reported while the rest runs; partial_execution_targets narrows the outputs; nothing left means "prompt has no outputs".
/// </summary>
public class AdmissionTests
{
    static readonly NodeRegistry Registry = new NodeRegistry().AddAssembly(typeof(AdmissionTests).Assembly);
    static PromptInput L(string id, int i = 0) => PromptInput.Link(id, i);

    /// <summary>2 is fine; 6 misses its input and feeds 4 and 7; 5 misses its own input.</summary>
    static PromptGraph Graph()
    {
        var g = new PromptGraph();
        g.Add("1", "IntSource");
        g.Add("2", "Sink", ("value", L("1")));
        g.Add("6", "AddNumbers");
        g.Add("4", "Sink", ("value", L("6")));
        g.Add("7", "Sink", ("value", L("6")));
        g.Add("5", "Sink");
        return g;
    }

    [Test]
    public void Broken_outputs_are_left_out_and_the_rest_runs()
    {
        var a = PromptAdmission.Check(Registry, Graph());
        Assert.That(a.Refused, Is.False);
        Assert.That(a.Outputs, Is.EqualTo(new[] { "2" }));
        Assert.That(a.Failures.Select(f => f.OutputId), Is.EqualTo(new[] { "4", "7", "5" }));
        Assert.That(a.Failures.Where(f => f.OutputId is "4" or "7").All(f => f.Result.Errors.Any(e => e.NodeId == "6" && e.Input == "a")), Is.True,
            "the error is on the node that has it, for every output it stops");
    }

    [Test]
    public void Targets_narrow_the_outputs_to_listed_output_nodes()
    {
        Assert.That(PromptAdmission.Check(Registry, Graph(), ["2"]), Has.Property("Outputs").EqualTo(new[] { "2" }).And.Property("Failures").Empty);
        var some = PromptAdmission.Check(Registry, Graph(), ["5", "2", "2"]);
        Assert.That((some.Outputs, some.Failures.Single().OutputId), Is.EqualTo((new[] { "2" }, "5")));
        foreach (var targets in new[] { new[] { "1" }, new[] { "99" }, [] })
        {
            var none = PromptAdmission.Check(Registry, Graph(), targets);
            Assert.That(none.Refused && none.Failures.Count == 0, Is.True, string.Join(",", targets));
            Assert.That(PromptAdmission.Refusal(none).Result.Errors.Single().Code, Is.EqualTo("prompt_no_outputs"));
        }
    }

    [Test]
    public void Nothing_that_validates_is_refused_with_every_outputs_errors()
    {
        var a = PromptAdmission.Check(Registry, Graph(), ["4", "5"]);
        Assert.That(a.Refused, Is.True);
        var refusal = PromptAdmission.Refusal(a);
        Assert.That(refusal.Failures.Select(f => f.OutputId), Is.EqualTo(new[] { "4", "5" }));
        Assert.That(refusal.Result.Errors.Select(e => e.NodeId), Is.EquivalentTo(new[] { "6", "5" }));

        var empty = new PromptGraph();
        empty.Add("1", "IntSource");
        Assert.That(PromptAdmission.Refusal(PromptAdmission.Check(Registry, empty)).Result.Errors.Single().Code, Is.EqualTo("prompt_no_outputs"));
    }
}
