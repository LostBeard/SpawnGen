using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>The Autogrow utility nodes run from prompts that name their inputs as ComfyUI does (values.a, values.value0).</summary>
public class FormulaNodeTests
{
    static async Task<object?[]> Run(string classType, params (string, object)[] inputs)
    {
        var g = new PromptGraph();
        g.Add("1", classType, inputs);
        g.Add("2", "PreviewAny", ("source", PromptInput.Link("1")));
        var r = await new PromptExecutor(new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly)).ExecuteAsync(g, new ExecutionOptions { OutputNodeIds = ["2"] });
        return [((IEnumerable<string>)r.Outputs["2"].Ui!["text"]!).Single()];
    }

    [Test]
    public async Task Format_text_fills_named_values_like_str_format()
    {
        Assert.That((await Run("StringFormat", ("values.a", 3L), ("values.b", 0.5), ("f_string", "{a:03d}-{b:.2f}-{c!r}"), ("values.c", "x")))[0], Is.EqualTo("003-0.50-'x'"));
    }

    [Test]
    public async Task Math_expression_evaluates_the_connected_numbers()
    {
        var g = new PromptGraph();
        g.Add("1", "ComfyMathExpression", ("expression", "max(values) * a + b // 2"), ("values.a", 2L), ("values.b", 7L));
        g.Add("2", "PreviewAny", ("source", PromptInput.Link("1", 0)));
        g.Add("3", "PreviewAny", ("source", PromptInput.Link("1", 1)));
        var r = await new PromptExecutor(new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly)).ExecuteAsync(g);
        Assert.That(((IEnumerable<string>)r.Outputs["2"].Ui!["text"]!).Single(), Is.EqualTo("17.0"));
        Assert.That(((IEnumerable<string>)r.Outputs["3"].Ui!["text"]!).Single(), Is.EqualTo("17"));
        var s = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).GetSchema("ComfyMathExpression")!;
        Assert.That(s.Inputs[1].Type.Name, Is.EqualTo("FLOAT,INT,BOOLEAN"));
        Assert.That(s.Outputs.Select(o => o.Name), Is.EqualTo(new[] { "FLOAT", "INT", "BOOL" }));
    }

    [Test]
    public async Task And_and_or_follow_python_truthiness()
    {
        Assert.That((await Run("ComfyAndNode", ("values.value0", 1L), ("values.value1", "x")))[0], Is.EqualTo("True"));
        Assert.That((await Run("ComfyAndNode", ("values.value0", 1L), ("values.value1", "")))[0], Is.EqualTo("False"));
        Assert.That((await Run("ComfyOrNode", ("values.value0", 0L), ("values.value1", 0.0)))[0], Is.EqualTo("False"));
    }
}
