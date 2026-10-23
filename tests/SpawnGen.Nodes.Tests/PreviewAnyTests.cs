using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

public class PreviewAnyTests
{
    [Test]
    public async Task Preview_Any_sends_its_text_to_the_editor_and_takes_a_linked_string()
    {
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly);
        var g = new PromptGraph();
        g.Add("1", "PrimitiveString", ("value", "hello\nworld"));
        g.Add("2", "PreviewAny", ("source", PromptInput.Link("1")));
        var result = await new PromptExecutor(registry).ExecuteAsync(g);
        Assert.That(((string[])result.Outputs["2"].Ui!["text"]!)[0], Is.EqualTo("hello\nworld"));
        Assert.That(result.Outputs["2"].Values[0], Is.EqualTo("hello\nworld"), "the text is also an output");
    }

    [TestCase(120L, "120")]
    [TestCase(1.0, "1.0")]
    [TestCase(0.1, "0.1")]
    [TestCase(1.5e-5, "1.5e-05")]
    [TestCase(1e16, "1e+16")]
    [TestCase(true, "True")]
    [TestCase(null, "None")]
    public void Values_read_as_Python_prints_them(object? value, string expected) => Assert.That(PreviewAny.Format(value), Is.EqualTo(expected));
}
