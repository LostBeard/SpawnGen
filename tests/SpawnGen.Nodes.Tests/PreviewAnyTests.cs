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
    }
}
