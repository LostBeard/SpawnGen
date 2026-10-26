using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Tests;

/// <summary>Shows the preview method the node was given (a sampler reads it to choose its previewer).</summary>
[Node(Category = "test", IsOutput = true)]
public sealed class ShowPreviewMethod
{
    public UiOutput Execute(NodeContext context) => new("method", new[] { context.PreviewMethod ?? "(none)" });
}

/// <summary>ComfyUI's extra_data.preview_method reaches every node of the run through the execution options.</summary>
public class PreviewMethodTests
{
    [TestCase("taesd")]
    [TestCase("none")]
    [TestCase(null)]
    public async Task The_prompts_preview_method_reaches_the_node(string? method)
    {
        var executor = new PromptExecutor(new NodeRegistry().AddAssembly(typeof(PreviewMethodTests).Assembly));
        var g = new PromptGraph();
        g.Add("1", "ShowPreviewMethod");
        var r = await executor.ExecuteAsync(g, new ExecutionOptions { PreviewMethod = method });
        Assert.That(r.Outputs["1"].Ui!["method"], Is.EqualTo(new[] { method ?? "(none)" }));
    }
}
