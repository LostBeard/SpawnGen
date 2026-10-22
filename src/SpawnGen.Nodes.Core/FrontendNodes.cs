using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.Core;

/// <summary>A bend in a link: one input, one output of the same value. The editor draws it as a small dot pair and compiles it away.</summary>
[Node(Category = "utils", DisplayName = "Reroute", FrontendOnly = true)]
public sealed class Reroute
{
    // Also works when a prompt that still contains one is submitted directly.
    public object? Execute(object? value) => value;
}

/// <summary>A sticky note on the canvas. It is never run.</summary>
[Node(Category = "utils", DisplayName = "Note", FrontendOnly = true)]
public sealed class Note
{
    public void Execute([Input(Multiline = true)] string text = "") { }
}

/// <summary>A note written in Markdown (shown as plain text here). It is never run.</summary>
[Node(Category = "utils", DisplayName = "Markdown Note", FrontendOnly = true)]
public sealed class MarkdownNote
{
    public void Execute([Input(Multiline = true)] string text = "") { }
}
