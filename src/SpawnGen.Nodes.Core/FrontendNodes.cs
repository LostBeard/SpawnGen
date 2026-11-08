using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.Core;

// Nodes that live only in the editor, as in ComfyUI's frontend (its server has no such node types): the compiler looks through a reroute and
// leaves notes out, so a run never sends them. Their interfaces are SpawnGen's.

[Node("Reroute", Category = "utils", DisplayName = "Reroute", FrontendOnly = true)]
public sealed class Reroute
{
    /// <summary>Passes the value through; the compiler links what reads a reroute to the reroute's source instead.</summary>
    public object? Execute(object? value) => value;
}

[Node("Note", Category = "utils", DisplayName = "Note", FrontendOnly = true)]
public sealed class Note
{
    public void Execute([Input(Multiline = true)] string text = "") { }
}

[Node("MarkdownNote", Category = "utils", DisplayName = "Markdown Note", FrontendOnly = true)]
public sealed class MarkdownNote
{
    public void Execute([Input(Multiline = true)] string text = "") { }
}
