using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.Core;

/// <summary>Shows a text in the editor: link any STRING output (a detection list, keypoints, a prompt) to read it in the node.</summary>
[Node("PreviewAny", Category = "utils", DisplayName = "Preview Any", IsOutput = true)]
public sealed class PreviewAny
{
    public UiOutput Execute([Input(Multiline = true)] string source = "") => new("text", new[] { source });
}

// Backend value nodes (ComfyUI's comfy_extras primitives): a widget that produces a value other nodes can read, so one number or text can feed many inputs.

[Node(Category = "utils/primitive", DisplayName = "String")]
public sealed class PrimitiveString
{
    public string Execute(string value = "") => value;
}

[Node(Category = "utils/primitive", DisplayName = "String (Multiline)")]
public sealed class PrimitiveStringMultiline
{
    public string Execute([Input(Multiline = true)] string value = "") => value;
}

[Node(Category = "utils/primitive", DisplayName = "Int")]
public sealed class PrimitiveInt
{
    public long Execute([Input(Min = -9007199254740991, Max = 9007199254740991)] long value = 0) => value;
}

[Node(Category = "utils/primitive", DisplayName = "Float")]
public sealed class PrimitiveFloat
{
    public double Execute([Input(Min = -1e15, Max = 1e15, Step = 0.01)] double value = 0) => value;
}

[Node(Category = "utils/primitive", DisplayName = "Boolean")]
public sealed class PrimitiveBoolean
{
    public bool Execute(bool value = false) => value;
}
