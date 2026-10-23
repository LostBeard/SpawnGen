using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Types;

/// <summary>
/// ComfyUI's DICT: a JSON-like object (Python <c>dict</c>) moving between nodes. Keys keep the order they were added in; values are null, bool,
/// whole numbers (long or BigInteger), double, string, <see cref="ArrayValue"/> or another <see cref="DictValue"/>.
/// </summary>
[Socket("DICT")]
public sealed class DictValue : OrderedDictionary<string, object?>
{
    public DictValue() : base(StringComparer.Ordinal) { }
}

/// <summary>ComfyUI's ARRAY: a JSON-like list (Python <c>list</c>) moving between nodes, with the same kinds of items as <see cref="DictValue"/>.</summary>
[Socket("ARRAY")]
public sealed class ArrayValue : List<object?>
{
    public ArrayValue() { }

    public ArrayValue(IEnumerable<object?> items) : base(items) { }
}
