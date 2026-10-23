using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.Core;

/// <summary>ComfyUI's If/Else Switch: passes on_true or on_false on, and only the chosen branch runs (both inputs are lazy).</summary>
[Node("ComfySwitchNode", Category = "utilities/logic", DisplayName = "If/Else Switch", OutputNames = ["output"])]
public sealed class SwitchNode
{
    /// <summary>Asks for the chosen branch once it is known to be missing (connected but not evaluated yet).</summary>
    public static IEnumerable<string> CheckLazyStatus(bool @switch, object? onFalse, object? onTrue, NodeContext context)
    {
        if (@switch && onTrue is null && context.IsConnected("on_true")) return ["on_true"];
        if (!@switch && onFalse is null && context.IsConnected("on_false")) return ["on_false"];
        return [];
    }

    public object? Execute(NodeContext context, bool @switch = false,
        [Input(Lazy = true, Optional = true)] object? onFalse = null, [Input(Lazy = true, Optional = true)] object? onTrue = null) => @switch ? onTrue : onFalse;
}

/// <summary>ComfyUI's Get Item From List: one item of a list (a list output upstream, such as Create List's), counted from 0; negative counts from the end like Python.</summary>
[Node("GetItemFromList", Category = "utilities", DisplayName = "Get Item From List", InputIsList = true)]
public sealed class GetItemFromList
{
    public object? Execute(IReadOnlyList<object?> list, [Input(Default = 0L)] IReadOnlyList<long> index)
    {
        long i = index[0];
        long at = i < 0 ? list.Count + i : i;
        if (at < 0 || at >= list.Count) throw new ArgumentOutOfRangeException(nameof(index), $"Index {i} is outside the list of {list.Count} items.");
        return list[(int)at];
    }
}
