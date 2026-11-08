using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// If/Else Switch: passes on <c>on_true</c> or <c>on_false</c>. Both are lazy, so only the chosen branch runs. Both sockets and the output share one
/// type (any type until a link decides it). Measured with ComfyUI: a chosen branch with nothing connected gives None.
/// </summary>
[Node("ComfySwitchNode", Category = "utilities/logic", DisplayName = "If/Else Switch", OutputNames = ["output"], OutputMatchTypes = ["switch"])]
public sealed class SwitchNode
{
    /// <summary>Asks for the chosen branch until it has been evaluated.</summary>
    public static IEnumerable<string> CheckLazyStatus(bool @switch, object? onFalse, object? onTrue) =>
        @switch ? onTrue is null ? ["on_true"] : [] : onFalse is null ? ["on_false"] : [];

    public object? Execute(bool @switch,
        [Input(Lazy = true, MatchType = "switch", Optional = true)] object? onFalse = null,
        [Input(Lazy = true, MatchType = "switch", Optional = true)] object? onTrue = null)
        => @switch ? onTrue : onFalse;
}

/// <summary>
/// Get Item From List: takes the whole list that reaches it and returns one item, indexed as Python indexes a list. Measured with ComfyUI on a
/// list of three: index -1 gives the last item; 3 and -4 fail with "IndexError: list index out of range". A single value is a list of one.
/// </summary>
[Node("GetItemFromList", Category = "utilities", DisplayName = "Get Item From List", InputIsList = true)]
public sealed class GetItemFromList
{
    public object? Execute(IReadOnlyList<object?> list, [Input(Default = 0L)] IReadOnlyList<long> index)
    {
        long i = index.Count > 0 ? index[0] : 0;
        if (i < 0) i += list.Count;
        if (i < 0 || i >= list.Count) throw new PyException("IndexError", "list index out of range");
        return list[(int)i];
    }
}
