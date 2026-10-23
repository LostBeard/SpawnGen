using System.Numerics;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.Core;

// ComfyUI's small utility nodes: comfy_extras/nodes_seed.py, nodes_number_convert.py and the backend half of nodes_logic.py.

[Node("SeedNode", Category = "utilities", DisplayName = "Seed", OutputNames = ["seed"])]
public sealed class SeedNode
{
    /// <summary>One seed that can feed several samplers. (ComfyUI starts this node's control at "fixed"; the editor gives seed widgets "randomize".)</summary>
    public long Execute([Input(Min = 0, Max = long.MaxValue)] long seed = 0) => seed;
}

[Node("ComfyNumberConvert", Category = "utilities", DisplayName = "Convert Number")]
public sealed class ComfyNumberConvert
{
    /// <summary>
    /// An INT, FLOAT, STRING or BOOLEAN as both a float and an int. A float's int is truncated toward zero; a text is read as Python's <c>float()</c> /
    /// <c>int()</c> read it (spaces around it, underscores between digits, "1e3" is 1000). Empty, unreadable or non-finite values are errors.
    /// </summary>
    public (double, long) Execute(object value)
    {
        double f;
        BigInteger i;
        switch (value)
        {
            case bool b:
                (f, i) = (b ? 1.0 : 0.0, b ? 1 : 0);
                break;
            case long or int or short or byte or BigInteger:
                i = value is BigInteger bi ? bi : new BigInteger(Convert.ToInt64(value));
                f = PyNumber.ToDouble(i);
                break;
            case double or float:
                f = Convert.ToDouble(value);
                if (double.IsNaN(f)) throw new ArgumentException("cannot convert float NaN to integer");
                if (double.IsInfinity(f)) throw new OverflowException("cannot convert float infinity to integer");
                i = PyNumber.ToBigInteger(f);
                break;
            case string s:
            {
                var text = PyText.Strip(s);
                if (text.Length == 0) throw new ArgumentException("Cannot convert empty string to number.");
                f = PyNumber.TryParseFloat(text) ?? throw new ArgumentException($"Cannot convert string to number: {PyText.ReprString(s)}");
                if (!double.IsFinite(f)) throw new ArgumentException($"Cannot convert non-finite value to number: {PyNumber.FloatRepr(f)}");
                i = PyNumber.TryParseInt(text) ?? PyNumber.ToBigInteger(f);
                break;
            }
            default:
                throw new ArgumentException($"Unsupported input type: {value.GetType().Name}");
        }
        if (!double.IsFinite(f)) throw new ArgumentException($"Cannot convert non-finite value to number: {PyNumber.FloatRepr(f)}");
        if (i < long.MinValue || i > long.MaxValue) throw new OverflowException($"{i} does not fit in an INT (64 bits).");
        return (f, (long)i);
    }
}

[Node("ComfyNotNode", Category = "utilities/logic", DisplayName = "Not",
    Description = "Logical NOT operation. Returns true if the value is falsy. Uses Python's rules for truthiness.")]
public sealed class ComfyNotNode
{
    public bool Execute(object value) => !Truthiness.Of(value);
}

/// <summary>Python's truth value of the values that move between nodes.</summary>
public static class Truthiness
{
    /// <summary>
    /// False for None, False, 0, 0.0, "" and empty lists; a one-element IMAGE or MASK tensor is true when its value is not 0, a larger one is an error
    /// ("Boolean value of Tensor with more than one value is ambiguous", as in torch). Everything else (latents and audio are dicts in ComfyUI, models are objects) is true.
    /// </summary>
    public static bool Of(object? value) => value switch
    {
        Engine.Types.ImageBatch or Engine.Types.MaskBatch => ((Engine.Types.HostTensor)value).Data.Length switch
        {
            1 => ((Engine.Types.HostTensor)value).Data[0] != 0,
            0 => throw new InvalidOperationException("Boolean value of Tensor with no values is ambiguous"),
            _ => throw new InvalidOperationException("Boolean value of Tensor with more than one value is ambiguous"),
        },
        Engine.Types.LatentBatch => true,
        _ => PyExpression.Truthy(PyExpression.Normalize(value)),
    };
}
