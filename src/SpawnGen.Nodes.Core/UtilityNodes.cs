using System.Collections;
using System.Numerics;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

[Node("SeedNode", Category = "utilities", DisplayName = "Seed", OutputNames = ["seed"])]
public sealed class SeedNode
{
    /// <summary>The seed as it is (0 .. sys.maxsize, as ComfyUI's interface states).</summary>
    public long Execute([Input(Min = 0, Max = 9223372036854775807d)] long seed = 0) => seed;
}

[Node("ComfyNotNode", Category = "utilities/logic", DisplayName = "Not",
    Description = "Logical NOT operation. Returns true if the value is falsy. Uses Python's rules for truthiness.")]
public sealed class ComfyNotNode
{
    public bool Execute(object? value) => !Truthiness.Of(value);
}

/// <summary>
/// Python's truth value (the Python documentation, "Truth Value Testing") for the values SpawnGen's sockets carry. None, False, zero and empty
/// text, lists and dicts are false; everything else is true. ComfyUI's LATENT and AUDIO are dicts, so they are true. A tensor follows PyTorch
/// (<c>Tensor.__bool__</c>): one value is its own truth, more than one is an error. Measured with ComfyUI: a 1x1 RGB image (three values) fails
/// with "Boolean value of Tensor with more than one value is ambiguous"; a 1x1 mask of 0 is false.
/// </summary>
static class Truthiness
{
    public static bool Of(object? value) => value switch
    {
        null => false,
        bool b => b,
        long l => l != 0,
        int i => i != 0,
        BigInteger bi => !bi.IsZero,
        double d => d != 0, // NaN is true, as in Python
        float f => f != 0,
        string s => s.Length > 0,
        LatentBatch or AudioClip => true,
        HostTensor t => t.Data.Length switch
        {
            1 => t.Data[0] != 0,
            0 => throw new InvalidOperationException("Boolean value of Tensor with no values is ambiguous"),
            _ => throw new InvalidOperationException("Boolean value of Tensor with more than one value is ambiguous"),
        },
        ICollection c => c.Count > 0,
        _ => true,
    };
}

/// <summary>
/// Convert Number: a number, boolean or text as a FLOAT and an INT. Measured with ComfyUI:
/// <list type="bullet">
/// <item>A boolean is 1 or 0; an int is itself and <c>float(int)</c> (9007199254740993 gives 9007199254740992.0 and 9007199254740993).</item>
/// <item>A float is itself and truncated toward zero (-2.9 gives -2); infinity fails with "cannot convert float infinity to integer", NaN with
/// "cannot convert float NaN to integer".</item>
/// <item>Text is read with Python's whitespace stripped. Empty text fails with "Cannot convert empty string to number."; text Python reads as an
/// int (underscores and Unicode digits allowed) is that int; otherwise text Python reads as a float is that float, truncated for the INT
/// ("1e3" gives 1000.0 and 1000). Text that is neither fails with "Cannot convert string to number: ' abc '" (the text as sent, in Python's
/// repr); "inf", "Infinity" and "nan" fail with "Cannot convert non-finite value to number: inf" (the float's repr).</item>
/// <item>Anything else fails with "Unsupported input type: Tensor" (an IMAGE) or "Unsupported input type: NoneType".</item>
/// </list>
/// ComfyUI's INT is unbounded; SpawnGen's is 64 bits, so a result past that range is an error here.
/// </summary>
[Node("ComfyNumberConvert", Category = "utilities", DisplayName = "Convert Number")]
public sealed class ComfyNumberConvert
{
    public (double, long) Execute([Input(Type = "INT,FLOAT,STRING,BOOLEAN")] object? value)
    {
        switch (value)
        {
            case bool b:
                return (b ? 1 : 0, b ? 1 : 0);
            case long or int or short or byte:
            {
                long l = Convert.ToInt64(value, System.Globalization.CultureInfo.InvariantCulture);
                return (l, l);
            }
            case BigInteger bi:
                return (PyNumber.ToDouble(bi), FitInt(bi));
            case double or float:
            {
                double d = Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture);
                return (d, Truncate(d));
            }
            case string s:
                return FromText(s);
            default:
                throw new ArgumentException($"Unsupported input type: {TypeName(value)}");
        }
    }

    static (double, long) FromText(string text)
    {
        var stripped = PyText.Strip(text);
        if (stripped.Length == 0) throw new ArgumentException("Cannot convert empty string to number.");
        if (PyNumber.TryParseInt(stripped) is { } whole) return (PyNumber.ToDouble(whole), FitInt(whole));
        if (PyNumber.TryParseFloat(stripped) is not { } f) throw new ArgumentException($"Cannot convert string to number: {PyText.ReprString(text)}");
        if (!double.IsFinite(f)) throw new ArgumentException($"Cannot convert non-finite value to number: {PyNumber.FloatRepr(f)}");
        return (f, Truncate(f));
    }

    /// <summary>Python's <c>int(float)</c>: truncated toward zero.</summary>
    static long Truncate(double d)
    {
        if (double.IsNaN(d)) throw new ArgumentException("cannot convert float NaN to integer");
        if (double.IsInfinity(d)) throw new OverflowException("cannot convert float infinity to integer");
        double t = Math.Truncate(d);
        // 2^63 is the first double past long's range.
        if (t >= 9223372036854775808.0 || t < -9223372036854775808.0) throw new OverflowException($"{PyNumber.FloatRepr(d)} does not fit in an INT (64 bits).");
        return (long)t;
    }

    static long FitInt(BigInteger i) =>
        i >= long.MinValue && i <= long.MaxValue ? (long)i : throw new OverflowException($"{i} does not fit in an INT (64 bits).");

    /// <summary>The Python type name ComfyUI would report: its LATENT and AUDIO are dicts, its other tensors torch Tensors.</summary>
    static string TypeName(object? value) => value switch
    {
        LatentBatch or AudioClip => "dict",
        HostTensor => "Tensor",
        _ => PyValues.TypeName(value),
    };
}
