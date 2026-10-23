using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.Core;

/// <summary>ComfyUI's Preview as Text: shows any value as text in the node (and passes the text on), formatted as Python's str() would.</summary>
[Node("PreviewAny", Category = "utilities", DisplayName = "Preview as Text", IsOutput = true, Description = "Preview any input value as text.", OutputNames = ["STRING"])]
public sealed class PreviewAny
{
    public (string Text, UiOutput Ui) Execute(object? source)
    {
        var text = Format(source);
        return (text, new UiOutput("text", new[] { text }));
    }

    /// <summary>Text the way ComfyUI's node builds it: strings as they are; numbers and booleans as Python prints them (1.0, 1e-05, True); anything else by its description.</summary>
    public static string Format(object? value) => value switch
    {
        null => "None",
        string s => s,
        bool b => b ? "True" : "False",
        double d => PythonFloat(d),
        float f => PythonFloat(f),
        IFormattable n => n.ToString(null, System.Globalization.CultureInfo.InvariantCulture),
        Engine.Types.HostTensor t => $"{t.GetType().Name} [{string.Join(", ", t.Shape)}]",
        _ => value.ToString() ?? "None",
    };

    /// <summary>Python's repr of a float: the shortest text that reads back the same, with ".0" on whole numbers and Python's exponent form.</summary>
    static string PythonFloat(double d)
    {
        if (double.IsNaN(d)) return "nan";
        if (double.IsInfinity(d)) return d > 0 ? "inf" : "-inf";
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        double abs = Math.Abs(d);
        if (abs != 0 && (abs < 1e-4 || abs >= 1e16))
        {
            // Python switches to exponent form here; the mantissa is the shortest that still reads back as d.
            for (int digits = 1; ; digits++)
            {
                var candidate = d.ToString("E" + (digits - 1), inv);
                if (digits < 17 && double.Parse(candidate, inv) != d) continue;
                int e = candidate.IndexOf('E');
                var mantissa = candidate[..e].Contains('.') ? candidate[..e].TrimEnd('0').TrimEnd('.') : candidate[..e];
                int exponent = int.Parse(candidate[(e + 1)..], inv);
                return $"{mantissa}e{(exponent < 0 ? "-" : "+")}{Math.Abs(exponent):00}";
            }
        }
        var r = d.ToString("R", inv);
        return r.Contains('.') ? r : r + ".0";
    }
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
