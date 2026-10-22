namespace SpawnGen.Engine.Nodes;

/// <summary>
/// The type of a node socket. Names follow ComfyUI (<c>IMAGE</c>, <c>LATENT</c>, <c>MODEL</c>, ...) so workflows
/// round-trip; custom types are just new names. A comma-separated name (<c>"INT,FLOAT"</c>) is a union an
/// input accepts.
/// </summary>
public readonly record struct SocketType(string Name)
{
    public static readonly SocketType Any = new("*");
    public static readonly SocketType Int = new("INT");
    public static readonly SocketType Float = new("FLOAT");
    public static readonly SocketType String = new("STRING");
    public static readonly SocketType Boolean = new("BOOLEAN");
    public static readonly SocketType Image = new("IMAGE");
    public static readonly SocketType Mask = new("MASK");
    public static readonly SocketType Latent = new("LATENT");
    public static readonly SocketType Model = new("MODEL");
    public static readonly SocketType Clip = new("CLIP");
    public static readonly SocketType Vae = new("VAE");
    public static readonly SocketType Conditioning = new("CONDITIONING");

    public bool IsAny => Name == "*";

    /// <summary>True when an input of this type can be fed by an output of type <paramref name="output"/>.</summary>
    public bool Accepts(SocketType output)
    {
        if (IsAny || output.IsAny || Name == output.Name) return true;
        // Whole numbers widen to FLOAT; the binder coerces.
        if (output == Int && this == Float) return true;
        if (!Name.Contains(',')) return false;
        foreach (var part in Name.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
            if (part == output.Name || (part == "FLOAT" && output == Int)) return true;
        return false;
    }

    public override string ToString() => Name;
}
