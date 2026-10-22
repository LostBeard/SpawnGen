namespace SpawnGen.Studio.Canvas;

/// <summary>Colours and fonts of the node canvas, dark like ComfyUI. Socket colours follow ComfyUI's so workflows look familiar.</summary>
public static class Theme
{
    public const string Background = "#1c1c1c";
    public const string GridMinor = "#232323";
    public const string GridMajor = "#2b2b2b";
    public const string NodeBody = "#353535";
    public const string NodeBorder = "#161616";
    public const string NodeSelected = "#6aa9ff";
    public const string TitleText = "#e6e6e6";
    public const string LabelText = "#a8a8a8";
    public const string ValueText = "#e0e0e0";
    public const string WidgetFill = "#242424";
    public const string WidgetBorder = "#4a4a4a";
    public const string Accent = "#4f8fe8";
    public const string Error = "#e05555";

    public const string Font = "12px system-ui, 'Segoe UI', sans-serif";
    public const string TitleFont = "600 13px system-ui, 'Segoe UI', sans-serif";

    /// <summary>The title-bar colour for a node category (ComfyUI tints by category the same way).</summary>
    public static string TitleColor(string category) => category.Split('/')[0] switch
    {
        "loaders" => "#3d3550",
        "conditioning" => "#4d3d28",
        "sampling" => "#2e4a3a",
        "latent" => "#4a3550",
        "image" => "#2d4560",
        _ => "#2f2f2f",
    };

    public static string SocketColor(string type) => type switch
    {
        "MODEL" => "#B39DDB",
        "CLIP" => "#FFD500",
        "VAE" => "#FF6E6E",
        "CONDITIONING" => "#FFA931",
        "LATENT" => "#FF9CF9",
        "IMAGE" => "#64B5F6",
        "MASK" => "#81C784",
        "INT" => "#7aa2f7",
        "FLOAT" => "#9ece6a",
        "STRING" => "#e0af68",
        "BOOLEAN" => "#bb9af7",
        _ => "#9aa0a6",
    };
}
