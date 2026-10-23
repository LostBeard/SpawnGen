namespace SpawnGen.Studio.Canvas;

/// <summary>One set of canvas colours (ComfyUI's colour palettes). The page's own colours follow the same choice through CSS variables (data-theme on the html element).</summary>
public sealed record Palette(
    string Id, string Name,
    string Background, string GridMinor, string GridMajor,
    string NodeBody, string NodeBorder, string NodeSelected,
    string TitleText, string LabelText, string ValueText,
    string WidgetFill, string WidgetBorder, string WidgetArrow,
    string TextBoxFill, string TextBoxText, string ResizeGrip, string PinColor,
    string MinimapFill, string MinimapBorder, string MinimapNode, string MinimapView,
    string NoteBody, string NoteTitle, string BypassBody, string MissingTitle,
    string LinkTargetRing,
    IReadOnlyDictionary<string, string> CategoryTitles, string DefaultTitle);

/// <summary>Colours and fonts of the node canvas. Socket colours follow ComfyUI's so workflows look familiar, in every palette.</summary>
public static class Theme
{
    public static readonly Palette Dark = new(
        "dark", "Dark",
        Background: "#1c1c1c", GridMinor: "#232323", GridMajor: "#2b2b2b",
        NodeBody: "#353535", NodeBorder: "#161616", NodeSelected: "#6aa9ff",
        TitleText: "#e6e6e6", LabelText: "#a8a8a8", ValueText: "#e0e0e0",
        WidgetFill: "#242424", WidgetBorder: "#4a4a4a", WidgetArrow: "#6f6f6f",
        TextBoxFill: "#141414", TextBoxText: "#d8d8d8", ResizeGrip: "#5a5a5a", PinColor: "#d9d9d9",
        MinimapFill: "rgba(20,20,20,0.96)", MinimapBorder: "#555", MinimapNode: "#7a7a7a", MinimapView: "#f0f0f0",
        NoteBody: "#4d4630", NoteTitle: "#7a6a2a", BypassBody: "#4a3a5a", MissingTitle: "#5a2a2a",
        LinkTargetRing: "#ffffff",
        CategoryTitles: new Dictionary<string, string>
        {
            ["loaders"] = "#3d3550", ["conditioning"] = "#4d3d28", ["sampling"] = "#2e4a3a", ["latent"] = "#4a3550", ["image"] = "#2d4560",
        },
        DefaultTitle: "#2f2f2f");

    /// <summary>A light palette in the spirit of ComfyUI's "Light" one: light grey canvas, near-white nodes, dark text.</summary>
    public static readonly Palette Light = new(
        "light", "Light",
        Background: "#d6d6d6", GridMinor: "#cfcfcf", GridMajor: "#c4c4c4",
        NodeBody: "#f2f2f2", NodeBorder: "#9a9a9a", NodeSelected: "#2f6fd6",
        TitleText: "#1e1e1e", LabelText: "#555555", ValueText: "#1e1e1e",
        WidgetFill: "#e2e2e2", WidgetBorder: "#b4b4b4", WidgetArrow: "#7a7a7a",
        TextBoxFill: "#ffffff", TextBoxText: "#222222", ResizeGrip: "#9a9a9a", PinColor: "#444444",
        MinimapFill: "rgba(245,245,245,0.96)", MinimapBorder: "#9a9a9a", MinimapNode: "#9a9a9a", MinimapView: "#222222",
        NoteBody: "#f3e9b5", NoteTitle: "#e0cc70", BypassBody: "#e3d3f0", MissingTitle: "#e8a6a6",
        LinkTargetRing: "#1e1e1e",
        CategoryTitles: new Dictionary<string, string>
        {
            ["loaders"] = "#c9bfe0", ["conditioning"] = "#ead2b0", ["sampling"] = "#b8dcc6", ["latent"] = "#dcc0e4", ["image"] = "#b6cfe8",
        },
        DefaultTitle: "#c8c8c8");

    public static IReadOnlyList<Palette> All { get; } = [Dark, Light];

    /// <summary>The palette in use. The renderer reads it every frame, so a change shows on the next repaint.</summary>
    public static Palette Current { get; private set; } = Dark;

    /// <summary>Selects a palette by id; an unknown id selects the dark one.</summary>
    public static Palette Use(string? id) => Current = All.FirstOrDefault(p => p.Id == id) ?? Dark;

    public static string Background => Current.Background;
    public static string GridMinor => Current.GridMinor;
    public static string GridMajor => Current.GridMajor;
    public static string NodeBody => Current.NodeBody;
    public static string NodeBorder => Current.NodeBorder;
    public static string NodeSelected => Current.NodeSelected;
    public static string TitleText => Current.TitleText;
    public static string LabelText => Current.LabelText;
    public static string ValueText => Current.ValueText;
    public static string WidgetFill => Current.WidgetFill;
    public static string WidgetBorder => Current.WidgetBorder;
    public const string Accent = "#4f8fe8";
    public const string Error = "#e05555";
    public const string Running = "#4caf50";

    public const string Font = "12px system-ui, 'Segoe UI', sans-serif";
    public const string TitleFont = "600 13px system-ui, 'Segoe UI', sans-serif";

    /// <summary>The title-bar colour for a node category, by its first part; ComfyUI's newer "model/..." categories are tinted by the part after "model".</summary>
    public static string TitleColor(string category)
    {
        var parts = category.Split('/');
        var key = parts[0] == "model" && parts.Length > 1 ? parts[1] : parts[0];
        return Current.CategoryTitles.TryGetValue(key, out var c) ? c : Current.DefaultTitle;
    }

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
