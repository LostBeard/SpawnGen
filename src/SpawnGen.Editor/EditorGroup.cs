using System.Text.Json.Nodes;

namespace SpawnGen.Editor;

/// <summary>A labelled, coloured box on the canvas (ComfyUI's group). Dragging its title moves it together with the nodes inside it.</summary>
public sealed class EditorGroup
{
    public const double TitleHeight = 34;
    public const double MinWidth = 140, MinHeight = TitleHeight + 40;
    public const double ResizeSize = 18;
    public const string DefaultColor = "#3f789e";

    /// <summary>ComfyUI's node/group colour presets, name to colour.</summary>
    public static readonly IReadOnlyList<(string Name, string Color)> Palette =
    [
        ("red", "#A88"), ("brown", "#b06634"), ("green", "#8A8"), ("blue", "#88A"), ("pale blue", "#3f789e"),
        ("cyan", "#8AA"), ("purple", "#a1309b"), ("yellow", "#b58b2a"), ("black", "#444"),
    ];

    /// <summary>ComfyUI's node colour presets: name, title bar colour, body colour (what it writes as <c>color</c> and <c>bgcolor</c>).</summary>
    public static readonly IReadOnlyList<(string Name, string Color, string BgColor)> NodePalette =
    [
        ("red", "#322", "#533"), ("brown", "#332922", "#593930"), ("green", "#232", "#353"), ("blue", "#223", "#335"), ("pale blue", "#2a363b", "#3f5159"),
        ("cyan", "#233", "#355"), ("purple", "#323", "#535"), ("yellow", "#432", "#653"), ("black", "#222", "#000"),
    ];

    public required int Id { get; init; }
    public string Title { get; set; } = "Group";
    public double X { get; set; }
    public double Y { get; set; }
    public double Width { get; set; }
    public double Height { get; set; }
    public string Color { get; set; } = DefaultColor;
    public double FontSize { get; set; } = 24;
    /// <summary>Workflow fields the editor does not use (ComfyUI's <c>flags</c> and anything newer): kept and written back.</summary>
    public JsonObject? Extra { get; set; }

    /// <summary>A pinned group cannot be moved, resized or deleted from the canvas (ComfyUI keeps it in the group's <c>flags.pinned</c>, which this reads and writes).</summary>
    public bool Pinned
    {
        get => (bool?)Extra?["flags"]?["pinned"] == true;
        set
        {
            if (value == Pinned) return;
            Extra ??= new JsonObject();
            if (Extra["flags"] is not JsonObject flags) Extra["flags"] = flags = new JsonObject();
            if (value) flags["pinned"] = true;
            else
            {
                flags.Remove("pinned");
                if (flags.Count == 0) Extra.Remove("flags");
                if (Extra.Count == 0) Extra = null;
            }
        }
    }

    public Rect Bounds => new(X, Y, Width, Height);
    public Rect TitleBar => new(X, Y, Width, TitleHeight);
    public Rect ResizeHandle => new(X + Width - ResizeSize, Y + Height - ResizeSize, ResizeSize, ResizeSize);

    public EditorGroup Clone() => new() { Id = Id, Title = Title, X = X, Y = Y, Width = Width, Height = Height, Color = Color, FontSize = FontSize, Extra = (JsonObject?)Extra?.DeepClone() };
}
