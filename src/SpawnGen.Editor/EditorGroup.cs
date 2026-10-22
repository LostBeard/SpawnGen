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

    public Rect Bounds => new(X, Y, Width, Height);
    public Rect TitleBar => new(X, Y, Width, TitleHeight);
    public Rect ResizeHandle => new(X + Width - ResizeSize, Y + Height - ResizeSize, ResizeSize, ResizeSize);

    public EditorGroup Clone() => new() { Id = Id, Title = Title, X = X, Y = Y, Width = Width, Height = Height, Color = Color, FontSize = FontSize, Extra = (JsonObject?)Extra?.DeepClone() };
}
