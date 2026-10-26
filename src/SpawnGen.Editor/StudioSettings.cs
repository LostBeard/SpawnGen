using System.Text.Json;

namespace SpawnGen.Editor;

/// <summary>How links are drawn (ComfyUI's "link render mode").</summary>
public enum LinkStyle { Spline, Straight, Linear, Hidden }

/// <summary>The editor's user settings: saved as JSON in the browser's storage and applied to the view.</summary>
public sealed class StudioSettings
{
    public LinkStyle LinkStyle { get; set; } = LinkStyle.Spline;
    public bool SnapToGrid { get; set; }
    public int GridSize { get; set; } = 10;
    public bool ShowMinimap { get; set; } = true;
    /// <summary>Draws each node's id (#12) in its title bar, the number the saved workflow and error messages use (ComfyUI's node id badge).</summary>
    public bool ShowNodeIds { get; set; }
    /// <summary>The colour palette id ("dark" or "light"); the Studio keeps the list and falls back to dark for an id it does not know.</summary>
    public string Palette { get; set; } = "dark";
    /// <summary>How samplers preview while they work, sent with each prompt (ComfyUI's preview method): auto / latent2rgb (a quick colour guess), taesd (the
    /// tiny VAE from vae_approx, close to the final picture), none.</summary>
    public string PreviewMethod { get; set; } = "auto";

    /// <summary>The preview methods, as ComfyUI names them.</summary>
    public static readonly string[] PreviewMethods = ["auto", "latent2rgb", "taesd", "none"];

    public string ToJson() => JsonSerializer.Serialize(this);

    /// <summary>Reads settings back; anything missing, unknown or out of range falls back to the default, so an old or damaged value never breaks the page.</summary>
    public static StudioSettings FromJson(string? json)
    {
        var s = new StudioSettings();
        if (string.IsNullOrWhiteSpace(json)) return s;
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object) return s;
            if (root.TryGetProperty(nameof(LinkStyle), out var ls) && ls.ValueKind == JsonValueKind.Number && ls.TryGetInt32(out var li) && Enum.IsDefined(typeof(LinkStyle), li)) s.LinkStyle = (LinkStyle)li;
            if (root.TryGetProperty(nameof(SnapToGrid), out var sg) && sg.ValueKind is JsonValueKind.True or JsonValueKind.False) s.SnapToGrid = sg.GetBoolean();
            if (root.TryGetProperty(nameof(GridSize), out var gs) && gs.ValueKind == JsonValueKind.Number && gs.TryGetInt32(out var g) && g is >= 2 and <= 200) s.GridSize = g;
            if (root.TryGetProperty(nameof(ShowNodeIds), out var sn) && sn.ValueKind is JsonValueKind.True or JsonValueKind.False) s.ShowNodeIds = sn.GetBoolean();
            if (root.TryGetProperty(nameof(Palette), out var pa) && pa.ValueKind == JsonValueKind.String && pa.GetString() is { Length: > 0 and <= 40 } pid) s.Palette = pid;
            if (root.TryGetProperty(nameof(ShowMinimap), out var sm) && sm.ValueKind is JsonValueKind.True or JsonValueKind.False) s.ShowMinimap = sm.GetBoolean();
            if (root.TryGetProperty(nameof(PreviewMethod), out var pm) && pm.ValueKind == JsonValueKind.String && PreviewMethods.Contains(pm.GetString())) s.PreviewMethod = pm.GetString()!;
        }
        catch (JsonException) { }
        return s;
    }
}
