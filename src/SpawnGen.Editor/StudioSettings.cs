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
    /// <summary>
    /// LTX-2 LoRAs on 8-bit weights: true re-rounds the patched weights as ComfyUI does, so the result is ComfyUI's; false (default) keeps the exact
    /// float form (the LoRA's full effect). Sent with each prompt as a run setting (<see cref="SpawnGen.Engine.Execution.RunSettings.Ltx2ComfyFp8Lora"/>).
    /// </summary>
    public bool Ltx2ComfyFp8Lora { get; set; }
    /// <summary>LTX-2's arithmetic on CUDA: "fp32", "exact_tc" (default) or "fast" (<see cref="SpawnGen.Engine.Execution.RunSettings.Ltx2Precision"/>).</summary>
    public string Ltx2Precision { get; set; } = "exact_tc";
    /// <summary>LTX-2: host RAM (GB) for streamed blocks kept between steps; 0 = none (<see cref="SpawnGen.Engine.Execution.RunSettings.Ltx2HostCacheGb"/>).</summary>
    public double Ltx2HostCacheGb { get; set; }
    /// <summary>LTX-2: host RAM (GB) the block reader may run ahead; 0 = one block (<see cref="SpawnGen.Engine.Execution.RunSettings.Ltx2ReadAheadGb"/>).</summary>
    public double Ltx2ReadAheadGb { get; set; }

    /// <summary>
    /// The in-browser engine: an optional limit, in GB, on the downloaded models the browser keeps. 0 (the default) = no limit: every model
    /// stays until the user removes it in the Models panel, as in other model apps (TJ, 2026-11-24: downloading gigabytes again is worse for
    /// most people than using disk space they have). With a limit, the models used longest ago go first when a new one needs room, and a
    /// model larger than the limit streams from its host on every load. What protects the kept models from the browser itself is persistent
    /// storage (the engine asks for it): a browser under disk pressure clears whole sites that are not persistent, least recently used first.
    /// Saved as ModelCacheLimitGb; the first version's ModelCacheGb (which wrote its default of 24 into saved settings) is not read.
    /// </summary>
    public double ModelCacheLimitGb { get; set; }

    /// <summary><see cref="ModelCacheLimitGb"/> in bytes, null for no limit.</summary>
    public long? ModelCacheBytes => ModelCacheLimitGb > 0 ? (long)(ModelCacheLimitGb * 1e9) : null;

    /// <summary>
    /// GPU memory the loaded models may keep, in GB, for this browser's engine; 0 (the default) = one workflow's models at a time.
    /// With a number, the models of every workflow stay loaded while they fit it, and the one used longest ago is dropped when the next
    /// needs the room (TJ, 2026-11-25: an open workflow is probably still in use, and machines with far more memory than the test PC
    /// must be able to use it). It has to be the user's number: a browser tells a page nothing about GPU memory. It counts what the engine
    /// holds on the device, so leave room below the card's size for the browser, other tabs and a run's working memory.
    /// </summary>
    public double ModelMemoryGb { get; set; }

    /// <summary><see cref="ModelMemoryGb"/> in bytes, null for none set.</summary>
    public long? ModelMemoryBytes => ModelMemoryGb > 0 ? (long)(ModelMemoryGb * 1e9) : null;

    /// <summary>The run settings sent with each prompt (extra_data.spawngen_settings).</summary>
    public IReadOnlyDictionary<string, string> RunSettings() => new Dictionary<string, string>
    {
        [SpawnGen.Engine.Execution.RunSettings.Ltx2ComfyFp8Lora] = Ltx2ComfyFp8Lora ? "true" : "false",
        [SpawnGen.Engine.Execution.RunSettings.Ltx2Precision] = Ltx2Precision,
        [SpawnGen.Engine.Execution.RunSettings.Ltx2HostCacheGb] = Ltx2HostCacheGb.ToString(System.Globalization.CultureInfo.InvariantCulture),
        [SpawnGen.Engine.Execution.RunSettings.Ltx2ReadAheadGb] = Ltx2ReadAheadGb.ToString(System.Globalization.CultureInfo.InvariantCulture),
    };
    /// <summary>The user's keyboard shortcut changes, as <see cref="KeyBindings.ToJson"/> writes them (only what differs from the defaults).</summary>
    public System.Text.Json.Nodes.JsonObject Shortcuts { get; set; } = [];

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
            if (root.TryGetProperty(nameof(Shortcuts), out var sc) && sc.ValueKind == JsonValueKind.Object)
                s.Shortcuts = KeyBindings.FromJson(System.Text.Json.Nodes.JsonNode.Parse(sc.GetRawText())).ToJson();
            if (root.TryGetProperty(nameof(PreviewMethod), out var pm) && pm.ValueKind == JsonValueKind.String && PreviewMethods.Contains(pm.GetString())) s.PreviewMethod = pm.GetString()!;
            if (root.TryGetProperty(nameof(Ltx2ComfyFp8Lora), out var lf) && lf.ValueKind is JsonValueKind.True or JsonValueKind.False) s.Ltx2ComfyFp8Lora = lf.GetBoolean();
            if (root.TryGetProperty(nameof(Ltx2Precision), out var lp) && lp.ValueKind == JsonValueKind.String && SpawnGen.Engine.Execution.RunSettings.Ltx2Precisions.Contains(lp.GetString()))
                s.Ltx2Precision = lp.GetString()!;
            if (root.TryGetProperty(nameof(Ltx2HostCacheGb), out var hc) && hc.ValueKind == JsonValueKind.Number && hc.GetDouble() is >= 0 and <= 1024 and var cache) s.Ltx2HostCacheGb = cache;
            if (root.TryGetProperty(nameof(Ltx2ReadAheadGb), out var ra) && ra.ValueKind == JsonValueKind.Number && ra.GetDouble() is >= 0 and <= 1024 and var ahead) s.Ltx2ReadAheadGb = ahead;
            if (root.TryGetProperty(nameof(ModelCacheLimitGb), out var mc) && mc.ValueKind == JsonValueKind.Number && mc.GetDouble() is >= 0 and <= 4096 and var modelCache) s.ModelCacheLimitGb = modelCache;
            if (root.TryGetProperty(nameof(ModelMemoryGb), out var mm) && mm.ValueKind == JsonValueKind.Number && mm.GetDouble() is >= 0 and <= 4096 and var modelMemory) s.ModelMemoryGb = modelMemory;
        }
        catch (JsonException) { }
        return s;
    }
}
