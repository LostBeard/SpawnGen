using System.Text.Json;
using System.Text.Json.Nodes;

namespace SpawnGen.Editor;

/// <summary>A key with its modifiers (Ctrl, Shift, Alt): <c>Ctrl+Shift+E</c>. Keys are the browser's key names in lower case ("e", "enter", ".", "delete").</summary>
public readonly record struct KeyChord(string Key, Modifiers Mods)
{
    const Modifiers Mask = Modifiers.Ctrl | Modifiers.Shift | Modifiers.Alt;

    public static KeyChord Of(string key, Modifiers mods) => new(key.ToLowerInvariant(), mods & Mask);

    /// <summary>"Ctrl+Shift+E", "Alt+=", "Delete": what the shortcuts list shows and the settings store.</summary>
    public override string ToString()
    {
        var parts = new List<string>();
        if (Mods.HasFlag(Modifiers.Ctrl)) parts.Add("Ctrl");
        if (Mods.HasFlag(Modifiers.Alt)) parts.Add("Alt");
        if (Mods.HasFlag(Modifiers.Shift)) parts.Add("Shift");
        parts.Add(Key.Length == 1 ? (char.IsAsciiLetter(Key[0]) ? Key.ToUpperInvariant() : Key) : char.ToUpperInvariant(Key[0]) + Key[1..]);
        return string.Join("+", parts);
    }

    /// <summary>Reads "Ctrl+Shift+E" back (any case; "+" alone or last is the plus key). Null for text that is not a chord.</summary>
    public static KeyChord? Parse(string? text)
    {
        if (string.IsNullOrWhiteSpace(text)) return null;
        var mods = Modifiers.None;
        var rest = text.Trim();
        while (true)
        {
            int plus = rest.IndexOf('+', 1);
            if (plus < 0) break;
            var mod = rest[..plus].Trim().ToLowerInvariant();
            if (mod is "ctrl" or "control" or "cmd" or "meta") mods |= Modifiers.Ctrl;
            else if (mod == "shift") mods |= Modifiers.Shift;
            else if (mod == "alt") mods |= Modifiers.Alt;
            else return null;
            rest = rest[(plus + 1)..];
        }
        rest = rest.Trim();
        return rest.Length == 0 ? null : Of(rest, mods);
    }

    /// <summary>Keys that only modify, which a chord cannot end with.</summary>
    public static bool IsModifierKey(string key) => key.ToLowerInvariant() is "control" or "shift" or "alt" or "meta" or "altgraph" or "capslock" or "os";
}

/// <summary>A command the keyboard can run: its id (ComfyUI's where it has one), the name the shortcuts list shows, and its default keys.</summary>
public sealed record ShortcutCommand(string Id, string Label, IReadOnlyList<KeyChord> Defaults)
{
    /// <summary>The chord the editor and the Studio act on for this command (its first default): a rebound key is turned into this one.</summary>
    public KeyChord Canonical => Defaults[0];
}

/// <summary>
/// The keyboard shortcuts and the user's changes to them (Settings, Keyboard shortcuts; ComfyUI's Keybinding panel). The defaults are ComfyUI's
/// where it has the command (read from its keybinding panel), SpawnGen's own otherwise. A pressed chord is turned into the default chord of the
/// command it is bound to, so the editor keeps handling only the defaults; a default chord whose command was moved away does nothing.
/// </summary>
public sealed class KeyBindings
{
    static KeyChord K(string key, Modifiers mods = Modifiers.None) => KeyChord.Of(key, mods);
    const Modifiers Ctrl = Modifiers.Ctrl, Shift = Modifiers.Shift, Alt = Modifiers.Alt;

    /// <summary>Every command with its defaults; the order is the list's.</summary>
    public static readonly IReadOnlyList<ShortcutCommand> Commands =
    [
        new("Comfy.QueuePrompt", "Queue Prompt", [K("enter", Ctrl)]),
        new("Comfy.QueuePromptFront", "Queue Prompt (Front)", [K("enter", Ctrl | Shift)]),
        new("Comfy.Interrupt", "Interrupt", [K("enter", Ctrl | Alt)]),
        new("Comfy.SaveWorkflow", "Save Workflow", [K("s", Ctrl)]),
        new("Comfy.OpenWorkflow", "Open Workflow", [K("o", Ctrl)]),
        new("Comfy.Undo", "Undo", [K("z", Ctrl)]),
        new("Comfy.Redo", "Redo", [K("y", Ctrl), K("z", Ctrl | Shift)]),
        new("Comfy.Canvas.SelectAll", "Select All", [K("a", Ctrl)]),
        new("Comfy.Canvas.Copy", "Copy", [K("c", Ctrl)]),
        new("Comfy.Canvas.Paste", "Paste", [K("v", Ctrl)]),
        new("Comfy.Canvas.PasteWithConnect", "Paste with Connect", [K("v", Ctrl | Shift)]),
        new("Comfy.Canvas.DeleteSelectedItems", "Delete Selected Items", [K("delete"), K("backspace")]),
        new("Comfy.Graph.GroupSelectedNodes", "Group Selected Nodes", [K("g", Ctrl)]),
        new("Comfy.Graph.ConvertToSubgraph", "Convert Selection to Subgraph / Unpack", [K("e", Ctrl | Shift)]),
        new("Comfy.Canvas.ToggleSelectedNodes.Mute", "Mute/Unmute Selected Nodes", [K("m", Ctrl)]),
        new("Comfy.Canvas.ToggleSelectedNodes.Bypass", "Bypass/Unbypass Selected Nodes", [K("b", Ctrl)]),
        new("Comfy.Canvas.ToggleSelected.Pin", "Pin/Unpin Selected Items", [K("p")]),
        new("Comfy.Canvas.ToggleSelectedNodes.Collapse", "Collapse/Expand Selected Nodes", [K("c", Alt)]),
        new("Comfy.Canvas.FitView", "Fit view to selected nodes", [K("f"), K("."), K("home")]),
        new("Comfy.Canvas.ZoomIn", "Zoom In", [K("=", Alt), K("+", Alt | Shift), K("+", Alt)]),
        new("Comfy.Canvas.ZoomOut", "Zoom Out", [K("-", Alt)]),
        new("Comfy.Canvas.ToggleMinimap", "Canvas Toggle Minimap", [K("m", Alt | Shift)]),
        new("Comfy.ToggleLinear", "Toggle App Mode", [K("m", Alt), K("µ", Alt)]),
        new("Workspace.ToggleSidebarTab.node-library", "Toggle Node Library Sidebar", [K("n")]),
        new("Workspace.ToggleSidebarTab.workflows", "Toggle Workflows Sidebar", [K("w")]),
        new("Workspace.ToggleSidebarTab.model-library", "Toggle Models Panel", [K("m")]),
        new("SpawnGen.ToggleHistory", "Toggle History Panel", [K("h")]),
        new("SpawnGen.ToggleProperties", "Toggle Properties Panel", [K("i")]),
        new("Comfy.ShowSettingsDialog", "Show Settings Dialog", [K(",", Ctrl)]),
        new("Workspace.ToggleKeybindings", "Show Keyboard Shortcuts", [K("k", Ctrl | Shift)]),
    ];

    /// <summary>The user's keys per command id; a command not in here has its defaults. An empty list means "no keys".</summary>
    readonly Dictionary<string, List<KeyChord>> _changed = new(StringComparer.Ordinal);

    public static ShortcutCommand? Command(string id) => Commands.FirstOrDefault(c => c.Id == id);

    /// <summary>The keys a command answers to now.</summary>
    public IReadOnlyList<KeyChord> Keys(string id) =>
        _changed.TryGetValue(id, out var keys) ? keys : Command(id)?.Defaults ?? [];

    public bool IsChanged(string id) => _changed.ContainsKey(id);

    /// <summary>The command a chord runs now, or null.</summary>
    public ShortcutCommand? CommandFor(KeyChord chord) => Commands.FirstOrDefault(c => Keys(c.Id).Contains(chord));

    /// <summary>
    /// Gives a command one key (null: no key). The chord is taken off any other command that had it; that command is returned so the list can say
    /// so (null when there was none).
    /// </summary>
    public ShortcutCommand? Set(string id, KeyChord? chord)
    {
        if (Command(id) is null) return null;
        ShortcutCommand? taken = null;
        if (chord is { } c)
            foreach (var other in Commands.Where(o => o.Id != id && Keys(o.Id).Contains(c)).ToList())
            {
                _changed[other.Id] = [.. Keys(other.Id).Where(k => k != c)];
                taken = other;
            }
        _changed[id] = chord is { } k ? [k] : [];
        Normalise();
        return taken;
    }

    /// <summary>Back to the default keys, taking them off commands they were given to since.</summary>
    public void Reset(string id)
    {
        if (Command(id) is not { } command) return;
        _changed.Remove(id);
        foreach (var other in Commands.Where(o => o.Id != id && _changed.ContainsKey(o.Id)))
            _changed[other.Id] = [.. _changed[other.Id].Where(k => !command.Defaults.Contains(k))];
        Normalise();
    }

    public void ResetAll() => _changed.Clear();

    /// <summary>A change that ends where the defaults are is no change.</summary>
    void Normalise()
    {
        foreach (var id in _changed.Keys.ToList())
            if (Command(id) is { } c && _changed[id].SequenceEqual(c.Defaults)) _changed.Remove(id);
    }

    /// <summary>
    /// What a pressed chord means to the editor: the default chord of the command it is bound to; null for a default chord whose command now has
    /// other keys (nothing happens, the browser keeps the key); the chord itself for keys no command uses (Escape, Space ...).
    /// </summary>
    public KeyChord? Translate(KeyChord pressed)
    {
        if (CommandFor(pressed) is { } command) return command.Canonical;
        if (Commands.Any(c => c.Defaults.Contains(pressed))) return null;
        return pressed;
    }

    /// <summary>The changes only, as {command id: ["Ctrl+K", ...]} (what the settings keep).</summary>
    public JsonObject ToJson()
    {
        var o = new JsonObject();
        foreach (var (id, keys) in _changed) o[id] = new JsonArray([.. keys.Select(k => (JsonNode)k.ToString())]);
        return o;
    }

    /// <summary>Reads <see cref="ToJson"/> back; unknown commands and unreadable keys are skipped.</summary>
    public static KeyBindings FromJson(JsonNode? node)
    {
        var b = new KeyBindings();
        if (node is not JsonObject o) return b;
        foreach (var (id, value) in o)
            if (Command(id) is not null && value is JsonArray list)
                b._changed[id] = [.. list.Select(v => v is JsonValue s && s.TryGetValue<string>(out var t) ? KeyChord.Parse(t) : null).OfType<KeyChord>().Distinct()];
        b.Normalise();
        return b;
    }
}
