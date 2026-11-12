using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Web;
using SpawnGen.Editor;

namespace SpawnGen.Studio.Pages;

// Keyboard shortcuts (ComfyUI's Keybinding panel): every command with its keys; click the keys and press new ones to change them, x takes them
// off, the arrow puts the default back. Changes are kept with the settings and apply at once.
public partial class Studio
{
    KeyBindings _bindings = new();
    bool _shortcutsOpen;
    string _shortcutFilter = "";
    string? _capturing;               // the command whose keys are being pressed
    string? _shortcutNote;            // what the last change did ("Ctrl+B was Bypass's: it has no keys now")
    bool _focusCapture;
    ElementReference _captureRef;

    void ToggleShortcuts()
    {
        _shortcutsOpen = !_shortcutsOpen;
        _capturing = null;
        _shortcutNote = null;
        _settingsOpen = false;
        if (!_shortcutsOpen) _studio?.Focus();
    }

    IEnumerable<ShortcutCommand> ShortcutsShown()
    {
        var words = _shortcutFilter.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        return KeyBindings.Commands.Where(c => words.All(w =>
            c.Label.Contains(w, StringComparison.OrdinalIgnoreCase) || _bindings.Keys(c.Id).Any(k => k.ToString().Contains(w, StringComparison.OrdinalIgnoreCase))));
    }

    void OnShortcutFilter(ChangeEventArgs e) => _shortcutFilter = e.Value?.ToString() ?? "";

    void StartCapture(string id)
    {
        _capturing = id;
        _shortcutNote = "Press the new keys (Escape cancels).";
        _focusCapture = true;
    }

    /// <summary>The keys pressed while a command waits for its new ones: modifiers alone wait for the key; Escape cancels.</summary>
    void OnCaptureKey(KeyboardEventArgs e, string id)
    {
        if (_capturing != id || KeyChord.IsModifierKey(e.Key)) return;
        _capturing = null;
        if (e.Key == "Escape") { _shortcutNote = null; return; }
        var mods = (e.CtrlKey || e.MetaKey ? Modifiers.Ctrl : 0) | (e.ShiftKey ? Modifiers.Shift : 0) | (e.AltKey ? Modifiers.Alt : 0);
        var chord = KeyChord.Of(e.Key, mods);
        var taken = _bindings.Set(id, chord);
        _shortcutNote = taken is null ? $"{KeyBindings.Command(id)?.Label}: {chord}." : $"{chord} was {taken.Label}'s; it is {KeyBindings.Command(id)?.Label}'s now.";
        SaveShortcuts();
    }

    void UnbindShortcut(string id)
    {
        _bindings.Set(id, null);
        _shortcutNote = $"{KeyBindings.Command(id)?.Label} has no keys now.";
        SaveShortcuts();
    }

    void ResetShortcut(string id)
    {
        _bindings.Reset(id);
        _shortcutNote = null;
        SaveShortcuts();
    }

    void ResetAllShortcuts()
    {
        _bindings.ResetAll();
        _shortcutNote = "Every shortcut is back to its default.";
        SaveShortcuts();
    }

    void SaveShortcuts()
    {
        _settings.Shortcuts = _bindings.ToJson();
        ApplySettings(); // stores the settings and gives the canvas the new keys
    }
}
