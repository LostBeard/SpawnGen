using System.Text.Json.Nodes;
using SpawnGen.Editor;

namespace SpawnGen.Editor.Tests;

/// <summary>Keyboard shortcuts: ComfyUI's defaults, changing, removing and resetting keys, and what a pressed key becomes.</summary>
public class KeyBindingsTests
{
    static KeyChord C(string text) => KeyChord.Parse(text)!.Value;

    [Test]
    public void Chords_read_and_print()
    {
        Assert.That(C("Ctrl+Shift+E"), Is.EqualTo(KeyChord.Of("e", Modifiers.Ctrl | Modifiers.Shift)));
        Assert.That(C("ctrl+shift+e").ToString(), Is.EqualTo("Ctrl+Shift+E"));
        Assert.That(C("Alt++"), Is.EqualTo(KeyChord.Of("+", Modifiers.Alt)));
        Assert.That(C("Alt+Shift++").ToString(), Is.EqualTo("Alt+Shift++"));
        Assert.That(C("Delete").ToString(), Is.EqualTo("Delete"));
        Assert.That(C("Alt+µ").ToString(), Is.EqualTo("Alt+µ"), "only a-z are written in capitals");
        Assert.That(C(".").Key, Is.EqualTo("."));
        Assert.That(KeyChord.Parse("Hyper+K"), Is.Null);
        Assert.That(KeyChord.Parse(""), Is.Null);
        Assert.That(KeyChord.Of("K", Modifiers.Ctrl | Modifiers.Space), Is.EqualTo(C("Ctrl+K")), "Space (held to pan) is not part of a chord");
    }

    [Test]
    public void ComfyUIs_defaults_are_there_and_each_key_runs_one_command()
    {
        var b = new KeyBindings();
        Assert.That(b.CommandFor(C("Ctrl+Enter"))!.Id, Is.EqualTo("Comfy.QueuePrompt"));
        Assert.That(b.CommandFor(C("Ctrl+Alt+Enter"))!.Id, Is.EqualTo("Comfy.Interrupt"));
        Assert.That(b.CommandFor(C("."))!.Id, Is.EqualTo("Comfy.Canvas.FitView"));
        Assert.That(b.CommandFor(C("Alt+C"))!.Id, Is.EqualTo("Comfy.Canvas.ToggleSelectedNodes.Collapse"));
        Assert.That(b.CommandFor(C("Ctrl+Shift+K"))!.Id, Is.EqualTo("Workspace.ToggleKeybindings"));
        var all = KeyBindings.Commands.SelectMany(c => c.Defaults).ToList();
        Assert.That(all.Distinct().Count(), Is.EqualTo(all.Count), "no key is the default of two commands");
        Assert.That(b.Translate(C(".")), Is.EqualTo(C("F")), "a second default becomes the first, which the editor handles");
        Assert.That(b.Translate(C("Escape")), Is.EqualTo(C("Escape")), "keys no command uses pass through");
    }

    [Test]
    public void A_new_key_runs_the_command_and_the_old_one_does_nothing()
    {
        var b = new KeyBindings();
        Assert.That(b.Set("Comfy.Canvas.ToggleSelectedNodes.Bypass", C("Ctrl+Q")), Is.Null);
        Assert.That(b.Translate(C("Ctrl+Q")), Is.EqualTo(C("Ctrl+B")));
        Assert.That(b.Translate(C("Ctrl+B")), Is.Null, "the old default does nothing (the browser keeps it)");
        Assert.That(b.Keys("Comfy.Canvas.ToggleSelectedNodes.Bypass"), Is.EqualTo(new[] { C("Ctrl+Q") }));

        var taken = b.Set("Comfy.Canvas.ToggleSelectedNodes.Mute", C("Ctrl+Q"));
        Assert.That(taken!.Id, Is.EqualTo("Comfy.Canvas.ToggleSelectedNodes.Bypass"), "a key in use moves, and the list says from where");
        Assert.That(b.Keys("Comfy.Canvas.ToggleSelectedNodes.Bypass"), Is.Empty);
        Assert.That(b.Translate(C("Ctrl+Q")), Is.EqualTo(C("Ctrl+M")));

        b.Set("Comfy.SaveWorkflow", null);
        Assert.That(b.Translate(C("Ctrl+S")), Is.Null, "no keys: Ctrl+S is the browser's again");
    }

    [Test]
    public void Reset_puts_the_defaults_back_and_takes_them_from_where_they_went()
    {
        var b = new KeyBindings();
        b.Set("Comfy.Canvas.ToggleSelectedNodes.Mute", C("Ctrl+B")); // takes Bypass's key
        Assert.That(b.Keys("Comfy.Canvas.ToggleSelectedNodes.Bypass"), Is.Empty);
        b.Reset("Comfy.Canvas.ToggleSelectedNodes.Bypass");
        Assert.That(b.Keys("Comfy.Canvas.ToggleSelectedNodes.Bypass"), Is.EqualTo(new[] { C("Ctrl+B") }));
        Assert.That(b.Keys("Comfy.Canvas.ToggleSelectedNodes.Mute"), Is.Empty, "Mute gave Ctrl+B back");
        b.Set("Comfy.Canvas.ToggleSelectedNodes.Mute", C("Ctrl+M"));
        Assert.That(b.IsChanged("Comfy.Canvas.ToggleSelectedNodes.Mute"), Is.False, "back at its default is no change");
        b.Set("Comfy.Undo", C("Alt+Z"));
        b.ResetAll();
        Assert.That(b.ToJson().Count, Is.EqualTo(0));
        Assert.That(b.Translate(C("Ctrl+Z")), Is.EqualTo(C("Ctrl+Z")));
    }

    [Test]
    public void Changes_are_stored_with_the_settings()
    {
        var b = new KeyBindings();
        b.Set("Comfy.Undo", C("Alt+Z"));
        b.Set("Comfy.SaveWorkflow", null);
        var json = b.ToJson();
        Assert.That(((string)json["Comfy.Undo"]![0]!, json["Comfy.SaveWorkflow"]!.AsArray().Count, json.Count), Is.EqualTo(("Alt+Z", 0, 2)), "only the changes");

        var settings = new StudioSettings { Shortcuts = json };
        var back = KeyBindings.FromJson(StudioSettings.FromJson(settings.ToJson()).Shortcuts);
        Assert.That(back.Translate(C("Alt+Z")), Is.EqualTo(C("Ctrl+Z")));
        Assert.That(back.Keys("Comfy.SaveWorkflow"), Is.Empty);

        var odd = KeyBindings.FromJson(JsonNode.Parse("""{"No.Such":["Ctrl+X"],"Comfy.Undo":["Hyper+Z","Alt+Z"],"Comfy.Redo":"Ctrl+R"}"""));
        Assert.That(odd.Keys("Comfy.Undo"), Is.EqualTo(new[] { C("Alt+Z") }), "unreadable keys and unknown commands are skipped");
        Assert.That(odd.Keys("Comfy.Redo"), Has.Count.EqualTo(2), "a value that is not a list leaves the defaults");
        Assert.That(StudioSettings.FromJson("""{"Shortcuts": 5}""").Shortcuts.Count, Is.EqualTo(0));
    }

    [Test]
    public void Alt_C_collapses_the_selection_and_Escape_with_nothing_selected_is_left_to_the_Studio()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        var view = new EditorView(doc);
        Assert.That(view.Key("escape"), Is.False, "nothing to clear");
        view.SelectAll();
        Assert.That(view.Key("c", Modifiers.Alt), Is.True);
        Assert.That(doc.Graph.Nodes.All(n => n.Collapsed), Is.True);
        view.Key("c", Modifiers.Alt);
        Assert.That(doc.Graph.Nodes.Any(n => n.Collapsed), Is.False, "all collapsed: Alt+C expands them");
        doc.Undo();
        Assert.That(doc.Graph.Nodes.All(n => n.Collapsed), Is.True, "one undo step");
        Assert.That(view.Key("escape"), Is.True);
        Assert.That(view.SelectedNodes, Is.Empty);
    }
}
