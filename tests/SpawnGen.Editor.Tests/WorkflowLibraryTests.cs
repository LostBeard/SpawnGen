using SpawnGen.Editor;

namespace SpawnGen.Editor.Tests;

/// <summary>Names in the workflow library (ComfyUI's workflows/ folder, apps as .app.json) and the tab that remembers its saved file.</summary>
public class WorkflowLibraryTests
{
    [Test]
    public void Typed_names_become_library_paths()
    {
        Assert.That(WorkflowLibrary.PathFor("My flow", app: false), Is.EqualTo("My flow.json"));
        Assert.That(WorkflowLibrary.PathFor("My flow", app: true), Is.EqualTo("My flow.app.json"));
        Assert.That(WorkflowLibrary.PathFor("  portraits/ hires fix ", app: false), Is.EqualTo("portraits/hires fix.json"), "a / makes a folder");
        Assert.That(WorkflowLibrary.PathFor("a.json", app: false), Is.EqualTo("a.json"), "a typed suffix is not doubled");
        Assert.That(WorkflowLibrary.PathFor("a.app.json", app: false), Is.EqualTo("a.json"));
        Assert.That(WorkflowLibrary.PathFor("what? <now>: \"x\" | *", app: false), Is.EqualTo("what_ _now__ _x_ _ _.json"));
        Assert.That(WorkflowLibrary.PathFor("../../etc/passwd", app: false), Is.EqualTo("etc/passwd.json"), "no way out of the folder");
        Assert.That(WorkflowLibrary.PathFor(@"a\b", app: false), Is.EqualTo("a/b.json"));
        Assert.That(WorkflowLibrary.PathFor("   ", app: false), Is.Null);
        Assert.That(WorkflowLibrary.PathFor("/./..", app: true), Is.Null);
    }

    [Test]
    public void Apps_are_told_apart_named_and_kept_apps_when_renamed()
    {
        var app = new WorkflowEntry("sub/Portrait.app.json", 10, 0);
        var flow = new WorkflowEntry("Portrait.json", 10, 0);
        Assert.That((app.IsApp, app.Name), Is.EqualTo((true, "sub/Portrait")));
        Assert.That((flow.IsApp, flow.Name), Is.EqualTo((false, "Portrait")));
        Assert.That(WorkflowLibrary.Renamed(app.Path, "Faces"), Is.EqualTo("Faces.app.json"));
        Assert.That(WorkflowLibrary.Renamed(flow.Path, "Faces"), Is.EqualTo("Faces.json"));
    }

    [Test]
    public void The_list_is_sorted_by_name_and_searched_by_words()
    {
        var entries = new[] { new WorkflowEntry("b.json", 1, 0), new WorkflowEntry("A hires.app.json", 1, 0), new WorkflowEntry("sub/c.json", 1, 0) };
        Assert.That(WorkflowLibrary.Sorted(entries).Select(e => e.Path), Is.EqualTo(new[] { "A hires.app.json", "b.json", "sub/c.json" }));
        Assert.That(entries.Where(e => WorkflowLibrary.Matches(e, "HIRES a")).Select(e => e.Path), Is.EqualTo(new[] { "A hires.app.json" }));
        Assert.That(entries.Count(e => WorkflowLibrary.Matches(e, " ")), Is.EqualTo(3));
        Assert.That(entries.Count(e => WorkflowLibrary.Matches(e, "json")), Is.EqualTo(0), "the suffix is not part of the name");
    }

    [Test]
    public void A_tab_remembers_its_saved_file_across_a_reload()
    {
        var tabs = new WorkflowTabs();
        tabs.Add("one").Path = "sub/one.app.json";
        tabs.Add("two");
        var back = WorkflowTabs.FromJson(tabs.ToJson())!;
        Assert.That(back.Tabs.Select(t => t.Path), Is.EqualTo(new[] { "sub/one.app.json", null }));
    }
}
