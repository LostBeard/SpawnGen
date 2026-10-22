using SpawnGen.Editor;

namespace SpawnGen.Editor.Tests;

public class WorkflowTabsTests
{
    [Test]
    public void Adding_a_tab_makes_it_active_and_names_stay_unique()
    {
        var tabs = new WorkflowTabs();
        var a = tabs.Add("Portrait");
        var b = tabs.Add("Portrait");
        var c = tabs.Add("  ");
        Assert.That(tabs.ActiveId, Is.EqualTo(c.Id));
        Assert.That(tabs.Tabs.Select(t => t.Name), Is.EqualTo(new[] { "Portrait", "Portrait (2)", "Workflow" }));
        Assert.That(new[] { a.Id, b.Id, c.Id }.Distinct().Count(), Is.EqualTo(3));
    }

    [Test]
    public void Closing_the_active_tab_activates_its_left_neighbour_and_the_last_tab_cannot_be_closed()
    {
        var tabs = new WorkflowTabs();
        var a = tabs.Add("A"); var b = tabs.Add("B"); var c = tabs.Add("C");
        tabs.Activate(b.Id);
        Assert.That(tabs.Close(b.Id), Is.True);
        Assert.That(tabs.ActiveId, Is.EqualTo(a.Id));
        Assert.That(tabs.Close(c.Id), Is.True);
        Assert.That(tabs.ActiveId, Is.EqualTo(a.Id), "closing another tab keeps the active one");
        Assert.That(tabs.Close(a.Id), Is.False);
        Assert.That(tabs.Tabs, Has.Count.EqualTo(1));
        Assert.That(tabs.Close("nope"), Is.False);
    }

    [Test]
    public void Closing_the_first_tab_activates_the_next()
    {
        var tabs = new WorkflowTabs();
        var a = tabs.Add("A"); var b = tabs.Add("B");
        tabs.Activate(a.Id);
        tabs.Close(a.Id);
        Assert.That(tabs.ActiveId, Is.EqualTo(b.Id));
    }

    [Test]
    public void Rename_keeps_names_unique_and_ignores_blank_names()
    {
        var tabs = new WorkflowTabs();
        var a = tabs.Add("A"); var b = tabs.Add("B");
        tabs.Rename(b.Id, "A");
        Assert.That(b.Name, Is.EqualTo("A (2)"));
        tabs.Rename(b.Id, "A (2)");
        Assert.That(b.Name, Is.EqualTo("A (2)"), "a tab may keep its own name");
        tabs.Rename(a.Id, "   ");
        Assert.That(a.Name, Is.EqualTo("A"));
    }

    [Test]
    public void Tabs_survive_a_round_trip_through_text_with_workflow_and_view()
    {
        var tabs = new WorkflowTabs();
        var a = tabs.Add("First", "{\"nodes\": []}");
        a.Scale = 0.5; a.OffsetX = 120; a.OffsetY = -40;
        var b = tabs.Add("Second");
        tabs.Activate(a.Id);

        var back = WorkflowTabs.FromJson(tabs.ToJson())!;
        Assert.That(back.Tabs.Select(t => t.Name), Is.EqualTo(new[] { "First", "Second" }));
        Assert.That(back.ActiveId, Is.EqualTo(a.Id));
        Assert.That(back.Tabs[0].Json, Is.EqualTo("{\"nodes\": []}"));
        Assert.That((back.Tabs[0].Scale, back.Tabs[0].OffsetX, back.Tabs[0].OffsetY), Is.EqualTo((0.5, 120.0, -40.0)));
        Assert.That(back.Tabs[1].Json, Is.Null);
        Assert.That(back.Tabs[1].Id, Is.EqualTo(b.Id));
    }

    [TestCase(null)]
    [TestCase("")]
    [TestCase("not json")]
    [TestCase("{\"version\": 99, \"tabs\": []}")]
    [TestCase("{\"version\": 1, \"tabs\": []}")]
    [TestCase("[1, 2]")]
    public void Missing_damaged_or_unknown_stored_text_starts_fresh(string? text) => Assert.That(WorkflowTabs.FromJson(text), Is.Null);

    [Test]
    public void A_stored_active_id_that_no_longer_exists_falls_back_to_the_first_tab_and_views_are_clamped()
    {
        var back = WorkflowTabs.FromJson("""{"version": 1, "active": "gone", "tabs": [{"id": "a", "name": "A", "json": null, "scale": 99, "x": 0, "y": 0}, {"id": "a", "name": "dup"}]}""")!;
        Assert.That(back.ActiveId, Is.EqualTo("a"));
        Assert.That(back.Tabs, Has.Count.EqualTo(1), "a duplicate id is dropped");
        Assert.That(back.Tabs[0].Scale, Is.EqualTo(Viewport.MaxScale));
    }
}
