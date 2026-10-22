using SpawnGen.Editor;

namespace SpawnGen.Editor.Tests;

/// <summary>Pointer and keyboard behaviour of the editor, driven the way a canvas would drive it. No browser, no GPU.</summary>
public class EditorViewTests
{
    static (EditorView View, EditorDocument Doc, Dictionary<string, int> Ids) Setup()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        return (new EditorView(doc), doc, ids);
    }

    static Point Screen(EditorView v, Point world) => v.Viewport.WorldToScreen(world);

    static void Drag(EditorView v, Point fromWorld, Point toWorld, Modifiers mods = Modifiers.None, int steps = 6)
    {
        var a = Screen(v, fromWorld);
        var b = Screen(v, toWorld);
        v.PointerDown(a, PointerButton.Left, mods);
        for (int i = 1; i <= steps; i++)
            v.PointerMove(new Point(a.X + (b.X - a.X) * i / steps, a.Y + (b.Y - a.Y) * i / steps), mods);
        v.PointerUp(b, PointerButton.Left, mods);
    }

    static Point TitleOf(EditorView v, EditorNode n) { var g = v.Hits.Geometry(n); return new Point(g.TitleBar.X + 80, g.TitleBar.Y + 15); }
    static Point OutputOf(EditorView v, int node, int output) => v.Hits.OutputPosition(node, output)!.Value;
    static Point InputOf(EditorView v, int node, string input) => v.Hits.InputPosition(node, input)!.Value;

    // ---- viewport ----

    [Test]
    public void Wheel_zoom_keeps_the_world_point_under_the_cursor_fixed()
    {
        var (view, _, _) = Setup();
        var cursor = new Point(400, 300);
        var before = view.Viewport.ScreenToWorld(cursor);
        view.Wheel(cursor, -300);
        view.Wheel(cursor, -300);
        var after = view.Viewport.ScreenToWorld(cursor);
        Assert.That(view.Viewport.Scale, Is.GreaterThan(1.5));
        Assert.That(after.X, Is.EqualTo(before.X).Within(1e-9));
        Assert.That(after.Y, Is.EqualTo(before.Y).Within(1e-9));
    }

    [Test]
    public void Zoom_is_clamped()
    {
        var (view, _, _) = Setup();
        for (int i = 0; i < 60; i++) view.Wheel(new Point(10, 10), -500);
        Assert.That(view.Viewport.Scale, Is.EqualTo(Viewport.MaxScale));
        for (int i = 0; i < 120; i++) view.Wheel(new Point(10, 10), 500);
        Assert.That(view.Viewport.Scale, Is.EqualTo(Viewport.MinScale));
    }

    [Test]
    public void Dragging_the_empty_background_pans_and_a_plain_click_on_it_deselects()
    {
        var (view, _, ids) = Setup();
        view.SelectOnly(ids["pos"]);
        var empty = new Point(-300, -300); // nothing there
        Drag(view, empty, new Point(-200, -250));
        Assert.That((view.Viewport.OffsetX, view.Viewport.OffsetY), Is.EqualTo((100.0, 50.0)));
        Assert.That(view.SelectedNodes, Has.Count.EqualTo(1), "A pan must not deselect.");

        view.PointerDown(new Point(5, 5), PointerButton.Left);
        view.PointerUp(new Point(5, 5), PointerButton.Left);
        Assert.That(view.SelectedNodes, Is.Empty);
    }

    [Test]
    public void Fit_shows_every_node()
    {
        var (view, doc, _) = Setup();
        view.ScreenSize = new Point(900, 600);
        view.FitToView();
        var visible = view.Viewport.VisibleWorld(900, 600);
        foreach (var n in doc.Graph.Nodes)
        {
            var b = view.Hits.Geometry(n).Bounds;
            Assert.That(visible.Contains(new Point(b.X, b.Y)) && visible.Contains(new Point(b.Right, b.Bottom)), Is.True, $"node {n.Id} is off screen");
        }
    }

    // ---- selecting and moving ----

    [Test]
    public void Clicking_a_node_selects_it_and_shift_click_adds_another()
    {
        var (view, doc, ids) = Setup();
        view.PointerDown(Screen(view, TitleOf(view, doc.Graph.GetNode(ids["pos"]))), PointerButton.Left);
        view.PointerUp(Screen(view, TitleOf(view, doc.Graph.GetNode(ids["pos"]))), PointerButton.Left);
        Assert.That(view.SelectedNodes, Is.EquivalentTo(new[] { ids["pos"] }));

        var neg = Screen(view, TitleOf(view, doc.Graph.GetNode(ids["neg"])));
        view.PointerDown(neg, PointerButton.Left, Modifiers.Shift);
        view.PointerUp(neg, PointerButton.Left, Modifiers.Shift);
        Assert.That(view.SelectedNodes, Is.EquivalentTo(new[] { ids["pos"], ids["neg"] }));
    }

    [Test]
    public void Dragging_a_node_moves_every_selected_node_together_as_one_undo_step()
    {
        var (view, doc, ids) = Setup();
        view.SelectOnly(ids["pos"]);
        view.ToggleSelect(ids["neg"]);
        var pos = doc.Graph.GetNode(ids["pos"]);
        var neg = doc.Graph.GetNode(ids["neg"]);
        var other = doc.Graph.GetNode(ids["sampler"]);
        (double px, double py, double nx, double ny, double ox) = (pos.X, pos.Y, neg.X, neg.Y, other.X);

        Drag(view, TitleOf(view, pos), TitleOf(view, pos) + new Point(120, 70));

        Assert.That(pos.X, Is.EqualTo(px + 120).Within(1e-6)); Assert.That(pos.Y, Is.EqualTo(py + 70).Within(1e-6));
        Assert.That(neg.X, Is.EqualTo(nx + 120).Within(1e-6)); Assert.That(neg.Y, Is.EqualTo(ny + 70).Within(1e-6));
        Assert.That(other.X, Is.EqualTo(ox), "An unselected node must not move.");
        doc.Undo();
        Assert.That(new[] { pos.X, pos.Y, neg.X, neg.Y }, Is.EqualTo(new[] { px, py, nx, ny }).Within(1e-6));
    }

    [Test]
    public void A_click_without_movement_does_not_move_or_create_an_undo_step()
    {
        var (view, doc, ids) = Setup();
        var node = doc.Graph.GetNode(ids["pos"]);
        double x = node.X;
        var p = Screen(view, TitleOf(view, node));
        view.PointerDown(p, PointerButton.Left);
        view.PointerMove(p + new Point(1, 1)); // under the drag threshold
        view.PointerUp(p, PointerButton.Left);
        Assert.That(node.X, Is.EqualTo(x));
        Assert.That(doc.CanUndo, Is.False);
    }

    [Test]
    public void Box_select_with_ctrl_selects_the_nodes_it_touches()
    {
        var (view, _, ids) = Setup();
        // Around the two text-encode nodes (x 400, y 0 and 250) but not the sampler (x 850).
        Drag(view, new Point(380, -40), new Point(700, 520), Modifiers.Ctrl);
        Assert.That(view.SelectedNodes, Is.EquivalentTo(new[] { ids["pos"], ids["neg"], ids["latent"] }));
        Assert.That(view.SelectionBox, Is.Null, "The box goes away on release.");
    }

    [Test]
    public void Clicking_a_node_brings_it_to_the_front()
    {
        var (view, doc, ids) = Setup();
        var a = doc.Graph.GetNode(ids["pos"]);
        view.SelectOnly(ids["ckpt"]);
        Assert.That(doc.Graph.GetNode(ids["ckpt"]).ZIndex, Is.GreaterThan(a.ZIndex));
    }

    // ---- hit testing ----

    [Test]
    public void Slots_win_over_the_node_body_and_report_name_and_index()
    {
        var (view, _, ids) = Setup();
        var hit = view.Hits.Test(InputOf(view, ids["sampler"], "positive"));
        Assert.That((hit.Kind, hit.NodeId, hit.Name), Is.EqualTo((HitKind.InputSlot, ids["sampler"], "positive")));

        var outHit = view.Hits.Test(OutputOf(view, ids["ckpt"], 2));
        Assert.That((outHit.Kind, outHit.NodeId, outHit.Index), Is.EqualTo((HitKind.OutputSlot, ids["ckpt"], 2)));
    }

    [Test]
    public void A_link_is_picked_by_its_curve_but_a_node_on_top_of_it_wins()
    {
        var (view, doc, ids) = Setup();
        var link = doc.Graph.LinkInto(ids["decode"], "samples")!;
        var mid = LinkGeometry.Midpoint(OutputOf(view, link.FromNode, link.FromOutput), InputOf(view, link.ToNode, link.ToInput));
        var hit = view.Hits.Test(mid);
        Assert.That((hit.Kind, hit.LinkId), Is.EqualTo((HitKind.Link, link.Id)));

        // Put a node over the midpoint: now the node is hit, not the link.
        var cover = doc.AddNode("VAEDecode", mid.X - 40, mid.Y - 20);
        Assert.That(view.Hits.Test(mid).NodeId, Is.EqualTo(cover.Id));
    }

    [Test]
    public void The_collapse_button_folds_a_node_and_its_links_stay_attached()
    {
        var (view, doc, ids) = Setup();
        var g = view.Hits.Geometry(doc.Graph.GetNode(ids["sampler"]));
        var button = new Point(g.TitleBar.X + 10, g.TitleBar.Y + 15);
        view.PointerDown(Screen(view, button), PointerButton.Left);
        view.PointerUp(Screen(view, button), PointerButton.Left);

        var node = doc.Graph.GetNode(ids["sampler"]);
        Assert.That(node.Collapsed, Is.True);
        var collapsed = view.Hits.Geometry(node);
        Assert.That(collapsed.Bounds.Height, Is.EqualTo(NodeLayout.TitleHeight));
        Assert.That(view.Hits.InputPosition(ids["sampler"], "positive"), Is.EqualTo(new Point(collapsed.Bounds.X, collapsed.Bounds.Center.Y)));
    }

    // ---- links ----

    [Test]
    public void Dragging_from_an_output_to_a_compatible_input_connects_them()
    {
        var (view, doc, ids) = Setup();
        var extra = doc.AddNode("CLIPTextEncode", 400, 800);
        view.PointerDown(Screen(view, OutputOf(view, ids["ckpt"], 1)), PointerButton.Left); // CLIP out
        Assert.That(view.Draft, Is.Not.Null);
        view.PointerMove(Screen(view, InputOf(view, extra.Id, "clip")) + new Point(2, 0));
        Assert.That(view.Draft!.Target, Is.Not.Null, "A valid target is highlighted while hovering it.");
        view.PointerUp(Screen(view, InputOf(view, extra.Id, "clip")), PointerButton.Left);

        var link = doc.Graph.LinkInto(extra.Id, "clip");
        Assert.That((link?.FromNode, link?.FromOutput), Is.EqualTo((ids["ckpt"], 1)));
        Assert.That(view.Draft, Is.Null);
    }

    [Test]
    public void Dropping_on_an_incompatible_input_reports_why_and_changes_nothing()
    {
        var (view, doc, ids) = Setup();
        string? message = null;
        view.Message += m => message = m;
        int links = doc.Graph.Links.Count;

        // MODEL output onto the VAE decode's "samples" (LATENT) input.
        view.PointerDown(Screen(view, OutputOf(view, ids["ckpt"], 0)), PointerButton.Left);
        view.PointerMove(Screen(view, InputOf(view, ids["decode"], "samples")));
        Assert.That(view.Draft!.Problem, Does.Contain("MODEL"));
        view.PointerUp(Screen(view, InputOf(view, ids["decode"], "samples")), PointerButton.Left);

        Assert.That(message, Does.Contain("MODEL cannot connect to LATENT"));
        Assert.That(doc.Graph.Links.Count, Is.EqualTo(links));
    }

    [Test]
    public void Dropping_a_link_on_a_node_body_connects_to_its_first_matching_free_input()
    {
        var (view, doc, ids) = Setup();
        var node = doc.AddNode("KSampler", 800, 700);
        var g = view.Hits.Geometry(node);

        view.PointerDown(Screen(view, OutputOf(view, ids["ckpt"], 0)), PointerButton.Left); // MODEL
        view.PointerMove(Screen(view, new Point(g.Bounds.X + 60, g.Bounds.Y + 12)));         // over the title, not a slot
        view.PointerUp(Screen(view, new Point(g.Bounds.X + 60, g.Bounds.Y + 12)), PointerButton.Left);

        Assert.That(doc.Graph.LinkInto(node.Id, "model")?.FromNode, Is.EqualTo(ids["ckpt"]));
    }

    [Test]
    public void Dragging_a_connected_input_moves_its_link_to_another_input_as_one_undo_step()
    {
        var (view, doc, ids) = Setup();
        // Move the positive prompt's link from sampler.positive over to sampler.negative: the old negative link is replaced.
        var pos = doc.Graph.LinkInto(ids["sampler"], "positive")!;
        view.PointerDown(Screen(view, InputOf(view, ids["sampler"], "positive")), PointerButton.Left);
        Assert.That(view.Draft!.PickedUp?.Id, Is.EqualTo(pos.Id));
        view.PointerMove(Screen(view, InputOf(view, ids["sampler"], "negative")));
        view.PointerUp(Screen(view, InputOf(view, ids["sampler"], "negative")), PointerButton.Left);

        Assert.That(doc.Graph.LinkInto(ids["sampler"], "positive"), Is.Null);
        Assert.That(doc.Graph.LinkInto(ids["sampler"], "negative")!.FromNode, Is.EqualTo(ids["pos"]));

        doc.Undo();
        Assert.That(doc.Graph.LinkInto(ids["sampler"], "positive")!.FromNode, Is.EqualTo(ids["pos"]));
        Assert.That(doc.Graph.LinkInto(ids["sampler"], "negative")!.FromNode, Is.EqualTo(ids["neg"]));
    }

    [Test]
    public void Dragging_a_connected_input_off_into_empty_space_removes_the_link_and_offers_search()
    {
        var (view, doc, ids) = Setup();
        SpawnGen.Editor.LinkDraft? offered = null;
        Point? at = null;
        view.SearchRequested += (p, d) => { at = p; offered = d; };

        var from = InputOf(view, ids["sampler"], "positive");
        Drag(view, from, new Point(1000, 900));

        Assert.That(doc.Graph.LinkInto(ids["sampler"], "positive"), Is.Null);
        Assert.That(at, Is.Not.Null);
        Assert.That((offered?.StartsAtOutput, offered?.NodeId), Is.EqualTo((true, ids["pos"])));
    }

    [Test]
    public void Releasing_a_new_link_on_empty_canvas_offers_the_node_search_and_placing_a_node_connects_it()
    {
        var (view, doc, ids) = Setup();
        SpawnGen.Editor.LinkDraft? draft = null;
        Point at = default;
        view.SearchRequested += (p, d) => { at = p; draft = d; };

        Drag(view, OutputOf(view, ids["latent"], 0), new Point(900, 700)); // LATENT out dropped in the open
        Assert.That(draft, Is.Not.Null);

        var node = view.PlaceNode("VAEDecode", at, draft);
        Assert.That(doc.Graph.LinkInto(node.Id, "samples")?.FromNode, Is.EqualTo(ids["latent"]));
        Assert.That(view.SelectedNodes, Is.EquivalentTo(new[] { node.Id }));
        doc.Undo();
        Assert.That(doc.Graph.FindNode(node.Id), Is.Null, "Add and connect undo together.");
    }

    [Test]
    public void Clicking_a_slot_without_dragging_does_nothing()
    {
        var (view, doc, ids) = Setup();
        bool searched = false;
        view.SearchRequested += (_, _) => searched = true;
        var p = Screen(view, OutputOf(view, ids["latent"], 0));
        view.PointerDown(p, PointerButton.Left);
        view.PointerUp(p, PointerButton.Left);
        Assert.That(searched, Is.False);
        Assert.That(doc.CanUndo, Is.False);
    }

    [Test]
    public void Double_click_on_the_background_requests_the_node_search()
    {
        var (view, _, _) = Setup();
        Point? at = null;
        view.SearchRequested += (p, _) => at = p;
        view.PointerDown(new Point(-300, -300), PointerButton.Left, Modifiers.None, clickCount: 2); // empty canvas
        Assert.That(at, Is.EqualTo(new Point(-300, -300)));
    }

    // ---- widgets, resize, title ----

    [Test]
    public void Clicking_a_widget_asks_the_ui_to_edit_it_but_dragging_from_it_does_not()
    {
        var (view, doc, ids) = Setup();
        var notes = new List<(int, string)>();
        view.WidgetActivated += (n, w) => notes.Add((n, w));
        var steps = view.Hits.Geometry(doc.Graph.GetNode(ids["sampler"])).Widgets.First(w => w.Name == "steps").Bounds.Center;

        view.PointerDown(Screen(view, steps), PointerButton.Left);
        view.PointerUp(Screen(view, steps), PointerButton.Left);
        Assert.That(notes, Is.EqualTo(new[] { (ids["sampler"], "steps") }));

        Drag(view, steps, steps + new Point(80, 0));
        Assert.That(notes, Has.Count.EqualTo(1));
    }

    [Test]
    public void Dragging_the_corner_resizes_the_node_width_as_one_undo_step()
    {
        var (view, doc, ids) = Setup();
        var node = doc.Graph.GetNode(ids["sampler"]);
        var before = view.Hits.Geometry(node).Bounds.Width;
        var handle = view.Hits.Geometry(node).ResizeHandle.Center;

        Drag(view, handle, handle + new Point(100, 0));
        Assert.That(view.Hits.Geometry(node).Bounds.Width, Is.EqualTo(before + 100).Within(1));
        doc.Undo();
        Assert.That(node.Width, Is.Null);
        Assert.That(doc.CanUndo, Is.False);
    }

    [Test]
    public void Double_clicking_a_title_requests_a_rename()
    {
        var (view, doc, ids) = Setup();
        int? renamed = null;
        view.TitleEditRequested += id => renamed = id;
        view.PointerDown(Screen(view, TitleOf(view, doc.Graph.GetNode(ids["sampler"]))), PointerButton.Left, Modifiers.None, 2);
        Assert.That(renamed, Is.EqualTo(ids["sampler"]));
    }

    // ---- keyboard ----

    [Test]
    public void Delete_removes_the_selected_nodes_and_ctrl_z_brings_them_back()
    {
        var (view, doc, ids) = Setup();
        view.SelectOnly(ids["pos"]);
        view.ToggleSelect(ids["neg"]);
        int nodes = doc.Graph.Nodes.Count, links = doc.Graph.Links.Count;

        Assert.That(view.Key("delete"), Is.True);
        Assert.That(doc.Graph.Nodes.Count, Is.EqualTo(nodes - 2));
        Assert.That(view.SelectedNodes, Is.Empty);

        view.Key("z", Modifiers.Ctrl);
        Assert.That((doc.Graph.Nodes.Count, doc.Graph.Links.Count), Is.EqualTo((nodes, links)));
        view.Key("y", Modifiers.Ctrl);
        Assert.That(doc.Graph.Nodes.Count, Is.EqualTo(nodes - 2));
    }

    [Test]
    public void Delete_removes_a_selected_link_and_leaves_the_nodes()
    {
        var (view, doc, ids) = Setup();
        var link = doc.Graph.LinkInto(ids["decode"], "samples")!;
        var mid = LinkGeometry.Midpoint(OutputOf(view, link.FromNode, link.FromOutput), InputOf(view, link.ToNode, link.ToInput));
        view.PointerDown(Screen(view, mid), PointerButton.Left);
        view.PointerUp(Screen(view, mid), PointerButton.Left);
        Assert.That(view.SelectedLink, Is.EqualTo(link.Id));

        int nodes = doc.Graph.Nodes.Count;
        view.Key("delete");
        Assert.That(doc.Graph.FindLink(link.Id), Is.Null);
        Assert.That(doc.Graph.Nodes.Count, Is.EqualTo(nodes));
    }

    [Test]
    public void Copy_and_paste_duplicates_nodes_with_their_internal_links_and_widget_values()
    {
        var (view, doc, ids) = Setup();
        doc.SetWidget(ids["sampler"], "steps", 4L);
        view.SelectOnly(ids["sampler"]);
        view.ToggleSelect(ids["decode"]); // sampler -> decode is an internal link of the selection
        view.Key("c", Modifiers.Ctrl);
        int nodes = doc.Graph.Nodes.Count;

        view.Paste(new Point(2000, 100));

        Assert.That(doc.Graph.Nodes.Count, Is.EqualTo(nodes + 2));
        var pasted = view.SelectedNodes.Select(doc.Graph.GetNode).ToList();
        var sampler = pasted.Single(n => n.ClassType == "KSampler");
        var decode = pasted.Single(n => n.ClassType == "VAEDecode");
        Assert.That(sampler.Widgets["steps"], Is.EqualTo(4L));
        Assert.That(sampler.X, Is.EqualTo(2000).Within(0.001), "The top-left of the pasted group lands at the paste point.");
        Assert.That(doc.Graph.LinkInto(decode.Id, "samples")?.FromNode, Is.EqualTo(sampler.Id));
        Assert.That(doc.Graph.LinkInto(sampler.Id, "model"), Is.Null, "External links are not copied by default.");

        doc.Undo();
        Assert.That(doc.Graph.Nodes.Count, Is.EqualTo(nodes), "A paste is one undo step.");
    }

    [Test]
    public void Shift_paste_keeps_the_links_into_the_copied_nodes_from_the_originals()
    {
        var (view, doc, ids) = Setup();
        view.SelectOnly(ids["sampler"]);
        view.Key("c", Modifiers.Ctrl);
        view.Paste(new Point(2000, 100), keepExternalInputs: true);
        var copy = view.SelectedNodes.Select(doc.Graph.GetNode).Single();
        Assert.That(doc.Graph.LinkInto(copy.Id, "model")?.FromNode, Is.EqualTo(ids["ckpt"]));
        Assert.That(doc.Graph.LinkInto(copy.Id, "positive")?.FromNode, Is.EqualTo(ids["pos"]));
    }

    [Test]
    public void Ctrl_m_mutes_and_ctrl_b_bypasses_the_selection_and_pressing_again_restores_it()
    {
        var (view, doc, ids) = Setup();
        view.SelectOnly(ids["sampler"]);
        view.Key("m", Modifiers.Ctrl);
        Assert.That(doc.Graph.GetNode(ids["sampler"]).Mode, Is.EqualTo(NodeMode.Never));
        view.Key("m", Modifiers.Ctrl);
        Assert.That(doc.Graph.GetNode(ids["sampler"]).Mode, Is.EqualTo(NodeMode.Always));
        view.Key("b", Modifiers.Ctrl);
        Assert.That(doc.Graph.GetNode(ids["sampler"]).Mode, Is.EqualTo(NodeMode.Bypass));
    }

    [Test]
    public void Escape_cancels_a_link_in_progress_before_clearing_the_selection()
    {
        var (view, doc, ids) = Setup();
        view.SelectOnly(ids["pos"]);
        view.PointerDown(Screen(view, OutputOf(view, ids["latent"], 0)), PointerButton.Left);
        view.PointerMove(Screen(view, new Point(900, 900)));
        view.Key("escape");
        Assert.That(view.Draft, Is.Null);
        Assert.That(view.SelectedNodes, Has.Count.EqualTo(1));
        view.PointerUp(Screen(view, new Point(900, 900)), PointerButton.Left); // the release after Escape must not create anything
        Assert.That(doc.CanUndo, Is.False);
        view.Key("escape");
        Assert.That(view.SelectedNodes, Is.Empty);
    }

    [Test]
    public void Select_all_selects_every_node()
    {
        var (view, doc, _) = Setup();
        view.Key("a", Modifiers.Ctrl);
        Assert.That(view.SelectedNodes, Has.Count.EqualTo(doc.Graph.Nodes.Count));
    }

    [Test]
    public void Unregistered_node_types_still_load_as_movable_placeholders()
    {
        var (view, doc, _) = Setup();
        // A workflow from a node pack that is not installed: the type is unknown to the registry.
        var geometry = view.Hits.Geometry(new EditorNode { Id = 99, ClassType = "SomeMissingNode", X = 5, Y = 5 });
        Assert.That(geometry.Title, Does.Contain("missing"));
        Assert.That(geometry.Bounds.Width, Is.GreaterThan(0));
    }
}
