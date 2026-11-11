using System.Text.Json.Nodes;
using SpawnGen.Editor;

namespace SpawnGen.Editor.Tests;

/// <summary>App Mode's setup in ComfyUI's workflow format (extra.linearData / extra.linearMode) and the App Builder's picking on the canvas.</summary>
public class AppModeTests
{
    [Test]
    public void ComfyUIs_linear_data_is_read_with_string_and_number_ids()
    {
        var extra = JsonNode.Parse("""{"ds": {"scale": 1}, "linearMode": true, "linearData": {"inputs": [["6", "text"], [3, "seed"]], "outputs": ["9", 12]}}""")!.AsObject();
        var app = AppModeConfig.Read(extra);
        Assert.That(app.Configured, Is.True);
        Assert.That(app.OpensAsApp, Is.True);
        Assert.That(app.Inputs, Is.EqualTo(new[] { new AppInput("6", "text"), new AppInput("3", "seed") }));
        Assert.That(app.Outputs, Is.EqualTo(new[] { "9", "12" }));

        var none = AppModeConfig.Read(JsonNode.Parse("""{"ds": {}}""")!.AsObject());
        Assert.That((none.Configured, none.OpensAsApp, none.Inputs.Count), Is.EqualTo((false, (bool?)null, 0)));
        Assert.That(AppModeConfig.Read(null).Configured, Is.False);
    }

    [Test]
    public void The_setup_is_saved_in_the_workflow_and_read_back_with_the_rest_of_extra_kept()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        doc.Extra = JsonNode.Parse("""{"ds": {"scale": 2}, "linearData": {"inputs": [], "outputs": [], "someday": 1}}""")!.AsObject();
        var app = AppModeConfig.Read(doc.Extra);
        app.ToggleInput(ids["pos"], "text");
        app.ToggleInput(ids["sampler"], "seed");
        app.ToggleOutput(ids["save"]);
        app.OpensAsApp = true;
        app.Write(doc);

        var json = ComfyWorkflow.Save(doc);
        var root = JsonNode.Parse(json)!;
        Assert.That(root["extra"]!["linearMode"]!.GetValue<bool>(), Is.True);
        Assert.That(root["extra"]!["linearData"]!["inputs"]!.ToJsonString(), Is.EqualTo($"[[\"{ids["pos"]}\",\"text\"],[\"{ids["sampler"]}\",\"seed\"]]"));
        Assert.That(root["extra"]!["linearData"]!["outputs"]!.ToJsonString(), Is.EqualTo($"[\"{ids["save"]}\"]"));
        Assert.That(root["extra"]!["linearData"]!["someday"]!.GetValue<int>(), Is.EqualTo(1), "what linearData also holds is kept");
        Assert.That(root["extra"]!["ds"]!["scale"]!.GetValue<int>(), Is.EqualTo(2));

        var reopened = new EditorDocument(doc.Registry);
        ComfyWorkflow.Load(json, reopened);
        var back = AppModeConfig.Read(reopened.Extra);
        Assert.That(back.Inputs, Is.EqualTo(app.Inputs));
        Assert.That(back.Outputs, Is.EqualTo(app.Outputs));
        Assert.That(back.OpensAsApp, Is.True);

        back.OpensAsApp = null;
        back.Write(reopened);
        Assert.That(reopened.Extra!.ContainsKey("linearMode"), Is.False, "no default view chosen: none written");
    }

    [Test]
    public void Picks_toggle_and_reorder()
    {
        var app = new AppModeConfig();
        Assert.That(app.ToggleInput(1, "a"), Is.True);
        Assert.That(app.ToggleInput(2, "b"), Is.True);
        Assert.That(app.ToggleInput(3, "c"), Is.True);
        Assert.That(app.ToggleInput(2, "b"), Is.False, "a second click takes it out");
        Assert.That(app.Inputs.Select(i => i.NodeId), Is.EqualTo(new[] { "1", "3" }));
        AppModeConfig.Move(app.Inputs, 1, -1);
        Assert.That(app.Inputs.Select(i => i.NodeId), Is.EqualTo(new[] { "3", "1" }));
        AppModeConfig.Move(app.Inputs, 0, -5);
        Assert.That(app.Inputs.Select(i => i.NodeId), Is.EqualTo(new[] { "3", "1" }), "clamped");
        Assert.That(app.HasInput(1, "a") && !app.HasInput(1, "b"), Is.True);
        Assert.That(app.ToggleOutput(9) && app.HasOutput(9) && !app.ToggleOutput(9) && !app.HasOutput(9), Is.True);
    }

    [Test]
    public void While_picking_a_click_picks_the_widget_or_node_and_edits_nothing()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        var view = new EditorView(doc);
        var sampler = doc.Graph.GetNode(ids["sampler"]);
        var stepsBefore = sampler.Widgets["steps"];
        var row = view.Hits.Geometry(sampler).Widgets.First(w => w.Name == "steps").Bounds;
        var at = view.Viewport.WorldToScreen(row.Center);
        var picked = new List<(int, string)>();
        var nodes = new List<int>();
        bool activated = false;
        view.WidgetPicked += (n, w) => picked.Add((n, w));
        view.NodePicked += nodes.Add;
        view.WidgetActivated += (_, _) => activated = true;

        view.Picking = PickTarget.Widgets;
        view.PointerDown(at, PointerButton.Left);
        view.PointerMove(new Point(at.X + 40, at.Y)); // a drag would scrub the number when not picking
        view.PointerUp(new Point(at.X + 40, at.Y), PointerButton.Left);
        Assert.That(picked, Is.EqualTo(new[] { (ids["sampler"], "steps") }));
        Assert.That(sampler.Widgets["steps"], Is.EqualTo(stepsBefore));
        Assert.That(activated, Is.False);

        var save = doc.Graph.GetNode(ids["save"]);
        var title = view.Hits.Geometry(save).TitleBar;
        var (x, y) = (save.X, save.Y);
        view.Picking = PickTarget.Nodes;
        var titleAt = view.Viewport.WorldToScreen(new Point(title.X + 30, title.Y + 10));
        view.PointerDown(titleAt, PointerButton.Left);
        view.PointerMove(new Point(titleAt.X + 50, titleAt.Y + 50));
        view.PointerUp(new Point(titleAt.X + 50, titleAt.Y + 50), PointerButton.Left);
        Assert.That(nodes, Is.EqualTo(new[] { ids["save"] }));
        Assert.That((save.X, save.Y), Is.EqualTo((x, y)), "picking does not move the node");
        Assert.That(view.SelectedNodes, Is.Empty);

        view.Picking = PickTarget.None;
        view.PointerDown(at, PointerButton.Left);
        view.PointerUp(at, PointerButton.Left);
        Assert.That(activated, Is.True, "back to editing");
    }

    [Test]
    public void Ids_resolve_in_the_workflow_and_inside_its_subgraphs()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        Assert.That(AppModeConfig.Resolve(doc.Graph, ids["sampler"].ToString())?.ClassType, Is.EqualTo("KSampler"));
        Assert.That(AppModeConfig.Resolve(doc.Graph, "9999"), Is.Null);
        Assert.That(AppModeConfig.Resolve(doc.Graph, "x"), Is.Null);
        var instance = doc.ConvertToSubgraph([ids["pos"]], "Prompt", out var error);
        Assert.That(instance, Is.Not.Null, error);
        var def = SubgraphLibrary.Get(instance!.ClassType)!;
        var inner = def.Nodes.Single(n => n.ClassType == "CLIPTextEncode");
        Assert.That(AppModeConfig.Resolve(doc.Graph, inner.Id.ToString()), Is.SameAs(inner));
    }

    [Test]
    public void Executed_events_match_plain_and_subgraph_node_ids()
    {
        Assert.That(AppModeConfig.IsNode("12", "12"), Is.True);
        Assert.That(AppModeConfig.IsNode("5:12", "12"), Is.True);
        Assert.That(AppModeConfig.IsNode("112", "12"), Is.False);
        Assert.That(AppModeConfig.IsNode(null, "12"), Is.False);
        var (doc, ids) = Fixtures.Txt2Img();
        Assert.That(AppModeConfig.IsOutputNode(doc.Registry, doc.Graph.GetNode(ids["save"])), Is.True);
        Assert.That(AppModeConfig.IsOutputNode(doc.Registry, doc.Graph.GetNode(ids["sampler"])), Is.False);
    }
}
