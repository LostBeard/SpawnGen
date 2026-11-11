using SpawnGen.Editor;

namespace SpawnGen.Editor.Tests;

/// <summary>Context-menu actions: Clone, Add Reroute on a link, Disconnect on an output slot. Each is one undo step.</summary>
public class MenuActionsTests
{
    [Test]
    public void Add_reroute_puts_a_reroute_into_the_link_and_undo_takes_it_out()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        var link = doc.Graph.LinkInto(ids["decode"], "samples")!;
        int nodes = doc.Graph.Nodes.Count, links = doc.Graph.Links.Count;
        var reroute = doc.InsertReroute(link.Id, 1100, 50)!;
        doc.EndGesture();
        Assert.That(reroute.ClassType, Is.EqualTo("Reroute"));
        Assert.That((reroute.X + NodeLayout.RerouteWidth / 2, reroute.Y + NodeLayout.RerouteHeight / 2), Is.EqualTo((1100.0, 50.0)), "centred on the click");
        Assert.That(doc.Graph.LinkInto(reroute.Id, "value")!.FromNode, Is.EqualTo(ids["sampler"]));
        Assert.That(doc.Graph.LinkInto(ids["decode"], "samples")!.FromNode, Is.EqualTo(reroute.Id));
        Assert.That((doc.Graph.Nodes.Count, doc.Graph.Links.Count), Is.EqualTo((nodes + 1, links + 1)));
        Assert.That(GraphCompiler.Compile(doc.Graph, doc.Registry).Graph.Nodes[ids["decode"].ToString()].Inputs["samples"].NodeId, Is.EqualTo(ids["sampler"].ToString()),
            "the run looks through the reroute");

        doc.Undo();
        Assert.That((doc.Graph.Nodes.Count, doc.Graph.Links.Count), Is.EqualTo((nodes, links)));
        Assert.That(doc.Graph.LinkInto(ids["decode"], "samples")!.FromNode, Is.EqualTo(ids["sampler"]));
        Assert.That(doc.InsertReroute(-5, 0, 0), Is.Null);
    }

    [Test]
    public void Disconnect_on_an_output_takes_out_all_its_links_in_one_step()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        Assert.That(doc.DisconnectOutput(ids["ckpt"], 1), Is.EqualTo(2), "CLIP feeds both prompts");
        doc.EndGesture();
        Assert.That(doc.Graph.LinksFrom(ids["ckpt"], 1), Is.Empty);
        Assert.That(doc.Graph.LinksFrom(ids["ckpt"], 0).Count(), Is.EqualTo(1), "the other outputs keep theirs");
        doc.Undo();
        Assert.That(doc.Graph.LinksFrom(ids["ckpt"], 1).Count(), Is.EqualTo(2));
    }

    [Test]
    public void Clone_copies_the_nodes_with_their_links_among_them_and_leaves_the_clipboard_alone()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        var view = new EditorView(doc);
        doc.SetWidget(ids["pos"], "text", "a red apple");
        doc.EndGesture();
        Assert.That(view.CanPaste, Is.False);
        int nodes = doc.Graph.Nodes.Count;
        view.Duplicate([ids["pos"], ids["sampler"]]);
        Assert.That(doc.Graph.Nodes.Count, Is.EqualTo(nodes + 2));
        var copies = view.SelectedNodes.Select(doc.Graph.GetNode).ToList();
        var pos = copies.Single(n => n.ClassType == "CLIPTextEncode");
        var sampler = copies.Single(n => n.ClassType == "KSampler");
        Assert.That(pos.Widgets["text"], Is.EqualTo("a red apple"));
        Assert.That((pos.X, pos.Y), Is.EqualTo((430.0, 30.0)), "a step down and right of the originals");
        Assert.That(doc.Graph.LinkInto(sampler.Id, "positive")!.FromNode, Is.EqualTo(pos.Id), "the link between them is copied");
        Assert.That(doc.Graph.LinkInto(sampler.Id, "model"), Is.Null, "links from outside are not");
        Assert.That(view.CanPaste, Is.False, "the clipboard is untouched");
        doc.Undo();
        Assert.That(doc.Graph.Nodes.Count, Is.EqualTo(nodes));
    }
}
