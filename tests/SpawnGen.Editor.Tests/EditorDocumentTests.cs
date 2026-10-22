using SpawnGen.Editor;
using SpawnGen.Engine.Nodes;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Editor.Tests;

static class Fixtures
{
    public static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly);

    /// <summary>Loader -> two text encodes -> empty latent -> KSampler -> VAE decode -> save, laid out left to right.</summary>
    public static (EditorDocument Doc, Dictionary<string, int> Ids) Txt2Img()
    {
        var doc = new EditorDocument(Registry());
        var ids = new Dictionary<string, int>();
        int Add(string key, string type, double x, double y) => ids[key] = doc.AddNode(type, x, y).Id;
        Add("ckpt", "CheckpointLoaderSimple", 0, 0);
        Add("pos", "CLIPTextEncode", 400, 0);
        Add("neg", "CLIPTextEncode", 400, 250);
        Add("latent", "EmptyLatentImage", 400, 500);
        Add("sampler", "KSampler", 850, 0);
        Add("decode", "VAEDecode", 1250, 0);
        Add("save", "SaveImage", 1500, 0);
        void C(string from, int o, string to, string input)
        {
            var link = doc.Connect(ids[from], o, ids[to], input, out var err);
            Assert.That(link, Is.Not.Null, err);
        }
        C("ckpt", 1, "pos", "clip"); C("ckpt", 1, "neg", "clip");
        C("ckpt", 0, "sampler", "model"); C("pos", 0, "sampler", "positive"); C("neg", 0, "sampler", "negative");
        C("latent", 0, "sampler", "latent_image"); C("sampler", 0, "decode", "samples"); C("ckpt", 2, "decode", "vae");
        C("decode", 0, "save", "images");
        doc.ClearHistory();
        return (doc, ids);
    }
}

public class EditorDocumentTests
{
    [Test]
    public void A_new_node_starts_with_its_widgets_at_their_defaults()
    {
        var doc = new EditorDocument(Fixtures.Registry());
        var node = doc.AddNode("EmptyLatentImage", 10, 20);
        Assert.That((node.X, node.Y), Is.EqualTo((10.0, 20.0)));
        Assert.That(node.Widgets["width"], Is.EqualTo(512L));
        Assert.That(node.Widgets["height"], Is.EqualTo(512L));
        Assert.That(node.Widgets["batch_size"], Is.EqualTo(1L));
    }

    [Test]
    public void Connecting_checks_socket_types()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        // IMAGE from the decoder cannot feed the sampler's LATENT input.
        var link = doc.Connect(ids["decode"], 0, ids["sampler"], "latent_image", out var error);
        Assert.That(link, Is.Null);
        Assert.That(error, Does.Contain("IMAGE").And.Contain("LATENT"));
    }

    [Test]
    public void Connecting_to_a_taken_input_replaces_the_link_and_undo_brings_it_back()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        var other = doc.AddNode("EmptyLatentImage", 400, 800).Id;
        var before = doc.Graph.LinkInto(ids["sampler"], "latent_image")!;

        doc.Connect(other, 0, ids["sampler"], "latent_image", out _);
        Assert.That(doc.Graph.LinkInto(ids["sampler"], "latent_image")!.FromNode, Is.EqualTo(other));
        Assert.That(doc.Graph.Links.Count(l => l.ToInput == "latent_image"), Is.EqualTo(1), "An input takes one link.");

        doc.Undo();
        Assert.That(doc.Graph.LinkInto(ids["sampler"], "latent_image")!.Id, Is.EqualTo(before.Id));
    }

    [Test]
    public void A_link_that_would_close_a_cycle_is_refused()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        var self = doc.Connect(ids["sampler"], 0, ids["sampler"], "latent_image", out var selfError);
        Assert.That(self, Is.Null);
        Assert.That(selfError, Does.Contain("itself"));

        // decode (IMAGE) -> VAEEncode -> (LATENT) -> sampler.latent_image would make sampler -> decode -> encode -> sampler.
        var encode = doc.AddNode("VAEEncode", 0, 600).Id;
        Assert.That(doc.Connect(ids["decode"], 0, encode, "pixels", out var e1), Is.Not.Null, e1);
        Assert.That(doc.Connect(ids["ckpt"], 2, encode, "vae", out var e2), Is.Not.Null, e2);
        var loop = doc.Connect(encode, 0, ids["sampler"], "latent_image", out var loopError);
        Assert.That(loop, Is.Null);
        Assert.That(loopError, Does.Contain("cycle"));
        Assert.That(doc.Graph.LinkInto(ids["sampler"], "latent_image")!.FromNode, Is.EqualTo(ids["latent"]), "The refused link must not disturb the existing one.");
    }


    [Test]
    public void Deleting_a_node_removes_its_links_and_undo_restores_them_all()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        int nodes = doc.Graph.Nodes.Count, links = doc.Graph.Links.Count;

        doc.DeleteNodes([ids["sampler"]]);
        Assert.That(doc.Graph.Nodes.Count, Is.EqualTo(nodes - 1));
        Assert.That(doc.Graph.Links.Count, Is.EqualTo(links - 5), "Five links touch the sampler: model, positive, negative, latent_image, and its output.");

        doc.Undo();
        Assert.That(doc.Graph.Nodes.Count, Is.EqualTo(nodes));
        Assert.That(doc.Graph.Links.Count, Is.EqualTo(links));
        doc.Redo();
        Assert.That(doc.Graph.FindNode(ids["sampler"]), Is.Null);
    }

    [Test]
    public void A_drag_is_one_undo_step_and_a_new_gesture_starts_a_new_one()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        var node = doc.Graph.GetNode(ids["pos"]);
        double x0 = node.X, y0 = node.Y;

        for (int i = 0; i < 20; i++) doc.MoveNodes([ids["pos"]], 3, 2); // pointer-move events of one drag
        doc.EndGesture();
        doc.MoveNodes([ids["pos"]], 100, 0);                            // a second drag

        Assert.That((node.X, node.Y), Is.EqualTo((x0 + 160, y0 + 40)));
        doc.Undo();
        Assert.That((node.X, node.Y), Is.EqualTo((x0 + 60, y0 + 40)), "Undo takes back the second drag only.");
        doc.Undo();
        Assert.That((node.X, node.Y), Is.EqualTo((x0, y0)), "And the next undo takes back the whole first drag.");
        Assert.That(doc.CanUndo, Is.False);
    }

    [Test]
    public void Scrubbing_a_widget_merges_into_one_undo_step()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        for (long v = 1; v <= 30; v++) doc.SetWidget(ids["sampler"], "steps", v);
        doc.EndGesture();
        doc.SetWidget(ids["sampler"], "steps", 50L);

        doc.Undo();
        Assert.That(doc.Graph.GetNode(ids["sampler"]).Widgets["steps"], Is.EqualTo(30L));
        doc.Undo();
        Assert.That(doc.Graph.GetNode(ids["sampler"]).Widgets["steps"], Is.EqualTo(20L), "The default before the scrub.");
    }

    [Test]
    public void Group_makes_several_edits_one_undo_step()
    {
        var (doc, ids) = Fixtures.Txt2Img();
        doc.Group(() =>
        {
            doc.SetWidget(ids["sampler"], "steps", 4L);
            doc.SetWidget(ids["sampler"], "cfg", 1.0);
            doc.SetTitle(ids["sampler"], "Turbo sampler");
        });
        var n = doc.Graph.GetNode(ids["sampler"]);
        Assert.That((n.Widgets["steps"], n.Widgets["cfg"], n.Title), Is.EqualTo((4L, 1.0, "Turbo sampler")));

        doc.Undo();
        Assert.That((n.Widgets["steps"], n.Widgets["cfg"], n.Title), Is.EqualTo((20L, 8.0, (string?)null)));
        Assert.That(doc.CanUndo, Is.False);
    }

    [Test]
    public void Changed_fires_for_edits_undo_and_redo()
    {
        var doc = new EditorDocument(Fixtures.Registry());
        int fired = 0;
        doc.Changed += () => fired++;
        var id = doc.AddNode("EmptyLatentImage", 0, 0).Id;
        doc.SetWidget(id, "width", 768L);
        doc.Undo();
        doc.Redo();
        Assert.That(fired, Is.EqualTo(4));
    }

    [Test]
    public void New_edits_after_an_undo_discard_the_redo_history()
    {
        var doc = new EditorDocument(Fixtures.Registry());
        doc.AddNode("EmptyLatentImage", 0, 0);
        doc.Undo();
        Assert.That(doc.CanRedo, Is.True);
        doc.AddNode("EmptyLatentImage", 5, 5);
        Assert.That(doc.CanRedo, Is.False);
    }
}
