using System.Text.Json.Nodes;
using SpawnGen.Editor;

namespace SpawnGen.Editor.Tests;

/// <summary>
/// ComfyUI's "choose file to upload" button on file combos (Load Image, Load Image (as Mask), Load Audio, Load Video, Load Image (from Outputs)).
/// Order and saved values were measured on ComfyUI frontend 1.61.3 (cloud.comfy.org, 2026-11-18).
/// </summary>
public class UploadWidgetTests
{
    static EditorNode Add(EditorDocument doc, string type) => doc.Graph.GetNode(doc.AddNode(type, 0, 0).Id);

    [TestCase("LoadImage", "image", "image")]
    [TestCase("LoadImageMask", "image", "image")]
    [TestCase("LoadImageOutput", "image", "image")]
    [TestCase("LoadAudio", "audio", "audio")]
    [TestCase("LoadVideo", "file", "video")]
    public void A_file_combo_gets_the_upload_button_as_its_last_row(string type, string combo, string kind)
    {
        var doc = new EditorDocument(Fixtures.Registry());
        var node = Add(doc, type);
        var schema = doc.Registry.GetSchema(type)!;
        Assert.That(UploadWidget.ComboFor(schema)?.Name, Is.EqualTo(combo));
        Assert.That(UploadWidget.ComboFor(schema)!.Widget!.Upload, Is.EqualTo(kind));

        var geo = NodeLayout.Compute(node, schema, doc.Graph);
        var last = geo.Widgets[^1];
        Assert.That(last.IsUpload, Is.True, string.Join(", ", geo.Widgets.Select(w => w.Name)));
        Assert.That(last.UploadFor, Is.EqualTo(combo));
        Assert.That(last.Label, Is.EqualTo("choose file to upload"));
        Assert.That(geo.Widgets.Count(w => w.IsUpload), Is.EqualTo(1));
        Assert.That(geo.Inputs.Any(i => i.Name == UploadWidget.Name), Is.False, "the button has no socket");
        Assert.That(last.Bounds.Y, Is.GreaterThanOrEqualTo(geo.Widgets[^2].Bounds.Bottom), "under the other rows");
        Assert.That(geo.Bounds.Bottom, Is.GreaterThan(last.Bounds.Bottom));
    }

    [Test]
    public void Load_Image_as_Mask_puts_the_button_after_channel_as_ComfyUI_does()
    {
        var doc = new EditorDocument(Fixtures.Registry());
        var node = Add(doc, "LoadImageMask");
        var names = NodeLayout.Compute(node, doc.Registry.GetSchema("LoadImageMask")!, doc.Graph).Widgets.Select(w => w.Name);
        Assert.That(names, Is.EqualTo(new[] { "image", "channel", UploadWidget.Name }));
    }

    [Test]
    public void Nodes_without_an_upload_combo_have_no_button()
    {
        var doc = new EditorDocument(Fixtures.Registry());
        foreach (var type in new[] { "CheckpointLoaderSimple", "KSampler", "SaveImage", "LoadImageDataSetFromFolder" })
        {
            var node = Add(doc, type);
            Assert.That(NodeLayout.Compute(node, doc.Registry.GetSchema(type)!, doc.Graph).Widgets.Any(w => w.IsUpload), Is.False, type);
        }
    }

    /// <summary>The values ComfyUI's frontend 1.61.3 serializes for these nodes (node.serialize().widgets_values).</summary>
    [TestCase("LoadImage", "a.png", "[\"a.png\",\"image\"]")]
    [TestCase("LoadImageMask", "a.png", "[\"a.png\",\"alpha\",\"image\"]")]
    [TestCase("LoadVideo", "a.webm", "[\"a.webm\",\"image\"]")]
    [TestCase("LoadAudio", "a.wav", "[\"a.wav\",\"\"]")]
    // ComfyUI saves ["name", false, "refresh", "image"]: an auto-refresh toggle and a refresh button SpawnGen does not have. Writing the button's
    // value alone would put "image" where ComfyUI reads the toggle, so nothing extra is written.
    [TestCase("LoadImageOutput", "a.png [output]", "[\"a.png [output]\"]")]
    public void Saved_widget_values_end_with_the_upload_buttons_value_as_ComfyUI_saves_them(string type, string file, string expected)
    {
        var doc = new EditorDocument(Fixtures.Registry());
        var node = Add(doc, type);
        var combo = UploadWidget.ComboFor(doc.Registry.GetSchema(type)!)!.Name;
        doc.SetWidget(node.Id, combo, file);
        var saved = JsonNode.Parse(ComfyWorkflow.Save(doc))!["nodes"]!.AsArray().Single()!["widgets_values"]!;
        Assert.That(saved.ToJsonString(), Is.EqualTo(expected));
    }

    [Test]
    public void ComfyUIs_saved_values_with_the_buttons_value_read_back_into_the_right_widgets()
    {
        // As ComfyUI 1.61.3 saves Load Image (as Mask) with channel "red": the button's "image" comes after channel.
        var json = """
        {"last_node_id":1,"last_link_id":0,"nodes":[{"id":1,"type":"LoadImageMask","pos":[0,0],"size":[300,300],"flags":{},"order":0,"mode":0,
          "inputs":[],"outputs":[{"name":"MASK","type":"MASK","links":null}],"properties":{},"widgets_values":["photo.png","red","image"]}],
         "links":[],"groups":[],"config":{},"extra":{},"version":0.4}
        """;
        var doc = new EditorDocument(Fixtures.Registry());
        var warnings = ComfyWorkflow.Load(json, doc);
        var node = doc.Graph.Nodes.Single();
        Assert.That(warnings, Is.Empty);
        Assert.That(node.Widgets["image"], Is.EqualTo("photo.png"));
        Assert.That(node.Widgets["channel"], Is.EqualTo("red"));
        Assert.That(node.Widgets.ContainsKey(UploadWidget.Name), Is.False, "the button holds no value");
        Assert.That(JsonNode.Parse(ComfyWorkflow.Save(doc))!["nodes"]![0]!["widgets_values"]!.ToJsonString(), Is.EqualTo("[\"photo.png\",\"red\",\"image\"]"), "saved back the same");
    }

    [Test]
    public void The_button_is_not_sent_with_the_prompt()
    {
        var doc = new EditorDocument(Fixtures.Registry());
        var node = Add(doc, "LoadImageMask");
        doc.SetWidget(node.Id, "image", "a.png");
        var inputs = GraphCompiler.Compile(doc.Graph, doc.Registry).Graph.Nodes[node.Id.ToString()].Inputs.Keys;
        Assert.That(inputs, Is.EquivalentTo(new[] { "image", "channel" }));
    }

    [Test]
    public void An_upload_into_another_folder_is_named_with_that_folders_annotation()
    {
        var doc = new EditorDocument(Fixtures.Registry());
        var output = UploadWidget.ComboFor(doc.Registry.GetSchema("LoadImageOutput")!)!.Widget!;
        var input = UploadWidget.ComboFor(doc.Registry.GetSchema("LoadImage")!)!.Widget!;
        Assert.That(output.UploadFolder, Is.EqualTo("output"));
        Assert.That(UploadWidget.ValueFor(output, "a.png"), Is.EqualTo("a.png [output]"), "as Load Image (from Outputs) lists its files");
        Assert.That(UploadWidget.ValueFor(input, "a.png"), Is.EqualTo("a.png"));
    }
}
