using SpawnGen.Engine.Graph;

namespace SpawnGen.Engine.Tests;

public class PromptJsonTests
{
    // Shape of a real ComfyUI "Save (API Format)" export.
    const string ComfyApi = """
    {
      "3": { "class_type": "KSampler", "inputs": { "seed": 156680208700286, "steps": 20, "cfg": 8, "sampler_name": "euler",
             "denoise": 1.0, "model": ["4", 0], "positive": ["6", 0], "latent_image": ["5", 0] }, "_meta": { "title": "KSampler" } },
      "4": { "class_type": "CheckpointLoaderSimple", "inputs": { "ckpt_name": "v1-5.safetensors" } },
      "5": { "class_type": "EmptyLatentImage", "inputs": { "width": 512, "height": 512, "batch_size": 1 } },
      "6": { "class_type": "CLIPTextEncode", "inputs": { "text": "a cat", "clip": ["4", 1] } }
    }
    """;

    [Test]
    public void Parses_comfy_api_format()
    {
        var g = PromptJson.Parse(ComfyApi);
        Assert.That(g.Nodes.Keys, Is.EqualTo(new[] { "3", "4", "5", "6" }));
        var k = g.Nodes["3"];
        Assert.That(k.ClassType, Is.EqualTo("KSampler"));
        Assert.That(k.Title, Is.EqualTo("KSampler"));
        Assert.That(k.Inputs["seed"].Literal, Is.EqualTo(156680208700286L));
        Assert.That(k.Inputs["denoise"].Literal, Is.EqualTo(1.0));
        Assert.That(k.Inputs["sampler_name"].Literal, Is.EqualTo("euler"));
        Assert.That(k.Inputs["model"], Is.EqualTo(PromptInput.Link("4", 0)));
        Assert.That(g.Nodes["6"].Inputs["clip"], Is.EqualTo(PromptInput.Link("4", 1)));
    }

    [Test]
    public void Round_trips()
    {
        var g = PromptJson.Parse(ComfyApi);
        var again = PromptJson.Parse(PromptJson.ToJson(g));
        Assert.That(PromptJson.ToJson(again), Is.EqualTo(PromptJson.ToJson(g)));
        Assert.That(again.Nodes["3"].Inputs["seed"].Literal, Is.EqualTo(156680208700286L));
    }

    [Test]
    public void Accepts_the_prompt_wrapper_used_by_the_comfy_http_api()
    {
        var g = PromptJson.Parse("{\"prompt\":" + ComfyApi + ",\"client_id\":\"x\"}");
        Assert.That(g.Nodes.Count, Is.EqualTo(4));
    }

    [Test]
    public void Numeric_node_ids_in_links_are_accepted()
    {
        var g = PromptJson.Parse("""{ "1": { "class_type": "A", "inputs": {} }, "2": { "class_type": "B", "inputs": { "x": [1, 0] } } }""");
        Assert.That(g.Nodes["2"].Inputs["x"], Is.EqualTo(PromptInput.Link("1", 0)));
    }

    [Test]
    public void Malformed_input_is_a_format_error()
    {
        Assert.Throws<FormatException>(() => PromptJson.Parse("""{ "1": { "inputs": {} } }"""));
        Assert.Throws<FormatException>(() => PromptJson.Parse("""{ "1": { "class_type": "A", "inputs": { "x": {"a":1} } } }"""));
        Assert.Throws<FormatException>(() => PromptJson.Parse("[]"));
    }
}
