using System.Net.Http.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Nodes.Core;
using SpawnGen.Server;

namespace SpawnGen.Server.Tests;

/// <summary>Custom node packs: assemblies in the custom nodes folder are loaded at start, show up in the catalogue and run; the catalogue reads back into schemas.</summary>
public class CustomNodesTests
{
    WebApplication _app = null!;
    HttpClient _http = null!;
    string _dir = null!;

    [SetUp]
    public async Task SetUp()
    {
        _dir = Path.Combine(Path.GetTempPath(), "spawngen-custom-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(_dir, "custom_nodes"));
        // The sample pack was built next to the tests (a project reference); a copy of its file is the "installed" pack.
        File.Copy(Path.Combine(AppContext.BaseDirectory, "SpawnGen.SampleNodes.dll"), Path.Combine(_dir, "custom_nodes", "SpawnGen.SampleNodes.dll"));
        File.WriteAllText(Path.Combine(_dir, "custom_nodes", "broken.dll"), "this is not an assembly");
        _app = ServerApp.Build(
            ["--urls", "http://127.0.0.1:0", $"--SpawnGen:ModelsPath={_dir}/models", $"--SpawnGen:OutputPath={_dir}/output", $"--SpawnGen:InputPath={_dir}/input",
             $"--SpawnGen:TempPath={_dir}/temp", $"--SpawnGen:CustomNodesPath={_dir}/custom_nodes", "--environment", "Development"]);
        await _app.StartAsync();
        var address = _app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
        _http = new HttpClient { BaseAddress = new Uri(address) };
    }

    [TearDown]
    public async Task TearDown()
    {
        _http.Dispose();
        await _app.StopAsync();
        await _app.DisposeAsync();
        try { Directory.Delete(_dir, true); } catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { } // the loaded pack keeps its file locked
    }

    [Test]
    public async Task The_pack_is_loaded_listed_in_the_catalogue_and_a_broken_file_is_reported_without_stopping_the_server()
    {
        var info = JsonNode.Parse(await _http.GetStringAsync("/object_info"))!.AsObject();
        Assert.That(info.ContainsKey("SampleReverseText"), Is.True);
        Assert.That((string?)info["SampleReverseText"]!["category"], Is.EqualTo("sample"));
        Assert.That((string?)info["SampleTintImage"]!["display_name"], Is.EqualTo("Tint Image"));
        Assert.That(info.ContainsKey("KSampler"), Is.True, "the built-in nodes are still there");

        var stats = JsonNode.Parse(await _http.GetStringAsync("/system_stats"))!;
        var packs = stats["custom_nodes"]!.AsArray();
        var ok = packs.First(p => (string?)p!["file"] == "SpawnGen.SampleNodes.dll")!;
        Assert.That(ok["nodes"]!.AsArray().Select(n => (string?)n), Is.EquivalentTo(new[] { "SampleReverseText", "SampleTintImage" }));
        var broken = packs.First(p => (string?)p!["file"] == "broken.dll")!;
        Assert.That((string?)broken["error"], Is.Not.Null.And.Not.Empty);
    }

    [Test]
    public async Task A_custom_node_runs_in_a_prompt()
    {
        var prompt = new JsonObject
        {
            ["1"] = new JsonObject { ["class_type"] = "PrimitiveString", ["inputs"] = new JsonObject { ["value"] = "SpawnGen" } },
            ["2"] = new JsonObject { ["class_type"] = "SampleReverseText", ["inputs"] = new JsonObject { ["text"] = new JsonArray("1", 0) } },
            ["3"] = new JsonObject { ["class_type"] = "PreviewAny", ["inputs"] = new JsonObject { ["source"] = new JsonArray("2", 0) } },
        };
        var response = await _http.PostAsJsonAsync("/prompt", new JsonObject { ["prompt"] = prompt, ["client_id"] = "t" });
        var id = (string)(await response.Content.ReadFromJsonAsync<JsonObject>())!["prompt_id"]!;
        JsonObject? entry = null;
        for (int i = 0; i < 100 && entry is null; i++)
        {
            await Task.Delay(100);
            entry = JsonNode.Parse(await _http.GetStringAsync("/history"))!.AsObject()[id] as JsonObject;
        }
        Assert.That(entry, Is.Not.Null);
        Assert.That((string?)entry!["status"]!["status_str"], Is.EqualTo("success"));
        Assert.That((string?)entry["outputs"]!["3"]!["text"]![0], Is.EqualTo(Reverse("SpawnGen")));
    }

    static string Reverse(string s) => new(s.Reverse().ToArray());

    [Test]
    public async Task The_catalogue_reads_back_into_schemas_with_widgets_combos_and_outputs()
    {
        var info = JsonNode.Parse(await _http.GetStringAsync("/object_info"))!.AsObject();
        var schemas = NodeSchemaJson.FromObjectInfo(info).ToDictionary(s => s.Id);
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(SpawnGen.Nodes.ML.KSampler).Assembly);

        var ks = schemas["KSampler"];
        var original = registry.GetSchema("KSampler")!;
        Assert.That(ks.Inputs.Select(i => i.Name), Is.EqualTo(original.Inputs.Select(i => i.Name)));
        Assert.That(ks.Inputs.Select(i => i.Type.Name), Is.EqualTo(original.Inputs.Select(i => i.Type.Name)));
        Assert.That(ks.Outputs.Select(o => (o.Name, o.Type.Name)), Is.EqualTo(original.Outputs.Select(o => (o.Name, o.Type.Name))));
        var steps = ks.FindInput("steps")!.Widget!;
        Assert.That((steps.Kind, steps.Min, steps.Max), Is.EqualTo((WidgetKind.Int, 1.0, 10000.0)));
        var sampler = ks.FindInput("sampler_name")!.Widget!;
        Assert.That(sampler.Kind, Is.EqualTo(WidgetKind.Combo));
        Assert.That(sampler.Options, Does.Contain("euler").And.Contain("dpmpp_2m"));
        Assert.That(ks.FindInput("model")!.Widget, Is.Null, "a socket input has no widget");
        Assert.That(schemas["SaveImage"].IsOutputNode, Is.True);
        Assert.That(schemas["SampleTintImage"].FindInput("strength")!.Widget!.Default, Is.EqualTo(0.3).Within(1e-6));
        Assert.That(schemas["SampleReverseText"].FindInput("text")!.Widget!.Multiline, Is.True);
    }

    [Test]
    public async Task Match_type_force_input_and_dev_only_read_back_from_the_catalogue()
    {
        var info = JsonNode.Parse(await _http.GetStringAsync("/object_info"))!.AsObject();
        var start = NodeSchemaJson.FromObjectInfo(info).Single(s => s.Id == "StartLoop");
        // The List mode's socket is a MatchType input that still belongs to its dynamic combo option.
        var list = start.FindInput("mode.list")!;
        Assert.That((list.MatchTemplate, list.DynamicParent, list.Widget), Is.EqualTo(("list_item", (string?)"mode", (WidgetSpec?)null)));
        Assert.That(list.DynamicOptions, Is.EqualTo(new[] { "List" }));
        Assert.That(start.FindInput("initial_iteration_value")!.MatchTemplate, Is.EqualTo("carried_value"));
        var parent = start.FindInput("parent_iteration")!;
        Assert.That((parent.ForceInput, parent.Widget), Is.EqualTo((true, (WidgetSpec?)null)));
        Assert.That(start.Outputs.Select(o => o.MatchTemplate), Is.EqualTo(new string?[] { null, null, null, "list_item", "carried_value" }));
        Assert.That(start.Outputs[4].Type, Is.EqualTo(SocketType.Any));
        // A MatchType input limited to some types (ComfyUI's allowed_types) and the output that follows it.
        var resize = NodeSchemaJson.FromObjectInfo(info).Single(s => s.Id == "ResizeImageMaskNode");
        Assert.That((resize.FindInput("input")!.MatchTemplate, resize.FindInput("input")!.Type.Name), Is.EqualTo(("input_type", "IMAGE,MASK")));
        Assert.That(resize.Outputs[0].MatchTemplate, Is.EqualTo("input_type"));
        var iteration = NodeSchemaJson.FromObjectInfo(info).Single(s => s.Id == "LoopIteration");
        Assert.That(iteration.DevOnly, Is.True);
        Assert.That(NodeSchemaJson.FromObjectInfo(info).Single(s => s.Id == "EndLoop").DevOnly, Is.False);
    }

    [Test]
    public async Task An_Autogrow_group_is_listed_where_it_was_declared_and_only_its_minimum_must_be_linked()
    {
        var info = JsonNode.Parse(await _http.GetStringAsync("/object_info"))!.AsObject();
        // ComfyUI lists String Format's values group (minimum 0) under required, Batch Images' (minimum 1) too, End Loop's terminations under optional.
        Assert.That(info["StringFormat"]!["input"]!["required"]!.AsObject().ContainsKey("values.a"), Is.True);
        Assert.That(info["BatchImagesNode"]!["input"]!["required"]!.AsObject().ContainsKey("images.image1"), Is.True);
        Assert.That(info["EndLoop"]!["input"]!["optional"]!.AsObject().ContainsKey("terminations.termination0"), Is.True);
        var schemas = NodeSchemaJson.FromObjectInfo(info).ToDictionary(s => s.Id);
        Assert.That(schemas["StringFormat"].FindInput("values.a")!.Required, Is.False);
        Assert.That(schemas["BatchImagesNode"].FindInput("images.image0")!.Required, Is.True);
        Assert.That(schemas["BatchImagesNode"].FindInput("images.image1")!.Required, Is.False);
        Assert.That(schemas["EndLoop"].FindInput("terminations.termination0")!.Required, Is.False);
    }

    [Test]
    public async Task A_text_socket_without_a_widget_is_listed_as_forceInput_and_reads_back_as_a_socket()
    {
        var info = JsonNode.Parse(await _http.GetStringAsync("/object_info"))!.AsObject();
        // ComfyUI's frontend draws a widget for a STRING input unless it says forceInput.
        var texts = info["ShuffleImageTextDataset"]!["input"]!["required"]!["texts"]!;
        Assert.That(((string?)texts[0], (bool?)texts[1]!["forceInput"]), Is.EqualTo(("STRING", (bool?)true)));
        Assert.That((bool?)info["SaveImageTextDataSetToFolder"]!["input"]!["optional"]!["texts"]![1]!["forceInput"], Is.True);
        Assert.That(info["TruncateText"]!["input"]!["required"]!["texts"]![1]!["forceInput"], Is.Null, "a text widget");

        var schemas = NodeSchemaJson.FromObjectInfo(info).ToDictionary(s => s.Id);
        Assert.That(schemas["ShuffleImageTextDataset"].FindInput("texts")!.Widget, Is.Null);
        Assert.That(schemas["TruncateText"].FindInput("texts")!.Widget!.Kind, Is.EqualTo(WidgetKind.String));
    }

    [Test]
    public void A_remote_node_can_be_placed_and_validated_but_is_not_run_here()
    {
        var schema = NodeSchemaJson.FromObjectInfo(JsonNode.Parse("""
            { "FarAway": { "input": { "required": { "x": ["INT", { "default": 3, "min": 0, "max": 9 }] } }, "input_order": { "required": ["x"] },
              "output": ["INT"], "output_name": ["out"], "display_name": "Far Away", "category": "remote", "output_node": true } }
            """)!.AsObject()).Single();
        var registry = new NodeRegistry().AddRemote(schema);
        Assert.That(registry.GetSchema("FarAway")!.DisplayName, Is.EqualTo("Far Away"));

        var g = new PromptGraph();
        g.Add("1", "FarAway", ("x", 4));
        var executor = new PromptExecutor(registry);
        Assert.That(executor.Validate(g).Errors, Is.Empty);
        var ex = Assert.ThrowsAsync<NodeExecutionException>(() => executor.ExecuteAsync(g));
        Assert.That(ex!.InnerException!.Message, Does.Contain("custom node"));

        // A node that is really here is never replaced by a remote description of the same id.
        var real = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly);
        real.AddRemote(new NodeSchema { Id = "SaveImage", DisplayName = "Imposter" });
        Assert.That(real.GetSchema("SaveImage")!.DisplayName, Is.Not.EqualTo("Imposter"));
    }
}
