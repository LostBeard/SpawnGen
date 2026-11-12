using System.Net;
using System.Text;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using SpawnGen.Server;

namespace SpawnGen.Server.Tests;

/// <summary>
/// POST /prompt with broken output nodes and with partial_execution_targets: the answers (codes, node_errors with dependent_outputs) and what
/// runs are what ComfyUI's server gave for the same prompts.
/// </summary>
public class PromptAdmissionTests
{
    string _dir = "";
    WebApplication _app = null!;
    HttpClient _http = null!;

    [SetUp]
    public async Task SetUp()
    {
        _dir = Path.Combine(Path.GetTempPath(), "spawngen-admission-" + Guid.NewGuid().ToString("N"));
        _app = ServerApp.Build(
            ["--urls", "http://127.0.0.1:0", $"--SpawnGen:ModelsPath={_dir}/models", $"--SpawnGen:OutputPath={_dir}/output", $"--SpawnGen:InputPath={_dir}/input",
             $"--SpawnGen:TempPath={_dir}/temp", $"--SpawnGen:UserPath={_dir}/user", $"--SpawnGen:CustomNodesPath={_dir}/custom_nodes", "--environment", "Development"]);
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
        try { Directory.Delete(_dir, true); } catch (IOException) { }
    }

    static JsonObject Text(string value) => new() { ["class_type"] = "PrimitiveString", ["inputs"] = new JsonObject { ["value"] = value } };
    static JsonObject Show(string? from) => new() { ["class_type"] = "PreviewAny", ["inputs"] = from is null ? new JsonObject() : new JsonObject { ["source"] = new JsonArray(from, 0) } };

    /// <summary>2 and 4 show text; 5 has no input.</summary>
    static JsonObject TwoGoodOneBroken() => new() { ["1"] = Text("a"), ["2"] = Show("1"), ["3"] = Text("b"), ["4"] = Show("3"), ["5"] = Show(null) };

    async Task<(HttpStatusCode Code, JsonObject Body)> Post(JsonObject prompt, JsonNode? targets = null)
    {
        var body = new JsonObject { ["prompt"] = prompt, ["client_id"] = "test" };
        if (targets is not null) body["partial_execution_targets"] = targets;
        using var response = await _http.PostAsync("api/prompt", new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json"));
        return (response.StatusCode, (JsonNode.Parse(await response.Content.ReadAsStringAsync()) as JsonObject)!);
    }

    /// <summary>The output nodes the finished run has results for.</summary>
    async Task<string[]> Ran(JsonObject accepted)
    {
        var id = (string)accepted["prompt_id"]!;
        for (int i = 0; i < 200; i++)
        {
            var history = JsonNode.Parse(await _http.GetStringAsync("api/history/" + id)) as JsonObject;
            if (history?[id] is JsonObject entry && entry["status"]?["completed"]?.GetValue<bool>() == true)
                return [.. (entry["outputs"] as JsonObject ?? []).Select(kv => kv.Key).Order()];
            await Task.Delay(50);
        }
        Assert.Fail("the run did not finish");
        return [];
    }

    [Test]
    public async Task A_broken_output_is_reported_and_the_others_run()
    {
        var (code, body) = await Post(TwoGoodOneBroken());
        Assert.That(code, Is.EqualTo(HttpStatusCode.OK));
        var five = body["node_errors"]!["5"]!;
        Assert.That(five["errors"]![0]!["type"]!.GetValue<string>(), Is.EqualTo("required_input_missing"));
        Assert.That(five["errors"]![0]!["extra_info"]!["input_name"]!.GetValue<string>(), Is.EqualTo("source"));
        Assert.That(five["dependent_outputs"]!.ToJsonString(), Is.EqualTo("""["5"]"""));
        Assert.That(five["class_type"]!.GetValue<string>(), Is.EqualTo("PreviewAny"));
        Assert.That(await Ran(body), Is.EqualTo(new[] { "2", "4" }));
    }

    [Test]
    public async Task An_error_lists_every_output_it_stops()
    {
        var prompt = new JsonObject
        {
            ["1"] = Text("x"), ["2"] = Show("1"),
            // 6 reads a node that is not there (the error is keyed on 6, the node holding the link, as ComfyUI keys it)
            ["6"] = new JsonObject { ["class_type"] = "StringConcatenate", ["inputs"] = new JsonObject { ["string_a"] = new JsonArray("99", 0), ["string_b"] = "", ["delimiter"] = "" } },
            ["4"] = Show("6"), ["7"] = Show("6"),
        };
        var (code, body) = await Post(prompt);
        Assert.That(code, Is.EqualTo(HttpStatusCode.OK));
        Assert.That((body["node_errors"] as JsonObject)!.Select(kv => kv.Key), Is.EqualTo(new[] { "6" }));
        Assert.That(body["node_errors"]!["6"]!["dependent_outputs"]!.ToJsonString(), Is.EqualTo("""["4","7"]"""));
        Assert.That(await Ran(body), Is.EqualTo(new[] { "2" }));
    }

    [Test]
    public async Task Partial_execution_targets_run_only_those_outputs()
    {
        var (code, body) = await Post(TwoGoodOneBroken(), new JsonArray("2"));
        Assert.That((code, body["node_errors"]!.ToJsonString()), Is.EqualTo((HttpStatusCode.OK, "{}")));
        Assert.That(await Ran(body), Is.EqualTo(new[] { "2" }));

        (code, body) = await Post(TwoGoodOneBroken(), JsonValue.Create("4"));
        Assert.That(await Ran(body), Is.EqualTo(new[] { "4" }), "a single id is a list of one");

        (code, body) = await Post(TwoGoodOneBroken(), new JsonArray("5", "2"));
        Assert.That((code, (body["node_errors"] as JsonObject)!.Count), Is.EqualTo((HttpStatusCode.OK, 1)));
        Assert.That(await Ran(body), Is.EqualTo(new[] { "2" }));
    }

    [Test]
    public async Task Nothing_to_run_is_refused_as_ComfyUI_refuses_it()
    {
        foreach (var targets in new JsonNode[] { new JsonArray("1"), new JsonArray("99"), new JsonArray() })
        {
            var (code, body) = await Post(TwoGoodOneBroken(), targets);
            Assert.That((code, body["error"]!["type"]!.GetValue<string>(), body["error"]!["message"]!.GetValue<string>(), body["node_errors"]!.ToJsonString()),
                Is.EqualTo((HttpStatusCode.BadRequest, "prompt_no_outputs", "Prompt has no outputs", "{}")), targets.ToJsonString());
        }
        var (noOutputs, refusal) = await Post(new JsonObject { ["1"] = Text("x") });
        Assert.That((noOutputs, refusal["error"]!["type"]!.GetValue<string>()), Is.EqualTo((HttpStatusCode.BadRequest, "prompt_no_outputs")));

        var (allBroken, failed) = await Post(TwoGoodOneBroken(), new JsonArray("5"));
        Assert.That((allBroken, failed["error"]!["type"]!.GetValue<string>()), Is.EqualTo((HttpStatusCode.BadRequest, "prompt_outputs_failed_validation")));
        Assert.That(failed["node_errors"]!["5"]!["dependent_outputs"]!.ToJsonString(), Is.EqualTo("""["5"]"""));
    }
}
