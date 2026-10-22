using System.Net;
using System.Net.Http.Json;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Server;

namespace SpawnGen.Server.Tests;

/// <summary>Nodes only the tests use: one that waits (to test interrupt and ordering) and one that fails.</summary>
[Node(Category = "test")]
public sealed class WaitNode
{
    public async Task<ImageBatch> ExecuteAsync(ImageBatch images, NodeContext context, [Input(Min = 0, Max = 600)] int seconds = 30)
    {
        context.Status("waiting");
        await Task.Delay(TimeSpan.FromSeconds(seconds), context.CancellationToken);
        return images;
    }
}

[Node(Category = "test")]
public sealed class FailNode
{
    public ImageBatch Execute(ImageBatch images) => throw new InvalidOperationException("this node always fails");
}

/// <summary>Starts a real server on a free port with its folders in a temp directory, and talks to it with an HTTP client and a websocket.</summary>
public class ServerTests
{
    WebApplication _app = null!;
    HttpClient _http = null!;
    string _dir = null!;
    ClientWebSocket? _ws;
    readonly List<JsonObject> _messages = [];
    readonly object _lock = new();

    [SetUp]
    public async Task SetUp()
    {
        lock (_lock) _messages.Clear(); // NUnit shares one fixture instance across tests
        _ws = null;
        _dir = Path.Combine(Path.GetTempPath(), "spawngen-server-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(_dir, "input"));
        _app = ServerApp.Build(
            ["--urls", "http://127.0.0.1:0", $"--SpawnGen:ModelsPath={_dir}/models", $"--SpawnGen:OutputPath={_dir}/output",
             $"--SpawnGen:InputPath={_dir}/input", $"--SpawnGen:TempPath={_dir}/temp", "--environment", "Development"],
            builder => builder.Services.AddSingleton(new NodeRegistry()
                .AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(SpawnGen.Nodes.ML.KSampler).Assembly).AddAssembly(typeof(WaitNode).Assembly)));
        await _app.StartAsync();
        var address = _app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
        _http = new HttpClient { BaseAddress = new Uri(address) };
        InputFiles.Directory = Path.Combine(_dir, "input");
    }

    [TearDown]
    public async Task TearDown()
    {
        _ws?.Dispose();
        _http.Dispose();
        await _app.StopAsync();
        await _app.DisposeAsync();
        try { Directory.Delete(_dir, true); } catch (IOException) { }
    }

    // ---- helpers ----

    async Task ConnectAsync()
    {
        _ws = new ClientWebSocket();
        await _ws.ConnectAsync(new Uri(_http.BaseAddress!.ToString().Replace("http", "ws") + "ws?clientId=test"), CancellationToken.None);
        _ = Task.Run(async () =>
        {
            var buffer = new byte[64 * 1024];
            try
            {
                while (_ws.State == WebSocketState.Open)
                {
                    var r = await _ws.ReceiveAsync(buffer, CancellationToken.None);
                    if (r.MessageType == WebSocketMessageType.Close) break;
                    var json = JsonNode.Parse(Encoding.UTF8.GetString(buffer, 0, r.Count)) as JsonObject;
                    if (json is not null) lock (_lock) _messages.Add(json);
                }
            }
            catch (WebSocketException) { }
            catch (ObjectDisposedException) { }
        });
        await WaitForAsync(m => (string?)m["type"] == "status");
    }

    List<JsonObject> Messages() { lock (_lock) return [.. _messages]; }

    async Task<JsonObject> WaitForAsync(Func<JsonObject, bool> match, int seconds = 15)
    {
        var deadline = DateTime.UtcNow.AddSeconds(seconds);
        while (DateTime.UtcNow < deadline)
        {
            var hit = Messages().FirstOrDefault(match);
            if (hit is not null) return hit;
            await Task.Delay(25);
        }
        throw new TimeoutException("Expected websocket message never arrived. Got: " + string.Join(", ", Messages().Select(m => (string?)m["type"])));
    }

    static Func<JsonObject, bool> Type(string type) => m => (string?)m["type"] == type;

    string WriteInputPng(string name, int size = 16)
    {
        var img = new ImageBatch(1, size, size, 3);
        for (int i = 0; i < img.Data.Length; i++) img.Data[i] = (i * 7 % 255) / 255f;
        File.WriteAllBytes(Path.Combine(_dir, "input", name), PngWriter.Encode(img));
        return name;
    }

    static JsonObject Prompt(string image, string? middle = null, string output = "SaveImage")
    {
        var p = new JsonObject
        {
            ["1"] = new JsonObject { ["class_type"] = "LoadImage", ["inputs"] = new JsonObject { ["image"] = image } },
        };
        string last = "1";
        if (middle is not null)
        {
            p["2"] = new JsonObject { ["class_type"] = middle, ["inputs"] = new JsonObject { ["images"] = new JsonArray("1", 0) } };
            last = "2";
        }
        p["9"] = new JsonObject { ["class_type"] = output, ["inputs"] = new JsonObject { ["images"] = new JsonArray(last, 0) } };
        return p;
    }

    Task<HttpResponseMessage> PostPrompt(JsonObject prompt) =>
        _http.PostAsJsonAsync("/prompt", new JsonObject { ["prompt"] = prompt, ["client_id"] = "test" });

    // ---- tests ----

    [Test]
    public async Task Object_info_describes_nodes_in_ComfyUIs_format()
    {
        var info = JsonNode.Parse(await _http.GetStringAsync("/object_info"))!.AsObject();
        var sampler = info["KSampler"]!.AsObject();
        Assert.That((string?)sampler["display_name"], Is.EqualTo("KSampler"));
        Assert.That(sampler["output"]!.AsArray().Select(x => (string?)x), Is.EqualTo(new[] { "LATENT" }));
        var required = sampler["input"]!["required"]!.AsObject();
        Assert.That((string?)required["model"]![0], Is.EqualTo("MODEL"));
        Assert.That((string?)required["steps"]![0], Is.EqualTo("INT"));
        Assert.That((double?)required["steps"]![1]!["default"], Is.EqualTo(20));
        Assert.That(required["sampler_name"]![0]!.AsArray().Select(x => (string?)x), Does.Contain("euler_ancestral"), "A combo lists its options.");
        Assert.That((bool?)info["SaveImage"]!["output_node"], Is.True);
    }

    [Test]
    public async Task A_prompt_runs_and_reports_every_step_then_serves_the_result()
    {
        await ConnectAsync();
        var image = WriteInputPng("in.png");

        var response = await PostPrompt(Prompt(image));
        Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK));
        var queued = (await response.Content.ReadFromJsonAsync<JsonObject>())!;
        var id = (string)queued["prompt_id"]!;

        var executed = await WaitForAsync(m => (string?)m["type"] == "executed" && (string?)m["data"]!["node"] == "9");
        await WaitForAsync(Type("execution_success"));
        var types = Messages().Select(m => (string?)m["type"]).ToList();
        Assert.That(types, Is.SupersetOf(new[] { "execution_start", "executing", "executed", "execution_success" }));
        Assert.That(types.IndexOf("execution_start"), Is.LessThan(types.IndexOf("executed")));
        Assert.That((string?)executed["data"]!["prompt_id"], Is.EqualTo(id));

        // The UI output names the saved file, and /view serves exactly what was written.
        var file = executed["data"]!["output"]!["images"]![0]!.AsObject();
        Assert.That((string?)file["type"], Is.EqualTo("output"));
        var served = await _http.GetByteArrayAsync($"/view?filename={file["filename"]}&subfolder=&type=output");
        Assert.That(served, Is.EqualTo(await File.ReadAllBytesAsync(Path.Combine(_dir, "output", (string)file["filename"]!))));
        Assert.That(served.Take(4), Is.EqualTo(new byte[] { 0x89, 0x50, 0x4E, 0x47 }), "It is a PNG.");

        var history = JsonNode.Parse(await _http.GetStringAsync($"/history/{id}"))!.AsObject();
        Assert.That((string?)history[id]!["status"]!["status_str"], Is.EqualTo("success"));
        Assert.That(history[id]!["outputs"]!["9"]!["images"]!.AsArray(), Has.Count.EqualTo(1));
        Assert.That((await _http.GetStringAsync("/history")), Does.Contain(id));
    }

    [Test]
    public async Task A_prompt_that_cannot_run_is_refused_with_errors_per_node_and_nothing_is_queued()
    {
        var bad = new JsonObject
        {
            ["1"] = new JsonObject { ["class_type"] = "LoadImage", ["inputs"] = new JsonObject { ["image"] = "x.png" } },
            ["9"] = new JsonObject { ["class_type"] = "SaveImage", ["inputs"] = new JsonObject() }, // images is missing
        };
        var response = await PostPrompt(bad);
        Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.BadRequest));
        var body = (await response.Content.ReadFromJsonAsync<JsonObject>())!;
        Assert.That((string?)body["error"]!["type"], Is.EqualTo("prompt_outputs_failed_validation"));
        Assert.That(body["node_errors"]!["9"], Is.Not.Null, "The error is attached to the node that has it.");
        Assert.That((string?)body["node_errors"]!["9"]!["class_type"], Is.EqualTo("SaveImage"));

        var queue = JsonNode.Parse(await _http.GetStringAsync("/queue"))!;
        Assert.That(queue["queue_pending"]!.AsArray(), Is.Empty);
        Assert.That(queue["queue_running"]!.AsArray(), Is.Empty);
    }

    [Test]
    public async Task A_saved_picture_carries_the_prompt_and_the_workflow_that_made_it()
    {
        await ConnectAsync();
        var image = WriteInputPng("in.png");
        var workflow = new JsonObject { ["nodes"] = new JsonArray(), ["marker"] = "from the editor" };
        var body = new JsonObject
        {
            ["prompt"] = Prompt(image), ["client_id"] = "test",
            ["extra_data"] = new JsonObject { ["extra_pnginfo"] = new JsonObject { ["workflow"] = workflow } },
        };
        Assert.That((await _http.PostAsJsonAsync("/prompt", body)).StatusCode, Is.EqualTo(HttpStatusCode.OK));
        var executed = await WaitForAsync(m => (string?)m["type"] == "executed" && (string?)m["data"]!["node"] == "9");

        var file = (string)executed["data"]!["output"]!["images"]![0]!["filename"]!;
        var text = PngReader.ReadText(await File.ReadAllBytesAsync(Path.Combine(_dir, "output", file)));
        Assert.That(JsonNode.Parse(text["prompt"])!["1"]!["class_type"]!.ToString(), Is.EqualTo("LoadImage"), "The API graph is embedded.");
        Assert.That(JsonNode.Parse(text["workflow"])!["marker"]!.ToString(), Is.EqualTo("from the editor"), "So is the editor workflow.");
    }

    [Test]
    public async Task Garbage_in_the_request_is_a_400_not_a_crash()
    {
        Assert.That((await _http.PostAsync("/prompt", new StringContent("{nope", Encoding.UTF8, "application/json"))).StatusCode, Is.EqualTo(HttpStatusCode.BadRequest));
        Assert.That((await _http.PostAsJsonAsync("/prompt", new JsonObject { ["hello"] = 1 })).StatusCode, Is.EqualTo(HttpStatusCode.BadRequest));
        Assert.That((await _http.PostAsJsonAsync("/prompt", new JsonObject { ["prompt"] = new JsonObject { ["1"] = new JsonObject { ["class_type"] = "NoSuchNode", ["inputs"] = new JsonObject() } } })).StatusCode,
            Is.EqualTo(HttpStatusCode.BadRequest));
    }

    [Test]
    public async Task Running_the_same_prompt_again_is_served_from_the_cache()
    {
        await ConnectAsync();
        var image = WriteInputPng("in.png");
        await PostPrompt(Prompt(image));
        await WaitForAsync(Type("execution_success"));
        lock (_lock) _messages.Clear();

        await PostPrompt(Prompt(image));
        await WaitForAsync(Type("execution_success"));
        var types = Messages().Select(m => (string?)m["type"]).ToList();
        Assert.That(types, Does.Contain("execution_cached"));
        Assert.That(types, Does.Not.Contain("executing"), "Nothing needs to run: every node is cached.");
    }

    [Test]
    public async Task Interrupt_stops_the_running_prompt_and_the_next_one_still_runs()
    {
        await ConnectAsync();
        var image = WriteInputPng("in.png");
        var slow = Prompt(image, middle: "WaitNode");
        slow["2"]!["inputs"]!["seconds"] = 60;
        var first = (string)(await (await PostPrompt(slow)).Content.ReadFromJsonAsync<JsonObject>())!["prompt_id"]!;
        await WaitForAsync(m => (string?)m["type"] == "executing" && (string?)m["data"]!["node"] == "2");

        var second = (string)(await (await PostPrompt(Prompt(image))).Content.ReadFromJsonAsync<JsonObject>())!["prompt_id"]!;
        var queue = JsonNode.Parse(await _http.GetStringAsync("/queue"))!;
        Assert.That(queue["queue_running"]!.AsArray(), Has.Count.EqualTo(1));
        Assert.That(queue["queue_pending"]!.AsArray(), Has.Count.EqualTo(1), "The second prompt waits its turn.");

        await _http.PostAsync("/interrupt", null);
        await WaitForAsync(Type("execution_interrupted"));
        await WaitForAsync(m => (string?)m["type"] == "execution_success" && (string?)m["data"]!["prompt_id"] == second);

        var history = JsonNode.Parse(await _http.GetStringAsync("/history"))!.AsObject();
        Assert.That((bool?)history[first]!["status"]!["interrupted"], Is.True);
        Assert.That((string?)history[second]!["status"]!["status_str"], Is.EqualTo("success"));
    }

    [Test]
    public async Task A_waiting_prompt_can_be_deleted_from_the_queue_but_the_running_one_is_not_touched()
    {
        await ConnectAsync();
        var image = WriteInputPng("in.png");
        var slow = Prompt(image, middle: "WaitNode");
        slow["2"]!["inputs"]!["seconds"] = 60;
        var running = (string)(await (await PostPrompt(slow)).Content.ReadFromJsonAsync<JsonObject>())!.AsObject()["prompt_id"]!;
        await WaitForAsync(m => (string?)m["type"] == "executing" && (string?)m["data"]!["node"] == "2");
        var waiting = (string)(await (await PostPrompt(Prompt(image))).Content.ReadFromJsonAsync<JsonObject>())!["prompt_id"]!;
        Assert.That(JsonNode.Parse(await _http.GetStringAsync("/queue"))!["queue_pending"]!.AsArray(), Has.Count.EqualTo(1));

        // Deleting the running prompt's id does nothing to it; deleting the waiting one removes it.
        await _http.PostAsJsonAsync("/queue", new JsonObject { ["delete"] = new JsonArray(running) });
        var queue = JsonNode.Parse(await _http.GetStringAsync("/queue"))!;
        Assert.That(queue["queue_running"]!.AsArray(), Has.Count.EqualTo(1));
        await _http.PostAsJsonAsync("/queue", new JsonObject { ["delete"] = new JsonArray(waiting) });
        queue = JsonNode.Parse(await _http.GetStringAsync("/queue"))!;
        Assert.That(queue["queue_pending"]!.AsArray(), Is.Empty);
        Assert.That(queue["queue_running"]!.AsArray(), Has.Count.EqualTo(1));
        await _http.PostAsync("/interrupt", null);
    }

    [Test]
    public async Task Progress_messages_from_a_node_reach_the_client()
    {
        await ConnectAsync();
        var image = WriteInputPng("in.png");
        var p = Prompt(image, middle: "WaitNode");
        p["2"]!["inputs"]!["seconds"] = 1;
        await PostPrompt(p);
        await WaitForAsync(Type("execution_success"));
        var status = Messages().First(m => (string?)m["type"] == "node_status");
        Assert.That((string?)status["data"]!["message"], Is.EqualTo("waiting"));
        Assert.That((string?)status["data"]!["node"], Is.EqualTo("2"));
    }

    [Test]
    public async Task A_node_that_throws_is_reported_with_its_message_and_the_queue_carries_on()
    {
        await ConnectAsync();
        var image = WriteInputPng("in.png");
        var id = (string)(await (await PostPrompt(Prompt(image, middle: "FailNode"))).Content.ReadFromJsonAsync<JsonObject>())!["prompt_id"]!;

        var error = await WaitForAsync(Type("execution_error"));
        Assert.That((string?)error["data"]!["node_id"], Is.EqualTo("2"));
        Assert.That((string?)error["data"]!["node_type"], Is.EqualTo("FailNode"));
        Assert.That((string?)error["data"]!["exception_message"], Is.EqualTo("this node always fails"));

        // Give the worker a moment to record it, then check history and that a good prompt still runs.
        for (int i = 0; i < 100 && !(await _http.GetStringAsync("/history")).Contains(id); i++) await Task.Delay(25);
        var history = JsonNode.Parse(await _http.GetStringAsync($"/history/{id}"))!.AsObject();
        Assert.That((string?)history[id]!["status"]!["status_str"], Is.EqualTo("error"));
        Assert.That((string?)history[id]!["status"]!["error"], Does.Contain("always fails"));

        lock (_lock) _messages.Clear();
        await PostPrompt(Prompt(image));
        await WaitForAsync(Type("execution_success"));
    }

    [Test]
    public async Task An_uploaded_image_can_be_loaded_by_a_prompt()
    {
        await ConnectAsync();
        var png = PngWriter.Encode(new ImageBatch(1, 8, 8, 3));
        using var form = new MultipartFormDataContent { { new ByteArrayContent(png), "image", "upload one.png" } };
        var upload = await _http.PostAsync("/upload/image", form);
        Assert.That(upload.StatusCode, Is.EqualTo(HttpStatusCode.OK));
        var name = (string)(await upload.Content.ReadFromJsonAsync<JsonObject>())!["name"]!;
        Assert.That(File.Exists(Path.Combine(_dir, "input", name)));

        Assert.That((await PostPrompt(Prompt(name))).StatusCode, Is.EqualTo(HttpStatusCode.OK));
        await WaitForAsync(Type("execution_success"));
    }

    [Test]
    public async Task Preview_images_are_written_to_the_temp_folder_and_served_from_there()
    {
        await ConnectAsync();
        var image = WriteInputPng("in.png");
        await PostPrompt(Prompt(image, output: "PreviewImage"));
        var executed = await WaitForAsync(m => (string?)m["type"] == "executed" && (string?)m["data"]!["node"] == "9");
        var file = executed["data"]!["output"]!["images"]![0]!;
        Assert.That((string?)file["type"], Is.EqualTo("temp"));
        var bytes = await _http.GetByteArrayAsync($"/view?filename={file["filename"]}&type=temp");
        Assert.That(bytes, Is.Not.Empty);
        Assert.That(Directory.GetFiles(Path.Combine(_dir, "output")), Is.Empty, "A preview is not a result: nothing in output.");
    }

    [Test]
    public async Task The_view_endpoint_will_not_leave_its_folders()
    {
        File.WriteAllText(Path.Combine(_dir, "secret.txt"), "private");
        foreach (var url in new[]
        {
            "/view?filename=..%2fsecret.txt&type=output",
            "/view?filename=secret.txt&subfolder=..&type=output",
            "/view?filename=..%5csecret.txt&type=input",
            $"/view?filename={Uri.EscapeDataString(Path.Combine(_dir, "secret.txt"))}&type=output",
        })
            Assert.That((await _http.GetAsync(url)).StatusCode, Is.EqualTo(HttpStatusCode.NotFound), url);
        Assert.That((await _http.GetAsync("/view?filename=x.png&type=models")).StatusCode, Is.EqualTo(HttpStatusCode.BadRequest));
    }

    [Test]
    public void SafePath_accepts_files_inside_the_root_only()
    {
        var root = Path.Combine(Path.GetTempPath(), "root");
        Assert.That(ServerApp.SafePath(root, null, "a.png"), Is.EqualTo(Path.Combine(Path.GetFullPath(root), "a.png")));
        Assert.That(ServerApp.SafePath(root, "sub", "a.png"), Is.Not.Null);
        Assert.That(ServerApp.SafePath(root, "..", "a.png"), Is.Null);
        Assert.That(ServerApp.SafePath(root, "sub/../..", "a.png"), Is.Null);
        Assert.That(ServerApp.SafePath(root, null, "../a.png"), Is.Null);
        Assert.That(ServerApp.SafePath(root, null, ""), Is.Null);
    }

}
