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
/// The user's files through ComfyUI's userdata and settings API. The expected codes and shapes are what ComfyUI's own server answered to the same
/// requests (a black-box probe of it), so ComfyUI clients and the Studio see the same library.
/// </summary>
public class UserDataTests
{
    string _dir = "";
    WebApplication _app = null!;
    HttpClient _http = null!;

    [SetUp]
    public async Task SetUp()
    {
        _dir = Path.Combine(Path.GetTempPath(), "spawngen-userdata-" + Guid.NewGuid().ToString("N"));
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

    async Task<(HttpStatusCode Code, string Body)> Send(HttpMethod method, string url, string? body = null)
    {
        using var request = new HttpRequestMessage(method, url);
        if (body is not null) request.Content = new StringContent(body, Encoding.UTF8, "application/json");
        using var response = await _http.SendAsync(request);
        return (response.StatusCode, await response.Content.ReadAsStringAsync());
    }

    string OnDisk(string relative) => Path.Combine(_dir, "user", "default", relative);

    [Test]
    public async Task Files_are_written_read_listed_moved_and_deleted_as_ComfyUI_does()
    {
        Assert.That(await Send(HttpMethod.Get, "api/userdata"), Is.EqualTo((HttpStatusCode.BadRequest, "Directory not provided")));
        Assert.That(await Send(HttpMethod.Get, "api/userdata?dir=nope"), Is.EqualTo((HttpStatusCode.NotFound, "Directory not found")));
        Assert.That(await Send(HttpMethod.Get, "api/userdata?dir=workflows"), Is.EqualTo((HttpStatusCode.OK, "[]")), "the library's folder is made at start (ComfyUI makes it on first save)");

        Assert.That(await Send(HttpMethod.Post, "api/userdata/workflows%2Fa.json", """{"nodes":[1]}"""), Is.EqualTo((HttpStatusCode.OK, "\"workflows/a.json\"")));
        Assert.That(await Send(HttpMethod.Post, "api/userdata/workflows%2Fa.json", """{"nodes":[2]}"""), Is.EqualTo((HttpStatusCode.OK, "\"workflows/a.json\"")), "overwrite is the default");
        Assert.That(await Send(HttpMethod.Post, "api/userdata/workflows%2Fa.json?overwrite=false", "{}"), Is.EqualTo((HttpStatusCode.Conflict, "File already exists")));
        var info = await Send(HttpMethod.Post, "api/userdata/workflows%2Fa.json?overwrite=true&full_info=true", """{"nodes":[4]}""");
        var o = JsonNode.Parse(info.Body)!;
        Assert.That((o["path"]!.GetValue<string>(), o["size"]!.GetValue<long>()), Is.EqualTo(("workflows/a.json", 13L)));
        Assert.That(o["modified"]!.GetValue<long>(), Is.GreaterThan(1_600_000_000_000L), "milliseconds");
        Assert.That(o["created"], Is.Not.Null);
        await Send(HttpMethod.Post, "api/userdata/workflows%2Fsub%2Fb.app.json", """{"extra":{}}""");
        Assert.That(File.Exists(OnDisk("workflows/sub/b.app.json")), Is.True, "folders are made");

        Assert.That((await Send(HttpMethod.Get, "api/userdata/workflows%2Fa.json")).Body, Is.EqualTo("""{"nodes":[4]}"""));
        Assert.That((await Send(HttpMethod.Get, "api/userdata/workflows/a.json")).Code, Is.EqualTo(HttpStatusCode.NotFound), "an unescaped slash is not part of a name");
        Assert.That((await Send(HttpMethod.Get, "api/userdata/workflows%2Fmissing.json")).Code, Is.EqualTo(HttpStatusCode.NotFound));

        Assert.That((await Send(HttpMethod.Get, "api/userdata?dir=workflows")).Body, Is.EqualTo("""["a.json"]"""));
        Assert.That((await Send(HttpMethod.Get, "api/userdata?dir=workflows/")).Body, Is.EqualTo("""["a.json"]"""));
        Assert.That((await Send(HttpMethod.Get, "api/userdata?dir=workflows&recurse=true")).Body, Is.EqualTo("""["a.json","sub/b.app.json"]"""));
        Assert.That((await Send(HttpMethod.Get, "api/userdata?dir=workflows&recurse=true&split=true")).Body,
            Is.EqualTo("""[["a.json","a.json"],["sub/b.app.json","sub","b.app.json"]]"""));
        var full = JsonNode.Parse((await Send(HttpMethod.Get, "api/userdata?dir=workflows&recurse=true&split=false&full_info=true")).Body)!.AsArray();
        Assert.That(full.Select(f => f!["path"]!.GetValue<string>()), Is.EqualTo(new[] { "a.json", "sub/b.app.json" }));
        Assert.That(full[1]!["size"]!.GetValue<long>(), Is.EqualTo(12));
        Assert.That((await Send(HttpMethod.Get, "api/userdata?dir=workflows%2Fsub")).Body, Is.EqualTo("""["b.app.json"]"""));
        Assert.That((await Send(HttpMethod.Get, "userdata?dir=workflows")).Body, Is.EqualTo("""["a.json"]"""), "without /api too");

        Assert.That(await Send(HttpMethod.Post, "api/userdata/workflows%2Fa.json/move/workflows%2Fc.json"), Is.EqualTo((HttpStatusCode.OK, "\"workflows/c.json\"")));
        await Send(HttpMethod.Post, "api/userdata/workflows%2Fx.json", "{}");
        Assert.That(await Send(HttpMethod.Post, "api/userdata/workflows%2Fx.json/move/workflows%2Fc.json?overwrite=false"), Is.EqualTo((HttpStatusCode.Conflict, "File already exists")));
        var moved = JsonNode.Parse((await Send(HttpMethod.Post, "api/userdata/workflows%2Fx.json/move/workflows%2Fdeep%2Fd.json?full_info=true")).Body)!;
        Assert.That(moved["path"]!.GetValue<string>(), Is.EqualTo("workflows/deep/d.json"));
        Assert.That((await Send(HttpMethod.Post, "api/userdata/workflows%2Fnone.json/move/workflows%2Fq.json")).Code, Is.EqualTo(HttpStatusCode.NotFound));

        Assert.That((await Send(HttpMethod.Delete, "api/userdata/workflows%2Fdeep%2Fd.json")).Code, Is.EqualTo(HttpStatusCode.NoContent));
        Assert.That((await Send(HttpMethod.Delete, "api/userdata/workflows%2Fdeep%2Fd.json")).Code, Is.EqualTo(HttpStatusCode.NotFound));
        Assert.That((await Send(HttpMethod.Get, "api/userdata?dir=workflows&recurse=true")).Body, Is.EqualTo("""["c.json","sub/b.app.json"]"""));

        var v2 = JsonNode.Parse((await Send(HttpMethod.Get, "api/v2/userdata?path=workflows")).Body)!.AsArray();
        Assert.That(v2.Select(e => $"{e!["type"]}:{e["path"]}"), Is.EqualTo(new[]
            { "directory:workflows/deep", "directory:workflows/sub", "file:workflows/sub/b.app.json", "file:workflows/c.json" }));
        Assert.That((await Send(HttpMethod.Get, "api/v2/userdata?path=nope")).Code, Is.EqualTo(HttpStatusCode.NotFound));
    }

    [Test]
    public async Task Paths_outside_the_users_folder_are_refused()
    {
        foreach (var name in new[] { "..%2Fevil.json", "workflows%2F..%2F..%2Fevil.json", "%2Fabs.json", "C%3A%2Fevil.json" })
            Assert.That((await Send(HttpMethod.Post, "api/userdata/" + name, "{}")).Code, Is.EqualTo(HttpStatusCode.Forbidden), name);
        Assert.That(await Send(HttpMethod.Get, "api/userdata?dir=.."), Is.EqualTo((HttpStatusCode.Forbidden, "Invalid directory")));
        Assert.That(await Send(HttpMethod.Get, "api/userdata?dir=workflows%2F..%2F.."), Is.EqualTo((HttpStatusCode.Forbidden, "Invalid directory")));
        await Send(HttpMethod.Post, "api/userdata/workflows%2Fa.json", "{}");
        Assert.That((await Send(HttpMethod.Post, "api/userdata/workflows%2Fa.json/move/..%2Fout.json")).Code, Is.EqualTo(HttpStatusCode.NotFound));
        Assert.That((await Send(HttpMethod.Post, "api/userdata/workflows", "{}")).Code, Is.EqualTo(HttpStatusCode.BadRequest), "a folder is not a file");
        Assert.That(File.Exists(Path.Combine(_dir, "user", "evil.json")) || File.Exists(Path.Combine(_dir, "evil.json")), Is.False);
    }

    [Test]
    public async Task Settings_are_kept_in_comfy_settings_json()
    {
        Assert.That((await Send(HttpMethod.Get, "api/users")).Body, Is.EqualTo("""{"storage":"server","migrated":true}"""));
        Assert.That((await Send(HttpMethod.Get, "api/settings")).Body, Is.EqualTo("{}"));
        Assert.That((await Send(HttpMethod.Post, "api/settings/Comfy.Foo", "42")).Code, Is.EqualTo(HttpStatusCode.OK));
        Assert.That((await Send(HttpMethod.Post, "api/settings", """{"A.b":"c"}""")).Code, Is.EqualTo(HttpStatusCode.OK));
        Assert.That(JsonNode.Parse((await Send(HttpMethod.Get, "api/settings")).Body)!.ToJsonString(), Is.EqualTo("""{"Comfy.Foo":42,"A.b":"c"}"""));
        Assert.That((await Send(HttpMethod.Get, "api/settings/Comfy.Foo")).Body, Is.EqualTo("42"));
        Assert.That((await Send(HttpMethod.Get, "api/settings/Missing")).Body, Is.EqualTo("null"));
        Assert.That(File.Exists(OnDisk("comfy.settings.json")), Is.True);
        Assert.That((await Send(HttpMethod.Get, "api/userdata?dir=.")).Body, Is.EqualTo("""["comfy.settings.json"]"""));
    }

    [Test]
    public async Task Every_route_also_answers_under_api()
    {
        Assert.That((await Send(HttpMethod.Get, "api/queue")).Code, Is.EqualTo(HttpStatusCode.OK));
        Assert.That((await Send(HttpMethod.Get, "api/object_info/SaveImage")).Body, Does.Contain("SaveImage"));
    }
}
