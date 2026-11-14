using System.Net;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;

namespace SpawnGen.Server.Tests;

/// <summary>
/// ComfyUI's workflow templates at /templates/&lt;file&gt;, as ComfyUI's server serves them (measured: index.json, a template's .json, its -1.webp).
/// Offline here: the source is turned off and the kept copies are served.
/// </summary>
public class TemplatesTests
{
    string _dir = "";
    WebApplication _app = null!;
    HttpClient _http = null!;

    [SetUp]
    public async Task SetUp()
    {
        _dir = Path.Combine(Path.GetTempPath(), "spawngen-templates-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(_dir, "templates"));
        File.WriteAllText(Path.Combine(_dir, "templates", "index.json"), """[{"title":"Image","templates":[{"name":"image_z_image_turbo"}]}]""");
        File.WriteAllText(Path.Combine(_dir, "templates", "image_z_image_turbo.json"), """{"nodes":[],"links":[]}""");
        File.WriteAllBytes(Path.Combine(_dir, "templates", "image_z_image_turbo-1.webp"), [0x52, 0x49, 0x46, 0x46]);
        _app = ServerApp.Build(
            ["--urls", "http://127.0.0.1:0", $"--SpawnGen:ModelsPath={_dir}/models", $"--SpawnGen:OutputPath={_dir}/output", $"--SpawnGen:InputPath={_dir}/input",
             $"--SpawnGen:TempPath={_dir}/temp", $"--SpawnGen:UserPath={_dir}/user", $"--SpawnGen:CustomNodesPath={_dir}/custom_nodes",
             $"--SpawnGen:TemplatesPath={_dir}/templates", "--SpawnGen:TemplatesSource=", "--environment", "Development"]);
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

    [Test]
    public async Task The_index_a_template_and_its_thumbnail_are_served_with_their_types()
    {
        var index = await _http.GetAsync("/templates/index.json");
        Assert.That(index.StatusCode, Is.EqualTo(HttpStatusCode.OK));
        Assert.That(index.Content.Headers.ContentType?.MediaType, Is.EqualTo("application/json"));
        Assert.That(await index.Content.ReadAsStringAsync(), Does.Contain("image_z_image_turbo"));
        Assert.That((await _http.GetAsync("/templates/image_z_image_turbo.json")).StatusCode, Is.EqualTo(HttpStatusCode.OK));
        var thumb = await _http.GetAsync("/templates/image_z_image_turbo-1.webp");
        Assert.That(thumb.Content.Headers.ContentType?.MediaType, Is.EqualTo("image/webp"));
    }

    [TestCase("missing.json")]
    [TestCase("..%2Fuser%2Fsecret.json")]
    [TestCase(".hidden")]
    public async Task Missing_files_and_anything_but_a_plain_name_are_not_found(string file)
    {
        Assert.That((await _http.GetAsync("/templates/" + file)).StatusCode, Is.EqualTo(HttpStatusCode.NotFound));
    }

    [Test]
    public void Only_plain_names_are_servable()
    {
        Assert.That(Templates.IsServable("image_z_image_turbo-1.webp"), Is.True);
        Assert.That(Templates.IsServable("../x.json"), Is.False);
        Assert.That(Templates.IsServable("a/b.json"), Is.False);
        Assert.That(Templates.IsServable("a..b"), Is.False);
        Assert.That(Templates.IsServable(".x"), Is.False);
    }
}
