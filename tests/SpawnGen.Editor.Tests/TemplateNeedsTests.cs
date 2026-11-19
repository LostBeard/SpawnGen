using SpawnGen.Editor;

namespace SpawnGen.Editor.Tests;

/// <summary>The template browser's check (TemplateNeeds) and the shipped list of what ComfyUI's templates need (wwwroot/template-needs.json).</summary>
public class TemplateNeedsTests
{
    static string Repo()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "SpawnGen.slnx"))) dir = dir.Parent;
        return dir!.FullName;
    }

    static string NeedsJson() => File.ReadAllText(Path.Combine(Repo(), "src", "SpawnGen.Studio", "wwwroot", "template-needs.json"));

    [Test]
    public void A_template_with_a_node_SpawnGen_lacks_does_not_open()
    {
        var registry = EditorDocumentFixtures.Registry();
        var needs = TemplateNeeds.Parse("""
            {"commit": "c", "templates": {
              "plain": {"nodes": ["KSampler", "SaveImage"], "models": []},
              "lacks": {"nodes": ["KSampler", "NoSuchNodeAnywhere"], "models": []}}}
            """);
        var files = TemplateNeeds.OfferedFiles(registry);
        Assert.That(needs.Check("plain", registry, files), Is.EqualTo(new TemplateStatus([], [])).Using<TemplateStatus>(Same));
        var lacks = needs.Check("lacks", registry, files)!;
        Assert.That(lacks.MissingNodes, Is.EqualTo(new[] { "NoSuchNodeAnywhere" }));
        Assert.That(lacks.Opens, Is.False);
        Assert.That(needs.Check("not-in-the-list", registry, files), Is.Null, "a template newer than the list: unknown, not \"ready\"");
    }

    [Test]
    public void A_model_file_counts_when_a_loader_offers_it_by_name()
    {
        var registry = EditorDocumentFixtures.Registry();
        var needs = TemplateNeeds.Parse("""{"commit": "c", "templates": {"t": {"nodes": ["KSampler"], "models": ["no_such_model_v9.safetensors"]}}}""");
        var status = needs.Check("t", registry, TemplateNeeds.OfferedFiles(registry))!;
        Assert.That(status.Opens, Is.True);
        Assert.That(status.Ready, Is.False);
        Assert.That(status.MissingModels, Is.EqualTo(new[] { "no_such_model_v9.safetensors" }));

        // Offered in a sub-folder (as the desktop lists models\sub\file): found by its name.
        var offered = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "no_such_model_v9.safetensors" };
        Assert.That(needs.Check("t", registry, offered)!.Ready, Is.True);
    }

    [Test]
    public void The_shipped_list_is_for_the_commit_the_Studio_and_the_server_open()
    {
        var needs = TemplateNeeds.Parse(NeedsJson());
        Assert.That(needs.Commit, Does.Match("^[0-9a-f]{40}$"));
        var studio = File.ReadAllText(Path.Combine(Repo(), "src", "SpawnGen.Studio", "Pages", "Studio.razor"));
        var server = File.ReadAllText(Path.Combine(Repo(), "src", "SpawnGen.Server", "ServerOptions.cs"));
        Assert.That(studio, Does.Contain($"Comfy-Org/workflow_templates/{needs.Commit}/templates/"), "Studio.razor TemplatesUpstream");
        Assert.That(server, Does.Contain($"Comfy-Org/workflow_templates/{needs.Commit}/templates\""), "ServerOptions.TemplatesSource");
    }

    [Test]
    public void The_shipped_list_reads_the_LTX_2_template_as_measured()
    {
        // video_ltx2_t2v is the template PARITY.md measures the LTX-2 nodes on: its nodes and model files as the template names them.
        var needs = TemplateNeeds.Parse(NeedsJson());
        Assert.That(needs.TryGet("video_ltx2_t2v", out var ltx2), Is.True);
        Assert.That(ltx2.Nodes, Does.Contain("LTXVAudioVAELoader").And.Contain("SamplerCustomAdvanced").And.Not.Contain("MarkdownNote"));
        Assert.That(ltx2.Models, Does.Contain("ltx-2-19b-dev-fp8.safetensors").And.Contain("gemma_3_12B_it_fp4_mixed.safetensors"));
        Assert.That(ltx2.Nodes.Any(n => n.Length == 36 && n.Count(c => c == '-') == 4), Is.False, "sub-graph ids are not node types");
    }

    static bool Same(TemplateStatus a, TemplateStatus b) => a.MissingNodes.SequenceEqual(b.MissingNodes) && a.MissingModels.SequenceEqual(b.MissingModels);
}
