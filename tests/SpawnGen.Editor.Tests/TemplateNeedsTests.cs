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
    public void A_listed_model_SpawnGen_has_not_run_keeps_a_template_out_of_Ready()
    {
        // Found 2026-11-19: Wan 2.2 14B's template showed "ready" because its files are listed, though SpawnGen runs only the 5B model.
        var registry = EditorDocumentFixtures.Registry();
        var needs = TemplateNeeds.Parse("""
            {"commit": "c", "verified_models": ["ran_before.safetensors"], "templates": {
              "verified": {"nodes": ["KSampler"], "models": ["ran_before.safetensors"]},
              "untested": {"nodes": ["KSampler"], "models": ["ran_before.safetensors", "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"]}}}
            """);
        var offered = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { "ran_before.safetensors", "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors" };
        Assert.That(needs.Check("verified", registry, offered)!.Ready, Is.True);
        var untested = needs.Check("untested", registry, offered)!;
        Assert.That(untested.Opens, Is.True);
        Assert.That(untested.MissingModels, Is.Empty);
        Assert.That(untested.UnverifiedModels, Is.EqualTo(new[] { "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors" }));
        Assert.That(untested.Ready, Is.False);
    }

    [Test]
    public void The_shipped_list_names_the_verified_models_and_not_Wan_14B()
    {
        var json = System.Text.Json.JsonDocument.Parse(NeedsJson()).RootElement.GetProperty("verified_models");
        var verified = json.EnumerateArray().Select(e => e.GetString()).ToList();
        Assert.That(verified, Does.Contain("ltx-2-19b-dev-fp8.safetensors").And.Contain("z_image_turbo_bf16.safetensors"));
        Assert.That(verified.Any(v => v!.Contains("14B")), Is.False);
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
