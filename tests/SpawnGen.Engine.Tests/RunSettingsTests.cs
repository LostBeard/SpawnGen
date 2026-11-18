using System.Text.Json.Nodes;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Engine.Tests;

/// <summary>SpawnGen's run settings (extra_data.spawngen_settings): read from JSON, seen by the nodes, and part of the cache signature.</summary>
public class RunSettingsTests
{
    NodeRegistry _registry = null!;
    PromptExecutor _executor = null!;

    [SetUp]
    public void SetUp()
    {
        Calls.Reset();
        _registry = new NodeRegistry().AddAssembly(typeof(RunSettingsTests).Assembly);
        _executor = new PromptExecutor(_registry);
    }

    static PromptGraph Echo()
    {
        var g = new PromptGraph();
        g.Add("1", "SettingsEcho");
        g.Add("2", "StringSink", ("text", PromptInput.Link("1", 0)));
        return g;
    }

    static ExecutionOptions With(params (string Key, string Value)[] settings) =>
        new() { Settings = settings.ToDictionary(s => s.Key, s => s.Value) };

    static string Text(ExecutionResult r) => (string)r.Outputs["2"].Ui!["text"]!;

    [Test]
    public async Task Nodes_see_the_runs_settings()
    {
        var r = await _executor.ExecuteAsync(Echo(), With((RunSettings.Ltx2ComfyFp8Lora, "false"), ("b.other", "1")));
        Assert.That(Text(r), Is.EqualTo("b.other=1;ltx2.comfy_fp8_lora=false"), "sorted by key");
        r = await _executor.ExecuteAsync(Echo(), new ExecutionOptions());
        Assert.That(Text(r), Is.EqualTo(""));
    }

    [Test]
    public async Task Another_setting_runs_again_and_the_same_one_is_cached()
    {
        await _executor.ExecuteAsync(Echo(), With((RunSettings.Ltx2ComfyFp8Lora, "true")));
        var same = await _executor.ExecuteAsync(Echo(), With((RunSettings.Ltx2ComfyFp8Lora, "true")));
        Assert.That(same.Executed, Is.Empty, "the same settings: all cached");
        var other = await _executor.ExecuteAsync(Echo(), With((RunSettings.Ltx2ComfyFp8Lora, "false")));
        Assert.That(other.Executed, Is.EquivalentTo(new[] { "1", "2" }), "another value of a setting changes every signature");
        Assert.That(Text(other), Is.EqualTo("ltx2.comfy_fp8_lora=false"));
        var none = await _executor.ExecuteAsync(Echo(), new ExecutionOptions());
        Assert.That(none.Executed, Is.EquivalentTo(new[] { "1", "2" }), "no settings is not the same as some");
        Assert.That(Calls.Of("SettingsEcho"), Is.EqualTo(3));
    }

    [Test]
    public void Settings_are_read_from_extra_data_and_flags_fall_back()
    {
        var read = RunSettings.FromJson(JsonNode.Parse("""{"ltx2.comfy_fp8_lora": false, "a": "x", "n": 3, "skip": {"nested": 1}}"""));
        Assert.That(read, Is.EquivalentTo(new Dictionary<string, string> { [RunSettings.Ltx2ComfyFp8Lora] = "false", ["a"] = "x", ["n"] = "3" }));
        Assert.That(RunSettings.Flag(read, RunSettings.Ltx2ComfyFp8Lora, true), Is.False);
        Assert.That(RunSettings.Flag(read, "a", true), Is.True, "not a boolean: the fallback");
        Assert.That(RunSettings.Flag(null, RunSettings.Ltx2ComfyFp8Lora, true), Is.True);
        Assert.That(RunSettings.FromJson(JsonNode.Parse("{}")), Is.Null);
        Assert.That(RunSettings.FromJson(null), Is.Null);
    }
}
