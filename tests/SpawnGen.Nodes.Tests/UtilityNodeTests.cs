using System.Text;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>The text, logic and utility nodes' schemas against ComfyUI's, and the nodes run through the engine.</summary>
public class UtilityNodeTests
{
    static readonly NodeRegistry Registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly);

    sealed class TestServices(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }

    /// <summary>Input names in order with their widget default (null for a socket), and output types, as ComfyUI declares them.</summary>
    static void AssertSchema(string id, string display, string category, (string Name, object? Default)[] inputs, string[] outputs)
    {
        var s = Registry.GetSchema(id) ?? throw new AssertionException($"{id} is not registered");
        Assert.Multiple(() =>
        {
            Assert.That(s.Id, Is.EqualTo(id));
            Assert.That(s.DisplayName, Is.EqualTo(display), id);
            Assert.That(s.Category, Is.EqualTo(category), id);
            Assert.That(s.Inputs.Select(i => i.Name), Is.EqualTo(inputs.Select(i => i.Name)), id + " inputs");
            foreach (var (name, dflt) in inputs)
                Assert.That(s.FindInput(name)?.Widget?.Default, Is.EqualTo(dflt), $"{id}.{name} default");
            Assert.That(s.Outputs.Select(o => o.Type.Name), Is.EqualTo(outputs), id + " outputs");
        });
    }

    [Test]
    public void String_node_schemas_match_comfyui()
    {
        AssertSchema("StringConcatenate", "Concatenate Text", "text", [("string_a", ""), ("string_b", ""), ("delimiter", "")], ["STRING"]);
        AssertSchema("StringSubstring", "Substring", "text", [("string", ""), ("start", 0L), ("end", 0L)], ["STRING"]);
        AssertSchema("StringLength", "Text Length", "text", [("string", "")], ["INT"]);
        AssertSchema("CaseConverter", "Convert Text Case", "text", [("string", ""), ("mode", "UPPERCASE")], ["STRING"]);
        AssertSchema("StringTrim", "Trim Text", "text", [("string", ""), ("mode", "Both")], ["STRING"]);
        AssertSchema("StringReplace", "Replace Text", "text", [("string", ""), ("find", ""), ("replace", "")], ["STRING"]);
        AssertSchema("StringContains", "Contains Text", "text", [("string", ""), ("substring", ""), ("case_sensitive", true)], ["BOOLEAN"]);
        AssertSchema("StringCompare", "Compare Text", "text", [("string_a", ""), ("string_b", ""), ("mode", "Starts With"), ("case_sensitive", true)], ["BOOLEAN"]);
        AssertSchema("RegexMatch", "Match Text", "text", [("string", ""), ("regex_pattern", ""), ("case_insensitive", true), ("multiline", false), ("dotall", false)], ["BOOLEAN"]);
        AssertSchema("RegexExtract", "Extract Text", "text",
            [("string", ""), ("regex_pattern", ""), ("mode", "First Match"), ("case_insensitive", true), ("multiline", false), ("dotall", false), ("group_index", 1L)], ["STRING"]);
        AssertSchema("RegexReplace", "Replace Text (Regex)", "text",
            [("string", ""), ("regex_pattern", ""), ("replace", ""), ("case_insensitive", true), ("multiline", false), ("dotall", false), ("count", 0L)], ["STRING"]);
        AssertSchema("JsonExtractString", "Extract Text from JSON", "text", [("json_string", ""), ("key", "")], ["STRING"]);
        AssertSchema("ConvertDictionaryToString", "Convert Dictionary to String", "text", [("dictionary", null), ("indent", 2L)], ["STRING"]);
        AssertSchema("ConvertArrayToString", "Convert Array to String", "text", [("array", null), ("indent", 2L)], ["STRING"]);
        AssertSchema("SaveText", "Save Text", "text", [("text", null), ("filename_prefix", "ComfyUI"), ("format", "txt")], ["STRING"]);

        Assert.That(Registry.GetSchema("CaseConverter")!.FindInput("mode")!.Widget!.ResolveOptions(), Is.EqualTo(new[] { "UPPERCASE", "lowercase", "Capitalize", "Title Case" }));
        Assert.That(Registry.GetSchema("StringTrim")!.FindInput("mode")!.Widget!.ResolveOptions(), Is.EqualTo(new[] { "Both", "Left", "Right" }));
        Assert.That(Registry.GetSchema("StringCompare")!.FindInput("mode")!.Widget!.ResolveOptions(), Is.EqualTo(new[] { "Starts With", "Ends With", "Equal" }));
        Assert.That(Registry.GetSchema("RegexExtract")!.FindInput("mode")!.Widget!.ResolveOptions(), Is.EqualTo(new[] { "First Match", "All Matches", "First Group", "All Groups" }));
        Assert.That(Registry.GetSchema("SaveText")!.FindInput("format")!.Widget!.ResolveOptions(), Is.EqualTo(new[] { "txt", "csv", "md", "json" }));
        var group = Registry.GetSchema("RegexExtract")!.FindInput("group_index")!.Widget!;
        Assert.That((group.Min, group.Max), Is.EqualTo(((double?)0, (double?)100)));
        var count = Registry.GetSchema("RegexReplace")!.FindInput("count")!.Widget!;
        Assert.That((count.Min, count.Max), Is.EqualTo(((double?)0, (double?)100)));
        var indent = Registry.GetSchema("ConvertDictionaryToString")!.FindInput("indent")!.Widget!;
        Assert.That((indent.Min, indent.Max), Is.EqualTo(((double?)0, (double?)8)));
        Assert.That(Registry.GetSchema("ConvertDictionaryToString")!.FindInput("dictionary")!.Type.Name, Is.EqualTo("DICT"));
        Assert.That(Registry.GetSchema("ConvertArrayToString")!.FindInput("array")!.Type.Name, Is.EqualTo("ARRAY"));
        Assert.That(Registry.GetSchema("SaveText")!.IsOutputNode, Is.True);
        Assert.That(Registry.GetSchema("SaveText")!.Outputs[0].Name, Is.EqualTo("text"));
    }

    [Test]
    public void Utility_node_schemas_match_comfyui()
    {
        AssertSchema("SeedNode", "Seed", "utilities", [("seed", 0L)], ["INT"]);
        var seed = Registry.GetSchema("SeedNode")!.FindInput("seed")!.Widget!;
        Assert.That((seed.Min, seed.Max), Is.EqualTo(((double?)0, (double?)long.MaxValue)), "0 .. sys.maxsize");
        AssertSchema("ComfyNumberConvert", "Convert Number", "utilities", [("value", null)], ["FLOAT", "INT"]);
        Assert.That(Registry.GetSchema("ComfyNumberConvert")!.Outputs.Select(o => o.Name), Is.EqualTo(new[] { "FLOAT", "INT" }));
        AssertSchema("ComfyNotNode", "Not", "utilities/logic", [("value", null)], ["BOOLEAN"]);
        Assert.That(Registry.GetSchema("ComfyNotNode")!.FindInput("value")!.Required, Is.True);
    }

    [Test]
    public void Seed_passes_the_seed_through() => Assert.That(new SeedNode().Execute(123456789012L), Is.EqualTo(123456789012L));

    [Test]
    public void Not_uses_python_truthiness()
    {
        var not = new ComfyNotNode();
        Assert.Multiple(() =>
        {
            Assert.That(not.Execute(false), Is.True);
            Assert.That(not.Execute(true), Is.False);
            Assert.That(not.Execute(0L), Is.True);
            Assert.That(not.Execute(3L), Is.False);
            Assert.That(not.Execute(0.0), Is.True);
            Assert.That(not.Execute(-0.5), Is.False);
            Assert.That(not.Execute(""), Is.True);
            Assert.That(not.Execute("0"), Is.False, "a non-empty text is true, even \"0\"");
            Assert.That(not.Execute(new ArrayValue()), Is.True);
            Assert.That(not.Execute(new DictValue { ["a"] = 1L }), Is.False);
            Assert.That(not.Execute(new LatentBatch(1, 4, 8, 8)), Is.False, "a LATENT is a dict in ComfyUI: true");
            Assert.That(not.Execute(new MaskBatch(1, 1, 1, [0f])), Is.True, "a one-value tensor is its value");
            Assert.That(() => not.Execute(new ImageBatch(1, 2, 2, 3)), Throws.InvalidOperationException, "torch refuses the truth value of a larger tensor");
        });
    }

    [Test]
    public void Number_convert_errors_like_comfyui()
    {
        var c = new ComfyNumberConvert();
        Assert.That(() => c.Execute("  "), Throws.ArgumentException.With.Message.EqualTo("Cannot convert empty string to number."));
        Assert.That(() => c.Execute("abc"), Throws.ArgumentException.With.Message.EqualTo("Cannot convert string to number: 'abc'"));
        Assert.That(() => c.Execute("inf"), Throws.ArgumentException.With.Message.EqualTo("Cannot convert non-finite value to number: inf"));
        Assert.That(() => c.Execute(new ImageBatch(1, 1, 1)), Throws.ArgumentException);
        Assert.That(c.Execute("  3.75 "), Is.EqualTo((3.75, 3L)));
        Assert.That(c.Execute("1e3"), Is.EqualTo((1000.0, 1000L)), "int('1e3') fails, so int(float) is used");
        Assert.That(c.Execute(-2.9), Is.EqualTo((-2.9, -2L)), "truncated toward zero");
    }

    [Test]
    public void Regex_nodes_follow_python_not_dotnet()
    {
        // .NET numbers named groups after the unnamed ones; Python numbers all groups left to right.
        Assert.That(new RegexExtract().Execute("ab12", @"(?P<letters>[a-z]+)(\d+)", "First Group", true, false, false, 2), Is.EqualTo("12"));
        // \Z is the very end in Python (not "before a final newline").
        Assert.That(new RegexMatchNode().Execute("end\n", @"end\Z", false, false, false), Is.False);
        // Python's {,n} is {0,n}; in .NET it would be literal text.
        Assert.That(new RegexExtract().Execute("aaaa", "a{,2}", "First Match", false, false, false, 1), Is.EqualTo("aa"));
        // A pattern Python cannot compile: Match Text gives false, Extract Text "", Replace Text (Regex) fails.
        Assert.That(new RegexMatchNode().Execute("x", "(", true, false, false), Is.False);
        Assert.That(new RegexExtract().Execute("x", "(", "First Match", true, false, false, 1), Is.EqualTo(""));
        Assert.That(() => new RegexReplaceNode().Execute("x", "(", "y", true, false, false, 0), Throws.TypeOf<PyException>());
        // An emoji is one character in Python.
        Assert.That(new RegexReplaceNode().Execute("a\U0001F600\U0001F600b", "\U0001F600+", "-", true, false, false, 0), Is.EqualTo("a-b"));
    }

    [Test]
    public async Task Save_Text_writes_numbered_files_through_the_result_store_and_passes_the_text_on()
    {
        var store = new MemoryResultStore();
        var services = new TestServices(store);
        var g = new PromptGraph();
        g.Add("1", "PrimitiveStringMultiline", ("value", "{\"name\": \"cat\", \"legs\": 4} trailing"));
        g.Add("2", "JsonExtractString", ("json_string", PromptInput.Link("1")), ("key", "legs"));
        g.Add("3", "SaveText", ("text", PromptInput.Link("2")), ("filename_prefix", "notes/legs"), ("format", "txt"));
        g.Add("4", "SaveText", ("text", PromptInput.Link("1")), ("filename_prefix", "raw"), ("format", "md"));
        var result = await new PromptExecutor(Registry, services).ExecuteAsync(g);

        var ui = result.Outputs["3"].Ui!;
        Assert.That(ui["text"], Is.EqualTo(new[] { "4" }));
        var file = (ResultFile)((object[])ui["files"]!)[0];
        Assert.That((file.Filename, file.Subfolder, file.Type), Is.EqualTo(("legs_00001_.txt", "notes", "output")));
        Assert.That(Encoding.UTF8.GetString(store.Get("output", "legs_00001_.txt", "notes")!), Is.EqualTo("4"));
        Assert.That(result.Outputs["3"].Values, Is.EqualTo(new object[] { "4" }), "the text is also an output");
        Assert.That(Encoding.UTF8.GetString(store.Get("output", "raw_00001_.md")!), Is.EqualTo("{\"name\": \"cat\", \"legs\": 4} trailing"));
    }

    [Test]
    public async Task Save_Text_json_pretty_prints_valid_json_like_python()
    {
        var store = new MemoryResultStore();
        foreach (var (text, expected) in PythonVectors.SaveTextJson)
        {
            var (passed, ui) = await new SaveText().ExecuteAsync(text, new TestServices(store), "data", "json");
            var file = (ResultFile)((object[])ui["files"]!)[0];
            var bytes = store.Get("output", file.Filename)!;
            Assert.That(Encoding.UTF8.GetString(bytes), Is.EqualTo(expected), text);
            Assert.That(bytes.Take(3), Is.Not.EqualTo(new byte[] { 0xEF, 0xBB, 0xBF }), "no byte order mark");
            Assert.That(passed, Is.EqualTo(text), "the output is the text as given");
        }
        // Only text: a number linked in is refused before anything runs, as ComfyUI's STRING socket refuses it.
        var g = new PromptGraph();
        g.Add("1", "PrimitiveInt", ("value", 5));
        g.Add("2", "SaveText", ("text", PromptInput.Link("1")));
        Assert.That(new PromptExecutor(Registry).Validate(g).Errors.Select(e => (e.NodeId, e.Input, e.Code)), Does.Contain(((string?)"2", (string?)"text", "type_mismatch")));
    }

    [Test]
    public void Dict_and_array_sockets_carry_python_values()
    {
        var d = new DictValue { ["b"] = 1L, ["a"] = new ArrayValue([1L, 2.5, "x", null, true]) };
        Assert.That(new ConvertDictionaryToString().Execute(d, 0), Is.EqualTo("{\"b\": 1, \"a\": [1, 2.5, \"x\", null, true]}"), "insertion order, Python's separators");
        Assert.That(new ConvertArrayToString().Execute(new ArrayValue([1.0, 1e16]), 1), Is.EqualTo("[\n 1.0,\n 1e+16\n]"));
    }
}
