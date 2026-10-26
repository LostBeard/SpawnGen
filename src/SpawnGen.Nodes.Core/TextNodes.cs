using System.Text;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// More of ComfyUI's text nodes (comfy_extras/nodes_string.py and nodes_text.py): regex extraction, JSON in and out, saving text files.

[Node("RegexExtract", Category = "text", DisplayName = "Extract Text")]
public sealed class RegexExtract
{
    public static string[] Modes() => ["First Match", "All Matches", "First Group", "All Groups"];

    /// <summary>
    /// The first match, every match (one per line; with groups, group 1 of each), the first match's group <c>group_index</c>, or that group of every match.
    /// A pattern Python cannot compile gives "" (ComfyUI catches <c>re.error</c>).
    /// </summary>
    public string Execute([Input(Multiline = true)] string @string = "", [Input(Multiline = true)] string regexPattern = "",
        [Input(OptionsProvider = nameof(Modes))] string mode = "First Match", bool caseInsensitive = true, bool multiline = false, bool dotall = false,
        [Input(Min = 0, Max = 100)] long groupIndex = 1)
    {
        PyRegex regex;
        try { regex = PyRegex.Compile(regexPattern, caseInsensitive, multiline, dotall); }
        catch (PyException e) when (e.PyType == "error") { return ""; }
        switch (mode)
        {
            case "First Match":
                return regex.Search(@string)?.Value ?? "";
            case "All Matches":
                // re.findall: the whole match without groups, else group 1 of each match ("" when it did not take part)
                return string.Join("\n", regex.FindIter(@string).Select(m => regex.Groups == 0 ? m.Value : PyRegex.Group(m, 1) ?? ""));
            case "First Group":
            {
                var m = regex.Search(@string);
                // A group that did not take part is None in ComfyUI; an empty text here.
                return m is not null && regex.Groups >= groupIndex ? PyRegex.Group(m, (int)groupIndex) ?? "" : "";
            }
            case "All Groups":
            {
                var parts = new List<string>();
                if (regex.Groups > 0 && regex.Groups >= groupIndex)
                    foreach (var m in regex.FindIter(@string))
                        parts.Add(PyRegex.Group(m, (int)groupIndex)
                            ?? throw new PyException("TypeError", $"sequence item {parts.Count}: expected str instance, NoneType found")); // "\n".join fails on None
                return string.Join("\n", parts);
            }
            default:
                return "";
        }
    }
}

[Node("JsonExtractString", Category = "text", DisplayName = "Extract Text from JSON",
    Description = "Extract a key's value from the first valid JSON object in the text, including Markdown code blocks.")]
public sealed class JsonExtractString
{
    /// <summary>
    /// The value of <c>key</c> in the first <c>{...}</c> that is valid JSON (Python's <c>json</c> rules), as Python's <c>str()</c> writes it:
    /// a string as is, numbers as Python prints them, lists and objects in Python's notation (<c>{'a': 1}</c>); "" when the key is missing or null.
    /// </summary>
    public string Execute([Input(Multiline = true)] string jsonString = "", string key = "")
    {
        for (int i = jsonString.IndexOf('{'); i >= 0; i = jsonString.IndexOf('{', i + 1))
        {
            if (!PyJson.TryRawDecode(jsonString, i, out var data, out _)) continue;
            var value = ((DictValue)data!).GetValueOrDefault(key);
            return value is null ? "" : PyText.Str(value);
        }
        return "";
    }
}

[Node("ConvertDictionaryToString", Category = "text", DisplayName = "Convert Dictionary to String")]
public sealed class ConvertDictionaryToString
{
    /// <summary>The dictionary as JSON, written like Python's <c>json.dumps(..., ensure_ascii=False)</c>.</summary>
    public string Execute(DictValue dictionary, [Input(Min = 0, Max = 8, Tooltip = "Spaces per indent level. 0 produces compact single-line string.")] long indent = 2) =>
        PyJson.Dumps(dictionary, (int)indent);
}

[Node("ConvertArrayToString", Category = "text", DisplayName = "Convert Array to String")]
public sealed class ConvertArrayToString
{
    /// <summary>The array as JSON, written like Python's <c>json.dumps(..., ensure_ascii=False)</c>.</summary>
    public string Execute(ArrayValue array, [Input(Min = 0, Max = 8, Tooltip = "Spaces per indent level. 0 produces compact single-line string.")] long indent = 2) =>
        PyJson.Dumps(array, (int)indent);
}

/// <summary>
/// ComfyUI's Save Text: writes the text to a numbered file in the output folder (through the result store, like Save Image) and passes it on.
/// The json format pretty-prints valid JSON with two spaces (anything else is written as it is). The file is UTF-8 without a byte order mark.
/// </summary>
[Node("SaveText", Category = "text", DisplayName = "Save Text", Description = "Save text content to a file in the output directory.", IsOutput = true)]
public sealed class SaveText
{
    public static string[] Formats() => ["txt", "csv", "md", "json"];

    public async Task<(string Text, UiOutput Ui)> ExecuteAsync([Input(ForceInput = true)] string text, IServiceProvider services, string filenamePrefix = "ComfyUI",
        [Input(OptionsProvider = nameof(Formats))] string format = "txt")
    {
        var content = text;
        if (!Formats().Contains(format)) throw new ArgumentException($"Unsupported text format: '{format}'", nameof(format));
        var body = content;
        if (format == "json")
        {
            try { body = PyJson.Dumps(PyJson.Loads(content), 2); }
            catch (PyException e) when (e.PyType == "JSONDecodeError") { }
        }
        // ComfyUI computes the file name variables for a 1x1 "image".
        var prefix = SavePrefix.ComputeVars(filenamePrefix, 1, 1, DateTime.Now);
        var file = await Results.For(services).SaveAsync("output", prefix, new UTF8Encoding(false).GetBytes(body), format);
        var ui = new UiOutput { ["text"] = new[] { content }, ["files"] = new object[] { file } };
        return (content, ui);
    }
}
