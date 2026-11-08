using System.Text;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Text nodes over Python values: regular expressions follow Python's re module (PyRegex), JSON follows Python's json module (PyJson), and values
// turn into text as Python's str() does (PyText.Str). What the Python documentation leaves to the node (which group a mode returns, what an
// unknown mode gives, how a missing value reads) was measured by running ComfyUI.

[Node("ConvertDictionaryToString", Category = "text", DisplayName = "Convert Dictionary to String")]
public sealed class ConvertDictionaryToString
{
    /// <summary>
    /// The dictionary as JSON text, keys in their order, non-ASCII text written as it is. Indent 0 is one line with ", " and ": " between items;
    /// otherwise each item is on its own line, indented by that many spaces per level.
    /// </summary>
    public string Execute(DictValue dictionary,
        [Input(Min = 0, Max = 8, Tooltip = "Spaces per indent level. 0 produces compact single-line string.")] long indent = 2)
        => PyJson.Dumps(dictionary, (int)Math.Clamp(indent, 0, int.MaxValue));
}

[Node("ConvertArrayToString", Category = "text", DisplayName = "Convert Array to String")]
public sealed class ConvertArrayToString
{
    /// <summary>The array as JSON text, written as <see cref="ConvertDictionaryToString"/> writes a dictionary.</summary>
    public string Execute(ArrayValue array,
        [Input(Min = 0, Max = 8, Tooltip = "Spaces per indent level. 0 produces compact single-line string.")] long indent = 2)
        => PyJson.Dumps(array, (int)Math.Clamp(indent, 0, int.MaxValue));
}

[Node("JsonExtractString", Category = "text", DisplayName = "Extract Text from JSON",
    Description = "Extract a key's value from the first valid JSON object in the text, including Markdown code blocks.")]
public sealed class JsonExtractString
{
    /// <summary>
    /// The value under <paramref name="key"/> in the first JSON object found in the text. Measured with ComfyUI: the text is searched from each "{" in
    /// turn and the first one that starts a complete JSON object is used (text before or after it, a code fence, a broken object before it, are all
    /// skipped); only that object is looked at, so a key that is only in a later or a nested object gives "". The key is matched exactly (no
    /// trimming). A missing key or a null value gives ""; a string is returned as it is; anything else as Python's str() writes it
    /// ("True", "1.0", "inf", "[1, 'two', None]", "{'k': 1}").
    /// </summary>
    public string Execute([Input(Multiline = true)] string jsonString = "", string key = "")
    {
        for (int at = jsonString.IndexOf('{'); at >= 0; at = jsonString.IndexOf('{', at + 1))
        {
            if (!PyJson.TryRawDecode(jsonString, at, out var value, out _) || value is not DictValue found) continue;
            return found.TryGetValue(key, out var v) && v is not null ? PyText.Str(v) : "";
        }
        return "";
    }
}

[Node("RegexExtract", Category = "text", DisplayName = "Extract Text")]
public sealed class RegexExtract
{
    /// <summary>
    /// Text the Python regular expression finds. Measured with ComfyUI:
    /// <list type="bullet">
    /// <item>"First Match": the whole first match.</item>
    /// <item>"All Matches": one line per match: the whole match when the pattern has no groups, otherwise its first group ("" where that group took no
    /// part), as Python's <c>re.findall</c> gives them.</item>
    /// <item>"First Group": group <paramref name="groupIndex"/> of the first match (0 is the whole match); "" when there is no match, no such group,
    /// or the group took no part.</item>
    /// <item>"All Groups": one line per match with group <paramref name="groupIndex"/>; "" when the pattern has no groups or not that many. A match
    /// where the group took no part fails the run with Python's <c>str.join</c> TypeError.</item>
    /// </list>
    /// A pattern that does not compile, or a mode outside the list, gives "". A negative group fails with "IndexError: no such group".
    /// </summary>
    public string Execute([Input(Multiline = true)] string @string = "", [Input(Multiline = true)] string regexPattern = "",
        [Input(Options = ["First Match", "All Matches", "First Group", "All Groups"])] string mode = "First Match",
        bool caseInsensitive = true, bool multiline = false, bool dotall = false,
        [Input(Min = 0, Max = 100)] long groupIndex = 1)
    {
        PyRegex regex;
        try
        {
            regex = PyRegex.Compile(regexPattern, caseInsensitive, multiline, dotall);
        }
        catch (PyException)
        {
            return "";
        }

        switch (mode)
        {
            case "First Match":
                return regex.Search(@string)?.Value ?? "";
            case "All Matches":
                return string.Join("\n", regex.FindIter(@string).Select(m => regex.Groups == 0 ? m.Value : PyRegex.Group(m, 1) ?? ""));
            case "First Group":
            {
                var m = regex.Search(@string);
                if (m is null || groupIndex > regex.Groups) return "";
                return PyRegex.Group(m, GroupNumber(groupIndex)) ?? "";
            }
            case "All Groups":
            {
                if (regex.Groups == 0 || groupIndex > regex.Groups) return "";
                int g = -1;
                var lines = new List<string>();
                foreach (var m in regex.FindIter(@string))
                {
                    if (g < 0) g = GroupNumber(groupIndex);
                    lines.Add(PyRegex.Group(m, g)
                        ?? throw new PyException("TypeError", $"sequence item {lines.Count}: expected str instance, NoneType found"));
                }
                return string.Join("\n", lines);
            }
            default:
                return "";
        }
    }

    /// <summary>The group number to read; Python's <c>match.group(n)</c> refuses a negative one.</summary>
    static int GroupNumber(long groupIndex) => groupIndex < 0 ? throw new PyException("IndexError", "no such group") : (int)groupIndex;
}

[Node("SaveText", Category = "text", DisplayName = "Save Text", Description = "Save text content to a file in the output directory.", IsOutput = true,
    OutputNames = ["text"])]
public sealed class SaveText
{
    static readonly UTF8Encoding Utf8 = new(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);

    /// <summary>
    /// Writes the text to <c>prefix_00001.ext</c> (no trailing "_"; the extension is the format) in the output folder as UTF-8 without a byte order
    /// mark, shows it, and passes it on unchanged. Measured with ComfyUI:
    /// <list type="bullet">
    /// <item>"json": text that Python's <c>json.loads</c> accepts is written pretty-printed (two-space indent, non-ASCII as it is, NaN and Infinity
    /// kept); any other text is written as it is. The other formats write the text as it is.</item>
    /// <item>%width% and %height% in the prefix read as 1 (the file-name variables are computed as for a 1x1 picture); the time variables are filled.</item>
    /// <item>The node shows <c>{"text": [text], "files": [file]}</c>.</item>
    /// </list>
    /// ComfyUI on Windows writes in text mode, so every "\n" becomes "\r\n" there; this node writes the line breaks as they are (what ComfyUI writes
    /// on Linux and macOS).
    /// </summary>
    public async Task<(string Text, UiOutput Ui)> ExecuteAsync([Input(ForceInput = true)] string text, IServiceProvider services,
        string filenamePrefix = "ComfyUI", [Input(Options = ["txt", "csv", "md", "json"])] string format = "txt")
    {
        string content = text;
        if (format == "json")
        {
            try
            {
                content = PyJson.Dumps(PyJson.Loads(text), 2);
            }
            catch (PyException)
            {
                content = text;
            }
        }
        var prefix = SavePrefix.ComputeVars(filenamePrefix, 1, 1, DateTime.Now);
        var file = await Results.For(services).SaveAsync("output", prefix, Utf8.GetBytes(content), format, trailingUnderscore: false);
        var ui = new UiOutput { ["text"] = new[] { text }, ["files"] = new object[] { file } };
        return (text, ui);
    }
}
