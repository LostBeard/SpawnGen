using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>SVG documents (ComfyUI's SVG type: one file per entry), as produced by vector nodes.</summary>
[Socket("SVG")]
public sealed record SvgImages(IReadOnlyList<byte[]> Data);

/// <summary>A rectangle in pixels (ComfyUI's BOUNDING_BOX: x, y, width, height).</summary>
[Socket("BOUNDING_BOX")]
public sealed record BoundingBox(int X, int Y, int Width, int Height);

[Node(Category = "utilities/primitive", DisplayName = "Bounding Box")]
public sealed class PrimitiveBoundingBox
{
    public BoundingBox Execute([Input(Min = 0, Max = 16384)] int x = 0, [Input(Min = 0, Max = 16384)] int y = 0,
        [Input(Min = 1, Max = 16384)] int width = 512, [Input(Min = 1, Max = 16384)] int height = 512) => new(x, y, width, height);
}

internal static class ComfyJson
{
    /// <summary>Python's <c>json.dumps</c> default <c>ensure_ascii</c>: every character above '~' written as <c>\uXXXX</c> (they can only be inside strings).</summary>
    public static string EnsureAscii(string json)
    {
        if (json.All(c => c <= '~')) return json;
        var sb = new StringBuilder(json.Length + 16);
        foreach (var c in json)
            if (c <= '~') sb.Append(c);
            else sb.Append("\\u").Append(((int)c).ToString("x4"));
        return sb.ToString();
    }
}

[Node(Category = "image", DisplayName = "Save Animated PNG", IsOutput = true)]
public sealed class SaveAnimatedPNG
{
    /// <summary>
    /// The whole batch as one animated PNG (<c>prefix_00001_.png</c>) at <paramref name="fps"/> (each frame lasts <c>int(1000 / fps)</c> ms), looping.
    /// Like ComfyUI the prompt and workflow go into private <c>comf</c> chunks after the image data. The pictures are passed on.
    /// </summary>
    public async Task<(ImageBatch Images, UiOutput Ui)> ExecuteAsync(ImageBatch images, NodeContext context, IServiceProvider services,
        string filenamePrefix = "ComfyUI",
        [Input(Min = 0.01, Max = 1000, Step = 0.01)] double fps = 6.0,
        [Input(Min = 0, Max = 9)] int compressLevel = 4)
    {
        var chunks = Results.Metadata(context)
            .Select(kv => ("comf", (byte[])[.. Encoding.Latin1.GetBytes(kv.Key), 0, .. Encoding.Latin1.GetBytes(ComfyJson.EnsureAscii(kv.Value))]))
            .ToList();
        var png = PngWriter.EncodeAnimated(images, (int)(1000.0 / fps), compressLevel, chunks);
        var prefix = SavePrefix.ComputeVars(filenamePrefix, images.Width, images.Height, DateTime.Now);
        var file = await Results.For(services).SaveAsync("output", prefix, png);
        var ui = new UiOutput("images", new object[] { file });
        if (images.Batch > 1) ui["animated"] = new object[] { true };
        return (images, ui);
    }
}

[Node(Category = "image", DisplayName = "Save SVG", Description = "Save SVG files on disk.", IsOutput = true)]
public sealed class SaveSVGNode
{
    /// <summary>
    /// Writes each SVG (<c>prefix_00001_.svg</c>, <c>%batch_num%</c> in the prefix becomes its index) with ComfyUI's <c>&lt;metadata&gt;</c> block (prompt and
    /// workflow as indented JSON in CDATA) after every <c>&lt;svg ...&gt;</c> tag, and passes the SVGs on.
    /// </summary>
    public async Task<(SvgImages Svg, UiOutput Ui)> ExecuteAsync(SvgImages svg, NodeContext context, IServiceProvider services,
        [Input(Tooltip = "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.")]
        string filenamePrefix = "svg/ComfyUI")
    {
        var metadata = new JsonObject();
        foreach (var (key, value) in Results.Metadata(context))
        {
            JsonNode? node;
            try { node = JsonNode.Parse(value); } catch (JsonException) { node = JsonValue.Create(value); }
            metadata[key] = node;
        }
        string json = ComfyJson.EnsureAscii(metadata.ToJsonString(new JsonSerializerOptions
        {
            WriteIndented = true, IndentSize = 2, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        }).ReplaceLineEndings("\n"));
        // ComfyUI's block, whitespace included.
        string element = "  <metadata>\n                <![CDATA[\n            " + json + "\n                ]]>\n            </metadata>\n            ";

        var store = Results.For(services);
        var prefix = SavePrefix.ComputeVars(filenamePrefix, 0, 0, DateTime.Now);
        var saved = new List<object>();
        for (int i = 0; i < svg.Data.Count; i++)
        {
            var content = Encoding.UTF8.GetString(svg.Data[i]);
            if (metadata.Count > 0) content = Regex.Replace(content, "(<svg[^>]*>)", m => m.Groups[1].Value + "\n" + element);
            saved.Add(await store.SaveAsync("output", prefix.Replace("%batch_num%", i.ToString(System.Globalization.CultureInfo.InvariantCulture)), Encoding.UTF8.GetBytes(content), "svg"));
        }
        return (svg, new UiOutput("images", saved));
    }
}
