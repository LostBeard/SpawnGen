using System.Globalization;
using System.Text;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries and SpawnGen's tests (ComfyImageNodeTests: the chunk layout of Save
// Animated PNG, the metadata block of Save SVG). The APNG itself comes from PngWriter.EncodeAnimated (Pillow's rules, Pillow is MIT-CMU).

/// <summary>Metadata text as Python's json module writes it with its default ensure_ascii: every non-ASCII UTF-16 unit as \uXXXX.</summary>
static class AsciiJson
{
    public static string Escape(string json)
    {
        var sb = new StringBuilder(json.Length);
        foreach (var c in json)
            if (c < 0x80) sb.Append(c);
            else sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
        return sb.ToString();
    }
}

/// <summary>
/// Save Animated PNG: the batch as one APNG in the output folder, each picture shown for <c>int(1000 / fps)</c> ms, with the prompt and the
/// workflow in private "comf" chunks after the image data (keyword, a zero byte, the JSON with non-ASCII escaped). A batch whose pictures are all
/// alike is a still PNG, and the ui then has no "animated" entry.
/// </summary>
[Node("SaveAnimatedPNG", Category = "image", DisplayName = "Save Animated PNG", IsOutput = true)]
public sealed class SaveAnimatedPNG
{
    public async Task<(ImageBatch Images, UiOutput Ui)> ExecuteAsync(ImageBatch images, NodeContext context, IServiceProvider services,
        string filenamePrefix = "ComfyUI",
        [Input(Min = 0.01, Max = 1000.0, Step = 0.01)] double fps = 6.0,
        [Input(Min = 0, Max = 9)] long compressLevel = 4)
    {
        var chunks = Results.Metadata(context).Select(kv => ("comf", Encoding.Latin1.GetBytes($"{kv.Key}\0{AsciiJson.Escape(kv.Value)}"))).ToList();
        var png = PngWriter.EncodeAnimated(images, (int)(1000.0 / fps), (int)compressLevel, chunks);
        var prefix = SavePrefix.ComputeVars(filenamePrefix, images.Width, images.Height, DateTime.Now);
        var file = (await Results.For(services).SaveAsync("output", prefix, png)).AsImage(images.Width, images.Height);
        var ui = new UiOutput("images", new object[] { file });
        if (IsAnimated(png)) ui["animated"] = new object[] { true };
        return (images, ui);
    }

    /// <summary>Whether the chunk after IHDR is acTL (an APNG rather than a still PNG).</summary>
    static bool IsAnimated(byte[] png) => png.Length >= 41 && Encoding.ASCII.GetString(png, 37, 4) == "acTL";
}

/// <summary>SVG pictures, each the file's bytes (ComfyUI's SVG type).</summary>
[Socket("SVG")]
public sealed class SvgImages(IReadOnlyList<byte[]> data)
{
    public IReadOnlyList<byte[]> Data { get; } = data;
}

/// <summary>
/// Save SVG: each SVG as a file in the output folder, named like Save Image's files with ".svg" ("%batch_num%" in the prefix becomes the
/// picture's index, each name counted on its own). The prompt and the extra metadata (the workflow) go into a <c>&lt;metadata&gt;</c> CDATA block
/// right after the opening svg tag, as JSON indented by 2 with non-ASCII escaped (the layout SpawnGen's test pins).
/// </summary>
[Node("SaveSVGNode", Category = "image", DisplayName = "Save SVG", IsOutput = true, Description = "Save SVG files on disk.")]
public sealed class SaveSVGNode
{
    public async Task<(SvgImages Svg, UiOutput Ui)> ExecuteAsync(SvgImages svg, NodeContext context, IServiceProvider services,
        [Input(Tooltip = "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.")]
        string filenamePrefix = "svg/ComfyUI")
    {
        var store = Results.For(services);
        var block = MetadataBlock(context);
        var files = new List<object>();
        for (int i = 0; i < svg.Data.Count; i++)
        {
            var prefix = filenamePrefix.Replace("%batch_num%", i.ToString(CultureInfo.InvariantCulture), StringComparison.Ordinal);
            files.Add(await store.SaveAsync("output", prefix, WithMetadata(svg.Data[i], block), "svg"));
        }
        return (svg, new UiOutput("images", files));
    }

    static string? MetadataBlock(NodeContext? context)
    {
        var metadata = Results.Metadata(context);
        if (metadata.Count == 0) return null;
        var all = new DictValue();
        foreach (var (key, json) in metadata) all[key] = PyJson.Loads(json);
        return "\n  <metadata>\n                <![CDATA[\n            " + AsciiJson.Escape(PyJson.Dumps(all, 2)) +
               "\n                ]]>\n            </metadata>\n            ";
    }

    /// <summary>The SVG with the block inserted after the first "&lt;svg ...&gt;" tag; unchanged without a block or such a tag.</summary>
    static byte[] WithMetadata(byte[] svg, string? block)
    {
        if (block is null) return svg;
        var text = Encoding.UTF8.GetString(svg);
        int start = text.IndexOf("<svg", StringComparison.Ordinal);
        int end = start < 0 ? -1 : text.IndexOf('>', start);
        return end < 0 ? svg : Encoding.UTF8.GetBytes(text[..(end + 1)] + block + text[(end + 1)..]);
    }
}

/// <summary>Bounding Box: a BOUNDING_BOX value, the dict {"x", "y", "width", "height"} the bounding box nodes read.</summary>
[Node("PrimitiveBoundingBox", Category = "utilities/primitive", DisplayName = "Bounding Box", OutputNames = ["BOUNDING_BOX"], OutputTypes = ["BOUNDING_BOX"])]
public sealed class PrimitiveBoundingBox
{
    public object Execute([Input(Min = 0, Max = 16384)] long x = 0, [Input(Min = 0, Max = 16384)] long y = 0,
        [Input(Min = 1, Max = 16384)] long width = 512, [Input(Min = 1, Max = 16384)] long height = 512) =>
        new DictValue { ["x"] = x, ["y"] = y, ["width"] = width, ["height"] = height };
}
