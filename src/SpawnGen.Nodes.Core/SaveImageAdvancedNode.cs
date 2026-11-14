using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// Save Image (Advanced): the pictures as PNG (8 or 16 bits per sample), OpenEXR (half or float, from sRGB, HLG or linear input) or AVIF.
/// Measured with ComfyUI (2026-11-13, Fixtures/image_save_advanced): one file per picture named <c>{prefix}_00001.png</c> (no "_" before the dot,
/// counting on from Save Image's files of the same prefix and back), the prompt and workflow inside each file (PNG text, EXR string attributes),
/// a pHYs chunk in the PNGs, the pictures passed on. AVIF needs an AV1 still-picture writer that SpawnDev.Codecs does not have yet.
/// </summary>
[Node("SaveImageAdvanced", Category = "image", DisplayName = "Save Image (Advanced)", IsOutput = true,
    Description = "Saves the input images to your ComfyUI output directory.", OutputNames = ["images"])]
public sealed class SaveImageAdvanced
{
    const string ExrSpaceTip = "Colorspace of the input tensor. The EXR is always written as scene-linear in the matching gamut.\n" +
        "sRGB — input is sRGB-encoded Rec.709; the inverse sRGB EOTF is applied.\n" +
        "HDR — input is HLG-encoded Rec.2020 (BT.2100); the inverse HLG OETF is applied to get scene-linear light.\n" +
        "linear — input is already scene-linear (Rec.709 primaries); written through unchanged. Use this for renderer/compositor output.";

    public static IEnumerable<DynamicOption> Formats() =>
    [
        new("png", DynamicInput.Combo("bit_depth", ["8-bit", "16-bit"], "8-bit"), DynamicInput.Combo("input_color_space", ["sRGB"], "sRGB")),
        new("exr", DynamicInput.Combo("bit_depth", ["32-bit float", "16-bit float"], "16-bit float"),
            DynamicInput.Combo("input_color_space", ["sRGB", "HDR", "linear"], "sRGB", ExrSpaceTip)),
        new("avif",
            DynamicInput.Combo("bit_depth", ["auto", "8-bit YUV420", "10-bit YUV420"], "auto", "Auto uses 8-bit YUV420 for sRGB and 10-bit YUV420 for HDR."),
            DynamicInput.Combo("input_color_space", ["sRGB", "HDR", "HDR PQ"], "sRGB",
                "Colorspace of the input images. HDR selects BT.2020/HLG and HDR PQ selects BT.2020/PQ."),
            DynamicInput.Int("crf", 18, 1, 63, tooltip: "Lower values produce higher quality and larger files."),
            DynamicInput.Nested("save_mode",
            [
                new DynamicOption("still images"),
                new DynamicOption("animated", DynamicInput.Float("fps", 6.0, 0.01, 1000.0, 0.01),
                    DynamicInput.Int("loop_count", 0, 0, 1000, tooltip: "Number of times to loop the animation. 0 loops forever.")),
            ])),
    ];

    public async Task<(ImageBatch Images, UiOutput Ui)> ExecuteAsync(
        [Input(Tooltip = "The images to save.")] ImageBatch images, NodeContext context, IServiceProvider services,
        [Input(Tooltip = "The prefix for the file to save. May include formatting tokens such as %date:yyyy-MM-dd% or %Empty Latent Image.width%.")]
        string filenamePrefix = "ComfyUI",
        [DynamicCombo(nameof(Formats)), Input(Tooltip = "The file format in which to save the image.")] DynamicChoice? format = null)
    {
        var prefix = SavePrefix.ComputeVars(filenamePrefix, images.Width, images.Height, DateTime.Now);
        var metadata = Results.Metadata(context);
        string key = format?.Key ?? "png";
        List<object> files;
        switch (key)
        {
            case "png":
                int bits = format?.Get("bit_depth", "8-bit") == "16-bit" ? 16 : 8;
                files = await Results.SaveEachAsync(images, services, "output", prefix, i => PngWriter.Encode(images, i, metadata, bits, aspectChunk: true),
                    "png", trailingUnderscore: false);
                break;
            case "exr":
                bool half = format?.Get("bit_depth", "16-bit float") != "32-bit float";
                var input = format?.Get("input_color_space", "sRGB") switch
                {
                    "HDR" => ExrInput.Hlg,
                    "linear" => ExrInput.Linear,
                    "sRGB" => ExrInput.Srgb,
                    var other => throw new ArgumentException($"Unknown input_color_space '{other}' for EXR."),
                };
                files = await Results.SaveEachAsync(images, services, "output", prefix, i => ExrWriter.Encode(images, i, input, half, metadata),
                    "exr", trailingUnderscore: false);
                break;
            case "avif":
                throw new NotSupportedException("AVIF is not available yet: it needs an AV1 still-picture (AVIF) writer in SpawnDev.Codecs. Choose png or exr.");
            default:
                throw new ArgumentException($"Unknown format '{key}'; Save Image (Advanced) writes png, exr or avif.");
        }
        return (images, new UiOutput("images", files));
    }
}
