using SpawnGen.Engine.Nodes;

namespace SpawnGen.Editor;

/// <summary>
/// ComfyUI's upload button: a file combo with an upload flag (<see cref="WidgetSpec.Upload"/>: Load Image, Load Audio, Load Video ...) gets a
/// "choose file to upload" row right under it. Pressing it picks a file, uploads it to the engine and selects it in the combo. It is not an input:
/// no slot, no link, and it is not sent with the prompt.
/// </summary>
public static class UploadWidget
{
    /// <summary>The row's name in the node's widget rows (ComfyUI's frontend names the button widget "upload").</summary>
    public const string Name = "upload";

    /// <summary>The button's text, as ComfyUI's frontend shows it.</summary>
    public const string Label = "choose file to upload";

    /// <summary>The row's spec: a button, so nothing to edit; the combo it serves is found with <see cref="ComboFor"/>.</summary>
    public static readonly WidgetSpec Spec = new(WidgetKind.String) { Hidden = true };

    /// <summary>True for an input whose combo gets the button.</summary>
    public static bool HasUpload(InputSpec input) => input.Widget is { Kind: WidgetKind.Combo, Upload: not null };

    /// <summary>The input the node's upload button serves (its first file combo with an upload flag), or null when it has none.</summary>
    public static InputSpec? ComboFor(NodeSchema schema) => schema.Inputs.FirstOrDefault(HasUpload);

    /// <summary>
    /// The upload button's value ComfyUI's frontend saves last in widgets_values (measured on frontend 1.61.3: Load Image ["example.png", "image"],
    /// Load Image (as Mask) [name, "alpha", "image"], Load Video [name, "image"], Load Audio [name, ""]); reading skips it, as it comes after the
    /// last widget. Null when nothing is saved: a node without the button, and Load Image (from Outputs), whose saved list also holds an
    /// auto-refresh toggle and a refresh button SpawnGen does not have, so a value here would land in the toggle's place.
    /// </summary>
    public static string? SavedValue(NodeSchema schema) => ComboFor(schema)?.Widget switch
    {
        { UploadFolder: not (null or "input") } => null,
        { Upload: "image" or "video" } => "image",
        { Upload: "audio" } => "",
        _ => null,
    };

    /// <summary>The files the picker offers for an upload kind (what SpawnGen can load: the combo's own lists filter the same way).</summary>
    public static string Accept(string? kind) => kind switch
    {
        "image" => ".png,.jpg,.jpeg,.webp,.gif,.bmp",
        "audio" => ".wav,.flac",
        "video" => ".3gp,.avi,.m4v,.mkv,.mov,.mp4,.mpeg,.mpg,.ogv,.webm,.wmv",
        _ => "",
    };

    /// <summary>
    /// The combo value for a file uploaded under <paramref name="stored"/> into the combo's folder (<see cref="WidgetSpec.UploadFolder"/>): the name
    /// itself for the input folder, "name [output]" for a combo that lists the output folder (Load Image (from Outputs) lists its files that way).
    /// </summary>
    public static string ValueFor(WidgetSpec combo, string stored) => combo.UploadFolder is null or "input" ? stored : $"{stored} [{combo.UploadFolder}]";
}
