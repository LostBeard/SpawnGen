using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries, SpawnGen's tests and fixtures, and black-box measurements of ComfyUI's
// Load Image, Load Image (as Mask), Save Image, Preview Image and Compare Images (tools/parity/make_image_node_fixtures.py).

/// <summary>Where the Load Image nodes read pictures: the input store (a folder on the desktop, memory in the browser).</summary>
public static class InputFiles
{
    static string _directory = Path.Combine(AppContext.BaseDirectory, "input");

    /// <summary>The input store the nodes read from; setting <see cref="Directory"/> replaces it with that folder.</summary>
    public static IInputStore Store { get; set; } = new DiskInputStore(() => _directory);

    /// <summary>The input folder on disk. Setting it points <see cref="Store"/> at that folder.</summary>
    public static string Directory
    {
        get => _directory;
        set
        {
            _directory = value;
            Store = new DiskInputStore(() => _directory);
        }
    }
}

/// <summary>The output pictures Load Image (from Outputs) offers, newest first, each with ComfyUI's " [output]" annotation (set by the host).</summary>
public static class OutputFiles
{
    public static Func<IReadOnlyList<string>>? Lister { get; set; }

    public static string[] List() => Lister is { } lister ? [.. lister()] : [];
}

/// <summary>Saving node results: the result store, the metadata saved pictures carry, and temporary previews.</summary>
internal static class Results
{
    /// <summary>The host's result store, or a folder store from its <see cref="OutputOptions"/> (defaults when it has none).</summary>
    public static IResultStore For(IServiceProvider? services) =>
        services?.GetService(typeof(IResultStore)) as IResultStore
        ?? new DiskResultStore(services?.GetService(typeof(OutputOptions)) as OutputOptions ?? new OutputOptions());

    /// <summary>
    /// The text a saved file carries (measured with ComfyUI: tEXt "prompt", then each extra_pnginfo entry such as "workflow", in that order); empty
    /// without a running prompt.
    /// </summary>
    public static IReadOnlyDictionary<string, string> Metadata(NodeContext? context)
    {
        var text = new Dictionary<string, string>();
        if (context is null) return text;
        text["prompt"] = context.PromptJson;
        foreach (var (key, value) in context.ExtraPngInfo) text[key] = value;
        return text;
    }

    /// <summary>
    /// Stores pictures of the batch as temporary results (the first <paramref name="count"/> when given) and returns their entries for a node's
    /// ui. Measured with ComfyUI: the files are named <c>{prefix}_temp_{five random letters}_00001_.png</c>, counting up; the previews of its newer
    /// nodes (Compare Images, Preview Mask, the compositor) carry no PNG text, only Preview Image's carry the prompt.
    /// </summary>
    public static Task<List<object>> PreviewAsync(ImageBatch images, NodeContext? context, IServiceProvider services, int? count = null, string prefix = "ComfyUI") =>
        SaveTempAsync(images, null, services, $"{prefix}_temp_{RandomLetters()}", count);

    internal static async Task<List<object>> SaveTempAsync(ImageBatch images, IReadOnlyDictionary<string, string>? metadata, IServiceProvider services, string prefix, int? count = null)
    {
        var store = For(services);
        var files = new List<object>();
        for (int i = 0; i < Math.Min(images.Batch, count ?? images.Batch); i++)
            files.Add((await store.SaveAsync("temp", prefix, PngWriter.Encode(images, i, metadata))).AsImage(images.Width, images.Height));
        return files;
    }

    internal static string RandomLetters() => string.Create(5, 0, (span, _) =>
    {
        for (int i = 0; i < span.Length; i++) span[i] = (char)('a' + Random.Shared.Next(26));
    });

    /// <summary>
    /// Saves each picture of the batch as <c>{prefix}_00001_.png</c>, counting up, with the prompt's metadata. Measured with ComfyUI: "%batch_num%"
    /// in the prefix becomes the picture's index, while the counter is the one for the prefix as written (so such names start at 00001 on every run,
    /// replacing the earlier files) and goes up by one per picture: bn_0_00001_, bn_1_00002_, ...
    /// </summary>
    internal static async Task<List<object>> SaveImagesAsync(ImageBatch images, NodeContext? context, IServiceProvider services, string type, string prefix)
    {
        var store = For(services);
        var metadata = Metadata(context);
        var files = new List<object>();
        if (!prefix.Contains("%batch_num%", StringComparison.Ordinal))
        {
            for (int i = 0; i < images.Batch; i++)
                files.Add((await store.SaveAsync(type, prefix, PngWriter.Encode(images, i, metadata))).AsImage(images.Width, images.Height));
            return files;
        }
        var (folder, name) = SavePrefix.Split(prefix);
        store.CreateFolder(type, folder);
        var counter = SavePrefix.NextCounter(store.List(type, folder) ?? [], name);
        for (int i = 0; i < images.Batch; i++)
        {
            var file = SavePrefix.FileName(name.Replace("%batch_num%", i.ToString(System.Globalization.CultureInfo.InvariantCulture), StringComparison.Ordinal), counter + i, "png");
            files.Add((await store.WriteAsync(type, folder, file, PngWriter.Encode(images, i, metadata))).AsImage(images.Width, images.Height));
        }
        return files;
    }
}

/// <summary>Shared by the Load Image nodes: reading a picture into an IMAGE and a MASK.</summary>
static class Pictures
{
    /// <summary>
    /// Every frame as IMAGE, and MASK = 1 - alpha per frame. Measured with ComfyUI: a picture without alpha gets an empty 64 x 64 mask per frame
    /// (an RGB APNG of 4 frames: 4 x 64 x 64).
    /// </summary>
    public static async Task<(ImageBatch Image, MaskBatch Mask)> LoadAsync(byte[] data)
    {
        var (images, alpha) = await ImageDecoder.DecodeFramesAsync(data);
        return (images, MaskFromAlpha(alpha, images.Batch, images.Height, images.Width));
    }

    public static MaskBatch MaskFromAlpha(float[]? alpha, int frames, int height, int width)
    {
        if (alpha is null) return new MaskBatch(frames, 64, 64);
        var mask = new MaskBatch(frames, height, width);
        for (int i = 0; i < alpha.Length; i++) mask.Data[i] = 1f - alpha[i];
        return mask;
    }

    /// <summary>The name without ComfyUI's " [input]" / " [output]" / " [temp]" annotation, and that annotation's folder type (null without one).</summary>
    public static (string Name, string? Type) Annotated(string name)
    {
        name = name.Trim();
        foreach (var type in (string[])["input", "output", "temp"])
            if (name.EndsWith($" [{type}]", StringComparison.Ordinal)) return (name[..^(type.Length + 3)].Trim(), type);
        return (name, null);
    }
}

/// <summary>Load Image: a picture from the input folder; every frame of an animated one, and its transparency as a mask (1 = transparent).</summary>
[Node(Category = "image", DisplayName = "Load Image", OutputNames = ["IMAGE", "MASK"])]
public sealed class LoadImage
{
    public static string[] ImageNames() => [.. InputFiles.Store.List(ImageDecoder.IsSupportedName)];

    public async Task<(ImageBatch Image, MaskBatch Mask)> ExecuteAsync([Input(OptionsProvider = nameof(ImageNames))] string image) =>
        await Pictures.LoadAsync(await InputFiles.Store.ReadAsync(Pictures.Annotated(image).Name));
}

/// <summary>
/// Load Image (from Outputs): a picture from the results ("name.png [output]", "sub/name.png [output]", or " [temp]"); a name without an annotation
/// (or with " [input]") is read from the input folder.
/// </summary>
[Node(Category = "image", DisplayName = "Load Image (from Outputs)", OutputNames = ["IMAGE", "MASK"],
    Description = "Load an image from the output folder. When the refresh button is clicked, the node will update the image list and automatically select the first image, allowing for easy iteration.")]
public sealed class LoadImageOutput
{
    public static string[] OutputImages() => OutputFiles.List();

    public async Task<(ImageBatch Image, MaskBatch Mask)> ExecuteAsync([Input(OptionsProvider = nameof(OutputImages))] string image, IServiceProvider services)
    {
        var (name, type) = Pictures.Annotated(image);
        if (type is null or "input") return await Pictures.LoadAsync(await InputFiles.Store.ReadAsync(name));
        name = name.Replace('\\', '/');
        int slash = name.LastIndexOf('/');
        var data = await Results.For(services).ReadAsync(type, slash < 0 ? "" : name[..slash], name[(slash + 1)..])
            ?? throw new FileNotFoundException($"There is no {type} picture '{name}'.", name);
        return await Pictures.LoadAsync(data);
    }
}

/// <summary>
/// Load Image (as Mask): one channel of every frame as a mask. Measured with ComfyUI: "alpha" gives 1 - alpha (an empty 64 x 64 mask per frame when
/// the picture has no alpha; a palette picture always has it), "red" / "green" / "blue" give the channel's value as it is.
/// </summary>
[Node(Category = "image", DisplayName = "Load Image (as Mask)")]
public sealed class LoadImageMask
{
    public static string[] ImageNames() => LoadImage.ImageNames();

    public async Task<MaskBatch> ExecuteAsync([Input(OptionsProvider = nameof(ImageNames))] string image,
        [Input(Options = ["alpha", "red", "green", "blue"])] string channel = "alpha")
    {
        var (picture, alpha) = await ImageDecoder.DecodeFramesAsync(await InputFiles.Store.ReadAsync(Pictures.Annotated(image).Name));
        if (channel == "alpha") return Pictures.MaskFromAlpha(alpha, picture.Batch, picture.Height, picture.Width);
        int c = channel switch
        {
            "red" => 0,
            "green" => 1,
            "blue" => 2,
            _ => throw new ArgumentException($"Unknown channel '{channel}': use alpha, red, green or blue.", nameof(channel)),
        };
        var mask = new MaskBatch(picture.Batch, picture.Height, picture.Width);
        for (int i = 0; i < mask.Data.Length; i++) mask.Data[i] = picture.Data[i * picture.Channels + c];
        return mask;
    }
}

/// <summary>
/// Save Image: each picture as a PNG in the output folder, named <c>{prefix}_00001_.png</c> and counting up (a "folder/" part of the prefix is a
/// subfolder; %width%, %height% and %date:...% are filled in), with the prompt and the workflow as PNG text. The pictures are passed on.
/// </summary>
[Node(Category = "image", DisplayName = "Save Image", IsOutput = true, Description = "Saves the input images to your ComfyUI output directory.")]
public sealed class SaveImage
{
    public async Task<(ImageBatch Images, UiOutput Ui)> ExecuteAsync(
        [Input(Tooltip = "The images to save.")] ImageBatch images, NodeContext context, IServiceProvider services,
        [Input(Tooltip = "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.")]
        string filenamePrefix = "ComfyUI")
    {
        var prefix = SavePrefix.ComputeVars(filenamePrefix, images.Width, images.Height, DateTime.Now);
        return (images, new UiOutput("images", await Results.SaveImagesAsync(images, context, services, "output", prefix)));
    }
}

/// <summary>Preview Image: each picture as a temporary PNG shown in the node; nothing is kept in the output folder. The pictures are passed on.</summary>
[Node(Category = "image", DisplayName = "Preview Image", IsOutput = true, Description = "Preview the images without saving them to the ComfyUI output directory.")]
public sealed class PreviewImage
{
    public async Task<(ImageBatch Images, UiOutput Ui)> ExecuteAsync(ImageBatch images, NodeContext context, IServiceProvider services) =>
        (images, new UiOutput("images", await Results.SaveTempAsync(images, Results.Metadata(context), services, $"ComfyUI_temp_{Results.RandomLetters()}")));
}

/// <summary>
/// Compare Images: both sides as temporary pictures for the editor's slider (ui "a_images" and "b_images"; a side that is not connected is empty).
/// Measured with ComfyUI: the files are <c>comfy.compare.a_temp_{letters}_00001_.png</c> and <c>comfy.compare.b_temp_{letters}_...</c>, the same letters
/// on both sides.
/// </summary>
[Node(Category = "image", DisplayName = "Compare Images", IsOutput = true, Description = "Compares two images side by side with a slider.")]
public sealed class ImageCompare
{
    public async Task<UiOutput> ExecuteAsync(NodeContext context, IServiceProvider services,
        [Input(Type = "IMAGECOMPARE", Json = "{}")] object? compareView, ImageBatch? imageA = null, ImageBatch? imageB = null)
    {
        var letters = Results.RandomLetters();
        var ui = new UiOutput();
        ui["a_images"] = imageA is null ? new List<object>() : await Results.SaveTempAsync(imageA, null, services, $"comfy.compare.a_temp_{letters}");
        ui["b_images"] = imageB is null ? new List<object>() : await Results.SaveTempAsync(imageB, null, services, $"comfy.compare.b_temp_{letters}");
        return ui;
    }
}
