using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>Where <see cref="LoadImage"/> reads from. Hosts set <see cref="Store"/> (folder on the desktop, memory in the browser); the combo lists what it holds.</summary>
public static class InputFiles
{
    /// <summary>The input folder used by the default disk store.</summary>
    public static string Directory { get; set; } = Path.Combine(AppContext.BaseDirectory, "input");

    public static IInputStore Store { get; set; } = new DiskInputStore(() => Directory);

    public static string[] List() => [.. Store.List()];
}

internal static class Results
{
    /// <summary>What goes into a saved picture's text chunks, like ComfyUI: <c>prompt</c> (the API graph that made it) plus the editor's extras (<c>workflow</c>).</summary>
    public static IReadOnlyDictionary<string, string> Metadata(NodeContext context)
    {
        var text = new Dictionary<string, string> { ["prompt"] = context.PromptJson };
        foreach (var (key, value) in context.ExtraPngInfo) text[key] = value;
        return text;
    }

    /// <summary>
    /// The first <paramref name="count"/> pictures stored as temporary previews (ComfyUI's <c>UI.PreviewImage</c>), for nodes that show their result. An empty
    /// picture cannot be written, as Pillow refuses it.
    /// </summary>
    public static async Task<List<object>> PreviewAsync(ImageBatch images, NodeContext context, IServiceProvider services, int count = int.MaxValue)
    {
        if (images.Height == 0 || images.Width == 0) throw new ArgumentException("cannot write empty image", nameof(images));
        var store = For(services);
        var saved = new List<object>();
        for (int i = 0; i < Math.Min(count, images.Batch); i++)
            saved.Add(await store.SaveAsync("temp", "preview", PngWriter.Encode(images, i, Metadata(context))));
        return saved;
    }

    /// <summary>The result store a host registered, or a disk store over the output options (so the nodes also work with no host setup).</summary>
    public static IResultStore For(IServiceProvider services) =>
        services.GetService(typeof(IResultStore)) as IResultStore
        ?? new DiskResultStore(services.GetService(typeof(OutputOptions)) as OutputOptions ?? new OutputOptions());
}

[Node(Category = "image", DisplayName = "Load Image", OutputNames = ["IMAGE", "MASK"])]
public sealed class LoadImage
{
    public static string[] ImageNames() => InputFiles.List();

    /// <summary>
    /// The picture, and a mask from its transparency like ComfyUI's: 1 where the picture is transparent, 0 where it is opaque. Every frame of an animated
    /// WebP or GIF is one picture of the batch. A picture without transparency gets ComfyUI's empty mask: 64 x 64 zeros per picture.
    /// </summary>
    public async Task<(ImageBatch Image, MaskBatch Mask)> ExecuteAsync([Input(OptionsProvider = nameof(ImageNames))] string image) =>
        await LoadAsync(await InputFiles.Store.ReadAsync(image));

    internal static async Task<(ImageBatch Image, MaskBatch Mask)> LoadAsync(byte[] file)
    {
        var (img, alpha) = await ImageDecoder.DecodeFramesAsync(file);
        if (alpha is null) return (img, new MaskBatch(img.Batch, 64, 64));
        var mask = new MaskBatch(img.Batch, img.Height, img.Width);
        for (int i = 0; i < alpha.Length; i++) mask.Data[i] = 1f - alpha[i];
        return (img, mask);
    }
}

[Node(Category = "image", DisplayName = "Load Image (as Mask)")]
public sealed class LoadImageMask
{
    public static string[] ImageNames() => InputFiles.List();
    public static string[] Channels() => ["alpha", "red", "green", "blue"];

    /// <summary>ComfyUI's: Load Image's mask for the alpha channel (transparent = 1; 64 x 64 zeros without transparency), else that colour channel, per frame.</summary>
    public async Task<MaskBatch> ExecuteAsync([Input(OptionsProvider = nameof(ImageNames))] string image,
        [Input(OptionsProvider = nameof(Channels))] string channel = "alpha")
    {
        var (img, alphaMask) = await LoadImage.LoadAsync(await InputFiles.Store.ReadAsync(image));
        int c = channel switch { "red" => 0, "green" => 1, "blue" => 2, "alpha" => -1, _ => throw new ArgumentException($"Unknown channel '{channel}'.", nameof(channel)) };
        if (c < 0) return alphaMask;
        var mask = new MaskBatch(img.Batch, img.Height, img.Width);
        for (int i = 0; i < mask.Data.Length; i++) mask.Data[i] = img.Data[i * 3 + c];
        return mask;
    }
}

[Node(Category = "image", DisplayName = "Save Image", IsOutput = true)]
public sealed class SaveImage
{
    /// <summary>Writes every picture as a PNG with the workflow inside, and passes the pictures on (ComfyUI's node has the same output).</summary>
    public async Task<(ImageBatch Images, UiOutput Ui)> ExecuteAsync([Input(Tooltip = "The images to save.")] ImageBatch images, NodeContext context, IServiceProvider services,
        [Input(Tooltip = "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.")] string filenamePrefix = "ComfyUI")
    {
        var store = Results.For(services);
        var saved = new List<object>();
        filenamePrefix = SavePrefix.ComputeVars(filenamePrefix, images.Width, images.Height, DateTime.Now);
        for (int i = 0; i < images.Batch; i++)
            saved.Add(await store.SaveAsync("output", filenamePrefix, PngWriter.Encode(images, i, Results.Metadata(context))));
        return (images, new UiOutput("images", saved));
    }
}

/// <summary>Shows images in the editor: stores them as temporary files (like ComfyUI's PreviewImage); they are not kept as results.</summary>
[Node(Category = "image", DisplayName = "Preview Image", IsOutput = true)]
public sealed class PreviewImage
{
    public async Task<(ImageBatch Images, UiOutput Ui)> ExecuteAsync(ImageBatch images, NodeContext context, IServiceProvider services)
    {
        var store = Results.For(services);
        var saved = new List<object>();
        for (int i = 0; i < images.Batch; i++)
            saved.Add(await store.SaveAsync("temp", "preview", PngWriter.Encode(images, i, Results.Metadata(context))));
        return (images, new UiOutput("images", saved));
    }
}
