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

    /// <summary>The result store a host registered, or a disk store over the output options (so the nodes also work with no host setup).</summary>
    public static IResultStore For(IServiceProvider services) =>
        services.GetService(typeof(IResultStore)) as IResultStore
        ?? new DiskResultStore(services.GetService(typeof(OutputOptions)) as OutputOptions ?? new OutputOptions());
}

[Node(Category = "image", DisplayName = "Load Image")]
public sealed class LoadImage
{
    public static string[] ImageNames() => InputFiles.List();

    /// <summary>The picture, and a mask from its transparency like ComfyUI's: 1 where the picture is transparent, 0 where it is opaque (all 0 for a picture without alpha).</summary>
    public async Task<(ImageBatch Image, MaskBatch Mask)> ExecuteAsync([Input(OptionsProvider = nameof(ImageNames))] string image)
    {
        var img = ImageDecoder.Decode(await InputFiles.Store.ReadAsync(image), out var alpha);
        var mask = new MaskBatch(1, img.Height, img.Width);
        if (alpha is not null) for (int i = 0; i < alpha.Length; i++) mask.Data[i] = 1f - alpha[i];
        return (img, mask);
    }
}

[Node(Category = "mask", DisplayName = "Load Image (as Mask)")]
public sealed class LoadImageMask
{
    public static string[] ImageNames() => InputFiles.List();
    public static string[] Channels() => ["alpha", "red", "green", "blue"];

    /// <summary>A mask from one channel of a picture; the alpha channel is inverted (transparent = 1) like ComfyUI's.</summary>
    public async Task<MaskBatch> ExecuteAsync([Input(OptionsProvider = nameof(ImageNames))] string image,
        [Input(OptionsProvider = nameof(Channels))] string channel = "alpha")
    {
        var img = ImageDecoder.Decode(await InputFiles.Store.ReadAsync(image), out var alpha);
        var mask = new MaskBatch(1, img.Height, img.Width);
        int c = channel switch { "red" => 0, "green" => 1, "blue" => 2, "alpha" => -1, _ => throw new ArgumentException($"Unknown channel '{channel}'.", nameof(channel)) };
        for (int i = 0; i < mask.Data.Length; i++)
            mask.Data[i] = c >= 0 ? img.Data[i * 3 + c] : alpha is null ? 0f : 1f - alpha[i];
        return mask;
    }
}

[Node(Category = "image", DisplayName = "Save Image", IsOutput = true)]
public sealed class SaveImage
{
    public async Task<UiOutput> ExecuteAsync(ImageBatch images, NodeContext context, IServiceProvider services, string filenamePrefix = "ComfyUI")
    {
        var store = Results.For(services);
        var saved = new List<object>();
        for (int i = 0; i < images.Batch; i++)
            saved.Add(await store.SaveAsync("output", filenamePrefix, PngWriter.Encode(images, i, Results.Metadata(context))));
        return new UiOutput("images", saved);
    }
}

/// <summary>Shows images in the editor: stores them as temporary files (like ComfyUI's PreviewImage); they are not kept as results.</summary>
[Node(Category = "image", DisplayName = "Preview Image", IsOutput = true)]
public sealed class PreviewImage
{
    public async Task<UiOutput> ExecuteAsync(ImageBatch images, NodeContext context, IServiceProvider services)
    {
        var store = Results.For(services);
        var saved = new List<object>();
        for (int i = 0; i < images.Batch; i++)
            saved.Add(await store.SaveAsync("temp", "preview", PngWriter.Encode(images, i, Results.Metadata(context))));
        return new UiOutput("images", saved);
    }
}
