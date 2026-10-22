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
    /// <summary>The result store a host registered, or a disk store over the output options (so the nodes also work with no host setup).</summary>
    public static IResultStore For(IServiceProvider services) =>
        services.GetService(typeof(IResultStore)) as IResultStore
        ?? new DiskResultStore(services.GetService(typeof(OutputOptions)) as OutputOptions ?? new OutputOptions());
}

[Node(Category = "image", DisplayName = "Load Image")]
public sealed class LoadImage
{
    public static string[] ImageNames() => InputFiles.List();

    public async Task<ImageBatch> ExecuteAsync([Input(OptionsProvider = nameof(ImageNames))] string image) =>
        PngReader.Decode(await InputFiles.Store.ReadAsync(image));
}

[Node(Category = "image", DisplayName = "Save Image", IsOutput = true)]
public sealed class SaveImage
{
    public async Task<UiOutput> ExecuteAsync(ImageBatch images, IServiceProvider services, string filenamePrefix = "ComfyUI")
    {
        var store = Results.For(services);
        var saved = new List<object>();
        for (int i = 0; i < images.Batch; i++)
            saved.Add(await store.SaveAsync("output", filenamePrefix, PngWriter.Encode(images, i)));
        return new UiOutput("images", saved);
    }
}

/// <summary>Shows images in the editor: stores them as temporary files (like ComfyUI's PreviewImage); they are not kept as results.</summary>
[Node(Category = "image", DisplayName = "Preview Image", IsOutput = true)]
public sealed class PreviewImage
{
    public async Task<UiOutput> ExecuteAsync(ImageBatch images, IServiceProvider services)
    {
        var store = Results.For(services);
        var saved = new List<object>();
        for (int i = 0; i < images.Batch; i++)
            saved.Add(await store.SaveAsync("temp", "preview", PngWriter.Encode(images, i)));
        return new UiOutput("images", saved);
    }
}
