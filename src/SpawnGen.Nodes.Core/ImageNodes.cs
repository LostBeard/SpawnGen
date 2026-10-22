using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>Where <see cref="SaveImage"/> writes. Register an instance as a service to change it (default: <c>./output</c>).</summary>
public sealed class OutputOptions
{
    public string Directory { get; init; } = Path.Combine(AppContext.BaseDirectory, "output");
}

/// <summary>Where <see cref="LoadImage"/> reads from. Set <see cref="Directory"/> at startup (default: <c>./input</c>); the combo lists its PNG files.</summary>
public static class InputFiles
{
    public static string Directory { get; set; } = Path.Combine(AppContext.BaseDirectory, "input");

    public static string[] List() =>
        System.IO.Directory.Exists(Directory)
            ? System.IO.Directory.EnumerateFiles(Directory, "*.png").Select(Path.GetFileName).OrderBy(n => n, StringComparer.OrdinalIgnoreCase).ToArray()!
            : [];
}

[Node(Category = "image", DisplayName = "Load Image")]
public sealed class LoadImage
{
    public static string[] ImageNames() => InputFiles.List();

    public ImageBatch Execute([Input(OptionsProvider = nameof(ImageNames))] string image)
    {
        // The file name only: a node must not be able to read outside the input folder.
        var path = Path.Combine(InputFiles.Directory, Path.GetFileName(image));
        return PngReader.Decode(File.ReadAllBytes(path));
    }
}

[Node(Category = "image", DisplayName = "Save Image", IsOutput = true)]
public sealed class SaveImage
{
    public UiOutput Execute(ImageBatch images, IServiceProvider services, string filenamePrefix = "SpawnGen")
    {
        var dir = (services.GetService(typeof(OutputOptions)) as OutputOptions ?? new OutputOptions()).Directory;
        Directory.CreateDirectory(dir);

        var saved = new List<object>();
        for (int i = 0; i < images.Batch; i++)
        {
            // Counter continues after the highest existing file so reruns never overwrite.
            int n = NextCounter(dir, filenamePrefix);
            var name = $"{filenamePrefix}_{n:D5}_.png";
            File.WriteAllBytes(Path.Combine(dir, name), PngWriter.Encode(images, i));
            saved.Add(new { filename = name, subfolder = "", type = "output" });
        }
        return new UiOutput("images", saved);
    }

    static int NextCounter(string dir, string prefix)
    {
        int max = 0;
        foreach (var f in Directory.EnumerateFiles(dir, $"{prefix}_*_.png"))
        {
            var stem = Path.GetFileNameWithoutExtension(f);
            var digits = stem[(prefix.Length + 1)..^1];
            if (int.TryParse(digits, out var v) && v > max) max = v;
        }
        return max + 1;
    }
}

/// <summary>Shows images in the editor without writing files: the PNG bytes travel in the node's UI output.</summary>
[Node(Category = "image", DisplayName = "Preview Image", IsOutput = true)]
public sealed class PreviewImage
{
    public UiOutput Execute(ImageBatch images)
    {
        var pngs = new byte[images.Batch][];
        for (int i = 0; i < images.Batch; i++) pngs[i] = PngWriter.Encode(images, i);
        return new UiOutput("images", pngs);
    }
}
