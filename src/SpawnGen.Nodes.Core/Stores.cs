namespace SpawnGen.Nodes.Core;

/// <summary>How a stored result is referred to in run events: <c>type</c> is "output" (kept) or "temp" (a preview). Same shape as ComfyUI's.</summary>
public sealed record ResultFile(
    [property: System.Text.Json.Serialization.JsonPropertyName("filename")] string Filename,
    [property: System.Text.Json.Serialization.JsonPropertyName("subfolder")] string Subfolder,
    [property: System.Text.Json.Serialization.JsonPropertyName("type")] string Type);

/// <summary>Where Save Image and Preview Image put their pictures. Files on the desktop and the server, memory in the browser.</summary>
public interface IResultStore
{
    /// <param name="type">"output" for a result to keep, "temp" for a throw-away preview.</param>
    Task<ResultFile> SaveAsync(string type, string prefix, byte[] png);

    /// <summary>Stores any result file (a WAV, say) with the given extension (no dot).</summary>
    Task<ResultFile> SaveAsync(string type, string prefix, byte[] data, string extension);
}

/// <summary>Where Load Image finds its pictures.</summary>
public interface IInputStore
{
    /// <summary>The pictures the store holds (what Load Image offers).</summary>
    IReadOnlyList<string> List();
    /// <summary>The files of any kind whose name passes <paramref name="filter"/> (Load Audio lists the .wav files).</summary>
    IReadOnlyList<string> List(Func<string, bool> filter);
    Task<byte[]> ReadAsync(string name);
    /// <summary>Stores an uploaded picture and returns the name it was stored under (changed when it would overwrite another).</summary>
    Task<string> SaveAsync(string name, byte[] bytes);
}

/// <summary>Where <see cref="SaveImage"/> writes. Register an instance as a service to change it (default: <c>./output</c>).</summary>
public sealed class OutputOptions
{
    public string Directory { get; init; } = Path.Combine(AppContext.BaseDirectory, "output");
    /// <summary>Where <see cref="PreviewImage"/> writes its throw-away files.</summary>
    public string TempDirectory { get; init; } = Path.Combine(AppContext.BaseDirectory, "temp");
}

/// <summary>A save prefix such as <c>folder/name</c> split into its subfolder and file name part. The subfolder must stay inside the output folder.</summary>
public static class SavePrefix
{
    public static (string Subfolder, string Name) Split(string prefix)
    {
        prefix = ComputeVars(prefix ?? "", 0, 0, DateTime.Now).Replace('\\', '/');
        int slash = prefix.LastIndexOf('/');
        if (slash < 0) return ("", prefix);
        var subfolder = prefix[..slash].Trim('/');
        var segments = subfolder.Split('/', StringSplitOptions.RemoveEmptyEntries);
        if (prefix.StartsWith('/') || subfolder.Contains(':') || segments.Any(s => s == ".." || s == "."))
            throw new ArgumentException($"The file name prefix '{prefix}' must stay inside the output folder: no '..', drive letters or leading slash.", nameof(prefix));
        return (string.Join('/', segments), prefix[(slash + 1)..]);
    }

    /// <summary>
    /// ComfyUI's <c>compute_vars</c>: %width% and %height% of the picture (0 for files that are not pictures), and %year%, %month%, %day%, %hour%,
    /// %minute%, %second% of the local time. The Studio has already filled %date:...% and %Node.widget% when it made the prompt.
    /// </summary>
    public static string ComputeVars(string prefix, int width, int height, DateTime now)
    {
        if (!prefix.Contains('%')) return prefix;
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        return prefix.Replace("%width%", width.ToString(inv)).Replace("%height%", height.ToString(inv))
            .Replace("%year%", now.Year.ToString(inv)).Replace("%month%", now.Month.ToString("00", inv)).Replace("%day%", now.Day.ToString("00", inv))
            .Replace("%hour%", now.Hour.ToString("00", inv)).Replace("%minute%", now.Minute.ToString("00", inv)).Replace("%second%", now.Second.ToString("00", inv));
    }
}

/// <summary>Results as PNG files in folders. Output files are numbered (<c>prefix_00001_.png</c>) and never overwritten.</summary>
public sealed class DiskResultStore(OutputOptions options) : IResultStore
{
    readonly object _gate = new();

    public Task<ResultFile> SaveAsync(string type, string prefix, byte[] png) => SaveAsync(type, prefix, png, "png");

    public Task<ResultFile> SaveAsync(string type, string prefix, byte[] png, string extension)
    {
        var (subfolder, prefixName) = SavePrefix.Split(prefix);
        prefix = prefixName;
        string dir = Path.Combine(type == "temp" ? options.TempDirectory : options.Directory, subfolder);
        Directory.CreateDirectory(dir);
        string name;
        lock (_gate)
        {
            // Output files continue after the highest existing number; previews just get a unique name.
            name = type == "temp"
                ? $"{prefix}_{Guid.NewGuid().ToString("N")[..10]}.{extension}"
                : $"{prefix}_{NextCounter(dir, prefix, extension):D5}_.{extension}";
            File.WriteAllBytes(Path.Combine(dir, name), png);
        }
        return Task.FromResult(new ResultFile(name, subfolder, type));
    }

    static int NextCounter(string dir, string prefix, string extension)
    {
        int max = 0;
        foreach (var f in Directory.EnumerateFiles(dir, $"{prefix}_*_.{extension}"))
        {
            var stem = Path.GetFileNameWithoutExtension(f);
            var digits = stem[(prefix.Length + 1)..^1];
            if (int.TryParse(digits, out var v) && v > max) max = v;
        }
        return max + 1;
    }
}

/// <summary>Pictures kept in memory (the browser, where there is no folder to write to). Numbering matches <see cref="DiskResultStore"/>.</summary>
public sealed class MemoryResultStore : IResultStore
{
    readonly Dictionary<string, byte[]> _files = [];
    readonly object _gate = new();

    public Task<ResultFile> SaveAsync(string type, string prefix, byte[] png) => SaveAsync(type, prefix, png, "png");

    public Task<ResultFile> SaveAsync(string type, string prefix, byte[] png, string extension)
    {
        var (subfolder, prefixName) = SavePrefix.Split(prefix);
        prefix = prefixName;
        lock (_gate)
        {
            string name;
            if (type == "temp") name = $"{prefix}_{Guid.NewGuid().ToString("N")[..10]}.{extension}";
            else
            {
                int n = 1;
                while (_files.ContainsKey(Key(type, subfolder, $"{prefix}_{n:D5}_.{extension}"))) n++;
                name = $"{prefix}_{n:D5}_.{extension}";
            }
            _files[Key(type, subfolder, name)] = png;
            var file = new ResultFile(name, subfolder, type);
            Saved?.Invoke(file, png);
            return Task.FromResult(file);
        }
    }

    public byte[]? Get(string type, string filename, string? subfolder = null)
    {
        lock (_gate) return _files.GetValueOrDefault(Key(type, subfolder ?? "", filename));
    }

    /// <summary>Raised when a picture is stored (the browser turns it into a blob URL the canvas can show).</summary>
    public event Action<ResultFile, byte[]>? Saved;

    static string Key(string type, string subfolder, string name) => type + "/" + (subfolder.Length == 0 ? "" : subfolder + "/") + name;
}

public sealed class DiskInputStore(Func<string> directory) : IInputStore
{
    public IReadOnlyList<string> List() =>
        Directory.Exists(directory())
            ? Directory.EnumerateFiles(directory()).Where(ImageDecoder.IsSupportedName).Select(f => Path.GetFileName(f)!).OrderBy(n => n, StringComparer.OrdinalIgnoreCase).ToArray()
            : [];

    public IReadOnlyList<string> List(Func<string, bool> filter) =>
        Directory.Exists(directory())
            ? Directory.EnumerateFiles(directory()).Select(f => Path.GetFileName(f)!).Where(filter).OrderBy(n => n, StringComparer.OrdinalIgnoreCase).ToArray()
            : [];

    public Task<byte[]> ReadAsync(string name) =>
        // The file name only: a node must not be able to read outside the input folder.
        File.ReadAllBytesAsync(Path.Combine(directory(), Path.GetFileName(name)));

    public async Task<string> SaveAsync(string name, byte[] bytes)
    {
        Directory.CreateDirectory(directory());
        name = Path.GetFileName(name);
        if (File.Exists(Path.Combine(directory(), name)))
            name = $"{Path.GetFileNameWithoutExtension(name)}_{Guid.NewGuid().ToString("N")[..6]}{Path.GetExtension(name)}";
        await File.WriteAllBytesAsync(Path.Combine(directory(), name), bytes);
        return name;
    }
}

public sealed class MemoryInputStore : IInputStore
{
    readonly SortedDictionary<string, byte[]> _files = new(StringComparer.OrdinalIgnoreCase);
    readonly object _gate = new();

    public IReadOnlyList<string> List() => List(ImageDecoder.IsSupportedName);

    public IReadOnlyList<string> List(Func<string, bool> filter)
    {
        lock (_gate) return [.. _files.Keys.Where(filter)];
    }

    public Task<byte[]> ReadAsync(string name)
    {
        lock (_gate)
            return _files.TryGetValue(name, out var bytes) ? Task.FromResult(bytes) : throw new FileNotFoundException($"No uploaded image named '{name}'.");
    }

    public Task<string> SaveAsync(string name, byte[] bytes)
    {
        lock (_gate)
        {
            var stored = name;
            for (int n = 2; _files.ContainsKey(stored); n++) stored = $"{Path.GetFileNameWithoutExtension(name)}_{n}{Path.GetExtension(name)}";
            _files[stored] = bytes;
            return Task.FromResult(stored);
        }
    }
}
