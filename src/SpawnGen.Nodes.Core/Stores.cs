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
}

/// <summary>Where Load Image finds its pictures.</summary>
public interface IInputStore
{
    IReadOnlyList<string> List();
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

/// <summary>Results as PNG files in folders. Output files are numbered (<c>prefix_00001_.png</c>) and never overwritten.</summary>
public sealed class DiskResultStore(OutputOptions options) : IResultStore
{
    readonly object _gate = new();

    public Task<ResultFile> SaveAsync(string type, string prefix, byte[] png)
    {
        string dir = type == "temp" ? options.TempDirectory : options.Directory;
        Directory.CreateDirectory(dir);
        string name;
        lock (_gate)
        {
            // Output files continue after the highest existing number; previews just get a unique name.
            name = type == "temp"
                ? $"{prefix}_{Guid.NewGuid().ToString("N")[..10]}.png"
                : $"{prefix}_{NextCounter(dir, prefix):D5}_.png";
            File.WriteAllBytes(Path.Combine(dir, name), png);
        }
        return Task.FromResult(new ResultFile(name, "", type));
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

/// <summary>Pictures kept in memory (the browser, where there is no folder to write to). Numbering matches <see cref="DiskResultStore"/>.</summary>
public sealed class MemoryResultStore : IResultStore
{
    readonly Dictionary<string, byte[]> _files = [];
    readonly object _gate = new();

    public Task<ResultFile> SaveAsync(string type, string prefix, byte[] png)
    {
        lock (_gate)
        {
            string name;
            if (type == "temp") name = $"{prefix}_{Guid.NewGuid().ToString("N")[..10]}.png";
            else
            {
                int n = 1;
                while (_files.ContainsKey(Key(type, $"{prefix}_{n:D5}_.png"))) n++;
                name = $"{prefix}_{n:D5}_.png";
            }
            _files[Key(type, name)] = png;
            Saved?.Invoke(new ResultFile(name, "", type), png);
            return Task.FromResult(new ResultFile(name, "", type));
        }
    }

    public byte[]? Get(string type, string filename)
    {
        lock (_gate) return _files.GetValueOrDefault(Key(type, filename));
    }

    /// <summary>Raised when a picture is stored (the browser turns it into a blob URL the canvas can show).</summary>
    public event Action<ResultFile, byte[]>? Saved;

    static string Key(string type, string name) => type + "/" + name;
}

public sealed class DiskInputStore(Func<string> directory) : IInputStore
{
    public IReadOnlyList<string> List() =>
        Directory.Exists(directory())
            ? Directory.EnumerateFiles(directory()).Where(ImageDecoder.IsSupportedName).Select(f => Path.GetFileName(f)!).OrderBy(n => n, StringComparer.OrdinalIgnoreCase).ToArray()
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

    public IReadOnlyList<string> List()
    {
        lock (_gate) return [.. _files.Keys];
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
