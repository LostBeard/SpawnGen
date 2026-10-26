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

    /// <summary>
    /// The names in a results folder (a relative "a/b" path, "" for the top), files and folders, as Python's <c>os.listdir</c> gives them; null when
    /// there is no such folder. Save Image (to Folder) numbers its files from these.
    /// </summary>
    IReadOnlyList<string>? List(string type, string folder);

    /// <summary>Makes a results folder and its parents (<c>os.makedirs(..., exist_ok=True)</c>).</summary>
    void CreateFolder(string type, string folder);

    /// <summary>
    /// Writes a file under exactly <paramref name="name"/> in an existing results folder, replacing a file of that name (Save Image (to Folder)).
    /// Throws <see cref="DirectoryNotFoundException"/> when the folder is not there, as Python's <c>open</c> does.
    /// </summary>
    Task<ResultFile> WriteAsync(string type, string folder, string name, byte[] data);
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

    /// <summary>Every folder inside the input folder, at any depth, as "a/b" paths in Python's sorted order (ComfyUI's get_input_subfolders).</summary>
    IReadOnlyList<string> Folders();

    /// <summary>
    /// What a folder holds (a relative "a/b" path, "" for the input folder itself) in the order the file system lists it (Python's <c>os.listdir</c>):
    /// each name and whether it is a folder. Throws <see cref="DirectoryNotFoundException"/> when there is no such folder.
    /// </summary>
    IReadOnlyList<(string Name, bool IsFolder)> Entries(string folder);

    /// <summary>Whether a file or a folder is at the relative path (<c>os.path.exists</c>).</summary>
    bool Exists(string path);

    /// <summary>The file at a relative "a/b/name" path.</summary>
    Task<byte[]> ReadFileAsync(string path);
}

/// <summary>Relative paths in the stores: "a/b" with forward slashes, checked to stay inside the store's folder.</summary>
public static class StorePaths
{
    /// <summary>
    /// Python's <c>os.path.normpath</c> of a relative path, with "/" between parts ("" for the folder itself): "." parts dropped, "a/.." folded.
    /// Throws when the path is absolute (a leading slash or a drive) or climbs out of the folder, like ComfyUI's secure_subfolder_path.
    /// </summary>
    public static string Normalize(string path, string what = "path")
    {
        // Windows paths take either slash and may name a drive; elsewhere a backslash and a colon are just characters of a name.
        var p = OperatingSystem.IsWindows() ? (path ?? "").Replace('\\', '/') : path ?? "";
        if (p.StartsWith('/') || OperatingSystem.IsWindows() && p.Contains(':'))
            throw new ArgumentException($"Invalid {what} '{path}': resolves outside of its folder.");
        var parts = new List<string>();
        foreach (var part in p.Split('/'))
        {
            if (part is "" or ".") continue;
            if (part == "..")
            {
                if (parts.Count == 0) throw new ArgumentException($"Invalid {what} '{path}': resolves outside of its folder.");
                parts.RemoveAt(parts.Count - 1);
            }
            else parts.Add(part);
        }
        return string.Join('/', parts);
    }

    /// <summary>The folder part of a normalized path ("" when it has none) and its last part.</summary>
    public static (string Folder, string Name) Split(string normalized)
    {
        int slash = normalized.LastIndexOf('/');
        return slash < 0 ? ("", normalized) : (normalized[..slash], normalized[(slash + 1)..]);
    }

    public static string Join(string folder, string name) => folder.Length == 0 ? name : name.Length == 0 ? folder : folder + "/" + name;

    /// <summary>A normalized relative path on disk under <paramref name="root"/>.</summary>
    internal static string OnDisk(string root, string normalized) =>
        normalized.Length == 0 ? root : Path.Combine(root, normalized.Replace('/', Path.DirectorySeparatorChar));

    /// <summary>A single file name: no folder part, not "." or "..".</summary>
    internal static void CheckName(string name)
    {
        if (name.Length == 0 || name.Contains('/') || name.Contains('\\') || name is "." or "..") throw new ArgumentException($"'{name}' is not a file name.", nameof(name));
    }
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

    string Root(string type) => type == "temp" ? options.TempDirectory : options.Directory;

    public IReadOnlyList<string>? List(string type, string folder)
    {
        var dir = StorePaths.OnDisk(Root(type), StorePaths.Normalize(folder, "folder"));
        return Directory.Exists(dir) ? [.. Directory.EnumerateFileSystemEntries(dir).Select(e => Path.GetFileName(e)!)] : null;
    }

    public void CreateFolder(string type, string folder) => Directory.CreateDirectory(StorePaths.OnDisk(Root(type), StorePaths.Normalize(folder, "folder")));

    public async Task<ResultFile> WriteAsync(string type, string folder, string name, byte[] data)
    {
        var normalized = StorePaths.Normalize(folder, "folder");
        StorePaths.CheckName(name);
        await File.WriteAllBytesAsync(Path.Combine(StorePaths.OnDisk(Root(type), normalized), name), data);
        return new ResultFile(name, normalized, type);
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

    readonly HashSet<string> _folders = [];

    /// <summary>The names directly in the folder, in key order (a memory store has no file system order of its own).</summary>
    public IReadOnlyList<string>? List(string type, string folder)
    {
        var top = Folder(type, StorePaths.Normalize(folder, "folder"));
        lock (_gate)
        {
            if (!FolderExists(top)) return null;
            var names = new List<string>();
            foreach (var key in _files.Keys.Concat(_folders).Order(StringComparer.Ordinal))
            {
                if (!key.StartsWith(top + "/", StringComparison.Ordinal)) continue;
                var first = key[(top.Length + 1)..].Split('/')[0];
                if (!names.Contains(first)) names.Add(first);
            }
            return names;
        }
    }

    public void CreateFolder(string type, string folder)
    {
        var parts = StorePaths.Normalize(folder, "folder");
        lock (_gate)
        {
            var path = type;
            foreach (var part in parts.Split('/', StringSplitOptions.RemoveEmptyEntries)) _folders.Add(path = path + "/" + part);
        }
    }

    public Task<ResultFile> WriteAsync(string type, string folder, string name, byte[] data)
    {
        var normalized = StorePaths.Normalize(folder, "folder");
        StorePaths.CheckName(name);
        lock (_gate)
        {
            if (!FolderExists(Folder(type, normalized))) throw new DirectoryNotFoundException($"No such results folder: '{normalized}'.");
            _files[Key(type, normalized, name)] = data;
            var file = new ResultFile(name, normalized, type);
            Saved?.Invoke(file, data);
            return Task.FromResult(file);
        }
    }

    static string Folder(string type, string normalized) => normalized.Length == 0 ? type : type + "/" + normalized;

    // The top of a type always exists; a folder exists once made, or once a file is in it (at any depth).
    bool FolderExists(string path) =>
        !path.Contains('/') || _folders.Contains(path) || _files.Keys.Any(k => k.StartsWith(path + "/", StringComparison.Ordinal));

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

    public IReadOnlyList<string> Folders()
    {
        var root = directory();
        if (!Directory.Exists(root)) return [];
        // os.walk: every folder at any depth, hidden ones too; a linked folder is not walked into (and so not listed).
        var options = new EnumerationOptions { RecurseSubdirectories = true, AttributesToSkip = FileAttributes.ReparsePoint, IgnoreInaccessible = true };
        return [.. Directory.EnumerateDirectories(root, "*", options).Select(d => Path.GetRelativePath(root, d).Replace(Path.DirectorySeparatorChar, '/')).Order(PyText.Order)];
    }

    public IReadOnlyList<(string Name, bool IsFolder)> Entries(string folder)
    {
        var dir = StorePaths.OnDisk(directory(), StorePaths.Normalize(folder, "folder"));
        if (!Directory.Exists(dir)) throw new DirectoryNotFoundException($"The system cannot find the path specified: '{folder}'");
        return [.. Directory.EnumerateFileSystemEntries(dir).Select(e => (Path.GetFileName(e)!, Directory.Exists(e)))];
    }

    public bool Exists(string path)
    {
        var full = StorePaths.OnDisk(directory(), StorePaths.Normalize(path));
        return File.Exists(full) || Directory.Exists(full);
    }

    public Task<byte[]> ReadFileAsync(string path) => File.ReadAllBytesAsync(StorePaths.OnDisk(directory(), StorePaths.Normalize(path)));

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

    /// <summary>The files at the top (a file saved as "a/b.png" is in folder "a", like a file in a sub folder of the input folder).</summary>
    public IReadOnlyList<string> List(Func<string, bool> filter)
    {
        lock (_gate) return [.. _files.Keys.Where(k => !k.Contains('/')).Where(filter)];
    }

    public IReadOnlyList<string> Folders()
    {
        lock (_gate)
        {
            var folders = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var key in _files.Keys)
                for (int slash = key.IndexOf('/'); slash > 0; slash = key.IndexOf('/', slash + 1)) folders.Add(key[..slash]);
            return [.. folders.Order(PyText.Order)];
        }
    }

    /// <summary>Names in the store's order (case-insensitive, as an NTFS folder lists them); a folder is listed where its first file is.</summary>
    public IReadOnlyList<(string Name, bool IsFolder)> Entries(string folder)
    {
        var top = StorePaths.Normalize(folder, "folder");
        var prefix = top.Length == 0 ? "" : top + "/";
        lock (_gate)
        {
            var entries = new List<(string, bool)>();
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var key in _files.Keys)
            {
                if (!key.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)) continue;
                var rest = key[prefix.Length..];
                int slash = rest.IndexOf('/');
                var name = slash < 0 ? rest : rest[..slash];
                if (seen.Add(name)) entries.Add((name, slash >= 0));
            }
            if (entries.Count == 0 && top.Length > 0) throw new DirectoryNotFoundException($"The system cannot find the path specified: '{folder}'");
            return entries;
        }
    }

    public bool Exists(string path)
    {
        var p = StorePaths.Normalize(path);
        lock (_gate) return p.Length == 0 || _files.ContainsKey(p) || _files.Keys.Any(k => k.StartsWith(p + "/", StringComparison.OrdinalIgnoreCase));
    }

    public Task<byte[]> ReadFileAsync(string path)
    {
        var p = StorePaths.Normalize(path);
        lock (_gate)
            return _files.TryGetValue(p, out var bytes) ? Task.FromResult(bytes) : throw new FileNotFoundException($"No such file: '{path}'.");
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
