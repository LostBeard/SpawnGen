using System.Globalization;
using System.Text.Json.Serialization;

namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md) from SpawnGen's tests and black-box measurements of ComfyUI: file numbering from
// Fixtures/comfy/save_counter.json (made by running ComfyUI), save prefixes, subfolders and temporary names from the Save Image, Preview Image,
// Save Latent and audio tests, and folder listings measured with ComfyUI's server (the dataset folder options, 2026-11-11).

/// <summary>
/// A file a node saved, as ComfyUI's ui entries name it: <c>{"filename", "subfolder", "type"}</c>, and for pictures and videos the
/// <c>"metadata"</c> ComfyUI adds (measured: Save / Preview Image and the animated savers give kind "image" with the size; audio has none).
/// </summary>
public sealed record ResultFile(
    [property: JsonPropertyName("filename")] string Filename,
    [property: JsonPropertyName("subfolder")] string Subfolder,
    [property: JsonPropertyName("type")] string Type)
{
    [JsonPropertyName("metadata"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public ResultMetadata? Metadata { get; init; }

    /// <summary>This file marked as a picture of that size.</summary>
    public ResultFile AsImage(int width, int height) => this with { Metadata = new ResultMetadata("image", width, height) };
}

/// <summary>ComfyUI's ui metadata of a saved file: its kind ("image", "video"), size, and for a video its length in seconds and frame rate.</summary>
public sealed record ResultMetadata(
    [property: JsonPropertyName("kind")] string Kind,
    [property: JsonPropertyName("width")] int Width,
    [property: JsonPropertyName("height")] int Height)
{
    [JsonPropertyName("duration"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? Duration { get; init; }

    [JsonPropertyName("fps"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public double? Fps { get; init; }
}

/// <summary>Where a folder result store keeps the "output" and "temp" results.</summary>
public sealed class OutputOptions
{
    public string Directory { get; set; } = Path.Combine(AppContext.BaseDirectory, "output");
    public string TempDirectory { get; set; } = Path.Combine(AppContext.BaseDirectory, "temp");
}

/// <summary>A name in a store folder and whether it is a folder.</summary>
public readonly record struct StoreEntry(string Name, bool IsFolder);

/// <summary>
/// The output side: results by type ("output", "temp"), each type a tree of folders. A folder on the desktop and the server, memory in the browser.
/// </summary>
public interface IResultStore
{
    /// <summary>
    /// Saves <paramref name="data"/> under the save prefix (see <see cref="SavePrefix"/>): <c>{name}_00001_.{extension}</c> in the prefix's subfolder,
    /// numbered after every name of the prefix already in that folder; the subfolder is created.
    /// </summary>
    Task<ResultFile> SaveAsync(string type, string prefix, byte[] data, string extension = "png", bool trailingUnderscore = true);

    /// <summary>
    /// <see cref="SaveAsync"/> for a large file: <paramref name="write"/> writes it into the stream the store opens (a folder store writes straight
    /// to disk).
    /// </summary>
    Task<ResultFile> SaveStreamAsync(string type, string prefix, string extension, Func<Stream, Task> write, bool trailingUnderscore = true);

    /// <summary>Writes a file under its own name into an existing folder (<see cref="DirectoryNotFoundException"/> when it is missing); replaces a file of that name.</summary>
    Task<ResultFile> WriteAsync(string type, string subfolder, string filename, byte[] data);

    /// <summary>Creates the folder and the folders above it.</summary>
    void CreateFolder(string type, string subfolder);

    /// <summary>The names directly in the folder (files and folders), sorted by code point; null when the folder does not exist.</summary>
    IReadOnlyList<string>? List(string type, string subfolder);

    /// <summary>The file's bytes, or null.</summary>
    byte[]? Get(string type, string filename, string subfolder = "");

    /// <summary>The file's bytes, or null.</summary>
    Task<byte[]?> ReadAsync(string type, string subfolder, string filename);

    /// <summary>The files directly in the type's top folder, newest first (Load Image (from Outputs), ComfyUI's /internal/files).</summary>
    IReadOnlyList<string> FilesNewestFirst(string type);
}

/// <summary>The input side: the pictures, audio and dataset folders the loader nodes read. Paths are relative, with "/" between folders.</summary>
public interface IInputStore
{
    /// <summary>The files directly in the input folder that pass <paramref name="filter"/> (pictures when null), sorted by code point.</summary>
    IReadOnlyList<string> List(Func<string, bool>? filter = null);

    /// <summary>Every folder at any depth ("set", "set/plain"), each followed by the folders inside it, in listing order.</summary>
    IReadOnlyList<string> Folders();

    /// <summary>The names directly in a folder ("" is the input folder), in the order the file system lists them; <see cref="DirectoryNotFoundException"/> when missing.</summary>
    IReadOnlyList<StoreEntry> Entries(string folder);

    /// <summary>Whether a file or folder is there.</summary>
    bool Exists(string path);

    /// <summary>The file's bytes (<see cref="FileNotFoundException"/> when missing).</summary>
    Task<byte[]> ReadAsync(string path);

    /// <summary>Stores a file (creating its folders) and returns its path.</summary>
    Task<string> SaveAsync(string path, byte[] data);
}

/// <summary>
/// Save prefixes as ComfyUI reads them (measured): "folder/name" saves into a subfolder of the output folder, the size and time variables are
/// filled in, and files are numbered with one counter per prefix and folder.
/// </summary>
public static class SavePrefix
{
    /// <summary>
    /// Fills %width%, %height%, %year%, %month%, %day%, %hour%, %minute% and %second% (two digits but the year). Other %...% parts stay
    /// (%date:...% and %Node.widget% are filled by the Studio before the prompt is sent).
    /// </summary>
    public static string ComputeVars(string prefix, int width, int height, DateTime now)
    {
        if (!prefix.Contains('%')) return prefix;
        var c = CultureInfo.InvariantCulture;
        return prefix
            .Replace("%width%", width.ToString(c), StringComparison.Ordinal)
            .Replace("%height%", height.ToString(c), StringComparison.Ordinal)
            .Replace("%year%", now.Year.ToString(c), StringComparison.Ordinal)
            .Replace("%month%", now.Month.ToString("00", c), StringComparison.Ordinal)
            .Replace("%day%", now.Day.ToString("00", c), StringComparison.Ordinal)
            .Replace("%hour%", now.Hour.ToString("00", c), StringComparison.Ordinal)
            .Replace("%minute%", now.Minute.ToString("00", c), StringComparison.Ordinal)
            .Replace("%second%", now.Second.ToString("00", c), StringComparison.Ordinal);
    }

    /// <summary>
    /// The subfolder ("" for none, "/" between folders) and the name of a prefix, after <see cref="ComputeVars"/> with sizes 0 and the current time.
    /// Either slash separates folders. A prefix that would leave the output folder (absolute, a drive, ".." above the top) is refused.
    /// </summary>
    public static (string Subfolder, string Name) Split(string prefix)
    {
        var path = ComputeVars(prefix, 0, 0, DateTime.Now).Replace('\\', '/');
        int slash = path.LastIndexOf('/');
        var folder = slash < 0 ? "" : path[..slash];
        if (path.StartsWith('/') || (path.Length >= 2 && path[1] == ':')) throw Outside(prefix);
        return (StorePaths.Normalize(folder) ?? throw Outside(prefix), path[(slash + 1)..]);
    }

    static ArgumentException Outside(string prefix) => new($"Saving outside the output folder is not allowed: '{prefix}'.", nameof(prefix));

    /// <summary>
    /// The number the next file of <paramref name="prefix"/> gets among <paramref name="names"/> (the names in its folder, files and folders alike):
    /// one more than the largest number found, or 1. Measured with ComfyUI (Fixtures/comfy/save_counter.json): a name counts when, without its
    /// extension, it starts with "{prefix}_" (ignoring case on Windows) and the text up to the next "_" reads as a Python int (spaces and a sign
    /// allowed), so "x_00002.flac", "x_00004_00009.png" and a folder "x_00003" all count, whatever the extension or trailing "_".
    /// </summary>
    public static long NextCounter(IEnumerable<string> names, string prefix)
    {
        var comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        long? largest = null;
        foreach (var name in names)
        {
            var stem = WithoutExtension(name);
            if (stem.Length <= prefix.Length || stem[prefix.Length] != '_' || !stem.StartsWith(prefix, comparison)) continue;
            var number = stem[(prefix.Length + 1)..];
            int end = number.IndexOf('_');
            if (end >= 0) number = number[..end];
            if (ParseInt(number) is long n && (largest is null || n > largest)) largest = n;
        }
        return largest is long l ? l + 1 : 1;
    }

    /// <summary><c>{name}_{counter:05}_.{extension}</c> (no "_" before the dot when <paramref name="trailingUnderscore"/> is false), the number as Python's <c>:05</c> writes it.</summary>
    public static string FileName(string name, long counter, string extension, bool trailingUnderscore = true) =>
        $"{name}_{Number5(counter)}{(trailingUnderscore ? "_" : "")}.{extension}";

    /// <summary>Python's <c>f"{n:05}"</c>: at least 5 characters, zeros after the sign ("-0002").</summary>
    internal static string Number5(long n) =>
        n < 0 ? "-" + Math.Abs((decimal)n).ToString(CultureInfo.InvariantCulture).PadLeft(4, '0') : n.ToString(CultureInfo.InvariantCulture).PadLeft(5, '0');

    /// <summary>A name without its extension, as Python's os.path.splitext cuts it: at the last dot, unless only dots come before it.</summary>
    internal static string WithoutExtension(string name)
    {
        int dot = name.LastIndexOf('.');
        for (int i = 0; i < dot; i++)
            if (name[i] != '.') return name[..dot];
        return name;
    }

    /// <summary>Python's <c>int(text)</c> for decimal text: surrounding whitespace, an optional sign, then digits; null when it is not one (or does not fit 64 bits).</summary>
    static long? ParseInt(string text)
    {
        text = PyText.Strip(text);
        int i = 0;
        bool negative = false;
        if (text.Length > 0 && text[0] is '+' or '-') { negative = text[0] == '-'; i = 1; }
        if (i == text.Length) return null;
        long value = 0;
        for (; i < text.Length; i++)
        {
            if (!char.IsDigit(text[i])) return null;
            int digit = (int)char.GetNumericValue(text[i]);
            if (value > (long.MaxValue - digit) / 10) return null;
            value = value * 10 + digit;
        }
        return negative ? -value : value;
    }
}

/// <summary>Relative store paths and folder listings shared by the stores.</summary>
public static class StorePaths
{
    /// <summary>
    /// A relative path with "/" between its parts, "." and empty parts dropped and ".." applied; null when it is absolute (a leading slash or a
    /// drive) or ".." goes above the top.
    /// </summary>
    public static string? Normalize(string path)
    {
        path = path.Replace('\\', '/');
        if (path.StartsWith('/') || (path.Length >= 2 && path[1] == ':')) return null;
        var parts = new List<string>();
        foreach (var part in path.Split('/'))
        {
            if (part is "" or ".") continue;
            if (part == "..")
            {
                if (parts.Count == 0) return null;
                parts.RemoveAt(parts.Count - 1);
                continue;
            }
            parts.Add(part);
        }
        return string.Join('/', parts);
    }

    /// <summary>The full path of <paramref name="relative"/> under <paramref name="root"/>; <see cref="ArgumentException"/> when it would leave it.</summary>
    public static string Inside(string root, string relative)
    {
        var rel = Normalize(relative) ?? throw new ArgumentException($"'{relative}' is outside the folder.", nameof(relative));
        var rootFull = Path.GetFullPath(root);
        var full = Path.GetFullPath(Path.Combine(rootFull, rel.Replace('/', Path.DirectorySeparatorChar)));
        var comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        var top = Path.TrimEndingDirectorySeparator(rootFull);
        if (!full.Equals(top, comparison) && !full.StartsWith(top + Path.DirectorySeparatorChar, comparison))
            throw new ArgumentException($"'{relative}' is outside the folder.", nameof(relative));
        return full;
    }

    /// <summary>A plain file name: no folders, not "." or "..".</summary>
    public static void CheckFileName(string filename)
    {
        if (filename.Length == 0 || filename is "." or ".." || filename.IndexOfAny(['/', '\\']) >= 0 || filename.Contains(':'))
            throw new ArgumentException($"'{filename}' is not a file name.", nameof(filename));
    }

    /// <summary>The files directly in a folder on disk, newest first (by last write time), without hidden ones ("."); empty when the folder is missing.</summary>
    public static IReadOnlyList<string> FilesNewestFirst(string directory) =>
        System.IO.Directory.Exists(directory)
            ? [.. new DirectoryInfo(directory).EnumerateFiles().Where(f => !f.Name.StartsWith('.')).OrderByDescending(f => f.LastWriteTimeUtc).Select(f => f.Name)]
            : [];

    /// <summary>Names in code point order, as Python's sorted() orders strings.</summary>
    public static readonly IComparer<string> CodePointOrder = Comparer<string>.Create(CompareCodePoints);

    static int CompareCodePoints(string? a, string? b)
    {
        if (a is null || b is null) return a is null ? b is null ? 0 : -1 : 1;
        var x = a.EnumerateRunes().GetEnumerator();
        var y = b.EnumerateRunes().GetEnumerator();
        while (true)
        {
            bool hasX = x.MoveNext(), hasY = y.MoveNext();
            if (!hasX || !hasY) return hasX ? 1 : hasY ? -1 : 0;
            int c = x.Current.Value.CompareTo(y.Current.Value);
            if (c != 0) return c;
        }
    }

    /// <summary>The order NTFS lists names in, which the memory stores follow: case ignored, then ordinal.</summary>
    public static readonly IComparer<string> FileSystemOrder = Comparer<string>.Create((a, b) =>
    {
        int c = string.Compare(a, b, StringComparison.OrdinalIgnoreCase);
        return c != 0 ? c : string.CompareOrdinal(a, b);
    });
}

/// <summary>What both result stores share: numbering a saved file from the names already in its folder.</summary>
public abstract class ResultStore : IResultStore
{
    readonly SemaphoreSlim _saving = new(1, 1);

    public async Task<ResultFile> SaveAsync(string type, string prefix, byte[] data, string extension = "png", bool trailingUnderscore = true)
    {
        var (subfolder, name) = SavePrefix.Split(prefix);
        await _saving.WaitAsync();
        try
        {
            CreateFolder(type, subfolder);
            var counter = SavePrefix.NextCounter(List(type, subfolder) ?? [], name);
            return await WriteAsync(type, subfolder, SavePrefix.FileName(name, counter, extension, trailingUnderscore), data);
        }
        finally { _saving.Release(); }
    }

    public async Task<ResultFile> SaveStreamAsync(string type, string prefix, string extension, Func<Stream, Task> write, bool trailingUnderscore = true)
    {
        var (subfolder, name) = SavePrefix.Split(prefix);
        await _saving.WaitAsync();
        try
        {
            CreateFolder(type, subfolder);
            var counter = SavePrefix.NextCounter(List(type, subfolder) ?? [], name);
            return await WriteStreamAsync(type, subfolder, SavePrefix.FileName(name, counter, extension, trailingUnderscore), write);
        }
        finally { _saving.Release(); }
    }

    /// <summary>Writes a file from a stream writer; by default through memory and <see cref="WriteAsync"/>.</summary>
    protected virtual async Task<ResultFile> WriteStreamAsync(string type, string subfolder, string filename, Func<Stream, Task> write)
    {
        using var buffer = new MemoryStream();
        await write(buffer);
        return await WriteAsync(type, subfolder, filename, buffer.ToArray());
    }

    public abstract Task<ResultFile> WriteAsync(string type, string subfolder, string filename, byte[] data);
    public abstract void CreateFolder(string type, string subfolder);
    public abstract IReadOnlyList<string>? List(string type, string subfolder);
    public abstract byte[]? Get(string type, string filename, string subfolder = "");
    public Task<byte[]?> ReadAsync(string type, string subfolder, string filename) => Task.FromResult(Get(type, filename, subfolder));
    public abstract IReadOnlyList<string> FilesNewestFirst(string type);

    /// <summary>The folder path in the store's form ("" for the top), or <see cref="ArgumentException"/> when it would leave the type's folder.</summary>
    protected static string Folder(string subfolder) =>
        StorePaths.Normalize(subfolder ?? "") ?? throw new ArgumentException($"'{subfolder}' is outside the output folder.", nameof(subfolder));
}

/// <summary>Results in the output and temp folders on disk (the desktop app and the server).</summary>
public sealed class DiskResultStore(OutputOptions options) : ResultStore
{
    string Root(string type) => type switch
    {
        "output" => options.Directory,
        "temp" => options.TempDirectory,
        _ => throw new ArgumentException($"There is no '{type}' result folder.", nameof(type)),
    };

    string FolderPath(string type, string subfolder) => StorePaths.Inside(Root(type), Folder(subfolder));

    public override async Task<ResultFile> WriteAsync(string type, string subfolder, string filename, byte[] data)
    {
        StorePaths.CheckFileName(filename);
        var folder = FolderPath(type, subfolder);
        if (!System.IO.Directory.Exists(folder)) throw new DirectoryNotFoundException($"There is no folder '{Folder(subfolder)}' in the {type} folder.");
        await File.WriteAllBytesAsync(Path.Combine(folder, filename), data);
        return new ResultFile(filename, Folder(subfolder), type);
    }

    protected override async Task<ResultFile> WriteStreamAsync(string type, string subfolder, string filename, Func<Stream, Task> write)
    {
        StorePaths.CheckFileName(filename);
        var folder = FolderPath(type, subfolder);
        if (!System.IO.Directory.Exists(folder)) throw new DirectoryNotFoundException($"There is no folder '{Folder(subfolder)}' in the {type} folder.");
        await using (var file = File.Create(Path.Combine(folder, filename))) await write(file);
        return new ResultFile(filename, Folder(subfolder), type);
    }

    public override void CreateFolder(string type, string subfolder) => System.IO.Directory.CreateDirectory(FolderPath(type, subfolder));

    public override IReadOnlyList<string>? List(string type, string subfolder)
    {
        var folder = FolderPath(type, subfolder);
        if (!System.IO.Directory.Exists(folder)) return null;
        return [.. new DirectoryInfo(folder).EnumerateFileSystemInfos().Select(e => e.Name).Order(StorePaths.CodePointOrder)];
    }

    public override byte[]? Get(string type, string filename, string subfolder = "")
    {
        StorePaths.CheckFileName(filename);
        var path = Path.Combine(FolderPath(type, subfolder), filename);
        return File.Exists(path) ? File.ReadAllBytes(path) : null;
    }

    public override IReadOnlyList<string> FilesNewestFirst(string type) => StorePaths.FilesNewestFirst(Root(type));
}

/// <summary>Results kept in memory (the browser).</summary>
public sealed class MemoryResultStore : ResultStore
{
    readonly Lock _gate = new();
    readonly HashSet<(string Type, string Folder)> _folders = [];
    readonly Dictionary<(string Type, string Folder, string Name), (byte[] Data, long Order)> _files = [];
    long _order;

    /// <summary>Raised after every file is written, with the file and its bytes.</summary>
    public event Action<ResultFile, byte[]>? Saved;

    bool FolderExists(string type, string folder) => folder.Length == 0 || _folders.Contains((type, folder));

    public override Task<ResultFile> WriteAsync(string type, string subfolder, string filename, byte[] data)
    {
        StorePaths.CheckFileName(filename);
        var folder = Folder(subfolder);
        lock (_gate)
        {
            if (!FolderExists(type, folder)) throw new DirectoryNotFoundException($"There is no folder '{folder}' in the {type} folder.");
            _files[(type, folder, filename)] = ((byte[])data.Clone(), ++_order);
        }
        var file = new ResultFile(filename, folder, type);
        Saved?.Invoke(file, data);
        return Task.FromResult(file);
    }

    public override void CreateFolder(string type, string subfolder)
    {
        var folder = Folder(subfolder);
        lock (_gate)
            for (var f = folder; f.Length > 0; f = f.LastIndexOf('/') is var s && s < 0 ? "" : f[..s]) _folders.Add((type, f));
    }

    public override IReadOnlyList<string>? List(string type, string subfolder)
    {
        var folder = Folder(subfolder);
        lock (_gate)
        {
            if (!FolderExists(type, folder)) return null;
            var names = _files.Keys.Where(k => k.Type == type && k.Folder == folder).Select(k => k.Name)
                .Concat(_folders.Where(f => f.Type == type && Parent(f.Folder) == folder).Select(f => f.Folder[(f.Folder.LastIndexOf('/') + 1)..]));
            return [.. names.Distinct().Order(StorePaths.CodePointOrder)];
        }
    }

    static string? Parent(string folder) => folder.LastIndexOf('/') is var s && s < 0 ? "" : folder[..s];

    public override byte[]? Get(string type, string filename, string subfolder = "")
    {
        var folder = Folder(subfolder);
        lock (_gate) return _files.TryGetValue((type, folder, filename), out var file) ? file.Data : null;
    }

    public override IReadOnlyList<string> FilesNewestFirst(string type)
    {
        lock (_gate) return [.. _files.Where(f => f.Key.Type == type && f.Key.Folder.Length == 0 && !f.Key.Name.StartsWith('.')).OrderByDescending(f => f.Value.Order).Select(f => f.Key.Name)];
    }
}

/// <summary>What both input stores share: the default picture filter and the folder walk.</summary>
public abstract class InputStore : IInputStore
{
    public IReadOnlyList<string> List(Func<string, bool>? filter = null)
    {
        filter ??= ImageDecoder.IsSupportedName;
        return [.. Entries("").Where(e => !e.IsFolder && filter(e.Name)).Select(e => e.Name).Order(StorePaths.CodePointOrder)];
    }

    /// <summary>Depth first: each folder, then the folders inside it, in listing order (measured with ComfyUI's folder options).</summary>
    public IReadOnlyList<string> Folders()
    {
        var folders = new List<string>();
        void Walk(string folder)
        {
            foreach (var e in Entries(folder))
            {
                if (!e.IsFolder) continue;
                var path = folder.Length == 0 ? e.Name : $"{folder}/{e.Name}";
                folders.Add(path);
                Walk(path);
            }
        }
        Walk("");
        return folders;
    }

    public abstract IReadOnlyList<StoreEntry> Entries(string folder);
    public abstract bool Exists(string path);
    public abstract Task<byte[]> ReadAsync(string path);
    public abstract Task<string> SaveAsync(string path, byte[] data);

    /// <summary>The path in the store's form, or <see cref="ArgumentException"/> when it would leave the input folder.</summary>
    protected static string Relative(string path) =>
        StorePaths.Normalize(path ?? "") ?? throw new ArgumentException($"'{path}' is outside the input folder.", nameof(path));
}

/// <summary>The input folder on disk.</summary>
public sealed class DiskInputStore(Func<string> directory) : InputStore
{
    string Full(string path) => StorePaths.Inside(directory(), Relative(path));

    public override IReadOnlyList<StoreEntry> Entries(string folder)
    {
        var full = Full(folder);
        if (!System.IO.Directory.Exists(full) && Relative(folder).Length == 0) return []; // no input folder yet: nothing in it
        if (!System.IO.Directory.Exists(full)) throw new DirectoryNotFoundException($"There is no folder '{folder}' in the input folder.");
        return [.. new DirectoryInfo(full).EnumerateFileSystemInfos().Select(e => new StoreEntry(e.Name, e is DirectoryInfo))];
    }

    public override bool Exists(string path)
    {
        try
        {
            var full = Full(path);
            return File.Exists(full) || System.IO.Directory.Exists(full);
        }
        catch (ArgumentException) { return false; }
    }

    public override Task<byte[]> ReadAsync(string path) => File.ReadAllBytesAsync(Full(path));

    public override async Task<string> SaveAsync(string path, byte[] data)
    {
        var rel = Relative(path);
        var full = Full(rel);
        System.IO.Directory.CreateDirectory(Path.GetDirectoryName(full)!);
        await File.WriteAllBytesAsync(full, data);
        return rel;
    }
}

/// <summary>Input files kept in memory (the browser). Names are listed case-insensitively, as NTFS lists them.</summary>
public sealed class MemoryInputStore : InputStore
{
    readonly Lock _gate = new();
    readonly Dictionary<string, byte[]> _files = new(StringComparer.Ordinal);
    readonly HashSet<string> _folders = new(StringComparer.Ordinal);

    public override IReadOnlyList<StoreEntry> Entries(string folder)
    {
        var dir = Relative(folder);
        lock (_gate)
        {
            if (dir.Length > 0 && !_folders.Contains(dir)) throw new DirectoryNotFoundException($"There is no folder '{folder}' in the input folder.");
            var prefix = dir.Length == 0 ? "" : dir + "/";
            static bool Direct(string path, string prefix) => path.StartsWith(prefix, StringComparison.Ordinal) && path.IndexOf('/', prefix.Length) < 0;
            return [.. _files.Keys.Where(p => Direct(p, prefix)).Select(p => new StoreEntry(p[prefix.Length..], false))
                .Concat(_folders.Where(p => Direct(p, prefix)).Select(p => new StoreEntry(p[prefix.Length..], true)))
                .OrderBy(e => e.Name, StorePaths.FileSystemOrder)];
        }
    }

    public override bool Exists(string path)
    {
        if (StorePaths.Normalize(path ?? "") is not { } rel) return false;
        lock (_gate) return rel.Length == 0 || _files.ContainsKey(rel) || _folders.Contains(rel);
    }

    public override Task<byte[]> ReadAsync(string path)
    {
        var rel = Relative(path);
        lock (_gate)
            return _files.TryGetValue(rel, out var data) ? Task.FromResult(data) : throw new FileNotFoundException($"There is no input file '{path}'.", path);
    }

    public override Task<string> SaveAsync(string path, byte[] data)
    {
        var rel = Relative(path);
        if (rel.Length == 0) throw new ArgumentException("A file needs a name.", nameof(path));
        lock (_gate)
        {
            for (int s = rel.IndexOf('/'); s >= 0; s = rel.IndexOf('/', s + 1)) _folders.Add(rel[..s]);
            _files[rel] = (byte[])data.Clone();
        }
        return Task.FromResult(rel);
    }
}
