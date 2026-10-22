using System.Reflection;
using System.Runtime.Loader;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Server;

/// <summary>
/// Custom nodes: assemblies in the custom nodes folder (each .dll there, or the .dll named like its sub-folder) are loaded into the registry at start-up, the way ComfyUI
/// loads its custom_nodes. A node pack is code the user chose to install, and runs with the server's rights.
/// </summary>
public static class CustomNodes
{
    public sealed record Loaded(string File, IReadOnlyList<string> Nodes, string? Error);

    /// <summary>Each plugin gets its own load context so its private dependencies stay private, while the host's assemblies (the engine above all) are shared: a node class must be the host's own <c>NodeAttribute</c> type.</summary>
    sealed class PluginContext(string path) : AssemblyLoadContext(Path.GetFileNameWithoutExtension(path))
    {
        readonly AssemblyDependencyResolver _resolver = new(path);

        protected override Assembly? Load(AssemblyName name)
        {
            var shared = Default.Assemblies.FirstOrDefault(a => a.GetName().Name == name.Name);
            if (shared is not null) return shared;
            var found = _resolver.ResolveAssemblyToPath(name);
            return found is null ? null : LoadFromAssemblyPath(found);
        }
    }

    public static IReadOnlyList<Loaded> Load(NodeRegistry registry, string folder)
    {
        var results = new List<Loaded>();
        if (!Directory.Exists(folder)) return results;
        var candidates = Directory.EnumerateFiles(folder, "*.dll", SearchOption.TopDirectoryOnly).ToList();
        foreach (var dir in Directory.EnumerateDirectories(folder))
        {
            var main = Path.Combine(dir, Path.GetFileName(dir) + ".dll");
            if (File.Exists(main)) candidates.Add(main);
        }
        foreach (var path in candidates.OrderBy(p => p, StringComparer.OrdinalIgnoreCase))
        {
            try
            {
                var assembly = new PluginContext(path).LoadFromAssemblyPath(path);
                var before = registry.Schemas.Select(s => s.Id).ToHashSet();
                registry.AddAssembly(assembly);
                var added = registry.Schemas.Select(s => s.Id).Where(id => !before.Contains(id)).OrderBy(id => id).ToList();
                results.Add(new Loaded(path, added, null));
            }
            catch (Exception ex) when (ex is BadImageFormatException or FileLoadException or FileNotFoundException or InvalidOperationException or ReflectionTypeLoadException or TypeLoadException)
            {
                results.Add(new Loaded(path, [], ex.Message)); // one broken pack must not stop the server
            }
        }
        return results;
    }
}
