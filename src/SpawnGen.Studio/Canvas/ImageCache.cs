using SpawnDev.SpawnJS.JSObjects;

namespace SpawnGen.Studio.Canvas;

/// <summary>Loads result pictures once and keeps them, so the canvas can draw them every frame without refetching.</summary>
public sealed class ImageCache : IDisposable
{
    sealed class Entry(HTMLImageElement image)
    {
        public HTMLImageElement Image { get; } = image;
        public bool Loaded { get; set; }
    }

    readonly Dictionary<string, Entry> _items = [];

    /// <summary>The picture for <paramref name="url"/>, or null while it is still loading. <paramref name="onLoaded"/> fires once it has (with its natural size).</summary>
    public HTMLImageElement? Get(string url, Action<double, double> onLoaded)
    {
        if (_items.TryGetValue(url, out var entry)) return entry.Loaded ? entry.Image : null;

        var image = new HTMLImageElement();
        entry = new Entry(image);
        _items[url] = entry;
        image.OnLoad += e =>
        {
            entry.Loaded = true;
            e.Dispose();
            onLoaded(image.NaturalWidth, image.NaturalHeight);
        };
        image.Src = url;
        return null;
    }

    public void Dispose()
    {
        foreach (var e in _items.Values) e.Image.Dispose();
        _items.Clear();
    }
}
