using SpawnDev.SpawnJS;
using SpawnDev.SpawnJS.JSObjects;

namespace SpawnGen.Studio.Canvas;

/// <summary>Loads result pictures once and keeps them, so the canvas can draw them every frame without refetching.</summary>
public sealed class ImageCache(SpawnJSRuntime js) : IDisposable
{
    readonly Dictionary<string, HTMLCanvasElement?> _animations = [];

    /// <summary>
    /// An animated picture playing on a canvas of its own (index.html's spawngenAnimate: WebCodecs decodes a frame at a time), or null while the
    /// first frame is not ready, when the browser cannot play it, or when the file is still: then draw <see cref="Get"/>'s picture.
    /// </summary>
    public HTMLCanvasElement? GetAnimation(string url)
    {
        if (!_animations.TryGetValue(url, out var canvas))
            _animations[url] = canvas = js.Call<string, HTMLCanvasElement?>("spawngenAnimate", url);
        return canvas is not null && canvas.Width > 0 ? canvas : null;
    }

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
        foreach (var canvas in _animations.Values)
        {
            if (canvas is null) continue;
            js.CallVoid("spawngenAnimateStop", canvas);
            canvas.Dispose();
        }
        _animations.Clear();
    }
}
