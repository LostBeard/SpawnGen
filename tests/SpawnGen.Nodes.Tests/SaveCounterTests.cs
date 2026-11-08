using System.Text.Json;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Output file numbering against ComfyUI's own folder_paths.get_save_image_path (Fixtures/comfy/save_counter.json, make_save_counter.py, made on Windows):
/// one counter for every name of the prefix in the folder, whatever its extension, trailing "_" or kind (folders count).
/// </summary>
public class SaveCounterTests
{
    static JsonElement Fixture => JsonDocument.Parse(File.ReadAllText(Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "comfy", "save_counter.json"))).RootElement;

    [Test]
    public void The_counter_is_ComfyUIs()
    {
        var fixture = Fixture;
        bool madeOnWindows = fixture.GetProperty("platform").GetString() == "win32";
        foreach (var c in fixture.GetProperty("cases").EnumerateArray())
        {
            var names = c.GetProperty("names").EnumerateArray().Select(n => n.GetString()!).ToList();
            string prefix = c.GetProperty("prefix").GetString()!;
            // os.path.normcase ignores case only on Windows: a name that matches the prefix only when case is ignored is checked where the fixture was made.
            bool dependsOnCase = names.Any(n => !n.StartsWith(prefix + "_", StringComparison.Ordinal) && n.StartsWith(prefix + "_", StringComparison.OrdinalIgnoreCase));
            if (dependsOnCase && madeOnWindows != OperatingSystem.IsWindows()) continue;
            Assert.That((long)SavePrefix.NextCounter(names, prefix), Is.EqualTo(c.GetProperty("counter").GetInt64()), $"{prefix}: {string.Join(", ", names)}");
        }
    }

    [Test]
    public void The_number_is_formatted_as_Pythons_05()
    {
        Assert.That(SavePrefix.FileName("a", 7, "png"), Is.EqualTo("a_00007_.png"));
        Assert.That(SavePrefix.FileName("a", 123456, "flac", trailingUnderscore: false), Is.EqualTo("a_123456.flac"));
        Assert.That(SavePrefix.FileName("a", -2, "png"), Is.EqualTo("a_-0002_.png"), "f\"{-2:05}\" is -0002");
    }

    [Test]
    public void Both_stores_number_every_file_of_a_prefix_together()
    {
        var dir = Path.Combine(Path.GetTempPath(), "spawngen-counter-" + Guid.NewGuid().ToString("N"));
        try
        {
            IResultStore[] stores = [new MemoryResultStore(), new DiskResultStore(new OutputOptions { Directory = Path.Combine(dir, "output"), TempDirectory = Path.Combine(dir, "temp") })];
            foreach (var store in stores)
            {
                var a = store.SaveAsync("output", "audio/clip", [1], "flac", trailingUnderscore: false).Result;
                var b = store.SaveAsync("output", "audio/clip", [2]).Result;
                var c = store.SaveAsync("output", "audio/clip", [3], "txt", trailingUnderscore: false).Result;
                Assert.That((a.Filename, b.Filename, c.Filename), Is.EqualTo(("clip_00001.flac", "clip_00002_.png", "clip_00003.txt")), store.GetType().Name);
            }
        }
        finally { if (Directory.Exists(dir)) Directory.Delete(dir, true); }
    }
}
