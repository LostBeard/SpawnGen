using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Load Image, Load Image (as Mask), Save Image, Preview Image and Compare Images against black-box measurements of ComfyUI:
/// Fixtures/image_nodes/load_image_modes.json.gz (tools/parity/make_image_node_fixtures.py) for the picture modes, and the file names, PNG text and
/// ui entries ComfyUI's server wrote for a prompt (noted at each test). ComfyUI fails on grey and 16-bit pictures (PyAV "Invalid argument"); SpawnGen
/// loads them (a documented difference in PARITY.md), checked here against Pillow.
/// </summary>
public class ImageNodeParityTests
{
    static readonly JsonElement Fixture = LoadFixture();

    static JsonElement LoadFixture()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "image_nodes", "load_image_modes.json.gz");
        using var gz = new GZipStream(File.OpenRead(path), CompressionMode.Decompress);
        return JsonDocument.Parse(gz).RootElement.Clone();
    }

    static IEnumerable<string> Files() => Fixture.EnumerateObject().Select(p => p.Name);

    static (int[] Shape, float[] Data) Tensor(JsonElement t) =>
        (t.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32()).ToArray(), t.GetProperty("data").EnumerateArray().Select(e => e.GetSingle()).ToArray());

    static async Task<T> WithInput<T>(string name, Func<Task<T>> run)
    {
        var store = new MemoryInputStore();
        await store.SaveAsync(name, Convert.FromBase64String(Fixture.GetProperty(name).GetProperty("file").GetString()!));
        var saved = InputFiles.Store;
        InputFiles.Store = store;
        try { return await run(); }
        finally { InputFiles.Store = saved; }
    }

    static void Same(HostTensor ours, JsonElement expected, string what, double tolerance)
    {
        var (shape, data) = Tensor(expected);
        Assert.That(ours.Shape, Is.EqualTo(shape), what + ": shape");
        double worst = 0;
        for (int i = 0; i < data.Length; i++) worst = Math.Max(worst, Math.Abs(ours.Data[i] - data[i]));
        Assert.That(worst, Is.LessThanOrEqualTo(tolerance), what + ": largest difference");
    }

    [TestCaseSource(nameof(Files))]
    public async Task Load_image_and_its_channel_masks_match_ComfyUI_or_Pillow_where_ComfyUI_fails(string file)
    {
        var fx = Fixture.GetProperty(file);
        bool jpeg = file.EndsWith(".jpg");
        // A JPEG is decoded by SpawnGen's own decoder: within 3/255 of libjpeg-turbo here (as JpegTests).
        double tolerance = jpeg ? 3 / 255.0 : 0;
        var (image, mask) = await WithInput(file, () => new LoadImage().ExecuteAsync(file));
        if (!fx.TryGetProperty("error", out _))
        {
            Same(image, fx.GetProperty("image"), file + " IMAGE", tolerance);
            Same(mask, fx.GetProperty("mask"), file + " MASK", 0);
            foreach (var channel in (string[])["alpha", "red", "green", "blue"])
            {
                var ours = await WithInput(file, () => new LoadImageMask().ExecuteAsync(file, channel));
                Same(ours, fx.GetProperty("mask_" + channel), $"{file} mask {channel}", channel == "alpha" ? 0 : tolerance);
            }
            return;
        }

        // ComfyUI raises on this picture; SpawnGen loads it as Pillow sees it (a 16-bit grey picture: its high bytes).
        Assert.That(fx.GetProperty("error").GetString(), Does.StartWith("ArgumentError"));
        int n = image.Height * image.Width;
        if (fx.GetProperty("pillow_rgba").ValueKind == JsonValueKind.Null)
        {
            var high = fx.GetProperty("high_byte").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            Assert.That(high, Has.Length.EqualTo(n));
            for (int i = 0; i < n; i++)
                for (int c = 0; c < 3; c++) Assert.That(image.Data[i * 3 + c], Is.EqualTo(high[i] / 255f), $"{file} pixel {i}");
            Assert.That(mask.Shape, Is.EqualTo(new[] { 1, 64, 64 }));
            return;
        }
        var rgba = fx.GetProperty("pillow_rgba").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        Assert.That(rgba, Has.Length.EqualTo(n * 4), file + ": size");
        bool hasAlpha = file is "la.png" or "rgba16.png";
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < 3; c++) Assert.That(image.Data[i * 3 + c], Is.EqualTo(rgba[i * 4 + c] / 255f).Within(tolerance), $"{file} pixel {i}");
            if (hasAlpha) Assert.That(mask.Data[i], Is.EqualTo(1f - rgba[i * 4 + 3] / 255f), $"{file} mask {i}");
        }
        Assert.That(mask.Shape, Is.EqualTo(hasAlpha ? new[] { 1, image.Height, image.Width } : new[] { 1, 64, 64 }));
    }

    [Test]
    public async Task Load_image_as_mask_gives_every_frame()
    {
        // Measured with ComfyUI's server: Load Image (as Mask) of the 4-frame APNG over_none.png gives 4 masks; of rgb.png (no alpha) 4 empty 64 x 64 ones.
        var dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "codecs", "apng");
        var store = new MemoryInputStore();
        foreach (var f in (string[])["over_none.png", "rgb.png"]) await store.SaveAsync(f, File.ReadAllBytes(Path.Combine(dir, f)));
        var saved = InputFiles.Store;
        InputFiles.Store = store;
        try
        {
            var (_, loadMask) = await new LoadImage().ExecuteAsync("over_none.png");
            var alpha = await new LoadImageMask().ExecuteAsync("over_none.png", "alpha");
            Assert.That(alpha.Shape, Is.EqualTo(new[] { 4, 16, 24 }));
            Assert.That(alpha.Data, Is.EqualTo(loadMask.Data), "the same mask as Load Image's");
            Assert.That((await new LoadImageMask().ExecuteAsync("over_none.png", "red")).Shape, Is.EqualTo(new[] { 4, 16, 24 }));
            Assert.That((await new LoadImageMask().ExecuteAsync("rgb.png", "alpha")).Shape, Is.EqualTo(new[] { 4, 64, 64 }));
        }
        finally { InputFiles.Store = saved; }
    }

    // ---- saving ----

    sealed class StoreServices(IResultStore store) : IServiceProvider
    {
        public object? GetService(Type serviceType) => serviceType == typeof(IResultStore) ? store : null;
    }

    static NodeContext Context(IServiceProvider services) =>
        (NodeContext)Activator.CreateInstance(typeof(NodeContext), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, null,
            ["1", "X", services, CancellationToken.None, null, new Lazy<string>(() => "{\"1\": {}}"), new Dictionary<string, string> { ["workflow"] = "{\"w\": 1}" }, null, null, 0], null)!;

    static List<string> ChunkTypes(byte[] png, out Dictionary<string, string> text)
    {
        var types = new List<string>();
        text = [];
        for (int p = 8; p < png.Length;)
        {
            int len = BinaryPrimitives.ReadInt32BigEndian(png.AsSpan(p));
            var type = Encoding.ASCII.GetString(png, p + 4, 4);
            types.Add(type);
            if (type == "tEXt")
            {
                var body = png.AsSpan(p + 8, len);
                int zero = body.IndexOf((byte)0);
                text[Encoding.Latin1.GetString(body[..zero])] = Encoding.Latin1.GetString(body[(zero + 1)..]);
            }
            p += 12 + len;
        }
        return types;
    }

    static ImageBatch Frames(int count) => new(count, 2, 3, 3, [.. Enumerable.Range(0, count * 18).Select(i => i % 7 / 7f)]);

    [Test]
    public async Task Save_image_numbers_files_and_writes_the_prompt_then_the_workflow()
    {
        // Measured with ComfyUI's server (prompt with extra_pnginfo {"workflow": ...}): tEXt prompt, tEXt workflow, then the image; prefix "pre/x"
        // gives subfolder "pre" and x_00001_.png, x_00002_.png; a later run goes on counting.
        var store = new MemoryResultStore();
        var services = new StoreServices(store);
        var (passed, ui) = await new SaveImage().ExecuteAsync(Frames(2), Context(services), services, "pre/x");
        Assert.That(passed.Batch, Is.EqualTo(2));
        var files = ((List<object>)ui["images"]!).Cast<ResultFile>().ToArray();
        Assert.That(files.Select(f => (f.Filename, f.Subfolder, f.Type)), Is.EqualTo(new[] { ("x_00001_.png", "pre", "output"), ("x_00002_.png", "pre", "output") }));
        var types = ChunkTypes(store.Get("output", "x_00001_.png", "pre")!, out var text);
        Assert.That(types, Is.EqualTo(new[] { "IHDR", "tEXt", "tEXt", "IDAT", "IEND" }));
        Assert.That(text.Keys, Is.EqualTo(new[] { "prompt", "workflow" }));
        Assert.That((text["prompt"], text["workflow"]), Is.EqualTo(("{\"1\": {}}", "{\"w\": 1}")));
        var again = await new SaveImage().ExecuteAsync(Frames(1), Context(services), services, "pre/x");
        Assert.That(((ResultFile)((List<object>)again.Ui["images"]!)[0]).Filename, Is.EqualTo("x_00003_.png"));
    }

    [Test]
    public async Task Save_image_fills_batch_num_and_counts_for_the_prefix_as_written()
    {
        // Measured with ComfyUI's server: "bn_%batch_num%" with 4 pictures gives bn_0_00001_ .. bn_3_00004_, and the next run the same names again.
        var store = new MemoryResultStore();
        var services = new StoreServices(store);
        for (int run = 0; run < 2; run++)
        {
            var (_, ui) = await new SaveImage().ExecuteAsync(Frames(4), Context(services), services, "bn_%batch_num%");
            Assert.That(((List<object>)ui["images"]!).Cast<ResultFile>().Select(f => f.Filename),
                Is.EqualTo(new[] { "bn_0_00001_.png", "bn_1_00002_.png", "bn_2_00003_.png", "bn_3_00004_.png" }), $"run {run}");
        }
        var (_, sized) = await new SaveImage().ExecuteAsync(Frames(1), Context(services), services, "s_%width%_%height%");
        Assert.That(((ResultFile)((List<object>)sized["images"]!)[0]).Filename, Is.EqualTo("s_3_2_00001_.png"), "measured: x_%width%_%height% -> x_2_1_ for 2 x 1");
    }

    [Test]
    public async Task Previews_are_temporary_files_with_random_letters_and_only_preview_image_writes_the_prompt()
    {
        // Measured with ComfyUI's server: Preview Image -> ComfyUI_temp_{5 letters}_00001_.png .. with tEXt prompt and workflow; Compare Images ->
        // comfy.compare.a_temp_{letters}_... and comfy.compare.b_temp_{same letters}_..., Preview Mask -> ComfyUI_temp_..., both without text.
        // A folder store: the memory store names its temporary files its own way (by content).
        var root = Path.Combine(Path.GetTempPath(), "spawngen-previews-" + Guid.NewGuid().ToString("N"));
        var store = new DiskResultStore(new OutputOptions { Directory = Path.Combine(root, "output"), TempDirectory = Path.Combine(root, "temp") });
        var services = new StoreServices(store);
        byte[] Get(ResultFile f) => File.ReadAllBytes(Path.Combine(root, "temp", f.Subfolder, f.Filename));
        var (_, ui) = await new PreviewImage().ExecuteAsync(Frames(2), Context(services), services);
        var previews = ((List<object>)ui["images"]!).Cast<ResultFile>().ToArray();
        Assert.That(previews.Select(f => f.Type), Is.All.EqualTo("temp"));
        // The part after the prefix is the store's: ComfyUI counts (_00001_.png), SpawnGen's stores name temporary files by content (see Stores).
        var m = Regex.Match(previews[0].Filename, "^ComfyUI_temp_([a-z]{5})_");
        Assert.That(m.Success, Is.True, previews[0].Filename);
        Assert.That(previews[1].Filename, Does.StartWith($"ComfyUI_temp_{m.Groups[1].Value}_"));
        ChunkTypes(Get(previews[0]), out var text);
        Assert.That(text.Keys, Is.EqualTo(new[] { "prompt", "workflow" }));

        var compare = await new ImageCompare().ExecuteAsync(Context(services), services, null, Frames(2), Frames(1));
        var a = ((List<object>)compare["a_images"]!).Cast<ResultFile>().ToArray();
        var b = ((List<object>)compare["b_images"]!).Cast<ResultFile>().ToArray();
        var letters = Regex.Match(a[0].Filename, "^comfy\\.compare\\.a_temp_([a-z]{5})_").Groups[1].Value;
        Assert.That(letters, Has.Length.EqualTo(5), a[0].Filename);
        Assert.That(a, Has.Length.EqualTo(2));
        Assert.That(a[1].Filename, Does.StartWith($"comfy.compare.a_temp_{letters}_"));
        Assert.That(b.Single().Filename, Does.StartWith($"comfy.compare.b_temp_{letters}_"));
        Assert.That(ChunkTypes(Get(b[0]), out _), Does.Not.Contain("tEXt"));

        var (_, maskUi) = await new MaskPreview().ExecuteAsync(new MaskBatch(2, 2, 3), Context(services), services);
        var masks = ((List<object>)maskUi["images"]!).Cast<ResultFile>().ToArray();
        Assert.That(masks, Has.Length.EqualTo(2));
        Assert.That(masks[0].Filename, Does.Match("^ComfyUI_temp_[a-z]{5}_"));
        Assert.That(ChunkTypes(Get(masks[0]), out _), Does.Not.Contain("tEXt"));
    }

    [Test]
    public async Task Create_bounding_boxes_previews_the_first_background_picture()
    {
        // Measured with ComfyUI: a 2-picture background gives one preview, comfy.bboxes.background_temp_{letters}_00001_.png.
        var root = Path.Combine(Path.GetTempPath(), "spawngen-bbox-" + Guid.NewGuid().ToString("N"));
        var services = new StoreServices(new DiskResultStore(new OutputOptions { Directory = Path.Combine(root, "output"), TempDirectory = Path.Combine(root, "temp") }));
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly);
        var executor = new SpawnGen.Engine.Execution.PromptExecutor(registry, services);
        var result = await executor.ExecuteAsync(SpawnGen.Engine.Graph.PromptJson.Parse("""
        {
          "1": { "class_type": "EmptyImage", "inputs": { "width": 40, "height": 30, "batch_size": 2, "color": 0 } },
          "2": { "class_type": "CreateBoundingBoxes", "inputs": { "background": ["1", 0], "width": 64, "height": 64,
                 "editor_state": {"__value__": []}, "last_incoming": {"__value__": []} } }
        }
        """));
        var files = ((List<object>)result.Outputs["2"].Ui!["background_images"]!).Cast<ResultFile>().ToArray();
        Assert.That(files.Single().Filename, Does.Match("^comfy\\.bboxes\\.background_temp_[a-z]{5}_"));
    }
}
