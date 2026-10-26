using System.IO.Compression;
using System.Text;
using System.Text.Json.Nodes;
using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// The dataset nodes against ComfyUI's own results (comfy_extras/nodes_dataset.py). Fixtures/dataset/set is the input folder (make_inputs.py);
/// comfy_reference.py made comfy_direct.json.gz by calling ComfyUI's node classes and comfy_graphs.json.gz by running graph_*.json on a ComfyUI server.
/// Pictures Pillow makes are compared as their 8-bit values, the tensor maths bit for bit, texts and file names character for character.
/// </summary>
[NonParallelizable]
public class DatasetNodeTests
{
    static string Fixture(params string[] parts) => Path.Combine([TestContext.CurrentContext.TestDirectory, "Fixtures", "dataset", .. parts]);

    static readonly Lazy<JsonObject> Direct = new(() => Load("comfy_direct.json.gz"));
    static readonly Lazy<JsonObject> Graphs = new(() => Load("comfy_graphs.json.gz"));

    static JsonObject Load(string name)
    {
        using var gz = new GZipStream(File.OpenRead(Fixture(name)), CompressionMode.Decompress);
        return JsonNode.Parse(gz)!.AsObject();
    }

    static int[] Shape(JsonNode n) => [.. n["shape"]!.AsArray().Select(x => (int)x!)];

    /// <summary>A picture ComfyUI's Pillow steps made: its shape and 8-bit values.</summary>
    static (int[] Shape, byte[] Values) Comfy8(JsonNode n) => (Shape(n), Convert.FromBase64String((string)n["u8"]!));

    static float[] ComfyFloats(JsonNode n)
    {
        var raw = Convert.FromBase64String((string)n["f32"]!);
        var values = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, values, 0, raw.Length);
        return values;
    }

    /// <summary>Our picture as 8-bit values; every value must be exactly k / 255, as pil_to_tensor makes it.</summary>
    static byte[] Ours8(ImageBatch image)
    {
        var bytes = new byte[image.Data.Length];
        for (int i = 0; i < bytes.Length; i++)
        {
            float v = image.Data[i];
            int k = (int)MathF.Round(v * 255f);
            Assert.That(k is >= 0 and <= 255 && v == k / 255f, $"value {i} ({v}) is not an 8-bit value / 255");
            bytes[i] = (byte)k;
        }
        return bytes;
    }

    static void AssertSame(ImageBatch ours, JsonNode comfy, string what)
    {
        var (shape, values) = Comfy8(comfy);
        Assert.That(ours.Shape, Is.EqualTo(shape), what + ": shape");
        var mine = Ours8(ours);
        int differ = 0, worst = 0;
        for (int i = 0; i < values.Length; i++)
            if (mine[i] != values[i]) { differ++; worst = Math.Max(worst, Math.Abs(mine[i] - values[i])); }
        Assert.That(differ, Is.EqualTo(0), $"{what}: {differ} of {values.Length} values differ (worst by {worst})");
    }

    static ImageBatch Rgb(string name) => ImageDecoder.DecodePillowRgb(File.ReadAllBytes(Fixture("set", name)));

    static ImageBatch Rgba(string name)
    {
        var rgb = PngReader.Decode(File.ReadAllBytes(Fixture("set", name)), out var alpha);
        var image = new ImageBatch(1, rgb.Height, rgb.Width, 4);
        for (int p = 0; p < rgb.Height * rgb.Width; p++)
        {
            for (int c = 0; c < 3; c++) image.Data[p * 4 + c] = rgb.Data[p * 3 + c];
            image.Data[p * 4 + 3] = alpha![p];
        }
        return image;
    }

    static ImageBatch Concat(params ImageBatch[] frames)
    {
        var first = frames[0];
        var batch = new ImageBatch(frames.Length, first.Height, first.Width, first.Channels);
        for (int i = 0; i < frames.Length; i++) Array.Copy(frames[i].Data, 0, batch.Data, i * first.Data.Length, first.Data.Length);
        return batch;
    }

    static ImageBatch Mirrored(ImageBatch image)
    {
        var m = new ImageBatch(image.Batch, image.Height, image.Width, image.Channels);
        int w = image.Width, c = image.Channels;
        for (int row = 0; row < image.Batch * image.Height; row++)
            for (int x = 0; x < w; x++) Array.Copy(image.Data, (row * w + x) * c, m.Data, (row * w + (w - 1 - x)) * c, c);
        return m;
    }

    /// <summary>
    /// The inputs comfy_reference.py built: a wide, a tall, a square picture, an RGBA one, a batch of three, a picture whose values are not 8-bit
    /// ("soft") and an 8x8 one whose halves differ only by Pillow's grey rounding ("halves"); the last two are stored in the fixture.
    /// </summary>
    static Dictionary<string, ImageBatch> Inputs()
    {
        var a = Rgb("a.png");
        var soft = Direct.Value["inputs"]!["soft"]!;
        var (halvesShape, halves) = Comfy8(Direct.Value["inputs"]!["halves"]!);
        int[] s = Shape(soft);
        return new()
        {
            ["wide"] = a, ["tall"] = Rgb("B.PNG"), ["square"] = Rgb("plain/p.png"), ["rgba"] = Rgba("c.png"),
            ["batch"] = Concat(a, Rgb("a_twin.png"), Mirrored(a)),
            ["soft"] = new ImageBatch(s[0], s[1], s[2], s[3], ComfyFloats(soft)),
            ["halves"] = new ImageBatch(halvesShape[0], halvesShape[1], halvesShape[2], halvesShape[3], [.. halves.Select(v => v / 255f)]),
        };
    }

    static string InputDirectory() => Fixture();

    [SetUp]
    public void UseTheFixtureFolder() => InputFiles.Directory = InputDirectory();

    // ---- loading -------------------------------------------------------------------------------------------------------------------------------

    [Test]
    public async Task Loading_a_folder_gives_ComfyUIs_pictures_captions_and_order()
    {
        var comfy = Direct.Value;
        // os.listdir and .NET list a folder in the file system's order: the same names in the same order.
        Assert.That(Directory.EnumerateFileSystemEntries(Fixture("set")).Select(e => Path.GetFileName(e)), Is.EqualTo(comfy["listdir"]!.AsArray().Select(x => (string?)x)));
        Assert.That(LoadImageTextDataSetFromFolder.Folders(), Does.Contain("set").And.Contain("set/3_kohya").And.Contain("set/plain"));

        var (images, texts) = await new LoadImageTextDataSetFromFolder().ExecuteAsync("set");
        var expected = comfy["load_text"]!["images"]!.AsArray();
        Assert.That(images, Has.Count.EqualTo(expected.Count).And.Count.EqualTo(9), "3 kohya repeats, 5 pictures, the plain folder's one");
        for (int i = 0; i < images.Count; i++) AssertSame(images[i], expected[i]!, $"loaded picture {i}");
        Assert.That(texts, Is.EqualTo(comfy["load_text"]!["texts"]!.AsArray().Select(x => (string?)x)));
        Assert.That(texts[3], Is.EqualTo("A Photo of a Stra\u00dfe\nwith \u03a3\u0391\u03a3 and \u0130stanbul"), "CRLF read as \\n, stripped");
        Assert.That(texts[6], Does.StartWith("\ufeff"), "a BOM stays, as Python's utf-8 codec keeps it");

        var plain = await new LoadImageDataSetFromFolder().ExecuteAsync("set");
        var expectedPlain = comfy["load"]!.AsArray();
        Assert.That(plain, Has.Count.EqualTo(expectedPlain.Count).And.Count.EqualTo(5), "only the pictures directly in the folder");
        for (int i = 0; i < plain.Count; i++) AssertSame(plain[i], expectedPlain[i]!, $"loaded picture {i}");
        // The 16-bit grey PNG: Pillow clips each 16-bit value at 255 instead of scaling it, so only the first pixel (3) is not white.
        var grey = plain[4];
        Assert.That((grey.Width, grey.Height, grey.Data[0], grey.Data[3]), Is.EqualTo((9, 7, 3 / 255f, 1f)));
    }

    [Test]
    public void A_folder_outside_the_input_folder_or_without_pictures_is_refused()
    {
        Assert.That(() => new LoadImageDataSetFromFolder().ExecuteAsync("../dataset").GetAwaiter().GetResult(), Throws.ArgumentException.With.Message.Contains("outside"));
        Assert.That(() => new LoadImageDataSetFromFolder().ExecuteAsync("set/3_kohya/../../..").GetAwaiter().GetResult(), Throws.ArgumentException);
        var empty = Path.Combine(Path.GetTempPath(), "spawngen-dataset-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(empty, "nothing"));
        File.WriteAllText(Path.Combine(empty, "nothing", "readme.txt"), "x");
        try
        {
            InputFiles.Directory = empty;
            Assert.That(() => new LoadImageTextDataSetFromFolder().ExecuteAsync("nothing").GetAwaiter().GetResult(),
                Throws.ArgumentException.With.Message.EqualTo("No valid images found in input"));
        }
        finally { Directory.Delete(empty, true); }
    }

    // ---- picture steps ---------------------------------------------------------------------------------------------------------------------------

    static ImageBatch RunSingle(string node, ImageBatch input, JsonObject p) => node switch
    {
        "ResizeImagesByShorterEdge" => new ResizeImagesByShorterEdge().Execute(input, (long)p["shorter_edge"]!),
        "ResizeImagesByLongerEdge" => new ResizeImagesByLongerEdge().Execute(input, (long)p["longer_edge"]!),
        "CenterCropImages" => new CenterCropImages().Execute(input, (long)p["width"]!, (long)p["height"]!),
        "RandomCropImages" => new RandomCropImages().Execute(input, (long)p["width"]!, (long)p["height"]!, (long)p["seed"]!),
        "NormalizeImages" => new NormalizeImages().Execute(input, (double)p["mean"]!, (double)p["std"]!),
        "AdjustBrightness" => new AdjustBrightness().Execute(input, (double)p["factor"]!),
        "AdjustContrast" => new AdjustContrast().Execute(input, (double)p["factor"]!),
        _ => throw new ArgumentException(node),
    };

    [Test]
    public void Resizes_and_crops_give_ComfyUIs_pixels()
    {
        var inputs = Inputs();
        var cases = Direct.Value["single"]!.AsArray();
        Assert.That(cases, Has.Count.EqualTo(98));
        foreach (var c in cases)
        {
            var node = (string)c!["node"]!;
            var name = (string)c["input"]!;
            var p = c["params"]!.AsObject();
            AssertSame(RunSingle(node, inputs[name], p), c["output"]!, $"{node} {name} {p.ToJsonString()}");
        }
    }

    [Test]
    public void Normalize_brightness_and_contrast_are_ComfyUIs_float32_maths_bit_for_bit()
    {
        var cases = Direct.Value["maths"]!.AsArray();
        Assert.That(cases, Has.Count.EqualTo(28));
        foreach (var c in cases)
        {
            int channels = (int)c!["channels"]!;
            var ramp = new ImageBatch(1, 1, 256, channels);
            for (int k = 0; k < 256; k++) for (int ch = 0; ch < channels; ch++) ramp.Data[k * channels + ch] = k / 255f;
            var ours = RunSingle((string)c["node"]!, ramp, c["params"]!.AsObject());
            var theirs = ComfyFloats(c["output"]!);
            Assert.That(ours.Shape, Is.EqualTo(Shape(c["output"]!)));
            int differ = Enumerable.Range(0, theirs.Length).Count(i => BitConverter.SingleToInt32Bits(ours.Data[i]) != BitConverter.SingleToInt32Bits(theirs[i]));
            Assert.That(differ, Is.EqualTo(0), $"{c["node"]} {c["params"]!.ToJsonString()} on {channels} channels: {differ} values differ");
        }
    }

    // ---- lists -------------------------------------------------------------------------------------------------------------------------------------

    static ImageBatch Dot(int i, int batch = 1)
    {
        var d = new ImageBatch(batch, 1, 1, 3);
        for (int b = 0; b < batch; b++) d.Data.AsSpan(b * 3, 3).Fill((i + b) / 255f);
        return d;
    }

    static int First(ImageBatch frame) => (int)MathF.Round(frame.Data[0] * 255f);

    [Test]
    public void Shuffles_follow_numpys_permutation_for_every_seed()
    {
        // 1x1 pictures 0..8 (value i / 255) so where each went is plain; "dots_batched" has 0, 1, 2 as one batch.
        var dots = Enumerable.Range(0, 9).Select(i => Dot(i)).ToList();
        IReadOnlyList<ImageBatch> batched = [Dot(0, 3), .. dots.Skip(3)];
        foreach (var c in Direct.Value["shuffle"]!.AsArray())
        {
            long seed = (long)c!["seed"]!;
            var input = (string)c["input"]! == "dots" ? dots : batched;
            var ours = new ShuffleDataset().Execute(input, [seed]);
            Assert.That(ours.Select(First), Is.EqualTo(c["order"]!.AsArray().Select(x => (int)x!)), $"seed {seed} {c["input"]}");
            Assert.That(ours.All(f => f.Batch == 1), "every batch is split into frames");
        }
        var texts = Enumerable.Range(0, batched.Count).Select(i => (object?)$"text {i}").ToList();
        foreach (var c in Direct.Value["shuffle_pairs"]!.AsArray())
        {
            long seed = (long)c!["seed"]!;
            var (images, shuffled) = new ShuffleImageTextDataset().Execute(batched, texts, [seed]);
            Assert.That(shuffled, Is.EqualTo(c["texts"]!.AsArray().Select(x => (string?)x)), $"seed {seed}");
            Assert.That(images.Select(First), Is.EqualTo(c["first"]!.AsArray().Select(x => (int)x!)), $"seed {seed}");
            Assert.That(images.Select(i => i.Shape), Is.EqualTo(c["shapes"]!.AsArray().Select(s => s!.AsArray().Select(x => (int)x!).ToArray())), "a batch stays one item here");
        }
    }

    [Test]
    public async Task Deduplicate_Images_keeps_and_drops_what_ComfyUI_does()
    {
        var (loaded, _) = await new LoadImageTextDataSetFromFolder().ExecuteAsync("set");
        int Index(IReadOnlyList<ImageBatch> list, ImageBatch frame) => list.Select((f, i) => (f, i)).First(t => t.f.Data.AsSpan().SequenceEqual(frame.Data)).i;
        foreach (var c in Direct.Value["dedup"]!.AsArray())
        {
            double threshold = (double)c!["threshold"]!;
            var kept = new ImageDeduplication().Execute(loaded, [threshold]);
            Assert.That(kept.Select(k => Index(loaded, k)), Is.EqualTo(c["kept"]!.AsArray().Select(x => (int)x!)), $"threshold {threshold}");
        }

        // Every pair's hash distance, found with the node the way the reference found it with ComfyUI's: the smallest d whose 1 - d / 64 makes them duplicates.
        var inputs = Inputs();
        var frames = loaded.Concat([inputs["rgba"]]).Concat(Enumerable.Range(0, 3).Select(b => Frame(inputs["batch"], b))).ToList();
        var comfy = Direct.Value["dedup_distances"]!.AsArray();
        int checkedPairs = 0, nonZero = 0;
        for (int i = 0; i < frames.Count; i++)
            for (int j = 0; j < frames.Count; j++)
            {
                int lo = 0, hi = 64;
                while (lo < hi)
                {
                    int mid = (lo + hi) / 2;
                    if (new ImageDeduplication().Execute([frames[i], frames[j]], [1.0 - mid / 64.0]).Count == 1) hi = mid; else lo = mid + 1;
                }
                Assert.That(lo, Is.EqualTo((int)comfy[i]![j]!), $"hash distance of pictures {i} and {j}");
                checkedPairs++;
                if (lo > 0) nonZero++;
            }
        TestContext.Out.WriteLine($"{checkedPairs} pairs, {nonZero} with differing hashes");
        Assert.That(nonZero, Is.GreaterThan(100), "the pictures hash differently");

        // The hashes themselves, against ComfyUI's compute_hash run on the same frames and on the soft and halves pictures.
        var hashes = Direct.Value["dedup_hashes"]!.AsArray().Select(h => (string)h!).ToList();
        var all = frames.Concat([inputs["soft"], inputs["halves"]]).ToList();
        Assert.That(all.Select(f => Convert.ToString((long)ImageDeduplication.Hash(f), 2).PadLeft(64, '0')), Is.EqualTo(hashes));
        Assert.That(hashes[^1], Is.EqualTo(new string('0', 64)), "Pillow's grey rounds 100.71 up to 101, so the halves are alike");
    }

    static ImageBatch Frame(ImageBatch batch, int b) =>
        new(1, batch.Height, batch.Width, batch.Channels, batch.Data.AsSpan(b * batch.Height * batch.Width * batch.Channels, batch.Height * batch.Width * batch.Channels).ToArray());

    [Test]
    public async Task Make_Image_Grid_is_ComfyUIs_grid()
    {
        var (loaded, _) = await new LoadImageTextDataSetFromFolder().ExecuteAsync("set");
        var inputs = Inputs();
        foreach (var c in Direct.Value["grid"]!.AsArray())
        {
            IReadOnlyList<ImageBatch> items = (string)c!["input"]! == "loaded" ? loaded : [inputs["rgba"], inputs["batch"], inputs["tall"]];
            var grid = new ImageGrid().Execute(items, [(long)c["columns"]!], [(long)c["cell_width"]!], [(long)c["cell_height"]!], [(long)c["padding"]!]);
            AssertSame(grid, c["output"]!, $"grid of {c["input"]}");
        }
        Assert.That(() => new ImageGrid().Execute([], [4], [32], [32], [0]), Throws.ArgumentException.With.Message.EqualTo("Cannot create grid from empty image list"));
    }

    [Test]
    public void Numpys_generator_draws_what_numpy_draws()
    {
        // The crops above pin randint; these pin the seeding rule: numpy refuses seeds of 2^32 and more, the nodes take seed % (2**32 - 1) first.
        Assert.That(() => new NumpyRandom(4294967296), Throws.InstanceOf<ArgumentOutOfRangeException>());
        Assert.That(new NumpyRandom(4294967295 % 4294967295).Permutation(9), Is.EqualTo(new NumpyRandom(0).Permutation(9)));
        Assert.That(new NumpyRandom(5).RandInt(3, 4), Is.EqualTo(3), "a range of one value draws nothing");
    }

    // ---- text steps -------------------------------------------------------------------------------------------------------------------------------

    [Test]
    public void Text_steps_put_out_a_list_of_one_text_with_Pythons_string_rules()
    {
        Assert.That(new TextToUppercase().Execute("stra\u00dfe"), Is.EqualTo(new object[] { "STRASSE" }));
        Assert.That(new TextToLowercase().Execute("\u0130"), Is.EqualTo(new object[] { "i\u0307" }));
        Assert.That(new TruncateText().Execute("a\U0001F600bc", 2), Is.EqualTo(new object[] { "a\U0001F600" }), "code points, not UTF-16 units");
        Assert.That(new StripWhitespace().Execute("\u2003 x \u001f"), Is.EqualTo(new object[] { "x" }));
        Assert.That(new ReplaceText().Execute("ab", "", "-"), Is.EqualTo(new object[] { "-a-b-" }));
        // An extra input that brings a text node's output (a list of one text) is that text; anything else fails as Python's + does.
        Assert.That(new AddTextPrefix().Execute("x", new object?[] { "p:" }), Is.EqualTo(new object[] { "p:x" }));
        Assert.That(() => new AddTextSuffix().Execute("x", new object?[] { "a", "b" }), Throws.ArgumentException.With.Message.Contains("can only concatenate str (not \"list\") to str"));
        // Truncate Text slices a list as Python does.
        var sliced = new TruncateText().Execute(new object?[] { "ONE" }, 2);
        Assert.That(sliced.Single(), Is.EqualTo(new object[] { "ONE" }));
    }

    // ---- prompts on the engine, compared with what the ComfyUI server did --------------------------------------------------------------------

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }

    static NodeRegistry Registry() => new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly);

    /// <summary>A work folder with the input folder under the name the prompts use, and an empty output folder.</summary>
    static (string Input, string Output) Work()
    {
        var root = Path.Combine(Path.GetTempPath(), "spawngen-dataset-" + Guid.NewGuid().ToString("N"));
        var input = Path.Combine(root, "input");
        CopyFolder(Fixture("set"), Path.Combine(input, (string)Graphs.Value["input_folder"]!));
        var output = Path.Combine(root, "output");
        Directory.CreateDirectory(output);
        return (input, output);
    }

    static void CopyFolder(string from, string to)
    {
        Directory.CreateDirectory(to);
        foreach (var f in Directory.EnumerateFiles(from)) File.Copy(f, Path.Combine(to, Path.GetFileName(f)));
        foreach (var d in Directory.EnumerateDirectories(from)) CopyFolder(d, Path.Combine(to, Path.GetFileName(d)));
    }

    static PromptGraph Prompt(string name) => PromptJson.Parse(File.ReadAllText(Fixture($"graph_{name}.json")));

    static string[] UiTexts(ExecutionResult result, string node) => [.. ((System.Collections.IEnumerable)result.Outputs[node].Ui!["text"]!).Cast<string>()];

    /// <summary>Every file the prompts wrote, compared with ComfyUI's: the same names, PNGs with the same pixels and no text chunks, the same caption bytes.</summary>
    static void AssertFilesLikeComfy(string output, Func<string, bool> which)
    {
        var dst = Path.Combine(output, (string)Graphs.Value["output_folder"]!);
        var comfy = Graphs.Value["files"]!.AsObject().Where(kv => which(kv.Key)).ToDictionary(kv => kv.Key, kv => kv.Value!);
        var ours = Directory.Exists(dst)
            ? Directory.EnumerateFiles(dst, "*", SearchOption.AllDirectories).Select(f => Path.GetRelativePath(dst, f).Replace('\\', '/')).Where(which).ToList()
            : [];
        Assert.That(ours.Order(StringComparer.Ordinal), Is.EqualTo(comfy.Keys.Order(StringComparer.Ordinal)), "the same files");
        foreach (var (name, entry) in comfy)
        {
            var bytes = File.ReadAllBytes(Path.Combine(dst, name));
            if (entry["bytes"] is { } text)
            {
                // Python's text mode writes "\n" as the platform's line end: ComfyUI on this platform writes the same with Environment.NewLine.
                var caption = Encoding.UTF8.GetString(Convert.FromBase64String((string)text!)).Replace((string)Graphs.Value["linesep"]!, "\n").Replace("\n", Environment.NewLine);
                Assert.That(bytes, Is.EqualTo(Encoding.UTF8.GetBytes(caption)), name);
                continue;
            }
            Assert.That(PngReader.ReadText(bytes), Is.Empty, $"{name}: ComfyUI's dataset PNGs carry no text chunks");
            var rgb = PngReader.Decode(bytes, out var alpha);
            var mode = (string)entry["mode"]!;
            int channels = mode switch { "RGB" => 3, "RGBA" => 4, _ => throw new AssertionException($"{name}: mode {mode}") };
            Assert.That(alpha is not null, Is.EqualTo(channels == 4), $"{name}: alpha");
            var size = entry["size"]!.AsArray().Select(x => (int)x!).ToArray();
            Assert.That((rgb.Width, rgb.Height), Is.EqualTo((size[0], size[1])), name);
            var expected = Convert.FromBase64String((string)entry["u8"]!);
            int differ = 0;
            for (int p = 0; p < rgb.Width * rgb.Height; p++)
            {
                for (int c = 0; c < 3; c++) if ((byte)MathF.Round(rgb.Data[p * 3 + c] * 255f) != expected[p * channels + c]) differ++;
                if (channels == 4 && (byte)MathF.Round(alpha![p] * 255f) != expected[p * 4 + 3]) differ++;
            }
            Assert.That(differ, Is.EqualTo(0), $"{name}: {differ} values differ");
        }
    }

    [Test]
    public async Task The_main_prompt_shows_and_writes_what_ComfyUI_did()
    {
        var (input, output) = Work();
        try
        {
            InputFiles.Directory = input;
            var result = await new PromptExecutor(Registry(), new Services(new OutputOptions { Directory = output })).ExecuteAsync(Prompt("main"));
            var answer = Graphs.Value["answers"]!["main"]!;
            Assert.That((string)answer["status"]!, Is.EqualTo("success"));
            var ui = answer["ui"]!.AsObject();
            Assert.That(ui.Count, Is.EqualTo(13));
            foreach (var (node, texts) in ui)
                Assert.That(UiTexts(result, node), Is.EqualTo(texts!.AsArray().Select(t => (string?)t)), $"Preview as Text node {node}");
            AssertFilesLikeComfy(output, f => !f.StartsWith("inc/", StringComparison.Ordinal) && !f.StartsWith("bad/", StringComparison.Ordinal));
        }
        finally { Directory.Delete(Path.GetDirectoryName(input)!, true); }
    }

    [Test]
    public async Task Increment_mode_numbers_after_the_files_already_there_like_ComfyUI()
    {
        var (input, output) = Work();
        try
        {
            InputFiles.Directory = input;
            var executor = new PromptExecutor(Registry(), new Services(new OutputOptions { Directory = output }));
            await executor.ExecuteAsync(Prompt("main"));
            await executor.ExecuteAsync(Prompt("increment"));
            // Overwrite mode wrote image_00000 ... image_00008 (and captions); increment mode then counts on from 8 and again before every picture.
            AssertFilesLikeComfy(output, f => f.StartsWith("inc/", StringComparison.Ordinal));
        }
        finally { Directory.Delete(Path.GetDirectoryName(input)!, true); }
    }

    [Test]
    public async Task Captions_that_are_not_text_fail_where_ComfyUI_fails()
    {
        var (input, output) = Work();
        try
        {
            InputFiles.Directory = input;
            var answer = Graphs.Value["answers"]!["failing"]!;
            Assert.That((string)answer["status"]!, Is.EqualTo("error"));
            var ex = Assert.ThrowsAsync<NodeExecutionException>(() => new PromptExecutor(Registry(), new Services(new OutputOptions { Directory = output })).ExecuteAsync(Prompt("failing")));
            Assert.That(ex!.NodeId, Is.EqualTo("3"));
            Assert.That(ex.InnerException!.Message, Is.EqualTo(((string)answer["error"]!["message"]!).TrimEnd('\n')));
            // The pictures were written, then the first caption file was created empty before the write failed.
            AssertFilesLikeComfy(output, f => f.StartsWith("bad/", StringComparison.Ordinal));
        }
        finally { Directory.Delete(Path.GetDirectoryName(input)!, true); }
    }

    [Test]
    public void Captions_are_written_with_the_platforms_line_end_as_Pythons_text_mode_does()
    {
        // The reference ran on Windows, so its caption files have "\r\n" where the text has "\n": the file comparisons check the translation.
        Assert.That((string)Graphs.Value["linesep"]!, Is.EqualTo("\r\n"));
        Assert.That(Graphs.Value["files"]!.AsObject().Where(f => f.Key.EndsWith(".txt")).Any(f => Encoding.UTF8.GetString(Convert.FromBase64String((string)f.Value!["bytes"]!)).Contains("\r\n")));
    }

    // ---- stores and schemas ----------------------------------------------------------------------------------------------------------------------

    [Test]
    public async Task The_memory_stores_have_folders_too()
    {
        var store = new MemoryInputStore();
        foreach (var f in Directory.EnumerateFiles(Fixture("set"), "*", SearchOption.AllDirectories))
            await store.SaveAsync("set/" + Path.GetRelativePath(Fixture("set"), f).Replace('\\', '/'), File.ReadAllBytes(f));
        await store.SaveAsync("top.png", File.ReadAllBytes(Fixture("set", "a.png")));
        var previous = InputFiles.Store;
        try
        {
            InputFiles.Store = store;
            Assert.That(store.List(), Is.EqualTo(new[] { "top.png" }), "Load Image lists the top only");
            Assert.That(store.Folders(), Is.EqualTo(new[] { "set", "set/3_kohya", "set/plain" }));
            Assert.That(store.Exists("set/a.txt") && store.Exists("set/plain") && !store.Exists("set/b.txt"));
            var (images, texts) = await new LoadImageTextDataSetFromFolder().ExecuteAsync("set");
            var comfy = Direct.Value["load_text"]!;
            // The memory store lists names case-insensitively, as NTFS does, so the order is ComfyUI's on Windows.
            Assert.That(texts, Is.EqualTo(comfy["texts"]!.AsArray().Select(x => (string?)x)));
            for (int i = 0; i < images.Count; i++) AssertSame(images[i], comfy["images"]![i]!, $"picture {i}");
        }
        finally { InputFiles.Store = previous; }

        var results = new MemoryResultStore();
        Assert.That(results.List("output", "d"), Is.Null);
        Assert.That(() => results.WriteAsync("output", "d", "x.png", [1]).GetAwaiter().GetResult(), Throws.InstanceOf<DirectoryNotFoundException>());
        var services = new Services(results);
        await new SaveImageDataSetToFolder().ExecuteAsync(services, [Rgb("a.png"), Rgb("B.PNG")], ["d"], ["p"], ["increment"]);
        Assert.That(results.List("output", "d"), Is.EqualTo(new[] { "p_00001_00000.png", "p_00002_00001.png" }));
        Assert.That(PngReader.Decode(results.Get("output", "p_00002_00001.png", "d")!).Width, Is.EqualTo(20));
    }

    [Test]
    public void Saving_never_leaves_the_output_folder()
    {
        var results = new MemoryResultStore();
        var services = new Services(results);
        var one = new[] { Rgb("a.png") };
        Assert.That(() => new SaveImageDataSetToFolder().ExecuteAsync(services, one, ["../x"], ["p"], ["overwrite"]).GetAwaiter().GetResult(), Throws.ArgumentException);
        Assert.That(() => new SaveImageDataSetToFolder().ExecuteAsync(services, one, ["d"], ["../p"], ["overwrite"]).GetAwaiter().GetResult(),
            Throws.ArgumentException.With.Message.Contains("outside"));
        Assert.That(() => new SaveImageDataSetToFolder().ExecuteAsync(services, one, ["d"], ["/abs"], ["increment"]).GetAwaiter().GetResult(),
            Throws.ArgumentException.With.Message.Contains("outside"));
        Assert.That(() => new SaveImageDataSetToFolder().ExecuteAsync(services, [Concat(Rgb("a.png"), Rgb("a.png"))], ["d"], ["p"], ["overwrite"]).GetAwaiter().GetResult(),
            Throws.ArgumentException.With.Message.Contains("batch of 2"), "Pillow cannot save a batch as one picture");
    }

    [Test]
    public void Schemas_match_ComfyUIs_sockets_and_widgets()
    {
        var registry = Registry();
        var shuffle = registry.GetSchema("ShuffleImageTextDataset")!;
        Assert.That(shuffle.InputIsList && shuffle.Outputs.Select(o => (o.Name, o.Type.Name, o.IsList)).SequenceEqual([("images", "IMAGE", true), ("texts", "STRING", true)]));
        Assert.That((shuffle.FindInput("texts")!.Type.Name, shuffle.FindInput("texts")!.Widget), Is.EqualTo(("STRING", (WidgetSpec?)null)), "force_input: a socket only");
        var truncate = registry.GetSchema("TruncateText")!;
        Assert.That(truncate.FindInput("texts")!.Widget!.Kind, Is.EqualTo(WidgetKind.String), "a text widget that also takes a list");
        Assert.That(truncate.Outputs.Single().IsList, Is.False);
        var save = registry.GetSchema("SaveImageTextDataSetToFolder")!;
        Assert.That(save.IsOutputNode && save.Outputs.Count == 0);
        Assert.That(save.FindInput("texts")!.Optional && save.FindInput("texts")!.Widget is null);
        Assert.That(save.FindInput("mode")!.Widget!.Options, Is.EqualTo(new[] { "overwrite", "increment" }));
        Assert.That(registry.GetSchema("ImageGrid")!.Outputs.Single().IsList, Is.False, "one grid picture");
        Assert.That(registry.GetSchema("RandomCropImages")!.FindInput("seed")!.Widget!.Max, Is.EqualTo(18446744073709551615d));
    }
}
