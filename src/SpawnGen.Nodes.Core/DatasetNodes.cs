using System.Collections;
using System.Numerics;
using System.Text;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries, SpawnGen's dataset tests and fixtures (Fixtures/dataset, outputs of
// ComfyUI), and black-box measurements of ComfyUI's server (2026-11-11): folder walk and repeat counts, captions, the contrast pivot, batches in
// Deduplicate Images and Make Image Grid, and the saving rules. Pillow (MIT-CMU) behavior comes from PilLanczos; numpy's legacy generator (BSD)
// from NumpyRandom.

/// <summary>Reading and writing the dataset folders, and the picture conversions the dataset nodes share.</summary>
static class Dataset
{
    /// <summary>The pictures a dataset folder offers (measured: PNG, JPG, JPEG and WEBP in any case; GIF and BMP are left out).</summary>
    public static bool IsPicture(string name) =>
        Path.GetExtension(name).ToLowerInvariant() is ".png" or ".jpg" or ".jpeg" or ".webp";

    public static string[] Folders() => [.. InputFiles.Store.Folders()];

    /// <summary>The folder as a path in the input store; it must be inside the input folder and exist.</summary>
    public static string InputFolder(string folder)
    {
        var path = StorePaths.Normalize(folder) ?? throw new ArgumentException($"The folder '{folder}' is outside the input folder.", nameof(folder));
        if (path.Length == 0 || !InputFiles.Store.Exists(path)) throw new ArgumentException($"There is no folder '{folder}' in the input folder.", nameof(folder));
        return path;
    }

    static string Join(string folder, string name) => folder.Length == 0 ? name : $"{folder}/{name}";

    /// <summary>A picture as Pillow's <c>convert("RGB")</c> gives it, one frame.</summary>
    public static async Task<ImageBatch> LoadPictureAsync(string path) => await ImageDecoder.DecodePillowRgbAsync(await InputFiles.Store.ReadAsync(path));

    /// <summary>
    /// The pictures directly in a folder, in listing order, each with the path its caption would have (same name, ".txt").
    /// </summary>
    public static IEnumerable<(string Picture, string Caption)> Pictures(string folder) =>
        InputFiles.Store.Entries(folder).Where(e => !e.IsFolder && IsPicture(e.Name))
            .Select(e => (Join(folder, e.Name), Join(folder, SavePrefix.WithoutExtension(e.Name) + ".txt")));

    /// <summary>
    /// How often a subfolder's pictures are loaded (measured): the part of its name before the first "_" (the whole name without one) when that
    /// is all digits ("3_kohya", "3", "007_x" give 3, 3, 7; "0_x" gives none); otherwise once ("-1_x", "+2_x", "2 _x", "x_y").
    /// </summary>
    public static int Repeats(string name)
    {
        int end = name.IndexOf('_');
        var head = end < 0 ? name : name[..end];
        if (head.Length == 0 || !head.All(char.IsDigit)) return 1;
        var n = BigInteger.Zero;
        foreach (var c in head) n = n * 10 + (int)char.GetNumericValue(c);
        return n > int.MaxValue ? int.MaxValue : (int)n;
    }

    /// <summary>A caption file's text: UTF-8 (a byte order mark stays a character), line ends read as "\n", whitespace stripped at both ends.</summary>
    public static string Caption(byte[] bytes) =>
        PyText.Strip(Encoding.UTF8.GetString(bytes).Replace("\r\n", "\n", StringComparison.Ordinal).Replace('\r', '\n'));

    public static ArgumentException NoPictures() => new("No valid images found in input");

    /// <summary>Each picture of every batch, one by one.</summary>
    public static List<ImageBatch> Frames(IEnumerable<ImageBatch> batches)
    {
        var frames = new List<ImageBatch>();
        foreach (var batch in batches)
            for (int b = 0; b < batch.Batch; b++) frames.Add(Frame(batch, b));
        return frames;
    }

    public static ImageBatch Frame(ImageBatch batch, int b)
    {
        int size = batch.Height * batch.Width * batch.Channels;
        return new ImageBatch(1, batch.Height, batch.Width, batch.Channels, batch.Data.AsSpan(b * size, size).ToArray());
    }

    /// <summary>The picture as Pillow sees it after the 8-bit conversion (255 * v, clamped, truncated) and back: values k / 255.</summary>
    public static ImageBatch EightBit(ImageBatch image) =>
        new(image.Batch, image.Height, image.Width, image.Channels, PilLanczos.FromBytes(PilLanczos.ToBytes(image.Data)));

    /// <summary>Every picture of the batch resized with Pillow's LANCZOS (RGBA premultiplied, as Pillow does).</summary>
    public static ImageBatch Resize(ImageBatch image, int width, int height)
    {
        var output = new ImageBatch(image.Batch, height, width, image.Channels);
        int src = image.Height * image.Width * image.Channels, dst = height * width * image.Channels;
        for (int b = 0; b < image.Batch; b++)
        {
            var frame = PilLanczos.Resize(image.Data.AsSpan(b * src, src).ToArray(), image.Width, image.Height, image.Channels, width, height);
            Array.Copy(frame, 0, output.Data, b * dst, dst);
        }
        return output;
    }

    /// <summary>The pixels of the rectangle (clipped to the picture) of every picture of the batch.</summary>
    public static ImageBatch Crop(ImageBatch image, int left, int top, int width, int height)
    {
        width = Math.Max(0, Math.Min(width, image.Width - left));
        height = Math.Max(0, Math.Min(height, image.Height - top));
        int c = image.Channels;
        var output = new ImageBatch(image.Batch, height, width, c);
        for (int b = 0; b < image.Batch; b++)
            for (int y = 0; y < height; y++)
                Array.Copy(image.Data, ((b * image.Height + top + y) * image.Width + left) * c, output.Data, ((b * height + y) * width) * c, width * c);
        return output;
    }

    /// <summary>numpy's legacy generator for a seed of the dataset nodes (0 to 2^64 - 1): seeded with seed % (2^32 - 1), which numpy accepts.</summary>
    public static NumpyRandom Random(long seed) => new((long)((ulong)seed % 4294967295UL));

    /// <summary>The first value of a widget input of a list node (all of them when it is linked to a list).</summary>
    public static T First<T>(IReadOnlyList<T> values, string name) =>
        values.Count > 0 ? values[0] : throw new ArgumentException($"The input '{name}' has no value.", name);

    /// <summary>
    /// Writes the pictures to the dataset folder (inside the output folder, created) as PNGs without text: <c>{prefix}_{i:05}.png</c> in "overwrite"
    /// mode, <c>{prefix}_{counter:05}_{i:05}.png</c> in "increment" mode, where the counter is counted again before every picture from the names
    /// already in the folder (measured). Returns the folder written to and each picture's name without ".png".
    /// </summary>
    public static async Task<(string Folder, List<string> Names)> SaveAsync(IServiceProvider services, IReadOnlyList<ImageBatch> images, string folderName, string prefix, string mode)
    {
        var store = Results.For(services);
        var folder = StorePaths.Normalize(folderName) ?? throw new ArgumentException($"The folder '{folderName}' is outside the output folder.", nameof(folderName));
        var (nameFolder, name) = NameIn(folder, prefix);
        store.CreateFolder("output", nameFolder);
        var names = new List<string>();
        for (int i = 0; i < images.Count; i++)
        {
            var image = images[i];
            if (image.Batch != 1)
                throw new ArgumentException($"Cannot save a batch of {image.Batch} pictures as one PNG; each picture of a dataset must be a single image.", nameof(images));
            var stem = mode == "increment"
                ? $"{name}_{SavePrefix.Number5(SavePrefix.NextCounter(store.List("output", nameFolder) ?? [], name))}_{SavePrefix.Number5(i)}"
                : $"{name}_{SavePrefix.Number5(i)}";
            await store.WriteAsync("output", nameFolder, stem + ".png", PngWriter.Encode(image));
            names.Add(stem);
        }
        return (nameFolder, names);
    }

    /// <summary>The folder a file of <paramref name="prefix"/> lands in and its name: the prefix may not lead out of the dataset folder.</summary>
    static (string Folder, string Name) NameIn(string folder, string prefix)
    {
        var p = prefix.Replace('\\', '/');
        var joined = p.StartsWith('/') || (p.Length >= 2 && p[1] == ':') ? null : StorePaths.Normalize(folder.Length == 0 ? p : $"{folder}/{p}");
        if (joined is null || p.EndsWith('/') || (folder.Length > 0 && !joined.StartsWith(folder + "/", StringComparison.Ordinal)))
            throw new ArgumentException($"The file name prefix '{prefix}' leads outside the dataset folder.", nameof(prefix));
        int slash = joined.LastIndexOf('/');
        return (slash < 0 ? "" : joined[..slash], joined[(slash + 1)..]);
    }
}

[Node("LoadImageDataSetFromFolder", Category = "image", DisplayName = "Load Image (from Folder)", OutputIsList = [true], OutputNames = ["images"],
    Description = "Load a dataset of images from a specified folder and return a list of images. Supported formats: PNG, JPG, JPEG, WEBP.")]
public sealed class LoadImageDataSetFromFolder
{
    public static string[] Folders() => Dataset.Folders();

    /// <summary>The pictures directly in the folder, in listing order (measured: subfolders are not read).</summary>
    public async Task<IReadOnlyList<ImageBatch>> ExecuteAsync([Input(OptionsProvider = nameof(Folders), Tooltip = "The folder to load images from.")] string folder)
    {
        var path = Dataset.InputFolder(folder);
        var images = new List<ImageBatch>();
        foreach (var (picture, _) in Dataset.Pictures(path)) images.Add(await Dataset.LoadPictureAsync(picture));
        return images.Count > 0 ? images : throw Dataset.NoPictures();
    }
}

[Node("LoadImageTextDataSetFromFolder", Category = "image", DisplayName = "Load Image-Text (from Folder)", OutputIsList = [true, true], OutputNames = ["images", "texts"],
    Description = "Load a dataset of pairs of images and text captions from a specified folder and return them as a list. Supported formats: PNG, JPG, JPEG, WEBP.")]
public sealed class LoadImageTextDataSetFromFolder
{
    public static string[] Folders() => Dataset.Folders();

    /// <summary>
    /// Measured with ComfyUI: the folder is read in listing order; a picture comes with the text of the ".txt" file of the same name ("" without
    /// one), and a subfolder (one level down) is read in its place, its pictures repeated as <see cref="Dataset.Repeats"/> says, the whole
    /// subfolder at a time (a, b, a, b).
    /// </summary>
    public async Task<(IReadOnlyList<ImageBatch> Images, IReadOnlyList<string> Texts)> ExecuteAsync(
        [Input(OptionsProvider = nameof(Folders), Tooltip = "The folder to load images and text captions from.")] string folder)
    {
        var path = Dataset.InputFolder(folder);
        var images = new List<ImageBatch>();
        var texts = new List<string>();
        async Task AddAsync(IEnumerable<(string Picture, string Caption)> pictures, int times)
        {
            var loaded = new List<(ImageBatch, string)>();
            foreach (var (picture, caption) in pictures)
                loaded.Add((await Dataset.LoadPictureAsync(picture),
                    InputFiles.Store.Exists(caption) ? Dataset.Caption(await InputFiles.Store.ReadAsync(caption)) : ""));
            for (int t = 0; t < times; t++)
                foreach (var (image, text) in loaded) { images.Add(image); texts.Add(text); }
        }
        foreach (var entry in InputFiles.Store.Entries(path))
        {
            if (entry.IsFolder) await AddAsync(Dataset.Pictures($"{path}/{entry.Name}"), Dataset.Repeats(entry.Name));
            else if (Dataset.IsPicture(entry.Name))
                await AddAsync([($"{path}/{entry.Name}", $"{path}/{SavePrefix.WithoutExtension(entry.Name)}.txt")], 1);
        }
        return images.Count > 0 ? (images, texts) : throw Dataset.NoPictures();
    }
}

[Node("SaveImageDataSetToFolder", Category = "image", DisplayName = "Save Image (to Folder) (DEPRECATED)", IsOutput = true, InputIsList = true,
    Description = "Save a dataset of images to a specified folder. Supported formats: PNG.")]
public sealed class SaveImageDataSetToFolder
{
    public async Task ExecuteAsync(IServiceProvider services,
        [Input(Tooltip = "List of images to save.")] IReadOnlyList<ImageBatch> images,
        [Input(Default = "dataset", Tooltip = "Name of the folder to save images to (inside output directory).")] IReadOnlyList<string> folderName,
        [Input(Default = "image", Tooltip = "Prefix for saved image filenames.")] IReadOnlyList<string> filenamePrefix,
        [Input(Default = "overwrite", Options = ["overwrite", "increment"], Tooltip = "Whether to overwrite existing files or increment filenames to avoid overwriting.")]
        IReadOnlyList<string> mode) =>
        await Dataset.SaveAsync(services, images, Dataset.First(folderName, "folder_name"), Dataset.First(filenamePrefix, "filename_prefix"), Dataset.First(mode, "mode"));
}

[Node("SaveImageTextDataSetToFolder", Category = "image", DisplayName = "Save Image-Text (to Folder)", IsOutput = true, InputIsList = true,
    Description = "Save a dataset of pairs of images and text captions to a specified folder. Images are saved as PNG files and captions are saved as TXT files with the same filename_prefix.")]
public sealed class SaveImageTextDataSetToFolder
{
    /// <summary>
    /// The pictures as Save Image (to Folder) writes them, then picture i's caption as "{its name}.txt" for each text given (measured: fewer texts
    /// than pictures leave the rest without one), UTF-8 with the platform's line ends, as Python's text mode writes.
    /// </summary>
    public async Task ExecuteAsync(IServiceProvider services,
        [Input(Tooltip = "List of images to save.")] IReadOnlyList<ImageBatch> images,
        [Input(Default = "dataset", Tooltip = "Name of the folder to save images to (inside output directory).")] IReadOnlyList<string> folderName,
        [Input(Default = "image", Tooltip = "Prefix for saved image filenames.")] IReadOnlyList<string> filenamePrefix,
        [Input(Default = "overwrite", Options = ["overwrite", "increment"], Tooltip = "Whether to overwrite existing files or increment filenames to avoid overwriting.")]
        IReadOnlyList<string> mode,
        [Input(Type = "STRING", ForceInput = true, Optional = true, Tooltip = "List of text captions to save.")] IReadOnlyList<object?>? texts = null)
    {
        var (subfolder, names) = await Dataset.SaveAsync(services, images, Dataset.First(folderName, "folder_name"),
            Dataset.First(filenamePrefix, "filename_prefix"), Dataset.First(mode, "mode"));
        if (texts is null) return;
        var store = Results.For(services);
        for (int i = 0; i < Math.Min(names.Count, texts.Count); i++)
        {
            if (texts[i] is not string text)
                throw new ArgumentException($"write() argument must be str, not {TextSteps.TypeName(texts[i])}", nameof(texts));
            await store.WriteAsync("output", subfolder, names[i] + ".txt", new UTF8Encoding(false).GetBytes(text.Replace("\n", Environment.NewLine, StringComparison.Ordinal)));
        }
    }
}

[Node("ResizeImagesByShorterEdge", Category = "image/transform", DisplayName = "Resize Images by Shorter Edge (DEPRECATED)", OutputNames = ["images"],
    Description = "Resize images so that the shorter edge matches the specified dimension while preserving aspect ratio.")]
public sealed class ResizeImagesByShorterEdge
{
    /// <summary>Measured: the other edge is <c>int(other * edge / shorter)</c> (truncated), resized with Pillow's LANCZOS from the 8-bit picture.</summary>
    public ImageBatch Execute([Input(Tooltip = "Image to process.")] ImageBatch images,
        [Input(Min = 1, Max = 8192, Tooltip = "Target dimension for the shorter edge.")] long shorterEdge = 512)
    {
        int w = images.Width, h = images.Height, e = (int)shorterEdge;
        return w < h ? Dataset.Resize(images, e, (int)((double)h * e / w)) : Dataset.Resize(images, (int)((double)w * e / h), e);
    }
}

[Node("ResizeImagesByLongerEdge", Category = "image/transform", DisplayName = "Resize Images by Longer Edge (DEPRECATED)", OutputNames = ["images"],
    Description = "Resize images so that the longer edge matches the specified dimension while preserving aspect ratio.")]
public sealed class ResizeImagesByLongerEdge
{
    /// <summary>As <see cref="ResizeImagesByShorterEdge"/>, the longer edge (the height of a square) set to <paramref name="longerEdge"/>.</summary>
    public ImageBatch Execute([Input(Tooltip = "Image to process.")] ImageBatch images,
        [Input(Min = 1, Max = 8192, Tooltip = "Target dimension for the longer edge.")] long longerEdge = 1024)
    {
        int w = images.Width, h = images.Height, e = (int)longerEdge;
        return w > h ? Dataset.Resize(images, e, (int)((double)h * e / w)) : Dataset.Resize(images, (int)((double)w * e / h), e);
    }
}

[Node("CenterCropImages", Category = "image/transform", DisplayName = "Crop Image (Center)", OutputNames = ["images"],
    Description = "Center crop an image to the specified dimensions.")]
public sealed class CenterCropImages
{
    /// <summary>Measured: the 8-bit picture's pixels from ((W - width) // 2, (H - height) // 2), at least 0; a crop larger than the picture keeps what there is.</summary>
    public ImageBatch Execute([Input(Tooltip = "Image to process.")] ImageBatch images,
        [Input(Min = 1, Max = 8192, Tooltip = "Crop width.")] long width = 512,
        [Input(Min = 1, Max = 8192, Tooltip = "Crop height.")] long height = 512)
    {
        int left = (int)Math.Max(0, Math.Floor((images.Width - width) / 2.0)), top = (int)Math.Max(0, Math.Floor((images.Height - height) / 2.0));
        return Dataset.Crop(Dataset.EightBit(images), left, top, (int)width, (int)height);
    }
}

[Node("RandomCropImages", Category = "image/transform", DisplayName = "Crop Image (Random)", OutputNames = ["images"],
    Description = "Randomly crop an image to the specified dimensions.")]
public sealed class RandomCropImages
{
    /// <summary>Measured: numpy's legacy generator draws the left edge from 0..max(0, W - width), then the top edge likewise; one rectangle for the batch.</summary>
    public ImageBatch Execute([Input(Tooltip = "Image to process.")] ImageBatch images,
        [Input(Min = 1, Max = 8192, Tooltip = "Crop width.")] long width = 512,
        [Input(Min = 1, Max = 8192, Tooltip = "Crop height.")] long height = 512,
        [Input(Min = 0, Max = 18446744073709551615d, Tooltip = "Random seed.")] long seed = 0)
    {
        var random = Dataset.Random(seed);
        long left = random.RandInt(0, Math.Max(0, images.Width - width) + 1);
        long top = random.RandInt(0, Math.Max(0, images.Height - height) + 1);
        return Dataset.Crop(Dataset.EightBit(images), (int)left, (int)top, (int)width, (int)height);
    }
}

/// <summary>The float32 colour steps: the first three channels change, a fourth (alpha) is kept (measured).</summary>
static class ColorSteps
{
    public static ImageBatch Map(ImageBatch images, Func<float, float> f)
    {
        var output = new ImageBatch(images.Batch, images.Height, images.Width, images.Channels, (float[])images.Data.Clone());
        int c = images.Channels, colors = Math.Min(3, c);
        for (int p = 0; p < output.Data.Length; p += c)
            for (int k = 0; k < colors; k++) output.Data[p + k] = f(output.Data[p + k]);
        return output;
    }

    public static float Clip(float v) => v < 0f ? 0f : v > 1f ? 1f : v;
}

[Node("NormalizeImages", Category = "image/color", DisplayName = "Normalize Image Colors", OutputNames = ["images"],
    Description = "Normalize images using mean and standard deviation.")]
public sealed class NormalizeImages
{
    /// <summary>(v - mean) / std in float32, not clamped.</summary>
    public ImageBatch Execute([Input(Tooltip = "Image to process.")] ImageBatch images,
        [Input(Min = 0.0, Max = 1.0, Tooltip = "Mean value for normalization.")] double mean = 0.5,
        [Input(Min = 0.001, Max = 1.0, Tooltip = "Standard deviation for normalization.")] double std = 0.5)
    {
        float m = (float)mean, s = (float)std;
        return ColorSteps.Map(images, v => (v - m) / s);
    }
}

[Node("AdjustBrightness", Category = "image/adjustments", DisplayName = "Adjust Brightness", OutputNames = ["images"],
    Description = "Adjust the brightness of an image.")]
public sealed class AdjustBrightness
{
    /// <summary>v * factor in float32, clamped to 0..1.</summary>
    public ImageBatch Execute([Input(Tooltip = "Image to process.")] ImageBatch images,
        [Input(Min = 0.0, Max = 2.0, Tooltip = "Brightness factor. 1.0 = no change, <1.0 = darker, >1.0 = brighter.")] double factor = 1.0)
    {
        float k = (float)factor;
        return ColorSteps.Map(images, v => ColorSteps.Clip(v * k));
    }
}

[Node("AdjustContrast", Category = "image/adjustments", DisplayName = "Adjust Contrast", OutputNames = ["images"],
    Description = "Adjust the contrast of an image.")]
public sealed class AdjustContrast
{
    /// <summary>(v - 0.5) * factor + 0.5 in float32, clamped to 0..1 (measured: the pivot is 0.5, not the picture's mean).</summary>
    public ImageBatch Execute([Input(Tooltip = "Image to process.")] ImageBatch images,
        [Input(Min = 0.0, Max = 2.0, Tooltip = "Contrast factor. 1.0 = no change, <1.0 = less contrast, >1.0 = more contrast.")] double factor = 1.0)
    {
        float k = (float)factor;
        return ColorSteps.Map(images, v => ColorSteps.Clip((v - 0.5f) * k + 0.5f));
    }
}

[Node("ShuffleDataset", Category = "image/batch", DisplayName = "Shuffle Images List", InputIsList = true, OutputIsList = [true], OutputNames = ["images"],
    Description = "Randomly shuffle the order of images in a list.")]
public sealed class ShuffleDataset
{
    /// <summary>Every picture of every batch, one by one, in the order of numpy's <c>permutation</c> for the seed.</summary>
    public IReadOnlyList<ImageBatch> Execute([Input(Tooltip = "List of images to process.")] IReadOnlyList<ImageBatch> images,
        [Input(Default = 0L, Min = 0, Max = 18446744073709551615d, Tooltip = "Random seed.")] IReadOnlyList<long> seed)
    {
        var frames = Dataset.Frames(images);
        return [.. Dataset.Random(Dataset.First(seed, "seed")).Permutation(frames.Count).Select(i => frames[i])];
    }
}

[Node("ShuffleImageTextDataset", Category = "image/batch", DisplayName = "Shuffle Pairs of Image-Text", InputIsList = true, OutputIsList = [true, true],
    OutputNames = ["images", "texts"], OutputTypes = [null, "STRING"], Description = "Randomly shuffle the order of pairs of image-text in a list.")]
public sealed class ShuffleImageTextDataset
{
    /// <summary>The pairs in the order of numpy's <c>permutation</c>; a batch stays one item. The lists must be as long as each other.</summary>
    public (IReadOnlyList<ImageBatch> Images, IReadOnlyList<object?> Texts) Execute(
        [Input(Tooltip = "List of images to shuffle.")] IReadOnlyList<ImageBatch> images,
        [Input(Type = "STRING", ForceInput = true, Tooltip = "List of texts to shuffle.")] IReadOnlyList<object?> texts,
        [Input(Default = 0L, Min = 0, Max = 18446744073709551615d, Tooltip = "Random seed.")] IReadOnlyList<long> seed)
    {
        if (texts.Count != images.Count) throw new ArgumentException($"There are {images.Count} images but {texts.Count} texts; each image needs its text.", nameof(texts));
        var order = Dataset.Random(Dataset.First(seed, "seed")).Permutation(images.Count);
        return ([.. order.Select(i => images[i])], [.. order.Select(i => texts[i])]);
    }
}

[Node("ImageDeduplication", Category = "image/batch", DisplayName = "Deduplicate Images", InputIsList = true, OutputIsList = [true], OutputNames = ["images"],
    Description = "Remove duplicate or very similar images from a list.")]
public sealed class ImageDeduplication
{
    /// <summary>
    /// Every picture (batches split, measured), in order, unless it is as similar as the threshold or more to a picture already kept; similarity
    /// is 1 - (differing bits of the two <see cref="Hash"/>es) / 64.
    /// </summary>
    public IReadOnlyList<ImageBatch> Execute([Input(Tooltip = "List of images to process.")] IReadOnlyList<ImageBatch> images,
        [Input(Default = 0.95, Min = 0.0, Max = 1.0, Tooltip = "Similarity threshold (0-1). Higher means more similar. Images above this threshold are considered duplicates.")]
        IReadOnlyList<double> similarityThreshold)
    {
        double threshold = Dataset.First(similarityThreshold, "similarity_threshold");
        var kept = new List<(ImageBatch Image, ulong Hash)>();
        foreach (var frame in Dataset.Frames(images))
        {
            var hash = Hash(frame);
            if (kept.Any(k => 1.0 - BitOperations.PopCount(k.Hash ^ hash) / 64.0 >= threshold)) continue;
            kept.Add((frame, hash));
        }
        return [.. kept.Select(k => k.Image)];
    }

    /// <summary>
    /// The average hash, measured against ComfyUI's: the 8-bit picture resized to 8 x 8 with Pillow's LANCZOS, made grey with Pillow's L conversion
    /// (ITU-R 601-2 luma in 16-bit fixed point, rounded), then one bit per pixel, row by row from the top bit: brighter than the mean.
    /// </summary>
    public static ulong Hash(ImageBatch frame)
    {
        int c = frame.Channels;
        var small = PilLanczos.Resize(PilLanczos.ToBytes(frame.Data.AsSpan(0, frame.Height * frame.Width * c).ToArray()), frame.Width, frame.Height, c, 8, 8);
        var grey = new int[64];
        for (int p = 0; p < 64; p++)
            grey[p] = c >= 3 ? (small[p * c] * 19595 + small[p * c + 1] * 38470 + small[p * c + 2] * 7471 + 0x8000) >> 16 : small[p * c];
        double mean = grey.Sum() / 64.0;
        ulong hash = 0;
        for (int p = 0; p < 64; p++) hash = (hash << 1) | (grey[p] > mean ? 1UL : 0UL);
        return hash;
    }
}

[Node("ImageGrid", Category = "image/batch", DisplayName = "Make Image Grid", InputIsList = true, OutputNames = ["images"],
    Description = "Arrange multiple images into a grid layout.")]
public sealed class ImageGrid
{
    /// <summary>
    /// Measured: every picture (batches split) stretched to the cell with Pillow's LANCZOS (an RGBA picture resized as RGBA, then its colour kept),
    /// row by row on a black RGB canvas of <c>columns</c> cells across (also when there are fewer pictures) and as many rows as needed, with
    /// <c>padding</c> pixels between cells.
    /// </summary>
    public ImageBatch Execute([Input(Tooltip = "List of images to process.")] IReadOnlyList<ImageBatch> images,
        [Input(Default = 4L, Min = 1, Max = 20, Tooltip = "Number of columns in the grid.")] IReadOnlyList<long> columns,
        [Input(Default = 256L, Min = 32, Max = 2048, Tooltip = "Width of each cell in the grid.")] IReadOnlyList<long> cellWidth,
        [Input(Default = 256L, Min = 32, Max = 2048, Tooltip = "Height of each cell in the grid.")] IReadOnlyList<long> cellHeight,
        [Input(Default = 4L, Min = 0, Max = 50, Tooltip = "Padding between images.")] IReadOnlyList<long> padding)
    {
        var frames = Dataset.Frames(images);
        if (frames.Count == 0) throw new ArgumentException("Cannot create grid from empty image list");
        int cols = (int)Dataset.First(columns, "columns"), cw = (int)Dataset.First(cellWidth, "cell_width"), ch = (int)Dataset.First(cellHeight, "cell_height");
        int pad = (int)Dataset.First(padding, "padding");
        int rows = (frames.Count + cols - 1) / cols;
        int width = cols * cw + (cols - 1) * pad, height = rows * ch + (rows - 1) * pad;
        var grid = new ImageBatch(1, height, width, 3);
        for (int i = 0; i < frames.Count; i++)
        {
            var f = frames[i];
            var cell = PilLanczos.Resize(PilLanczos.ToBytes(f.Data), f.Width, f.Height, f.Channels, cw, ch);
            int x0 = i % cols * (cw + pad), y0 = i / cols * (ch + pad);
            for (int y = 0; y < ch; y++)
                for (int x = 0; x < cw; x++)
                {
                    int s = (y * cw + x) * f.Channels, d = ((y0 + y) * width + x0 + x) * 3;
                    for (int k = 0; k < 3; k++) grid.Data[d + k] = cell[s + (f.Channels < 3 ? 0 : k)] / 255f;
                }
        }
        return grid;
    }
}

[Node("MergeImageLists", Category = "image/batch", DisplayName = "Merge Image Lists (DEPRECATED)", InputIsList = true, OutputIsList = [true], OutputNames = ["images"],
    Description = "Concatenate multiple image lists into one.")]
public sealed class MergeImageLists
{
    public IReadOnlyList<ImageBatch> Execute([Input(Tooltip = "List of images to process.")] IReadOnlyList<ImageBatch> images) => [.. images];
}

[Node("MergeTextLists", Category = "text", DisplayName = "Merge Text Lists (DEPRECATED)", InputIsList = true, OutputNames = ["texts"], OutputTypes = ["STRING"],
    Description = "Concatenate multiple text lists into one.")]
public sealed class MergeTextLists
{
    /// <summary>The texts as one value: a list (shown by Preview as Text as a JSON list).</summary>
    public object Execute([Input(Type = "STRING", Tooltip = "List of texts to process.")] IReadOnlyList<object?> texts)
    {
        var list = new ArrayValue();
        list.AddRange(texts);
        return list;
    }
}

/// <summary>The text steps: Python's string operations on the text, which may also be a list value (as Merge Text Lists puts out).</summary>
static class TextSteps
{
    public static string TypeName(object? value) => value switch
    {
        string => "str",
        null => "NoneType",
        IList => "list",
        _ => PyValues.TypeName(value),
    };

    /// <summary>A text, or a list of exactly one text, as the text (what an extra input brings); anything else stays as it is.</summary>
    public static object? Scalar(object? value) => value is IList { Count: 1 } list && list[0] is string s ? s : value;

    public static string Text(object? texts, string operation) =>
        texts as string ?? throw new ArgumentException($"'{TypeName(texts)}' object has no attribute '{operation}'", nameof(texts));

    /// <summary>Python's <c>a + b</c> for texts and lists.</summary>
    public static object Add(object? a, object? b)
    {
        if (a is string s && b is string t) return s + t;
        if (a is IList x && b is IList y)
        {
            var list = new ArrayValue();
            foreach (var v in x) list.Add(v);
            foreach (var v in y) list.Add(v);
            return list;
        }
        var kind = TypeName(a);
        throw new ArgumentException(kind is "str" or "list"
            ? $"can only concatenate {kind} (not \"{TypeName(b)}\") to {kind}"
            : $"unsupported operand type(s) for +: '{kind}' and '{TypeName(b)}'");
    }

    public const string Tooltip = "Text to process.";
    public const string Description = "Processed texts";
}

[Node("TextToLowercase", Category = "text", DisplayName = "Convert Text to Lowercase (DEPRECATED)", OutputNames = ["texts"], OutputTypes = ["STRING"],
    Description = "Convert text to lowercase.")]
public sealed class TextToLowercase
{
    public object Execute([Input(Type = "STRING", Tooltip = TextSteps.Tooltip)] object texts) => PyText.Lower(TextSteps.Text(texts, "lower"));
}

[Node("TextToUppercase", Category = "text", DisplayName = "Convert Text to Uppercase (DEPRECATED)", OutputNames = ["texts"], OutputTypes = ["STRING"],
    Description = "Convert text to uppercase.")]
public sealed class TextToUppercase
{
    public object Execute([Input(Type = "STRING", Tooltip = TextSteps.Tooltip)] object texts) => PyText.Upper(TextSteps.Text(texts, "upper"));
}

[Node("TruncateText", Category = "text", DisplayName = "Truncate Text", OutputNames = ["texts"], OutputTypes = ["STRING"],
    Description = "Truncate text to a maximum length.")]
public sealed class TruncateText
{
    /// <summary>The first <paramref name="maxLength"/> characters (code points) of a text, or items of a list.</summary>
    public object Execute([Input(Type = "STRING", Tooltip = TextSteps.Tooltip)] object texts,
        [Input(Min = 1, Max = 10000, Tooltip = "Maximum text length.")] long maxLength = 77)
    {
        if (texts is IList list)
        {
            var cut = new ArrayValue();
            for (int i = 0; i < Math.Min(list.Count, maxLength); i++) cut.Add(list[i]);
            return cut;
        }
        return PyText.Slice(TextSteps.Text(texts, "__getitem__"), null, maxLength);
    }
}

[Node("AddTextPrefix", Category = "text", DisplayName = "Add Text Prefix (DEPRECATED)", OutputNames = ["texts"], OutputTypes = ["STRING"],
    Description = "Add a prefix to all texts.")]
public sealed class AddTextPrefix
{
    public object Execute([Input(Type = "STRING", Tooltip = TextSteps.Tooltip)] object texts,
        [Input(Type = "STRING", Default = "", Tooltip = "Prefix to add.")] object? prefix = null) => TextSteps.Add(TextSteps.Scalar(prefix ?? ""), texts);
}

[Node("AddTextSuffix", Category = "text", DisplayName = "Add Text Suffix (DEPRECATED)", OutputNames = ["texts"], OutputTypes = ["STRING"],
    Description = "Add a suffix to all texts.")]
public sealed class AddTextSuffix
{
    public object Execute([Input(Type = "STRING", Tooltip = TextSteps.Tooltip)] object texts,
        [Input(Type = "STRING", Default = "", Tooltip = "Suffix to add.")] object? suffix = null) => TextSteps.Add(texts, TextSteps.Scalar(suffix ?? ""));
}

[Node("ReplaceText", Category = "text", DisplayName = "Replace Text (DEPRECATED)", OutputNames = ["texts"], OutputTypes = ["STRING"],
    Description = "Replace text in all texts.")]
public sealed class ReplaceText
{
    /// <summary>Python's <c>str.replace</c>: an empty find puts the replacement between every character and at both ends.</summary>
    public object Execute([Input(Type = "STRING", Tooltip = TextSteps.Tooltip)] object texts,
        [Input(Type = "STRING", Default = "", Tooltip = "Text to find.")] object? find = null,
        [Input(Type = "STRING", Default = "", Tooltip = "Text to replace with.")] object? replace = null)
    {
        var f = TextSteps.Scalar(find ?? "") as string ?? throw new ArgumentException($"replace() argument 1 must be str, not {TextSteps.TypeName(find)}", nameof(find));
        var r = TextSteps.Scalar(replace ?? "") as string ?? throw new ArgumentException($"replace() argument 2 must be str, not {TextSteps.TypeName(replace)}", nameof(replace));
        return PyText.Replace(TextSteps.Text(texts, "replace"), f, r);
    }
}

[Node("StripWhitespace", Category = "text", DisplayName = "Strip Whitespace (DEPRECATED)", OutputNames = ["texts"], OutputTypes = ["STRING"],
    Description = "Strip leading and trailing whitespace from all texts.")]
public sealed class StripWhitespace
{
    public object Execute([Input(Type = "STRING", Tooltip = TextSteps.Tooltip)] object texts) => PyText.Strip(TextSteps.Text(texts, "strip"));
}
