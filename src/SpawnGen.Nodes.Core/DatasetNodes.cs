using System.Numerics;
using System.Text;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// ComfyUI's dataset nodes (comfy_extras/nodes_dataset.py): folders of pictures and captions in and out, and the picture and text steps between.
// What the family shares, as ComfyUI does it:
// - Picture steps that go through Pillow quantise first (tensor_to_pil: np.clip(x * 255, 0, 255).astype(uint8), truncated) and come back as k / 255.
// - A node that is not a list node runs once per list item; given a batch it runs frame by frame and joins the frames (per_frame_process), except
//   the tensor maths (Normalize, Brightness, Contrast), which take the batch whole.
// - The list nodes (Shuffle Images List, Deduplicate Images, Make Image Grid, Merge Image Lists) first split every batch into single frames.
// - Seeds go through numpy's legacy generator: np.random.seed(seed % (2**32 - 1)).
// - A text node puts out a list holding its one text (ComfyUI's TextProcessingNode wraps the result in a list on an output that is not a list
//   output), so what flows on is that list: Preview as Text shows it as JSON, another text node's text input refuses it (Truncate Text slices it,
//   as Python slices a list), and an extra input (prefix, suffix, find, replace) unwraps a list of one.

/// <summary>Helpers the dataset nodes share.</summary>
internal static class Dataset
{
    /// <summary>
    /// tensor_to_pil's <c>Image.fromarray</c>: 2 channels become an LA picture, 3 RGB, 4 RGBA; anything else is refused with Pillow's TypeError.
    /// </summary>
    public static void CheckPillowChannels(int channels)
    {
        if (channels is < 2 or > 4) throw new ArgumentException($"Cannot handle this data type: (1, 1, {channels}), |u1");
    }

    /// <summary>ImageProcessingNode._ensure_image_list: every frame of every list item as a picture of its own, in order.</summary>
    public static List<ImageBatch> Frames(IReadOnlyList<ImageBatch> images)
    {
        var frames = new List<ImageBatch>();
        foreach (var item in images)
            for (int b = 0; b < item.Batch; b++)
                frames.Add(item.Batch == 1 ? item : new ImageBatch(1, item.Height, item.Width, item.Channels, Pixels.Frame(item, b)));
        return frames;
    }

    /// <summary>
    /// An individual picture node on a batch: the step runs on each frame (with the frame's width and height) and the frames are joined again
    /// (torch.cat), which needs every result to be the same size, as the steps here make them.
    /// </summary>
    public static ImageBatch EachFrame(ImageBatch images, Func<float[], (float[] Pixels, int Width, int Height)> step)
    {
        var frames = new List<float[]>(images.Batch);
        int w = 0, h = 0;
        for (int b = 0; b < images.Batch; b++)
        {
            var (px, fw, fh) = step(Pixels.Frame(images, b));
            if (b > 0 && (fw != w || fh != h)) throw new InvalidOperationException("Sizes of tensors must match to join the frames again.");
            (w, h) = (fw, fh);
            frames.Add(px);
        }
        return Pixels.From(frames, h, w, images.Channels);
    }

    /// <summary>The base classes' scalar extraction for a list node's extra inputs: a list of one is its value; a longer list stays a list and Python's maths on it fails.</summary>
    public static T One<T>(IReadOnlyList<T> values, string input) =>
        values.Count == 1 ? values[0] : throw new ArgumentException($"Input '{input}' got a list of {values.Count} values; this node takes one (Python: unsupported operand type(s) for 'list').");

    /// <summary>The same extraction for a node that runs per item: an extra input that brings a list of one (a text node's output) is its one item.</summary>
    public static object? Unwrap(object? value) => value is IReadOnlyList<object?> { Count: 1 } list ? list[0] : value;

    /// <summary>Python's name for a value's type, for the TypeErrors Python would raise.</summary>
    public static string PyTypeName(object? value) => value switch
    {
        null => "NoneType",
        string => "str",
        bool => "bool",
        long or int or BigInteger => "int",
        double or float => "float",
        System.Collections.IList => "list",
        _ => value.GetType().Name,
    };

    /// <summary><c>np.random.seed(seed % (2**32 - 1))</c>.</summary>
    public static NumpyRandom Seeded(long seed) => new(seed % 4294967295L);
}

// ---- picture steps, one list item at a time --------------------------------------------------------------------------------------------------------

[Node("ResizeImagesByShorterEdge", Category = "image/transform", DisplayName = "Resize Images by Shorter Edge (DEPRECATED)",
    Description = "Resize images so that the shorter edge matches the specified dimension while preserving aspect ratio.", OutputNames = ["images"])]
public sealed class ResizeImagesByShorterEdge
{
    /// <summary>Pillow's LANCZOS to the new size; the other edge is <c>int(edge * (shorter_edge / this edge))</c>, truncated. Ties count as wide.</summary>
    public ImageBatch Execute([Input(Tooltip = "Image to process.")] ImageBatch images,
        [Input(Min = 1, Max = 8192, Tooltip = "Target dimension for the shorter edge.")] long shorterEdge = 512)
    {
        Dataset.CheckPillowChannels(images.Channels);
        int w = images.Width, h = images.Height;
        var (newW, newH) = w < h ? ((int)shorterEdge, (int)(h * ((double)shorterEdge / w))) : ((int)(w * ((double)shorterEdge / h)), (int)shorterEdge);
        return Dataset.EachFrame(images, f => (DatasetPictures.Resize(f, w, h, images.Channels, newW, newH), newW, newH));
    }
}

[Node("ResizeImagesByLongerEdge", Category = "image/transform", DisplayName = "Resize Images by Longer Edge (DEPRECATED)",
    Description = "Resize images so that the longer edge matches the specified dimension while preserving aspect ratio.", OutputNames = ["images"])]
public sealed class ResizeImagesByLongerEdge
{
    /// <summary>Pillow's LANCZOS to the new size; the other edge is <c>int(edge * (longer_edge / this edge))</c>, truncated. Ties count as tall.</summary>
    public ImageBatch Execute([Input(Tooltip = "Image to process.")] ImageBatch images,
        [Input(Min = 1, Max = 8192, Tooltip = "Target dimension for the longer edge.")] long longerEdge = 1024)
    {
        Dataset.CheckPillowChannels(images.Channels);
        int w = images.Width, h = images.Height;
        var (newW, newH) = w > h ? ((int)longerEdge, (int)(h * ((double)longerEdge / w))) : ((int)(w * ((double)longerEdge / h)), (int)longerEdge);
        return Dataset.EachFrame(images, f => (DatasetPictures.Resize(f, w, h, images.Channels, newW, newH), newW, newH));
    }
}

[Node("CenterCropImages", Category = "image/transform", DisplayName = "Crop Image (Center)", Description = "Center crop an image to the specified dimensions.",
    OutputNames = ["images"])]
public sealed class CenterCropImages
{
    /// <summary>The middle part (left and top rounded down); a crop larger than the picture keeps the whole of that edge. 8-bit like every Pillow step.</summary>
    public ImageBatch Execute([Input(Tooltip = "Image to process.")] ImageBatch images,
        [Input(Min = 1, Max = 8192, Tooltip = "Crop width.")] long width = 512, [Input(Min = 1, Max = 8192, Tooltip = "Crop height.")] long height = 512)
    {
        Dataset.CheckPillowChannels(images.Channels);
        int w = images.Width, h = images.Height;
        int left = (int)Math.Max(0, (w - width) / 2), top = (int)Math.Max(0, (h - height) / 2); // only a positive difference is halved, so / floors here
        int right = (int)Math.Min(w, left + width), bottom = (int)Math.Min(h, top + height);
        return Dataset.EachFrame(images, f => (DatasetPictures.Crop(f, w, images.Channels, left, top, right, bottom), right - left, bottom - top));
    }
}

[Node("RandomCropImages", Category = "image/transform", DisplayName = "Crop Image (Random)", Description = "Randomly crop an image to the specified dimensions.",
    OutputNames = ["images"])]
public sealed class RandomCropImages
{
    /// <summary>
    /// numpy seeded with <c>seed % (2**32 - 1)</c>, then <c>randint(0, max_left + 1)</c> for the left edge and the same for the top (each drawn only
    /// when there is room to move). The generator is seeded again for every frame, so every frame of a batch gets the same crop.
    /// </summary>
    public ImageBatch Execute([Input(Tooltip = "Image to process.")] ImageBatch images,
        [Input(Min = 1, Max = 8192, Tooltip = "Crop width.")] long width = 512, [Input(Min = 1, Max = 8192, Tooltip = "Crop height.")] long height = 512,
        [Input(Min = 0, Max = 18446744073709551615d, Tooltip = "Random seed.")] long seed = 0)
    {
        Dataset.CheckPillowChannels(images.Channels);
        int w = images.Width, h = images.Height;
        var rng = Dataset.Seeded(seed);
        long maxLeft = Math.Max(0, w - width), maxTop = Math.Max(0, h - height);
        int left = maxLeft > 0 ? (int)rng.RandInt(0, maxLeft + 1) : 0;
        int top = maxTop > 0 ? (int)rng.RandInt(0, maxTop + 1) : 0;
        int right = (int)Math.Min(w, left + width), bottom = (int)Math.Min(h, top + height);
        return Dataset.EachFrame(images, f => (DatasetPictures.Crop(f, w, images.Channels, left, top, right, bottom), right - left, bottom - top));
    }
}

[Node("NormalizeImages", Category = "image/color", DisplayName = "Normalize Image Colors", Description = "Normalize images using mean and standard deviation.",
    OutputNames = ["images"])]
public sealed class NormalizeImages
{
    /// <summary><c>(image - mean) / std</c> in float32 over the whole batch, not clamped; a fourth channel (alpha) is kept as it was.</summary>
    public ImageBatch Execute([Input(Tooltip = "Image to process.")] ImageBatch images,
        [Input(Min = 0.0, Max = 1.0, Tooltip = "Mean value for normalization.")] double mean = 0.5,
        [Input(Min = 0.001, Max = 1.0, Tooltip = "Standard deviation for normalization.")] double std = 0.5)
    {
        float m = (float)mean, s = (float)std; // torch takes the Python floats into the float32 maths as float32
        return DatasetPictures.Elementwise(images, x => (x - m) / s);
    }
}

[Node("AdjustBrightness", Category = "image/adjustments", DisplayName = "Adjust Brightness", Description = "Adjust the brightness of an image.", OutputNames = ["images"])]
public sealed class AdjustBrightness
{
    /// <summary><c>(image * factor).clamp(0, 1)</c> in float32; a fourth channel (alpha) is kept as it was.</summary>
    public ImageBatch Execute([Input(Tooltip = "Image to process.")] ImageBatch images,
        [Input(Min = 0.0, Max = 2.0, Tooltip = "Brightness factor. 1.0 = no change, <1.0 = darker, >1.0 = brighter.")] double factor = 1.0)
    {
        float f = (float)factor;
        return DatasetPictures.Elementwise(images, x => Math.Clamp(x * f, 0f, 1f));
    }
}

[Node("AdjustContrast", Category = "image/adjustments", DisplayName = "Adjust Contrast", Description = "Adjust the contrast of an image.", OutputNames = ["images"])]
public sealed class AdjustContrast
{
    /// <summary><c>((image - 0.5) * factor + 0.5).clamp(0, 1)</c> in float32, one rounding per step; a fourth channel (alpha) is kept as it was.</summary>
    public ImageBatch Execute([Input(Tooltip = "Image to process.")] ImageBatch images,
        [Input(Min = 0.0, Max = 2.0, Tooltip = "Contrast factor. 1.0 = no change, <1.0 = less contrast, >1.0 = more contrast.")] double factor = 1.0)
    {
        float f = (float)factor;
        return DatasetPictures.Elementwise(images, x => Math.Clamp((x - 0.5f) * f + 0.5f, 0f, 1f));
    }
}

/// <summary>The picture steps the nodes above share.</summary>
internal static class DatasetPictures
{
    /// <summary>tensor_to_pil, Pillow's resize to (newW, newH) with LANCZOS, pil_to_tensor. Pillow refuses a size of 0.</summary>
    public static float[] Resize(float[] frame, int w, int h, int c, int newW, int newH)
    {
        if (newW <= 0 || newH <= 0) throw new ArgumentException("height and width must be > 0");
        return PilLanczos.Resize(frame, w, h, c, newW, newH);
    }

    /// <summary>tensor_to_pil, Pillow's crop of the box (left, top, right, bottom), pil_to_tensor.</summary>
    public static float[] Crop(float[] frame, int w, int c, int left, int top, int right, int bottom)
    {
        var px = PilLanczos.ToBytes(frame);
        int cw = right - left, ch = bottom - top;
        var cut = new byte[cw * ch * c];
        for (int y = 0; y < ch; y++) Array.Copy(px, ((top + y) * w + left) * c, cut, y * cw * c, cw * c);
        return PilLanczos.FromBytes(cut);
    }

    /// <summary>A float32 step on every value of the batch; with 4 channels the alpha values are put back afterwards.</summary>
    public static ImageBatch Elementwise(ImageBatch images, Func<float, float> step)
    {
        var result = new ImageBatch(images.Batch, images.Height, images.Width, images.Channels);
        var src = images.Data;
        var dst = result.Data;
        for (int i = 0; i < src.Length; i++) dst[i] = step(src[i]);
        if (images.Channels == 4)
            for (int i = 3; i < src.Length; i += 4) dst[i] = src[i];
        return result;
    }

    /// <summary>The picture as Pillow pixels at a new size (tensor_to_pil then resize with LANCZOS).</summary>
    public static byte[] PillowResized(ImageBatch frame, int newW, int newH)
    {
        CheckSize(newW, newH);
        return PilLanczos.Resize(PilLanczos.ToBytes(frame.Data), frame.Width, frame.Height, frame.Channels, newW, newH);
    }

    static void CheckSize(int w, int h)
    {
        if (w <= 0 || h <= 0) throw new ArgumentException("height and width must be > 0");
    }
}

// ---- picture lists ---------------------------------------------------------------------------------------------------------------------------------

[Node("ShuffleDataset", Category = "image/batch", DisplayName = "Shuffle Images List", Description = "Randomly shuffle the order of images in a list.",
    InputIsList = true, OutputIsList = [true], OutputNames = ["images"])]
public sealed class ShuffleDataset
{
    /// <summary>Every frame of the list (batches split) in numpy's <c>permutation(n)</c> order for <c>seed % (2**32 - 1)</c>.</summary>
    public IReadOnlyList<ImageBatch> Execute([Input(Tooltip = "List of images to process.")] IReadOnlyList<ImageBatch> images,
        [Input(Min = 0, Max = 18446744073709551615d, Default = 0L, Tooltip = "Random seed.")] IReadOnlyList<long> seed)
    {
        var frames = Dataset.Frames(images);
        var order = Dataset.Seeded(Dataset.One(seed, "seed")).Permutation(frames.Count);
        return [.. order.Select(i => frames[i])];
    }
}

[Node("ShuffleImageTextDataset", Category = "image/batch", DisplayName = "Shuffle Pairs of Image-Text", Description = "Randomly shuffle the order of pairs of image-text in a list.",
    InputIsList = true, OutputIsList = [true, true], OutputNames = ["images", "texts"], OutputTypes = [null, "STRING"])]
public sealed class ShuffleImageTextDataset
{
    /// <summary>
    /// The list items (a batch stays one item here) and the texts in the same numpy <c>permutation(len(images))</c> order; a text list shorter than
    /// the pictures fails as Python's indexing does, a longer one loses its extra texts.
    /// </summary>
    public (IReadOnlyList<ImageBatch> Images, IReadOnlyList<object?> Texts) Execute([Input(Tooltip = "List of images to shuffle.")] IReadOnlyList<ImageBatch> images,
        [Input(Type = "STRING", ForceInput = true, Tooltip = "List of texts to shuffle.")] IReadOnlyList<object?> texts,
        [Input(Min = 0, Max = 18446744073709551615d, Default = 0L, Tooltip = "Random seed.")] IReadOnlyList<long> seed)
    {
        var order = Dataset.Seeded(seed[0]).Permutation(images.Count);
        if (order.Length > 0 && texts.Count < images.Count) throw new ArgumentException($"list index out of range: {images.Count} pictures but {texts.Count} texts.", nameof(texts));
        return ([.. order.Select(i => images[i])], [.. order.Select(i => texts[i])]);
    }
}

[Node("ImageDeduplication", Category = "image/batch", DisplayName = "Deduplicate Images", Description = "Remove duplicate or very similar images from a list.",
    InputIsList = true, OutputIsList = [true], OutputNames = ["images"])]
public sealed class ImageDeduplication
{
    /// <summary>
    /// ComfyUI's average hash: each frame through Pillow at 8x8 (LANCZOS), to grey (Pillow's L: (19595 R + 38470 G + 7471 B + 0x8000) >> 16), one
    /// bit per pixel above the mean. Frames are kept in order unless one already kept is at least <c>similarity_threshold</c> alike
    /// (1 - differing bits / 64).
    /// </summary>
    public IReadOnlyList<ImageBatch> Execute([Input(Tooltip = "List of images to process.")] IReadOnlyList<ImageBatch> images,
        [Input(Min = 0.0, Max = 1.0, Default = 0.95, Tooltip = "Similarity threshold (0-1). Higher means more similar. Images above this threshold are considered duplicates.")]
        IReadOnlyList<double> similarityThreshold)
    {
        var frames = Dataset.Frames(images);
        double threshold = Dataset.One(similarityThreshold, "similarity_threshold");
        if (frames.Count == 0) return [];
        var hashes = frames.Select(Hash).ToList();
        var keep = new List<int>();
        for (int i = 0; i < frames.Count; i++)
        {
            bool duplicate = false;
            foreach (int j in keep)
            {
                double similarity = 1.0 - BitOperations.PopCount(hashes[i] ^ hashes[j]) / 64.0;
                if (similarity >= threshold) { duplicate = true; break; }
            }
            if (!duplicate) keep.Add(i);
        }
        return [.. keep.Select(i => frames[i])];
    }

    /// <summary>The 64 bits of the hash, the first pixel in the top bit (only the count of differing bits matters).</summary>
    public static ulong Hash(ImageBatch frame)
    {
        Dataset.CheckPillowChannels(frame.Channels);
        int c = frame.Channels;
        var small = DatasetPictures.PillowResized(frame, 8, 8);
        var grey = new int[64];
        for (int p = 0; p < 64; p++)
            grey[p] = c == 2 ? small[p * 2] : (small[p * c] * 19595 + small[p * c + 1] * 38470 + small[p * c + 2] * 7471 + 0x8000) >> 16; // LA keeps L
        double average = grey.Sum() / 64.0;
        ulong bits = 0;
        for (int p = 0; p < 64; p++) bits = bits << 1 | (grey[p] > average ? 1UL : 0UL);
        return bits;
    }
}

[Node("ImageGrid", Category = "image/batch", DisplayName = "Make Image Grid", Description = "Arrange multiple images into a grid layout.",
    InputIsList = true, OutputNames = ["images"])]
public sealed class ImageGrid
{
    /// <summary>
    /// One RGB picture on black: every frame resized to the cell with Pillow's LANCZOS and pasted row by row, <c>padding</c> pixels apart. Pasting
    /// keeps a picture's colour channels (an RGBA picture's alpha is dropped, an LA picture becomes grey).
    /// </summary>
    public ImageBatch Execute([Input(Tooltip = "List of images to process.")] IReadOnlyList<ImageBatch> images,
        [Input(Min = 1, Max = 20, Default = 4L, Tooltip = "Number of columns in the grid.")] IReadOnlyList<long> columns,
        [Input(Min = 32, Max = 2048, Default = 256L, Tooltip = "Width of each cell in the grid.")] IReadOnlyList<long> cellWidth,
        [Input(Min = 32, Max = 2048, Default = 256L, Tooltip = "Height of each cell in the grid.")] IReadOnlyList<long> cellHeight,
        [Input(Min = 0, Max = 50, Default = 4L, Tooltip = "Padding between images.")] IReadOnlyList<long> padding)
    {
        var frames = Dataset.Frames(images);
        int cols = (int)Dataset.One(columns, "columns"), cw = (int)Dataset.One(cellWidth, "cell_width"), ch = (int)Dataset.One(cellHeight, "cell_height");
        int pad = (int)Dataset.One(padding, "padding");
        if (frames.Count == 0) throw new ArgumentException("Cannot create grid from empty image list");
        int rows = (frames.Count + cols - 1) / cols;
        int gw = cols * cw + (cols - 1) * pad, gh = rows * ch + (rows - 1) * pad;
        var grid = new byte[gw * gh * 3];
        for (int i = 0; i < frames.Count; i++)
        {
            var frame = frames[i];
            Dataset.CheckPillowChannels(frame.Channels);
            int c = frame.Channels;
            var cell = DatasetPictures.PillowResized(frame, cw, ch);
            int x0 = i % cols * (cw + pad), y0 = i / cols * (ch + pad);
            for (int y = 0; y < ch; y++)
                for (int x = 0; x < cw; x++)
                {
                    int s = (y * cw + x) * c, d = ((y0 + y) * gw + x0 + x) * 3;
                    grid[d] = cell[s];
                    grid[d + 1] = c == 2 ? cell[s] : cell[s + 1];
                    grid[d + 2] = c == 2 ? cell[s] : cell[s + 2];
                }
        }
        return new ImageBatch(1, gh, gw, 3, PilLanczos.FromBytes(grid));
    }
}

[Node("MergeImageLists", Category = "image/batch", DisplayName = "Merge Image Lists (DEPRECATED)", Description = "Concatenate multiple image lists into one.",
    InputIsList = true, OutputIsList = [true], OutputNames = ["images"])]
public sealed class MergeImageLists
{
    /// <summary>The list it is given, with every batch split into frames.</summary>
    public IReadOnlyList<ImageBatch> Execute([Input(Tooltip = "List of images to process.")] IReadOnlyList<ImageBatch> images) => Dataset.Frames(images);
}

// ---- text steps --------------------------------------------------------------------------------------------------------------------------------------

[Node("TextToLowercase", Category = "text", DisplayName = "Convert Text to Lowercase (DEPRECATED)", Description = "Convert text to lowercase.",
    OutputNames = ["texts"], OutputTypes = ["STRING"])]
public sealed class TextToLowercase
{
    /// <summary>Python's <c>str.lower</c> (full Unicode mappings), as a list of one text (see the notes at the top of this file).</summary>
    public IReadOnlyList<object?> Execute([Input(Tooltip = "Text to process.")] string texts) => [PyText.Lower(texts)];
}

[Node("TextToUppercase", Category = "text", DisplayName = "Convert Text to Uppercase (DEPRECATED)", Description = "Convert text to uppercase.",
    OutputNames = ["texts"], OutputTypes = ["STRING"])]
public sealed class TextToUppercase
{
    /// <summary>Python's <c>str.upper</c> ("ß" becomes "SS"), as a list of one text.</summary>
    public IReadOnlyList<object?> Execute([Input(Tooltip = "Text to process.")] string texts) => [PyText.Upper(texts)];
}

[Node("TruncateText", Category = "text", DisplayName = "Truncate Text", Description = "Truncate text to a maximum length.", OutputNames = ["texts"], OutputTypes = ["STRING"])]
public sealed class TruncateText
{
    /// <summary>
    /// <c>texts[:max_length]</c>, in code points, as a list of one text. Python slices a list the same way, so a text node's output (a list of one
    /// text) comes through as that list, inside another list.
    /// </summary>
    public IReadOnlyList<object?> Execute([Input(Type = "STRING", Tooltip = "Text to process.")] object texts,
        [Input(Min = 1, Max = 10000, Tooltip = "Maximum text length.")] long maxLength = 77) => texts switch
    {
        string s => [PyText.Slice(s, null, maxLength)],
        IReadOnlyList<object?> list => [PyText.SliceItems(list, null, maxLength, null).ToArray()],
        _ => throw new ArgumentException($"'{Dataset.PyTypeName(texts)}' object is not subscriptable", nameof(texts)),
    };
}

[Node("AddTextPrefix", Category = "text", DisplayName = "Add Text Prefix (DEPRECATED)", Description = "Add a prefix to all texts.", OutputNames = ["texts"], OutputTypes = ["STRING"])]
public sealed class AddTextPrefix
{
    /// <summary><c>prefix + texts</c>, as a list of one text; a prefix that comes as a list of one text is that text.</summary>
    public IReadOnlyList<object?> Execute([Input(Tooltip = "Text to process.")] string texts,
        [Input(Type = "STRING", Default = "", Tooltip = "Prefix to add.")] object prefix) =>
        Dataset.Unwrap(prefix) is string p ? [p + texts]
            : throw new ArgumentException($"can only concatenate {Dataset.PyTypeName(Dataset.Unwrap(prefix))} (not \"str\") to {Dataset.PyTypeName(Dataset.Unwrap(prefix))}", nameof(prefix));
}

[Node("AddTextSuffix", Category = "text", DisplayName = "Add Text Suffix (DEPRECATED)", Description = "Add a suffix to all texts.", OutputNames = ["texts"], OutputTypes = ["STRING"])]
public sealed class AddTextSuffix
{
    /// <summary><c>texts + suffix</c>, as a list of one text; a suffix that comes as a list of one text is that text.</summary>
    public IReadOnlyList<object?> Execute([Input(Tooltip = "Text to process.")] string texts,
        [Input(Type = "STRING", Default = "", Tooltip = "Suffix to add.")] object suffix) =>
        Dataset.Unwrap(suffix) is string s ? [texts + s]
            : throw new ArgumentException($"can only concatenate str (not \"{Dataset.PyTypeName(Dataset.Unwrap(suffix))}\") to str", nameof(suffix));
}

[Node("ReplaceText", Category = "text", DisplayName = "Replace Text (DEPRECATED)", Description = "Replace text in all texts.", OutputNames = ["texts"], OutputTypes = ["STRING"])]
public sealed class ReplaceText
{
    /// <summary>Python's <c>str.replace</c> (an empty find puts the replacement around every character), as a list of one text.</summary>
    public IReadOnlyList<object?> Execute([Input(Tooltip = "Text to process.")] string texts,
        [Input(Type = "STRING", Default = "", Tooltip = "Text to find.")] object find,
        [Input(Type = "STRING", Default = "", Tooltip = "Text to replace with.")] object replace)
    {
        if (Dataset.Unwrap(find) is not string f) throw new ArgumentException($"replace() argument 1 must be str, not {Dataset.PyTypeName(Dataset.Unwrap(find))}", nameof(find));
        if (Dataset.Unwrap(replace) is not string r) throw new ArgumentException($"replace() argument 2 must be str, not {Dataset.PyTypeName(Dataset.Unwrap(replace))}", nameof(replace));
        return [PyText.Replace(texts, f, r)];
    }
}

[Node("StripWhitespace", Category = "text", DisplayName = "Strip Whitespace (DEPRECATED)", Description = "Strip leading and trailing whitespace from all texts.",
    OutputNames = ["texts"], OutputTypes = ["STRING"])]
public sealed class StripWhitespace
{
    /// <summary>Python's <c>str.strip()</c> (Python's whitespace, not .NET's), as a list of one text.</summary>
    public IReadOnlyList<object?> Execute([Input(Tooltip = "Text to process.")] string texts) => [PyText.Strip(texts)];
}

[Node("MergeTextLists", Category = "text", DisplayName = "Merge Text Lists (DEPRECATED)", Description = "Concatenate multiple text lists into one.",
    InputIsList = true, OutputNames = ["texts"], OutputTypes = ["STRING"])]
public sealed class MergeTextLists
{
    /// <summary>The whole list it is given, as one value: a list holding that list (ComfyUI wraps it once more, the output not being a list output).</summary>
    public IReadOnlyList<object?> Execute([Input(Type = "STRING", Default = "", Tooltip = "List of texts to process.")] IReadOnlyList<object?> texts) => [texts.ToArray()];
}

// ---- folders -------------------------------------------------------------------------------------------------------------------------------------------

/// <summary>Reading and writing dataset folders as ComfyUI's dataset nodes do.</summary>
internal static class DatasetFiles
{
    static readonly string[] PictureEndings = [".png", ".jpg", ".jpeg", ".webp"];

    /// <summary><c>any(f.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"])</c>.</summary>
    public static bool IsPicture(string name)
    {
        var lower = PyText.Lower(name);
        return PictureEndings.Any(e => lower.EndsWith(e, StringComparison.Ordinal));
    }

    /// <summary>The input folder a node names, checked to stay inside the input folder (ComfyUI's secure_subfolder_path).</summary>
    public static string InputFolder(string folder) => StorePaths.Normalize(folder, "folder name");

    /// <summary>
    /// load_and_process_images: each file through Pillow (<c>Image.open(...).convert("RGB")</c>, no EXIF rotation), one list item each. WebP files
    /// are listed as ComfyUI lists them but SpawnGen cannot decode WebP yet, so a folder with one fails on it.
    /// </summary>
    public static async Task<IReadOnlyList<ImageBatch>> LoadAsync(IReadOnlyList<string> paths)
    {
        if (paths.Count == 0) throw new ArgumentException("No valid images found in input");
        var images = new List<ImageBatch>(paths.Count);
        foreach (var path in paths) images.Add(ImageDecoder.DecodePillowRgb(await InputFiles.Store.ReadFileAsync(path)));
        return images;
    }

    /// <summary>A caption file read as Python's <c>open(path, "r", encoding="utf-8").read().strip()</c>: strict UTF-8 (a BOM stays), newlines made "\n".</summary>
    public static string Caption(byte[] bytes)
    {
        var text = new UTF8Encoding(false, true).GetString(bytes);
        return PyText.Strip(text.Replace("\r\n", "\n").Replace('\r', '\n'));
    }

    /// <summary>Python's <c>os.path.splitext(path)[1]</c>: from the last dot of the last part, unless that part is only dots before it.</summary>
    public static string Extension(string path)
    {
        int sep = path.LastIndexOf('/'), dot = path.LastIndexOf('.');
        if (dot <= sep) return "";
        for (int i = sep + 1; i < dot; i++) if (path[i] != '.') return path[dot..];
        return "";
    }

    /// <summary>
    /// save_images_to_folder: every list item (one picture each; Pillow cannot take a batch of more) as a PNG with no text chunks, named
    /// <c>prefix_00000.png</c> by its place in the list (overwrite), or <c>prefix_CCCCC_00000.png</c> with the counter of ComfyUI's
    /// get_save_image_path worked out again before every picture (increment), so the counter rises with each one. Returns the names (relative to
    /// <paramref name="folder"/>) in order.
    /// </summary>
    public static async Task<List<string>> SaveImagesAsync(IResultStore store, IReadOnlyList<ImageBatch> images, string folder, string prefix, bool overwrite)
    {
        store.CreateFolder("output", folder);
        var saved = new List<string>(images.Count);
        for (int idx = 0; idx < images.Count; idx++)
        {
            var image = images[idx];
            if (image.Batch != 1)
                throw new ArgumentException($"Cannot handle this data type: list item {idx} is a batch of {image.Batch} pictures; Save Image (to Folder) saves one picture per list item.");
            Dataset.CheckPillowChannels(image.Channels);
            string filename;
            if (overwrite) filename = $"{prefix}_{idx:D5}.png";
            else
            {
                var (counter, resolved) = Counter(store, folder, prefix);
                filename = $"{resolved}_{counter.ToString("D5", System.Globalization.CultureInfo.InvariantCulture)}_{idx:D5}.png";
            }
            var (dir, name) = StorePaths.Split(Within(folder, filename));
            await store.WriteAsync("output", dir, name, PngWriter.Encode(image));
            saved.Add(filename);
        }
        return saved;
    }

    /// <summary>
    /// Captions next to the pictures: <c>name.replace(".png", ".txt")</c> holding the text, UTF-8, "\n" written as the platform's line end (Python's
    /// text mode). Pairs up to the shorter list. A caption that is not text fails after its file has been created empty, as Python's open + write does.
    /// </summary>
    public static async Task SaveCaptionsAsync(IResultStore store, string folder, IReadOnlyList<string> saved, IReadOnlyList<object?> texts)
    {
        for (int i = 0; i < Math.Min(saved.Count, texts.Count); i++)
        {
            var (dir, name) = StorePaths.Split(Within(folder, PyText.Replace(saved[i], ".png", ".txt")));
            if (texts[i] is not string caption)
            {
                await store.WriteAsync("output", dir, name, []);
                throw new ArgumentException($"write() argument must be str, not {Dataset.PyTypeName(texts[i])}");
            }
            await store.WriteAsync("output", dir, name, new UTF8Encoding(false, true).GetBytes(caption.Replace("\n", Environment.NewLine)));
        }
    }

    /// <summary>
    /// A path under the dataset folder, which it must not leave. ComfyUI checks that only in increment mode (for the prefix's folder), and in overwrite
    /// mode would even follow an absolute prefix; SpawnGen never writes outside the folder a node was given.
    /// </summary>
    static string Within(string folder, string relative)
    {
        var rooted = relative.StartsWith('/') || OperatingSystem.IsWindows() && (relative.StartsWith('\\') || relative.Contains(':'));
        var path = rooted ? null : StorePaths.Normalize(StorePaths.Join(folder, relative), "file name");
        if (path is null || folder.Length > 0 && path != folder && !path.StartsWith(folder + "/", StringComparison.Ordinal))
            throw new ArgumentException("Saving image outside the output folder is not allowed.");
        return path;
    }

    /// <summary>
    /// get_save_image_path(prefix, folder): %year% and the like filled in (width and height 0), then 1 + the highest number among the names in the
    /// prefix's folder that start with the prefix's last part and "_" (compared as os.path.normcase does) and have a number up to the next "_" or
    /// ".", or 1. Returns the counter and the filled-in prefix.
    /// </summary>
    static (BigInteger Counter, string Prefix) Counter(IResultStore store, string folder, string prefix)
    {
        if (prefix.Contains('%')) prefix = SavePrefix.ComputeVars(prefix, 0, 0, DateTime.Now);
        var norm = NormPath(prefix);
        int slash = norm.LastIndexOf('/');
        string sub = slash < 0 ? "" : slash == 0 ? "/" : norm[..slash], filename = norm[(slash + 1)..];
        var outputFolder = Within(folder, sub);
        var names = store.List("output", outputFolder);
        if (names is null) { store.CreateFolder("output", outputFolder); return (1, prefix); }
        int prefixLength = PyText.Length(BaseName(prefix));
        BigInteger? best = null;
        foreach (var name in names)
        {
            var head = PyText.Slice(name, null, prefixLength + 1);
            if (head.Length == 0 || head[^1] != '_' || NormCase(head[..^1]) != NormCase(filename)) continue;
            var rest = PyText.Slice(name, prefixLength + 1, null);
            var digits = PyNumber.TryParseInt(rest.Split('.')[0].Split('_')[0]) ?? BigInteger.Zero;
            if (best is null || digits > best) best = digits;
        }
        return (best is { } max ? max + 1 : 1, prefix);
    }

    static string BaseName(string path)
    {
        int cut = Math.Max(path.LastIndexOf('/'), OperatingSystem.IsWindows() ? path.LastIndexOf('\\') : -1);
        return path[(cut + 1)..];
    }

    static string NormCase(string s) => OperatingSystem.IsWindows() ? PyText.Lower(s).Replace('/', '\\') : s;

    /// <summary>Python's <c>os.path.normpath</c> with "/" between parts: "." parts dropped, "a/.." folded, leading ".." kept, "" becomes ".".</summary>
    static string NormPath(string path)
    {
        var p = OperatingSystem.IsWindows() ? path.Replace('\\', '/') : path;
        bool rooted = p.StartsWith('/');
        var parts = new List<string>();
        foreach (var part in p.Split('/'))
        {
            if (part is "" or ".") continue;
            if (part == ".." && parts.Count > 0 && parts[^1] != "..") parts.RemoveAt(parts.Count - 1);
            else if (part != ".." || !rooted) parts.Add(part);
        }
        var joined = (rooted ? "/" : "") + string.Join('/', parts);
        return joined.Length == 0 ? "." : joined;
    }
}

[Node("LoadImageDataSetFromFolder", Category = "image", DisplayName = "Load Image (from Folder)",
    Description = "Load a dataset of images from a specified folder and return a list of images. Supported formats: PNG, JPG, JPEG, WEBP.",
    OutputIsList = [true], OutputNames = ["images"])]
public sealed class LoadImageDataSetFromFolder
{
    /// <summary>Every folder in the input folder, at any depth.</summary>
    public static string[] Folders() => [.. InputFiles.Store.Folders()];

    /// <summary>The pictures directly in the folder (.png, .jpg, .jpeg, .webp in any case), in the order the file system lists them; one list item each.</summary>
    public async Task<IReadOnlyList<ImageBatch>> ExecuteAsync([Input(OptionsProvider = nameof(Folders), Tooltip = "The folder to load images from.")] string folder)
    {
        var dir = DatasetFiles.InputFolder(folder);
        return await DatasetFiles.LoadAsync([.. InputFiles.Store.Entries(dir).Where(e => DatasetFiles.IsPicture(e.Name)).Select(e => StorePaths.Join(dir, e.Name))]);
    }
}

[Node("LoadImageTextDataSetFromFolder", Category = "image", DisplayName = "Load Image-Text (from Folder)",
    Description = "Load a dataset of pairs of images and text captions from a specified folder and return them as a list. Supported formats: PNG, JPG, JPEG, WEBP.",
    OutputIsList = [true, true], OutputNames = ["images", "texts"])]
public sealed class LoadImageTextDataSetFromFolder
{
    public static string[] Folders() => [.. InputFiles.Store.Folders()];

    /// <summary>
    /// The pictures in the folder in file system order, and from each folder inside it its pictures, all of them repeated N times when the folder's
    /// name starts with digits and "_" (kohya-ss "N_name", N = 0 leaves them out). Each picture's caption is the file named like it with the
    /// extension replaced by ".txt" (Python's str.replace, so every occurrence of the extension in the path), stripped, or "" when there is none.
    /// </summary>
    public async Task<(IReadOnlyList<ImageBatch> Images, IReadOnlyList<string> Texts)> ExecuteAsync(
        [Input(OptionsProvider = nameof(Folders), Tooltip = "The folder to load images and text captions from.")] string folder)
    {
        var store = InputFiles.Store;
        var dir = DatasetFiles.InputFolder(folder);
        var files = new List<string>();
        foreach (var (name, isFolder) in store.Entries(dir))
        {
            var path = StorePaths.Join(dir, name);
            if (DatasetFiles.IsPicture(name)) files.Add(path);
            else if (isFolder)
            {
                long repeat = 1;
                var head = name.Split('_')[0];
                if (PyExpression.StrPredicate(head, "isdigit"))
                {
                    var n = PyNumber.TryParseInt(head) ?? throw new ArgumentException($"invalid literal for int() with base 10: {PyText.ReprString(head)}");
                    repeat = n > int.MaxValue ? throw new OverflowException("cannot fit 'int' into an index-sized integer") : (long)n;
                }
                var inner = store.Entries(path).Where(e => DatasetFiles.IsPicture(e.Name)).Select(e => StorePaths.Join(path, e.Name)).ToList();
                for (long r = 0; r < repeat; r++) files.AddRange(inner);
            }
        }

        var captions = new List<string>(files.Count);
        foreach (var file in files)
        {
            var captionPath = PyText.Replace(file, DatasetFiles.Extension(file), ".txt");
            captions.Add(store.Exists(captionPath) ? DatasetFiles.Caption(await store.ReadFileAsync(captionPath)) : "");
        }
        return (await DatasetFiles.LoadAsync(files), captions);
    }
}

[Node("SaveImageDataSetToFolder", Category = "image", DisplayName = "Save Image (to Folder) (DEPRECATED)",
    Description = "Save a dataset of images to a specified folder. Supported formats: PNG.", IsOutput = true, InputIsList = true)]
public sealed class SaveImageDataSetToFolder
{
    /// <summary>Writes every list item as a PNG into the folder inside the output folder (see <see cref="DatasetFiles.SaveImagesAsync"/>).</summary>
    public async Task ExecuteAsync(IServiceProvider services, [Input(Tooltip = "List of images to save.")] IReadOnlyList<ImageBatch> images,
        [Input(Default = "dataset", Tooltip = "Name of the folder to save images to (inside output directory).")] IReadOnlyList<string> folderName,
        [Input(Default = "image", Tooltip = "Prefix for saved image filenames.")] IReadOnlyList<string> filenamePrefix,
        [Input(Options = ["overwrite", "increment"], Tooltip = "Whether to overwrite existing files or increment filenames to avoid overwriting.")] IReadOnlyList<string> mode)
    {
        var folder = StorePaths.Normalize(folderName[0], "folder name");
        await DatasetFiles.SaveImagesAsync(Results.For(services), images, folder, filenamePrefix[0], mode[0] == "overwrite");
    }
}

[Node("SaveImageTextDataSetToFolder", Category = "image", DisplayName = "Save Image-Text (to Folder)",
    Description = "Save a dataset of pairs of images and text captions to a specified folder. Images are saved as PNG files and captions are saved as TXT files with the same filename_prefix.",
    IsOutput = true, InputIsList = true)]
public sealed class SaveImageTextDataSetToFolder
{
    /// <summary>The pictures as Save Image (to Folder) writes them, then a caption file next to each picture that has a text.</summary>
    public async Task ExecuteAsync(IServiceProvider services, [Input(Tooltip = "List of images to save.")] IReadOnlyList<ImageBatch> images,
        [Input(Default = "dataset", Tooltip = "Name of the folder to save images to (inside output directory).")] IReadOnlyList<string> folderName,
        [Input(Default = "image", Tooltip = "Prefix for saved image filenames.")] IReadOnlyList<string> filenamePrefix,
        [Input(Options = ["overwrite", "increment"], Tooltip = "Whether to overwrite existing files or increment filenames to avoid overwriting.")] IReadOnlyList<string> mode,
        [Input(Type = "STRING", ForceInput = true, Optional = true, Tooltip = "List of text captions to save.")] IReadOnlyList<object?>? texts = null)
    {
        var store = Results.For(services);
        var folder = StorePaths.Normalize(folderName[0], "folder name");
        var saved = await DatasetFiles.SaveImagesAsync(store, images, folder, filenamePrefix[0], mode[0] == "overwrite");
        if (texts is { Count: > 0 }) await DatasetFiles.SaveCaptionsAsync(store, folder, saved, texts);
    }
}
