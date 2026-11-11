using System.Numerics;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries, SpawnGen's tests and fixture (Fixtures/bbox/bbox.json.gz) and black-box
// probes of ComfyUI (2026-11-11). Python's semantics (dict access, slicing, arithmetic, the errors they raise) come from the Python documentation through
// the Py* helpers; Pillow's rectangle from PilDraw; PyTorch's area resize from CommonUpscale.

/// <summary>
/// Crop Image (ImageCropV2), as measured: x = min(region.get("x", 0), W - 1), y = min(region.get("y", 0), H - 1), then Python's slice
/// image[:, y:y + height, x:x + width] (width and height default to 512; negative and out-of-range ends follow slicing, so the result can be empty).
/// The region must be a dict and the indices ints (bools count); it shows the crop as a preview.
/// </summary>
[Node("ImageCropV2", Category = "image/transform", DisplayName = "Crop Image", HasIntermediateOutput = true, OutputNames = ["IMAGE"],
    Description = "Crop an image to the specified dimensions.")]
public sealed class ImageCropV2
{
    public async Task<(ImageBatch Image, UiOutput Ui)> ExecuteAsync(ImageBatch image,
        [Input(Type = "BOUNDING_BOX", Json = """{"x":0,"y":0,"width":512,"height":512}""")] object? cropRegion,
        NodeContext context, IServiceProvider services)
    {
        var cropped = Crop(image, cropRegion);
        return (cropped, new UiOutput("images", await Results.PreviewAsync(cropped, context, services)));
    }

    public static ImageBatch Crop(ImageBatch image, object? cropRegion)
    {
        var x = PyValues.Get(cropRegion, "x", 0L);
        var y = PyValues.Get(cropRegion, "y", 0L);
        var width = PyValues.Get(cropRegion, "width", 512L);
        var height = PyValues.Get(cropRegion, "height", 512L);
        x = PyValues.Min(x, (long)image.Width - 1);
        y = PyValues.Min(y, (long)image.Height - 1);
        var yEnd = PyValues.Add(y, height);
        var xEnd = PyValues.Add(x, width);
        var (y0, y1) = PySlice.Bounds(y, yEnd, image.Height);
        var (x0, x1) = PySlice.Bounds(x, xEnd, image.Width);
        int h = y1 - y0, w = x1 - x0, c = image.Channels;
        var result = new ImageBatch(image.Batch, h, w, c);
        for (int b = 0; b < image.Batch; b++)
            for (int row = 0; row < h; row++)
                Array.Copy(image.Data, (((long)b * image.Height + y0 + row) * image.Width + x0) * c, result.Data, (((long)b * h + row) * w) * c, (long)w * c);
        return result;
    }
}

/// <summary>Python's slice bounds for a sequence of length n.</summary>
internal static class PySlice
{
    /// <summary>
    /// <c>seq[start:stop]</c>: both must be ints (TypeError otherwise); negative values count from the end; both are clamped to 0..n; an empty slice
    /// when stop &lt;= start.
    /// </summary>
    public static (int Start, int Stop) Bounds(object? start, object? stop, int n)
    {
        if (!PyValues.IsInt(start) || !PyValues.IsInt(stop))
            throw new PyException("TypeError", "slice indices must be integers or None or have an __index__ method");
        int Norm(BigInteger v)
        {
            if (v < 0) v += n;
            return (int)BigInteger.Clamp(v, 0, n);
        }
        int s = Norm(PyValues.Int(start)), e = Norm(PyValues.Int(stop));
        return (s, Math.Max(s, e));
    }
}

/// <summary>
/// Draw BBoxes, as measured: every picture goes to 8 bits as uint8(float32(v) * 255) (C truncation, wrapping like a numpy cast) and back as b / 255; each
/// box {x, y, width, height} is Pillow's rectangle outline [x, y, x + width, y + height] (coordinates truncated), 3 pixels wide, in the colour of its
/// label's COCO class (the class index modulo an 8-colour cycle; any other label, or none, is red), drawn in ascending score order (stable, no score
/// = 1). A dict, or a list of dicts, applies to every
/// picture; a list of lists gives picture f the list min(f, last). Without an image the canvas is black and as large as the first picture's boxes
/// reach (int of the largest x + width and y + height), 640 x 640 when there are none. Pillow refuses the colour on 1 and 2 channel pictures (only when a
/// box is drawn). ComfyUI also writes the label (and score) as text inside known-label boxes; SpawnGen draws no text yet.
/// </summary>
[Node("DrawBBoxes", Category = "image/detection", DisplayName = "Draw BBoxes", OutputNames = ["out_image"])]
public sealed class DrawBBoxes
{
    static readonly string[] Coco =
    [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
        "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush",
    ];

    static readonly byte[][] Colors =
    [
        [255, 0, 0], [0, 200, 0], [0, 0, 255], [255, 165, 0], [128, 0, 128], [0, 255, 255], [255, 20, 147], [100, 149, 237],
    ];

    public ImageBatch Execute(
        [Input(Type = "BOUNDING_BOX", Json = """{"x":0,"y":0,"width":512,"height":512}""", ForceInput = true)] object bboxes,
        [Input(Optional = true)] ImageBatch? image = null)
    {
        var picture = image ?? Canvas(bboxes);
        int h = picture.Height, w = picture.Width, c = picture.Channels, pixels = h * w * c;
        var result = new ImageBatch(picture.Batch, h, w, c);
        var px = new byte[pixels];
        for (int f = 0; f < picture.Batch; f++)
        {
            for (int i = 0; i < pixels; i++) px[i] = ToByte(picture.Data[(long)f * pixels + i]);
            // Drawn from the lowest score up (stable; a box without a score counts as 1), so confident boxes end on top.
            foreach (var box in BoxesOf(bboxes, f).OrderBy(b => PyValues.Float(PyValues.Get(b, "score", 1.0))))
            {
                var x = PyValues.Item(box, "x");
                var y = PyValues.Item(box, "y");
                var bw = PyValues.Item(box, "width");
                var bh = PyValues.Item(box, "height");
                var label = PyValues.Get(box, "label");
                double x1 = PyValues.Float(PyValues.Add(x, bw)), y1 = PyValues.Float(PyValues.Add(y, bh));
                PilDraw.Rectangle(px, w, h, c, PyValues.Float(x), PyValues.Float(y), x1, y1, Ink(label, c), fill: false, width: 3);
            }
            for (int i = 0; i < pixels; i++) result.Data[(long)f * pixels + i] = px[i] / 255f;
        }
        return result;
    }

    static byte ToByte(float v)
    {
        float scaled = v * 255f;
        return float.IsNaN(scaled) ? (byte)0 : unchecked((byte)(long)scaled);
    }

    static byte[] Ink(object? label, int channels)
    {
        int index = label is string s ? Array.IndexOf(Coco, s) : -1;
        var rgb = Colors[index < 0 ? 0 : index % Colors.Length];
        return channels switch
        {
            1 => throw new PyException("TypeError", "color must be int or single-element tuple"),
            2 => throw new PyException("TypeError", "color must be int, or tuple of one or two elements"),
            3 => rgb,
            4 => [rgb[0], rgb[1], rgb[2], 255],
            _ => throw new ArgumentException($"Draw BBoxes cannot draw on a {channels}-channel picture."),
        };
    }

    /// <summary>The boxes drawn on picture <paramref name="frame"/>.</summary>
    static IEnumerable<object?> BoxesOf(object bboxes, int frame)
    {
        if (bboxes is DictValue) return [bboxes];
        if (bboxes is ArrayValue list)
        {
            if (list.Count == 0) return [];
            if (list[0] is ArrayValue) return PyValues.Iterate(list[Math.Min(frame, list.Count - 1)]);
            return list;
        }
        return PyValues.Iterate(bboxes);
    }

    /// <summary>The black canvas drawn on when there is no image: as large as the first picture's boxes reach, or 640 x 640.</summary>
    static ImageBatch Canvas(object bboxes)
    {
        var boxes = BoxesOf(bboxes, 0).ToList();
        if (boxes.Count == 0) return new ImageBatch(1, 640, 640, 3);
        object? right = null, bottom = null;
        foreach (var box in boxes)
        {
            var r = PyValues.Add(PyValues.Item(box, "x"), PyValues.Item(box, "width"));
            var b = PyValues.Add(PyValues.Item(box, "y"), PyValues.Item(box, "height"));
            if (right is null || PyValues.Less(right, r)) right = r;
            if (bottom is null || PyValues.Less(bottom, b)) bottom = b;
        }
        return new ImageBatch(1, (int)PyValues.ToInt(bottom), (int)PyValues.ToInt(right), 3);
    }
}

/// <summary>
/// Crop By Bounding Boxes, as measured: picture f takes the boxes of list min(f, last) (a dict: the same box for all; an empty list: the picture gives
/// nothing), their union min x, min y, max(x + width), max(y + height), grown by padding on each side, truncated to ints and clamped to the picture.
/// An empty region falls back to an s x s square, s = int(0.3 min(H, W)), at ((W - s) // 2, int(0.1 H)); if that is empty too the crop is black. The crop is resized with PyTorch's area mode:
/// stretched to the output size, or (keep_aspect pad) scaled by min(width / cw, height / ch) to Python round()ed sizes and centred on zeros. With no crop
/// at all the input comes back unchanged.
/// </summary>
[Node("CropByBBoxes", Category = "image/transform", DisplayName = "Crop By Bounding Boxes", OutputNames = ["IMAGE"],
    Description = "Crop and resize regions from the input image batch based on provided bounding boxes.")]
public sealed class CropByBBoxes
{
    public ImageBatch Execute(ImageBatch image,
        [Input(Type = "BOUNDING_BOX", Json = """{"x":0,"y":0,"width":512,"height":512}""", ForceInput = true)] object bboxes,
        [Input(Min = 64, Max = 4096, Step = 8, Tooltip = "Width each crop is resized to.")] long outputWidth = 512,
        [Input(Min = 64, Max = 4096, Step = 8, Tooltip = "Height each crop is resized to.")] long outputHeight = 512,
        [Input(Min = 0, Max = 1024, Step = 1, Tooltip = "Extra padding in pixels added on each side of the bbox before cropping.")] long padding = 0,
        [Input(Options = ["stretch", "pad"], Tooltip = "Whether to stretch the crop to fit the output size, or pad with black pixels to preserve aspect ratio.")]
        string keepAspect = "stretch")
    {
        int ow = (int)outputWidth, oh = (int)outputHeight, c = image.Channels, h = image.Height, w = image.Width;
        var crops = new List<float[]>();
        for (int f = 0; f < image.Batch; f++)
        {
            var boxes = BoxesOf(bboxes, f).ToList();
            if (boxes.Count == 0) continue;
            object? left = null, top = null, right = null, bottom = null;
            foreach (var box in boxes) { var v = PyValues.Item(box, "x"); if (left is null || PyValues.Less(v, left)) left = v; }
            foreach (var box in boxes) { var v = PyValues.Item(box, "y"); if (top is null || PyValues.Less(v, top)) top = v; }
            foreach (var box in boxes)
            {
                var v = PyValues.Add(PyValues.Item(box, "x"), PyValues.Item(box, "width"));
                if (right is null || PyValues.Less(right, v)) right = v;
            }
            foreach (var box in boxes)
            {
                var v = PyValues.Add(PyValues.Item(box, "y"), PyValues.Item(box, "height"));
                if (bottom is null || PyValues.Less(bottom, v)) bottom = v;
            }
            int x0 = (int)Math.Max(0, Truncate(Subtract(left, padding))), y0 = (int)Math.Max(0, Truncate(Subtract(top, padding)));
            int x1 = (int)Math.Min(w, Truncate(PyValues.Add(right, padding))), y1 = (int)Math.Min(h, Truncate(PyValues.Add(bottom, padding)));
            if (x1 <= x0 || y1 <= y0)
            {
                int s = (int)(0.3 * Math.Min(h, w));
                x0 = (w - s) / 2;
                y0 = (int)(0.1 * h);
                x1 = x0 + s;
                y1 = y0 + s;
            }
            int cw = x1 - x0, ch = y1 - y0;
            if (cw <= 0 || ch <= 0) { crops.Add(new float[ow * oh * c]); continue; } // a picture too small even for that: black
            var region = new float[cw * ch * c];
            for (int row = 0; row < ch; row++)
                Array.Copy(image.Data, (((long)f * h + y0 + row) * w + x0) * c, region, (long)row * cw * c, (long)cw * c);
            crops.Add(keepAspect == "pad" ? Pad(region, cw, ch, c, ow, oh) : CommonUpscale.Resize(region, cw, ch, c, ow, oh, "area"));
        }
        if (crops.Count == 0) return image;
        var result = new ImageBatch(crops.Count, oh, ow, c);
        for (int i = 0; i < crops.Count; i++) Array.Copy(crops[i], 0, result.Data, (long)i * crops[i].Length, crops[i].Length);
        return result;
    }

    static float[] Pad(float[] region, int cw, int ch, int c, int ow, int oh)
    {
        double scale = Math.Min((double)ow / cw, (double)oh / ch);
        int nw = (int)PyValues.Round(cw * scale), nh = (int)PyValues.Round(ch * scale);
        var resized = CommonUpscale.Resize(region, cw, ch, c, nw, nh, "area");
        var canvas = new float[ow * oh * c];
        int ox = (ow - nw) / 2, oy = (oh - nh) / 2;
        for (int row = 0; row < nh; row++) Array.Copy(resized, (long)row * nw * c, canvas, ((long)(oy + row) * ow + ox) * c, (long)nw * c);
        return canvas;
    }

    /// <summary>The boxes of picture <paramref name="frame"/>: a dict for every picture, else the list min(frame, last) (iterated as Python would).</summary>
    static IEnumerable<object?> BoxesOf(object bboxes, int frame) => bboxes switch
    {
        DictValue => [bboxes],
        ArrayValue { Count: 0 } => [],
        ArrayValue list => PyValues.Iterate(list[Math.Min(frame, list.Count - 1)]),
        _ => PyValues.Iterate(bboxes),
    };

    /// <summary>Python's <c>a - b</c> for an int b.</summary>
    static object Subtract(object? a, long b)
    {
        if (PyValues.IsInt(a)) return PyValues.Normal(PyValues.Int(a) - b);
        if (PyValues.IsNumber(a)) return PyValues.Float(a) - b;
        throw new PyException("TypeError", $"unsupported operand type(s) for -: '{PyValues.TypeName(a)}' and 'int'");
    }

    /// <summary>Python's <c>int(v)</c> of a number (toward zero).</summary>
    static long Truncate(object? v) => (long)PyValues.ToInt(v);
}
