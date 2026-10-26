using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

[Node("ImageCropV2", Category = "image/transform", DisplayName = "Crop Image", OutputNames = ["IMAGE"], Description = "Crop an image to the specified dimensions.")]
public sealed class ImageCropV2
{
    /// <summary>
    /// ComfyUI's Crop Image: the region's x and y are kept inside the picture (at most width - 1, height - 1), then the picture is sliced
    /// <c>[y : y + height, x : x + width]</c> with Python's slice rules (a negative start counts from the end, the end is clipped); a missing size is 512.
    /// The crop is shown in the node.
    /// </summary>
    public async Task<(ImageBatch Image, UiOutput? Ui)> ExecuteAsync(ImageBatch image,
        [Input(Type = "BOUNDING_BOX", Json = """{"x": 0, "y": 0, "width": 512, "height": 512}""")] object cropRegion,
        NodeContext? context = null, IServiceProvider? services = null)
    {
        var crop = Crop(image, cropRegion);
        return (crop, context is not null && services is not null ? new UiOutput("images", await Results.PreviewAsync(crop, context, services)) : null);
    }

    public static ImageBatch Crop(ImageBatch image, object? region)
    {
        object? x = PyValues.Get(region, "x", 0L), y = PyValues.Get(region, "y", 0L);
        object? width = PyValues.Get(region, "width", 512L), height = PyValues.Get(region, "height", 512L);
        x = PyValues.Min(x, (long)image.Width - 1);
        y = PyValues.Min(y, (long)image.Height - 1);
        object toX = PyValues.Add(width, x), toY = PyValues.Add(height, y);
        var (y0, _, _, rows) = PyText.SliceIndices(image.Height, SliceIndex(y), SliceIndex(toY), 1);
        var (x0, _, _, cols) = PyText.SliceIndices(image.Width, SliceIndex(x), SliceIndex(toX), 1);
        int c = image.Channels;
        var result = new ImageBatch(image.Batch, (int)rows, (int)cols, c);
        for (int b = 0; b < image.Batch; b++)
            for (int r = 0; r < rows; r++)
                Array.Copy(image.Data, (((long)b * image.Height + y0 + r) * image.Width + x0) * c, result.Data, ((long)b * rows + r) * cols * c, cols * c);
        return result;
    }

    /// <summary>A slice bound: an int (clamped to long, which slicing clips anyway); anything else is Python's TypeError.</summary>
    static long SliceIndex(object? v)
    {
        if (!PyValues.IsInt(v)) throw new PyException("TypeError", "slice indices must be integers or None or have an __index__ method");
        var i = PyValues.Int(v);
        return i > long.MaxValue ? long.MaxValue : i < long.MinValue ? long.MinValue : (long)i;
    }
}

[Node("DrawBBoxes", Category = "image/detection", DisplayName = "Draw BBoxes", OutputNames = ["out_image"])]
public sealed class DrawBBoxes
{
    /// <summary>The COCO classes RT-DETR detects; a box whose label is one of them is drawn in that class's colour and labelled.</summary>
    public static readonly string[] CocoClasses =
    [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
        "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
        "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    ];

    static readonly byte[][] Colors = [[255, 0, 0], [0, 200, 0], [0, 0, 255], [255, 165, 0], [128, 0, 128], [0, 255, 255], [255, 20, 147], [100, 149, 237]];

    /// <summary>
    /// ComfyUI's Draw BBoxes: each picture with its frame's boxes drawn (3 px outlines, lowest score first; a COCO label picks the colour, else red). A box
    /// or a flat list of boxes is drawn on every picture; a list of frames goes picture by picture (missing frames draw nothing). Without a picture the
    /// boxes of the first frame are drawn on black, as large as they reach (640 x 640 when there are none). The picture goes through 8 bits as ComfyUI's
    /// PIL round trip does. The label text ComfyUI writes in a labelled box needs a font engine, which SpawnGen does not have yet; it is not drawn.
    /// </summary>
    public ImageBatch Execute(
        [Input(Type = "BOUNDING_BOX", ForceInput = true, Json = """{"x": 0, "y": 0, "width": 512, "height": 512}""")] object bboxes,
        ImageBatch? image = null)
    {
        int batch = image?.Batch ?? 1;
        List<object?> frames = bboxes switch
        {
            DictValue d => [new ArrayValue { d }],
            ArrayValue { Count: > 0 } a when a[0] is DictValue => [a],
            ArrayValue { Count: > 0 } a => [.. a],
            _ => [new ArrayValue()],
        };
        if (frames.Count == 1) frames = Enumerable.Repeat(frames[0], batch).ToList();
        frames = frames.Concat(Enumerable.Repeat<object?>(new ArrayValue(), batch)).Take(batch).ToList();

        if (image is null)
        {
            batch = frames.Count;
            long maxW = 640, maxH = 640;
            var all = frames.SelectMany(PyValues.Iterate).ToList();
            if (all.Count > 0)
            {
                maxW = all.Max(d => (long)PyValues.ToInt(PyValues.Add(PyValues.Item(d, "x"), PyValues.Item(d, "width"))));
                maxH = all.Max(d => (long)PyValues.ToInt(PyValues.Add(PyValues.Item(d, "y"), PyValues.Item(d, "height"))));
            }
            if (maxW < 0 || maxH < 0) throw new PyException("RuntimeError", $"Trying to create tensor with negative dimension: [{batch}, {maxH}, {maxW}, 3]");
            image = new ImageBatch(batch, (int)maxH, (int)maxW, 3);
        }

        int h = image.Height, w = image.Width, c = image.Channels, plane = h * w * c;
        if (c is < 1 or > 4) throw new ArgumentException($"Draw BBoxes takes pictures of 1 to 4 channels (PIL's L, LA, RGB, RGBA), got {c}.", nameof(image));
        var result = new ImageBatch(batch, h, w, c);
        var px = new byte[plane];
        for (int i = 0; i < batch; i++)
        {
            // ToPILImage: mul(255).byte(), truncated and wrapped to 8 bits.
            for (int k = 0; k < plane; k++) px[k] = unchecked((byte)(int)(image.Data[i * plane + k] * 255f));
            var detections = frames[i];
            if (PyValues.Truthy(detections)) Draw(px, w, h, c, PyValues.Iterate(detections).ToList());
            for (int k = 0; k < plane; k++) result.Data[i * plane + k] = px[k] / 255f;
        }
        return result;
    }

    static void Draw(byte[] px, int w, int h, int c, List<object?> detections)
    {
        // torch.tensor of the corners: int64 when every value is an int, else float32 (each value rounded to float32).
        var corners = detections.Select(d => new[]
        {
            PyValues.Item(d, "x"), PyValues.Item(d, "y"), PyValues.Add(PyValues.Item(d, "x"), PyValues.Item(d, "width")),
            PyValues.Add(PyValues.Item(d, "y"), PyValues.Item(d, "height")),
        }).ToList();
        bool intBoxes = corners.All(b => b.All(PyValues.IsInt));
        var labels = detections.Select(d => PyValues.Get(d, "label") is var l && CocoClasses.Any(k => PyValues.Equal(l, k)) ? (string)l! : null).ToList();
        var scoreValues = detections.Select(d => PyValues.Get(d, "score", 1.0)).ToList();
        bool intScores = scoreValues.All(PyValues.IsInt);
        var scores = scoreValues.Select(s => intScores ? PyValues.Float(s) : (double)(float)PyValues.Float(s)).ToList();
        // PIL's ink for a one- or two-band picture cannot be a colour tuple.
        if (c is 1 or 2) throw new PyException("TypeError", c == 1 ? "color must be int or single-element tuple" : "color must be int, or tuple of one or two elements");

        foreach (var k in Enumerable.Range(0, detections.Count).OrderBy(k => scores[k])) // sorted(): stable, ascending score
        {
            var box = corners[k].Select(v => intBoxes ? PyValues.Float(v) : (double)(float)PyValues.Float(v)).ToArray();
            int colorIndex = labels[k] is { } label ? Array.IndexOf(CocoClasses, label) : 0;
            var color = Colors[colorIndex % Colors.Length];
            byte[] ink = c == 4 ? [color[0], color[1], color[2], 255] : color;
            PilDraw.Rectangle(px, w, h, c, box[0], box[1], box[2], box[3], ink, fill: false, width: 3);
            // ComfyUI writes f"{label} {score:.2f}" at (x1 + 2, y1 + 2) here; text needs a font engine (not drawn).
        }
    }
}

[Node("CropByBBoxes", Category = "image/transform", DisplayName = "Crop By Bounding Boxes",
    Description = "Crop and resize regions from the input image batch based on provided bounding boxes.")]
public sealed class CropByBBoxes
{
    /// <summary>
    /// ComfyUI's Crop By Bounding Boxes: per picture, the union of its frame's boxes (the last frame serves the pictures after it), widened by
    /// <paramref name="padding"/> and kept inside the picture, resized ("area") to the output size, stretched or letterboxed on black ("pad"). A box
    /// that leaves nothing falls back to a centred square of 30 % of the shorter side, 10 % from the top. Pictures with no boxes are skipped; with no crops
    /// at all the picture comes back as it is.
    /// </summary>
    public ImageBatch Execute(ImageBatch image,
        [Input(Type = "BOUNDING_BOX", ForceInput = true, Json = """{"x": 0, "y": 0, "width": 512, "height": 512}""")] object bboxes,
        [Input(Min = 64, Max = 4096, Step = 8, Tooltip = "Width each crop is resized to.")] long outputWidth = 512,
        [Input(Min = 64, Max = 4096, Step = 8, Tooltip = "Height each crop is resized to.")] long outputHeight = 512,
        [Input(Min = 0, Max = 1024, Step = 1, Tooltip = "Extra padding in pixels added on each side of the bbox before cropping.")] long padding = 0,
        [Input(Options = ["stretch", "pad"], Tooltip = "Whether to stretch the crop to fit the output size, or pad with black pixels to preserve aspect ratio.")]
        string keepAspect = "stretch")
    {
        int imgH = image.Height, imgW = image.Width, c = image.Channels, ow = (int)outputWidth, oh = (int)outputHeight;
        ArrayValue frames;
        if (bboxes is not ArrayValue list) frames = [new ArrayValue { bboxes }];
        else if (list.Count == 0) return image;
        else frames = list;

        var crops = new List<float[]>();
        for (int f = 0; f < image.Batch; f++)
        {
            var frameBoxes = frames[Math.Min(f, frames.Count - 1)];
            if (!PyValues.Truthy(frameBoxes)) continue;
            var boxes = PyValues.Iterate(frameBoxes).ToList();
            // Python evaluates min(x1) over every box, then min(y1), max(x2), max(y2): the first failing value raises.
            long x1 = (long)boxes.Select(b => PyValues.ToInt(PyValues.Item(b, "x"))).ToList().Min();
            long y1 = (long)boxes.Select(b => PyValues.ToInt(PyValues.Item(b, "y"))).ToList().Min();
            long x2 = (long)boxes.Select(b => PyValues.ToInt(PyValues.Add(PyValues.Item(b, "x"), PyValues.Item(b, "width")))).ToList().Max();
            long y2 = (long)boxes.Select(b => PyValues.ToInt(PyValues.Add(PyValues.Item(b, "y"), PyValues.Item(b, "height")))).ToList().Max();
            if (padding > 0)
            {
                x1 = Math.Max(0, x1 - padding);
                y1 = Math.Max(0, y1 - padding);
                x2 = Math.Min(imgW, x2 + padding);
                y2 = Math.Min(imgH, y2 + padding);
            }
            (x1, x2) = (Math.Max(0, x1), Math.Min(imgW, x2));
            (y1, y2) = (Math.Max(0, y1), Math.Min(imgH, y2));

            if (x2 <= x1 || y2 <= y1)
            {
                long fallback = (long)(Math.Min(imgH, imgW) * 0.3);
                long fx1 = Math.Max(0, FloorDiv(imgW - fallback, 2));
                long fy1 = Math.Max(0, (long)(imgH * 0.1));
                long fx2 = Math.Min(imgW, fx1 + fallback), fy2 = Math.Min(imgH, fy1 + fallback);
                if (fx2 <= fx1 || fy2 <= fy1)
                {
                    crops.Add(new float[oh * ow * c]);
                    continue;
                }
                (x1, y1, x2, y2) = (fx1, fy1, fx2, fy2);
            }

            int cw = (int)(x2 - x1), ch = (int)(y2 - y1);
            var crop = new float[ch * cw * c];
            for (int r = 0; r < ch; r++)
                Array.Copy(image.Data, (((long)f * imgH + y1 + r) * imgW + x1) * c, crop, (long)r * cw * c, cw * c);

            if (keepAspect == "pad")
            {
                double sx = (double)ow / cw, sy = (double)oh / ch, scale = sy < sx ? sy : sx;
                int sw = (int)PyValues.Round(cw * scale), sh = (int)PyValues.Round(ch * scale);
                var scaled = CommonUpscale.Resize(crop, cw, ch, c, sw, sh, "area");
                int left = (int)FloorDiv(ow - sw, 2), top = (int)FloorDiv(oh - sh, 2);
                var padded = new float[oh * ow * c];
                for (int r = 0; r < sh; r++)
                    Array.Copy(scaled, (long)r * sw * c, padded, ((long)(top + r) * ow + left) * c, sw * c);
                crops.Add(padded);
            }
            else crops.Add(CommonUpscale.Resize(crop, cw, ch, c, ow, oh, "area"));
        }

        if (crops.Count == 0) return image;
        var result = new ImageBatch(crops.Count, oh, ow, c);
        for (int i = 0; i < crops.Count; i++) crops[i].CopyTo(result.Data, (long)i * oh * ow * c);
        return result;
    }

    /// <summary>Python's <c>a // b</c> for ints (rounds toward negative infinity).</summary>
    static long FloorDiv(long a, long b) => (long)Math.Floor((double)a / b);
}
