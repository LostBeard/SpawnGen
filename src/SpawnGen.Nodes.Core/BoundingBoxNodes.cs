using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// ComfyUI's BOUNDING_BOX values (comfy_extras/nodes_bounding_boxes.py): a box is a dict <c>{"x", "y", "width", "height"}</c> in pixels, optionally with
/// <c>metadata</c> (type, text, desc, palette); a frame is a list of boxes and a BOUNDING_BOX output is a list of frames. Ideogram "elements" carry a
/// <c>bbox</c> of <c>[ymin, xmin, ymax, xmax]</c> on a 0..1000 grid. The values are JSON-like (<see cref="DictValue"/>, <see cref="ArrayValue"/>) and are
/// read with Python's semantics, errors included.
/// </summary>
public static class BoundingBoxes
{
    /// <summary>A canvas region as fractions of the canvas (x, y, w, h may be negative), with the box's metadata.</summary>
    public sealed record Region(double X, double Y, double W, double H, object? Type, object? Text, object? Desc, object? Palette);

    static PyException ValueError(string message) => new("ValueError", message);

    public static Region PixelsToFractions(DictValue box, long width, long height, object? type, object? text, object? desc, object? palette)
    {
        long w = width != 0 ? width : 1, h = height != 0 ? height : 1;
        return new Region(PyValues.TrueDivide(PyValues.Get(box, "x", 0L), w), PyValues.TrueDivide(PyValues.Get(box, "y", 0L), h),
            PyValues.TrueDivide(PyValues.Get(box, "width", 0L), w), PyValues.TrueDivide(PyValues.Get(box, "height", 0L), h), type, text, desc, palette);
    }

    /// <summary>fractions_to_pixels: a region back to a pixel box (a negative size flips the box), every value rounded half to even.</summary>
    public static DictValue FractionsToPixels(Region r, long width, long height)
    {
        double x = r.X, y = r.Y, w = r.W, h = r.H;
        if (w < 0) (x, w) = (x + w, -w);
        if (h < 0) (y, h) = (y + h, -h);
        return new DictValue
        {
            ["x"] = PyValues.Round(x * width), ["y"] = PyValues.Round(y * height),
            ["width"] = PyValues.Round(w * width), ["height"] = PyValues.Round(h * height),
        };
    }

    /// <summary>fractions_to_bbox_frame: one frame of pixel boxes, or no frame at all when there are no regions.</summary>
    public static ArrayValue FractionsToBBoxFrame(IReadOnlyList<Region> regions, long width, long height)
    {
        var pixels = new ArrayValue(regions.Select(r => (object?)FractionsToPixels(r, width, height)));
        return pixels.Count > 0 ? new ArrayValue { pixels } : [];
    }

    /// <summary>boxes_to_regions: each dict of the list as a region of the canvas with its metadata (type defaults to "obj"); anything but a list gives none.</summary>
    public static List<Region> BoxesToRegions(object? boxes, long width, long height)
    {
        var regions = new List<Region>();
        if (boxes is not ArrayValue list) return regions;
        foreach (var item in list)
        {
            if (item is not DictValue box) continue;
            var meta = PyValues.Get(box, "metadata") as DictValue ?? [];
            regions.Add(PixelsToFractions(box, width, height, PyValues.Get(meta, "type", "obj"), PyValues.Get(meta, "text", ""), PyValues.Get(meta, "desc", ""),
                PyValues.Get(meta, "palette", new ArrayValue())));
        }
        return regions;
    }

    /// <summary>normalize_incoming_boxes: the first frame of the value (a dict is a frame of one), each box reduced to x, y, width, height and its metadata dict.</summary>
    public static ArrayValue NormalizeIncomingBoxes(object? bboxes)
    {
        IEnumerable<object?> frame = bboxes switch
        {
            DictValue d => [d],
            ArrayValue { Count: > 0 } a when a[0] is DictValue => a,
            ArrayValue { Count: > 0 } a => a[0] as ArrayValue ?? [],
            _ => [],
        };
        var boxes = new ArrayValue();
        foreach (var item in frame)
        {
            if (item is not DictValue box) continue;
            var norm = new DictValue
            {
                ["x"] = PyValues.Get(box, "x", 0L), ["y"] = PyValues.Get(box, "y", 0L),
                ["width"] = PyValues.Get(box, "width", 0L), ["height"] = PyValues.Get(box, "height", 0L),
            };
            if (PyValues.Get(box, "metadata") is DictValue meta) norm["metadata"] = meta;
            boxes.Add(norm);
        }
        return boxes;
    }

    static bool LooksLikeElement(DictValue box) => PyValues.Get(box, "bbox") is ArrayValue { Count: 4 };

    static bool LooksLikeBBox(DictValue box) => box.ContainsKey("x") && box.ContainsKey("y") && box.ContainsKey("width") && box.ContainsKey("height");

    /// <summary>elements_to_boxes: Ideogram elements (<c>bbox</c> = [ymin, xmin, ymax, xmax] on 0..1000) as pixel boxes on a width x height canvas.</summary>
    public static ArrayValue ElementsToBoxes(IEnumerable<object?> elements, long width, long height)
    {
        var boxes = new ArrayValue();
        foreach (var item in elements)
        {
            if (item is not DictValue element) continue;
            if (PyValues.Get(element, "bbox") is not ArrayValue { Count: 4 } bbox)
                throw ValueError("bboxes element is missing a valid 'bbox' [ymin, xmin, ymax, xmax]");
            double ymin, xmin, ymax, xmax;
            try
            {
                ymin = PyValues.ToFloat(bbox[0]) / 1000.0;
                xmin = PyValues.ToFloat(bbox[1]) / 1000.0;
                ymax = PyValues.ToFloat(bbox[2]) / 1000.0;
                xmax = PyValues.ToFloat(bbox[3]) / 1000.0;
            }
            catch (PyException e) when (e.PyType is "TypeError" or "ValueError")
            {
                throw ValueError("bboxes element 'bbox' must contain four numbers");
            }
            bool text = PyValues.Equal(PyValues.Get(element, "type"), "text");
            var palette = PyValues.Get(element, "color_palette", new ArrayValue());
            boxes.Add(new DictValue
            {
                ["x"] = PyValues.Round(PyMin(xmin, xmax) * width),
                ["y"] = PyValues.Round(PyMin(ymin, ymax) * height),
                ["width"] = PyValues.Round(Math.Abs(xmax - xmin) * width),
                ["height"] = PyValues.Round(Math.Abs(ymax - ymin) * height),
                ["metadata"] = new DictValue
                {
                    ["type"] = text ? "text" : "obj",
                    ["text"] = text ? PyValues.Get(element, "text", "") : "",
                    ["desc"] = PyValues.Get(element, "desc", ""),
                    ["palette"] = PyValues.Truthy(palette) ? palette : new ArrayValue(),
                },
            });
        }
        return boxes;
    }

    /// <summary>Python's <c>min(a, b)</c> of floats: <paramref name="a"/> unless <paramref name="b"/> is smaller (so a NaN first stays).</summary>
    static double PyMin(double a, double b) => b < a ? b : a;

    /// <summary>
    /// boxes_from_input: the pixel boxes a <c>bboxes</c> input gives on a width x height canvas: bounding boxes (a box, a frame, a list of frames: the first),
    /// Ideogram elements (one or a list), or a JSON text of any of them; nothing for null or empty text. Other values are a ValueError.
    /// </summary>
    public static ArrayValue FromInput(object? data, long width, long height)
    {
        if (data is null) return [];
        if (data is string s)
        {
            var text = PyText.Strip(s);
            if (text.Length == 0) return [];
            try { data = PyJson.Loads(text); }
            catch (PyException e) { throw ValueError($"bboxes string input is not valid JSON: {e.PyMessage}"); }
        }
        if (data is DictValue d)
        {
            if (LooksLikeElement(d)) return ElementsToBoxes([d], width, height);
            if (LooksLikeBBox(d)) return NormalizeIncomingBoxes(d);
            throw ValueError("bboxes dict must be a bounding box (x, y, width, height) or an element (with a 'bbox')");
        }
        if (data is not ArrayValue list)
            throw ValueError($"bboxes input must be bounding boxes, elements, or a JSON string, got {PyValues.TypeName(data)}");
        if (list.Count == 0) return [];
        var first = list[0];
        if (first is ArrayValue) return NormalizeIncomingBoxes(list);
        if (first is DictValue f)
        {
            if (LooksLikeElement(f)) return ElementsToBoxes(list, width, height);
            if (LooksLikeBBox(f)) return NormalizeIncomingBoxes(list);
            throw ValueError("bboxes items must be bounding boxes (x, y, width, height) or elements (with a 'bbox')");
        }
        throw ValueError($"bboxes list must contain bounding boxes or elements, got {PyValues.TypeName(first)}");
    }

    /// <summary>color_util.normalize_palette: the non-empty strings of a list (a dict's values), upper-cased.</summary>
    public static ArrayValue NormalizePalette(object? colors)
    {
        if (colors is DictValue d) colors = new ArrayValue(d.Values);
        return new ArrayValue(PyValues.Iterate(colors).Where(c => c is string { Length: > 0 }).Select(c => (object?)PyText.Upper((string)c!)));
    }

    /// <summary>_norm_bbox: a region as [ymin, xmin, ymax, xmax] on the 0..1000 grid (rounded, clamped, ordered).</summary>
    static ArrayValue NormBBox(Region r)
    {
        static long Grid(double v) => Math.Max(0, Math.Min(1000, PyValues.Round(v * 1000)));
        long ymin = Grid(r.Y), xmin = Grid(r.X), ymax = Grid(r.Y + r.H), xmax = Grid(r.X + r.W);
        if (ymin > ymax) (ymin, ymax) = (ymax, ymin);
        if (xmin > xmax) (xmin, xmax) = (xmax, xmin);
        return [ymin, xmin, ymax, xmax];
    }

    /// <summary>build_elements: Ideogram prompt elements for the regions (type, bbox, text for text regions, desc, up to five palette colours).</summary>
    public static ArrayValue BuildElements(IReadOnlyList<Region> regions)
    {
        var elements = new ArrayValue();
        foreach (var r in regions)
        {
            bool text = PyValues.Equal(r.Type, "text");
            var element = new DictValue { ["type"] = text ? "text" : "obj", ["bbox"] = NormBBox(r) };
            if (text) element["text"] = r.Text;
            element["desc"] = r.Desc;
            var palette = NormalizePalette(r.Palette);
            if (palette.Count > 0) element["color_palette"] = new ArrayValue(palette.Take(5));
            elements.Add(element);
        }
        return elements;
    }

    /// <summary>color_util.hex_to_rgb: "#rrggbb" (any number of leading '#') as bytes, white when it is not six hex digits as int(..., 16) reads them.</summary>
    public static (byte R, byte G, byte B) HexToRgb(object? value)
    {
        if (value is not string s) throw new PyException("AttributeError", $"'{PyValues.TypeName(value)}' object has no attribute 'lstrip'");
        var h = s.TrimStart('#');
        if (PyText.Length(h) != 6) return (255, 255, 255);
        var parts = new long[3];
        for (int i = 0; i < 3; i++)
        {
            var v = PyNumber.TryParseInt(PyText.Slice(h, i * 2, i * 2 + 2), 16);
            if (v is null) return (255, 255, 255);
            parts[i] = (long)v.Value;
        }
        // Pillow's ink takes the ints clipped to 0..255 (a "+f"-style pair is still a small number).
        return ((byte)Math.Clamp(parts[0], 0, 255), (byte)Math.Clamp(parts[1], 0, 255), (byte)Math.Clamp(parts[2], 0, 255));
    }

    /// <summary>
    /// render_preview without its text: the canvas (the background dimmed to a quarter, long edge at most 1024, else grey 32) with each region's outline
    /// (2 px, its first palette colour or grey 140) and up to five palette swatches along its top edge. The number tags and the descriptions ComfyUI writes
    /// into the boxes need a font engine, which SpawnGen does not have yet; their pixels are left as the boxes and swatches leave them.
    /// </summary>
    public static ImageBatch RenderPreview(IReadOnlyList<Region> regions, long width, long height, ImageBatch? background)
    {
        const int LongEdgeMax = 1024;
        var bg = BackgroundBytes(background, out int iw, out int ih);
        int rw, rh;
        byte[] px;
        if (bg is not null)
        {
            long longEdge = Math.Max(iw, ih);
            if (longEdge == 0) longEdge = 1;
            double scale = Math.Min(1.0, (double)LongEdgeMax / longEdge);
            rw = (int)Math.Max(1, PyValues.Round(iw * scale));
            rh = (int)Math.Max(1, PyValues.Round(ih * scale));
            px = rw == iw && rh == ih ? bg : PilLanczos.Resize(bg, iw, ih, 3, rw, rh);
            // ImageEnhance.Brightness(0.25): Image.blend with black, (UINT8)(0.25f * v).
            for (int i = 0; i < px.Length; i++) px[i] = (byte)(0.25f * px[i]);
        }
        else
        {
            long longEdge = Math.Max(width, height);
            if (longEdge == 0) longEdge = 1;
            double scale = Math.Min(1.0, (double)LongEdgeMax / longEdge);
            rw = (int)Math.Max(1, PyValues.Round(width * scale));
            rh = (int)Math.Max(1, PyValues.Round(height * scale));
            px = new byte[rw * rh * 3];
            Array.Fill(px, (byte)32); // round(0.25 * 128)
        }

        long fs = Math.Max(10, PyValues.Round(rh / 64.0));
        foreach (var region in regions)
        {
            var palette = PyValues.Truthy(region.Palette) ? PyValues.Iterate(region.Palette).Where(PyValues.Truthy).ToList() : [];
            var (r, g, b) = palette.Count > 0 ? HexToRgb(palette[0]) : ((byte)140, (byte)140, (byte)140);
            long x1 = Math.Max(0, Math.Min(rw, PyValues.Round(region.X * rw)));
            long y1 = Math.Max(0, Math.Min(rh, PyValues.Round(region.Y * rh)));
            long x2 = Math.Max(0, Math.Min(rw, PyValues.Round((region.X + region.W) * rw)));
            long y2 = Math.Max(0, Math.Min(rh, PyValues.Round((region.Y + region.H) * rh)));
            if (x2 < x1) (x1, x2) = (x2, x1);
            if (y2 < y1) (y1, y2) = (y2, y1);

            PilDraw.Rectangle(px, rw, rh, 3, x1, y1, x2, y2, [r, g, b], fill: false, width: 2);

            var swatches = palette.Take(5).ToList();
            if (swatches.Count > 0 && x2 - x1 > 2)
            {
                long sh = Math.Max(5, fs / 2);
                double seg = (double)(x2 - x1) / swatches.Count;
                for (int p = 0; p < swatches.Count; p++)
                {
                    long sx = x1 + PyValues.Round(p * seg);
                    var (sr, sg, sb) = HexToRgb(swatches[p]);
                    PilDraw.Rectangle(px, rw, rh, 3, sx, y1, x1 + PyValues.Round((p + 1) * seg), y1 + sh, [sr, sg, sb], fill: true);
                }
            }

            // ComfyUI then writes the region's number tag and its description (wrapped) into the box; the text is not drawn here (no font engine yet).
            // The description must still be text where ComfyUI would write it, as there.
            bool isText = PyValues.Equal(region.Type, "text");
            object? body = PyValues.Truthy(region.Desc) ? region.Desc : "";
            if (isText && PyValues.Truthy(region.Text))
            {
                if (PyValues.Truthy(body) && body is not string)
                    throw new PyException("TypeError", $"can only concatenate str (not \"{PyValues.TypeName(body)}\") to str");
                body = "text";
            }
            if (PyValues.Truthy(body) && x2 - x1 > 8 && body is not string)
                throw new PyException("AttributeError", $"'{PyValues.TypeName(body)}' object has no attribute 'split'");
        }

        var image = new ImageBatch(1, rh, rw, 3);
        for (int i = 0; i < px.Length; i++) image.Data[i] = px[i] / 255f;
        return image;
    }

    /// <summary>
    /// _bg_from_image: the first picture as 8-bit RGB ((v * 255).clip(0, 255) truncated, then Pillow's convert("RGB")); null where ComfyUI gets none
    /// (no picture, an empty batch, or a channel count Image.fromarray cannot take: 1 or more than 4).
    /// </summary>
    static byte[]? BackgroundBytes(ImageBatch? image, out int w, out int h)
    {
        w = h = 0;
        if (image is null || image.Batch == 0 || image.Channels is not (2 or 3 or 4)) return null;
        w = image.Width;
        h = image.Height;
        int c = image.Channels;
        var rgb = new byte[w * h * 3];
        for (int p = 0; p < w * h; p++)
            for (int k = 0; k < 3; k++)
                rgb[p * 3 + k] = (byte)Math.Clamp(image.Data[p * c + (c == 2 ? 0 : k)] * 255f, 0f, 255f); // LA -> RGB repeats L; RGBA -> RGB drops A
        return rgb;
    }
}

[Node(Category = "utilities", DisplayName = "Create Bounding Boxes", IsOutput = true, OutputNames = ["preview", "bboxes", "elements"],
    OutputTypes = [null, "BOUNDING_BOX", null],
    Description = "Draw bounding boxes in a canvas. Outputs Ideogram prompt elements, pixel-space bounding boxes, and a preview image.")]
public sealed class CreateBoundingBoxes
{
    /// <summary>
    /// ComfyUI's Create Bounding Boxes: the boxes come from the canvas (<paramref name="editorState"/>) unless a new upstream value arrived (it differs from
    /// <paramref name="lastIncoming"/>, the one that last initialized the canvas). Outputs the preview, the boxes as one frame of pixel boxes and the Ideogram
    /// elements; the editor gets the canvas size, the upstream boxes and the background.
    /// </summary>
    public async Task<(ImageBatch Preview, object Bboxes, ArrayValue Elements, UiOutput Ui)> ExecuteAsync(
        [Input(Tooltip = "Optional image used as background in the canvas and preview.")] ImageBatch? background = null,
        [Input(Type = "BOUNDING_BOX,ARRAY,STRING", Tooltip = "Bounding boxes, elements, or a JSON string to initialize the canvas. A new upstream value initializes the canvas; edits made on the canvas take priority and are kept until the upstream value changes again.")]
        object? bboxes = null,
        [Input(Min = 64, Max = 16384, Step = 16, Tooltip = "Width of the canvas and the pixel grid for the bounding boxes.")] long width = 1024,
        [Input(Min = 64, Max = 16384, Step = 16, Tooltip = "Height of the canvas and the pixel grid for the bounding boxes.")] long height = 1024,
        [Input(Type = "BOUNDING_BOXES", Json = "[]", JsonSocket = true, Tooltip = "Draw bounding boxes and set each box type, text, description, color palette. Start with background element first and foreground last.")]
        object? editorState = null,
        [Input(Type = "BOUNDING_BOXES", Json = "[]", Optional = true, Tooltip = "Internal state managed by the canvas: the upstream bboxes value that last initialized it. Leave empty to re-initialize the canvas from the bboxes input on the next run.")]
        object? lastIncoming = null,
        NodeContext? context = null, IServiceProvider? services = null)
    {
        var incoming = BoundingBoxes.FromInput(bboxes, width, height);
        var applied = lastIncoming as ArrayValue ?? [];
        bool upstreamChanged = incoming.Count > 0 && !PyValues.Equal(incoming, applied);
        object? source = upstreamChanged ? incoming : PyValues.Truthy(editorState) ? editorState : new ArrayValue();
        var regions = BoundingBoxes.BoxesToRegions(source, width, height);
        var preview = BoundingBoxes.RenderPreview(regions, width, height, background);

        var ui = new UiOutput("dims", new[] { width, height });
        if (incoming.Count > 0) ui["input_bboxes"] = incoming;
        if (background is { Batch: > 0 } && context is not null && services is not null)
            ui["background_images"] = await Results.PreviewAsync(background, context, services, count: 1);
        return (preview, BoundingBoxes.FractionsToBBoxFrame(regions, width, height), BoundingBoxes.BuildElements(regions), ui);
    }
}
