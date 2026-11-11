using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries, SpawnGen's tests and fixture (Fixtures/bbox/bbox.json.gz) and black-box
// probes of ComfyUI (2026-11-11: coordinate rounding, clamping, the palette strip, the background's dimming and scaling). Python's semantics come from the
// Py* helpers; Pillow's rectangle from PilDraw and its LANCZOS resize from PilLanczos.

/// <summary>
/// The bounding boxes input Create Bounding Boxes and Layers From Bounding Boxes share (BOUNDING_BOX, ARRAY or STRING): a pixel box
/// {x, y, width, height}, an element {bbox: [y0, x0, y1, x1] on a 0..1000 grid}, a list of them (a list of lists: the first list), or JSON text of any
/// of these. Whitespace-only text and None mean no boxes.
/// </summary>
internal static class BBoxInput
{
    public abstract record Item(DictValue Raw);

    /// <summary>A pixel box; its values as given.</summary>
    public sealed record Box(DictValue Raw) : Item(Raw);

    /// <summary>An element, its corners ordered (Y0 &lt;= Y1, X0 &lt;= X1), on the 0..1000 grid.</summary>
    public sealed record Element(DictValue Raw, double Y0, double X0, double Y1, double X1) : Item(Raw);

    static PyException Error(string message) => new("ValueError", message);

    public static List<Item> Parse(object? bboxes)
    {
        if (bboxes is null) return [];
        if (bboxes is string text)
        {
            text = PyText.Strip(text);
            if (text.Length == 0) return [];
            try { bboxes = PyJson.Loads(text); }
            catch (PyException e) when (e.PyType == "JSONDecodeError") { throw Error("bboxes string input is not valid JSON: " + e.PyMessage); }
            if (bboxes is null) return [];
        }
        switch (bboxes)
        {
            case DictValue d:
                return [Classify(d) ?? throw Error("bboxes dict must be a bounding box (x, y, width, height) or an element (with a 'bbox')")];
            case ArrayValue list:
                if (list.Count > 0 && list[0] is ArrayValue first) list = first;
                var items = new List<Item>();
                foreach (var v in list)
                {
                    if (v is not DictValue item) throw Error($"bboxes list must contain bounding boxes or elements, got {PyValues.TypeName(v)}");
                    items.Add(Classify(item) ?? throw Error("bboxes items must be bounding boxes (x, y, width, height) or elements (with a 'bbox')"));
                }
                return items;
            default:
                throw Error($"bboxes input must be bounding boxes, elements, or a JSON string, got {PyValues.TypeName(bboxes)}");
        }
    }

    static Item? Classify(DictValue d)
    {
        if (d.TryGetValue("bbox", out var b) && b is ArrayValue { Count: 4 } corners)
        {
            var v = new double[4];
            for (int i = 0; i < 4; i++)
            {
                try { v[i] = PyValues.ToFloat(corners[i]); }
                catch (PyException) { throw Error("bboxes element 'bbox' must contain four numbers"); }
            }
            return new Element(d, Math.Min(v[0], v[2]), Math.Min(v[1], v[3]), Math.Max(v[0], v[2]), Math.Max(v[1], v[3]));
        }
        return d.ContainsKey("x") && d.ContainsKey("y") && d.ContainsKey("width") && d.ContainsKey("height") ? new Box(d) : null;
    }

    /// <summary>An element's corners in pixels of a <paramref name="w"/> x <paramref name="h"/> canvas: Python's round of x0 * w / 1000 and of the scaled sizes.</summary>
    public static (long X, long Y, long W, long H) Pixels(Element e, double w, double h) =>
        (PyValues.Round(e.X0 * w / 1000), PyValues.Round(e.Y0 * h / 1000), PyValues.Round((e.X1 - e.X0) * w / 1000), PyValues.Round((e.Y1 - e.Y0) * h / 1000));
}

/// <summary>
/// Create Bounding Boxes, as measured. The canvas boxes come from editor_state (a list of {x, y, width, height, metadata: {type, text, desc, palette}};
/// anything else is none), unless the bboxes input brings boxes that differ from last_incoming: those (elements turned into pixel boxes with their type,
/// text, desc and colour palette as metadata) replace them, and the ui reports them as input_bboxes. A negative size flips its box.
/// <para>
/// Outputs: the pixel boxes as ints, the boxes as Ideogram elements (type, bbox [y0, x0, y1, x1] rounded on a 0..1000 grid and clamped, text for text
/// boxes, desc, up to 5 upper-cased palette colours), and a preview: the background (frame 0 as 8 bits, truncated; one channel is ignored, two are grey)
/// or a (32, 32, 32) canvas, at most 1024 on its longer side, darkened to a quarter, with each box outlined 2 pixels wide in its first palette colour
/// (grey without one) and a 6-row strip of up to 5 palette swatches along its top. Box corners are round((x / W + w / W) * preview width), clamped to
/// 0..preview size. A colour is "#rrggbb" (any number of leading '#'), anything else white. ComfyUI also draws a number tag and the description as
/// text; SpawnGen draws no text yet.
/// </para>
/// </summary>
[Node("CreateBoundingBoxes", Category = "image/detection", DisplayName = "Create Bounding Boxes", IsOutput = true,
    Description = "Draw bounding boxes in a canvas. Outputs Ideogram prompt elements, pixel-space bounding boxes, and a preview image.",
    OutputNames = ["preview", "bboxes", "elements"], OutputTypes = [null, "BOUNDING_BOX", "ARRAY"])]
public sealed class CreateBoundingBoxes
{
    static readonly byte[] Grey = [140, 140, 140], White = [255, 255, 255];

    public async Task<(ImageBatch Preview, object Bboxes, ArrayValue Elements, UiOutput Ui)> ExecuteAsync(
        [Input(Optional = true, Tooltip = "Optional image used as background in the canvas and preview.")] ImageBatch? background = null,
        [Input(Type = "BOUNDING_BOX,ARRAY,STRING", Optional = true,
            Tooltip = "Bounding boxes, elements, or a JSON string to initialize the canvas. A new upstream value initializes the canvas; edits made on the canvas take priority and are kept until the upstream value changes again.")]
        object? bboxes = null,
        [Input(Min = 64, Max = 16384, Step = 16, Tooltip = "Width of the canvas and the pixel grid for the bounding boxes.")] long width = 1024,
        [Input(Min = 64, Max = 16384, Step = 16, Tooltip = "Height of the canvas and the pixel grid for the bounding boxes.")] long height = 1024,
        [Input(Type = "BOUNDING_BOXES", Json = "[]", JsonSocket = true,
            Tooltip = "Draw bounding boxes and set each box type, text, description, color palette. Start with background element first and foreground last.")]
        object? editorState = null,
        [Input(Type = "BOUNDING_BOXES", Json = "[]", Optional = true,
            Tooltip = "Internal state managed by the canvas: the upstream bboxes value that last initialized it. Leave empty to re-initialize the canvas from the bboxes input on the next run.")]
        object? lastIncoming = null,
        NodeContext? context = null)
    {
        double W = width, H = height;
        var incoming = new ArrayValue();
        foreach (var item in BBoxInput.Parse(bboxes)) incoming.Add(EditorBox(item, W, H));
        var state = editorState as ArrayValue ?? [];
        if (incoming.Count > 0 && !PyValues.Equal(incoming, lastIncoming)) state = incoming;

        var boxes = new List<(double X, double Y, double W, double H, object? Meta)>();
        foreach (var b in state)
        {
            double x = PyValues.ToFloat(PyValues.Get(b, "x", 0L)), y = PyValues.ToFloat(PyValues.Get(b, "y", 0L));
            double w = PyValues.ToFloat(PyValues.Get(b, "width", 0L)), h = PyValues.ToFloat(PyValues.Get(b, "height", 0L));
            if (w < 0) { x += w; w = -w; }
            if (h < 0) { y += h; h = -h; }
            boxes.Add((x, y, w, h, PyValues.Get(b, "metadata")));
        }

        var (px, pw, ph) = Canvas(background, width, height);
        foreach (var (x, y, w, h, meta) in boxes)
        {
            long x0 = Math.Clamp(PyValues.Round(x / W * pw), 0, pw), x1 = Math.Clamp(PyValues.Round((x / W + w / W) * pw), 0, pw);
            long y0 = Math.Clamp(PyValues.Round(y / H * ph), 0, ph), y1 = Math.Clamp(PyValues.Round((y / H + h / H) * ph), 0, ph);
            var palette = meta is DictValue m && PyValues.Truthy(PyValues.Get(m, "palette")) ? PyValues.Iterate(m["palette"]).Take(5).ToList() : [];
            var colours = palette.Select(Colour).ToList();
            PilDraw.Rectangle(px, pw, ph, 3, x0, y0, x1, y1, colours.Count > 0 ? colours[0] : Grey, fill: false, width: 2);
            for (int i = 0; i < colours.Count; i++)
            {
                long s0 = x0 + PyValues.Round((double)i * (x1 - x0) / colours.Count), s1 = x0 + PyValues.Round((double)(i + 1) * (x1 - x0) / colours.Count);
                PilDraw.Rectangle(px, pw, ph, 3, s0, y0, s1, y0 + 5, colours[i], fill: true);
            }
            // The label ComfyUI writes: '"text" desc' for a text box, else desc; split into words when the box is wider than 8 pixels.
            object? desc = meta is DictValue md ? PyValues.Get(md, "desc", "") : "";
            object? label = desc;
            if (meta is DictValue mt && PyValues.Equal(PyValues.Get(mt, "type", "obj"), "text") && PyValues.Get(mt, "text", "") is { } t && PyValues.Truthy(t))
                label = PyValues.Add(PyValues.Add(PyValues.Add("\"", t), "\" "), desc);
            if (x1 - x0 > 8 && PyValues.Truthy(label) && label is not string)
                throw new PyException("AttributeError", $"'{PyValues.TypeName(label)}' object has no attribute 'split'");
        }

        var pixels = new ArrayValue();
        var elements = new ArrayValue();
        foreach (var (x, y, w, h, meta) in boxes)
        {
            pixels.Add(new DictValue
            {
                ["x"] = PyValues.Normal(PyValues.ToInt(x)), ["y"] = PyValues.Normal(PyValues.ToInt(y)),
                ["width"] = PyValues.Normal(PyValues.ToInt(w)), ["height"] = PyValues.Normal(PyValues.ToInt(h)),
            });
            elements.Add(Element(x, y, w, h, meta as DictValue, W, H));
        }

        var preview = new ImageBatch(1, ph, pw, 3, PilLanczos.FromBytes(px));
        var ui = new UiOutput("dims", new ArrayValue { width, height });
        if (incoming.Count > 0) ui["input_bboxes"] = incoming;
        if (background is not null && context is not null) ui["background_images"] = await Results.PreviewAsync(background, context, context.Services, 1);
        return (preview, pixels.Count > 0 ? new ArrayValue { pixels } : new ArrayValue(), elements, ui);
    }

    /// <summary>An upstream box as the canvas holds it: a pixel box as given, an element in pixels with its metadata.</summary>
    static DictValue EditorBox(BBoxInput.Item item, double w, double h)
    {
        if (item is BBoxInput.Box box)
        {
            var d = new DictValue { ["x"] = box.Raw["x"], ["y"] = box.Raw["y"], ["width"] = box.Raw["width"], ["height"] = box.Raw["height"] };
            if (box.Raw.TryGetValue("metadata", out var meta)) d["metadata"] = meta;
            return d;
        }
        var e = (BBoxInput.Element)item;
        var (x, y, bw, bh) = BBoxInput.Pixels(e, w, h);
        return new DictValue
        {
            ["x"] = x, ["y"] = y, ["width"] = bw, ["height"] = bh,
            ["metadata"] = new DictValue
            {
                ["type"] = PyValues.Get(e.Raw, "type", "obj"), ["text"] = PyValues.Get(e.Raw, "text", ""), ["desc"] = PyValues.Get(e.Raw, "desc", ""),
                ["palette"] = PyValues.Get(e.Raw, "color_palette", new ArrayValue()),
            },
        };
    }

    static DictValue Element(double x, double y, double w, double h, DictValue? meta, double W, double H)
    {
        long Grid(double v) => Math.Clamp(PyValues.Round(v * 1000), 0, 1000);
        var type = meta is null ? "obj" : PyValues.Get(meta, "type", "obj");
        var e = new DictValue
        {
            ["type"] = type,
            ["bbox"] = new ArrayValue { Grid(y / H), Grid(x / W), Grid(y / H + h / H), Grid(x / W + w / W) },
        };
        if (PyValues.Equal(type, "text")) e["text"] = meta is null ? "" : PyValues.Get(meta, "text", "");
        e["desc"] = meta is null ? "" : PyValues.Get(meta, "desc", "");
        var palette = new ArrayValue();
        switch (meta is null ? null : PyValues.Get(meta, "palette"))
        {
            case ArrayValue list:
                palette.AddRange(list.Take(5).Select(Upper));
                break;
            case DictValue dict:
                palette.AddRange(dict.Values.Where(PyValues.Truthy).Take(5).Select(Upper));
                break;
        }
        if (palette.Count > 0) e["color_palette"] = palette;
        return e;
    }

    static object? Upper(object? v) => v is string s ? s.ToUpperInvariant()
        : throw new PyException("AttributeError", $"'{PyValues.TypeName(v)}' object has no attribute 'upper'");

    /// <summary>A palette colour: "#rrggbb" after removing leading '#'s (each pair read as Python's int(pair, 16) reads it), white otherwise.</summary>
    static byte[] Colour(object? entry)
    {
        if (entry is not string s) throw new PyException("AttributeError", $"'{PyValues.TypeName(entry)}' object has no attribute 'lstrip'");
        s = s.TrimStart('#');
        if (s.Length != 6) return White;
        var rgb = new byte[3];
        for (int i = 0; i < 3; i++)
        {
            if (HexPair(s.Substring(2 * i, 2)) is not { } v) return White;
            rgb[i] = (byte)Math.Clamp(v, 0, 255);
        }
        return rgb;
    }

    /// <summary>Python's int(text, 16) for two characters: two hex digits, a sign and a digit, or a digit with surrounding whitespace.</summary>
    static int? HexPair(string text)
    {
        static int Digit(char c) => Uri.IsHexDigit(c) ? Convert.ToInt32(c.ToString(), 16) : -1;
        var t = text.Trim();
        if (t.Length == 2 && Digit(t[0]) >= 0 && Digit(t[1]) >= 0) return Digit(t[0]) * 16 + Digit(t[1]);
        if (t.Length == 2 && t[0] is '+' or '-' && Digit(t[1]) >= 0) return t[0] == '-' ? -Digit(t[1]) : Digit(t[1]);
        if (t.Length == 1 && Digit(t[0]) >= 0) return Digit(t[0]);
        return null;
    }

    /// <summary>The preview canvas: the background's frame 0 (or a (32, 32, 32) canvas of the box grid's size) scaled to at most 1024, darkened to a quarter.</summary>
    static (byte[] Px, int W, int H) Canvas(ImageBatch? background, long width, long height)
    {
        bool usable = background is { Channels: >= 2, Height: > 0, Width: > 0 };
        long bw = usable ? background!.Width : width, bh = usable ? background!.Height : height;
        double scale = Math.Max(bw, bh) > 1024 ? 1024.0 / Math.Max(bw, bh) : 1.0;
        int pw = (int)Math.Max(1, PyValues.Round(bw * scale)), ph = (int)Math.Max(1, PyValues.Round(bh * scale));
        if (!usable)
        {
            var flat = new byte[pw * ph * 3];
            Array.Fill(flat, (byte)32);
            return (flat, pw, ph);
        }
        int c = background!.Channels, n = (int)(bw * bh);
        var rgb = new byte[n * 3];
        var frame = PilLanczos.ToBytes(background.Data.AsSpan(0, n * c).ToArray());
        for (int i = 0; i < n; i++)
            for (int k = 0; k < 3; k++) rgb[i * 3 + k] = frame[i * c + (c == 2 ? 0 : k)];
        if (pw != bw || ph != bh) rgb = PilLanczos.Resize(rgb, (int)bw, (int)bh, 3, pw, ph);
        for (int i = 0; i < rgb.Length; i++) rgb[i] = (byte)(rgb[i] >> 2);
        return (rgb, pw, ph);
    }
}
