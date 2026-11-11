using System.Security.Cryptography;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries and their tooltips, SpawnGen's tests and fixture
// (Fixtures/bbox/compositor.json.gz) and black-box probes of ComfyUI (2026-11-11). Pillow's resize and rotation come from PilLanczos / PilTransform,
// PyTorch's bilinear resize from TorchInterpolate, Python's semantics from the Py* helpers. ComfyUI's layer fingerprint could not be derived from its
// behaviour; SpawnGen hashes the same inputs its own way (a documented difference, PARITY.md).

/// <summary>ComfyUI's LAYERS: a layer stack built by Add Layer / Layers From Bounding Boxes for Create Layered Image.</summary>
[Socket("LAYERS")]
public sealed class LayerDocument
{
    public int Version { get; init; } = 1;
    /// <summary>The document canvas (width, height); null derives it from the placed layers.</summary>
    public (int, int)? Canvas { get; init; }
    public List<LayerItem> Layers { get; init; } = [];
}

/// <summary>
/// One item of a LAYERS document: a picture (a batch expands to consecutive layers) and the keys ComfyUI stores when they are set: placement,
/// stacking, a transparency mask (1 = transparent), name, opacity, blend mode, visibility, flips, rotation (radians, clockwise) and display size.
/// </summary>
public sealed record LayerItem(ImageBatch Image)
{
    public string Type { get; init; } = "raster";
    public long X { get; init; }
    public long Y { get; init; }
    public long ZIndex { get; init; }
    public MaskBatch? Mask { get; init; }
    public string? Name { get; init; }
    public double? Opacity { get; init; }
    public string? BlendMode { get; init; }
    public bool? Visible { get; init; }
    public bool FlipH { get; init; }
    public bool FlipV { get; init; }
    public double? Rotation { get; init; }
    public long? W { get; init; }
    public long? H { get; init; }
}

/// <summary>
/// Add Layer, as measured: appends one item to a copy of the incoming stack (version 1, its canvas kept). x, y and z_index are always stored; the
/// mask, a non-empty name, opacity other than 1, a blend mode other than normal, a rotation other than 0 (in radians), a width / height other than 0
/// and the flips only when set.
/// </summary>
[Node("AddLayer", Category = "image/compositing", DisplayName = "Add Layer", OutputNames = ["LAYERS"])]
public sealed class AddLayer
{
    public static IEnumerable<string> BlendModes() => CompositorBlend.ModeNames;

    public LayerDocument Execute(
        [Input(Tooltip = "Layer content at its native size. A batch expands to consecutive layers.")] ImageBatch image,
        [Input(Optional = true, Tooltip = "Layer stack to append to. Leave unconnected to start a new stack.")] LayerDocument? layers = null,
        [Input(Optional = true, Tooltip = "Transparency mask for this layer. Masked areas (value 1) become transparent, multiplying with any alpha channel the image already carries.")]
        MaskBatch? mask = null,
        [Input(Optional = true, Tooltip = "Layer name shown in the compositor editor.")] string name = "",
        [Input(Optional = true, Min = -16384, Max = 16384, Tooltip = "Initial horizontal placement on the canvas.")] long x = 0,
        [Input(Optional = true, Min = -16384, Max = 16384, Tooltip = "Initial vertical placement on the canvas.")] long y = 0,
        [Input(Optional = true, Min = 0.0, Max = 1.0, Step = 0.01, Tooltip = "Initial layer opacity.")] double opacity = 1.0,
        [Input(Optional = true, OptionsProvider = nameof(BlendModes),
            Tooltip = "Initial blend mode, applied against the layers below. On the bottom layer over the default transparent background, non-normal modes produce transparency.")]
        string blendMode = "normal",
        [Input(Optional = true, Min = -360.0, Max = 360.0, Step = 1.0, Tooltip = "Initial rotation in degrees, clockwise.")] double rotation = 0.0,
        [Input(Optional = true, Min = 0, Max = 16384, Tooltip = "Initial display width. 0 keeps the image's native width.")] long width = 0,
        [Input(Optional = true, Min = 0, Max = 16384, Tooltip = "Initial display height. 0 keeps the image's native height.")] long height = 0,
        [Input(Optional = true, Min = -1000, Max = 1000, Tooltip = "Stacking override. Layers are stable-sorted by z_index; equal values keep their list order.")] long zIndex = 0,
        [Input(Optional = true, Tooltip = "Flip the layer horizontally.")] bool flipH = false,
        [Input(Optional = true, Tooltip = "Flip the layer vertically.")] bool flipV = false)
    {
        var item = new LayerItem(image)
        {
            X = x, Y = y, ZIndex = zIndex, Mask = mask,
            Name = string.IsNullOrEmpty(name) ? null : name,
            Opacity = opacity != 1.0 ? opacity : null,
            BlendMode = blendMode != "normal" ? blendMode : null,
            Rotation = rotation != 0 ? rotation * (Math.PI / 180.0) : null, // Python's math.radians
            W = width != 0 ? width : null,
            H = height != 0 ? height : null,
            FlipH = flipH, FlipV = flipV,
        };
        return new LayerDocument { Canvas = layers?.Canvas, Layers = [.. layers?.Layers ?? [], item] };
    }
}

/// <summary>
/// Layers From Bounding Boxes, as measured: frame f of the batch becomes one layer placed by box f (bounding boxes, elements or JSON text, as Create
/// Bounding Boxes reads them; a list of lists gives its first list). Pixel box numbers are truncated (bools and text count as missing), a width / height
/// above 0 scales the layer; elements need the canvas size to resolve (Python's round of the scaled corner and size) and name the layer by their desc.
/// A box's metadata names the layer (name, else desc), sets its z_index, and with crop_to_content its content_rect [x, y, w, h] (clamped to the frame)
/// crops the frame and mask and moves the layer by the rect's offset, its size then the cropped one. Frames without a box sit at the origin. z_index
/// defaults to one above the stack's highest (0 for a new stack) plus the frame index. A mask batch of 1 applies to every frame, otherwise frame by frame.
/// </summary>
[Node("LayersFromBoundingBoxes", Category = "image/compositing", DisplayName = "Layers From Bounding Boxes", OutputNames = ["LAYERS"],
    Description = "Turn an image batch plus its bounding boxes into a layer stack, one layer per frame, each placed by its own box. Use this when a node emits layers as a batch - a batch carries a single placement for every frame, so the individual positions are otherwise lost.")]
public sealed class LayersFromBoundingBoxes
{
    public LayerDocument Execute(
        [Input(Tooltip = "Image batch; each frame becomes one layer.")] ImageBatch image,
        [Input(Type = "BOUNDING_BOX,ARRAY,STRING",
            Tooltip = "Placement boxes, index-aligned with the image batch. Accepts bounding boxes (x, y, width, height), normalized elements (with a 'bbox' - these need canvas_width/canvas_height to resolve to pixels), or a JSON string of either. Frames without a matching box are placed at the origin. A box's width/height scales the layer to fit it. metadata.name (or desc) and metadata.z_index are used when present, and metadata.content_rect (frame-relative) crops the frame to its real content.")]
        object bboxes,
        [Input(Optional = true, Tooltip = "Per-frame transparency, index-aligned with the image batch (1 = transparent, LoadImage convention).")] MaskBatch? mask = null,
        [Input(Optional = true, Tooltip = "Layer stack to append to. Leave unconnected to start a new stack.")] LayerDocument? layers = null,
        [Input(Optional = true,
            Tooltip = "Crop each frame to metadata.content_rect where present and place the content at the box position plus the rect offset. Leave on for batches whose frames are padded - it keeps only the real content at its true spot.")]
        bool cropToContent = true,
        [Input(Optional = true, Min = 0, Max = 16384, Tooltip = "Document canvas width. 0 derives it from the placed layers.")] long canvasWidth = 0,
        [Input(Optional = true, Min = 0, Max = 16384, Tooltip = "Document canvas height. 0 derives it from the placed layers.")] long canvasHeight = 0)
    {
        var boxes = BBoxInput.Parse(bboxes);
        bool canvas = canvasWidth > 0 && canvasHeight > 0;
        if (!canvas && boxes.Any(b => b is BBoxInput.Element))
            throw new PyException("ValueError", "normalized element boxes need canvas_width and canvas_height to resolve to pixels");
        var existing = layers?.Layers ?? [];
        long baseZ = existing.Count > 0 ? existing.Max(l => l.ZIndex) + 1 : 0;
        var items = new List<LayerItem>(existing);
        for (int f = 0; f < image.Batch; f++)
        {
            var frame = Frame(image, f);
            var frameMask = mask is null ? null : mask.Batch == 1 ? MaskFrame(mask, 0) : f < mask.Batch ? MaskFrame(mask, f) : null;
            long x = 0, y = 0, z = baseZ + f;
            long? w = null, h = null;
            string? name = null;
            switch (f < boxes.Count ? boxes[f] : null)
            {
                case BBoxInput.Element e:
                {
                    var (ex, ey, ew, eh) = BBoxInput.Pixels(e, canvasWidth, canvasHeight);
                    (x, y) = (ex, ey);
                    if (ew > 0) w = ew;
                    if (eh > 0) h = eh;
                    name = NonEmpty(PyValues.Get(e.Raw, "desc"));
                    break;
                }
                case BBoxInput.Box b:
                {
                    x = Number(b.Raw["x"]) ?? 0;
                    y = Number(b.Raw["y"]) ?? 0;
                    if (Number(b.Raw["width"]) is > 0 and var bw) w = bw;
                    if (Number(b.Raw["height"]) is > 0 and var bh) h = bh;
                    if (b.Raw.TryGetValue("metadata", out var m) && m is DictValue meta)
                    {
                        name = NonEmpty(PyValues.Get(meta, "name")) ?? NonEmpty(PyValues.Get(meta, "desc"));
                        if (Number(PyValues.Get(meta, "z_index")) is { } mz) z = mz;
                        if (cropToContent && ContentRect(PyValues.Get(meta, "content_rect"), frame.Width, frame.Height) is var (cx0, cy0, cx1, cy1))
                        {
                            frame = Crop(frame, cx0, cy0, cx1, cy1);
                            if (frameMask is not null) frameMask = CropMask(frameMask, cx0, cy0, cx1, cy1);
                            x += cx0;
                            y += cy0;
                            w = h = null;
                        }
                    }
                    break;
                }
            }
            items.Add(new LayerItem(frame) { X = x, Y = y, ZIndex = z, W = w, H = h, Mask = frameMask, Name = name });
        }
        return new LayerDocument { Canvas = canvas ? ((int)canvasWidth, (int)canvasHeight) : layers?.Canvas, Layers = items };
    }

    static string? NonEmpty(object? v) => v is string { Length: > 0 } s ? s : null;

    /// <summary>int(v) of an int or float (bools and anything else: none).</summary>
    static long? Number(object? v) => v is bool || !PyValues.IsNumber(v) ? null : (long)PyValues.ToInt(v);

    static (int X0, int Y0, int X1, int Y1)? ContentRect(object? rect, int w, int h)
    {
        if (rect is not ArrayValue { Count: 4 } r) return null;
        var v = new long[4];
        for (int i = 0; i < 4; i++)
            if (Number(r[i]) is { } n) v[i] = n;
            else return null;
        long x0 = Math.Max(0, v[0]), y0 = Math.Max(0, v[1]), x1 = Math.Min(w, x0 + v[2]), y1 = Math.Min(h, y0 + v[3]);
        return x1 > x0 && y1 > y0 ? ((int)x0, (int)y0, (int)x1, (int)y1) : null;
    }

    internal static ImageBatch Frame(ImageBatch image, int f)
    {
        int n = image.Height * image.Width * image.Channels;
        return new ImageBatch(1, image.Height, image.Width, image.Channels, image.Data.AsSpan(f * n, n).ToArray());
    }

    internal static MaskBatch MaskFrame(MaskBatch mask, int f)
    {
        int n = mask.Height * mask.Width;
        return new MaskBatch(1, mask.Height, mask.Width, mask.Data.AsSpan(f * n, n).ToArray());
    }

    static ImageBatch Crop(ImageBatch frame, int x0, int y0, int x1, int y1)
    {
        int w = x1 - x0, h = y1 - y0, c = frame.Channels;
        var result = new ImageBatch(1, h, w, c);
        for (int row = 0; row < h; row++) Array.Copy(frame.Data, ((y0 + row) * frame.Width + x0) * c, result.Data, row * w * c, w * c);
        return result;
    }

    static MaskBatch CropMask(MaskBatch mask, int x0, int y0, int x1, int y1)
    {
        // The mask is sliced like the frame (Python slicing clamps to its own size).
        x0 = Math.Min(x0, mask.Width); x1 = Math.Min(x1, mask.Width); y0 = Math.Min(y0, mask.Height); y1 = Math.Min(y1, mask.Height);
        int w = x1 - x0, h = y1 - y0;
        var result = new MaskBatch(1, h, w);
        for (int row = 0; row < h; row++) Array.Copy(mask.Data, (y0 + row) * mask.Width + x0, result.Data, row * w, w);
        return result;
    }
}

/// <summary>
/// Create Layered Image (ImageCompositor), as measured. Items expand frame by frame (a mask batch of 1 for every frame, otherwise frame by frame; a mask of
/// another size is resized bilinearly; alpha *= 1 - mask) and are stable-sorted by z_index. Each layer goes to 8 bits (rounded), is resized with Pillow's
/// LANCZOS to its display size, flipped (horizontal, then vertical) and rotated with Pillow (bicubic, expanded), then decoded to linear light and composited
/// in stacking order with its blend mode and opacity (<see cref="CompositorBlend"/>) onto a transparent canvas: the document's, else the extent of the
/// placed layers. A layer's picture is centred on its box; a rotated one sits where its turned box's corners reach. The result is sRGB, RGB with an empty
/// mask when fully opaque, else RGBA with mask = 1 - alpha.
/// <para>
/// A saved composition (the editor's JSON) replays when it matches the current layers' fingerprints: its canvas, each layer's transform, opacity, blend,
/// visibility and flips, the drawing order and a background colour. One that does not match, or is malformed, is reported stale and the initial
/// composition is drawn.
/// </para>
/// </summary>
[Node("ImageCompositor", Category = "image/compositing", DisplayName = "Create Layered Image", IsOutput = true, HasIntermediateOutput = true,
    OutputNames = ["IMAGE", "MASK"])]
public sealed class ImageCompositor
{
    const int MaxLayers = 50, MaxCanvas = 16384;

    sealed class Layer
    {
        public required ImageBatch Frame;
        public MaskBatch? Mask;
        public required LayerItem Item;
        public long NativeW, NativeH;
        public double X, Y, Wf, Hf;
        public long W, H;
        public double Rotation, Opacity = 1;
        public string Blend = "normal";
        public bool Visible = true, FlipH, FlipV, Replay;
        public byte[] Rgba = [];
        public float[] Px = [];
        public int Pw, Ph;
    }

    public async Task<(ImageBatch Image, MaskBatch Mask, UiOutput Ui)> ExecuteAsync(
        [Input(Tooltip = "Layer stack to composite; build it with Add Layer. Items are stacked by z_index, batch frames inside an item expand to consecutive layers, and item placement, opacity, and blend mode define the initial composition. Without an explicit document canvas the size is a best-effort maximum extent of the placed layers. A saved composition that matches the current inputs takes priority.")]
        LayerDocument? layers,
        [Input(Type = "COMPOSITOR", Json = "{}", Tooltip = "Layered composition saved by the compositor editor.")] object? compositor,
        NodeContext? context = null)
    {
        var expanded = Expand(layers);
        var ui = new UiOutput();
        if (expanded.Count == 0)
        {
            ui["compositor_inputs"] = new ArrayValue();
            ui["compositor_bboxes"] = new ArrayValue();
            var black = new ImageBatch(1, 64, 64, 3);
            if (context is not null) ui["images"] = await Results.PreviewAsync(black, context, context.Services);
            return (black, new MaskBatch(1, 64, 64), ui);
        }
        foreach (var l in expanded) Prepare(l);
        var fingerprints = new ArrayValue(expanded.Select(l => (object?)Fingerprint(l)));
        var bboxes = new ArrayValue(expanded.Select(l => (object?)new DictValue
        {
            ["x"] = l.Item.X, ["y"] = l.Item.Y, ["width"] = l.W, ["height"] = l.H, ["rotation"] = l.Rotation, ["name"] = l.Item.Name,
            ["visible"] = l.Visible, ["opacity"] = l.Opacity, ["blend"] = l.Blend, ["flipH"] = l.FlipH, ["flipV"] = l.FlipV,
        }));

        int cw, ch;
        if (layers!.Canvas is { } docCanvas) (cw, ch) = docCanvas;
        else
        {
            cw = expanded.Max(l => Place(l).X + l.Pw);
            ch = expanded.Max(l => Place(l).Y + l.Ph);
        }
        var drawn = expanded;
        float[]? background = null;
        bool stale = false;
        switch (Saved(compositor, out stale))
        {
            case DictValue st when !stale:
                if (!Matches(st, fingerprints, expanded.Count, out double sw, out double sh)) { stale = true; break; }
                long rw = PyValues.Round(sw), rh = PyValues.Round(sh);
                if (rw > MaxCanvas || rh > MaxCanvas)
                    throw new PyException("ValueError", $"Compositor canvas {rw}x{rh} exceeds the maximum supported size of {MaxCanvas}x{MaxCanvas}");
                var saved = (ArrayValue)st["layers"]!;
                drawn = [.. expanded.Select((l, k) => Replayed(l, saved[k]))];
                if (PyValues.Get(st, "order") is ArrayValue order && order.All(o => o is long or System.Numerics.BigInteger) && order.Select(o => (long)PyValues.Int(o)).Order().SequenceEqual(Enumerable.Range(0, drawn.Count).Select(i => (long)i)))
                    drawn = [.. order.Select(o => drawn[(int)PyValues.Int(o)])];
                (cw, ch) = ((int)rw, (int)rh);
                background = Background(PyValues.Get(st, "background"));
                break;
        }

        var (image, mask) = Render(drawn, cw, ch, background);
        ui["compositor_inputs"] = fingerprints;
        ui["compositor_bboxes"] = bboxes;
        ui["compositor_canvas"] = new ArrayValue { new DictValue { ["w"] = (long)cw, ["h"] = (long)ch } };
        if (stale) ui["compositor_state_stale"] = new ArrayValue { true };
        if (context is not null)
        {
            ui["images"] = await Results.PreviewAsync(image, context, context.Services);
            var previews = new List<object>();
            foreach (var l in expanded)
                previews.AddRange(await Results.PreviewAsync(new ImageBatch(1, (int)l.NativeH, (int)l.NativeW, 4, PilLanczos.FromBytes(l.Rgba)), context, context.Services));
            ui["compositor_layers"] = previews;
        }
        return (image, mask, ui);
    }

    /// <summary>SpawnGen's fingerprints of the document's layers in stacking order: what a saved composition is matched against.</summary>
    public static ArrayValue Fingerprints(LayerDocument? layers)
    {
        var expanded = Expand(layers);
        foreach (var l in expanded) l.Rgba = Rgba(l.Frame, l.Mask);
        return new ArrayValue(expanded.Select(l => (object?)Fingerprint(l)));
    }

    static List<Layer> Expand(LayerDocument? doc)
    {
        if (doc is null) return [];
        if (doc.Version != 1) throw new PyException("ValueError", $"LAYERS document version {doc.Version} is not supported");
        var result = new List<Layer>();
        foreach (var item in doc.Layers)
        {
            if (item.Type != "raster") throw new PyException("ValueError", $"LAYERS item type '{item.Type}' is not supported yet");
            var blend = string.IsNullOrEmpty(item.BlendMode) ? "normal" : item.BlendMode;
            if (!CompositorBlend.ModeNames.Contains(blend)) throw new PyException("ValueError", $"LAYERS item blend_mode '{blend}' is not a known blend mode");
            var image = item.Image;
            for (int f = 0; f < image.Batch; f++)
            {
                var m = item.Mask is not { } mk ? null : mk.Batch == 1 ? LayersFromBoundingBoxes.MaskFrame(mk, 0) : f < mk.Batch ? LayersFromBoundingBoxes.MaskFrame(mk, f) : null;
                if (m is not null && (m.Height != image.Height || m.Width != image.Width))
                    m = new MaskBatch(1, image.Height, image.Width, TorchInterpolate.Resize(m.Data, 1, m.Height, m.Width, image.Height, image.Width, "bilinear"));
                long w = item.W is > 0 ? item.W.Value : image.Width, h = item.H is > 0 ? item.H.Value : image.Height;
                result.Add(new Layer
                {
                    Frame = LayersFromBoundingBoxes.Frame(image, f), Mask = m, Item = item, NativeW = image.Width, NativeH = image.Height,
                    X = item.X, Y = item.Y, W = w, H = h, Wf = w, Hf = h, Rotation = item.Rotation ?? 0, Opacity = item.Opacity ?? 1, Blend = blend,
                    Visible = item.Visible ?? true, FlipH = item.FlipH, FlipV = item.FlipV,
                });
            }
        }
        if (result.Count > MaxLayers) throw new PyException("ValueError", $"Compositor supports at most {MaxLayers} layers, got {result.Count}");
        return [.. result.OrderBy(l => l.Item.ZIndex)]; // stable
    }

    /// <summary>The frame as 8-bit RGBA: alpha (1 for RGB) times 1 - mask, every value round(v * 255) in float32, clamped.</summary>
    static byte[] Rgba(ImageBatch frame, MaskBatch? mask)
    {
        int n = frame.Height * frame.Width, c = frame.Channels;
        var px = new byte[n * 4];
        static byte Q(float v)
        {
            float r = MathF.Round(v * 255f, MidpointRounding.ToEven);
            return (byte)(r < 0 ? 0 : r > 255 ? 255 : r);
        }
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < 3; k++) px[i * 4 + k] = Q(frame.Data[i * c + (c >= 3 ? k : 0)]);
            float a = c == 4 ? frame.Data[i * c + 3] : 1f;
            if (mask is not null) a *= 1f - mask.Data[i];
            px[i * 4 + 3] = Q(a);
        }
        return px;
    }

    /// <summary>The layer's picture at its display size, flipped and rotated, in linear light.</summary>
    static void Prepare(Layer l)
    {
        if (l.Rgba.Length == 0) l.Rgba = Rgba(l.Frame, l.Mask);
        var px = l.Rgba;
        int w = (int)l.NativeW, h = (int)l.NativeH;
        if (l.W != w || l.H != h) { px = PilLanczos.Resize(px, w, h, 4, (int)l.W, (int)l.H); (w, h) = ((int)l.W, (int)l.H); }
        if (l.FlipH) px = PilTransform.FlipLeftRight(px, w, h, 4);
        if (l.FlipV) px = PilTransform.FlipTopBottom(px, w, h, 4);
        if (l.Rotation != 0) (px, w, h) = PilTransform.RotateExpand(px, w, h, -(l.Rotation * (180.0 / Math.PI))); // Python's math.degrees
        var lin = new float[px.Length];
        for (int i = 0; i < px.Length; i++)
        {
            float v = px[i] / 255f;
            lin[i] = i % 4 == 3 ? v : CompositorBlend.SrgbToLinear(v);
        }
        (l.Px, l.Pw, l.Ph) = (lin, w, h);
    }

    /// <summary>Where the layer's picture goes: centred on its box; rotated, at the reach of its turned corners (a replay centres on the saved size).</summary>
    static (int X, int Y) Place(Layer l)
    {
        if (l.Rotation != 0 && !l.Replay)
        {
            double c = Math.Cos(l.Rotation), s = Math.Sin(l.Rotation), hw = l.W / 2.0, hh = l.H / 2.0, cx = l.X + hw, cy = l.Y + hh;
            double minX = double.PositiveInfinity, minY = double.PositiveInfinity;
            foreach (var (px, py) in new[] { (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh) })
            {
                minX = Math.Min(minX, cx + px * c + py * s);
                minY = Math.Min(minY, cy + -(px * s) + py * c);
            }
            return ((int)Math.Floor(minX), (int)Math.Floor(minY));
        }
        if (l.Rotation != 0) return ((int)Math.Floor(l.X + l.Wf / 2 - l.Pw / 2.0), (int)Math.Floor(l.Y + l.Hf / 2 - l.Ph / 2.0));
        return ((int)Math.Floor(l.X + l.W / 2.0 - l.Pw / 2.0), (int)Math.Floor(l.Y + l.H / 2.0 - l.Ph / 2.0));
    }

    static (ImageBatch, MaskBatch) Render(List<Layer> layers, int cw, int ch, float[]? background)
    {
        var canvas = new float[cw * ch * 4];
        if (background is not null)
            for (int i = 0; i < cw * ch; i++) background.CopyTo(canvas, i * 4);
        foreach (var l in layers)
        {
            if (!l.Visible) continue;
            var (ox, oy) = Place(l);
            int x0 = Math.Max(ox, 0), y0 = Math.Max(oy, 0), x1 = Math.Min(ox + l.Pw, cw), y1 = Math.Min(oy + l.Ph, ch);
            if (x1 <= x0 || y1 <= y0) continue;
            CompositorBlend.Composite(l.Blend, canvas, y0 * cw + x0, cw, l.Px, (y0 - oy) * l.Pw + (x0 - ox), l.Pw, x1 - x0, y1 - y0, l.Opacity);
        }
        bool opaque = true;
        for (int i = 3; i < canvas.Length; i += 4) if (canvas[i] != 1f) { opaque = false; break; }
        int c = opaque ? 3 : 4;
        var image = new ImageBatch(1, ch, cw, c);
        var mask = new MaskBatch(1, ch, cw);
        for (int i = 0; i < cw * ch; i++)
        {
            for (int k = 0; k < 3; k++) image.Data[i * c + k] = CompositorBlend.LinearToSrgb(Math.Clamp(canvas[i * 4 + k], 0f, 1f));
            if (!opaque)
            {
                image.Data[i * c + 3] = canvas[i * 4 + 3];
                mask.Data[i] = 1f - canvas[i * 4 + 3];
            }
        }
        return (image, mask);
    }

    /// <summary>The compositor value as a dict to replay: none for nothing saved ({} or empty), stale for text that is not JSON.</summary>
    static object? Saved(object? compositor, out bool stale)
    {
        stale = false;
        if (compositor is string text)
        {
            if (PyText.Strip(text).Length == 0) return null;
            try { compositor = PyJson.Loads(text); }
            catch (PyException) { stale = true; return null; }
        }
        return compositor is DictValue { Count: > 0 } d ? d : null;
    }

    /// <summary>Whether a saved composition fits the current layers: fingerprints, version 1, one entry per layer and a finite positive canvas.</summary>
    static bool Matches(DictValue st, ArrayValue fingerprints, int count, out double w, out double h)
    {
        w = h = double.NaN;
        var cv = PyValues.Get(st, "canvas") as DictValue ?? [];
        try
        {
            w = PyValues.ToFloat(PyValues.Get(cv, "w"));
            h = PyValues.ToFloat(PyValues.Get(cv, "h"));
        }
        catch (PyException) { w = h = double.NaN; }
        return PyValues.Equal(PyValues.Get(st, "inputs"), fingerprints) && PyValues.Equal(PyValues.Get(st, "version"), 1L)
            && PyValues.Get(st, "layers") is ArrayValue saved && saved.Count == count
            && w > 0 && h > 0 && double.IsFinite(w) && double.IsFinite(h);
    }

    /// <summary>A number the editor saved (bools count, text does not); null otherwise.</summary>
    static double? Num(object? v) => v is bool b ? (b ? 1 : 0) : PyValues.IsNumber(v) ? PyValues.Float(v) : null;

    static Layer Replayed(Layer l, object? saved)
    {
        var s = saved as DictValue ?? [];
        var t = PyValues.Get(s, "transform") as DictValue ?? [];
        double Coord(string key) => Num(PyValues.Get(t, key, 0L)) is { } v && double.IsFinite(v) ? v : 0;
        (long Size, double Float) Size(string key, long native) =>
            Num(PyValues.Get(t, key)) is { } v && double.IsFinite(v) && v > 0 ? (Math.Max(1, PyValues.Round(v)), v) : (native, native);
        var (w, wf) = Size("w", l.NativeW);
        var (h, hf) = Size("h", l.NativeH);
        double rotation = Num(PyValues.Get(t, "rotation", 0L)) is { } r && double.IsFinite(r) ? r : 0;
        double opacity = Num(PyValues.Get(s, "opacity", 1L)) ?? 1;
        opacity = double.IsNaN(opacity) ? 1 : Math.Clamp(opacity, 0, 1);
        var blend = PyValues.Get(s, "blend", "normal") is string b && CompositorBlend.ModeNames.Contains(b) ? b : "normal";
        var replay = new Layer
        {
            Frame = l.Frame, Mask = l.Mask, Item = l.Item, NativeW = l.NativeW, NativeH = l.NativeH, Rgba = l.Rgba,
            X = Coord("x"), Y = Coord("y"), W = w, H = h, Wf = wf, Hf = hf, Rotation = rotation, Opacity = opacity, Blend = blend,
            Visible = PyValues.Truthy(PyValues.Get(s, "visible", true)), FlipH = PyValues.Truthy(PyValues.Get(s, "flipH", false)),
            FlipV = PyValues.Truthy(PyValues.Get(s, "flipV", false)), Replay = true,
        };
        Prepare(replay);
        return replay;
    }

    /// <summary>The saved background as a linear RGBA pixel: "#rgb" / "#rrggbb" (white otherwise) at its opacity; none when hidden or transparent.</summary>
    static float[]? Background(object? bg)
    {
        if (bg is not DictValue d || !PyValues.Truthy(PyValues.Get(d, "visible", true))) return null;
        double op = Num(PyValues.Get(d, "opacity", 1L)) ?? 1;
        op = double.IsNaN(op) ? 1 : Math.Clamp(op, 0, 1);
        if (op <= 0) return null;
        double[] rgb = [1, 1, 1];
        if (PyValues.Get(d, "color") is string c && System.Text.RegularExpressions.Regex.IsMatch(c, "^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})\\z"))
        {
            var hex = c[1..];
            if (hex.Length == 3) hex = string.Concat(hex.Select(ch => $"{ch}{ch}"));
            for (int i = 0; i < 3; i++) rgb[i] = Convert.ToInt32(hex.Substring(2 * i, 2), 16) / 255.0;
        }
        return [CompositorBlend.SrgbToLinear((float)rgb[0]), CompositorBlend.SrgbToLinear((float)rgb[1]), CompositorBlend.SrgbToLinear((float)rgb[2]), (float)op];
    }

    /// <summary>
    /// SpawnGen's layer fingerprint: 16 hex digits of SHA-256 over the 8-bit pixels (mask applied), their size and the initial placement, size, rotation,
    /// opacity, blend mode, flips and visibility. ComfyUI hashes the same things its own way.
    /// </summary>
    static string Fingerprint(Layer l)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        hash.AppendData(l.Rgba);
        var text = string.Join("|", l.NativeW, l.NativeH, l.Item.X, l.Item.Y, l.W, l.H, BitConverter.DoubleToInt64Bits(l.Rotation),
            BitConverter.DoubleToInt64Bits(l.Opacity), l.Blend, l.FlipH, l.FlipV, l.Visible);
        hash.AppendData(System.Text.Encoding.UTF8.GetBytes(text));
        return Convert.ToHexStringLower(hash.GetHashAndReset())[..16];
    }
}
