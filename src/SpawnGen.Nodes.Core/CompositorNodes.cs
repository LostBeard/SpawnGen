using System.Security.Cryptography;
using System.Text;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// ComfyUI's LAYERS value (comfy_api Layers.Document): a stack of raster layers with an optional document canvas. Add Layer and Layers From Bounding
/// Boxes build it; Create Layered Image composites it.
/// </summary>
[Socket("LAYERS")]
public sealed class LayerDocument
{
    public int Version { get; init; } = 1;
    /// <summary>The document canvas (width, height), when one was given.</summary>
    public (int W, int H)? Canvas { get; init; }
    public IReadOnlyList<LayerItem> Layers { get; init; } = [];
}

/// <summary>
/// One layer item (Layers.LayerItem): the picture (a batch expands to consecutive layers), its placement and look. Null fields are absent in ComfyUI's
/// dict and take its defaults (opacity 1, blend "normal", visible, no rotation, the picture's own size).
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
    /// <summary>Clockwise, in radians.</summary>
    public double? Rotation { get; init; }
    public long? W { get; init; }
    public long? H { get; init; }
}

/// <summary>ComfyUI's compositor (comfy_extras/nodes_compositor.py): the layer stack expanded to frames, placed, blended and flattened.</summary>
public static class Compositor
{
    public const int MaxLayers = 50;
    public const int MaxResolution = 16384;
    const float OpaqueEpsilon = 1e-3f;

    /// <summary>One frame of a layer item, as expand_item_frames makes it.</summary>
    public sealed record Frame(float[] Pixels, int Height, int Width, int Channels, float[]? Mask, int MaskHeight, int MaskWidth, string? Name, long X, long Y,
        long W, long H, double Rotation, double Opacity, string Blend, bool Visible, bool FlipH, bool FlipV);

    /// <summary>document_items: the raster items, checked (version 1, known blend modes) and stably sorted by z_index.</summary>
    public static List<LayerItem> DocumentItems(LayerDocument? doc)
    {
        if (doc is null) return [];
        if (doc.Version != 1) throw new PyException("ValueError", $"LAYERS document version {doc.Version} is not supported");
        var items = new List<LayerItem>();
        foreach (var item in doc.Layers)
        {
            if (item.Type != "raster") throw new PyException("ValueError", $"LAYERS item type {PyText.Repr(item.Type)} is not supported yet");
            if (item.BlendMode is { } blend && !CompositorBlend.ModeNames.Contains(blend))
                throw new PyException("ValueError", $"LAYERS item blend_mode {PyText.Repr(blend)} is not a known blend mode");
            items.Add(item);
        }
        return [.. items.OrderBy(i => i.ZIndex)];
    }

    /// <summary>_item_mask_frame: a one-frame mask serves every frame, else the frame's own (none past the mask's batch).</summary>
    public static MaskBatch? MaskFrame(MaskBatch? mask, int index) =>
        mask is null ? null : mask.Batch == 1 ? Slice(mask, 0) : index < mask.Batch ? Slice(mask, index) : null;

    static MaskBatch Slice(MaskBatch m, int index)
    {
        int plane = m.Height * m.Width;
        return new MaskBatch(1, m.Height, m.Width, m.Data[(index * plane)..((index + 1) * plane)]);
    }

    public static List<Frame> ExpandItemFrames(IEnumerable<LayerItem> items)
    {
        var frames = new List<Frame>();
        foreach (var item in items)
        {
            var image = item.Image;
            int plane = image.Height * image.Width * image.Channels;
            for (int index = 0; index < image.Batch; index++)
            {
                var mask = MaskFrame(item.Mask, index);
                long w = item.W ?? 0, h = item.H ?? 0;
                frames.Add(new Frame(image.Data[(index * plane)..((index + 1) * plane)], image.Height, image.Width, image.Channels,
                    mask?.Data, mask?.Height ?? 0, mask?.Width ?? 0, item.Name, item.X, item.Y, w > 0 ? w : image.Width, h > 0 ? h : image.Height,
                    item.Rotation ?? 0.0, item.Opacity ?? 1.0, item.BlendMode ?? "normal", item.Visible ?? true, item.FlipH, item.FlipV));
            }
        }
        if (frames.Count > MaxLayers) throw new PyException("ValueError", $"Compositor supports at most {MaxLayers} layers, got {frames.Count}");
        return frames;
    }

    /// <summary>frame_alpha: the picture's alpha channel times (1 - mask), the mask resized bilinearly to the picture when its size differs; null without either.</summary>
    public static float[]? FrameAlpha(Frame f)
    {
        float[]? alpha = null;
        int n = f.Height * f.Width;
        if (f.Channels == 4)
        {
            alpha = new float[n];
            for (int p = 0; p < n; p++) alpha[p] = f.Pixels[p * 4 + 3];
        }
        if (f.Mask is null) return alpha;
        var m = f.MaskHeight != f.Height || f.MaskWidth != f.Width
            ? TorchInterpolate.Resize(f.Mask, 1, f.MaskHeight, f.MaskWidth, f.Height, f.Width, "bilinear")
            : f.Mask;
        var inv = new float[n];
        for (int p = 0; p < n; p++) inv[p] = Math.Clamp(1f - m[p], 0f, 1f);
        if (alpha is null) return inv;
        for (int p = 0; p < n; p++) alpha[p] *= inv[p];
        return alpha;
    }

    static byte ToByte(float v) => (byte)Math.Clamp(MathF.Round(v * 255f, MidpointRounding.ToEven), 0f, 255f); // np.clip(np.rint(v * 255.0), 0, 255)

    /// <summary>input_fingerprints: what a saved composition remembers of each layer (picture, alpha and placement), the first 16 hex digits of a SHA-256.</summary>
    public static List<string> Fingerprints(IReadOnlyList<Frame> frames, IReadOnlyList<float[]?> alphas)
    {
        var result = new List<string>();
        for (int i = 0; i < frames.Count; i++)
        {
            var f = frames[i];
            using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            sha.AppendData(Encoding.UTF8.GetBytes($"(1, {f.Height}, {f.Width}, {f.Channels})"));
            int n = f.Height * f.Width;
            var rgb8 = new byte[n * 3];
            for (int p = 0; p < n; p++) for (int k = 0; k < 3; k++) rgb8[p * 3 + k] = ToByte(f.Pixels[p * f.Channels + k]);
            sha.AppendData(rgb8);
            if (alphas[i] is { } alpha) sha.AppendData(alpha.Select(ToByte).ToArray());
            var placement = string.Join(", ", f.X.ToString(System.Globalization.CultureInfo.InvariantCulture), f.Y.ToString(System.Globalization.CultureInfo.InvariantCulture),
                f.W.ToString(System.Globalization.CultureInfo.InvariantCulture), f.H.ToString(System.Globalization.CultureInfo.InvariantCulture),
                PyNumber.FloatRepr(f.Rotation), PyNumber.FloatRepr(f.Opacity), PyText.Repr(f.Blend), f.Visible ? "True" : "False", f.FlipH ? "True" : "False",
                f.FlipV ? "True" : "False");
            sha.AppendData(Encoding.UTF8.GetBytes($"({placement})"));
            result.Add(Convert.ToHexStringLower(sha.GetHashAndReset())[..16]);
        }
        return result;
    }

    /// <summary>A composition to render: the canvas size, one entry per layer (JSON-like, as the editor saves them), the background and the stacking order.</summary>
    public sealed record State((int W, int H) Canvas, ArrayValue Layers, ArrayValue? Inputs, Background? Background, IReadOnlyList<int>? Order);

    public sealed record Background(string Color, double Opacity, bool Visible);

    static readonly HashSet<char> HexDigits = [.. "0123456789abcdef"];

    static string NormalizeHexColor(object? value)
    {
        if (value is string s)
        {
            var text = PyText.Strip(s).ToLowerInvariant();
            if (text.StartsWith('#'))
            {
                var digits = text[1..];
                if (digits.Length == 3 && digits.All(HexDigits.Contains)) digits = string.Concat(digits.Select(ch => new string(ch, 2)));
                if (digits.Length == 6 && digits.All(HexDigits.Contains)) return "#" + digits;
            }
        }
        return "#ffffff";
    }

    /// <summary>_number: a finite int or float as a double, else the default.</summary>
    static double Number(DictValue source, string key, double fallback)
    {
        var value = source.TryGetValue(key, out var v) ? v : fallback;
        if (!PyValues.IsNumber(value)) return fallback;
        double d = PyValues.Float(value);
        return double.IsFinite(d) ? d : fallback;
    }

    static double Clamp01(double v) => Math.Min(Math.Max(v, 0.0), 1.0);

    static Background? ParseBackground(object? entry) => entry is DictValue d
        ? new Background(NormalizeHexColor(PyValues.Get(d, "color")), Clamp01(Number(d, "opacity", 1.0)), PyValues.Truthy(PyValues.Get(d, "visible", true)))
        : null;

    static List<int>? ParseOrder(object? value, int layerCount)
    {
        if (value is not ArrayValue { Count: > 0 } list || !list.All(v => v is long or int || v is System.Numerics.BigInteger)) return null;
        var ints = list.Select(PyValues.Int).ToList();
        if (!ints.OrderBy(v => v).SequenceEqual(Enumerable.Range(0, layerCount).Select(v => (System.Numerics.BigInteger)v))) return null;
        return ints.Select(v => (int)v).ToList();
    }

    /// <summary>layer_state_provided: a non-empty dict, or text other than "" and "{}".</summary>
    public static bool StateProvided(object? raw) => raw switch
    {
        DictValue d => d.Count > 0,
        string s => s is not ("" or "{}"),
        _ => false,
    };

    /// <summary>parse_layer_state: the editor's saved composition (a dict or its JSON text), or null when it is not a usable one.</summary>
    public static State? ParseState(object? raw)
    {
        if (raw is string s)
        {
            if (PyText.Strip(s).Length == 0) return null;
            try { raw = PyJson.Loads(s); }
            catch (PyException) { return null; }
        }
        if (raw is not DictValue state) return null;
        var version = PyValues.Get(state, "version");
        if (version is not null && !PyValues.Equal(version, 1L)) return null;
        if (PyValues.Get(state, "canvas") is not DictValue canvas || PyValues.Get(state, "layers") is not ArrayValue { Count: > 0 } layers) return null;
        long w, h;
        try
        {
            w = PyValues.Round(PyValues.ToFloat(PyValues.Get(canvas, "w")));
            h = PyValues.Round(PyValues.ToFloat(PyValues.Get(canvas, "h")));
        }
        catch (PyException e) when (e.PyType is "TypeError" or "ValueError" or "OverflowError") { return null; }
        if (w <= 0 || h <= 0) return null;
        var inputs = PyValues.Get(state, "inputs") is ArrayValue given && given.Count == layers.Count && given.All(v => v is string) ? given : null;
        return new State(((int)Math.Min(w, int.MaxValue), (int)Math.Min(h, int.MaxValue)), layers, inputs, ParseBackground(PyValues.Get(state, "background")),
            ParseOrder(PyValues.Get(state, "order"), layers.Count));
    }

    /// <summary>state_from_items: the composition the layer items describe (no saved editing): placement, look, a hidden white background.</summary>
    public static State StateFromItems(IReadOnlyList<Frame> frames, (int W, int H) canvas)
    {
        var layers = new ArrayValue();
        foreach (var f in frames)
            layers.Add(new DictValue
            {
                ["name"] = f.Name, ["visible"] = f.Visible, ["opacity"] = f.Opacity, ["blend"] = f.Blend, ["flipH"] = f.FlipH, ["flipV"] = f.FlipV,
                ["transform"] = new DictValue { ["x"] = f.X, ["y"] = f.Y, ["w"] = f.W, ["h"] = f.H, ["rotation"] = f.Rotation },
            });
        return new State(canvas, layers, null, new Background("#ffffff", 1.0, false), null);
    }

    /// <summary>canvas_extent: the right and bottom edge every placed layer reaches (at least 1 x 1).</summary>
    public static (int W, int H) CanvasExtent(IEnumerable<Frame> frames)
    {
        long right = 1, bottom = 1;
        foreach (var f in frames)
        {
            var (bx, by, bw, bh) = CompositorBlend.PlacedBounds(f.X, f.Y, f.W, f.H, f.Rotation);
            right = Math.Max(right, (long)bx + bw);
            bottom = Math.Max(bottom, (long)by + bh);
        }
        return ((int)right, (int)bottom);
    }

    sealed record LayerParams(bool Visible, double Opacity, string Blend, double X, double Y, double W, double H, double Rotation, bool FlipH, bool FlipV);

    static LayerParams ParamsOf(object? entry, int naturalW, int naturalH)
    {
        var e = entry as DictValue ?? [];
        var t = PyValues.Get(e, "transform") as DictValue ?? [];
        static double Size(double value, int natural) => value <= 0 ? natural : Math.Min(value, MaxResolution);
        return new LayerParams(
            PyValues.Truthy(PyValues.Get(e, "visible", true)),
            // The state is untrusted (saved in the workflow, or posted to /prompt): opacity is clamped as the background's is.
            Clamp01(Number(e, "opacity", 1.0)),
            PyValues.Get(e, "blend") as string ?? "normal",
            Math.Min(Math.Max(Number(t, "x", 0.0), -MaxResolution), MaxResolution),
            Math.Min(Math.Max(Number(t, "y", 0.0), -MaxResolution), MaxResolution),
            Size(Number(t, "w", naturalW), naturalW),
            Size(Number(t, "h", naturalH), naturalH),
            Number(t, "rotation", 0.0),
            PyValues.Truthy(PyValues.Get(e, "flipH", false)),
            PyValues.Truthy(PyValues.Get(e, "flipV", false)));
    }

    /// <summary>_prepare_layer_bitmap: the layer as 8-bit RGBA, flipped, resized (Lanczos) and turned (bicubic, canvas grown) as Pillow does it.</summary>
    static (byte[] Px, int W, int H) PrepareBitmap(Frame f, LayerParams p, float[]? alpha)
    {
        int w = f.Width, h = f.Height, n = w * h;
        var px = new byte[n * 4];
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < 3; k++) px[i * 4 + k] = ToByte(f.Pixels[i * f.Channels + k]);
            px[i * 4 + 3] = alpha is null ? (byte)255 : ToByte(alpha[i]);
        }
        if (p.FlipH) px = PilTransform.FlipLeftRight(px, w, h, 4);
        if (p.FlipV) px = PilTransform.FlipTopBottom(px, w, h, 4);
        int tw = (int)Math.Max(1, PyValues.Round(p.W)), th = (int)Math.Max(1, PyValues.Round(p.H));
        if (tw != w || th != h)
        {
            px = PilLanczos.Resize(px, w, h, 4, tw, th);
            (w, h) = (tw, th);
        }
        if (p.Rotation != 0) (px, w, h) = PilTransform.RotateExpand(px, w, h, -(p.Rotation * PilTransform.RadToDeg));
        return (px, w, h);
    }

    static long FloorDiv(long a, long b) => (long)Math.Floor((double)a / b);

    /// <summary>_place_in_bounds: the bitmap centred in a bw x bh buffer of straight linear-light RGBA (0 outside it).</summary>
    static float[] PlaceInBounds(byte[] px, int aw, int ah, int bw, int bh)
    {
        var buf = new float[(long)bh * bw * 4];
        long ox = FloorDiv(bw - aw, 2), oy = FloorDiv(bh - ah, 2);
        long dx0 = Math.Max(ox, 0), dy0 = Math.Max(oy, 0), dx1 = Math.Min(ox + aw, bw), dy1 = Math.Min(oy + ah, bh);
        for (long y = dy0; y < dy1; y++)
            for (long x = dx0; x < dx1; x++)
            {
                long s = ((y - oy) * aw + (x - ox)) * 4, d = (y * bw + x) * 4;
                for (int k = 0; k < 3; k++) buf[d + k] = CompositorBlend.SrgbToLinear(px[s + k] / 255f);
                buf[d + 3] = px[s + 3] / 255f;
            }
        return buf;
    }

    /// <summary>composite_from_state: the layers in order over the (optional) background on the state's canvas; straight RGBA in sRGB, 0..1.</summary>
    public static ImageBatch CompositeFromState(IReadOnlyList<Frame> frames, State state, IReadOnlyList<float[]?> alphas)
    {
        var (cw, ch) = state.Canvas;
        if (cw > MaxResolution || ch > MaxResolution)
            throw new PyException("ValueError", $"Compositor canvas {cw}x{ch} exceeds the maximum supported size of {MaxResolution}x{MaxResolution}");
        var canvas = new float[(long)ch * cw * 4];
        if (state.Background is { Visible: true, Opacity: > 0 } bg)
        {
            var (r, g, b) = BoundingBoxes.HexToRgb(bg.Color);
            float lr = CompositorBlend.SrgbToLinear(r / 255f), lg = CompositorBlend.SrgbToLinear(g / 255f), lb = CompositorBlend.SrgbToLinear(b / 255f);
            var layer = new float[canvas.Length];
            for (long i = 0; i < layer.Length; i += 4) { layer[i] = lr; layer[i + 1] = lg; layer[i + 2] = lb; layer[i + 3] = 1f; }
            CompositorBlend.Composite("normal", canvas, 0, cw * 4, layer, 0, cw * 4, cw, ch, bg.Opacity);
        }
        var order = state.Order ?? Enumerable.Range(0, frames.Count).ToList();
        foreach (var index in order)
        {
            if (index < 0 || index >= frames.Count) continue;
            var f = frames[index];
            var p = ParamsOf(index < state.Layers.Count ? state.Layers[index] : null, f.Width, f.Height);
            if (!p.Visible) continue;
            var (px, aw, ah) = PrepareBitmap(f, p, index < alphas.Count ? alphas[index] : null);
            var (bx, by, bw, bh) = CompositorBlend.PlacedBounds(p.X, p.Y, p.W, p.H, p.Rotation);
            var buf = PlaceInBounds(px, aw, ah, bw, bh);
            int x0 = Math.Max(bx, 0), y0 = Math.Max(by, 0), x1 = Math.Min(bx + bw, cw), y1 = Math.Min(by + bh, ch);
            if (x0 >= x1 || y0 >= y1) continue;
            CompositorBlend.Composite(p.Blend, canvas, (y0 * cw + x0) * 4, cw * 4, buf, ((y0 - by) * bw + (x0 - bx)) * 4, bw * 4, x1 - x0, y1 - y0, p.Opacity);
        }
        var result = new ImageBatch(1, ch, cw, 4);
        for (long i = 0; i < canvas.Length; i += 4)
        {
            for (int k = 0; k < 3; k++) result.Data[i + k] = CompositorBlend.LinearToSrgb(Math.Clamp(canvas[i + k], 0f, 1f));
            result.Data[i + 3] = Math.Clamp(canvas[i + 3], 0f, 1f);
        }
        return result;
    }

    /// <summary>composite_outputs: an opaque composite as RGB with an all-zero mask, else RGBA with mask = 1 - alpha.</summary>
    public static (ImageBatch Image, MaskBatch Mask) Outputs(ImageBatch rgba)
    {
        int n = rgba.Height * rgba.Width;
        var mask = new MaskBatch(1, rgba.Height, rgba.Width);
        bool opaque = true;
        for (int p = 0; p < n; p++) if (!(rgba.Data[p * 4 + 3] >= 1f - OpaqueEpsilon)) { opaque = false; break; }
        if (opaque)
        {
            var rgb = new ImageBatch(1, rgba.Height, rgba.Width, 3);
            for (int p = 0; p < n; p++) for (int k = 0; k < 3; k++) rgb.Data[p * 3 + k] = rgba.Data[p * 4 + k];
            return (rgb, mask);
        }
        for (int p = 0; p < n; p++) mask.Data[p] = Math.Clamp(1f - rgba.Data[p * 4 + 3], 0f, 1f);
        return (rgba, mask);
    }

    /// <summary>layer_preview_tensor: the layer's RGB with its alpha (when it has one) as a picture.</summary>
    public static ImageBatch LayerPreview(Frame f, float[]? alpha)
    {
        int n = f.Height * f.Width, c = alpha is null ? 3 : 4;
        var img = new ImageBatch(1, f.Height, f.Width, c);
        for (int p = 0; p < n; p++)
        {
            for (int k = 0; k < 3; k++) img.Data[p * c + k] = f.Pixels[p * f.Channels + k];
            if (alpha is not null) img.Data[p * 4 + 3] = alpha[p];
        }
        return img;
    }
}

[Node(Category = "image", DisplayName = "Create Layered Image", IsOutput = true, OutputNames = ["IMAGE", "MASK"])]
public sealed class ImageCompositor
{
    /// <summary>
    /// ComfyUI's Create Layered Image: the stack composited on its document canvas (else the extent of the placed layers); a saved composition from the
    /// compositor editor takes over while its layer fingerprints match the inputs. Outputs the picture (RGBA when it has transparent areas) and its
    /// transparency mask; the editor gets the layer previews, fingerprints, boxes and canvas. With no layers: 64 x 64 black.
    /// </summary>
    public async Task<(ImageBatch Image, MaskBatch Mask, UiOutput Ui)> ExecuteAsync(
        [Input(Tooltip = "Layer stack to composite; build it with Add Layer. Items are stacked by z_index, batch frames inside an item expand to consecutive layers, and item placement, opacity, and blend mode define the initial composition. Without an explicit document canvas the size is a best-effort maximum extent of the placed layers. A saved composition that matches the current inputs takes priority.")]
        LayerDocument layers,
        [Input(Type = "COMPOSITOR", Json = "{}", Tooltip = "Layered composition saved by the compositor editor.")] object? compositor = null,
        NodeContext? context = null, IServiceProvider? services = null)
    {
        var frames = Compositor.ExpandItemFrames(Compositor.DocumentItems(layers));
        var alphas = frames.Select(Compositor.FrameAlpha).ToList();
        var fingerprints = Compositor.Fingerprints(frames, alphas);
        var state = Compositor.ParseState(compositor);
        bool replay = state is not null && frames.Count > 0 && state.Inputs is { } inputs && inputs.Cast<string>().SequenceEqual(fingerprints);
        (int W, int H)? canvas = null;
        ImageBatch output;
        if (replay)
        {
            canvas = state!.Canvas;
            output = Compositor.CompositeFromState(frames, state, alphas);
        }
        else if (frames.Count > 0)
        {
            canvas = layers.Canvas ?? Compositor.CanvasExtent(frames);
            output = Compositor.CompositeFromState(frames, Compositor.StateFromItems(frames, canvas.Value), alphas);
        }
        else output = new ImageBatch(1, 64, 64, 3);
        bool stale = Compositor.StateProvided(compositor) && !replay;
        var (image, mask) = output.Channels == 4 ? Compositor.Outputs(output) : (output, new MaskBatch(1, output.Height, output.Width));

        var ui = new UiOutput();
        if (context is not null && services is not null)
        {
            ui["images"] = await Results.PreviewAsync(image, context, services);
            var layerRefs = new List<object>();
            for (int i = 0; i < frames.Count; i++) layerRefs.AddRange(await Results.PreviewAsync(Compositor.LayerPreview(frames[i], alphas[i]), context, services));
            ui["compositor_layers"] = layerRefs;
        }
        ui["compositor_inputs"] = new ArrayValue(fingerprints);
        ui["compositor_bboxes"] = new ArrayValue(frames.Select(f => (object?)new DictValue
        {
            ["x"] = f.X, ["y"] = f.Y, ["width"] = f.W, ["height"] = f.H, ["rotation"] = f.Rotation, ["name"] = f.Name, ["visible"] = f.Visible,
            ["opacity"] = f.Opacity, ["blend"] = f.Blend, ["flipH"] = f.FlipH, ["flipV"] = f.FlipV,
        }));
        if (canvas is { } c) ui["compositor_canvas"] = new ArrayValue { new DictValue { ["w"] = (long)c.W, ["h"] = (long)c.H } };
        if (stale) ui["compositor_state_stale"] = new ArrayValue { true };
        return (image, mask, ui);
    }
}

[Node(Category = "image", DisplayName = "Add Layer")]
public sealed class AddLayer
{
    /// <summary>ComfyUI's Add Layer: the stack with one more item (only what differs from the defaults is stated); the document canvas is kept.</summary>
    public LayerDocument Execute(
        [Input(Tooltip = "Layer content at its native size. A batch expands to consecutive layers.")] ImageBatch image,
        [Input(Tooltip = "Layer stack to append to. Leave unconnected to start a new stack.")] LayerDocument? layers = null,
        [Input(Tooltip = "Transparency mask for this layer. Masked areas (value 1) become transparent, multiplying with any alpha channel the image already carries.")] MaskBatch? mask = null,
        [Input(Optional = true, Tooltip = "Layer name shown in the compositor editor.")] string name = "",
        [Input(Optional = true, Min = -16384, Max = 16384, Tooltip = "Initial horizontal placement on the canvas.")] long x = 0,
        [Input(Optional = true, Min = -16384, Max = 16384, Tooltip = "Initial vertical placement on the canvas.")] long y = 0,
        [Input(Optional = true, Min = 0.0, Max = 1.0, Step = 0.01, Tooltip = "Initial layer opacity.")] double opacity = 1.0,
        [Input(Optional = true, OptionsProvider = nameof(BlendModes), Tooltip = "Initial blend mode, applied against the layers below. On the bottom layer over the default transparent background, non-normal modes produce transparency.")] string blendMode = "normal",
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
            Name = name.Length > 0 ? name : null,
            Opacity = opacity != 1.0 ? opacity : null,
            BlendMode = blendMode != "normal" ? blendMode : null,
            Rotation = rotation != 0.0 ? rotation * PilTransform.DegToRad : null,
            W = width > 0 ? width : null,
            H = height > 0 ? height : null,
            FlipH = flipH, FlipV = flipV,
        };
        return new LayerDocument { Layers = [.. layers?.Layers ?? [], item], Canvas = layers?.Canvas };
    }

    public static string[] BlendModes() => CompositorBlend.ModeNames;
}

[Node(Category = "image", DisplayName = "Layers From Bounding Boxes",
    Description = "Turn an image batch plus its bounding boxes into a layer stack, one layer per frame, each placed by its own box. Use this when a node emits layers as a batch - a batch carries a single placement for every frame, so the individual positions are otherwise lost.")]
public sealed class LayersFromBoundingBoxes
{
    /// <summary>_int: an int or float (not a bool) as a whole number (truncated), else the default.</summary>
    static long Int(object? value, long fallback) => PyValues.IsNumber(value) && value is not bool ? (long)PyValues.ToInt(value) : fallback;

    /// <summary>_bbox_list: the boxes as pixel boxes; normalized elements need the canvas size.</summary>
    static ArrayValue BBoxList(object? bboxes, long canvasWidth, long canvasHeight)
    {
        if (bboxes is null) return [];
        if (bboxes is string s)
        {
            var text = PyText.Strip(s);
            if (text.Length == 0) return [];
            try { bboxes = PyJson.Loads(text); }
            catch (PyException e) { throw new PyException("ValueError", $"bboxes string input is not valid JSON: {e.PyMessage}"); }
        }
        IReadOnlyList<object?> probe = bboxes as ArrayValue ?? [bboxes];
        if (probe.Count > 0 && probe[0] is ArrayValue first) probe = first;
        bool hasElements = probe.Any(box => box is DictValue d && PyValues.Get(d, "bbox") is ArrayValue);
        if (hasElements && (canvasWidth <= 0 || canvasHeight <= 0))
            throw new PyException("ValueError", "normalized element boxes need canvas_width and canvas_height to resolve to pixels");
        return BoundingBoxes.FromInput(bboxes, canvasWidth, canvasHeight);
    }

    public LayerDocument Execute(
        [Input(Tooltip = "Image batch; each frame becomes one layer.")] ImageBatch image,
        [Input(Type = "BOUNDING_BOX,ARRAY,STRING", Tooltip = "Placement boxes, index-aligned with the image batch. Accepts bounding boxes (x, y, width, height), normalized elements (with a 'bbox' - these need canvas_width/canvas_height to resolve to pixels), or a JSON string of either. Frames without a matching box are placed at the origin. A box's width/height scales the layer to fit it. metadata.name (or desc) and metadata.z_index are used when present, and metadata.content_rect (frame-relative) crops the frame to its real content.")]
        object bboxes,
        [Input(Tooltip = "Per-frame transparency, index-aligned with the image batch (1 = transparent, LoadImage convention).")] MaskBatch? mask = null,
        [Input(Tooltip = "Layer stack to append to. Leave unconnected to start a new stack.")] LayerDocument? layers = null,
        [Input(Optional = true, Tooltip = "Crop each frame to metadata.content_rect where present and place the content at the box position plus the rect offset. Leave on for batches whose frames are padded - it keeps only the real content at its true spot.")] bool cropToContent = true,
        [Input(Optional = true, Min = 0, Max = 16384, Tooltip = "Document canvas width. 0 derives it from the placed layers.")] long canvasWidth = 0,
        [Input(Optional = true, Min = 0, Max = 16384, Tooltip = "Document canvas height. 0 derives it from the placed layers.")] long canvasHeight = 0)
    {
        var boxes = BBoxList(bboxes, canvasWidth, canvasHeight);
        var items = new List<LayerItem>(layers?.Layers ?? []);
        long baseZ = (items.Count > 0 ? items.Max(i => i.ZIndex) : -1) + 1;
        int W = image.Width, H = image.Height, c = image.Channels, plane = H * W * c;
        for (int index = 0; index < image.Batch; index++)
        {
            var box = index < boxes.Count ? boxes[index] as DictValue ?? [] : [];
            var meta = PyValues.Get(box, "metadata") as DictValue ?? [];
            var frame = new ImageBatch(1, H, W, c, image.Data[(index * plane)..((index + 1) * plane)]);
            var frameMask = Compositor.MaskFrame(mask, index);
            long x = Int(PyValues.Get(box, "x"), 0), y = Int(PyValues.Get(box, "y"), 0);
            long boxW = Int(PyValues.Get(box, "width"), 0), boxH = Int(PyValues.Get(box, "height"), 0);
            bool cropped = false;
            if (cropToContent && PyValues.Get(meta, "content_rect") is ArrayValue { Count: 4 } rect)
            {
                long left = Math.Min(Math.Max(Int(rect[0], 0), 0), W), top = Math.Min(Math.Max(Int(rect[1], 0), 0), H);
                long cw = Math.Min(Math.Max(Int(rect[2], 0), 0), W - left), ch = Math.Min(Math.Max(Int(rect[3], 0), 0), H - top);
                if (cw > 0 && ch > 0)
                {
                    frame = Crop(frame, (int)left, (int)top, (int)cw, (int)ch);
                    if (frameMask is not null) frameMask = CropMask(frameMask, left, top, cw, ch);
                    x += left;
                    y += top;
                    cropped = true;
                }
            }
            var name = PyValues.Get(meta, "name") is string { Length: > 0 } n ? n : PyValues.Get(meta, "desc") as string;
            items.Add(new LayerItem(frame)
            {
                X = x, Y = y, ZIndex = Int(PyValues.Get(meta, "z_index"), baseZ + index),
                W = !cropped && boxW > 0 ? boxW : null,
                H = !cropped && boxH > 0 ? boxH : null,
                Mask = frameMask,
                Name = name is { Length: > 0 } ? name : null,
            });
        }
        (int, int)? canvas = canvasWidth > 0 && canvasHeight > 0 ? ((int)canvasWidth, (int)canvasHeight) : layers?.Canvas;
        return new LayerDocument { Layers = items, Canvas = canvas };
    }

    static ImageBatch Crop(ImageBatch frame, int left, int top, int cw, int ch)
    {
        int c = frame.Channels;
        var o = new ImageBatch(1, ch, cw, c);
        for (int r = 0; r < ch; r++) Array.Copy(frame.Data, ((top + r) * frame.Width + left) * c, o.Data, r * cw * c, cw * c);
        return o;
    }

    /// <summary><c>mask[:, top:top + ch, left:left + cw]</c> with Python's slice rules (the mask may be smaller than the picture).</summary>
    static MaskBatch CropMask(MaskBatch m, long left, long top, long cw, long ch)
    {
        var (y0, _, _, rows) = PyText.SliceIndices(m.Height, top, top + ch, 1);
        var (x0, _, _, cols) = PyText.SliceIndices(m.Width, left, left + cw, 1);
        var o = new MaskBatch(1, (int)rows, (int)cols);
        for (int r = 0; r < rows; r++) Array.Copy(m.Data, (y0 + r) * m.Width + x0, o.Data, r * cols, cols);
        return o;
    }
}
