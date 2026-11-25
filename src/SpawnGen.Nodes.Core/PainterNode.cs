using System.Globalization;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// Painter: a picture painted in the node's own widget, over a background colour or over a linked picture. IMAGE is the painting drawn over
/// the base with its alpha; MASK is the painting's alpha.
/// </summary>
/// <remarks>
/// Rebuilt from black-box measurements of ComfyUI (2026-11-25: tools/studio-browser-test/probe-comfy-painter*.js, painter-resize-check.py;
/// its code was not read): the widget uploads the painting as an RGBA PNG of the canvas size (straight alpha, transparent where nothing is
/// painted) when the prompt is built and puts its name in <c>mask</c> ("painter-&lt;node&gt;-&lt;ms&gt;.png [input]"). Without a linked image the
/// output is <c>width</c> x <c>height</c>; with one it has the image's size (width and height are then ignored) and only the batch's first
/// picture is used. A painting of another size than the output is resized with Pillow's Lanczos on its RGBA bytes (exact on the measured
/// files; bilinear, bicubic, box, nearest and torch's interpolate modes all differ). IMAGE = painting x alpha + base x (1 - alpha);
/// MASK = alpha. With an empty <c>mask</c> the IMAGE is the base and the MASK is zero. Not measured: an image with an alpha channel as the
/// base (its colour channels are used here), and a colour that is not "#rrggbb" (black here).
/// </remarks>
[Node(Category = "image", DisplayName = "Painter", OutputNames = ["IMAGE", "MASK"])]
public sealed class Painter
{
    public async Task<(ImageBatch Image, MaskBatch Mask)> ExecuteAsync(
        [Input(WidgetType = "PAINTER", Default = "", Socketless = true)] string mask,
        [Input(Hidden = true, Socketless = true, Default = 512L, Min = 64, Max = 4096, Step = 64)] long width,
        [Input(Hidden = true, Socketless = true, Default = 512L, Min = 64, Max = 4096, Step = 64)] long height,
        [Input(Name = "bg_color", Color = true, Default = "#000000")] string bgColor,
        [Input(Optional = true, Tooltip = "Optional base image to paint over")] ImageBatch? image = null)
    {
        int w = image is not null ? image.Width : (int)width, h = image is not null ? image.Height : (int)height;
        byte[]? painting = null;
        if (!string.IsNullOrWhiteSpace(mask))
        {
            var (name, type) = Pictures.Annotated(mask);
            if (type is not (null or "input")) throw new PyException("ValueError", $"Painter reads its painting from the input folder, not from '{type}'");
            var (picture, alpha) = await ImageDecoder.DecodeFirstAsync(await InputFiles.Store.ReadAsync(name));
            painting = Rgba(picture, alpha);
            if (picture.Width != w || picture.Height != h) painting = PilLanczos.Resize(painting, picture.Width, picture.Height, 4, w, h);
        }
        return Compose(painting, w, h, image, Colour(bgColor));
    }

    /// <summary>The painting (RGBA8, <paramref name="w"/> x <paramref name="h"/>, or null for none) over the base: the picture's first frame, or the colour.</summary>
    public static (ImageBatch Image, MaskBatch Mask) Compose(byte[]? painting, int w, int h, ImageBatch? image, (float R, float G, float B) colour)
    {
        var result = new ImageBatch(1, h, w, 3);
        var mask = new MaskBatch(1, h, w);
        int c = image?.Channels ?? 0;
        Span<float> bg = [colour.R, colour.G, colour.B];
        for (int i = 0; i < w * h; i++)
        {
            float a = painting is null ? 0f : painting[i * 4 + 3] / 255f;
            for (int k = 0; k < 3; k++)
            {
                float under = image is not null ? image.Data[i * c + (c >= 3 ? k : 0)] : bg[k];
                float over = painting is null ? 0f : painting[i * 4 + k] / 255f;
                result.Data[i * 3 + k] = over * a + under * (1f - a);
            }
            mask.Data[i] = a;
        }
        return (result, mask);
    }

    static byte[] Rgba(ImageBatch picture, float[]? alpha)
    {
        int n = picture.Width * picture.Height, c = picture.Channels;
        var px = new byte[n * 4];
        static byte Q(float v) => (byte)Math.Clamp(MathF.Round(v * 255f, MidpointRounding.ToEven), 0, 255);
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < 3; k++) px[i * 4 + k] = Q(picture.Data[i * c + (c >= 3 ? k : 0)]);
            px[i * 4 + 3] = alpha is not null ? Q(alpha[i]) : c == 4 ? Q(picture.Data[i * c + 3]) : (byte)255;
        }
        return px;
    }

    /// <summary>"#rrggbb" as 0..1 values; anything else is black.</summary>
    public static (float R, float G, float B) Colour(string? text)
    {
        if (text is { Length: 7 } && text[0] == '#' && int.TryParse(text.AsSpan(1), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out int rgb))
            return (((rgb >> 16) & 255) / 255f, ((rgb >> 8) & 255) / 255f, (rgb & 255) / 255f);
        return (0, 0, 0);
    }
}
