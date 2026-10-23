using System.IO.Compression;
using System.Text.Json;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// The image nodes ported from comfy_extras (nodes_images, nodes_compositing, nodes_morphology, nodes_canny, nodes_color) against values ComfyUI's
/// own code computed: Fixtures/comfy/*.json.gz hold the inputs and the outputs of each node's <c>execute</c> (and of comfy.utils.common_upscale), run
/// unchanged on torch 2.14 (CPU), kornia 0.8.3 and Pillow 13 with the server parts stubbed out.
/// </summary>
public class ComfyImageParityTests
{
    public sealed record Tensor(int[] Shape, float[] Data);

    public sealed record Case(string Node, JsonElement Args, Dictionary<string, Tensor> Inputs, JsonElement[] Out, Tensor[] Tensors)
    {
        public override string ToString() => Node + " " + Args.GetRawText();
    }

    static readonly Dictionary<string, List<Case>> Loaded = [];

    static List<Case> Load(string file)
    {
        lock (Loaded)
        {
            if (Loaded.TryGetValue(file, out var cached)) return cached;
            var path = Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "comfy", $"comfy_image_nodes_{file}.json.gz");
            using var gz = new GZipStream(File.OpenRead(path), CompressionMode.Decompress);
            using var doc = JsonDocument.Parse(gz);
            var tensors = doc.RootElement.GetProperty("tensors").EnumerateArray().Select(ReadTensor).ToArray();
            var cases = doc.RootElement.GetProperty("cases").EnumerateArray().Select(c => new Case(
                c.GetProperty("node").GetString()!,
                c.GetProperty("args").Clone(),
                c.GetProperty("inputs").EnumerateObject().ToDictionary(p => p.Name, p => tensors[p.Value.GetInt32()]),
                c.GetProperty("out").EnumerateArray().Select(o => o.Clone()).ToArray(),
                tensors)).ToList();
            Loaded[file] = cases;
            return cases;
        }
    }

    static Tensor ReadTensor(JsonElement t)
    {
        var shape = t.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        var data = t.TryGetProperty("bytes", out var bytes)
            ? bytes.EnumerateArray().Select(v => v.GetInt32() / 255f).ToArray()
            : t.GetProperty("data").EnumerateArray().Select(v => (float)v.GetDouble()).ToArray();
        return new Tensor(shape, data);
    }

    static IEnumerable<TestCaseData> Cases(string file, string? node = null) =>
        Load(file).Where(c => node is null || c.Node == node).Select((c, i) => new TestCaseData(c).SetName($"{c.Node}_{i:D3}"));

    static IEnumerable<TestCaseData> MainCases() => Cases("main");
    static IEnumerable<TestCaseData> CropCases() => Cases("crop");

    static ImageBatch Img(Tensor t) => new(t.Shape[0], t.Shape[1], t.Shape[2], t.Shape[3], (float[])t.Data.Clone());
    static MaskBatch Mask(Tensor t) => new(t.Shape[0], t.Shape[1], t.Shape[2], (float[])t.Data.Clone());

    static string S(Case c, string name) => c.Args.GetProperty(name).GetString()!;
    static int I(Case c, string name) => c.Args.GetProperty(name).GetInt32();
    static double D(Case c, string name) => c.Args.GetProperty(name).GetDouble();
    static bool B(Case c, string name) => c.Args.GetProperty(name).GetBoolean();

    static void Same(HostTensor actual, JsonElement expected, double tolerance, string what = "")
    {
        var e = ReadTensor(expected);
        Assert.That(actual.Shape, Is.EqualTo(e.Shape), "shape " + what);
        double worst = 0;
        int at = -1;
        for (int i = 0; i < e.Data.Length; i++)
        {
            double d = Math.Abs(actual.Data[i] - e.Data[i]);
            if (!(d <= worst)) { worst = d; at = i; }
        }
        Assert.That(worst, Is.LessThanOrEqualTo(tolerance), $"{what} worst at {at}: {(at >= 0 ? actual.Data[at] : 0)} vs {(at >= 0 ? e.Data[at] : 0)}");
    }

    [TestCaseSource(nameof(MainCases))]
    public void Matches_ComfyUI(Case c)
    {
        var o = c.Out;
        switch (c.Node)
        {
            case "common_upscale":
            {
                var m = S(c, "method");
                // Pillow's lanczos is integer arithmetic: identical; the torch modes to float rounding.
                Same(CommonUpscale.Resize(Img(c.Inputs["image"]), I(c, "width"), I(c, "height"), m, S(c, "crop")), o[0], m == "lanczos" ? 0 : 2e-6);
                break;
            }
            case "ImageScaleToMaxDimension":
                Same(new ImageScaleToMaxDimension().Execute(Img(c.Inputs["image"]), S(c, "upscale_method"), I(c, "largest_size")), o[0], 2e-6);
                break;
            case "ResizeAndPadImage":
                Same(new ResizeAndPadImage().Execute(Img(c.Inputs["image"]), I(c, "target_width"), I(c, "target_height"), S(c, "padding_color"), S(c, "interpolation")), o[0], 2e-6);
                break;
            case "ImageStitch":
                Same(new ImageStitch().Execute(Img(c.Inputs["image1"]), S(c, "direction"), B(c, "match_image_size"), I(c, "spacing_width"), S(c, "spacing_color"),
                    c.Inputs.TryGetValue("image2", out var image2) ? Img(image2) : null), o[0], 0);
                break;
            case "PorterDuffImageComposite":
            {
                var (image, mask) = new PorterDuffImageComposite().Execute(Img(c.Inputs["source"]), Mask(c.Inputs["source_alpha"]), Img(c.Inputs["destination"]),
                    Mask(c.Inputs["destination_alpha"]), S(c, "mode"));
                Same(image, o[0], 2e-5, "image");
                Same(mask, o[1], 2e-6, "mask");
                break;
            }
            case "SplitImageWithAlpha":
            {
                var (image, mask) = new SplitImageWithAlpha().Execute(Img(c.Inputs["image"]));
                Same(image, o[0], 0); Same(mask, o[1], 0);
                break;
            }
            case "JoinImageWithAlpha":
                Same(new JoinImageWithAlpha().Execute(Img(c.Inputs["image"]), Mask(c.Inputs["alpha"])), o[0], 2e-6);
                break;
            case "ImageRGBToYUV":
            {
                var (y, u, v) = new ImageRGBToYUV().Execute(Img(c.Inputs["image"]));
                Same(y, o[0], 1e-6, "Y"); Same(u, o[1], 1e-6, "U"); Same(v, o[2], 1e-6, "V");
                break;
            }
            case "ImageYUVToRGB":
                Same(new ImageYUVToRGB().Execute(Img(c.Inputs["Y"]), Img(c.Inputs["U"]), Img(c.Inputs["V"])), o[0], 1e-6);
                break;
            case "Morphology":
                Same(new Morphology().Execute(Img(c.Inputs["image"]), S(c, "operation"), I(c, "kernel_size")), o[0], 0);
                break;
            case "Canny":
                Same(new Canny().Execute(Img(c.Inputs["image"]), (float)D(c, "low_threshold"), (float)D(c, "high_threshold")), o[0], 0);
                break;
            case "ImageAddNoise":
                Same(new ImageAddNoise().Execute(Img(c.Inputs["image"]), c.Args.GetProperty("seed").GetInt64(), (float)D(c, "strength")), o[0], 2e-6);
                break;
            case "ImageColorSpace":
                Same(new ImageColorSpace().Execute(Img(c.Inputs["image"]), S(c, "source"), S(c, "destination")), o[0], 2e-5);
                break;
            case "ColorToRGBInt":
                if (o[0].ValueKind == JsonValueKind.String && o[0].GetString() == "error")
                    Assert.Throws<ArgumentException>(() => new ColorToRGBInt().Execute(S(c, "color")));
                else
                {
                    var (rgb, hex, alpha) = new ColorToRGBInt().Execute(S(c, "color"));
                    Assert.That(rgb, Is.EqualTo(o[0].GetInt32()));
                    Assert.That(hex, Is.EqualTo(o[1].GetString()));
                    Assert.That(alpha, Is.EqualTo(o[2].GetDouble()));
                }
                break;
            default:
                Assert.Fail("No runner for " + c.Node);
                break;
        }
    }

    [TestCaseSource(nameof(CropCases))]
    public void Crop_to_mask_matches_ComfyUI(Case c)
    {
        var r = new ImageCropToMask().Execute(Img(c.Inputs["images"]), Mask(c.Inputs["masks"]), I(c, "width"), I(c, "height"), D(c, "pad_factor"), I(c, "grow_mask"), S(c, "background"));
        Same(r, c.Out[0], 1e-6);
    }
}
