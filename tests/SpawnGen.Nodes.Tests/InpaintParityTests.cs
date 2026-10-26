using System.IO.Compression;
using System.Text.Json;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// VAE Encode, VAE Encode (for Inpainting) and InpaintModelConditioning against ComfyUI's own nodes (Fixtures/inpaint/make_inpaint.py -> inpaint.json.gz):
/// ComfyUI ran them with a stand-in VAE that returns what it is given, so the fixture holds the pictures the VAE gets and the masks the nodes return, for
/// pictures that are and are not multiples of 8, masks of other sizes, soft masks, one mask for two pictures, and several growths.
/// </summary>
public class InpaintParityTests
{
    static readonly JsonElement[] Cases = Load();

    static JsonElement[] Load()
    {
        using var gz = new GZipStream(File.OpenRead(Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "inpaint", "inpaint.json.gz")), CompressionMode.Decompress);
        return JsonDocument.Parse(gz).RootElement.EnumerateArray().ToArray();
    }

    public static IEnumerable<int> CaseIndexes() => Enumerable.Range(0, 4);

    static (int[] Shape, float[] Data) Tensor(JsonElement e) =>
        (e.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray(), e.GetProperty("data").EnumerateArray().Select(v => v.GetSingle()).ToArray());

    static ImageBatch Pixels(JsonElement c)
    {
        var (s, d) = Tensor(c.GetProperty("pixels"));
        return new ImageBatch(s[0], s[1], s[2], s[3], d);
    }

    static MaskBatch Mask(JsonElement c)
    {
        var (s, d) = Tensor(c.GetProperty("mask"));
        return new MaskBatch(s[0], s[1], s[2], d);
    }

    /// <summary>Same shape as ComfyUI's (its masks carry a channel axis of 1) and the same values; the rounding of the fixture's 7 decimals and of
    /// float32 interpolation is the only slack.</summary>
    static void Same(HostTensor ours, JsonElement theirs, string what)
    {
        var (shape, data) = Tensor(theirs);
        Assert.That(ours.Shape, Is.EqualTo(shape.Where((n, i) => !(shape.Length == 4 && i == 1 && n == 1 && ours.Shape.Length == 3)).ToArray()), what + " shape");
        Assert.That(ours.Data, Is.EqualTo(data).Within(2e-6f), what);
    }

    [TestCaseSource(nameof(CaseIndexes))]
    public void VAE_Encode_crops_like_ComfyUI(int index)
    {
        var c = Cases[index];
        Same(VAEEncode.CropForVae(Pixels(c)), c.GetProperty("vae_encode"), "picture given to the VAE");
    }

    [TestCaseSource(nameof(CaseIndexes))]
    public void VAE_Encode_for_Inpainting_greys_and_grows_like_ComfyUI(int index)
    {
        var c = Cases[index];
        foreach (var run in c.GetProperty("for_inpaint").EnumerateArray())
        {
            int grow = run.GetProperty("grow").GetInt32();
            var (greyed, noiseMask) = VAEEncodeForInpaint.Prepare(Pixels(c), Mask(c), grow);
            // ComfyUI's VAE crops again; the picture is already a multiple of 8 then.
            Same(VAEEncode.CropForVae(greyed), run.GetProperty("encoded"), $"grow {grow}: picture given to the VAE");
            Same(noiseMask, run.GetProperty("noise_mask"), $"grow {grow}: noise mask");
        }
    }

    [TestCaseSource(nameof(CaseIndexes))]
    public void InpaintModelConditioning_prepares_like_ComfyUI(int index)
    {
        var c = Cases[index];
        var (picture, mask, greyed) = InpaintPixels.Prepare(Pixels(c), Mask(c));
        foreach (var run in c.GetProperty("model_conditioning").EnumerateArray())
        {
            // The latent is the whole picture encoded (the VAE crops it), the concat latent the greyed one; the mask is both the noise mask and the concat mask.
            Same(picture, run.GetProperty("encoded"), "picture encoded for the latent");
            Same(greyed, run.GetProperty("concat"), "picture encoded for the concat latent");
            Same(mask, run.GetProperty("concat_mask"), "concat mask");
            if (run.GetProperty("noise_mask_on").GetBoolean()) Same(mask, run.GetProperty("noise_mask"), "noise mask");
            else Assert.That(run.TryGetProperty("noise_mask", out _), Is.False);
        }
    }
}
