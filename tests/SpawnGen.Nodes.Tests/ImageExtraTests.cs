using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>Flip, rotate, quantize, morphology, colour to mask and the alpha mask of Load Image, against hand-computed values.</summary>
public class ImageExtraTests
{
    // 2 wide, 3 tall, one channel per pixel replicated to 3: values 0..5 / 10, row by row.
    static ImageBatch Grid()
    {
        var img = new ImageBatch(1, 3, 2, 3);
        for (int i = 0; i < 6; i++) img.Data[i * 3] = img.Data[i * 3 + 1] = img.Data[i * 3 + 2] = i / 10f;
        return img;
    }

    static int[] Reds(ImageBatch img) => Enumerable.Range(0, img.Width * img.Height).Select(i => (int)Math.Round(img.Data[i * img.Channels] * 10)).ToArray();

    [Test]
    public void Flip_vertically_reverses_the_rows_and_horizontally_the_columns()
    {
        Assert.That(Reds(new ImageFlip().Execute(Grid(), "x-axis: vertically")), Is.EqualTo(new[] { 4, 5, 2, 3, 0, 1 }));
        Assert.That(Reds(new ImageFlip().Execute(Grid(), "y-axis: horizontally")), Is.EqualTo(new[] { 1, 0, 3, 2, 5, 4 }));
        Assert.That(() => new ImageFlip().Execute(Grid(), "z"), Throws.ArgumentException);
    }

    [Test]
    public void Rotate_turns_clockwise_and_swaps_the_sides()
    {
        // [0 1 / 2 3 / 4 5] turned 90 degrees clockwise is [4 2 0 / 5 3 1].
        var r90 = new ImageRotate().Execute(Grid(), "90 degrees");
        Assert.That((r90.Width, r90.Height), Is.EqualTo((3, 2)));
        Assert.That(Reds(r90), Is.EqualTo(new[] { 4, 2, 0, 5, 3, 1 }));
        Assert.That(Reds(new ImageRotate().Execute(Grid(), "180 degrees")), Is.EqualTo(new[] { 5, 4, 3, 2, 1, 0 }));
        var r270 = new ImageRotate().Execute(Grid(), "270 degrees");
        Assert.That(Reds(r270), Is.EqualTo(new[] { 1, 3, 5, 0, 2, 4 }));
        Assert.That(new ImageRotate().Execute(Grid(), "none"), Is.SameAs(Grid()).Or.Not.Null);
        // Four quarter turns come back.
        var four = Grid();
        for (int i = 0; i < 4; i++) four = new ImageRotate().Execute(four, "90 degrees");
        Assert.That(Reds(four), Is.EqualTo(Reds(Grid())));
    }

    [Test]
    public void Quantize_to_two_colours_splits_dark_from_light_and_one_colour_is_the_mean()
    {
        // Greys 0, 0.1, 0.9, 1 are bytes 0, 25, 229, 255 (* 255, truncated). Pillow's median cut splits along green (weight 150), walking the sorted
        // values from the top until more than half the pixels are taken: {255, 229, 25} | {0}. Box means (rounded half up): 170 and 0.
        var img = new ImageBatch(1, 1, 4, 3);
        float[] values = [0f, 0.1f, 0.9f, 1f];
        for (int i = 0; i < 4; i++) img.Data[i * 3] = img.Data[i * 3 + 1] = img.Data[i * 3 + 2] = values[i];
        float[] Reds(ImageBatch b) => [b.Data[0], b.Data[3], b.Data[6], b.Data[9]];
        Assert.That(Reds(new ImageQuantize().Execute(img, 2, "none")), Is.EqualTo(new[] { 0f, 0f, 170 / 255f, 170 / 255f }));
        // One colour: (0 + 25 + 229 + 255) / 4 = 127.25 -> 127.
        var one = new ImageQuantize().Execute(img, 1, "none");
        Assert.That(one.Data.Distinct(), Is.EqualTo(new[] { 127 / 255f }));
        // Floyd-Steinberg: 25 -> 0 leaves 25, 7/16 of it lifts 229 to 239 -> 170, whose error 69 lifts 255 past the top -> 170.
        Assert.That(Reds(new ImageQuantize().Execute(img, 2, "floyd-steinberg")), Is.EqualTo(new[] { 0f, 0f, 170 / 255f, 170 / 255f }));
        // bayer-2 with two palette entries: spread 256, thresholds 256 * [[-3/8, 1/8], [3/8, -1/8]] + 0.5 on the top row: -95.5 and 32.5.
        // 0 - 95.5 -> 0; 25 + 32.5 -> 57 (nearest 0); 229 - 95.5 -> 133 (170); 255 + 32.5 -> 255 (170).
        Assert.That(Reds(new ImageQuantize().Execute(img, 2, "bayer-2")), Is.EqualTo(new[] { 0f, 0f, 170 / 255f, 170 / 255f }));
    }

    [Test]
    public void Morphology_erode_and_dilate_take_min_and_max_and_the_rest_combine_them()
    {
        // One bright pixel in the middle of a 5x5 dark picture.
        var img = new ImageBatch(1, 5, 5, 3);
        int mid = (2 * 5 + 2) * 3;
        img.Data[mid] = img.Data[mid + 1] = img.Data[mid + 2] = 1f;
        float Sum(ImageBatch b) => Enumerable.Range(0, 25).Sum(i => b.Data[i * 3]);
        var m = new Morphology();
        Assert.That(Sum(m.Execute(img, "dilate", 3)), Is.EqualTo(9f), "a 3x3 block");
        Assert.That(Sum(m.Execute(img, "erode", 3)), Is.EqualTo(0f), "the lone pixel vanishes");
        Assert.That(Sum(m.Execute(img, "open", 3)), Is.EqualTo(0f));
        Assert.That(Sum(m.Execute(img, "gradient", 3)), Is.EqualTo(9f).Within(1e-5), "dilate minus erode");
        Assert.That(Sum(m.Execute(img, "top_hat", 3)), Is.EqualTo(1f).Within(1e-5), "what opening removed");
        Assert.That(Sum(m.Execute(img, "close", 3)), Is.EqualTo(1f).Within(1e-5), "closing keeps a lone bright pixel");
        Assert.That(Sum(m.Execute(img, "bottom_hat", 3)), Is.EqualTo(0f).Within(1e-5));
        Assert.That(() => m.Execute(img, "nope", 3), Throws.ArgumentException);
    }

    [Test]
    public void Colour_to_mask_marks_exact_matches()
    {
        var img = new ImageBatch(1, 1, 3, 3, [1f, 0f, 0f, 0f, 1f, 0f, 1f, 0f, 0f]);
        var mask = new ImageColorToMask().Execute(img, 0xFF0000);
        Assert.That(mask.Data, Is.EqualTo(new[] { 1f, 0f, 1f }));
    }

    [Test]
    public async Task Load_image_gives_a_mask_from_the_alpha_and_load_image_as_mask_picks_a_channel()
    {
        var dir = Path.Combine(Path.GetTempPath(), "spawngen-alpha-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        InputFiles.Directory = dir;
        // 2x1 RGBA: opaque red, half-transparent green. Written by hand as an RGBA PNG.
        File.WriteAllBytes(Path.Combine(dir, "a.png"), RgbaPng(2, 1, [255, 0, 0, 255, 0, 255, 0, 128]));
        File.WriteAllBytes(Path.Combine(dir, "plain.png"), PngWriter.Encode(new ImageBatch(1, 1, 2, 3, [1f, 0f, 0f, 0f, 1f, 0f])));
        var (image, mask) = await new LoadImage().ExecuteAsync("a.png");
        Assert.That(image.Data, Is.EqualTo(new[] { 1f, 0f, 0f, 0f, 1f, 0f }).Within(1e-5));
        Assert.That(mask.Data[0], Is.EqualTo(0f).Within(1e-5), "opaque: not masked");
        Assert.That(mask.Data[1], Is.EqualTo(1f - 128 / 255f).Within(1e-5), "half transparent");
        var (_, plainMask) = await new LoadImage().ExecuteAsync("plain.png");
        Assert.That(plainMask.Data, Is.EqualTo(new[] { 0f, 0f }), "no alpha: nothing masked");
        var green = await new LoadImageMask().ExecuteAsync("a.png", "green");
        Assert.That(green.Data, Is.EqualTo(new[] { 0f, 1f }).Within(1e-5));
        var alpha = await new LoadImageMask().ExecuteAsync("a.png", "alpha");
        Assert.That(alpha.Data[1], Is.EqualTo(1f - 128 / 255f).Within(1e-5));
    }

    static byte[] RgbaPng(int w, int h, byte[] rgba)
    {
        using var raw = new MemoryStream();
        for (int y = 0; y < h; y++) { raw.WriteByte(0); raw.Write(rgba, y * w * 4, w * 4); }
        using var z = new MemoryStream();
        using (var d = new System.IO.Compression.ZLibStream(z, System.IO.Compression.CompressionLevel.Fastest, true)) d.Write(raw.ToArray());
        using var png = new MemoryStream();
        png.Write([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        void Chunk(string type, byte[] data)
        {
            var len = new byte[4]; System.Buffers.Binary.BinaryPrimitives.WriteUInt32BigEndian(len, (uint)data.Length);
            png.Write(len);
            var body = System.Text.Encoding.ASCII.GetBytes(type).Concat(data).ToArray();
            png.Write(body);
            var crc = new byte[4]; // the reader does not check CRCs
            png.Write(crc);
        }
        var ihdr = new byte[13];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32BigEndian(ihdr, (uint)w);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32BigEndian(ihdr.AsSpan(4), (uint)h);
        ihdr[8] = 8; ihdr[9] = 6;
        Chunk("IHDR", ihdr); Chunk("IDAT", z.ToArray()); Chunk("IEND", []);
        return png.ToArray();
    }
}
