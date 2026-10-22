using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// The JPEG decoder against libjpeg-turbo (via Pillow): Fixtures/jpeg holds tiny synthetic files (a gradient with a disc, 70x50) and the PNG Pillow decodes each to.
/// Regenerate them with a Pillow script if a case is added; they contain no photographs.
/// </summary>
public class JpegTests
{
    static string Fixture(string name) => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "jpeg", name);

    static (double Mean, double Max) Compare(ImageBatch a, ImageBatch b)
    {
        Assert.That((a.Width, a.Height, a.Channels), Is.EqualTo((b.Width, b.Height, b.Channels)));
        double sum = 0, max = 0;
        for (int i = 0; i < a.Data.Length; i++) { double d = Math.Abs(a.Data[i] - b.Data[i]) * 255; sum += d; max = Math.Max(max, d); }
        return (sum / a.Data.Length, max);
    }

    [TestCase("baseline444", 0.2, 3.0)]
    [TestCase("baseline420", 0.6, 4.0)]
    [TestCase("baseline422", 0.6, 4.0)]
    [TestCase("progressive444", 0.2, 3.0)]
    [TestCase("progressive420", 0.6, 4.0)]
    [TestCase("grey", 0.2, 3.0)]
    [TestCase("orientation6", 0.2, 3.0)]
    public void Decoding_matches_libjpeg(string name, double meanLimit, double maxLimit)
    {
        var ours = ImageDecoder.Decode(File.ReadAllBytes(Fixture(name + ".jpg")), out var alpha);
        var expected = PngReader.Decode(File.ReadAllBytes(Fixture(name + ".ref.png")));
        Assert.That(alpha, Is.Null);
        var (mean, max) = Compare(ours, expected);
        TestContext.Out.WriteLine($"{name}: mean abs diff {mean:F3}/255, max {max:F0}");
        Assert.That(mean, Is.LessThan(meanLimit));
        Assert.That(max, Is.LessThan(maxLimit));
    }

    [Test]
    public void A_progressive_file_decodes_like_the_baseline_file_of_the_same_picture()
    {
        var b = JpegReader.Decode(File.ReadAllBytes(Fixture("baseline444.jpg")));
        var p = JpegReader.Decode(File.ReadAllBytes(Fixture("progressive444.jpg")));
        Assert.That(Compare(b, p).Max, Is.LessThan(1.01), "same quantised coefficients, so the same pixels");
    }

    [Test]
    public void Orientation_6_turns_the_picture_so_it_is_taller_than_wide()
    {
        var img = JpegReader.Decode(File.ReadAllBytes(Fixture("orientation6.jpg")));
        Assert.That((img.Width, img.Height), Is.EqualTo((50, 70)));
    }

    [Test]
    public void Files_that_are_not_pictures_say_so()
    {
        Assert.That(() => ImageDecoder.Decode([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), Throws.TypeOf<InvalidDataException>());
        Assert.That(() => ImageDecoder.Decode("RIFF\0\0\0\0WEBPVP8 "u8.ToArray()), Throws.TypeOf<NotSupportedException>().With.Message.Contains("WebP"));
        var truncated = File.ReadAllBytes(Fixture("baseline444.jpg"))[..200];
        Assert.That(() => JpegReader.Decode(truncated), Throws.InstanceOf<Exception>());
    }

    [Test]
    public async Task Load_image_reads_a_jpeg_from_the_input_folder()
    {
        var dir = Path.Combine(Path.GetTempPath(), "spawngen-jpg-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        File.Copy(Fixture("baseline444.jpg"), Path.Combine(dir, "photo.jpg"));
        InputFiles.Directory = dir;
        Assert.That(LoadImage.ImageNames(), Does.Contain("photo.jpg"));
        var (image, mask) = await new LoadImage().ExecuteAsync("photo.jpg");
        Assert.That((image.Width, image.Height), Is.EqualTo((70, 50)));
        Assert.That(mask.Data.Sum(), Is.EqualTo(0f), "a JPEG has no transparency");
    }
}
