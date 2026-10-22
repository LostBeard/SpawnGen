using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

public class ResampleTests
{
    static readonly ResampleMethod[] All = Enum.GetValues<ResampleMethod>();

    static float[] Ramp(int w, int h, int c = 1)
    {
        var d = new float[w * h * c];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                for (int k = 0; k < c; k++) d[(y * w + x) * c + k] = (x + y * 0.5f) * (k + 1);
        return d;
    }

    [TestCaseSource(nameof(All))]
    public void A_picture_resized_to_its_own_size_is_unchanged(ResampleMethod m)
    {
        var src = Ramp(7, 5, 3);
        Assert.That(Resample.Resize(src, 7, 5, 3, 7, 5, m), Is.EqualTo(src));
    }

    [Test]
    public void Constant_pictures_stay_constant_for_every_method_up_and_down()
    {
        foreach (var m in All)
            foreach (var (nw, nh) in new[] { (16, 12), (3, 2), (5, 9), (1, 1) })
            {
                var flat = Enumerable.Repeat(0.37f, 8 * 6 * 3).ToArray();
                var r = Resample.Resize(flat, 8, 6, 3, nw, nh, m);
                Assert.That(r, Has.Length.EqualTo(nw * nh * 3));
                Assert.That(r.All(v => Math.Abs(v - 0.37f) < 1e-5f), Is.True, $"{m} {nw}x{nh}");
            }
    }

    [Test]
    public void Nearest_exact_doubles_pixels_going_up_and_takes_the_nearer_pixel_going_down()
    {
        float[] src = [1, 2, 3, 4];                  // 2x2, one channel
        var up = Resample.Resize(src, 2, 2, 1, 4, 4, ResampleMethod.NearestExact);
        Assert.That(up, Is.EqualTo(new float[] { 1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4 }));
        var down = Resample.Resize(Ramp(4, 4), 4, 4, 1, 2, 2, ResampleMethod.NearestExact);
        Assert.That(down, Is.EqualTo(new[] { Ramp(4, 4)[1 * 4 + 1], Ramp(4, 4)[1 * 4 + 3], Ramp(4, 4)[3 * 4 + 1], Ramp(4, 4)[3 * 4 + 3] }));
    }

    [Test]
    public void Area_downscale_averages_the_blocks_it_covers()
    {
        float[] src = [1, 3, 5, 7,
                       1, 3, 5, 7,
                       2, 2, 4, 4,
                       2, 2, 4, 4];
        var r = Resample.Resize(src, 4, 4, 1, 2, 2, ResampleMethod.Area);
        Assert.That(r, Is.EqualTo(new float[] { 2, 6, 2, 4 }).Within(1e-5f));
        // A non-integer ratio still averages everything it covers: the mean of the whole picture is kept.
        var big = Ramp(9, 9);
        var small = Resample.Resize(big, 9, 9, 1, 4, 4, ResampleMethod.Area);
        Assert.That(small.Average(), Is.EqualTo(big.Average()).Within(0.05));
    }

    [Test]
    public void Bilinear_upscale_matches_PyTorch_with_align_corners_false()
    {
        // 1x2 ramp [0, 1] to 1x4: [0, 0.25, 0.75, 1] (edges clamp).
        var r = Resample.Resize([0f, 1f], 2, 1, 1, 4, 1, ResampleMethod.Bilinear);
        Assert.That(r, Is.EqualTo(new[] { 0f, 0.25f, 0.75f, 1f }).Within(1e-6f));
    }

    [TestCase(ResampleMethod.Bilinear)]
    [TestCase(ResampleMethod.Bicubic)]
    [TestCase(ResampleMethod.Lanczos)]
    public void A_linear_ramp_stays_a_ramp_in_the_interior_when_upscaled(ResampleMethod m)
    {
        var src = Enumerable.Range(0, 16).Select(i => i / 15f).ToArray();      // 16x1 ramp 0..1
        var up = Resample.Resize(src, 16, 1, 1, 48, 1, m);
        for (int i = 12; i < 36; i++)
        {
            double expected = ((i + 0.5) / 3.0 - 0.5) / 15.0;                  // the ramp sampled at the output pixel's centre
            Assert.That(up[i], Is.EqualTo(expected).Within(0.01), $"{m} sample {i}");
        }
    }

    [Test]
    public void Channels_are_resized_independently_and_layout_is_kept()
    {
        var src = Ramp(6, 4, 3);
        var r = Resample.Resize(src, 6, 4, 3, 12, 8, ResampleMethod.Bilinear);
        // Channel k of the source ramp is (k+1) times channel 0, and bilinear is linear: the same holds after resizing.
        for (int i = 0; i < 12 * 8; i++)
        {
            Assert.That(r[i * 3 + 1], Is.EqualTo(r[i * 3] * 2).Within(1e-4f));
            Assert.That(r[i * 3 + 2], Is.EqualTo(r[i * 3] * 3).Within(1e-4f));
        }
    }

    [Test]
    public void Lanczos_downscale_does_not_alias_a_checkerboard()
    {
        // A 1-pixel checkerboard averages to mid grey when shrunk; aliasing would leave stripes.
        var src = new float[32 * 32];
        for (int y = 0; y < 32; y++) for (int x = 0; x < 32; x++) src[y * 32 + x] = (x + y) % 2;
        var r = Resample.Resize(src, 32, 32, 1, 8, 8, ResampleMethod.Lanczos);
        Assert.That(r.All(v => Math.Abs(v - 0.5f) < 0.05f), Is.True);
    }

    [Test]
    public void Method_names_parse_and_unknown_ones_say_what_is_known()
    {
        foreach (var name in Resample.MethodNames) Assert.DoesNotThrow(() => Resample.Parse(name));
        var ex = Assert.Throws<ArgumentException>(() => Resample.Parse("sinc"));
        Assert.That(ex!.Message, Does.Contain("lanczos"));
    }

    [Test]
    public void Bad_sizes_are_refused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Resample.Resize([1f], 1, 1, 1, 0, 4, ResampleMethod.Bilinear));
    }
}
