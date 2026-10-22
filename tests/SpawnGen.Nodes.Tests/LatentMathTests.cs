using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

public class LatentMathTests
{
    static LatentBatch L(params float[] values) => new(1, values.Length, 1, 1, values);   // one position, one channel per value

    [Test]
    public void Add_subtract_and_multiply_work_per_value()
    {
        Assert.That(new LatentAdd().Execute(L(1, 2, 3, 4), L(10, 20, 30, 40)).Data, Is.EqualTo(new[] { 11f, 22f, 33f, 44f }));
        Assert.That(new LatentSubtract().Execute(L(10, 20, 30, 40), L(1, 2, 3, 4)).Data, Is.EqualTo(new[] { 9f, 18f, 27f, 36f }));
        Assert.That(new LatentMultiply().Execute(L(1, -2, 3, 4), -0.5f).Data, Is.EqualTo(new[] { -0.5f, 1f, -1.5f, -2f }));
    }

    [Test]
    public void The_second_latent_is_read_around_when_it_has_fewer_pictures()
    {
        var two = new LatentBatch(2, 4, 1, 1, [1, 1, 1, 1, 2, 2, 2, 2]);
        var one = L(10, 10, 10, 10);
        Assert.That(new LatentAdd().Execute(two, one).Data, Is.EqualTo(new[] { 11f, 11f, 11f, 11f, 12f, 12f, 12f, 12f }));
    }

    [Test]
    public void Interpolate_keeps_the_strength_and_hits_the_ends()
    {
        var a = L(3, 4, 0, 0);   // length 5
        var b = L(0, 0, 0, 10);  // length 10
        Assert.That(new LatentInterpolate().Execute(a, b, 1f).Data, Is.EqualTo(a.Data).Within(1e-5), "ratio 1 is the first");
        Assert.That(new LatentInterpolate().Execute(a, b, 0f).Data, Is.EqualTo(b.Data).Within(1e-5), "ratio 0 is the second");
        var mid = new LatentInterpolate().Execute(a, b, 0.5f).Data;
        double length = Math.Sqrt(mid.Sum(v => v * v));
        Assert.That(length, Is.EqualTo(7.5).Within(1e-4), "the length is the mean of the two lengths (5 and 10), not shrunk like an average of opposite-ish vectors");
        var plain = new[] { 1.5f, 2f, 0f, 5f };
        Assert.That(Math.Sqrt(plain.Sum(v => v * v)), Is.LessThan(length), "a plain average is shorter");
        Assert.That(mid[1] / mid[0], Is.EqualTo(4 / 3.0).Within(1e-4), "the first two channels keep their direction");
    }

    [Test]
    public void A_latent_of_another_size_is_resized_to_the_first()
    {
        var big = new LatentBatch(1, 4, 4, 4);
        var small = new LatentBatch(1, 4, 2, 2);
        for (int i = 0; i < small.Data.Length; i++) small.Data[i] = 1;
        var sum = new LatentAdd().Execute(big, small);
        Assert.That((sum.Width, sum.Height), Is.EqualTo((4, 4)));
        Assert.That(sum.Data.Average(), Is.EqualTo(1).Within(0.05), "a constant stays a constant when resized");
    }
}
