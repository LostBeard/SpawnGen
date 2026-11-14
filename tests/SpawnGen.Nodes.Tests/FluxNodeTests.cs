using System.Text.Json;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// The model-independent FLUX nodes against ComfyUI (bef11fc, measured 2026-11-13 with the probe nodes; Fixtures/flux): Flux2Scheduler's sigmas,
/// FluxKontextImageScale's sizes and pixels, Empty Flux 2 Latent, Trim Video Latent, and what Flux Guidance, Set Reference Latent and Edit Model
/// Reference Method put in a conditioning.
/// </summary>
public class FluxNodeTests
{
    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "flux");

    static float[] Raw(string name)
    {
        var b = File.ReadAllBytes(Path.Combine(Dir, name));
        var d = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, d, 0, b.Length);
        return d;
    }

    [Test]
    public void Flux2Scheduler_gives_ComfyUIs_sigmas_for_every_measured_case()
    {
        var cases = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "flux2_scheduler_sigmas.json"))).RootElement.GetProperty("cases");
        int n = 0;
        foreach (var c in cases.EnumerateArray())
        {
            int steps = c.GetProperty("steps").GetInt32(), w = c.GetProperty("width").GetInt32(), h = c.GetProperty("height").GetInt32();
            var expected = c.GetProperty("sigmas").EnumerateArray().Select(v => (float)v.GetDouble()).ToArray();
            var ours = new Flux2Scheduler().Execute(steps, w, h).Values;
            Assert.That(ours.Length, Is.EqualTo(expected.Length), $"{steps} steps {w}x{h}");
            for (int i = 0; i < ours.Length; i++)
                Assert.That(ours[i], Is.EqualTo(expected[i]).Within(2e-7), $"{steps} steps {w}x{h} sigma {i}");
            n++;
        }
        Assert.That(n, Is.EqualTo(33));
    }

    [Test]
    public void Flux2Schedulers_mu_has_its_two_regimes()
    {
        // Above 4300 tokens the step count does not matter (2048 x 2048: mu 3.2300 at 4, 20 and 50 steps); below, it does.
        Assert.That(Flux2Scheduler.Mu(16384, 4), Is.EqualTo(Flux2Scheduler.Mu(16384, 50)));
        Assert.That(Flux2Scheduler.Mu(16384, 4), Is.EqualTo(3.22999).Within(1e-4));
        Assert.That(Flux2Scheduler.Mu(4096, 4), Is.EqualTo(2.29118).Within(1e-4));
        Assert.That(Flux2Scheduler.Mu(4096, 50), Is.EqualTo(2.02335).Within(1e-4));
        Assert.That(Flux2Scheduler.Tokens(100, 36), Is.EqualTo(14), "14.0625 rounds to 14");
        Assert.That((Flux2Scheduler.Tokens(30, 30), Flux2Scheduler.Tokens(50, 50)), Is.EqualTo((4.0, 10.0)), "3.52 and 9.77 round up");
        Assert.That((Flux2Scheduler.Tokens(16, 24), Flux2Scheduler.Tokens(16, 40), Flux2Scheduler.Tokens(16, 56), Flux2Scheduler.Tokens(16, 72)),
            Is.EqualTo((2.0, 2.0, 4.0, 4.0)), "halves to even, as measured");
    }

    [Test]
    public void FluxKontextImageScale_picks_ComfyUIs_size_for_every_measured_aspect()
    {
        var sizes = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "kontext_sizes.json"))).RootElement.GetProperty("sizes");
        int n = 0;
        foreach (var s in sizes.EnumerateObject())
        {
            var parts = s.Name.Split('x').Select(int.Parse).ToArray();
            var expected = (s.Value[0].GetInt32(), s.Value[1].GetInt32());
            Assert.That(FluxKontextImageScale.Target(parts[0], parts[1]), Is.EqualTo(expected), s.Name);
            n++;
        }
        Assert.That(n, Is.EqualTo(251));
    }

    [Test]
    public void FluxKontextImageScale_pixels_are_ComfyUIs()
    {
        // 64 x 16 (aspect 4) to 1568 x 672: only Lanczos after a centre crop matches (without the crop the rows differ by up to 1.0).
        var input = new ImageBatch(1, 16, 64, 3, Raw("kontext_in_1x16x64x3.raw"));
        var ours = new FluxKontextImageScale().Execute(input);
        Assert.That((ours.Width, ours.Height), Is.EqualTo((1568, 672)));
        var rows = Raw("kontext_out_rows_0_1_335_671_of_672x1568x3.raw");
        int stride = 1568 * 3, k = 0;
        foreach (var row in new[] { 0, 1, 335, 671 })
        {
            for (int i = 0; i < stride; i++) Assert.That(ours.Data[row * stride + i], Is.EqualTo(rows[k * stride + i]), $"row {row} value {i}");
            k++;
        }
    }

    [TestCase(1024, 1024, 1, new[] { 1, 128, 64, 64 })]
    [TestCase(100, 36, 2, new[] { 2, 128, 2, 6 })]
    [TestCase(272, 160, 1, new[] { 1, 128, 10, 17 })]
    [TestCase(17, 33, 1, new[] { 1, 128, 2, 1 })]
    public void EmptyFlux2LatentImage_is_ComfyUIs_zeros(int w, int h, int batch, int[] shape)
    {
        var latent = new EmptyFlux2LatentImage().Execute(w, h, batch);
        Assert.That(latent.Shape, Is.EqualTo(shape));
        Assert.That(latent.Data, Is.All.EqualTo(0f));
    }

    static LatentBatch Frames()
    {
        // Measured input: frame f, channel c holds f * 10 + c / 100 everywhere.
        var latent = new LatentBatch(1, 16, 5, 4, 6);
        for (int c = 0; c < 16; c++)
            for (int f = 0; f < 5; f++)
                for (int i = 0; i < 24; i++) latent.Data[(c * 5 + f) * 24 + i] = f * 10 + c / 100f;
        return latent;
    }

    [TestCase(0, new[] { 0f, 10f, 20f, 30f, 40f })]
    [TestCase(1, new[] { 10f, 20f, 30f, 40f })]
    [TestCase(3, new[] { 30f, 40f })]
    [TestCase(4, new[] { 40f })]
    [TestCase(5, new float[0])]
    [TestCase(9, new float[0])]
    public void TrimVideoLatent_drops_the_first_frames_as_ComfyUI(int trim, float[] frames)
    {
        var mask = new MaskBatch(1, 32, 48);
        var source = new LatentBatch(1, 16, 5, 4, 6, Frames().Data) { NoiseMask = mask };
        var trimmed = new TrimVideoLatent().Execute(source, trim);
        Assert.That(trimmed.Shape, Is.EqualTo(new[] { 1, 16, frames.Length, 4, 6 }));
        for (int f = 0; f < frames.Length; f++)
        {
            Assert.That(trimmed.Data[f * 24], Is.EqualTo(frames[f]), $"channel 0 frame {f}");
            Assert.That(trimmed.Data[(15 * frames.Length + f) * 24 + 23], Is.EqualTo(frames[f] + 0.15f).Within(1e-5), $"channel 15 frame {f}");
        }
        Assert.That(trimmed.NoiseMask, Is.SameAs(mask), "measured: the noise mask is passed on untouched");
    }

    [Test]
    public void TrimVideoLatent_trims_a_picture_latents_rows_as_ComfyUI()
    {
        // Measured: [1, 4, 5, 6] trimmed by 2 is [1, 4, 3, 6], the first two rows of every channel gone.
        var latent = new LatentBatch(1, 4, 5, 6, [.. Enumerable.Range(0, 120).Select(i => (float)i)]);
        var trimmed = new TrimVideoLatent().Execute(latent, 2);
        Assert.That(trimmed.Shape, Is.EqualTo(new[] { 1, 4, 3, 6 }));
        Assert.That(trimmed.Data[0], Is.EqualTo(12f));
        Assert.That(trimmed.Data[18], Is.EqualTo(42f), "channel 1 starts at its row 2");
    }

    [Test]
    public void Guidance_and_reference_latents_go_into_every_entry_as_ComfyUI_puts_them()
    {
        var plain = new Conditioning([new CondEntry(new float[4], [1, 2, 2]), new CondEntry(new float[4], [1, 2, 2])]);
        var guided = new FluxGuidance().Execute(plain, 4.25);
        Assert.That(guided.Entries.Select(e => e.Guidance), Is.All.EqualTo(4.25));

        var a = new LatentBatch(1, 128, 2, 4);
        var b = new LatentBatch(1, 16, 4, 4);
        var once = new ReferenceLatent().Execute(guided, a);
        var twice = new ReferenceLatent().Execute(once, b);
        Assert.That(twice.Entries.Select(e => e.ReferenceLatents!.Count), Is.All.EqualTo(2), "chained: both, in order");
        Assert.That(twice.Entries[0].ReferenceLatents![0], Is.SameAs(a));
        Assert.That(twice.Entries[1].ReferenceLatents![1], Is.SameAs(b));
        Assert.That(twice.Entries[0].Guidance, Is.EqualTo(4.25), "the guidance stays");
        Assert.That(once.Entries[0].ReferenceLatents!.Count, Is.EqualTo(1), "the earlier conditioning is not changed");
        Assert.That(new ReferenceLatent().Execute(plain), Is.SameAs(plain), "no latent: passed on");

        Assert.That(new FluxKontextMultiReferenceLatentMethod().Execute(twice, "uxo/uno").Entries.Select(e => e.ReferenceLatentsMethod), Is.All.EqualTo("uxo"));
        Assert.That(new FluxKontextMultiReferenceLatentMethod().Execute(twice, "index_timestep_zero").Entries[0].ReferenceLatentsMethod, Is.EqualTo("index_timestep_zero"));
    }
}
