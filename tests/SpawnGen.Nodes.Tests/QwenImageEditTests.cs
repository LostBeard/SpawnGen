using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// Qwen-Image Edit 2509's TextEncodeQwenImageEditPlus against ComfyUI (bef11fc, CPU float32; Fixtures/qwenedit, the dumps sent to SpawnDev.ILGPU.ML
/// on 2026-11-15): picture A (256 x 192) alone and with picture B (200 x 120), prompt "Turn the picture into a watercolor painting".
/// </summary>
public class QwenImageEditTests
{
    static string Dir => Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "qwenedit");

    static float[] Raw(string name)
    {
        var b = File.ReadAllBytes(Path.Combine(Dir, name));
        var d = new float[b.Length / 4];
        Buffer.BlockCopy(b, 0, d, 0, b.Length);
        return d;
    }

    static double RelRms(float[] ours, float[] theirs)
    {
        double num = 0, den = 0;
        for (int i = 0; i < ours.Length; i++) { double d = ours[i] - theirs[i]; num += d * d; den += (double)theirs[i] * theirs[i]; }
        return Math.Sqrt(num / den);
    }

    static ImageBatch Picture(string name, int height, int width)
    {
        var p = new ImageBatch(1, height, width, 3);
        Raw(name).CopyTo(p.Data, 0);
        return p;
    }

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }

    static NodeContext Context(Services services, string classType) =>
        (NodeContext)Activator.CreateInstance(typeof(NodeContext), System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, null,
            ["1", classType, services, CancellationToken.None, null, new Lazy<string>(() => "{}"), new Dictionary<string, string>(), null, null, 0], null)!;

    static string? Root()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        ModelFolders.Scan(root!);
        return root;
    }

    [Test]
    public void The_vision_and_reference_sizes_are_ComfyUIs()
    {
        // Measured from ComfyUI's vision grids ([1, 24, 32], [1, 22, 36]) and reference latents ([.., 111, 148], [.., 99, 165]).
        Assert.That(TextEncodeQwenImageEditPlus.VisionSize(192, 256), Is.EqualTo(((333, 443), (336, 448))));
        Assert.That(TextEncodeQwenImageEditPlus.VisionSize(120, 200), Is.EqualTo(((297, 496), (308, 504))));
        Assert.That(TextEncodeQwenImageEditPlus.ReferenceSize(192, 256), Is.EqualTo((888, 1184)));
        Assert.That(TextEncodeQwenImageEditPlus.ReferenceSize(120, 200), Is.EqualTo((792, 1320)));
    }

    [Test]
    public void The_vision_picture_is_ComfyUIs()
    {
        // ComfyUI's picture as its vision tower received it (the patch rows put back in place, un-normalised): area, then bilinear, no 8-bit step.
        foreach (var (pic, h, w, want, wh, ww) in new[] { ("f2_img_1x192x256x3.raw", 192, 256, "qe_vision_picture_A_f32_336x448x3.raw", 336, 448),
            ("f2_img_1x120x200x3.raw", 120, 200, "qe_vision_picture_B_f32_308x504x3.raw", 308, 504) })
        {
            var ours = TextEncodeQwenImageEditPlus.VisionPicture(Picture(pic, h, w));
            Assert.That((ours.Height, ours.Width), Is.EqualTo((wh, ww)));
            var theirs = Raw(want);
            double worst = 0;
            for (int i = 0; i < theirs.Length; i++) worst = Math.Max(worst, Math.Abs(ours.Data[i] - theirs[i]));
            Assert.That(worst, Is.LessThan(1e-5), $"{w} x {h}: largest difference {worst:E2} (1.3e-6 measured in float64)");
        }
    }

    static int[] Ids(string name) =>
        System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, name))).RootElement.GetProperty("calls")[0].GetProperty("ids").EnumerateArray().Select(v => v.GetInt32()).ToArray();

    const string NeedsAll = "Needs SPAWNGEN_MODELS_ROOT with text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors, vae/qwen_image_vae.safetensors and a GPU.";

    /// <summary>The node: A alone (the prompt and the empty one) and A with B; ids, conditioning and reference latents against ComfyUI's.</summary>
    [Test, Explicit(NeedsAll)]
    public async Task The_edit_conditioning_is_ComfyUIs()
    {
        var root = Root();
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var services = new Services(runtime);
        var registry = new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly);
        Assert.That(registry.GetSchema("TextEncodeQwenImageEditPlus")!.DisplayName, Is.EqualTo("Text Encode Qwen Image Edit Plus"));
        // The ids, the image-pad ids taken out (ComfyUI's embedding table never sees them: vision start and end sit together).
        var clip = await new CLIPLoader().ExecuteAsync("qwen_2.5_vl_7b_fp8_scaled.safetensors", "qwen_image", Context(services, "CLIPLoader"), services);
        var encoder = (QwenImageTextEncoderModel)clip.Encoder;
        const string Prompt = "Turn the picture into a watercolor painting";
        foreach (var (text, grids, file) in new[] { (Prompt, new[] { (24, 32) }, "qe_te1_ids.json"), ("", new[] { (24, 32) }, "qe_te1_neg_ids.json"),
            (Prompt, new[] { (24, 32), (22, 36) }, "qe_te2_ids.json") })
        {
            var (ids, starts) = await encoder.TokenizeEditAsync(text, grids, CancellationToken.None);
            Assert.That(ids.Where(i => i != QwenImageTextEncoderModel.ImagePad), Is.EqualTo(Ids(file)), file);
            Assert.That(ids.Take(QwenImageTextEncoderModel.EditDropped).Last(), Is.EqualTo(198), "the dropped rows end with <|im_start|>user\n");
            Assert.That(starts[0], Is.EqualTo(70));
        }
        var a = Picture("f2_img_1x192x256x3.raw", 192, 256);
        var b = Picture("f2_img_1x120x200x3.raw", 120, 200);
        var node = new TextEncodeQwenImageEditPlus();
        var clock = System.Diagnostics.Stopwatch.StartNew();
        var vae = await new VAELoader().ExecuteAsync("qwen_image_vae.safetensors", Context(services, "VAELoader"), services);
        var one = (await node.ExecuteAsync(clip, Context(services, "TextEncodeQwenImageEditPlus"), Prompt, vae, a)).Entries.Single();
        var oneNeg = (await node.ExecuteAsync(clip, Context(services, "TextEncodeQwenImageEditPlus"), "", vae, a)).Entries.Single();
        var two = (await node.ExecuteAsync(clip, Context(services, "TextEncodeQwenImageEditPlus"), Prompt, vae, a, b)).Entries.Single();
        Assert.That(one.Shape, Is.EqualTo(new[] { 1, 212, 3584 }));
        Assert.That(oneNeg.Shape, Is.EqualTo(new[] { 1, 204, 3584 }));
        Assert.That(two.Shape, Is.EqualTo(new[] { 1, 417, 3584 }));
        double e1 = RelRms(one.Embeddings, Raw("qe_te1_cond_0_cond_f32_1x212x3584.raw")), e1n = RelRms(oneNeg.Embeddings, Raw("qe_te1_neg_cond_0_cond_f32_1x204x3584.raw")),
            e2 = RelRms(two.Embeddings, Raw("qe_te2_cond_0_cond_f32_1x417x3584.raw"));
        Assert.That(one.ReferenceLatents!.Select(r => r.Shape), Is.EqualTo(new[] { new[] { 1, 16, 1, 111, 148 } }));
        Assert.That(two.ReferenceLatents!.Select(r => r.Shape), Is.EqualTo(new[] { new[] { 1, 16, 1, 111, 148 }, new[] { 1, 16, 1, 99, 165 } }));
        double r1 = RelRms(one.ReferenceLatents![0].Data, Raw("qe_te1_cond_0_reference_latents_0_f32_1x16x1x111x148.raw")),
            r2 = RelRms(two.ReferenceLatents![1].Data, Raw("qe_te2_cond_0_reference_latents_1_f32_1x16x1x99x165.raw"));
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: one picture {e1:E2} (negative {e1n:E2}), two pictures {e2:E2}; reference latents A {r1:E2}, B {r2:E2}; {clock.Elapsed.TotalSeconds:F1} s");
        Assert.That(e1, Is.LessThan(3e-4), "4.4e-5 on the RX 590");
        Assert.That(e1n, Is.LessThan(3e-4), "4.5e-5 on the RX 590");
        Assert.That(e2, Is.LessThan(3e-4), "7.3e-5 on the RX 590");
        Assert.That(r1, Is.LessThan(1e-5), "1.2e-6 on the RX 590");
        Assert.That(r2, Is.LessThan(1e-5), "1.4e-6 on the RX 590");
    }

    const string NeedsVae = "Needs SPAWNGEN_MODELS_ROOT with vae/qwen_image_vae.safetensors and a GPU.";

    /// <summary>
    /// Which resampler ComfyUI uses for the reference latent's picture (about 1 MP: each side times sqrt(1024^2 / (w h)), rounded to a multiple of
    /// 8; A -> 1184 x 888, B -> 1320 x 792, measured from the latents' sizes): each candidate encoded with the same VAE (8e-7 from ComfyUI's) and
    /// compared with ComfyUI's reference latent. Prints every candidate; asserts the node's own choice is the best and within 1e-4.
    /// </summary>
    [Test, Explicit(NeedsVae)]
    public async Task The_reference_latent_resize_is_ComfyUIs()
    {
        var root = Root();
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var services = new Services(runtime);
        var vae = await new VAELoader().ExecuteAsync("qwen_image_vae.safetensors", Context(services, "VAELoader"), services);
        var report = new List<string>();
        foreach (var (pic, h, w, refFile, lh, lw) in new[]
        {
            ("f2_img_1x192x256x3.raw", 192, 256, "qe_te1_cond_0_reference_latents_0_f32_1x16x1x111x148.raw", 111, 148),
            ("f2_img_1x120x200x3.raw", 120, 200, "qe_te2_cond_0_reference_latents_1_f32_1x16x1x99x165.raw", 99, 165),
        })
        {
            var picture = Picture(pic, h, w);
            var (rh, rw) = TextEncodeQwenImageEditPlus.ReferenceSize(h, w);
            Assert.That((rh, rw), Is.EqualTo((lh * 8, lw * 8)), $"{w} x {h}: the reference picture's size");
            var want = Raw(refFile);
            var errors = new Dictionary<string, double>();
            foreach (var method in new[] { "area", "bilinear", "bicubic", "lanczos", "nearest-exact" })
            {
                var latent = await new VAEEncode().ExecuteAsync(CommonUpscale.Resize(picture, rw, rh, method), vae, Context(services, "VAEEncode"), services);
                errors[method] = RelRms(latent.Data, want);
            }
            report.Add($"{w} x {h} -> {rw} x {rh}: " + string.Join(", ", errors.OrderBy(e => e.Value).Select(e => $"{e.Key} {e.Value:E2}")));
            var best = errors.OrderBy(e => e.Value).First();
            Assert.That(best.Key, Is.EqualTo(TextEncodeQwenImageEditPlus.ReferenceMethod), string.Join("; ", report));
            Assert.That(best.Value, Is.LessThan(1e-4), string.Join("; ", report));
        }
        TestContext.Out.WriteLine($"{runtime.Accelerator.Name}: " + string.Join("; ", report));
    }
}
