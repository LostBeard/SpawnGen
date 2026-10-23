using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;
using SpawnGen.Nodes.ML;

namespace SpawnGen.Nodes.Tests;

/// <summary>
/// SD-Turbo end to end against ComfyUI itself. Fixtures/sampling/comfyui_sd_turbo holds a noised start latent (make_start.py) and what ComfyUI made from it
/// (run_comfy.py: sd_turbo.safetensors, "a photo of a red apple on a wooden table", empty negative, KSampler (Advanced) with add_noise disabled, 4 steps,
/// cfg 1, normal). With the same start and no fresh noise the deterministic samplers must land on ComfyUI's latent: a seed-only comparison cannot tell a
/// conditioning or guidance difference from a different noise draw, this can.
/// </summary>
public class ComfyUIRenderParityTests
{
    [Test]
    public void SD2_prompts_are_padded_with_token_0_and_SD1_prompts_with_the_end_token()
    {
        int[] ids = [49406, 320, 1125, 49407, 49407, 49407];
        Assert.That(ClipPadding.Pad(ids, ClipPadding.ForTextWidth(1024)), Is.EqualTo(new[] { 49406, 320, 1125, 49407, 0, 0 }), "OpenCLIP-H (SD 2.x, SD-Turbo): ComfyUI's pad 0");
        Assert.That(ClipPadding.Pad(ids, ClipPadding.ForTextWidth(768)), Is.EqualTo(ids), "CLIP-L (SD 1.x): ComfyUI's pad_with_end");
        Assert.That(ClipPadding.Pad([49406, 49407, 49407], 0), Is.EqualTo(new[] { 49406, 49407, 0 }), "the empty prompt keeps one end token");
        Assert.That(ids[4], Is.EqualTo(49407), "the input is not changed");
    }

    [TestCase("euler")]
    [TestCase("res_multistep_cfg_pp")] // CFG++: leans on the empty prompt's estimate, which the padding changed most
    [TestCase("dpm_fast")]
    [Explicit("Needs SPAWNGEN_MODELS (sd_turbo) and a GPU.")]
    public async Task Same_start_latent_gives_ComfyUIs_result(string sampler)
    {
        var models = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS");
        if (string.IsNullOrEmpty(models)) Assert.Ignore("SPAWNGEN_MODELS is not set.");
        var dir = Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "sampling", "comfyui_sd_turbo");
        var outDir = Path.Combine(Path.GetTempPath(), "spawngen-comfy-parity-" + Guid.NewGuid().ToString("N"));
        var previousInput = InputFiles.Directory;
        InputFiles.Directory = dir;
        try
        {
            using var runtime = new MlRuntime(new FolderModelSource(models!));
            var executor = new PromptExecutor(new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly),
                new Services(runtime, new OutputOptions { Directory = outDir }));
            var g = new PromptGraph();
            g.Add("1", "CheckpointLoaderSimple", ("ckpt_name", "sd_turbo"));
            g.Add("2", "CLIPTextEncode", ("text", "a photo of a red apple on a wooden table"), ("clip", PromptInput.Link("1", 1)));
            g.Add("3", "CLIPTextEncode", ("text", ""), ("clip", PromptInput.Link("1", 1)));
            g.Add("4", "LoadLatent", ("latent", "start.latent"));
            g.Add("5", "KSamplerAdvanced", ("model", PromptInput.Link("1")), ("add_noise", "disable"), ("noise_seed", 0), ("steps", 4), ("cfg", 1.0),
                ("sampler_name", sampler), ("scheduler", "normal"), ("positive", PromptInput.Link("2")), ("negative", PromptInput.Link("3")),
                ("latent_image", PromptInput.Link("4")), ("start_at_step", 0), ("end_at_step", 10000), ("return_with_leftover_noise", "disable"));
            g.Add("6", "SaveLatent", ("samples", PromptInput.Link("5")), ("filename_prefix", "parity"));
            g.Add("7", "VAEDecode", ("samples", PromptInput.Link("5")), ("vae", PromptInput.Link("1", 2)));
            g.Add("8", "SaveImage", ("images", PromptInput.Link("7")), ("filename_prefix", "parity_" + sampler));
            await executor.ExecuteAsync(g);

            var ours = LoadLatent.Decode(File.ReadAllBytes(Directory.GetFiles(outDir, "parity_*.latent", SearchOption.AllDirectories).Single())).Data;
            // ComfyUI's LATENT is in VAE space; SpawnGen's is the model's (x 0.18215).
            var comfy = LoadLatent.Decode(File.ReadAllBytes(Path.Combine(dir, sampler + ".latent"))).Data.Select(v => v * 0.18215f).ToArray();
            double diff = Math.Sqrt(ours.Zip(comfy, (a, b) => (double)(a - b) * (a - b)).Average());
            double scale = Math.Sqrt(comfy.Average(v => (double)v * v));
            TestContext.Out.WriteLine($"{sampler}: rms difference {diff:F4} of rms {scale:F4}; picture {Directory.GetFiles(outDir, "parity_*.png").Single()}");
            // ONNX fp32 on the GPU against torch fp32 on the CPU: about 0.002 to 0.008. The old SD-2 padding gave 0.39 to 0.80.
            Assert.That(diff, Is.LessThan(0.03 * scale));
        }
        finally
        {
            InputFiles.Directory = previousInput;
        }
    }

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }
}
