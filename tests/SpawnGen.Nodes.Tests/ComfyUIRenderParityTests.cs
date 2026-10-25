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
        using var runtime = new MlRuntime(new FolderModelSource(models!));
        var (diff, scale) = await RunAsync(runtime, "sd_turbo", sampler, "comfyui_sd_turbo", 4, 1.0);
        // ONNX fp32 on the GPU against torch fp32 on the CPU: about 0.002 to 0.008. The old SD-2 padding gave 0.39 to 0.80.
        Assert.That(diff, Is.LessThan(0.03 * scale));
    }

    /// <summary>
    /// The same comparison with the very file ComfyUI loaded (sd_turbo.safetensors in &lt;models&gt;/checkpoints), through the library's single-file loader
    /// (SdCheckpoint: SD 2.x's OpenCLIP-H text encoder, UNet and VAE from the checkpoint, weights held as fp16 on the GPU).
    /// </summary>
    [TestCase("euler")]
    [TestCase("res_multistep_cfg_pp")]
    [TestCase("dpm_fast")]
    [Explicit("Needs SPAWNGEN_MODELS_ROOT (a models folder with checkpoints/sd_turbo.safetensors) and a GPU.")]
    public async Task Same_start_latent_from_the_safetensors_checkpoint_gives_ComfyUIs_result(string sampler)
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        CheckpointCatalog.ScanCheckpointFiles(root!);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var (diff, scale) = await RunAsync(runtime, "sd_turbo.safetensors", sampler, "comfyui_sd_turbo", 4, 1.0);
        Assert.That(diff, Is.LessThan(0.03 * scale));
    }

    /// <summary>
    /// SD 1.5 from the original single file (v1-5-pruned-emaonly.safetensors in &lt;models&gt;/checkpoints) against ComfyUI's render of the same start latent
    /// (Fixtures/sampling/comfyui_sd15, run_comfy.py: euler, 10 steps, cfg 7.5, normal, ComfyUI on the CPU in float32). Real classifier-free guidance, so the
    /// empty prompt's encoding and the CFG blend are checked too.
    /// </summary>
    [TestCase("v1-5-pruned-emaonly.safetensors")]
    [TestCase("v1-5-pruned-emaonly-fp16.safetensors")] // ComfyUI's fp16 archive: downloaded from Hugging Face on first use unless it is in checkpoints/
    [Explicit("Needs SPAWNGEN_MODELS_ROOT (a models folder with checkpoints/v1-5-pruned-emaonly.safetensors) and a GPU.")]
    public async Task SD15_from_the_safetensors_checkpoint_gives_ComfyUIs_result(string checkpoint)
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        CheckpointCatalog.ScanCheckpointFiles(root!);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var (diff, scale) = await RunAsync(runtime, checkpoint, "euler", "comfyui_sd15", 10, 7.5);
        Assert.That(diff, Is.LessThan(0.03 * scale));
    }

    /// <summary>
    /// SDXL base from the original single file at 1024x1024 against ComfyUI's render of the same start latent (Fixtures/sampling/comfyui_sdxl: euler, 6 steps,
    /// cfg 7, normal). Checks both text towers and their padding, the pooled embedding, the size label ComfyUI builds from the latent's size, and SDXL's
    /// latent scale 0.13025.
    /// </summary>
    [Test, Explicit("Needs SPAWNGEN_MODELS_ROOT (a models folder with checkpoints/sd_xl_base_1.0.safetensors) and a GPU with 8 GB.")]
    public async Task SDXL_from_the_safetensors_checkpoint_gives_ComfyUIs_result()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        CheckpointCatalog.ScanCheckpointFiles(root!);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var (diff, scale) = await RunAsync(runtime, "sd_xl_base_1.0.safetensors", "euler", "comfyui_sdxl", 6, 7.0);
        Assert.That(diff, Is.LessThan(0.03 * scale));
    }

    /// <summary>
    /// SD 1.5 with a LoRA (latent-consistency/lcm-lora-sdv1-5 as &lt;models&gt;/loras/lcm-lora-sdv1-5.safetensors) through Load LoRA (Model and CLIP) at strength 1,
    /// against ComfyUI's render of the same graph (euler, 4 steps, cfg 1).
    /// </summary>
    [Test, Explicit("Needs SPAWNGEN_MODELS_ROOT (checkpoints/v1-5-pruned-emaonly.safetensors, loras/lcm-lora-sdv1-5.safetensors) and a GPU.")]
    public async Task SD15_with_a_LoRA_gives_ComfyUIs_result()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        CheckpointCatalog.ScanCheckpointFiles(root!);
        CheckpointCatalog.ScanLoraFiles(root!);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var (diff, scale) = await RunAsync(runtime, "v1-5-pruned-emaonly.safetensors", "euler", "comfyui_sd15", 4, 1.0, lora: "lcm-lora-sdv1-5.safetensors", result: "euler_lcm_lora");
        Assert.That(diff, Is.LessThan(0.03 * scale));
    }

    static async Task<(double Diff, double Scale)> RunAsync(MlRuntime runtime, string checkpoint, string sampler, string fixture, int steps, double cfg,
        string? lora = null, string? result = null)
    {
        var dir = Path.Combine(TestContext.CurrentContext.TestDirectory, "Fixtures", "sampling", fixture);
        var outDir = Path.Combine(Path.GetTempPath(), "spawngen-comfy-parity-" + Guid.NewGuid().ToString("N"));
        var previousInput = InputFiles.Directory;
        InputFiles.Directory = dir;
        try
        {
            var executor = new PromptExecutor(new NodeRegistry().AddAssembly(typeof(SaveImage).Assembly).AddAssembly(typeof(KSampler).Assembly),
                new Services(runtime, new OutputOptions { Directory = outDir }));
            var g = new PromptGraph();
            g.Add("1", "CheckpointLoaderSimple", ("ckpt_name", checkpoint));
            // With a LoRA, Load LoRA (Model and CLIP) sits between the checkpoint and everything that used it (as run_comfy.py builds it).
            string from = "1";
            if (lora is not null)
            {
                g.Add("9", "LoraLoader", ("model", PromptInput.Link("1")), ("clip", PromptInput.Link("1", 1)), ("lora_name", lora), ("strength_model", 1.0), ("strength_clip", 1.0));
                from = "9";
            }
            g.Add("2", "CLIPTextEncode", ("text", "a photo of a red apple on a wooden table"), ("clip", PromptInput.Link(from, 1)));
            g.Add("3", "CLIPTextEncode", ("text", ""), ("clip", PromptInput.Link(from, 1)));
            g.Add("4", "LoadLatent", ("latent", "start.latent"));
            g.Add("5", "KSamplerAdvanced", ("model", PromptInput.Link(from)), ("add_noise", "disable"), ("noise_seed", 0), ("steps", steps), ("cfg", cfg),
                ("sampler_name", sampler), ("scheduler", "normal"), ("positive", PromptInput.Link("2")), ("negative", PromptInput.Link("3")),
                ("latent_image", PromptInput.Link("4")), ("start_at_step", 0), ("end_at_step", 10000), ("return_with_leftover_noise", "disable"));
            g.Add("6", "SaveLatent", ("samples", PromptInput.Link("5")), ("filename_prefix", "parity"));
            g.Add("7", "VAEDecode", ("samples", PromptInput.Link("5")), ("vae", PromptInput.Link("1", 2)));
            g.Add("8", "SaveImage", ("images", PromptInput.Link("7")), ("filename_prefix", "parity_" + sampler));
            await executor.ExecuteAsync(g);

            var ours = LoadLatent.Decode(File.ReadAllBytes(Directory.GetFiles(outDir, "parity_*.latent", SearchOption.AllDirectories).Single())).Data;
            // ComfyUI saved its LATENT (the VAE's values), which is what SpawnGen's LATENT holds too.
            var comfy = LoadLatent.Decode(File.ReadAllBytes(Path.Combine(dir, (result ?? sampler) + ".latent"))).Data;
            double diff = Math.Sqrt(ours.Zip(comfy, (a, b) => (double)(a - b) * (a - b)).Average());
            double scale = Math.Sqrt(comfy.Average(v => (double)v * v));
            TestContext.Out.WriteLine($"{checkpoint} {sampler}: rms difference {diff:F4} of rms {scale:F4}; picture {Directory.GetFiles(outDir, "parity_*.png").Single()}");
            return (diff, scale);
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
