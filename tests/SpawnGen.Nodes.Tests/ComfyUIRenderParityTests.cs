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

    /// <summary>
    /// SD 1.5 from ComfyUI's own component files (ModelSave / CLIPSave / VAESave of v1-5-pruned-emaonly: diffusion_models/sd15_unet.safetensors with
    /// model.diffusion_model. names, text_encoders/sd15_clip_l.safetensors in Hugging Face names, vae/sd15_vae.safetensors in LDM names) through Load Diffusion
    /// Model, Load CLIP and Load VAE: the same graph as the single file must land on ComfyUI's single-file render.
    /// </summary>
    [Test, Explicit("Needs SPAWNGEN_MODELS_ROOT (diffusion_models/sd15_unet, text_encoders/sd15_clip_l, vae/sd15_vae .safetensors) and a GPU.")]
    public async Task SD15_from_split_files_gives_ComfyUIs_single_file_result()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        ModelFolders.Scan(root!);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var (diff, scale) = await RunAsync(runtime, "", "euler", "comfyui_sd15", 10, 7.5, load: g =>
        {
            g.Add("11", "UNETLoader", ("unet_name", "sd15_unet.safetensors"), ("weight_dtype", "default"));
            g.Add("12", "CLIPLoader", ("clip_name", "sd15_clip_l.safetensors"), ("type", "stable_diffusion"));
            g.Add("13", "VAELoader", ("vae_name", "sd15_vae.safetensors"));
            return (PromptInput.Link("11"), PromptInput.Link("12"), PromptInput.Link("13"));
        });
        Assert.That(diff, Is.LessThan(0.03 * scale));
    }

    /// <summary>
    /// A real standalone VAE: Stability's vae-ft-mse-840000-ema-pruned (&lt;models&gt;/vae) through Load VAE on SD 1.5's samples, against ComfyUI's picture decoded
    /// with the same file (Fixtures/sampling/comfyui_sd15/euler_ftmse.png). Its decoder differs from the checkpoint's, so the stock picture (euler.png) would fail.
    /// </summary>
    [Test, Explicit("Needs SPAWNGEN_MODELS_ROOT (checkpoints/v1-5-pruned-emaonly.safetensors, vae/vae-ft-mse-840000-ema-pruned.safetensors) and a GPU.")]
    public async Task SD15_with_a_standalone_VAE_gives_ComfyUIs_picture()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        CheckpointCatalog.ScanCheckpointFiles(root!);
        ModelFolders.Scan(root!);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var (diff, scale) = await RunAsync(runtime, "", "euler", "comfyui_sd15", 10, 7.5, picture: "euler_ftmse", load: g =>
        {
            g.Add("1", "CheckpointLoaderSimple", ("ckpt_name", "v1-5-pruned-emaonly.safetensors"));
            g.Add("13", "VAELoader", ("vae_name", "vae-ft-mse-840000-ema-pruned.safetensors"));
            return (PromptInput.Link("1"), PromptInput.Link("1", 1), PromptInput.Link("13"));
        });
        Assert.That(diff, Is.LessThan(0.03 * scale));
    }

    /// <summary>
    /// SDXL from ComfyUI's own component files (ModelSave / CLIPSave / VAESave of sd_xl_base_1.0) through Load Diffusion Model, Load CLIP (Dual) and Load VAE,
    /// against ComfyUI's single-file render at 1024 (latent and picture). The two CLIP files are given bigG first: ComfyUI takes them in either order.
    /// </summary>
    [Test, Explicit("Needs SPAWNGEN_MODELS_ROOT (diffusion_models/sdxl_unet, text_encoders/sdxl_clip_l + sdxl_clip_g, vae/sdxl_vae .safetensors) and a GPU with 8 GB.")]
    public async Task SDXL_from_split_files_gives_ComfyUIs_single_file_result()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        ModelFolders.Scan(root!);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var (diff, scale) = await RunAsync(runtime, "", "euler", "comfyui_sdxl", 6, 7.0, load: g =>
        {
            g.Add("11", "UNETLoader", ("unet_name", "sdxl_unet.safetensors"), ("weight_dtype", "default"));
            g.Add("12", "DualCLIPLoader", ("clip_name1", "sdxl_clip_g.safetensors"), ("clip_name2", "sdxl_clip_l.safetensors"), ("type", "sdxl"));
            g.Add("13", "VAELoader", ("vae_name", "sdxl_vae.safetensors"));
            return (PromptInput.Link("11"), PromptInput.Link("12"), PromptInput.Link("13"));
        });
        Assert.That(diff, Is.LessThan(0.03 * scale));
    }

    /// <summary>
    /// LoRA on components loaded alone: SD 1.5's split files (see <see cref="SD15_from_split_files_gives_ComfyUIs_single_file_result"/>) plus the LCM-LoRA through
    /// Load LoRA (Model and CLIP), against ComfyUI's checkpoint + LoRA render (the same weights, merged into the standalone UNet and text encoder).
    /// </summary>
    [Test, Explicit("Needs SPAWNGEN_MODELS_ROOT (the SD 1.5 split files, loras/lcm-lora-sdv1-5.safetensors) and a GPU.")]
    public async Task SD15_split_files_with_a_LoRA_give_ComfyUIs_result()
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        ModelFolders.Scan(root!);
        CheckpointCatalog.ScanLoraFiles(root!);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var (diff, scale) = await RunAsync(runtime, "", "euler", "comfyui_sd15", 4, 1.0, lora: "lcm-lora-sdv1-5.safetensors", result: "euler_lcm_lora", load: g =>
        {
            g.Add("11", "UNETLoader", ("unet_name", "sd15_unet.safetensors"), ("weight_dtype", "default"));
            g.Add("12", "CLIPLoader", ("clip_name", "sd15_clip_l.safetensors"), ("type", "stable_diffusion"));
            g.Add("13", "VAELoader", ("vae_name", "sd15_vae.safetensors"));
            return (PromptInput.Link("11"), PromptInput.Link("12"), PromptInput.Link("13"));
        });
        Assert.That(diff, Is.LessThan(0.03 * scale));
    }

    /// <summary>
    /// SD 2.x from component files: ComfyUI's ModelSave / CLIPSave / VAESave of sd_turbo.safetensors (an OpenCLIP-H text encoder, a UNet-only file whose
    /// prediction type is read from its weights) against ComfyUI's single-file render of the same start latent.
    /// </summary>
    [TestCase("euler")]
    [TestCase("dpm_fast")]
    [Explicit("Needs SPAWNGEN_MODELS_ROOT (diffusion_models/sdturbo_unet, text_encoders/sdturbo_clip_h, vae/sdturbo_vae .safetensors) and a GPU.")]
    public async Task SD_Turbo_from_split_files_gives_ComfyUIs_result(string sampler)
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        ModelFolders.Scan(root!);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var (diff, scale) = await RunAsync(runtime, "", sampler, "comfyui_sd_turbo", 4, 1.0, load: g =>
        {
            g.Add("11", "UNETLoader", ("unet_name", "sdturbo_unet.safetensors"), ("weight_dtype", "default"));
            g.Add("12", "CLIPLoader", ("clip_name", "sdturbo_clip_h.safetensors"), ("type", "stable_diffusion"));
            g.Add("13", "VAELoader", ("vae_name", "sdturbo_vae.safetensors"));
            return (PromptInput.Link("11"), PromptInput.Link("12"), PromptInput.Link("13"));
        });
        Assert.That(diff, Is.LessThan(0.03 * scale));
    }

    /// <summary>
    /// ControlNet: SD 1.5 + control_v11p_sd15_canny_fp16 (&lt;models&gt;/controlnet) through Load ControlNet Model and Apply ControlNet, with ComfyUI's own canny edge
    /// map of its SD 1.5 apple (Fixtures/sampling/comfyui_sd15/canny_hint.png, from its Detect Edges node), against ComfyUI's render of the same graph: strength 1
    /// over the whole run, and strength 0.8 over the first half only (the percent range).
    /// </summary>
    [TestCase(1.0, 1.0, "euler_cn_full")]
    [TestCase(0.8, 0.5, "euler_cn_half")]
    [Explicit("Needs SPAWNGEN_MODELS_ROOT (checkpoints/v1-5-pruned-emaonly.safetensors, controlnet/control_v11p_sd15_canny_fp16.safetensors) and a GPU.")]
    public async Task SD15_with_a_ControlNet_gives_ComfyUIs_result(double strength, double endPercent, string result)
    {
        var root = Environment.GetEnvironmentVariable("SPAWNGEN_MODELS_ROOT");
        if (string.IsNullOrEmpty(root)) Assert.Ignore("SPAWNGEN_MODELS_ROOT is not set.");
        CheckpointCatalog.ScanCheckpointFiles(root!);
        ModelFolders.Scan(root!);
        using var runtime = new MlRuntime(new DesktopModelSource(root!, MlRuntime.RepoFolders));
        var (diff, scale) = await RunAsync(runtime, "v1-5-pruned-emaonly.safetensors", "euler", "comfyui_sd15", 10, 7.5, result: result,
            controlNet: ("control_v11p_sd15_canny_fp16.safetensors", "canny_hint.png", strength, endPercent));
        Assert.That(diff, Is.LessThan(0.03 * scale));
    }

    static async Task<(double Diff, double Scale)> RunAsync(MlRuntime runtime, string checkpoint, string sampler, string fixture, int steps, double cfg,
        string? lora = null, string? result = null, Func<PromptGraph, (PromptInput Model, PromptInput Clip, PromptInput Vae)>? load = null, string? picture = null,
        (string Name, string Hint, double Strength, double EndPercent)? controlNet = null)
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
            // The model, CLIP and VAE come from Load Checkpoint, or from the loaders the test adds.
            PromptInput model, clip, vae;
            if (load is null)
            {
                g.Add("1", "CheckpointLoaderSimple", ("ckpt_name", checkpoint));
                (model, clip, vae) = (PromptInput.Link("1"), PromptInput.Link("1", 1), PromptInput.Link("1", 2));
            }
            else (model, clip, vae) = load(g);
            // With a LoRA, Load LoRA (Model and CLIP) sits between the loader and everything that used it (as run_comfy.py builds it).
            if (lora is not null)
            {
                g.Add("9", "LoraLoader", ("model", model), ("clip", clip), ("lora_name", lora), ("strength_model", 1.0), ("strength_clip", 1.0));
                (model, clip) = (PromptInput.Link("9"), PromptInput.Link("9", 1));
            }
            g.Add("2", "CLIPTextEncode", ("text", "a photo of a red apple on a wooden table"), ("clip", clip));
            g.Add("3", "CLIPTextEncode", ("text", ""), ("clip", clip));
            g.Add("4", "LoadLatent", ("latent", "start.latent"));
            // With a ControlNet, Apply ControlNet sits between the prompts and the sampler (as the ComfyUI reference graph has it).
            PromptInput positive = PromptInput.Link("2"), negative = PromptInput.Link("3");
            if (controlNet is { } cn)
            {
                g.Add("20", "ControlNetLoader", ("control_net_name", cn.Name));
                g.Add("21", "LoadImage", ("image", cn.Hint));
                g.Add("22", "ControlNetApplyAdvanced", ("positive", positive), ("negative", negative), ("control_net", PromptInput.Link("20")),
                    ("image", PromptInput.Link("21")), ("strength", cn.Strength), ("start_percent", 0.0), ("end_percent", cn.EndPercent));
                (positive, negative) = (PromptInput.Link("22"), PromptInput.Link("22", 1));
            }
            g.Add("5", "KSamplerAdvanced", ("model", model), ("add_noise", "disable"), ("noise_seed", 0), ("steps", steps), ("cfg", cfg),
                ("sampler_name", sampler), ("scheduler", "normal"), ("positive", positive), ("negative", negative),
                ("latent_image", PromptInput.Link("4")), ("start_at_step", 0), ("end_at_step", 10000), ("return_with_leftover_noise", "disable"));
            g.Add("6", "SaveLatent", ("samples", PromptInput.Link("5")), ("filename_prefix", "parity"));
            g.Add("7", "VAEDecode", ("samples", PromptInput.Link("5")), ("vae", vae));
            g.Add("8", "SaveImage", ("images", PromptInput.Link("7")), ("filename_prefix", "parity_" + sampler));
            await executor.ExecuteAsync(g);

            var ours = LoadLatent.Decode(File.ReadAllBytes(Directory.GetFiles(outDir, "parity_*.latent", SearchOption.AllDirectories).Single())).Data;
            // ComfyUI saved its LATENT (the VAE's values), which is what SpawnGen's LATENT holds too.
            var comfy = LoadLatent.Decode(File.ReadAllBytes(Path.Combine(dir, (result ?? sampler) + ".latent"))).Data;
            double diff = Math.Sqrt(ours.Zip(comfy, (a, b) => (double)(a - b) * (a - b)).Average());
            double scale = Math.Sqrt(comfy.Average(v => (double)v * v));
            TestContext.Out.WriteLine($"{(load is null ? checkpoint : "split files")} {sampler}: rms difference {diff:F4} of rms {scale:F4}; picture {Directory.GetFiles(outDir, "parity_*.png").Single()}");
            // Where the fixture has ComfyUI's decoded picture too, the VAE is checked: mean and largest difference per channel, in 1/255 steps.
            var comfyPng = Path.Combine(dir, (picture ?? result ?? sampler) + ".png");
            if (File.Exists(comfyPng))
            {
                var theirs = ImageDecoder.Decode(File.ReadAllBytes(comfyPng)).Data;
                var mine = ImageDecoder.Decode(File.ReadAllBytes(Directory.GetFiles(outDir, "parity_*.png").Single())).Data;
                Assert.That(mine, Has.Length.EqualTo(theirs.Length), "picture size");
                double mean = mine.Zip(theirs, (x, y) => Math.Abs(x - y)).Average() * 255, max = mine.Zip(theirs, (x, y) => Math.Abs(x - y)).Max() * 255;
                TestContext.Out.WriteLine($"  picture vs ComfyUI's: mean difference {mean:F2}/255, largest {max:F0}/255");
                Assert.That(mean, Is.LessThan(2.0), "the decoded picture is ComfyUI's (the VAE is checked)");
            }
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
