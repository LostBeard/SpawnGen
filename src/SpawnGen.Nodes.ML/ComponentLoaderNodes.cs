using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// The model files of ComfyUI's component folders under the models folder, listed by their path there (sub-folders too) as ComfyUI lists them:
/// diffusion_models (and the older unet), vae, text_encoders (and the older clip). Each file opens from the model source as repo <c>local/&lt;folder&gt;</c>.
/// </summary>
public static class ModelFolders
{
    public const string DiffusionModels = "diffusion_models", Vae = "vae", TextEncoders = "text_encoders", ControlNet = "controlnet", VaeApprox = "vae_approx";

    /// <summary>ComfyUI's folder names per kind: the current one first, then the older one it still reads.</summary>
    static readonly Dictionary<string, string[]> Folders = new()
    {
        [DiffusionModels] = ["diffusion_models", "unet"],
        [Vae] = ["vae"],
        [TextEncoders] = ["text_encoders", "clip"],
        [ControlNet] = ["controlnet", "t2i_adapter"],
        [VaeApprox] = ["vae_approx"],
    };

    static readonly object Gate = new();
    static readonly Dictionary<string, SortedDictionary<string, (string Repo, string File)>> Files = new();

    /// <summary>Lists every <c>.safetensors</c> file of every component folder. Returns how many were added.</summary>
    public static int Scan(string modelsFolder)
    {
        int added = 0;
        foreach (var (kind, folders) in Folders)
            foreach (var folder in folders)
            {
                var dir = Path.Combine(modelsFolder, folder);
                if (!Directory.Exists(dir)) continue;
                foreach (var path in Directory.EnumerateFiles(dir, "*.safetensors", SearchOption.AllDirectories))
                {
                    var name = Path.GetRelativePath(dir, path);
                    lock (Gate)
                    {
                        if (!Files.TryGetValue(kind, out var files)) Files[kind] = files = new(StringComparer.Ordinal);
                        if (files.TryAdd(name, ("local/" + folder, name))) added++;
                    }
                }
            }
        return added;
    }

    /// <summary>The names a loader offers for a kind.</summary>
    public static string[] Names(string kind) { lock (Gate) return Files.TryGetValue(kind, out var f) ? [.. f.Keys] : []; }

    /// <summary>Where a listed file opens from on the model source.</summary>
    public static (string Repo, string File) Locate(string kind, string name)
    {
        lock (Gate)
            if (Files.TryGetValue(kind, out var f) && f.TryGetValue(name, out var at)) return at;
        throw new ArgumentException($"'{name}' is not in the models folder's {Folders[kind][0]} folder.");
    }

    /// <summary>Repo id to folder for the model source (every component folder, whether or not it exists yet).</summary>
    public static IEnumerable<KeyValuePair<string, string>> RepoFolders => Folders.Values.SelectMany(f => f).Select(f => KeyValuePair.Create("local/" + f, f));
}

[Node(Category = "model/loaders", DisplayName = "Load Diffusion Model")]
public sealed class UNETLoader
{
    public static string[] UnetNames() => ModelFolders.Names(ModelFolders.DiffusionModels);
    public static string[] WeightDtypes() => ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"];

    /// <summary>
    /// A diffusion model (UNet) alone from <c>&lt;models&gt;/diffusion_models</c>: SD 1.x, SD 2.x or SDXL, in LDM names (with or without <c>model.diffusion_model.</c>)
    /// or diffusers names. The weights are held as fp16; the fp8 choices are not available.
    /// </summary>
    public async Task<ModelHandle> ExecuteAsync([Input(OptionsProvider = nameof(UnetNames))] string unetName,
        [Input(OptionsProvider = nameof(WeightDtypes))] string weightDtype, NodeContext context, IServiceProvider services)
    {
        if (weightDtype != "default")
            throw new NotSupportedException($"weight_dtype {weightDtype}: SpawnGen holds diffusion model weights as fp16; choose 'default'.");
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var (repo, file) = ModelFolders.Locate(ModelFolders.DiffusionModels, unetName);
        context.Status($"Loading {unetName}");
        var model = await runtime.GetOrLoadAsync("unet:" + repo + "/" + file, () => StandaloneDiffusionModel.LoadAsync(runtime, unetName, repo, file));
        return new ModelHandle(model);
    }
}

[Node(Category = "model/loaders", DisplayName = "Load VAE")]
public sealed class VAELoader
{
    /// <summary>The tiny VAEs SpawnGen runs (ComfyUI's image TAEs for SD 1.x / 2.x and SDXL) with their families' latent scales.</summary>
    static readonly Dictionary<string, float> Taes = new() { ["taesd"] = 0.18215f, ["taesdxl"] = 0.13025f };

    /// <summary>The VAE files, then each tiny VAE whose encoder and decoder are both in <c>&lt;models&gt;/vae_approx</c>, then pixel_space, as ComfyUI lists them.</summary>
    public static string[] VaeNames()
    {
        var approx = ModelFolders.Names(ModelFolders.VaeApprox);
        var taes = Taes.Keys.Where(t => TaeFile(approx, t, "decoder") is not null && TaeFile(approx, t, "encoder") is not null);
        return [.. ModelFolders.Names(ModelFolders.Vae), .. taes, "pixel_space"];
    }

    static string? TaeFile(IEnumerable<string> approx, string tae, string part) => approx.FirstOrDefault(n => n.StartsWith($"{tae}_{part}.", StringComparison.Ordinal));

    /// <summary>
    /// A VAE alone from <c>&lt;models&gt;/vae</c> (e.g. vae-ft-mse-840000-ema-pruned, sdxl_vae): LDM names with or without <c>first_stage_model.</c>, or diffusers names;
    /// or a tiny VAE (taesd, taesdxl) from <c>&lt;models&gt;/vae_approx</c>, fast and rougher.
    /// </summary>
    public async Task<VaeHandle> ExecuteAsync([Input(OptionsProvider = nameof(VaeNames))] string vaeName, NodeContext context, IServiceProvider services)
    {
        if (vaeName == "pixel_space") throw new NotSupportedException("pixel_space is for pixel-space models, which SpawnGen does not run yet.");
        var runtime = await GpuHelpers.RuntimeAsync(services);
        if (Taes.TryGetValue(vaeName, out var scale))
        {
            var approx = ModelFolders.Names(ModelFolders.VaeApprox);
            var decoder = ModelFolders.Locate(ModelFolders.VaeApprox, TaeFile(approx, vaeName, "decoder") ?? throw new ArgumentException($"No {vaeName}_decoder in vae_approx."));
            var encoder = ModelFolders.Locate(ModelFolders.VaeApprox, TaeFile(approx, vaeName, "encoder") ?? throw new ArgumentException($"No {vaeName}_encoder in vae_approx."));
            context.Status($"Loading {vaeName}");
            var tae = await runtime.GetOrLoadAsync($"tae:{vaeName}:{decoder.Repo}/{decoder.File}", async () =>
            {
                await using var dec = await runtime.ModelSource.OpenAsync(decoder.Repo, decoder.File);
                await using var enc = await runtime.ModelSource.OpenAsync(encoder.Repo, encoder.File);
                return new StandaloneTaesd(vaeName, await SdTaesd.LoadAsync(runtime.Accelerator, dec, decoder.File, enc, encoder.File, scale));
            });
            return new VaeHandle(tae);
        }
        var (repo, file) = ModelFolders.Locate(ModelFolders.Vae, vaeName);
        context.Status($"Loading {vaeName}");
        var vae = await runtime.GetOrLoadAsync("vae:" + repo + "/" + file, async () =>
            new StandaloneVae(vaeName, await SdVae.LoadAsync(runtime.Accelerator, runtime.ModelSource, repo, file)));
        return new VaeHandle(vae);
    }
}

[Node(Category = "model/loaders", DisplayName = "Load CLIP")]
public sealed class CLIPLoader
{
    public static string[] ClipNames() => ModelFolders.Names(ModelFolders.TextEncoders);
    public static string[] Types() => ["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream",
        "chroma", "ace", "omnigen2", "qwen_image", "hunyuan_image", "flux2", "ovis", "longcat_image", "cogvideox", "lens", "pixeldit", "ideogram4", "boogu", "krea2",
        "joyimage", "mage", "minimax", "yue2", "flux3", "moss_image", "fibo2"];
    public static string[] Devices() => ["default", "cpu"];

    /// <summary>
    /// One text encoder alone from <c>&lt;models&gt;/text_encoders</c>, type stable_diffusion: CLIP ViT-L (SD 1.x) or OpenCLIP ViT-H (SD 2.x), told apart by width,
    /// in Hugging Face or OpenCLIP names. <c>device</c> is accepted for workflow compatibility; the encoder runs on the GPU either way.
    /// </summary>
    public async Task<ClipHandle> ExecuteAsync([Input(OptionsProvider = nameof(ClipNames))] string clipName, [Input(OptionsProvider = nameof(Types))] string type,
        NodeContext context, IServiceProvider services, [Input(OptionsProvider = nameof(Devices), Optional = true)] string device = "default")
    {
        if (type != "stable_diffusion") throw new NotSupportedException($"Load CLIP type {type}: SpawnGen loads stable_diffusion text encoders (SD 1.x / 2.x).");
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var (repo, file) = ModelFolders.Locate(ModelFolders.TextEncoders, clipName);
        context.Status($"Loading {clipName}");
        var encoder = await runtime.GetOrLoadAsync("clip:" + repo + "/" + file, async () =>
        {
            var tokenizer = await Checkpoint.LoadClipTokenizerAsync(runtime.ModelSource, Checkpoint.ClipTokenizerRepo, "", null);
            return new StandaloneTextEncoder(clipName, await SdTextEncoder.LoadAsync(runtime.Accelerator, runtime.ModelSource, repo, file), tokenizer, runtime.ModelSource);
        });
        return new ClipHandle(encoder);
    }
}

[Node(Category = "model/loaders", DisplayName = "Load CLIP (Dual)")]
public sealed class DualCLIPLoader
{
    public static string[] ClipNames() => ModelFolders.Names(ModelFolders.TextEncoders);
    public static string[] Types() => ["sdxl", "sd3", "flux", "hunyuan_video", "hidream", "hunyuan_image", "hunyuan_video_15", "kandinsky5", "kandinsky5_image", "ltxv", "newbie", "ace"];
    public static string[] Devices() => ["default", "cpu"];

    /// <summary>
    /// SDXL's two text encoders from two files of <c>&lt;models&gt;/text_encoders</c> (CLIP-L and OpenCLIP bigG, in either order: they are told apart by width).
    /// <c>device</c> is accepted for workflow compatibility; the encoders run on the GPU either way.
    /// </summary>
    public async Task<ClipHandle> ExecuteAsync([Input(OptionsProvider = nameof(ClipNames))] string clipName1, [Input(OptionsProvider = nameof(ClipNames))] string clipName2,
        [Input(OptionsProvider = nameof(Types))] string type, NodeContext context, IServiceProvider services,
        [Input(OptionsProvider = nameof(Devices), Optional = true)] string device = "default")
    {
        if (type != "sdxl") throw new NotSupportedException($"Load CLIP (Dual) type {type}: SpawnGen loads the sdxl pair (CLIP-L + OpenCLIP bigG).");
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var first = ModelFolders.Locate(ModelFolders.TextEncoders, clipName1);
        var second = ModelFolders.Locate(ModelFolders.TextEncoders, clipName2);
        context.Status($"Loading {clipName1} + {clipName2}");
        var encoder = await runtime.GetOrLoadAsync($"clip2:{first.Repo}/{first.File}|{second.Repo}/{second.File}", async () =>
        {
            // ComfyUI takes the two files in either order; the bigG one is the second tower.
            bool firstIsG = await SafetensorsHeader.HasWidthAsync(runtime.ModelSource, first.Repo, first.File, 1280);
            var (l, g) = firstIsG ? (second, first) : (first, second);
            var tokenizer = await Checkpoint.LoadClipTokenizerAsync(runtime.ModelSource, Checkpoint.ClipTokenizerRepo, "", null);
            await using var ls = await runtime.ModelSource.OpenAsync(l.Repo, l.File);
            await using var gs = await runtime.ModelSource.OpenAsync(g.Repo, g.File);
            var te = await SdTextEncoder.LoadSdxlAsync(runtime.Accelerator, ls, $"{l.Repo}/{l.File}", gs, $"{g.Repo}/{g.File}");
            return new StandaloneTextEncoder($"{clipName1} + {clipName2}", te, tokenizer, runtime.ModelSource);
        });
        return new ClipHandle(encoder);
    }
}

/// <summary>Reads only the JSON header of a safetensors file (its tensor names and shapes).</summary>
internal static class SafetensorsHeader
{
    /// <summary>True when the file's final layer norm (CLIP's <c>final_layer_norm</c>, OpenCLIP's <c>ln_final</c>) is <paramref name="width"/> wide.</summary>
    public static async Task<bool> HasWidthAsync(SpawnDev.ILGPU.ML.Hub.IModelSource source, string repo, string file, int width)
    {
        await using var s = await source.OpenAsync(repo, file);
        var len = new byte[8];
        await s.ReadExactlyAsync(len);
        long n = BitConverter.ToInt64(len);
        if (n <= 0 || n > 100_000_000) throw new InvalidDataException($"'{file}' is not a safetensors file.");
        var header = new byte[n];
        await s.ReadExactlyAsync(header);
        using var doc = System.Text.Json.JsonDocument.Parse(header);
        foreach (var t in doc.RootElement.EnumerateObject())
            if ((t.Name.EndsWith("final_layer_norm.weight", StringComparison.Ordinal) || t.Name.EndsWith("ln_final.weight", StringComparison.Ordinal))
                && t.Value.TryGetProperty("shape", out var shape) && shape.GetArrayLength() == 1)
                return shape[0].GetInt32() == width;
        throw new InvalidDataException($"'{file}' has no CLIP final layer norm: it is not a text encoder.");
    }
}
