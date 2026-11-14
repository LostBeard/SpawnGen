using System.Buffers.Binary;
using System.Text.Json;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries (names, type lists, descriptions), its documented models folder layout
// (diffusion_models / unet, vae, text_encoders / clip, controlnet / t2i_adapter, vae_approx), ComponentLoaderTests, SpawnDev.ILGPU.ML's loaders
// (their docs) and the safetensors format (an 8-byte little-endian header length, then a JSON header of tensor shapes).

/// <summary>
/// The component files of the models folder, by kind: each kind reads its current folder and ComfyUI's older one, sub-folders included, and lists
/// the <c>.safetensors</c> files by their path there (with the platform's separator, as ComfyUI lists them).
/// </summary>
public static class ModelFolders
{
    public const string DiffusionModels = "diffusion_models";
    public const string Vae = "vae";
    public const string TextEncoders = "text_encoders";
    public const string ControlNet = "controlnet";
    public const string VaeApprox = "vae_approx";

    static readonly Dictionary<string, string[]> Folders = new()
    {
        [DiffusionModels] = ["diffusion_models", "unet"],
        [Vae] = ["vae"],
        [TextEncoders] = ["text_encoders", "clip"],
        [ControlNet] = ["controlnet", "t2i_adapter"],
        [VaeApprox] = ["vae_approx"],
    };

    static readonly Lock Gate = new();
    static readonly Dictionary<string, SortedDictionary<string, (string Repo, string File)>> Files =
        Folders.Keys.ToDictionary(k => k, _ => new SortedDictionary<string, (string, string)>(StringComparer.Ordinal));

    /// <summary>The repo id each folder's files are opened under ("local/vae" for &lt;models&gt;/vae), for the model source.</summary>
    public static IEnumerable<KeyValuePair<string, string>> RepoFolders =>
        Folders.Values.SelectMany(f => f).Select(f => new KeyValuePair<string, string>("local/" + f, f));

    /// <summary>Adds the component files under <paramref name="modelsFolder"/> that are not known yet; returns how many were added.</summary>
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
                        if (Files[kind].TryAdd(name, ("local/" + folder, name))) added++;
                }
            }
        return added;
    }

    /// <summary>The names of a kind's files, sorted.</summary>
    public static string[] Names(string kind)
    {
        lock (Gate) return [.. Files[kind].Keys];
    }

    /// <summary>Where a listed file is opened from: its folder's repo id and its path there.</summary>
    public static (string Repo, string File) Locate(string kind, string name)
    {
        lock (Gate)
            return Files[kind].TryGetValue(name, out var at) ? at
                : throw new ArgumentException($"'{name}' is not in the {Folders[kind][0]} folder of the models folder.", nameof(name));
    }
}

/// <summary>Reading a .safetensors file's header (the tensor names and shapes) without its data.</summary>
internal static class SafetensorsHeader
{
    /// <summary>The tensor names of the file.</summary>
    public static async Task<HashSet<string>> NamesAsync(IModelSource source, string repo, string file)
    {
        await using var stream = await source.OpenAsync(repo, file);
        var lengthBytes = new byte[8];
        await stream.ReadExactlyAsync(lengthBytes);
        long length = BinaryPrimitives.ReadInt64LittleEndian(lengthBytes);
        if (length <= 0 || length > 100_000_000) throw new InvalidDataException($"{file} is not a safetensors file.");
        var header = new byte[length];
        await stream.ReadExactlyAsync(header);
        using var doc = JsonDocument.Parse(header);
        return [.. doc.RootElement.EnumerateObject().Select(t => t.Name).Where(n => n != "__metadata__")];
    }

    /// <summary>Whether any tensor of the file has <paramref name="width"/> as its last dimension.</summary>
    public static async Task<bool> HasWidthAsync(IModelSource source, string repo, string file, int width)
    {
        await using var stream = await source.OpenAsync(repo, file);
        var lengthBytes = new byte[8];
        await stream.ReadExactlyAsync(lengthBytes);
        long length = BinaryPrimitives.ReadInt64LittleEndian(lengthBytes);
        if (length <= 0 || length > 100_000_000) throw new InvalidDataException($"{file} is not a safetensors file.");
        var header = new byte[length];
        await stream.ReadExactlyAsync(header);
        using var doc = JsonDocument.Parse(header);
        foreach (var tensor in doc.RootElement.EnumerateObject())
            if (tensor.Value.ValueKind == JsonValueKind.Object && tensor.Value.TryGetProperty("shape", out var shape)
                && shape.GetArrayLength() > 0 && shape[shape.GetArrayLength() - 1].GetInt64() == width)
                return true;
        return false;
    }
}

[Node("UNETLoader", Category = "model/loaders", DisplayName = "Load Diffusion Model")]
public sealed class UNETLoader
{
    /// <summary>The diffusion_models folder's files and the Wan 2.2 transformers the model host serves (downloaded on first use), sorted.</summary>
    public static string[] UnetNames() =>
        [.. ModelFolders.Names(ModelFolders.DiffusionModels).Union(WanFiles.DiffusionModels).Order(StringComparer.Ordinal)];

    public static string[] WeightDtypes() => ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"];

    /// <summary>A UNet-only file from diffusion_models (or unet), its prediction type read from its tensors. Weights are held as fp16 ("default" only).</summary>
    public async Task<ModelHandle> ExecuteAsync([Input(OptionsProvider = nameof(UnetNames))] string unetName,
        [Input(OptionsProvider = nameof(WeightDtypes))] string weightDtype, NodeContext context, IServiceProvider services)
    {
        if (weightDtype != "default")
            throw new NotSupportedException($"Weight dtype '{weightDtype}' is not available: SpawnGen holds diffusion model weights as fp16; choose 'default'.");
        var runtime = await GpuHelpers.RuntimeAsync(services);
        if (WanFiles.DiffusionModels.Contains(unetName))
            return new ModelHandle(await runtime.GetOrLoadAsync($"wan-dit:{WanFiles.Repo}/{unetName}", () => Task.FromResult(new WanDiffusionModel(runtime, unetName))));
        var (repo, file) = ModelFolders.Locate(ModelFolders.DiffusionModels, unetName);
        var names = await SafetensorsHeader.NamesAsync(runtime.ModelSource, repo, file);
        if (names.Contains(Flux2DiffusionModel.MarkerTensor))
            return new ModelHandle(await runtime.GetOrLoadAsync($"flux2:{repo}/{file}", () => Task.FromResult(new Flux2DiffusionModel(runtime, unetName, repo, file))));
        if (names.Contains(ZImageDiffusionModel.MarkerTensor))
            return new ModelHandle(await runtime.GetOrLoadAsync($"zimage:{repo}/{file}", () => Task.FromResult(new ZImageDiffusionModel(runtime, unetName, repo, file))));
        if (WanFiles.IsOriginalLayoutDit(names))
            throw new NotSupportedException($"'{unetName}' is a Wan transformer in its original layout (as ComfyUI's repackaged files ship it), which SpawnGen cannot load yet; pick {WanFiles.DiffusionModels[0]} (downloads on first use).");
        var model = await runtime.GetOrLoadAsync($"unet:{repo}/{file}", () => StandaloneDiffusionModel.LoadAsync(runtime, unetName, repo, file));
        return new ModelHandle(model);
    }
}

[Node("CLIPLoader", Category = "model/loaders", DisplayName = "Load CLIP",
    Description = "Recipes:\nsd: clip-l\nstable cascade: clip-g\nsd3: t5 xxl / clip-g / clip-l\nstable audio: t5 base\nmochi: t5 xxl\ncogvideox: t5 xxl (226-token padding)\ncosmos: old t5 xxl\nlumina2: gemma 2 2B\nwan: umt5 xxl\nhidream: llama-3.1 (Recommend) or t5\nomnigen2: qwen vl 2.5 3B\njoyimage: qwen3-vl 8B\nlens: gpt-oss-20b\npixeldit: gemma 2 2B elm\nminimax: MiniMax H3 Qwen3-VL or Music3 Qwen/RVQ\nflux3: qwen3-vl 4B\nfibo2: qwen3-vl 4B")]
public sealed class CLIPLoader
{
    /// <summary>The text_encoders folder's files and Wan 2.2's umT5 from the model host (downloaded on first use), sorted.</summary>
    public static string[] ClipNames() =>
        [.. ModelFolders.Names(ModelFolders.TextEncoders).Append(WanFiles.TextEncoder).Distinct().Order(StringComparer.Ordinal)];

    /// <summary>ComfyUI's full list, so workflows that name another type open; SpawnGen runs "stable_diffusion" and "wan".</summary>
    public static string[] Types() =>
    [
        "stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace",
        "omnigen2", "qwen_image", "hunyuan_image", "flux2", "ovis", "longcat_image", "cogvideox", "lens", "pixeldit", "ideogram4", "boogu", "krea2",
        "joyimage", "mage", "minimax", "yue2", "flux3", "moss_image", "fibo2",
    ];

    public static string[] Devices() => ["default", "cpu"];

    /// <summary>
    /// One text encoder file (CLIP-L for SD 1.x, OpenCLIP-H for SD 2.x; the family comes from its width), with the CLIP tokenizer; or, type "wan",
    /// Wan 2.2's umT5-XXL from the model host.
    /// </summary>
    public async Task<ClipHandle> ExecuteAsync([Input(OptionsProvider = nameof(ClipNames))] string clipName,
        [Input(OptionsProvider = nameof(Types))] string type, NodeContext context, IServiceProvider services,
        [Input(OptionsProvider = nameof(Devices), Optional = true)] string device = "default")
    {
        if (type == "wan")
        {
            if (clipName != WanFiles.TextEncoder)
                throw new NotSupportedException($"'{clipName}': SpawnGen runs Wan's umT5 from the library's export so far; pick {WanFiles.TextEncoder} (downloads on first use).");
            var rt = await GpuHelpers.RuntimeAsync(services);
            return new ClipHandle(await rt.GetOrLoadAsync($"wan-umt5:{WanFiles.Repo}/{clipName}", () => Task.FromResult(new WanTextEncoderModel(rt, clipName))));
        }
        if (type == "flux2")
        {
            var rt = await GpuHelpers.RuntimeAsync(services);
            var (qwenRepo, qwenFile) = ModelFolders.Locate(ModelFolders.TextEncoders, clipName);
            return new ClipHandle(await rt.GetOrLoadAsync($"flux2-qwen:{qwenRepo}/{qwenFile}", () => Task.FromResult(new Flux2TextEncoderModel(rt, clipName, qwenRepo, qwenFile))));
        }
        if (type == "lumina2")
        {
            // Z-Image (ComfyUI's Z-Image templates load qwen_3_4b with type lumina2).
            var rt = await GpuHelpers.RuntimeAsync(services);
            var (qwenRepo, qwenFile) = ModelFolders.Locate(ModelFolders.TextEncoders, clipName);
            return new ClipHandle(await rt.GetOrLoadAsync($"zimage-qwen:{qwenRepo}/{qwenFile}", () => Task.FromResult(new ZImageTextEncoderModel(rt, clipName, qwenRepo, qwenFile))));
        }
        if (type != "stable_diffusion")
            throw new NotSupportedException($"CLIP type '{type}' is not available: SpawnGen runs the Stable Diffusion text encoders, Wan's umT5, FLUX.2's Qwen3 and Z-Image's Qwen3; choose 'stable_diffusion', 'wan', 'flux2' or 'lumina2'.");
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var (repo, file) = ModelFolders.Locate(ModelFolders.TextEncoders, clipName);
        var model = await runtime.GetOrLoadAsync($"clip:{repo}/{file}", async () =>
        {
            var encoder = await SdTextEncoder.LoadAsync(runtime.Accelerator, runtime.ModelSource, repo, file);
            try
            {
                var tokenizer = await Checkpoint.LoadClipTokenizerAsync(runtime.ModelSource, Checkpoint.ClipTokenizerRepo, "", null);
                return new StandaloneTextEncoder(clipName, encoder, tokenizer, runtime.ModelSource);
            }
            catch { encoder.Dispose(); throw; }
        });
        return new ClipHandle(model);
    }
}

[Node("DualCLIPLoader", Category = "model/loaders", DisplayName = "Load CLIP (Dual)",
    Description = "Recipes:\nsdxl: clip-l, clip-g\nsd3: clip-l, clip-g / clip-l, t5 / clip-g, t5\nflux: clip-l, t5\nhidream: at least one of t5 or llama, recommended t5 and llama\nhunyuan_image: qwen2.5vl 7b and byt5 small\nnewbie: gemma-3-4b-it, jina clip v2")]
public sealed class DualCLIPLoader
{
    public static string[] ClipNames() => ModelFolders.Names(ModelFolders.TextEncoders);

    public static string[] Types() =>
        ["sdxl", "sd3", "flux", "hunyuan_video", "hidream", "hunyuan_image", "hunyuan_video_15", "kandinsky5", "kandinsky5_image", "ltxv", "newbie", "ace"];

    public static string[] Devices() => CLIPLoader.Devices();

    /// <summary>SDXL's two text encoders (CLIP-L and OpenCLIP-bigG, in either order: the bigG file is the one with 1280-wide tensors).</summary>
    public async Task<ClipHandle> ExecuteAsync([Input(OptionsProvider = nameof(ClipNames))] string clipName1, [Input(OptionsProvider = nameof(ClipNames))] string clipName2,
        [Input(OptionsProvider = nameof(Types))] string type, NodeContext context, IServiceProvider services,
        [Input(OptionsProvider = nameof(Devices), Optional = true)] string device = "default")
    {
        if (type != "sdxl")
            throw new NotSupportedException($"Dual CLIP type '{type}' is not available: SpawnGen runs SDXL's two text encoders; choose 'sdxl'.");
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var first = ModelFolders.Locate(ModelFolders.TextEncoders, clipName1);
        var second = ModelFolders.Locate(ModelFolders.TextEncoders, clipName2);
        var model = await runtime.GetOrLoadAsync($"clip2:{first.Repo}/{first.File}|{second.Repo}/{second.File}", async () =>
        {
            var source = runtime.ModelSource;
            var (l, g) = await SafetensorsHeader.HasWidthAsync(source, first.Repo, first.File, 1280) ? (second, first) : (first, second);
            SdTextEncoder encoder;
            await using (var ls = await source.OpenAsync(l.Repo, l.File))
            await using (var gs = await source.OpenAsync(g.Repo, g.File))
                encoder = await SdTextEncoder.LoadSdxlAsync(runtime.Accelerator, ls, l.File, gs, g.File);
            try
            {
                var tokenizer = await Checkpoint.LoadClipTokenizerAsync(source, Checkpoint.ClipTokenizerRepo, "", null);
                return new StandaloneTextEncoder($"{clipName1} + {clipName2}", encoder, tokenizer, source);
            }
            catch { encoder.Dispose(); throw; }
        });
        return new ClipHandle(model);
    }
}

[Node("VAELoader", Category = "model/loaders", DisplayName = "Load VAE")]
public sealed class VAELoader
{
    /// <summary>The tiny VAEs Load VAE offers when their decoder is in vae_approx: name, decoder file name, latent scale.</summary>
    static readonly (string Name, string Decoder, string Encoder, float Scale)[] Tiny =
    [
        ("taesd", "taesd_decoder", "taesd_encoder", 0.18215f),
        ("taesdxl", "taesdxl_decoder", "taesdxl_encoder", 0.13025f),
    ];

    static string? Approx(string stem) =>
        ModelFolders.Names(ModelFolders.VaeApprox).FirstOrDefault(n => Path.GetFileNameWithoutExtension(n) == stem);

    /// <summary>The vae folder's files, then the tiny VAEs whose decoder is there, then "pixel_space" (as ComfyUI lists them).</summary>
    public static string[] VaeNames() =>
        [.. ModelFolders.Names(ModelFolders.Vae).Append(WanFiles.Vae).Distinct().Order(StringComparer.Ordinal),
         .. Tiny.Where(t => Approx(t.Decoder) is not null).Select(t => t.Name), "pixel_space"];

    public async Task<VaeHandle> ExecuteAsync([Input(OptionsProvider = nameof(VaeNames))] string vaeName, NodeContext context, IServiceProvider services)
    {
        if (vaeName == "pixel_space") throw new NotSupportedException("'pixel_space' is for pixel-space models, which SpawnGen does not run.");
        var runtime = await GpuHelpers.RuntimeAsync(services);
        if (vaeName == WanFiles.Vae)
            return new VaeHandle(await runtime.GetOrLoadAsync($"wan-vae:{WanFiles.Repo}/{vaeName}", () => Task.FromResult(new WanVaeModel(runtime, vaeName))));
        if (Tiny.FirstOrDefault(t => t.Name == vaeName) is { Name: not null } tiny)
        {
            var decoder = Approx(tiny.Decoder) ?? throw new ArgumentException($"There is no {tiny.Decoder} in the vae_approx folder.", nameof(vaeName));
            var (repo, decoderFile) = ModelFolders.Locate(ModelFolders.VaeApprox, decoder);
            var encoderFile = Approx(tiny.Encoder) is { } e ? ModelFolders.Locate(ModelFolders.VaeApprox, e).File : null;
            var tae = await runtime.GetOrLoadAsync($"taesd:{repo}/{decoderFile}|{encoderFile}", async () =>
                new StandaloneTaesd(vaeName, await SdTaesd.LoadAsync(runtime.Accelerator, runtime.ModelSource, repo, decoderFile, encoderFile!, tiny.Scale)));
            return new VaeHandle(tae);
        }
        var (vaeRepo, file) = ModelFolders.Locate(ModelFolders.Vae, vaeName);
        var vae = await runtime.GetOrLoadAsync<IDisposable>($"vae:{vaeRepo}/{file}", async () =>
        {
            var sd = await SdVae.LoadAsync(runtime.Accelerator, runtime.ModelSource, vaeRepo, file);
            // FLUX.2's VAE (32 latent channels): its LATENT is ComfyUI's patchified, batch-normalised [128, H/16, W/16].
            return sd.LatentChannels == 32
                ? await Flux2VaeModel.CreateAsync(vaeName, sd, runtime.Accelerator, runtime.ModelSource, vaeRepo, file, context.CancellationToken)
                : new StandaloneVae(vaeName, sd);
        });
        return new VaeHandle((IVaeModel)vae);
    }
}
