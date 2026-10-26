using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

/// <summary>A loaded ControlNet (SD 1.x, ComfyUI's models/controlnet).</summary>
[Socket("CONTROL_NET")]
public sealed class ControlNetHandle(string name, SdControlNet net)
{
    public string Name { get; } = name;
    public SdControlNet Net { get; } = net;
}

/// <summary>
/// A ControlNet attached to a conditioning entry (ComfyUI's cond['control']): the net, its control picture, strength and the part of the run it is active for, chained
/// onto the control that was there before (<see cref="Previous"/>: several ControlNets add up).
/// </summary>
public sealed record ControlUse(ControlNetHandle ControlNet, ImageBatch Image, float Strength, float StartPercent = 0f, float EndPercent = 1f, ControlUse? Previous = null)
{
    /// <summary>The controls of the chain active at <paramref name="sigma"/> (ComfyUI's get_control: the previous ones first, each skipped outside its percent range).</summary>
    internal IEnumerable<ControlUse> Active(double sigma, ModelSampling sampling)
    {
        if (Previous is not null)
            foreach (var c in Previous.Active(sigma, sampling)) yield return c;
        if (sigma <= sampling.PercentToSigma(StartPercent) && sigma >= sampling.PercentToSigma(EndPercent)) yield return this;
    }

    /// <summary>The control picture for picture <paramref name="index"/> of a batch, resized to the latent's size in pixels as ComfyUI does (nearest-exact, centre crop), [3, H, W].</summary>
    internal float[] Hint(int index, int height, int width)
    {
        var one = new ImageBatch(1, Image.Height, Image.Width, Image.Channels);
        int per = Image.Height * Image.Width * Image.Channels;
        Array.Copy(Image.Data, (index % Image.Batch) * per, one.Data, 0, per);
        var sized = one.Width == width && one.Height == height ? one : new ImageScale().Execute(one, "nearest-exact", width, height, "center");
        var chw = new float[3 * height * width];
        int plane = height * width;
        for (int p = 0; p < plane; p++)
            for (int c = 0; c < 3; c++)
                chw[c * plane + p] = sized.Data[p * sized.Channels + Math.Min(c, sized.Channels - 1)];
        return chw;
    }
}

/// <summary>One ControlNet to run for one UNet call: the net, its hint for this picture ([3, 8h, 8w] in 0..1) and its strength.</summary>
public sealed record ControlCall(SdControlNet Net, Func<float[]> Hint, object HintKey, float Strength);

[Node(Category = "model/loaders", DisplayName = "Load ControlNet Model")]
public sealed class ControlNetLoader
{
    public static string[] ControlNetNames() => ModelFolders.Names(ModelFolders.ControlNet);

    /// <summary>A ControlNet from <c>&lt;models&gt;/controlnet</c> (SD 1.x ControlNet v1.0 / v1.1 in ComfyUI's single-file layout, e.g. control_v11p_sd15_canny_fp16).</summary>
    public async Task<ControlNetHandle> ExecuteAsync([Input(OptionsProvider = nameof(ControlNetNames))] string controlNetName, NodeContext context, IServiceProvider services)
    {
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var (repo, file) = ModelFolders.Locate(ModelFolders.ControlNet, controlNetName);
        context.Status($"Loading {controlNetName}");
        var net = await runtime.GetOrLoadAsync("controlnet:" + repo + "/" + file, () => SdControlNet.LoadAsync(runtime.Accelerator, runtime.ModelSource, repo, file));
        return new ControlNetHandle(controlNetName, net);
    }
}

[Node(Category = "model/conditioning/controlnet", DisplayName = "Apply ControlNet", OutputNames = ["positive", "negative"])]
public sealed class ControlNetApplyAdvanced
{
    /// <summary>
    /// Steers both conditionings with a ControlNet and its control picture (a canny edge map, a depth map, ...) at <c>strength</c>, from <c>start_percent</c> to
    /// <c>end_percent</c> of the run. A ControlNet already on an entry stays and the two add up, as in ComfyUI. <c>vae</c> is for ControlNets that take a latent hint
    /// (none of the SD 1.x ones do).
    /// </summary>
    public (Conditioning Positive, Conditioning Negative) Execute(Conditioning positive, Conditioning negative, ControlNetHandle controlNet, ImageBatch image,
        [Input(Min = 0, Max = 10, Step = 0.01, Default = 1.0)] double strength,
        [Input(Min = 0, Max = 1, Step = 0.001, Default = 0.0)] double startPercent,
        [Input(Min = 0, Max = 1, Step = 0.001, Default = 1.0)] double endPercent,
        [Input(Optional = true)] VaeHandle? vae = null)
    {
        if (strength == 0) return (positive, negative);
        // One control per previous control, shared by every entry that had it (ComfyUI's cnets dict).
        var made = new Dictionary<object, ControlUse>();
        ControlUse For(ControlUse? previous)
        {
            object key = previous ?? (object)made;
            if (!made.TryGetValue(key, out var use))
                made[key] = use = new ControlUse(controlNet, image, (float)strength, (float)startPercent, (float)endPercent, previous);
            return use;
        }
        Conditioning Apply(Conditioning c) => c.Map(e => e with { Control = For(e.Control), ControlApplyToUncond = false });
        return (Apply(positive), Apply(negative));
    }
}

[Node(Category = "model/conditioning/controlnet", DisplayName = "Apply ControlNet (DEPRECATED)")]
public sealed class ControlNetApply
{
    /// <summary>
    /// ComfyUI's old Apply ControlNet: steers one conditioning for the whole run; at sampling time the negative gets the same control where it has none (ComfyUI's
    /// control_apply_to_uncond). Use Apply ControlNet in new workflows.
    /// </summary>
    public Conditioning Execute(Conditioning conditioning, ControlNetHandle controlNet, ImageBatch image,
        [Input(Min = 0, Max = 10, Step = 0.01, Default = 1.0)] double strength)
    {
        if (strength == 0) return conditioning;
        return conditioning.Map(e => e with { Control = new ControlUse(controlNet, image, (float)strength, Previous: e.Control), ControlApplyToUncond = true });
    }
}

internal static class ControlFill
{
    /// <summary>
    /// ComfyUI's apply_empty_x_to_equal_area for 'control': the controls of the positive entries marked apply-to-uncond (and without an area) go onto the other
    /// conditioning's entries without an area and without a control, round-robin, unless that conditioning already has a control.
    /// </summary>
    public static Conditioning Fill(Conditioning positive, Conditioning other)
    {
        var condControls = positive.Entries.Where(e => e.ControlApplyToUncond && e.Area is null && e.Control is not null).Select(e => e.Control!).ToList();
        if (condControls.Count == 0) return other;
        var entries = other.Entries.ToList();
        var withoutArea = Enumerable.Range(0, entries.Count).Where(i => entries[i].Area is null).ToList();
        if (withoutArea.Any(i => entries[i].Control is not null)) return other;
        var originals = withoutArea.Select(i => (Entry: entries[i], Index: i)).ToList();
        if (originals.Count == 0) return other;
        for (int x = 0; x < condControls.Count; x++)
        {
            var (o, index) = originals[x % originals.Count];
            entries[index] = o with { Control = condControls[x] };
        }
        return new Conditioning(entries);
    }
}
