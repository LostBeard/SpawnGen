using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries and its documented ControlNet behaviour (a control applies to the
// conditionings it is attached to, chained controls add up, each active within its start / end percent), with SpawnDev.ILGPU.ML's SdControlNet
// doing the network (SdUNetRun takes the hint as 3 x H x W in 0..1).

/// <summary>CONTROL_NET: a loaded ControlNet.</summary>
[Socket("CONTROL_NET")]
public sealed class ControlNetHandle(string name, SdControlNet net)
{
    public string Name { get; } = name;
    public SdControlNet Net { get; } = net;
}

/// <summary>A ControlNet attached to a conditioning entry: its picture, strength and part of the run, after the controls attached before it.</summary>
public sealed record ControlUse(ControlNetHandle ControlNet, ImageBatch Image, float Strength, float StartPercent = 0f, float EndPercent = 1f, ControlUse? Previous = null)
{
    /// <summary>The controls of the chain active at <paramref name="sigma"/> (start sigma &gt;= sigma &gt;= end sigma), the earliest attached first.</summary>
    public IEnumerable<ControlUse> Active(double sigma, ModelSampling sampling)
    {
        var chain = new List<ControlUse>();
        for (var u = this; u is not null; u = u.Previous) chain.Add(u);
        chain.Reverse();
        foreach (var u in chain)
            if (sigma <= sampling.PercentToSigma(u.StartPercent) && sigma >= sampling.PercentToSigma(u.EndPercent)) yield return u;
    }

    /// <summary>
    /// The control picture for picture <paramref name="index"/> of the batch (the control batch repeats when shorter), scaled to fill height x width
    /// (nearest-exact, centred crop) as 3 x height x width values in 0..1.
    /// </summary>
    public float[] Hint(int index, int height, int width)
    {
        var frame = Frame(Image, index % Image.Batch);
        var sized = CommonUpscale.Resize(frame, width, height, "nearest-exact", "center");
        int plane = height * width, c = sized.Channels;
        var r = new float[3 * plane];
        for (int p = 0; p < plane; p++)
            for (int k = 0; k < 3; k++) r[k * plane + p] = sized.Data[p * c + Math.Min(k, c - 1)];
        return r;
    }

    static ImageBatch Frame(ImageBatch image, int b)
    {
        if (image.Batch == 1) return image;
        int size = image.Height * image.Width * image.Channels;
        return new ImageBatch(1, image.Height, image.Width, image.Channels, image.Data.AsSpan(b * size, size).ToArray());
    }
}

/// <summary>One ControlNet evaluation in a sampling step: the net, its hint (made on first use), the key the run caches the hint under, the strength.</summary>
public sealed record ControlCall(SdControlNet Net, Func<float[]> Hint, object HintKey, float Strength);

/// <summary>Controls meant for the unconditional pass too (Apply ControlNet's old form): copied onto the other conditioning's entries that have none.</summary>
internal static class ControlFill
{
    public static Conditioning Fill(Conditioning positive, Conditioning other)
    {
        var control = positive.Entries.FirstOrDefault(e => e.Control is not null && e.ControlApplyToUncond)?.Control;
        return control is null ? other : other.Map(e => e.Control is null ? e with { Control = control } : e);
    }
}

[Node("ControlNetLoader", Category = "model/loaders", DisplayName = "Load ControlNet Model")]
public sealed class ControlNetLoader
{
    public static string[] ControlNetNames() => ModelFolders.Names(ModelFolders.ControlNet);

    public async Task<ControlNetHandle> ExecuteAsync([Input(OptionsProvider = nameof(ControlNetNames))] string controlNetName, NodeContext context, IServiceProvider services)
    {
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var (repo, file) = ModelFolders.Locate(ModelFolders.ControlNet, controlNetName);
        return await runtime.GetOrLoadAsync($"controlnet:{repo}/{file}", async () =>
            new LoadedControlNet(new ControlNetHandle(controlNetName, await SdControlNet.LoadAsync(runtime.Accelerator, runtime.ModelSource, repo, file, context.CancellationToken))))
            .ContinueWith(t => t.Result.Handle, TaskContinuationOptions.ExecuteSynchronously);
    }

    /// <summary>The runtime's cache holds disposables; this disposes the net with the runtime.</summary>
    sealed class LoadedControlNet(ControlNetHandle handle) : IDisposable
    {
        public ControlNetHandle Handle { get; } = handle;
        public void Dispose() => Handle.Net.Dispose();
    }
}

[Node("ControlNetApply", Category = "model/conditioning/controlnet", DisplayName = "Apply ControlNet (DEPRECATED)")]
public sealed class ControlNetApply
{
    /// <summary>The ControlNet attached to every entry (after any already there), also applied to the unconditional pass; strength 0 changes nothing.</summary>
    public Conditioning Execute(Conditioning conditioning, ControlNetHandle controlNet, ImageBatch image,
        [Input(Min = 0.0, Max = 10.0, Step = 0.01, Default = 1.0)] double strength)
    {
        if (strength == 0) return conditioning;
        return conditioning.Map(e => e with { Control = new ControlUse(controlNet, image, (float)strength, 0f, 1f, e.Control), ControlApplyToUncond = true });
    }
}

[Node("ControlNetApplyAdvanced", Category = "model/conditioning/controlnet", DisplayName = "Apply ControlNet", OutputNames = ["positive", "negative"])]
public sealed class ControlNetApplyAdvanced
{
    /// <summary>
    /// The ControlNet attached to the entries of both conditionings within start / end percent of the run, after any control already there (entries
    /// that share a previous control share the new one); strength 0 changes nothing.
    /// </summary>
    public (Conditioning Positive, Conditioning Negative) Execute(Conditioning positive, Conditioning negative, ControlNetHandle controlNet, ImageBatch image,
        [Input(Min = 0.0, Max = 10.0, Step = 0.01, Default = 1.0)] double strength,
        [Input(Min = 0.0, Max = 1.0, Step = 0.001, Default = 0.0)] double startPercent,
        [Input(Min = 0.0, Max = 1.0, Step = 0.001, Default = 1.0)] double endPercent,
        [Input(Optional = true)] VaeHandle? vae = null)
    {
        if (strength == 0) return (positive, negative);
        var made = new Dictionary<object, ControlUse>(ReferenceEqualityComparer.Instance);
        var none = new object();
        ControlUse For(ControlUse? previous)
        {
            var key = (object?)previous ?? none;
            if (!made.TryGetValue(key, out var use))
                made[key] = use = new ControlUse(controlNet, image, (float)strength, (float)startPercent, (float)endPercent, previous);
            return use;
        }
        return (positive.Map(e => e with { Control = For(e.Control) }), negative.Map(e => e with { Control = For(e.Control) }));
    }
}
