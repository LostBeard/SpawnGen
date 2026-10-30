using System.Runtime.CompilerServices;
using SpawnDev.ILGPU.ML.StableDiffusion;

namespace SpawnGen.Nodes.ML;

/// <summary>
/// One step in making a UNet's or text encoder's weights, in graph order: a LoRA (<see cref="LoraUse"/>) or a merge with another model (<see cref="WeightMerge"/>).
/// A MODEL or CLIP carries its steps; they are put into the loaded weights when it samples or encodes, so the handles that come into a node never change
/// (ComfyUI's model.clone() and its patches).
/// </summary>
public abstract record WeightStep
{
    /// <summary>What the step does, as text: two step lists with the same keys make the same weights.</summary>
    internal abstract string Key { get; }

    internal static string KeyOf(IReadOnlyList<WeightStep> steps) => string.Join(";", steps.Select(s => s.Key));
}

/// <summary>One LoRA a model or CLIP carries: its file on the model source and how strongly it is merged.</summary>
public sealed record LoraUse(string Repo, string File, float Strength) : WeightStep
{
    internal override string Key => $"lora:{Repo}/{File}:{Strength:R}";
}

/// <summary>
/// A merge with another model's weights as they are at this point of the graph (ComfyUI's ModelMerge* / CLIPMerge* nodes): <see cref="Other"/> is the other
/// UNet or text encoder with the steps it carries; <see cref="Factors"/> gives each tensor's factors (<see cref="SdMerge"/>), <see cref="Name"/> says which
/// merge and settings it is, for the key.
/// </summary>
public sealed record WeightMerge(object Other, IReadOnlyList<WeightStep> OtherSteps, string Name, Func<string, SdMergeFactors?> Factors) : WeightStep
{
    static readonly ConditionalWeakTable<object, object> Ids = new();
    static int _next;

    internal override string Key => $"merge:{Name}:{Id(Other)}[{KeyOf(OtherSteps)}]";

    static string Id(object o) => (string)Ids.GetValue(o, _ => "m" + Interlocked.Increment(ref _next));
}

/// <summary>
/// The steps put into one loaded UNet or text encoder. They are put in again only when they change: a set of LoRAs alone is replaced in place (the library restores
/// the original weights first); with merges the weights go back to the file's and every step is replayed in order, since a merge reads the weights as they are
/// (LoRAs before it become part of them, LoRAs after it stay replaceable). Nodes run one at a time, so a sampling run or an encode always sees the steps it asked for.
/// </summary>
internal sealed class ComponentWeights(SpawnDev.ILGPU.ML.Hub.IModelSource source, string what,
    Func<IReadOnlyList<SdLoraSpec>, CancellationToken, Task<SdLoraReport>> applyLoras,
    Func<WeightMerge, CancellationToken, Task> merge, Func<Task> reset, bool clip) : IDisposable
{
    readonly SemaphoreSlim _gate = new(1, 1);
    string _applied = "";
    bool _appliedMerges;
    volatile bool _busy;

    /// <summary>True while steps are being put in (a merge whose other model is this one, in another state, cannot be done).</summary>
    public bool Busy => _busy;

    public async Task UseAsync(IReadOnlyList<WeightStep> steps, CancellationToken cancellationToken)
    {
        // Nodes run one at a time, so a call while steps are going in comes from a merge inside them: the same loaded model would be needed in two states.
        if (_busy)
            throw new NotSupportedException("This merge needs the same loaded model in two different states (a merge chain leads back to it); load the file a second time.");
        await _gate.WaitAsync(cancellationToken);
        _busy = true;
        try
        {
            string key = WeightStep.KeyOf(steps);
            if (key == _applied) return;
            _applied = "?"; // unknown until the steps are in: a failed step is redone next time
            bool merges = steps.Any(s => s is WeightMerge);
            if (merges || _appliedMerges) await reset();
            var loras = new List<LoraUse>();
            foreach (var step in steps)
            {
                if (step is LoraUse l) { loras.Add(l); continue; }
                if (loras.Count > 0) await ApplyLorasAsync(loras, cancellationToken);
                loras.Clear();
                await merge((WeightMerge)step, cancellationToken);
            }
            if (loras.Count > 0 || !merges) await ApplyLorasAsync(loras, cancellationToken); // with no merges an empty set removes the LoRAs merged before
            _applied = key;
            _appliedMerges = merges;
        }
        finally
        {
            _busy = false;
            _gate.Release();
        }
    }

    async Task ApplyLorasAsync(List<LoraUse> loras, CancellationToken cancellationToken)
    {
        // One entry per file, its uses summed; this component reads Strength (a UNet) or ClipStrength (a text encoder).
        var specs = loras.GroupBy(l => (l.Repo, l.File)).Select(g =>
        {
            float s = g.Sum(l => l.Strength);
            return SdLoraSpec.From(source, g.Key.Repo, g.Key.File, clip ? 0f : s, clip ? s : 0f);
        }).ToList();
        var report = await applyLoras(specs, cancellationToken);
        // ComfyUI warns about LoRA keys that match nothing (a LoRA made for another model family) and goes on; so does SpawnGen. The library reports the other
        // component's keys as unmatched too, so those are left out.
        foreach (var (lora, _, unmatched) in report.Loras)
        {
            var missed = unmatched.Where(k => !OtherPart(k)).ToList();
            if (missed.Count > 0) Console.WriteLine($"LoRA {lora}: {missed.Count} keys matched nothing in this {what} (first: {missed[0]})");
        }
    }

    static readonly string[] TextEncoderKeys = ["lora_te", "text_encoder", "te_", "te1", "te2", "cond_stage_model", "conditioner", "clip_l", "clip_g"];
    static readonly string[] UNetKeys = ["lora_unet", "unet.", "diffusion_model", "model.diffusion_model"];

    /// <summary>Whether a LoRA key names the other component (a UNet key while merging into a text encoder, and the other way round).</summary>
    bool OtherPart(string key) => (clip ? UNetKeys : TextEncoderKeys).Any(p => key.StartsWith(p, StringComparison.Ordinal));

    public void Dispose() => _gate.Dispose();

    /// <summary>The steps for a UNet.</summary>
    public static ComponentWeights For(SdUNet unet, SpawnDev.ILGPU.ML.Hub.IModelSource source) => new(source, "UNet", unet.ApplyLorasAsync,
        async (m, ct) =>
        {
            if (m.Other is not IDiffusionModel other || other.UNet is not { } otherUNet)
                throw new NotSupportedException("A model merges only with a model loaded from .safetensors (a checkpoint or Load Diffusion Model).");
            if (ReferenceEquals(otherUNet, unet))
                throw new NotSupportedException("Both models of this merge are the same loaded file in different states (for example with and without a LoRA); load the file a second time to merge them.");
            await other.UseWeightsAsync(m.OtherSteps, ct);
            await unet.MergeAsync(otherUNet, m.Factors, restorable: true, ct);
        }, unet.ResetAsync, clip: false);

    /// <summary>The steps for a text encoder (one tower or SDXL's two).</summary>
    public static ComponentWeights For(SdTextEncoder encoder, SpawnDev.ILGPU.ML.Hub.IModelSource source) => new(source, "text encoder", encoder.ApplyLorasAsync,
        async (m, ct) =>
        {
            if (m.Other is not ITextEncoderModel other || other.TextEncoder is not { } otherEncoder)
                throw new NotSupportedException("A CLIP merges only with a CLIP loaded from .safetensors (a checkpoint or Load CLIP).");
            if (ReferenceEquals(otherEncoder, encoder))
                throw new NotSupportedException("Both CLIPs of this merge are the same loaded file in different states (for example with and without a LoRA); load the file a second time to merge them.");
            await other.UseWeightsAsync(m.OtherSteps, ct);
            await encoder.MergeAsync(otherEncoder, m.Factors, restorable: true, ct);
        }, encoder.ResetAsync, clip: true);
}
