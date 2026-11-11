using System.Globalization;
using System.Runtime.CompilerServices;
using SpawnDev.ILGPU.ML.Hub;
using SpawnDev.ILGPU.ML.StableDiffusion;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md): SpawnGen's own design over SpawnDev.ILGPU.ML (ApplyLorasAsync, MergeAsync, ScaleAsync, ResetAsync, as its
// docs describe them), with the handle semantics LoraNodeTests pins.

/// <summary>
/// One change to a model's weights, kept on the handles as a list so that loading a LoRA or merging never touches the shared model until a run
/// needs those weights (<see cref="ComponentWeights"/>). Steps with the same <see cref="Key"/> make the same weights.
/// </summary>
public abstract record WeightStep
{
    public abstract string Key { get; }

    /// <summary>The key of a list of steps, in order ("" for none: the file's own weights).</summary>
    public static string KeyOf(IReadOnlyList<WeightStep> steps) => string.Join(" | ", steps.Select(s => s.Key));
}

/// <summary>A LoRA file (on the model source under <paramref name="Repo"/>) merged at <paramref name="Strength"/>.</summary>
public sealed record LoraUse(string Repo, string File, float Strength) : WeightStep
{
    public override string Key => $"lora {Repo}/{File} @ {Strength.ToString("R", CultureInfo.InvariantCulture)}";
}

/// <summary>
/// Another model's weights merged in, tensor by tensor, with the factors <paramref name="Factors"/> gives (SdMerge); <paramref name="Other"/> is the
/// other <see cref="IDiffusionModel"/> or <see cref="ITextEncoderModel"/> and <paramref name="OtherSteps"/> the steps its own handle carried. With
/// no other model the factors' Self value scales this model's tensors.
/// </summary>
public sealed record WeightMerge(object? Other, IReadOnlyList<WeightStep> OtherSteps, string Name, Func<string, SdMergeFactors?> Factors) : WeightStep
{
    public override string Key => $"merge {Name} with #{Identity.Of(Other)} [{KeyOf(OtherSteps)}]";
}

/// <summary>A number per object, for keys that must tell two models apart.</summary>
static class Identity
{
    static readonly ConditionalWeakTable<object, object> Ids = new();
    static long _next;

    public static long Of(object? o) => o is null ? 0 : (long)Ids.GetValue(o, _ => Interlocked.Increment(ref _next));
}

/// <summary>
/// The weights a loaded component (a UNet, or text encoder(s)) currently holds, brought to a list of steps when a run asks: if the list's key
/// differs from what is on the device, the component is reset to its file's weights, the merges are applied in order, then every LoRA in one call.
/// </summary>
internal sealed class ComponentWeights(
    IModelSource source,
    string what,
    Func<IReadOnlyList<SdLoraSpec>, CancellationToken, Task<SdLoraReport>> applyLoras,
    Func<WeightMerge, CancellationToken, Task> merge,
    Func<Task> reset,
    bool clip) : IDisposable
{
    readonly SemaphoreSlim _gate = new(1, 1);
    string? _current = ""; // null: unknown (a change failed part-way), reset before anything else

    /// <summary>Whether the weights are being changed right now.</summary>
    public bool Busy { get; private set; }

    public async Task UseAsync(IReadOnlyList<WeightStep> steps, CancellationToken cancellationToken)
    {
        var key = WeightStep.KeyOf(steps);
        await _gate.WaitAsync(cancellationToken);
        try
        {
            if (key == _current) return;
            Busy = true;
            if (_current != "") await reset();
            _current = null;
            foreach (var step in steps)
                if (step is WeightMerge m) await merge(m, cancellationToken);
                else if (step is not LoraUse) throw new NotSupportedException($"{what} cannot apply a {step.GetType().Name}.");
            var loras = steps.OfType<LoraUse>()
                .Select(l => SdLoraSpec.From(source, l.Repo, l.File, clip ? 0f : l.Strength, clip ? l.Strength : 0f))
                .ToList();
            if (loras.Count > 0) await applyLoras(loras, cancellationToken);
            _current = key;
        }
        finally
        {
            Busy = false;
            _gate.Release();
        }
    }

    public static ComponentWeights For(SdUNet unet, IModelSource source) => new(source, "The UNet", unet.ApplyLorasAsync,
        async (m, ct) =>
        {
            if (m.Other is null) { await unet.ScaleAsync(n => m.Factors(n)?.Self, true, ct); return; }
            var other = m.Other as IDiffusionModel ?? throw new ArgumentException("A model merge needs another diffusion model.");
            await other.UseWeightsAsync(m.OtherSteps, ct);
            var otherUnet = other.UNet ?? throw new NotSupportedException($"{other.Name} has no UNet SpawnGen can merge (an ONNX export).");
            if (ReferenceEquals(otherUnet, unet)) throw new NotSupportedException("A model cannot be merged with itself; load the second copy separately.");
            await unet.MergeAsync(otherUnet, m.Factors, true, ct);
        },
        unet.ResetAsync, clip: false);

    public static ComponentWeights For(SdTextEncoder encoder, IModelSource source) => new(source, "The text encoder", encoder.ApplyLorasAsync,
        async (m, ct) =>
        {
            if (m.Other is null) { await encoder.ScaleAsync(n => m.Factors(n)?.Self, true, ct); return; }
            var other = m.Other as ITextEncoderModel ?? throw new ArgumentException("A CLIP merge needs another text encoder.");
            await other.UseWeightsAsync(m.OtherSteps, ct);
            var otherEncoder = other.TextEncoder ?? throw new NotSupportedException($"{other.Name} has no text encoder SpawnGen can merge (an ONNX export).");
            if (ReferenceEquals(otherEncoder, encoder)) throw new NotSupportedException("A CLIP cannot be merged with itself; load the second copy separately.");
            await encoder.MergeAsync(otherEncoder, m.Factors, true, ct);
        },
        encoder.ResetAsync, clip: true);

    public void Dispose() => _gate.Dispose();
}
