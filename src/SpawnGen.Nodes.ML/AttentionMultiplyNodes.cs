using System.Globalization;
using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries for the attention experiments and SpawnDev.ILGPU.ML's SdMerge factor
// functions (UNetAttentionMultiply, UNetTemporalAttentionMultiply, CLIPAttentionMultiply, as its docs describe them). The scaling is a weight step on
// the handle, replayed in graph order with LoRAs and merges. Checked against ComfyUI's renders (ComfyUIRenderParityTests).

static class AttentionMultiply
{
    static string F(double v) => v.ToString("R", CultureInfo.InvariantCulture);

    /// <summary>A scaling step: the matching tensors times their factor (a WeightMerge without another model; the factor is its Self part).</summary>
    public static WeightMerge Step(string name, Func<string, float?> scale) =>
        new(null, [], name, n => scale(n) is { } s ? new SdMergeFactors(s, 0f) : null);

    public static ModelHandle Scale(ModelHandle model, string name, Func<string, float?> factors) =>
        new(model.Model, model.Sampling) { Patches = model.Patches, Weights = [.. model.Weights, Step(name, factors)] };

    public static string Name(string node, params double[] factors) => $"{node} {string.Join(" ", factors.Select(F))}";
}

[Node("UNetSelfAttentionMultiply", Category = "experimental/attention_experiments", DisplayName = "UNetSelfAttentionMultiply")]
public sealed class UNetSelfAttentionMultiply
{
    /// <summary>Every UNet self-attention (attn1) projection's weight and bias scaled: q, k, v and the output.</summary>
    public ModelHandle Execute(ModelHandle model,
        [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double q = 1.0, [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double k = 1.0,
        [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double v = 1.0, [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double @out = 1.0) =>
        AttentionMultiply.Scale(model, AttentionMultiply.Name("self attention", q, k, v, @out),
            SdMerge.UNetAttentionMultiply("attn1", (float)q, (float)k, (float)v, (float)@out));
}

[Node("UNetCrossAttentionMultiply", Category = "experimental/attention_experiments", DisplayName = "UNetCrossAttentionMultiply")]
public sealed class UNetCrossAttentionMultiply
{
    /// <summary>Every UNet cross-attention (attn2) projection's weight and bias scaled: q, k, v and the output.</summary>
    public ModelHandle Execute(ModelHandle model,
        [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double q = 1.0, [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double k = 1.0,
        [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double v = 1.0, [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double @out = 1.0) =>
        AttentionMultiply.Scale(model, AttentionMultiply.Name("cross attention", q, k, v, @out),
            SdMerge.UNetAttentionMultiply("attn2", (float)q, (float)k, (float)v, (float)@out));
}

[Node("UNetTemporalAttentionMultiply", Category = "experimental/attention_experiments", DisplayName = "UNetTemporalAttentionMultiply")]
public sealed class UNetTemporalAttentionMultiply
{
    /// <summary>The attention outputs scaled, separately for the temporal blocks of a video UNet (only the structural factors apply to SD 1.x / SDXL).</summary>
    public ModelHandle Execute(ModelHandle model,
        [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double selfStructural = 1.0, [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double selfTemporal = 1.0,
        [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double crossStructural = 1.0, [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double crossTemporal = 1.0) =>
        AttentionMultiply.Scale(model, AttentionMultiply.Name("temporal attention", selfStructural, selfTemporal, crossStructural, crossTemporal),
            SdMerge.UNetTemporalAttentionMultiply((float)selfStructural, (float)selfTemporal, (float)crossStructural, (float)crossTemporal));
}

[Node("CLIPAttentionMultiply", Category = "experimental/attention_experiments", DisplayName = "CLIPAttentionMultiply")]
public sealed class CLIPAttentionMultiply
{
    /// <summary>Every text encoder layer's self-attention projection's weight and bias scaled: q, k, v and the output.</summary>
    public ClipHandle Execute(ClipHandle clip,
        [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double q = 1.0, [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double k = 1.0,
        [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double v = 1.0, [Input(Min = 0.0, Max = 10.0, Step = 0.01)] double @out = 1.0) =>
        new(clip.Encoder)
        {
            Layer = clip.Layer,
            Weights = [.. clip.Weights, AttentionMultiply.Step(AttentionMultiply.Name("clip attention", q, k, v, @out),
                SdMerge.CLIPAttentionMultiply((float)q, (float)k, (float)v, (float)@out))],
        };
}
