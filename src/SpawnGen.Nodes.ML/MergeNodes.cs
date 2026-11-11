using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnGen.Engine.Nodes;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries (names, the per-block keys of the model-specific merges, defaults) and
// SpawnDev.ILGPU.ML's SdMerge factor functions, which state ComfyUI's merge formulas; saving goes through the library's SaveAsync methods.

/// <summary>Merging another model's or CLIP's weights into a copy of a handle (a weight step; the tensors change when a run needs them).</summary>
internal static class Merging
{
    public static ModelHandle Merge(ModelHandle model1, ModelHandle model2, string name, Func<string, SdMergeFactors?> factors) =>
        new(model1.Model, model1.Sampling) { Patches = model1.Patches, Weights = [.. model1.Weights, new WeightMerge(model2.Model, model2.Weights, name, factors)] };

    public static ClipHandle Merge(ClipHandle clip1, ClipHandle clip2, string name, Func<string, SdMergeFactors?> factors) =>
        new(clip1.Encoder) { Layer = clip1.Layer, Weights = [.. clip1.Weights, new WeightMerge(clip2.Encoder, clip2.Weights, name, factors)] };

    /// <summary>Each tensor takes the ratio of the longest key it starts with (SdMerge.Blocks).</summary>
    public static ModelHandle Blocks(ModelHandle model1, ModelHandle model2, (string Key, double Ratio)[] ratios) =>
        Merge(model1, model2, "blocks " + string.Join(",", ratios.Select(r => $"{r.Key}={r.Ratio:R}")),
            SdMerge.Blocks([.. ratios.Select(r => new KeyValuePair<string, float>(r.Key, (float)r.Ratio))]));
}

[Node("ModelMergeSimple", Category = "model/merging", DisplayName = "ModelMergeSimple")]
public sealed class ModelMergeSimple
{
    /// <summary>W = W1 * ratio + W2 * (1 - ratio).</summary>
    public ModelHandle Execute(ModelHandle model1, ModelHandle model2, [Input(Min = 0.0, Max = 1.0, Step = 0.01)] double ratio = 1.0) =>
        Merging.Merge(model1, model2, $"simple {ratio:R}", SdMerge.Simple((float)ratio));
}

[Node("ModelMergeAdd", Category = "model/merging", DisplayName = "ModelMergeAdd")]
public sealed class ModelMergeAdd
{
    /// <summary>W = W1 + W2.</summary>
    public ModelHandle Execute(ModelHandle model1, ModelHandle model2) => Merging.Merge(model1, model2, "add", SdMerge.Add());
}

[Node("ModelMergeSubtract", Category = "model/merging", DisplayName = "ModelMergeSubtract")]
public sealed class ModelMergeSubtract
{
    /// <summary>W = (W1 - W2) * multiplier.</summary>
    public ModelHandle Execute(ModelHandle model1, ModelHandle model2, [Input(Min = -10.0, Max = 10.0, Step = 0.01)] double multiplier = 1.0) =>
        Merging.Merge(model1, model2, $"subtract {multiplier:R}", SdMerge.Subtract((float)multiplier));
}

[Node("ModelMergeBlocks", Category = "model/merging", DisplayName = "ModelMergeBlocks")]
public sealed class ModelMergeBlocks
{
    public ModelHandle Execute(ModelHandle model1, ModelHandle model2,
        [Input(Min = 0.0, Max = 1.0, Step = 0.01)] double input = 1.0,
        [Input(Min = 0.0, Max = 1.0, Step = 0.01)] double middle = 1.0,
        [Input(Min = 0.0, Max = 1.0, Step = 0.01)] double @out = 1.0) =>
        Merging.Blocks(model1, model2, [("input", input), ("middle", middle), ("out", @out)]);
}

/// <summary>The model-specific merges' block keys, in object_info's order.</summary>
static class MergeKeys
{
    public static readonly string[] Sd =
    [
        "time_embed.", "label_emb.",
        "input_blocks.0.", "input_blocks.1.", "input_blocks.2.", "input_blocks.3.", "input_blocks.4.", "input_blocks.5.",
        "input_blocks.6.", "input_blocks.7.", "input_blocks.8.", "input_blocks.9.", "input_blocks.10.", "input_blocks.11.",
        "middle_block.0.", "middle_block.1.", "middle_block.2.",
        "output_blocks.0.", "output_blocks.1.", "output_blocks.2.", "output_blocks.3.", "output_blocks.4.", "output_blocks.5.",
        "output_blocks.6.", "output_blocks.7.", "output_blocks.8.", "output_blocks.9.", "output_blocks.10.", "output_blocks.11.",
        "out.",
    ];

    public static readonly string[] Sdxl =
    [
        "time_embed.", "label_emb.",
        "input_blocks.0", "input_blocks.1", "input_blocks.2", "input_blocks.3", "input_blocks.4", "input_blocks.5", "input_blocks.6", "input_blocks.7", "input_blocks.8",
        "middle_block.0", "middle_block.1", "middle_block.2",
        "output_blocks.0", "output_blocks.1", "output_blocks.2", "output_blocks.3", "output_blocks.4", "output_blocks.5", "output_blocks.6", "output_blocks.7", "output_blocks.8",
        "out.",
    ];

    public static (string, double)[] Pair(string[] keys, double[] ratios) => [.. keys.Select((k, i) => (k, ratios[i]))];
}

[Node("ModelMergeSD1", Category = "model/merging/model specific", DisplayName = "ModelMergeSD1")]
public sealed class ModelMergeSD1
{
    public ModelHandle Execute(ModelHandle model1, ModelHandle model2,
        [Input(Name = "time_embed.", Min = 0.0, Max = 1.0, Step = 0.01)] double b0 = 1, [Input(Name = "label_emb.", Min = 0.0, Max = 1.0, Step = 0.01)] double b1 = 1,
        [Input(Name = "input_blocks.0.", Min = 0.0, Max = 1.0, Step = 0.01)] double b2 = 1, [Input(Name = "input_blocks.1.", Min = 0.0, Max = 1.0, Step = 0.01)] double b3 = 1,
        [Input(Name = "input_blocks.2.", Min = 0.0, Max = 1.0, Step = 0.01)] double b4 = 1, [Input(Name = "input_blocks.3.", Min = 0.0, Max = 1.0, Step = 0.01)] double b5 = 1,
        [Input(Name = "input_blocks.4.", Min = 0.0, Max = 1.0, Step = 0.01)] double b6 = 1, [Input(Name = "input_blocks.5.", Min = 0.0, Max = 1.0, Step = 0.01)] double b7 = 1,
        [Input(Name = "input_blocks.6.", Min = 0.0, Max = 1.0, Step = 0.01)] double b8 = 1, [Input(Name = "input_blocks.7.", Min = 0.0, Max = 1.0, Step = 0.01)] double b9 = 1,
        [Input(Name = "input_blocks.8.", Min = 0.0, Max = 1.0, Step = 0.01)] double b10 = 1, [Input(Name = "input_blocks.9.", Min = 0.0, Max = 1.0, Step = 0.01)] double b11 = 1,
        [Input(Name = "input_blocks.10.", Min = 0.0, Max = 1.0, Step = 0.01)] double b12 = 1, [Input(Name = "input_blocks.11.", Min = 0.0, Max = 1.0, Step = 0.01)] double b13 = 1,
        [Input(Name = "middle_block.0.", Min = 0.0, Max = 1.0, Step = 0.01)] double b14 = 1, [Input(Name = "middle_block.1.", Min = 0.0, Max = 1.0, Step = 0.01)] double b15 = 1,
        [Input(Name = "middle_block.2.", Min = 0.0, Max = 1.0, Step = 0.01)] double b16 = 1, [Input(Name = "output_blocks.0.", Min = 0.0, Max = 1.0, Step = 0.01)] double b17 = 1,
        [Input(Name = "output_blocks.1.", Min = 0.0, Max = 1.0, Step = 0.01)] double b18 = 1, [Input(Name = "output_blocks.2.", Min = 0.0, Max = 1.0, Step = 0.01)] double b19 = 1,
        [Input(Name = "output_blocks.3.", Min = 0.0, Max = 1.0, Step = 0.01)] double b20 = 1, [Input(Name = "output_blocks.4.", Min = 0.0, Max = 1.0, Step = 0.01)] double b21 = 1,
        [Input(Name = "output_blocks.5.", Min = 0.0, Max = 1.0, Step = 0.01)] double b22 = 1, [Input(Name = "output_blocks.6.", Min = 0.0, Max = 1.0, Step = 0.01)] double b23 = 1,
        [Input(Name = "output_blocks.7.", Min = 0.0, Max = 1.0, Step = 0.01)] double b24 = 1, [Input(Name = "output_blocks.8.", Min = 0.0, Max = 1.0, Step = 0.01)] double b25 = 1,
        [Input(Name = "output_blocks.9.", Min = 0.0, Max = 1.0, Step = 0.01)] double b26 = 1, [Input(Name = "output_blocks.10.", Min = 0.0, Max = 1.0, Step = 0.01)] double b27 = 1,
        [Input(Name = "output_blocks.11.", Min = 0.0, Max = 1.0, Step = 0.01)] double b28 = 1, [Input(Name = "out.", Min = 0.0, Max = 1.0, Step = 0.01)] double b29 = 1) =>
        Merging.Blocks(model1, model2, MergeKeys.Pair(MergeKeys.Sd, [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24, b25, b26, b27, b28, b29]));
}

[Node("ModelMergeSD2", Category = "model/merging/model specific", DisplayName = "ModelMergeSD2")]
public sealed class ModelMergeSD2
{
    public ModelHandle Execute(ModelHandle model1, ModelHandle model2,
        [Input(Name = "time_embed.", Min = 0.0, Max = 1.0, Step = 0.01)] double b0 = 1, [Input(Name = "label_emb.", Min = 0.0, Max = 1.0, Step = 0.01)] double b1 = 1,
        [Input(Name = "input_blocks.0.", Min = 0.0, Max = 1.0, Step = 0.01)] double b2 = 1, [Input(Name = "input_blocks.1.", Min = 0.0, Max = 1.0, Step = 0.01)] double b3 = 1,
        [Input(Name = "input_blocks.2.", Min = 0.0, Max = 1.0, Step = 0.01)] double b4 = 1, [Input(Name = "input_blocks.3.", Min = 0.0, Max = 1.0, Step = 0.01)] double b5 = 1,
        [Input(Name = "input_blocks.4.", Min = 0.0, Max = 1.0, Step = 0.01)] double b6 = 1, [Input(Name = "input_blocks.5.", Min = 0.0, Max = 1.0, Step = 0.01)] double b7 = 1,
        [Input(Name = "input_blocks.6.", Min = 0.0, Max = 1.0, Step = 0.01)] double b8 = 1, [Input(Name = "input_blocks.7.", Min = 0.0, Max = 1.0, Step = 0.01)] double b9 = 1,
        [Input(Name = "input_blocks.8.", Min = 0.0, Max = 1.0, Step = 0.01)] double b10 = 1, [Input(Name = "input_blocks.9.", Min = 0.0, Max = 1.0, Step = 0.01)] double b11 = 1,
        [Input(Name = "input_blocks.10.", Min = 0.0, Max = 1.0, Step = 0.01)] double b12 = 1, [Input(Name = "input_blocks.11.", Min = 0.0, Max = 1.0, Step = 0.01)] double b13 = 1,
        [Input(Name = "middle_block.0.", Min = 0.0, Max = 1.0, Step = 0.01)] double b14 = 1, [Input(Name = "middle_block.1.", Min = 0.0, Max = 1.0, Step = 0.01)] double b15 = 1,
        [Input(Name = "middle_block.2.", Min = 0.0, Max = 1.0, Step = 0.01)] double b16 = 1, [Input(Name = "output_blocks.0.", Min = 0.0, Max = 1.0, Step = 0.01)] double b17 = 1,
        [Input(Name = "output_blocks.1.", Min = 0.0, Max = 1.0, Step = 0.01)] double b18 = 1, [Input(Name = "output_blocks.2.", Min = 0.0, Max = 1.0, Step = 0.01)] double b19 = 1,
        [Input(Name = "output_blocks.3.", Min = 0.0, Max = 1.0, Step = 0.01)] double b20 = 1, [Input(Name = "output_blocks.4.", Min = 0.0, Max = 1.0, Step = 0.01)] double b21 = 1,
        [Input(Name = "output_blocks.5.", Min = 0.0, Max = 1.0, Step = 0.01)] double b22 = 1, [Input(Name = "output_blocks.6.", Min = 0.0, Max = 1.0, Step = 0.01)] double b23 = 1,
        [Input(Name = "output_blocks.7.", Min = 0.0, Max = 1.0, Step = 0.01)] double b24 = 1, [Input(Name = "output_blocks.8.", Min = 0.0, Max = 1.0, Step = 0.01)] double b25 = 1,
        [Input(Name = "output_blocks.9.", Min = 0.0, Max = 1.0, Step = 0.01)] double b26 = 1, [Input(Name = "output_blocks.10.", Min = 0.0, Max = 1.0, Step = 0.01)] double b27 = 1,
        [Input(Name = "output_blocks.11.", Min = 0.0, Max = 1.0, Step = 0.01)] double b28 = 1, [Input(Name = "out.", Min = 0.0, Max = 1.0, Step = 0.01)] double b29 = 1) =>
        Merging.Blocks(model1, model2, MergeKeys.Pair(MergeKeys.Sd, [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, b24, b25, b26, b27, b28, b29]));
}

[Node("ModelMergeSDXL", Category = "model/merging/model specific", DisplayName = "ModelMergeSDXL")]
public sealed class ModelMergeSDXL
{
    public ModelHandle Execute(ModelHandle model1, ModelHandle model2,
        [Input(Name = "time_embed.", Min = 0.0, Max = 1.0, Step = 0.01)] double b0 = 1, [Input(Name = "label_emb.", Min = 0.0, Max = 1.0, Step = 0.01)] double b1 = 1,
        [Input(Name = "input_blocks.0", Min = 0.0, Max = 1.0, Step = 0.01)] double b2 = 1, [Input(Name = "input_blocks.1", Min = 0.0, Max = 1.0, Step = 0.01)] double b3 = 1,
        [Input(Name = "input_blocks.2", Min = 0.0, Max = 1.0, Step = 0.01)] double b4 = 1, [Input(Name = "input_blocks.3", Min = 0.0, Max = 1.0, Step = 0.01)] double b5 = 1,
        [Input(Name = "input_blocks.4", Min = 0.0, Max = 1.0, Step = 0.01)] double b6 = 1, [Input(Name = "input_blocks.5", Min = 0.0, Max = 1.0, Step = 0.01)] double b7 = 1,
        [Input(Name = "input_blocks.6", Min = 0.0, Max = 1.0, Step = 0.01)] double b8 = 1, [Input(Name = "input_blocks.7", Min = 0.0, Max = 1.0, Step = 0.01)] double b9 = 1,
        [Input(Name = "input_blocks.8", Min = 0.0, Max = 1.0, Step = 0.01)] double b10 = 1, [Input(Name = "middle_block.0", Min = 0.0, Max = 1.0, Step = 0.01)] double b11 = 1,
        [Input(Name = "middle_block.1", Min = 0.0, Max = 1.0, Step = 0.01)] double b12 = 1, [Input(Name = "middle_block.2", Min = 0.0, Max = 1.0, Step = 0.01)] double b13 = 1,
        [Input(Name = "output_blocks.0", Min = 0.0, Max = 1.0, Step = 0.01)] double b14 = 1, [Input(Name = "output_blocks.1", Min = 0.0, Max = 1.0, Step = 0.01)] double b15 = 1,
        [Input(Name = "output_blocks.2", Min = 0.0, Max = 1.0, Step = 0.01)] double b16 = 1, [Input(Name = "output_blocks.3", Min = 0.0, Max = 1.0, Step = 0.01)] double b17 = 1,
        [Input(Name = "output_blocks.4", Min = 0.0, Max = 1.0, Step = 0.01)] double b18 = 1, [Input(Name = "output_blocks.5", Min = 0.0, Max = 1.0, Step = 0.01)] double b19 = 1,
        [Input(Name = "output_blocks.6", Min = 0.0, Max = 1.0, Step = 0.01)] double b20 = 1, [Input(Name = "output_blocks.7", Min = 0.0, Max = 1.0, Step = 0.01)] double b21 = 1,
        [Input(Name = "output_blocks.8", Min = 0.0, Max = 1.0, Step = 0.01)] double b22 = 1, [Input(Name = "out.", Min = 0.0, Max = 1.0, Step = 0.01)] double b23 = 1) =>
        Merging.Blocks(model1, model2, MergeKeys.Pair(MergeKeys.Sdxl, [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23]));
}

[Node("CLIPMergeSimple", Category = "model/merging", DisplayName = "CLIPMergeSimple")]
public sealed class CLIPMergeSimple
{
    public ClipHandle Execute(ClipHandle clip1, ClipHandle clip2, [Input(Min = 0.0, Max = 1.0, Step = 0.01)] double ratio = 1.0) =>
        Merging.Merge(clip1, clip2, $"simple {ratio:R}", SdMerge.Simple((float)ratio));
}

[Node("CLIPMergeAdd", Category = "model/merging", DisplayName = "CLIPMergeAdd")]
public sealed class CLIPMergeAdd
{
    public ClipHandle Execute(ClipHandle clip1, ClipHandle clip2) => Merging.Merge(clip1, clip2, "add", SdMerge.Add());
}

[Node("CLIPMergeSubtract", Category = "model/merging", DisplayName = "CLIPMergeSubtract")]
public sealed class CLIPMergeSubtract
{
    public ClipHandle Execute(ClipHandle clip1, ClipHandle clip2, [Input(Min = -10.0, Max = 10.0, Step = 0.01)] double multiplier = 1.0) =>
        Merging.Merge(clip1, clip2, $"subtract {multiplier:R}", SdMerge.Subtract((float)multiplier));
}

/// <summary>Writing models as .safetensors files in the output folder.</summary>
internal static class ModelSaving
{
    /// <summary>
    /// The header metadata of a saved model: the prompt and each extra_pnginfo entry (the workflow) as JSON text. ComfyUI also writes Stability's
    /// modelspec keys; SpawnGen does not (docs/PARITY.md).
    /// </summary>
    public static Dictionary<string, string> Metadata(NodeContext? context, SdFamily? family, string title)
    {
        var metadata = new Dictionary<string, string>();
        if (context is null) return metadata;
        metadata["prompt"] = context.PromptJson;
        foreach (var (key, value) in context.ExtraPngInfo) metadata[key] = value;
        return metadata;
    }

    /// <summary>Writes a file named like Save Image's (<c>{prefix}_00001_.safetensors</c>, a "folder/" part a subfolder of the output folder).</summary>
    public static async Task<UiOutput> SaveAsync(IServiceProvider services, string prefix, Func<Stream, Task> write)
    {
        var store = services.GetService(typeof(IResultStore)) as IResultStore
            ?? new DiskResultStore(services.GetService(typeof(OutputOptions)) as OutputOptions ?? new OutputOptions());
        await store.SaveStreamAsync("output", SavePrefix.ComputeVars(prefix, 0, 0, DateTime.Now), "safetensors", write);
        return new UiOutput();
    }
}

[Node("ModelSave", Category = "model/merging", DisplayName = "ModelSave", IsOutput = true)]
public sealed class ModelSave
{
    /// <summary>The diffusion model's current weights (its LoRAs and merges applied) as a UNet-only file.</summary>
    public async Task<UiOutput> ExecuteAsync(ModelHandle model, NodeContext context, IServiceProvider services, string filenamePrefix = "diffusion_models/ComfyUI")
    {
        var unet = model.Model.UNet ?? throw new NotSupportedException($"{model.Model.Name} has no UNet SpawnGen can save (an ONNX export).");
        await model.Model.UseWeightsAsync(model.Weights, context.CancellationToken);
        var options = new SdSaveOptions { Metadata = ModelSaving.Metadata(context, model.Model.Family, "model") };
        return await ModelSaving.SaveAsync(services, filenamePrefix, s => unet.SaveAsync(s, options, context.CancellationToken));
    }
}

[Node("CheckpointSave", Category = "model/merging", DisplayName = "Save Checkpoint", IsOutput = true)]
public sealed class CheckpointSave
{
    /// <summary>
    /// The model, CLIP and VAE as one single-file checkpoint, with the model's and the CLIP's weight steps applied. All three must come from the same
    /// loaded single-file checkpoint (SpawnGen writes the loaded file's parts back).
    /// </summary>
    public async Task<UiOutput> ExecuteAsync(ModelHandle model, ClipHandle clip, VaeHandle vae, NodeContext context, IServiceProvider services,
        string filenamePrefix = "checkpoints/ComfyUI")
    {
        if (model.Model is not Checkpoint { File: { } file } checkpoint)
            throw new NotSupportedException("Save Checkpoint needs a model from a single-file checkpoint (Load Checkpoint with a .safetensors file).");
        if (!ReferenceEquals(clip.Encoder, checkpoint) || !ReferenceEquals(vae.Vae, checkpoint))
            throw new NotSupportedException("Save Checkpoint writes back one loaded checkpoint: the CLIP and the VAE must come from the same Load Checkpoint as the model.");
        await ((IDiffusionModel)checkpoint).UseWeightsAsync(model.Weights, context.CancellationToken);
        await ((ITextEncoderModel)checkpoint).UseWeightsAsync(clip.Weights, context.CancellationToken);
        var options = new SdSaveOptions { Metadata = ModelSaving.Metadata(context, checkpoint.Family, "checkpoint") };
        return await ModelSaving.SaveAsync(services, filenamePrefix, s => file.SaveAsync(s, options, context.CancellationToken));
    }
}

[Node("VAESave", Category = "model/merging", DisplayName = "VAESave", IsOutput = true)]
public sealed class VAESave
{
    public async Task<UiOutput> ExecuteAsync(VaeHandle vae, NodeContext context, IServiceProvider services, string filenamePrefix = "vae/ComfyUI_vae")
    {
        var sdVae = vae.Vae.SdVae ?? throw new NotSupportedException($"{vae.Vae.Name} cannot be saved as a VAE file (a tiny VAE or an ONNX export).");
        var options = new SdSaveOptions { Metadata = ModelSaving.Metadata(context, null, "vae") };
        return await ModelSaving.SaveAsync(services, filenamePrefix, s => sdVae.SaveAsync(s, options, context.CancellationToken));
    }
}
