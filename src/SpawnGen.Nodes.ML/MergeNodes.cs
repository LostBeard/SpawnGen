using SpawnDev.ILGPU.ML.StableDiffusion;
using SpawnGen.Engine.Nodes;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

// ComfyUI's model merging nodes (comfy_extras/nodes_model_merging*.py). A merge adds a step to a copy of the first model: when it samples, the second model's
// weights as they are at that point of the graph (its own LoRAs and merges included) are blended in on the GPU by SpawnDev.ILGPU.ML (SdMerge), so the
// models that come in never change. Merging works on models loaded from .safetensors (checkpoints, Load Diffusion Model, Load CLIP).

/// <summary>Builds the merge steps the nodes add.</summary>
internal static class Merging
{
    public static ModelHandle Merge(ModelHandle model1, ModelHandle model2, string name, Func<string, SdMergeFactors?> factors) =>
        new(model1.Model, model1.Sampling)
        {
            Patches = model1.Patches,
            Weights = [.. model1.Weights, new WeightMerge(model2.Model, model2.Weights, name, factors)],
        };

    public static ClipHandle Merge(ClipHandle clip1, ClipHandle clip2, string name, Func<string, SdMergeFactors?> factors) =>
        new(clip1.Encoder)
        {
            Layer = clip1.Layer,
            Weights = [.. clip1.Weights, new WeightMerge(clip2.Encoder, clip2.Weights, name, factors)],
        };

    /// <summary>ComfyUI's ModelMergeBlocks: each tensor takes the ratio of the longest block prefix its name starts with, the first block's when none does.</summary>
    public static ModelHandle Blocks(ModelHandle model1, ModelHandle model2, (string Block, double Ratio)[] ratios) =>
        Merge(model1, model2, "blocks " + string.Join(",", ratios.Select(r => $"{r.Block}={r.Ratio:R}")),
            SdMerge.Blocks([.. ratios.Select(r => new KeyValuePair<string, float>(r.Block, (float)r.Ratio))]));
}

[Node("ModelMergeSimple", Category = "model/merging", DisplayName = "ModelMergeSimple")]
public sealed class ModelMergeSimple
{
    /// <summary>model1 x ratio + model2 x (1 - ratio), tensor by tensor.</summary>
    public ModelHandle Execute(ModelHandle model1, ModelHandle model2, [Input(Min = 0, Max = 1, Step = 0.01)] double ratio = 1.0) =>
        Merging.Merge(model1, model2, $"simple {ratio:R}", SdMerge.Simple((float)ratio));
}

[Node("ModelMergeBlocks", Category = "model/merging", DisplayName = "ModelMergeBlocks")]
public sealed class ModelMergeBlocks
{
    /// <summary>Like ModelMergeSimple, with a ratio for the input blocks, the middle block and the output side (out also matches output_blocks, as in ComfyUI).</summary>
    public ModelHandle Execute(ModelHandle model1, ModelHandle model2,
        [Input(Min = 0, Max = 1, Step = 0.01)] double input = 1.0, [Input(Min = 0, Max = 1, Step = 0.01)] double middle = 1.0,
        [Input(Min = 0, Max = 1, Step = 0.01)] double @out = 1.0) =>
        Merging.Blocks(model1, model2, [("input", input), ("middle", middle), ("out", @out)]);
}

[Node("ModelMergeSubtract", Category = "model/merging", DisplayName = "ModelMergeSubtract")]
public sealed class ModelMergeSubtract
{
    /// <summary>(model1 - model2) x multiplier, tensor by tensor (ComfyUI's add_patches(-multiplier, multiplier)): with model1 a fine-tune of model2, its difference.</summary>
    public ModelHandle Execute(ModelHandle model1, ModelHandle model2, [Input(Min = -10, Max = 10, Step = 0.01)] double multiplier = 1.0) =>
        Merging.Merge(model1, model2, $"subtract {multiplier:R}", SdMerge.Subtract((float)multiplier));
}

[Node("ModelMergeAdd", Category = "model/merging", DisplayName = "ModelMergeAdd")]
public sealed class ModelMergeAdd
{
    /// <summary>model1 + model2, tensor by tensor (adds a difference made by ModelMergeSubtract).</summary>
    public ModelHandle Execute(ModelHandle model1, ModelHandle model2) => Merging.Merge(model1, model2, "add", SdMerge.Add());
}

[Node("CLIPMergeSimple", Category = "model/merging", DisplayName = "CLIPMergeSimple")]
public sealed class CLIPMergeSimple
{
    /// <summary>clip1 x ratio + clip2 x (1 - ratio), tensor by tensor; clip1's layer setting is kept.</summary>
    public ClipHandle Execute(ClipHandle clip1, ClipHandle clip2, [Input(Min = 0, Max = 1, Step = 0.01)] double ratio = 1.0) =>
        Merging.Merge(clip1, clip2, $"simple {ratio:R}", SdMerge.Simple((float)ratio));
}

[Node("CLIPMergeSubtract", Category = "model/merging", DisplayName = "CLIPMergeSubtract")]
public sealed class CLIPMergeSubtract
{
    /// <summary>(clip1 - clip2) x multiplier, tensor by tensor.</summary>
    public ClipHandle Execute(ClipHandle clip1, ClipHandle clip2, [Input(Min = -10, Max = 10, Step = 0.01)] double multiplier = 1.0) =>
        Merging.Merge(clip1, clip2, $"subtract {multiplier:R}", SdMerge.Subtract((float)multiplier));
}

[Node("CLIPMergeAdd", Category = "model/merging", DisplayName = "CLIPMergeAdd")]
public sealed class CLIPMergeAdd
{
    /// <summary>clip1 + clip2, tensor by tensor.</summary>
    public ClipHandle Execute(ClipHandle clip1, ClipHandle clip2) => Merging.Merge(clip1, clip2, "add", SdMerge.Add());
}

[Node("ModelMergeSD1", Category = "model/merging/model specific", DisplayName = "ModelMergeSD1")]
public sealed class ModelMergeSD1
{
    /// <summary>SD 1.x / 2.x block by block: one ratio per input, middle and output block (ComfyUI's ModelMergeSD1; ModelMergeSD2 is the same node).</summary>
    public ModelHandle Execute(ModelHandle model1, ModelHandle model2,
        [Input(Name = "time_embed.", Min = 0, Max = 1, Step = 0.01)] double b0 = 1.0,
        [Input(Name = "label_emb.", Min = 0, Max = 1, Step = 0.01)] double b1 = 1.0,
        [Input(Name = "input_blocks.0.", Min = 0, Max = 1, Step = 0.01)] double b2 = 1.0,
        [Input(Name = "input_blocks.1.", Min = 0, Max = 1, Step = 0.01)] double b3 = 1.0,
        [Input(Name = "input_blocks.2.", Min = 0, Max = 1, Step = 0.01)] double b4 = 1.0,
        [Input(Name = "input_blocks.3.", Min = 0, Max = 1, Step = 0.01)] double b5 = 1.0,
        [Input(Name = "input_blocks.4.", Min = 0, Max = 1, Step = 0.01)] double b6 = 1.0,
        [Input(Name = "input_blocks.5.", Min = 0, Max = 1, Step = 0.01)] double b7 = 1.0,
        [Input(Name = "input_blocks.6.", Min = 0, Max = 1, Step = 0.01)] double b8 = 1.0,
        [Input(Name = "input_blocks.7.", Min = 0, Max = 1, Step = 0.01)] double b9 = 1.0,
        [Input(Name = "input_blocks.8.", Min = 0, Max = 1, Step = 0.01)] double b10 = 1.0,
        [Input(Name = "input_blocks.9.", Min = 0, Max = 1, Step = 0.01)] double b11 = 1.0,
        [Input(Name = "input_blocks.10.", Min = 0, Max = 1, Step = 0.01)] double b12 = 1.0,
        [Input(Name = "input_blocks.11.", Min = 0, Max = 1, Step = 0.01)] double b13 = 1.0,
        [Input(Name = "middle_block.0.", Min = 0, Max = 1, Step = 0.01)] double b14 = 1.0,
        [Input(Name = "middle_block.1.", Min = 0, Max = 1, Step = 0.01)] double b15 = 1.0,
        [Input(Name = "middle_block.2.", Min = 0, Max = 1, Step = 0.01)] double b16 = 1.0,
        [Input(Name = "output_blocks.0.", Min = 0, Max = 1, Step = 0.01)] double b17 = 1.0,
        [Input(Name = "output_blocks.1.", Min = 0, Max = 1, Step = 0.01)] double b18 = 1.0,
        [Input(Name = "output_blocks.2.", Min = 0, Max = 1, Step = 0.01)] double b19 = 1.0,
        [Input(Name = "output_blocks.3.", Min = 0, Max = 1, Step = 0.01)] double b20 = 1.0,
        [Input(Name = "output_blocks.4.", Min = 0, Max = 1, Step = 0.01)] double b21 = 1.0,
        [Input(Name = "output_blocks.5.", Min = 0, Max = 1, Step = 0.01)] double b22 = 1.0,
        [Input(Name = "output_blocks.6.", Min = 0, Max = 1, Step = 0.01)] double b23 = 1.0,
        [Input(Name = "output_blocks.7.", Min = 0, Max = 1, Step = 0.01)] double b24 = 1.0,
        [Input(Name = "output_blocks.8.", Min = 0, Max = 1, Step = 0.01)] double b25 = 1.0,
        [Input(Name = "output_blocks.9.", Min = 0, Max = 1, Step = 0.01)] double b26 = 1.0,
        [Input(Name = "output_blocks.10.", Min = 0, Max = 1, Step = 0.01)] double b27 = 1.0,
        [Input(Name = "output_blocks.11.", Min = 0, Max = 1, Step = 0.01)] double b28 = 1.0,
        [Input(Name = "out.", Min = 0, Max = 1, Step = 0.01)] double b29 = 1.0) =>
        Merging.Blocks(model1, model2, [("time_embed.", b0), ("label_emb.", b1), ("input_blocks.0.", b2), ("input_blocks.1.", b3), ("input_blocks.2.", b4), ("input_blocks.3.", b5), ("input_blocks.4.", b6), ("input_blocks.5.", b7), ("input_blocks.6.", b8), ("input_blocks.7.", b9), ("input_blocks.8.", b10), ("input_blocks.9.", b11), ("input_blocks.10.", b12), ("input_blocks.11.", b13), ("middle_block.0.", b14), ("middle_block.1.", b15), ("middle_block.2.", b16), ("output_blocks.0.", b17), ("output_blocks.1.", b18), ("output_blocks.2.", b19), ("output_blocks.3.", b20), ("output_blocks.4.", b21), ("output_blocks.5.", b22), ("output_blocks.6.", b23), ("output_blocks.7.", b24), ("output_blocks.8.", b25), ("output_blocks.9.", b26), ("output_blocks.10.", b27), ("output_blocks.11.", b28), ("out.", b29)]);
}

[Node("ModelMergeSD2", Category = "model/merging/model specific", DisplayName = "ModelMergeSD2")]
public sealed class ModelMergeSD2
{
    /// <summary>ComfyUI's ModelMergeSD2: SD 2.x has SD 1.x's blocks, so it is ModelMergeSD1 under another name.</summary>
    public ModelHandle Execute(ModelHandle model1, ModelHandle model2,
        [Input(Name = "time_embed.", Min = 0, Max = 1, Step = 0.01)] double b0 = 1.0,
        [Input(Name = "label_emb.", Min = 0, Max = 1, Step = 0.01)] double b1 = 1.0,
        [Input(Name = "input_blocks.0.", Min = 0, Max = 1, Step = 0.01)] double b2 = 1.0,
        [Input(Name = "input_blocks.1.", Min = 0, Max = 1, Step = 0.01)] double b3 = 1.0,
        [Input(Name = "input_blocks.2.", Min = 0, Max = 1, Step = 0.01)] double b4 = 1.0,
        [Input(Name = "input_blocks.3.", Min = 0, Max = 1, Step = 0.01)] double b5 = 1.0,
        [Input(Name = "input_blocks.4.", Min = 0, Max = 1, Step = 0.01)] double b6 = 1.0,
        [Input(Name = "input_blocks.5.", Min = 0, Max = 1, Step = 0.01)] double b7 = 1.0,
        [Input(Name = "input_blocks.6.", Min = 0, Max = 1, Step = 0.01)] double b8 = 1.0,
        [Input(Name = "input_blocks.7.", Min = 0, Max = 1, Step = 0.01)] double b9 = 1.0,
        [Input(Name = "input_blocks.8.", Min = 0, Max = 1, Step = 0.01)] double b10 = 1.0,
        [Input(Name = "input_blocks.9.", Min = 0, Max = 1, Step = 0.01)] double b11 = 1.0,
        [Input(Name = "input_blocks.10.", Min = 0, Max = 1, Step = 0.01)] double b12 = 1.0,
        [Input(Name = "input_blocks.11.", Min = 0, Max = 1, Step = 0.01)] double b13 = 1.0,
        [Input(Name = "middle_block.0.", Min = 0, Max = 1, Step = 0.01)] double b14 = 1.0,
        [Input(Name = "middle_block.1.", Min = 0, Max = 1, Step = 0.01)] double b15 = 1.0,
        [Input(Name = "middle_block.2.", Min = 0, Max = 1, Step = 0.01)] double b16 = 1.0,
        [Input(Name = "output_blocks.0.", Min = 0, Max = 1, Step = 0.01)] double b17 = 1.0,
        [Input(Name = "output_blocks.1.", Min = 0, Max = 1, Step = 0.01)] double b18 = 1.0,
        [Input(Name = "output_blocks.2.", Min = 0, Max = 1, Step = 0.01)] double b19 = 1.0,
        [Input(Name = "output_blocks.3.", Min = 0, Max = 1, Step = 0.01)] double b20 = 1.0,
        [Input(Name = "output_blocks.4.", Min = 0, Max = 1, Step = 0.01)] double b21 = 1.0,
        [Input(Name = "output_blocks.5.", Min = 0, Max = 1, Step = 0.01)] double b22 = 1.0,
        [Input(Name = "output_blocks.6.", Min = 0, Max = 1, Step = 0.01)] double b23 = 1.0,
        [Input(Name = "output_blocks.7.", Min = 0, Max = 1, Step = 0.01)] double b24 = 1.0,
        [Input(Name = "output_blocks.8.", Min = 0, Max = 1, Step = 0.01)] double b25 = 1.0,
        [Input(Name = "output_blocks.9.", Min = 0, Max = 1, Step = 0.01)] double b26 = 1.0,
        [Input(Name = "output_blocks.10.", Min = 0, Max = 1, Step = 0.01)] double b27 = 1.0,
        [Input(Name = "output_blocks.11.", Min = 0, Max = 1, Step = 0.01)] double b28 = 1.0,
        [Input(Name = "out.", Min = 0, Max = 1, Step = 0.01)] double b29 = 1.0) =>
        Merging.Blocks(model1, model2, [("time_embed.", b0), ("label_emb.", b1), ("input_blocks.0.", b2), ("input_blocks.1.", b3), ("input_blocks.2.", b4), ("input_blocks.3.", b5), ("input_blocks.4.", b6), ("input_blocks.5.", b7), ("input_blocks.6.", b8), ("input_blocks.7.", b9), ("input_blocks.8.", b10), ("input_blocks.9.", b11), ("input_blocks.10.", b12), ("input_blocks.11.", b13), ("middle_block.0.", b14), ("middle_block.1.", b15), ("middle_block.2.", b16), ("output_blocks.0.", b17), ("output_blocks.1.", b18), ("output_blocks.2.", b19), ("output_blocks.3.", b20), ("output_blocks.4.", b21), ("output_blocks.5.", b22), ("output_blocks.6.", b23), ("output_blocks.7.", b24), ("output_blocks.8.", b25), ("output_blocks.9.", b26), ("output_blocks.10.", b27), ("output_blocks.11.", b28), ("out.", b29)]);
}

[Node("ModelMergeSDXL", Category = "model/merging/model specific", DisplayName = "ModelMergeSDXL")]
public sealed class ModelMergeSDXL
{
    /// <summary>SDXL block by block: one ratio per input, middle and output block (ComfyUI's names have no trailing dot here).</summary>
    public ModelHandle Execute(ModelHandle model1, ModelHandle model2,
        [Input(Name = "time_embed.", Min = 0, Max = 1, Step = 0.01)] double b0 = 1.0,
        [Input(Name = "label_emb.", Min = 0, Max = 1, Step = 0.01)] double b1 = 1.0,
        [Input(Name = "input_blocks.0", Min = 0, Max = 1, Step = 0.01)] double b2 = 1.0,
        [Input(Name = "input_blocks.1", Min = 0, Max = 1, Step = 0.01)] double b3 = 1.0,
        [Input(Name = "input_blocks.2", Min = 0, Max = 1, Step = 0.01)] double b4 = 1.0,
        [Input(Name = "input_blocks.3", Min = 0, Max = 1, Step = 0.01)] double b5 = 1.0,
        [Input(Name = "input_blocks.4", Min = 0, Max = 1, Step = 0.01)] double b6 = 1.0,
        [Input(Name = "input_blocks.5", Min = 0, Max = 1, Step = 0.01)] double b7 = 1.0,
        [Input(Name = "input_blocks.6", Min = 0, Max = 1, Step = 0.01)] double b8 = 1.0,
        [Input(Name = "input_blocks.7", Min = 0, Max = 1, Step = 0.01)] double b9 = 1.0,
        [Input(Name = "input_blocks.8", Min = 0, Max = 1, Step = 0.01)] double b10 = 1.0,
        [Input(Name = "middle_block.0", Min = 0, Max = 1, Step = 0.01)] double b11 = 1.0,
        [Input(Name = "middle_block.1", Min = 0, Max = 1, Step = 0.01)] double b12 = 1.0,
        [Input(Name = "middle_block.2", Min = 0, Max = 1, Step = 0.01)] double b13 = 1.0,
        [Input(Name = "output_blocks.0", Min = 0, Max = 1, Step = 0.01)] double b14 = 1.0,
        [Input(Name = "output_blocks.1", Min = 0, Max = 1, Step = 0.01)] double b15 = 1.0,
        [Input(Name = "output_blocks.2", Min = 0, Max = 1, Step = 0.01)] double b16 = 1.0,
        [Input(Name = "output_blocks.3", Min = 0, Max = 1, Step = 0.01)] double b17 = 1.0,
        [Input(Name = "output_blocks.4", Min = 0, Max = 1, Step = 0.01)] double b18 = 1.0,
        [Input(Name = "output_blocks.5", Min = 0, Max = 1, Step = 0.01)] double b19 = 1.0,
        [Input(Name = "output_blocks.6", Min = 0, Max = 1, Step = 0.01)] double b20 = 1.0,
        [Input(Name = "output_blocks.7", Min = 0, Max = 1, Step = 0.01)] double b21 = 1.0,
        [Input(Name = "output_blocks.8", Min = 0, Max = 1, Step = 0.01)] double b22 = 1.0,
        [Input(Name = "out.", Min = 0, Max = 1, Step = 0.01)] double b23 = 1.0) =>
        Merging.Blocks(model1, model2, [("time_embed.", b0), ("label_emb.", b1), ("input_blocks.0", b2), ("input_blocks.1", b3), ("input_blocks.2", b4), ("input_blocks.3", b5), ("input_blocks.4", b6), ("input_blocks.5", b7), ("input_blocks.6", b8), ("input_blocks.7", b9), ("input_blocks.8", b10), ("middle_block.0", b11), ("middle_block.1", b12), ("middle_block.2", b13), ("output_blocks.0", b14), ("output_blocks.1", b15), ("output_blocks.2", b16), ("output_blocks.3", b17), ("output_blocks.4", b18), ("output_blocks.5", b19), ("output_blocks.6", b20), ("output_blocks.7", b21), ("output_blocks.8", b22), ("out.", b23)]);
}

/// <summary>The model-file saving nodes (CheckpointSave, ModelSave, VAESave): the weights as they are after the graph's LoRAs and merges, fp16, LDM names.</summary>
internal static class ModelSaving
{
    /// <summary>ComfyUI's metadata: the prompt and the editor's extras, plus its modelspec keys for SDXL.</summary>
    public static Dictionary<string, string> Metadata(NodeContext context, SdFamily? family, string title)
    {
        var metadata = new Dictionary<string, string>(Results.Metadata(context));
        if (family == SdFamily.SDXL)
        {
            metadata["modelspec.architecture"] = "stable-diffusion-xl-v1-base";
            metadata["modelspec.sai_model_spec"] = "1.0.0";
            metadata["modelspec.implementation"] = "sgm";
            metadata["modelspec.title"] = title;
        }
        return metadata;
    }

    public static async Task<UiOutput> SaveAsync(IServiceProvider services, string prefix, Func<Stream, Task> write)
    {
        var file = await Results.For(services).SaveStreamAsync("output", prefix, "safetensors", write);
        return new UiOutput("models", new object[] { file });
    }
}

[Node("CheckpointSave", Category = "model/merging", DisplayName = "Save Checkpoint", IsOutput = true)]
public sealed class CheckpointSave
{
    /// <summary>
    /// Writes the model, CLIP and VAE as one .safetensors checkpoint (LDM names, fp16; v-prediction models get ComfyUI's v_pred marker) in the output folder:
    /// <c>checkpoints/ComfyUI_00001_.safetensors</c>. The three must come from the same loaded checkpoint for now.
    /// </summary>
    public async Task<UiOutput> ExecuteAsync(ModelHandle model, ClipHandle clip, VaeHandle vae, NodeContext context, IServiceProvider services,
        string filenamePrefix = "checkpoints/ComfyUI")
    {
        if (model.Model is not Checkpoint ckpt || !ReferenceEquals(clip.Encoder, ckpt) || !ReferenceEquals(vae.Vae, ckpt))
            throw new NotSupportedException("Save Checkpoint writes a model, CLIP and VAE from the same loaded checkpoint; use Save Model / Save VAE for parts loaded alone.");
        if (ckpt.File is not { } file)
            throw new NotSupportedException("Save Checkpoint writes checkpoints loaded from a .safetensors file; this is an ONNX export.");
        await ((IDiffusionModel)ckpt).UseWeightsAsync(model.Weights, context.CancellationToken);
        await ((ITextEncoderModel)ckpt).UseWeightsAsync(clip.Weights, context.CancellationToken);
        var options = new SdSaveOptions
        {
            DType = SdSaveDType.F16,
            VPrediction = ckpt.Sampling.Prediction == PredictionType.VPrediction,
            Metadata = ModelSaving.Metadata(context, ckpt.Family, filenamePrefix),
        };
        return await ModelSaving.SaveAsync(services, filenamePrefix, s => file.SaveAsync(s, options, context.CancellationToken));
    }
}

[Node("ModelSave", Category = "model/merging", DisplayName = "ModelSave", IsOutput = true)]
public sealed class ModelSave
{
    /// <summary>Writes the diffusion model alone (<c>model.diffusion_model.*</c>, fp16) in the output folder: <c>diffusion_models/ComfyUI_00001_.safetensors</c>.</summary>
    public async Task<UiOutput> ExecuteAsync(ModelHandle model, NodeContext context, IServiceProvider services, string filenamePrefix = "diffusion_models/ComfyUI")
    {
        if (model.Model?.UNet is not { } unet)
            throw new NotSupportedException("Save Model writes models loaded from .safetensors (a checkpoint or Load Diffusion Model); this is an ONNX export.");
        await model.Model.UseWeightsAsync(model.Weights, context.CancellationToken);
        var options = new SdSaveOptions
        {
            DType = SdSaveDType.F16,
            VPrediction = model.Model.Sampling.Prediction == PredictionType.VPrediction,
            Metadata = ModelSaving.Metadata(context, model.Model.Family, filenamePrefix),
        };
        return await ModelSaving.SaveAsync(services, filenamePrefix, s => unet.SaveAsync(s, options, context.CancellationToken));
    }
}

[Node("VAESave", Category = "model/merging", DisplayName = "VAESave", IsOutput = true)]
public sealed class VAESave
{
    /// <summary>Writes the VAE alone (<c>encoder.* / decoder.*</c>, fp16) in the output folder: <c>vae/ComfyUI_vae_00001_.safetensors</c>.</summary>
    public async Task<UiOutput> ExecuteAsync(VaeHandle vae, NodeContext context, IServiceProvider services, string filenamePrefix = "vae/ComfyUI_vae")
    {
        if (vae.Vae.SdVae is not { } sdVae)
            throw new NotSupportedException("Save VAE writes VAEs loaded from .safetensors (a checkpoint or Load VAE); not an ONNX export or a tiny VAE.");
        var options = new SdSaveOptions { DType = SdSaveDType.F16, Metadata = ModelSaving.Metadata(context, null, filenamePrefix) };
        return await ModelSaving.SaveAsync(services, filenamePrefix, s => sdVae.SaveAsync(s, options, context.CancellationToken));
    }
}
