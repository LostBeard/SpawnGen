using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries and LoraNodeTests. The LoRA itself is merged by SpawnDev.ILGPU.ML when a
// run needs the weights (ComponentWeights).

/// <summary>Load LoRA (Model and CLIP): adds a LoRA to copies of the model and the CLIP; a zero strength leaves that one as it came.</summary>
[Node("LoraLoader", Category = "model/loaders", DisplayName = "Load LoRA (Model and CLIP)", OutputNames = ["MODEL", "CLIP"],
    Description = "This LoRA loader is used to modify both diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together.")]
public sealed class LoraLoader
{
    public static string[] LoraNames() => [.. CheckpointCatalog.Loras];

    public (ModelHandle Model, ClipHandle Clip) Execute(
        [Input(Tooltip = "The diffusion model the LoRA will be applied to.")] ModelHandle model,
        [Input(Tooltip = "The CLIP model the LoRA will be applied to.")] ClipHandle clip,
        [Input(OptionsProvider = nameof(LoraNames), Tooltip = "The name of the LoRA.")] string loraName,
        [Input(Min = -100.0, Max = 100.0, Step = 0.01, Tooltip = "How strongly to modify the diffusion model. This value can be negative.")] double strengthModel = 1.0,
        [Input(Min = -100.0, Max = 100.0, Step = 0.01, Tooltip = "How strongly to modify the CLIP model. This value can be negative.")] double strengthClip = 1.0) =>
        (strengthModel == 0 ? model : WithLora(model, loraName, strengthModel), strengthClip == 0 ? clip : WithLora(clip, loraName, strengthClip));

    /// <summary>A copy of the model handle with the LoRA added to its weight steps.</summary>
    public static ModelHandle WithLora(ModelHandle model, string loraName, double strength) =>
        new(model.Model, model.Sampling) { Patches = model.Patches, Weights = [.. model.Weights, new LoraUse(CheckpointCatalog.LocalLorasRepo, loraName, (float)strength)] };

    /// <summary>A copy of the CLIP handle (its layer kept) with the LoRA added to its weight steps.</summary>
    public static ClipHandle WithLora(ClipHandle clip, string loraName, double strength) =>
        new(clip.Encoder) { Layer = clip.Layer, Weights = [.. clip.Weights, new LoraUse(CheckpointCatalog.LocalLorasRepo, loraName, (float)strength)] };
}

/// <summary>Load LoRA (model only).</summary>
[Node("LoraLoaderModelOnly", Category = "model/loaders", DisplayName = "Load LoRA", OutputNames = ["MODEL"],
    Description = "This LoRAs loader is used to modify the diffusion model, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together.")]
public sealed class LoraLoaderModelOnly
{
    public static string[] LoraNames() => LoraLoader.LoraNames();

    public ModelHandle Execute(ModelHandle model, [Input(OptionsProvider = nameof(LoraNames))] string loraName,
        [Input(Min = -100.0, Max = 100.0, Step = 0.01)] double strengthModel = 1.0) =>
        strengthModel == 0 ? model : LoraLoader.WithLora(model, loraName, strengthModel);
}
