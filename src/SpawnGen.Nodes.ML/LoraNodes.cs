using SpawnGen.Engine.Nodes;

namespace SpawnGen.Nodes.ML;

[Node(Category = "model/loaders", DisplayName = "Load LoRA (Model and CLIP)", OutputNames = ["MODEL", "CLIP"])]
public sealed class LoraLoader
{
    public static string[] LoraNames() => [.. CheckpointCatalog.Loras];

    /// <summary>
    /// A LoRA from <c>&lt;models&gt;/loras</c> on a copy of the model and of the CLIP: <c>strength_model</c> scales its UNet part, <c>strength_clip</c> its text-encoder part
    /// (either may be negative). The model and CLIP that come in are not changed; with both strengths 0 they pass through, as in ComfyUI.
    /// </summary>
    public (ModelHandle Model, ClipHandle Clip) Execute(ModelHandle model, ClipHandle clip,
        [Input(OptionsProvider = nameof(LoraNames))] string loraName,
        [Input(Min = -100, Max = 100, Step = 0.01, Default = 1.0)] double strengthModel,
        [Input(Min = -100, Max = 100, Step = 0.01, Default = 1.0)] double strengthClip)
    {
        if (strengthModel == 0 && strengthClip == 0) return (model, clip);
        return (WithLora(model, loraName, strengthModel), WithLora(clip, loraName, strengthClip));
    }

    internal static ModelHandle WithLora(ModelHandle model, string loraName, double strength) => strength == 0 ? model :
        new ModelHandle(model.Model, model.Sampling) { Patches = model.Patches, Loras = [.. model.Loras, Use(loraName, strength)] };

    internal static ClipHandle WithLora(ClipHandle clip, string loraName, double strength) => strength == 0 ? clip :
        new ClipHandle(clip.Encoder) { Layer = clip.Layer, Loras = [.. clip.Loras, Use(loraName, strength)] };

    static LoraUse Use(string loraName, double strength) => new(CheckpointCatalog.LocalLorasRepo, loraName, (float)strength);
}

[Node(Category = "model/loaders", DisplayName = "Load LoRA", OutputNames = ["MODEL"])]
public sealed class LoraLoaderModelOnly
{
    public static string[] LoraNames() => LoraLoader.LoraNames();

    /// <summary>A LoRA's UNet part on a copy of the model (ComfyUI's LoraLoaderModelOnly); the CLIP is left alone.</summary>
    public ModelHandle Execute(ModelHandle model, [Input(OptionsProvider = nameof(LoraNames))] string loraName,
        [Input(Min = -100, Max = 100, Step = 0.01, Default = 1.0)] double strengthModel) =>
        strengthModel == 0 ? model : LoraLoader.WithLora(model, loraName, strengthModel);
}
