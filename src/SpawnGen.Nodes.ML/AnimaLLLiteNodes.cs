using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entries for ModelPatchLoader and AnimaLLLiteApply and the black-box runs 4 / 5
// (docs/PLANS.md "Next: the Anima family"), with SpawnDev.ILGPU.ML's AnimaLLLite doing the network (its picture conventions are the library's,
// pinned on those runs: rgb in [-1, 1] with the masked part zeroed, the mask a 4th channel in {-1, 1}).

/// <summary>MODEL_PATCH: a file from the model_patches folder, opened by the node that applies it (Anima's LLLite so far).</summary>
[Socket("MODEL_PATCH")]
public sealed class ModelPatchHandle(string name, string repo, string file)
{
    public string Name { get; } = name;
    public string Repo { get; } = repo;
    public string File { get; } = file;
}

/// <summary>Anima's LLLite on a model: the patch, the picture (and mask), the strength, and the sigma window it is in force in.</summary>
public sealed record AnimaLLLiteUse(ModelPatchHandle Patch, ImageBatch Image, MaskBatch? Mask, float Strength, double SigmaStart, double SigmaEnd);

[Node("ModelPatchLoader", Category = "model/loaders", DisplayName = "Load Model Patch", OutputNames = ["MODEL_PATCH"])]
public sealed class ModelPatchLoader
{
    public static string[] PatchNames() => ModelFolders.Names(ModelFolders.ModelPatches);

    public ModelPatchHandle Execute([Input(OptionsProvider = nameof(PatchNames))] string name)
    {
        var (repo, file) = ModelFolders.Locate(ModelFolders.ModelPatches, name);
        return new ModelPatchHandle(name, repo, file);
    }
}

/// <summary>
/// Apply Anima LLLite: the model with the LLLite patch (anima-lllite-*.safetensors: ControlNet-LLLite v2 on Anima's DiT) steering every call
/// between start_percent and end_percent, from the picture and (for an inpainting patch) the mask, 1 = repaint. Measured with ComfyUI (runs 4 / 5:
/// anima-base-v1.0 + anima-lllite-inpainting-v2 on a 512 x 512 picture with a Painter mask): the library's AnimaLLLite on the SpawnGen run gives
/// ComfyUI's model calls; the picture is the latent's size x 8 (a picture of another size is refused: what ComfyUI does then is not measured).
/// </summary>
[Node("AnimaLLLiteApply", Category = "model/patch/anima", DisplayName = "Apply Anima LLLite")]
public sealed class AnimaLLLiteApply
{
    public ModelHandle Execute(ModelHandle model, [Input(Name = "model_patch")] ModelPatchHandle modelPatch, ImageBatch image,
        [Input(Min = -10.0, Max = 10.0, Step = 0.01)] double strength = 1.0,
        [Input(Name = "start_percent", Min = 0.0, Max = 1.0, Step = 0.001)] double startPercent = 0.0,
        [Input(Name = "end_percent", Min = 0.0, Max = 1.0, Step = 0.001)] double endPercent = 1.0,
        [Input(Optional = true)] MaskBatch? mask = null)
    {
        if (model.Model is not AnimaDiffusionModel)
            throw new NotSupportedException("Apply Anima LLLite works on an Anima model (Load Diffusion Model: anima-base-v1.0 / anima-preview3-base).");
        if (model.Patches.UNet?.AnimaLLLite is not null)
            throw new NotSupportedException("A second Anima LLLite on the same model: how ComfyUI combines two is not measured yet; apply one.");
        var use = new AnimaLLLiteUse(modelPatch, image, mask, (float)strength, model.Sampling.PercentToSigma(startPercent), model.Sampling.PercentToSigma(endPercent));
        return model.WithPatches(p => p with { UNet = (p.UNet ?? UNetOptions.None) with { AnimaLLLite = use } });
    }
}
