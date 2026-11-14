using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// Save 3D Model (ComfyUI's SaveGLB): writes the 3D file as it is to the results as <c>prefix_00001_.ext</c> and shows it in the editor. Measured with
/// ComfyUI (2026-11-13, a .glb from a FILE_3D_GLB input): the bytes are the input's unchanged (no metadata added), the name keeps the "_" before the
/// dot, "%batch_num%" stays as written, and the ui is "3d" (the stored file), "camera_info" [null] and "model_3d_info" []. Earlier SpawnGen workflows
/// used the id Save3DModel and the input name "model".
/// </summary>
[Node("SaveGLB", Category = "3d", DisplayName = "Save 3D Model", IsOutput = true, Aliases = ["Save3DModel"])]
public sealed class SaveGLB
{
    public async Task<UiOutput> ExecuteAsync(
        [Input(Type = "MESH,FILE_3D_GLB,FILE_3D_GLTF,FILE_3D_OBJ,FILE_3D_FBX,FILE_3D_STL,FILE_3D_USDZ,FILE_3D_PLY,FILE_3D_SPLAT,FILE_3D_SPZ,FILE_3D_KSPLAT,FILE_3D_SPLAT_ANY,FILE_3D_POINT_CLOUD_ANY,FILE_3D",
            Aliases = ["model"], Tooltip = "Mesh or 3D file to save")] Model3D mesh,
        IServiceProvider services, string filenamePrefix = "3d/ComfyUI")
    {
        var saved = await Results.For(services).SaveAsync("output", filenamePrefix, mesh.Bytes, mesh.Extension);
        var ui = new UiOutput("3d", new List<object> { saved });
        ui["camera_info"] = new List<object?> { null };
        ui["model_3d_info"] = new List<object>();
        return ui;
    }
}

/// <summary>Preview 3D Model: stores the model file as a temporary result and shows it in the editor's 3D viewer (ui "3d"); nothing is kept.</summary>
[Node("Preview3DModel", Category = "3d", DisplayName = "Preview 3D Model", IsOutput = true)]
public sealed class Preview3DModel
{
    public async Task<UiOutput> ExecuteAsync(Model3D model, IServiceProvider services)
    {
        var saved = await Results.For(services).SaveAsync("temp", "preview", model.Bytes, model.Extension);
        return new UiOutput("3d", new List<object> { saved });
    }
}
