using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

/// <summary>
/// Save 3D Model: writes the model file (a .glb from an image-to-3D node) to the results as <c>prefix_00001_.ext</c> and shows it in the editor (ui "3d":
/// the stored file, which the page opens in its 3D viewer).
/// </summary>
[Node("Save3DModel", Category = "3d", DisplayName = "Save 3D Model", IsOutput = true)]
public sealed class Save3DModel
{
    public async Task<UiOutput> ExecuteAsync(Model3D model, IServiceProvider services, string filenamePrefix = "ComfyUI/model")
    {
        var saved = await Results.For(services).SaveAsync("output", filenamePrefix, model.Bytes, model.Extension);
        return new UiOutput("3d", new List<object> { saved });
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
