using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;

namespace SpawnGen.Nodes.Core;

[Node(Category = "3d", DisplayName = "Save 3D Model", IsOutput = true)]
public sealed class Save3DModel
{
    /// <summary>Writes the model next to the saved pictures (<c>prefix_00001_.glb</c>); the Studio opens it in its 3D viewer.</summary>
    public async Task<UiOutput> ExecuteAsync(Model3D model, IServiceProvider services, string filenamePrefix = "ComfyUI/model")
    {
        var file = await Results.For(services).SaveAsync("output", filenamePrefix, model.Bytes, model.Extension);
        return new UiOutput("3d", new object[] { file });
    }
}

[Node(Category = "3d", DisplayName = "Preview 3D Model", IsOutput = true)]
public sealed class Preview3DModel
{
    public async Task<UiOutput> ExecuteAsync(Model3D model, IServiceProvider services)
    {
        var file = await Results.For(services).SaveAsync("temp", "preview", model.Bytes, model.Extension);
        return new UiOutput("3d", new object[] { file });
    }
}
