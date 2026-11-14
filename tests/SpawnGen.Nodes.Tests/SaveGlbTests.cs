using SpawnGen.Engine.Execution;
using SpawnGen.Engine.Graph;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.Tests;

/// <summary>Save 3D Model (ComfyUI's SaveGLB) against ComfyUI (bef11fc, 2026-11-13, a 416-byte triangle .glb from a FILE_3D_GLB input).</summary>
public class SaveGlbTests
{
    static readonly byte[] Glb = [.. "glTF"u8, 2, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    sealed class Services(params object[] instances) : IServiceProvider
    {
        public object? GetService(Type serviceType) => instances.FirstOrDefault(serviceType.IsInstanceOfType);
    }

    [Test]
    public async Task The_file_is_saved_as_it_is_with_ComfyUIs_name_and_ui()
    {
        var store = new MemoryResultStore();
        var ui = await new SaveGLB().ExecuteAsync(new Model3D(Glb), new Services(store));
        var file = (ResultFile)((List<object>)ui["3d"]!).Single();
        Assert.That((file.Subfolder, file.Filename, file.Type), Is.EqualTo(("3d", "ComfyUI_00001_.glb", "output")), "prefix 3d/ComfyUI, the '_' before the dot");
        Assert.That(store.Get("output", "ComfyUI_00001_.glb", "3d"), Is.EqualTo(Glb), "the bytes unchanged");
        Assert.That(ui["camera_info"], Is.EqualTo(new List<object?> { null }));
        Assert.That(ui["model_3d_info"], Is.Empty);
        var second = await new SaveGLB().ExecuteAsync(new Model3D(Glb), new Services(store), "probe_%batch_num%");
        Assert.That(((ResultFile)((List<object>)second["3d"]!).Single()).Filename, Is.EqualTo("probe_%batch_num%_00001_.glb"), "%batch_num% stays as written");
    }

    [Test]
    public void The_schema_is_ComfyUIs_and_old_SpawnGen_prompts_still_find_it()
    {
        var registry = new NodeRegistry().AddAssembly(typeof(SaveGLB).Assembly);
        var schema = registry.GetSchema("SaveGLB")!;
        Assert.That(schema.DisplayName, Is.EqualTo("Save 3D Model"));
        Assert.That(schema.FindInput("mesh")!.Type.Name, Does.StartWith("MESH,FILE_3D_GLB"));
        Assert.That(schema.FindInput("filename_prefix")!.Widget!.Default, Is.EqualTo("3d/ComfyUI"));
        Assert.That(registry.GetSchema("Save3DModel")?.Id, Is.EqualTo("SaveGLB"), "the earlier id");
    }
}
