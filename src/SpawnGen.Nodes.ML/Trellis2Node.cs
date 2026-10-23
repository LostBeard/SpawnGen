using SpawnDev.ILGPU.ML.Trellis2;
using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

[Node(Category = "3d", DisplayName = "Image to 3D (TRELLIS.2)")]
public sealed class TrellisImageTo3D
{
    public static string[] Resolutions() => ["512", "1024", "1024 cascade"];
    public static string[] TextureSizes() => ["1024", "2048", "4096"];

    /// <summary>
    /// Turns a picture into a textured 3D model (TRELLIS.2, 4B parameters): a .glb with positions, normals, UVs and base colour and metal-rough textures.
    /// The part of the picture that becomes the object is the <c>mask</c> (1 = object); without one the background is cut away first with RMBG 1.4.
    /// The first picture of a batch is used. The first run downloads about 10 GB; <c>resolution</c> 512 is the fast one, "1024 cascade" the detailed one.
    /// </summary>
    public async Task<Model3D> ExecuteAsync(ImageBatch image, NodeContext context, IServiceProvider services, MaskBatch? mask = null,
        long seed = 42, [Input(Min = 1000, Max = 1000000, Step = 1000)] int maxFaces = 150000,
        [Input(OptionsProvider = nameof(TextureSizes))] string textureSize = "2048", [Input(OptionsProvider = nameof(Resolutions))] string resolution = "512")
    {
        var runtime = await GpuHelpers.RuntimeAsync(services);
        // The weights are 2.5 GB files per stage: the browser needs the library's range-streaming source for them (RangeStreamModelSource), which this node does not set up yet.
        if (runtime.Accelerator.AcceleratorType is global::ILGPU.Runtime.AcceleratorType.WebGPU or global::ILGPU.Runtime.AcceleratorType.WebGL or global::ILGPU.Runtime.AcceleratorType.Wasm)
            throw new NotSupportedException("Image to 3D runs on the desktop engines (CUDA or OpenCL) for now: in the browser the 10 GB of weights need range streaming, which is not wired into this node yet.");
        int w = image.Width, h = image.Height;

        // The object's outline: the mask given, or the background cut-out.
        var foreground = new float[w * h];
        if (mask is not null && mask.Width == w && mask.Height == h) Array.Copy(mask.Data, foreground, foreground.Length);
        else if (mask is not null) foreground = Resample.Resize(mask.Data.AsSpan(0, mask.Width * mask.Height).ToArray(), mask.Width, mask.Height, 1, w, h, ResampleMethod.Bilinear);
        else
        {
            context.Status("cutting out the object");
            var cut = await RemoveBackground.MasksAsync(image, context, services, 0.5f);
            Array.Copy(cut.Data, foreground, foreground.Length);
        }

        var packed = Rgba.Pack(image, 0);
        for (int i = 0; i < packed.Length; i++) packed[i] = (packed[i] & 0x00FFFFFF) | (foreground[i] >= 0.5f ? 255 << 24 : 0);

        var generator = await runtime.GetOrLoadAsync("trellis2", () => Task.FromResult(
            new Trellis2ImageTo3D(runtime.Accelerator, Trellis2Weights.FromModelSource(runtime.ModelSource))));
        generator.StageDone = (stage, seconds) => context.Status($"{stage} ({seconds:0} s)");

        var options = new Trellis2Options
        {
            Seed = (int)((seed ^ (seed >> 31)) & 0x7FFFFFFF), // the library takes a 32-bit seed: fold the wider one so different seeds stay different MaxFaces = maxFaces, TextureSize = int.Parse(textureSize),
            Resolution = resolution.StartsWith("1024") ? 1024 : 512, Cascade = resolution.EndsWith("cascade"),
        };
        context.Status("generating");
        var glb = await generator.GenerateGlbAsync(packed, w, h, options, context.CancellationToken);
        return new Model3D(glb);
    }
}
