using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Glsl;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

[Node("GLSLShader", Category = "image/shader", DisplayName = "GLSL Shader", HasIntermediateOutput = true, OutputNames = ["IMAGE0", "IMAGE1", "IMAGE2", "IMAGE3"],
    Description = "Apply GLSL ES fragment shaders to images. u_resolution (vec2) is always available.")]
public sealed class GLSLShader
{
    public const string DefaultShader = """
        #version 300 es
        precision highp float;

        uniform sampler2D u_image0;
        uniform vec2 u_resolution;

        in vec2 v_texCoord;
        layout(location = 0) out vec4 fragColor0;

        void main() {
            fragColor0 = texture(u_image0, v_texCoord);
        }

        """;

    public static DynamicOption[] SizeModes() =>
    [
        new("from_input"),
        new("custom",
            new DynamicInput("width", SocketType.Int, new WidgetSpec(WidgetKind.Int) { Default = 512L, Min = 1, Max = 16384 }),
            new DynamicInput("height", SocketType.Int, new WidgetSpec(WidgetKind.Int) { Default = 512L, Min = 1, Max = 16384 })),
    ];

    /// <summary>
    /// ComfyUI's GLSL Shader: the fragment shader is compiled to a GPU kernel (SpawnGen.Glsl: GLSL ES 3.00 to ILGPU, on CUDA, OpenCL, WebGPU or the CPU) and
    /// run once per picture of the first image's batch, at the first image's size or a custom one. Images are u_image0..4, floats u_float0..19, ints
    /// u_int0..19, booleans u_bool0..9, curves u_curve0..3 (256-entry lookup textures), u_resolution the output size and u_pass the pass of
    /// <c>#pragma passes N</c>. The four outputs are RGBA (fragColor0..3, by layout location); outputs the shader does not write are black.
    /// </summary>
    public async Task<(ImageBatch Image0, ImageBatch Image1, ImageBatch Image2, ImageBatch Image3, UiOutput Ui)> ExecuteAsync(NodeContext context, IServiceProvider services,
        [Input(Multiline = true, Default = DefaultShader, Tooltip = "GLSL fragment shader source code (GLSL ES 3.00 / WebGL 2.0 compatible)")] string fragmentShader,
        [DynamicCombo(nameof(SizeModes)), Input(Tooltip = "Output size: 'from_input' uses first input image dimensions, 'custom' allows manual size")] DynamicChoice sizeMode,
        [Autogrow(Prefix = "image", Min = 1, Max = GlslShader.MaxImages), Input(Tooltip = "Images are available as u_image0-4 (sampler2D) in the shader code")] IReadOnlyList<ImageBatch> images,
        [Autogrow(Prefix = "u_float", Min = 0, Max = GlslShader.MaxFloats), Input(Tooltip = "Floats are available as u_float0-19 in the shader code")] IReadOnlyList<double> floats,
        [Autogrow(Prefix = "u_int", Min = 0, Max = GlslShader.MaxInts), Input(Tooltip = "Ints are available as u_int0-19 in the shader code")] IReadOnlyList<long> ints,
        [Autogrow(Prefix = "u_bool", Min = 0, Max = GlslShader.MaxBools), Input(Tooltip = "Booleans are available as u_bool0-9 (bool) in the shader code")] IReadOnlyList<bool> bools,
        [Autogrow(Prefix = "u_curve", Min = 0, Max = GlslShader.MaxCurves), Input(Tooltip = "Curves are available as u_curve0-3 (sampler2D, 1D LUT) in the shader code. Sample with texture(u_curve0, vec2(x, 0.5)).r")] IReadOnlyList<Curve> curves)
    {
        if (images.Count == 0) throw new ArgumentException("At least one input image is required");
        GlslShader shader;
        try { shader = GlslShader.Compile(fragmentShader); }
        catch (GlslException ex) { throw new InvalidOperationException($"Shader compilation failed:\n{ex.Message}", ex); }
        var first = images[0];
        int width = sizeMode.Key == "custom" ? sizeMode.Get("width", 512) : first.Width;
        int height = sizeMode.Key == "custom" ? sizeMode.Get("height", 512) : first.Height;
        var luts = curves.Select(c => c.ToLut().Select(v => (float)v).ToArray()).ToList();
        var accelerator = (await GpuHelpers.RuntimeAsync(services)).Accelerator;

        var outputs = Enumerable.Range(0, GlslShader.MaxOutputs).Select(_ => new ImageBatch(first.Batch, height, width, 4)).ToArray();
        int frame = height * width * 4;
        for (int b = 0; b < first.Batch; b++)
        {
            // ComfyUI takes picture b of every image (an image with a shorter batch would fail there too).
            var batch = images.Select(img => b < img.Batch
                ? new ShaderImage(img.Data.AsSpan(b * img.Height * img.Width * img.Channels, img.Height * img.Width * img.Channels).ToArray(), img.Width, img.Height, img.Channels)
                : throw new ArgumentException($"An input image has {img.Batch} pictures; the first has {first.Batch}.")).ToList<ShaderImage?>();
            var rendered = await shader.RunAsync(accelerator, width, height, batch,
                [.. floats.Select(f => (float)f)], [.. ints.Select(i => unchecked((int)i))], bools, luts);
            for (int k = 0; k < rendered.Length; k++) Array.Copy(rendered[k], 0, outputs[k].Data, b * frame, frame);
        }
        var ui = new UiOutput("images", await Results.PreviewAsync(outputs[0], context, services));
        return (outputs[0], outputs[1], outputs[2], outputs[3], ui);
    }
}
