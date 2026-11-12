using SpawnGen.Engine.Nodes;
using SpawnGen.Engine.Types;
using SpawnGen.Glsl;
using SpawnGen.Nodes.Core;

namespace SpawnGen.Nodes.ML;

// Written clean-room (docs/CLEANROOM.md) from ComfyUI's /object_info entry for GLSL Shader (inputs, defaults, tooltips, the four outputs, its
// intermediate output) and GlslShaderNodeTests; the shader runs on SpawnGen.Glsl, SpawnGen's own GLSL ES compiler.

[Node("GLSLShader", Category = "image/shader", DisplayName = "GLSL Shader", HasIntermediateOutput = true, OutputNames = ["IMAGE0", "IMAGE1", "IMAGE2", "IMAGE3"],
    Description = "Apply GLSL ES fragment shaders to images. u_resolution (vec2) is always available.")]
public sealed class GLSLShader
{
    public const string DefaultShader =
        "#version 300 es\nprecision highp float;\n\nuniform sampler2D u_image0;\nuniform vec2 u_resolution;\n\nin vec2 v_texCoord;\n" +
        "layout(location = 0) out vec4 fragColor0;\n\nvoid main() {\n    fragColor0 = texture(u_image0, v_texCoord);\n}\n";

    public static DynamicOption[] SizeModes() =>
    [
        new("from_input"),
        new("custom", DynamicInput.Int("width", 512, 1, 16384), DynamicInput.Int("height", 512, 1, 16384)),
    ];

    /// <summary>
    /// Renders each picture of the batch (the images' batches are walked together; a shorter batch repeats its last picture) at the first image's size
    /// or the custom one. Each output is RGBA; outputs the shader does not write are black. The first output's pictures are previewed.
    /// </summary>
    public async Task<(ImageBatch Image0, ImageBatch Image1, ImageBatch Image2, ImageBatch Image3, UiOutput Ui)> ExecuteAsync(
        NodeContext context, IServiceProvider services,
        [Input(Multiline = true, Default = DefaultShader, Tooltip = "GLSL fragment shader source code (GLSL ES 3.00 / WebGL 2.0 compatible)")] string fragmentShader,
        [DynamicCombo(nameof(SizeModes)), Input(Tooltip = "Output size: 'from_input' uses first input image dimensions, 'custom' allows manual size")] DynamicChoice sizeMode,
        [Autogrow(Prefix = "image", Min = 1, Max = 5), Input(Tooltip = "Images are available as u_image0-4 (sampler2D) in the shader code")] IReadOnlyList<ImageBatch> images,
        [Autogrow(Prefix = "u_float", Min = 0, Max = 20), Input(Tooltip = "Floats are available as u_float0-19 in the shader code")] IReadOnlyList<double> floats,
        [Autogrow(Prefix = "u_int", Min = 0, Max = 20), Input(Tooltip = "Ints are available as u_int0-19 in the shader code")] IReadOnlyList<long> ints,
        [Autogrow(Prefix = "u_bool", Min = 0, Max = 10), Input(Tooltip = "Booleans are available as u_bool0-9 (bool) in the shader code")] IReadOnlyList<bool> bools,
        [Autogrow(Prefix = "u_curve", Min = 0, Max = 4), Input(Tooltip = "Curves are available as u_curve0-3 (sampler2D, 1D LUT) in the shader code. Sample with texture(u_curve0, vec2(x, 0.5)).r")]
        IReadOnlyList<Curve> curves)
    {
        if (images.Count == 0) throw new ArgumentException("GLSL Shader needs at least one image.", nameof(images));
        GlslShader shader;
        try { shader = GlslShader.Compile(fragmentShader); }
        catch (GlslException e) { throw new InvalidOperationException(e.Message, e); }

        var first = images[0];
        int width = sizeMode.Key == "custom" ? (int)sizeMode.Get("width", 512L) : first.Width;
        int height = sizeMode.Key == "custom" ? (int)sizeMode.Get("height", 512L) : first.Height;
        int batch = images.Max(i => i.Batch);
        var runtime = await GpuHelpers.RuntimeAsync(services);
        var luts = curves.Select(c => c.ToLut(256).Select(v => (float)v).ToArray()).ToList();
        var outputs = Enumerable.Range(0, GlslShader.MaxOutputs).Select(_ => new ImageBatch(batch, height, width, 4)).ToArray();
        int frame = height * width * 4;

        for (int b = 0; b < batch; b++)
        {
            var pictures = images.Select(i => (ShaderImage?)Picture(i, Math.Min(b, i.Batch - 1))).ToList();
            float[][] rendered;
            try
            {
                rendered = await shader.RunAsync(runtime.Accelerator, width, height, pictures,
                    [.. floats.Select(f => (float)f)], [.. ints.Select(i => (int)i)], bools, luts);
            }
            catch (GlslException e) { throw new InvalidOperationException(e.Message, e); }
            for (int o = 0; o < Math.Min(rendered.Length, outputs.Length); o++) Array.Copy(rendered[o], 0, outputs[o].Data, b * frame, frame);
        }

        var store = services.GetService(typeof(IResultStore)) as IResultStore
            ?? new DiskResultStore(services.GetService(typeof(OutputOptions)) as OutputOptions ?? new OutputOptions());
        var prefix = "ComfyUI_temp_" + string.Create(5, 0, (span, _) => { for (int i = 0; i < span.Length; i++) span[i] = (char)('a' + Random.Shared.Next(26)); });
        var previews = new List<object>();
        for (int b = 0; b < batch; b++) previews.Add((await store.SaveAsync("temp", prefix, PngWriter.Encode(outputs[0], b))).AsImage(outputs[0].Width, outputs[0].Height));
        return (outputs[0], outputs[1], outputs[2], outputs[3], new UiOutput("images", previews));
    }

    static ShaderImage Picture(ImageBatch image, int b)
    {
        int size = image.Height * image.Width * image.Channels;
        return new ShaderImage(image.Data.AsSpan(b * size, size).ToArray(), image.Width, image.Height, image.Channels);
    }
}
