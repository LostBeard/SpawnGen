using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
using System.Text.RegularExpressions;
using ILGPU;
using ILGPU.Runtime;
using SpawnDev.ILGPU;

namespace SpawnGen.Glsl;

/// <summary>A picture handed to a shader: <paramref name="Pixels"/> holds Height x Width x Channels floats, rows top to bottom (3 channels get alpha 1).</summary>
public sealed record ShaderImage(float[] Pixels, int Width, int Height, int Channels);

/// <summary>
/// A GLSL ES 3.00 fragment shader compiled to an ILGPU kernel (the subset ComfyUI's GLSL Shader node uses: functions, loops, vectors and swizzles, int / uint
/// arithmetic, texture / texelFetch / textureSize and the usual built-ins; no structs, arrays or matrices). It runs the way ComfyUI renders: once per output pixel
/// with <c>v_texCoord</c> at the pixel centre, inputs sampled with LINEAR filtering and CLAMP_TO_EDGE, <c>#pragma passes N</c> ping-ponging the first output
/// through intermediate textures, up to four outputs (<c>layout(location = N) out vec4</c>).
/// </summary>
public sealed partial class GlslShader
{
    public const int MaxImages = 5, MaxFloats = 20, MaxInts = 20, MaxBools = 10, MaxCurves = 4, MaxOutputs = 4;
    const int CurveSlot = 5, EmptySlot = 9, PingSlot = 10, Slots = 12;

    readonly Action<Index1D, ShaderEnv> _kernel;
    readonly ConditionalWeakTable<Accelerator, Action<Index1D, ShaderEnv>> _loaded = new();

    /// <summary>How many passes <c>#pragma passes N</c> asks for (1 without it).</summary>
    public int Passes { get; }
    /// <summary>The output locations the shader writes.</summary>
    public IReadOnlyList<int> OutputLocations { get; }
    /// <summary>The number of outputs ComfyUI reads back: one more than the highest <c>fragColorN</c> named in the source (1 when none is).</summary>
    public int OutputCount { get; }

    GlslShader(Action<Index1D, ShaderEnv> kernel, int passes, int[] locations, int outputCount)
    {
        _kernel = kernel;
        Passes = passes;
        OutputLocations = locations;
        OutputCount = outputCount;
    }

    static readonly Dictionary<string, GlslShader> Cache = new(StringComparer.Ordinal);

    /// <summary>Compiles (or returns the already compiled) shader. Throws <see cref="GlslException"/> with GL-style messages for errors.</summary>
    public static GlslShader Compile(string source)
    {
        lock (Cache) if (Cache.TryGetValue(source, out var known)) return known;
        var ast = Parser.Parse(source);
        int passes = 1;
        foreach (var (text, line) in ast.Directives)
        {
            var m = PassesPragma().Match(text);
            if (m.Success) passes = Math.Max(1, int.Parse(m.Groups[1].Value));
            else if (text.StartsWith("#define", StringComparison.Ordinal) || text.StartsWith("#if", StringComparison.Ordinal) || text.StartsWith("#include", StringComparison.Ordinal))
                throw new GlslException($"'{text.Split(' ')[0]}' : preprocessor macros and conditionals are not supported", line);
        }
        var builder = new PersistedAssemblyBuilder(new AssemblyName("GlslShader_" + Guid.NewGuid().ToString("N")), typeof(object).Assembly);
        var module = builder.DefineDynamicModule("GlslShader");
        var locations = Compiler.Compile(ast, module);
        using var stream = new MemoryStream();
        builder.Save(stream);
        stream.Position = 0;
        var assembly = AssemblyLoadContext.Default.LoadFromStream(stream);
        var method = assembly.GetType("GlslShader")!.GetMethod("Kernel")!;
        var kernel = (Action<Index1D, ShaderEnv>)Delegate.CreateDelegate(typeof(Action<Index1D, ShaderEnv>), method);
        // ComfyUI counts outputs by the fragColorN names in the source (_detect_output_count), not by the declarations.
        var named = FragColorName().Matches(source).Select(m => int.Parse(m.Groups[1].Value)).ToList();
        int count = Math.Min(MaxOutputs, named.Count > 0 ? named.Max() + 1 : 1);
        var shader = new GlslShader(kernel, passes, locations, count);
        lock (Cache) Cache[source] = shader;
        return shader;
    }

    [GeneratedRegex(@"^#pragma\s+passes\s+(\d+)")]
    private static partial Regex PassesPragma();
    [GeneratedRegex(@"fragColor(\d+)")]
    private static partial Regex FragColorName();

    /// <summary>
    /// Renders <paramref name="width"/> x <paramref name="height"/> pixels. Returns <see cref="OutputCount"/> outputs, each Height x Width x 4 floats (RGBA), rows
    /// top to bottom. Missing images read as empty textures (0); curves are 1D lookup tables read as <c>u_curve0..3</c> (red channel).
    /// </summary>
    public async Task<float[][]> RunAsync(Accelerator accelerator, int width, int height, IReadOnlyList<ShaderImage?> images,
        IReadOnlyList<float> floats, IReadOnlyList<int> ints, IReadOnlyList<bool> bools, IReadOnlyList<float[]> curves)
    {
        if (width <= 0 || height <= 0) throw new ArgumentException("The output must be at least 1 x 1.");
        var kernel = _loaded.GetValue(accelerator, a => a.LoadAutoGroupedStreamKernel(_kernel));

        // Texels: every texture as GL holds it (RGBA floats, rows bottom up), then the two ping-pong textures.
        var table = new int[Slots * 3];
        var texels = new List<float[]>();
        long offset = 0;
        void Add(int slot, float[] rgba, int w, int h)
        {
            table[slot * 3] = checked((int)offset); table[slot * 3 + 1] = w; table[slot * 3 + 2] = h;
            texels.Add(rgba);
            offset += rgba.Length;
        }
        for (int i = 0; i < Math.Min(images.Count, MaxImages); i++)
            if (images[i] is { } img) Add(i, ToGl(img), img.Width, img.Height);
        for (int i = 0; i < Math.Min(curves.Count, MaxCurves); i++)
        {
            var lut = curves[i];
            if (lut.Length == 0) continue;
            var rgba = new float[lut.Length * 4];
            for (int k = 0; k < lut.Length; k++) { rgba[k * 4] = lut[k]; rgba[k * 4 + 3] = 1f; } // an R32F texture reads (r, 0, 0, 1)
            Add(CurveSlot + i, rgba, lut.Length, 1);
        }
        long pixels = (long)width * height;
        long[] ping = [offset, offset + pixels * 4];
        if (Passes > 1) offset += pixels * 8;
        var allTexels = new float[Math.Max(1, offset)];
        long at = 0;
        foreach (var t in texels) { Array.Copy(t, 0, allTexels, at, t.Length); at += t.Length; }

        var intValues = new int[ShaderEnv.TableBase + table.Length];
        for (int i = 0; i < Math.Min(ints.Count, MaxInts); i++) intValues[i] = ints[i];
        for (int i = 0; i < Math.Min(bools.Count, MaxBools); i++) intValues[ShaderEnv.BoolBase + i] = bools[i] ? 1 : 0;
        var floatValues = new float[MaxFloats];
        for (int i = 0; i < Math.Min(floats.Count, MaxFloats); i++) floatValues[i] = floats[i];

        using var texelBuffer = accelerator.Allocate1D(allTexels);
        using var floatBuffer = accelerator.Allocate1D(floatValues);
        using var intBuffer = accelerator.Allocate1D<int>(intValues.Length);
        using var output = accelerator.Allocate1D<float>(pixels * 4 * MaxOutputs);
        var env = new ShaderEnv
        {
            Texels = texelBuffer.View, Floats = floatBuffer.View, Ints = intBuffer.View, Output = output.View, Width = width, Height = height,
        };
        for (int p = 0; p < Passes; p++)
        {
            intValues[ShaderEnv.PassIndex] = p;
            Array.Copy(table, 0, intValues, ShaderEnv.TableBase, table.Length);
            // Later passes read the previous pass's first output as u_image0.
            if (p > 0) { intValues[ShaderEnv.TableBase] = (int)ping[(p - 1) % 2]; intValues[ShaderEnv.TableBase + 1] = width; intValues[ShaderEnv.TableBase + 2] = height; }
            intBuffer.View.CopyFromCPU(intValues);
            output.MemSetToZero(); // GL clears the framebuffer to 0 before each pass
            kernel((int)pixels, env);
            if (p < Passes - 1)
                output.View.SubView(0, pixels * 4).CopyTo(texelBuffer.View.SubView(ping[p % 2], pixels * 4));
        }
        await accelerator.SynchronizeAsync();
        var result = await output.CopyToHostAsync<float>(0, pixels * 4 * OutputCount);
        var outputs = new float[OutputCount][];
        for (int k = 0; k < OutputCount; k++)
        {
            var o = new float[pixels * 4];
            // Rows come back bottom up (GL): flip to top down.
            for (int y = 0; y < height; y++)
                Array.Copy(result, k * pixels * 4 + (long)(height - 1 - y) * width * 4, o, (long)y * width * 4, width * 4);
            outputs[k] = o;
        }
        return outputs;
    }

    /// <summary>An image as a GL texture: RGBA, rows bottom up (ComfyUI flips the picture vertically when it uploads it).</summary>
    static float[] ToGl(ShaderImage img)
    {
        int w = img.Width, h = img.Height, c = img.Channels;
        var rgba = new float[w * h * 4];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int s = ((h - 1 - y) * w + x) * c, d = (y * w + x) * 4;
                rgba[d] = img.Pixels[s];
                rgba[d + 1] = c > 1 ? img.Pixels[s + 1] : img.Pixels[s];
                rgba[d + 2] = c > 2 ? img.Pixels[s + 2] : img.Pixels[s];
                rgba[d + 3] = c > 3 ? img.Pixels[s + 3] : 1f;
            }
        return rgba;
    }
}
