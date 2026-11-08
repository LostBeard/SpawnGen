using ILGPU;
using ILGPU.Runtime;

namespace SpawnGen.Glsl;

// The types and helpers compiled shaders use. Generated IL works on these structs field by field and calls the static helpers; ILGPU compiles both
// into the kernel for every backend.

public struct Vec2 { public float X, Y; public Vec2(float x, float y) { X = x; Y = y; } }
public struct Vec3 { public float X, Y, Z; public Vec3(float x, float y, float z) { X = x; Y = y; Z = z; } }
public struct Vec4 { public float X, Y, Z, W; public Vec4(float x, float y, float z, float w) { X = x; Y = y; Z = z; W = w; } }
public struct IVec2 { public int X, Y; public IVec2(int x, int y) { X = x; Y = y; } }
public struct IVec3 { public int X, Y, Z; public IVec3(int x, int y, int z) { X = x; Y = y; Z = z; } }
public struct IVec4 { public int X, Y, Z, W; public IVec4(int x, int y, int z, int w) { X = x; Y = y; Z = z; W = w; } }
public struct UVec2 { public uint X, Y; public UVec2(uint x, uint y) { X = x; Y = y; } }
public struct UVec3 { public uint X, Y, Z; public UVec3(uint x, uint y, uint z) { X = x; Y = y; Z = z; } }
public struct UVec4 { public uint X, Y, Z, W; public UVec4(uint x, uint y, uint z, uint w) { X = x; Y = y; Z = z; W = w; } }
// Booleans are stored as ints (0 / 1): every backend can hold an int in a struct.
public struct BVec2 { public int X, Y; public BVec2(int x, int y) { X = x; Y = y; } }
public struct BVec3 { public int X, Y, Z; public BVec3(int x, int y, int z) { X = x; Y = y; Z = z; } }
public struct BVec4 { public int X, Y, Z, W; public BVec4(int x, int y, int z, int w) { X = x; Y = y; Z = z; W = w; } }

/// <summary>
/// Everything a shader run reads, as one kernel argument: the textures (RGBA floats, rows bottom to top as GL stores them) and their slot table, the uniforms,
/// the output buffer and the output size.
/// </summary>
public struct ShaderEnv
{
    /// <summary>Every texture's texels, RGBA, rows bottom up.</summary>
    public ArrayView1D<float, Stride1D.Dense> Texels;
    /// <summary>u_float0..19.</summary>
    public ArrayView1D<float, Stride1D.Dense> Floats;
    /// <summary>u_int0..19 at 0, u_bool0..9 at <see cref="BoolBase"/>, u_pass at <see cref="PassIndex"/>, then per texture slot: offset, width, height.</summary>
    public ArrayView1D<int, Stride1D.Dense> Ints;
    /// <summary>The outputs: location, then pixel (rows bottom up), then RGBA.</summary>
    public ArrayView1D<float, Stride1D.Dense> Output;
    public int Width, Height;

    public const int BoolBase = 20, PassIndex = 30, TableBase = 32;
}

/// <summary>The helpers generated shader code calls.</summary>
public static class GlslRuntime
{
    public static float Float(ShaderEnv env, int k) => env.Floats[k];
    public static int Int(ShaderEnv env, int k) => env.Ints[k];
    public static int Bool(ShaderEnv env, int k) => env.Ints[ShaderEnv.BoolBase + k] != 0 ? 1 : 0;
    public static int Pass(ShaderEnv env) => env.Ints[ShaderEnv.PassIndex];
    public static Vec2 Resolution(ShaderEnv env) => new(env.Width, env.Height);

    /// <summary>GL's texture() with LINEAR filtering and CLAMP_TO_EDGE wrapping; an empty slot (an image that is not connected) reads 0.</summary>
    public static Vec4 Texture(ShaderEnv env, int slot, Vec2 uv)
    {
        int b = ShaderEnv.TableBase + slot * 3;
        int off = env.Ints[b], w = env.Ints[b + 1], h = env.Ints[b + 2];
        if (w <= 0 || h <= 0) return default;
        float fx = uv.X * w - 0.5f, fy = uv.Y * h - 0.5f;
        float x0f = MathF.Floor(fx), y0f = MathF.Floor(fy);
        float tx = fx - x0f, ty = fy - y0f;
        int x0 = (int)x0f, y0 = (int)y0f;
        int xa = Clamp(x0, w), xb = Clamp(x0 + 1, w), ya = Clamp(y0, h), yb = Clamp(y0 + 1, h);
        Vec4 c00 = Texel(env, off, w, xa, ya), c10 = Texel(env, off, w, xb, ya), c01 = Texel(env, off, w, xa, yb), c11 = Texel(env, off, w, xb, yb);
        return new Vec4(
            Lerp2(c00.X, c10.X, c01.X, c11.X, tx, ty), Lerp2(c00.Y, c10.Y, c01.Y, c11.Y, tx, ty),
            Lerp2(c00.Z, c10.Z, c01.Z, c11.Z, tx, ty), Lerp2(c00.W, c10.W, c01.W, c11.W, tx, ty));
    }

    /// <summary>GL's texelFetch (level 0), clamped to the texture.</summary>
    public static Vec4 TexelFetch(ShaderEnv env, int slot, IVec2 p)
    {
        int b = ShaderEnv.TableBase + slot * 3;
        int off = env.Ints[b], w = env.Ints[b + 1], h = env.Ints[b + 2];
        if (w <= 0 || h <= 0) return default;
        return Texel(env, off, w, Clamp(p.X, w), Clamp(p.Y, h));
    }

    public static IVec2 TextureSize(ShaderEnv env, int slot)
    {
        int b = ShaderEnv.TableBase + slot * 3;
        return new IVec2(env.Ints[b + 1], env.Ints[b + 2]);
    }

    public static void WriteOutput(ShaderEnv env, int pixel, int location, Vec4 c)
    {
        int o = (location * env.Width * env.Height + pixel) * 4;
        env.Output[o] = c.X;
        env.Output[o + 1] = c.Y;
        env.Output[o + 2] = c.Z;
        env.Output[o + 3] = c.W;
    }

    static int Clamp(int v, int n) => v < 0 ? 0 : v >= n ? n - 1 : v;

    static Vec4 Texel(ShaderEnv env, int off, int w, int x, int y)
    {
        int i = off + (y * w + x) * 4;
        return new Vec4(env.Texels[i], env.Texels[i + 1], env.Texels[i + 2], env.Texels[i + 3]);
    }

    static float Lerp2(float c00, float c10, float c01, float c11, float tx, float ty)
    {
        float top = c00 + (c10 - c00) * tx, bottom = c01 + (c11 - c01) * tx;
        return top + (bottom - top) * ty;
    }
}

/// <summary>GLSL's scalar built-ins (the generated code applies them per component).</summary>
public static class GlslMath
{
    public static float Radians(float x) => x * (MathF.PI / 180f);
    public static float Degrees(float x) => x * (180f / MathF.PI);
    public static float Sin(float x) => MathF.Sin(x);
    public static float Cos(float x) => MathF.Cos(x);
    public static float Tan(float x) => MathF.Tan(x);
    public static float Asin(float x) => MathF.Asin(x);
    public static float Acos(float x) => MathF.Acos(x);
    public static float Atan(float x) => MathF.Atan(x);
    public static float Atan2(float y, float x) => MathF.Atan2(y, x);
    public static float Pow(float x, float y) => MathF.Pow(x, y);
    public static float Exp(float x) => MathF.Exp(x);
    public static float Log(float x) => MathF.Log(x);
    public static float Exp2(float x) => MathF.Pow(2f, x);
    public static float Log2(float x) => MathF.Log(x) * 1.44269504088896341f;
    public static float Sqrt(float x) => MathF.Sqrt(x);
    public static float InverseSqrt(float x) => 1f / MathF.Sqrt(x);
    public static float Abs(float x) => MathF.Abs(x);
    public static int Abs(int x) => x < 0 ? -x : x;
    public static float Sign(float x) => x > 0f ? 1f : x < 0f ? -1f : 0f;
    public static int Sign(int x) => x > 0 ? 1 : x < 0 ? -1 : 0;
    public static float Floor(float x) => MathF.Floor(x);
    public static float Ceil(float x) => MathF.Ceiling(x);
    public static float Trunc(float x) => MathF.Truncate(x);
    /// <summary>GLSL leaves round's halfway case to the implementation; GPUs round half to even, as here.</summary>
    public static float Round(float x) => MathF.Round(x);
    public static float RoundEven(float x) => MathF.Round(x);
    public static float Fract(float x) => x - MathF.Floor(x);
    public static float Mod(float x, float y) => x - y * MathF.Floor(x / y);
    public static float Min(float a, float b) => b < a ? b : a;
    public static int Min(int a, int b) => b < a ? b : a;
    public static uint Min(uint a, uint b) => b < a ? b : a;
    public static float Max(float a, float b) => a < b ? b : a;
    public static int Max(int a, int b) => a < b ? b : a;
    public static uint Max(uint a, uint b) => a < b ? b : a;
    public static float Clamp(float x, float lo, float hi) => Min(Max(x, lo), hi);
    public static int Clamp(int x, int lo, int hi) => Min(Max(x, lo), hi);
    public static uint Clamp(uint x, uint lo, uint hi) => Min(Max(x, lo), hi);
    public static float Mix(float x, float y, float a) => x * (1f - a) + y * a;
    public static float MixBool(float x, float y, int a) => a != 0 ? y : x;
    public static float Step(float edge, float x) => x < edge ? 0f : 1f;
    public static float SmoothStep(float e0, float e1, float x)
    {
        float t = Clamp((x - e0) / (e1 - e0), 0f, 1f);
        return t * t * (3f - 2f * t);
    }
    public static int IsNan(float x) => float.IsNaN(x) ? 1 : 0;
    public static int IsInf(float x) => float.IsInfinity(x) ? 1 : 0;
}
