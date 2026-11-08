using System.Reflection;

namespace SpawnGen.Glsl;

internal enum BaseType { Void, Float, Int, UInt, Bool, Sampler }

/// <summary>A GLSL type of the supported subset: a scalar or a 2-4 component vector of float / int / uint / bool, sampler2D, or void.</summary>
internal sealed record GlslType(BaseType Base, int Size)
{
    public static readonly GlslType Void = new(BaseType.Void, 0), Float = new(BaseType.Float, 1), Int = new(BaseType.Int, 1),
        UInt = new(BaseType.UInt, 1), Bool = new(BaseType.Bool, 1), Sampler = new(BaseType.Sampler, 1);

    public bool IsScalar => Size == 1 && Base is not (BaseType.Void or BaseType.Sampler);
    public bool IsVector => Size > 1;
    public bool IsNumeric => Base is BaseType.Float or BaseType.Int or BaseType.UInt;
    public bool IsIntegral => Base is BaseType.Int or BaseType.UInt;
    public GlslType Component => new(Base, Base is BaseType.Void or BaseType.Sampler ? Size : 1);
    public GlslType WithSize(int size) => new(Base, size);

    public static GlslType? FromName(string name) => name switch
    {
        "void" => Void, "float" => Float, "int" => Int, "uint" => UInt, "bool" => Bool, "sampler2D" => Sampler,
        "vec2" => new(BaseType.Float, 2), "vec3" => new(BaseType.Float, 3), "vec4" => new(BaseType.Float, 4),
        "ivec2" => new(BaseType.Int, 2), "ivec3" => new(BaseType.Int, 3), "ivec4" => new(BaseType.Int, 4),
        "uvec2" => new(BaseType.UInt, 2), "uvec3" => new(BaseType.UInt, 3), "uvec4" => new(BaseType.UInt, 4),
        "bvec2" => new(BaseType.Bool, 2), "bvec3" => new(BaseType.Bool, 3), "bvec4" => new(BaseType.Bool, 4),
        _ => null,
    };

    public static bool IsTypeName(string name) => FromName(name) is not null;

    public override string ToString() => (Base, Size) switch
    {
        (BaseType.Void, _) => "void",
        (BaseType.Sampler, _) => "sampler2D",
        (_, 1) => Base switch { BaseType.Float => "float", BaseType.Int => "int", BaseType.UInt => "uint", _ => "bool" },
        _ => (Base switch { BaseType.Float => "", BaseType.Int => "i", BaseType.UInt => "u", _ => "b" }) + "vec" + Size,
    };

    /// <summary>The CLR type the generated code holds the value in (samplers are texture slot numbers, booleans are ints 0 / 1).</summary>
    public Type Clr => (Base, Size) switch
    {
        (BaseType.Void, _) => typeof(void),
        (BaseType.Sampler, _) => typeof(int),
        (BaseType.Float, 1) => typeof(float), (BaseType.Int, 1) => typeof(int), (BaseType.UInt, 1) => typeof(uint), (BaseType.Bool, 1) => typeof(int),
        (BaseType.Float, 2) => typeof(Vec2), (BaseType.Float, 3) => typeof(Vec3), (BaseType.Float, 4) => typeof(Vec4),
        (BaseType.Int, 2) => typeof(IVec2), (BaseType.Int, 3) => typeof(IVec3), (BaseType.Int, 4) => typeof(IVec4),
        (BaseType.UInt, 2) => typeof(UVec2), (BaseType.UInt, 3) => typeof(UVec3), (BaseType.UInt, 4) => typeof(UVec4),
        (BaseType.Bool, 2) => typeof(BVec2), (BaseType.Bool, 3) => typeof(BVec3), (BaseType.Bool, 4) => typeof(BVec4),
        _ => throw new InvalidOperationException($"no CLR type for {this}"),
    };

    static readonly string[] FieldNames = ["X", "Y", "Z", "W"];

    /// <summary>The vector struct's field for component <paramref name="i"/>.</summary>
    public FieldInfo Field(int i) => Clr.GetField(FieldNames[i])!;

    public ConstructorInfo Constructor => Clr.GetConstructors().Single(c => c.GetParameters().Length == Size);
}
