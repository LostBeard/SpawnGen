using System.Reflection;
using System.Reflection.Emit;

namespace SpawnGen.Glsl;

internal sealed partial class Compiler
{
    static readonly HashSet<string> BuiltinNames =
    [
        "radians", "degrees", "sin", "cos", "tan", "asin", "acos", "atan", "pow", "exp", "log", "exp2", "log2", "sqrt", "inversesqrt",
        "abs", "sign", "floor", "ceil", "trunc", "round", "roundEven", "fract", "mod", "min", "max", "clamp", "mix", "step", "smoothstep",
        "isnan", "isinf", "length", "distance", "dot", "cross", "normalize", "reflect",
        "lessThan", "lessThanEqual", "greaterThan", "greaterThanEqual", "equal", "notEqual", "any", "all", "not",
        "texture", "textureLod", "texelFetch", "textureSize",
    ];

    GlslType EmitCall(CallExpr c)
    {
        if (GlslType.FromName(c.Name) is { } ctor) return EmitConstructor(ctor, c.Args, c.Line);
        var candidates = _functions.Where(f => f.Decl.Name == c.Name).ToList();
        if (candidates.Count > 0) return EmitUserCall(c, candidates);
        if (BuiltinNames.Contains(c.Name)) return EmitBuiltin(c);
        throw new GlslException($"'{c.Name}' : no matching overloaded function found", c.Line);
    }

    GlslType EmitConstructor(GlslType t, List<Expr> args, int line)
    {
        if (t.Base is BaseType.Void or BaseType.Sampler) throw new GlslException($"'{t}' : cannot construct this type", line);
        if (args.Count == 0) throw new GlslException($"'{t}' : constructor does not have any arguments", line);
        if (t.Size == 1)
        {
            if (args.Count != 1) throw new GlslException($"'{t}' : too many arguments", line);
            var at = EmitExpr(args[0]);
            if (at.Base is BaseType.Void or BaseType.Sampler) throw new GlslException($"'{t}' : cannot convert from '{at}'", line);
            if (at.Size > 1) LoadComp(Spill(at), at, 0);
            ConvertScalar(at.Base, t.Base);
            return t;
        }
        if (args.Count == 1 && TypeOf(args[0]) is { Size: 1 } single)
        {
            if (single.Base is BaseType.Void or BaseType.Sampler) throw new GlslException($"'{t}' : cannot convert from '{single}'", line);
            EmitExpr(args[0]);
            ConvertScalar(single.Base, t.Base);
            var l = Spill(t.Component);
            for (int i = 0; i < t.Size; i++) _il.Emit(OpCodes.Ldloc, l);
            Construct(t);
            return t;
        }
        int count = 0;
        foreach (var arg in args)
        {
            if (count >= t.Size) throw new GlslException($"'{t}' : too many arguments", line);
            var at = EmitExpr(arg);
            if (at.Base is BaseType.Void or BaseType.Sampler) throw new GlslException($"'{t}' : cannot convert from '{at}'", line);
            var l = Spill(at);
            for (int i = 0; i < at.Size && count < t.Size; i++, count++)
            {
                LoadComp(l, at, i);
                ConvertScalar(at.Base, t.Base);
            }
        }
        if (count < t.Size) throw new GlslException($"'{t}' : not enough data provided for construction", line);
        Construct(t);
        return t;
    }

    GlslType EmitUserCall(CallExpr c, List<Func> candidates)
    {
        var types = c.Args.Select(TypeOf).ToList();
        var f = candidates.FirstOrDefault(f => f.Params.Select(p => p.Type).SequenceEqual(types))
            ?? throw new GlslException($"'{c.Name}' : no matching overloaded function found for ({string.Join(", ", types)})", c.Line);
        _il.EmitArg(OpCodes.Ldarg, 0);
        _il.EmitArg(OpCodes.Ldarg, 1);
        // out / inout parameters are copied in and out through temporaries (GLSL's own semantics), which also lets a swizzle be an argument.
        var copies = new List<(LValue Target, LocalBuilder? Temp)>();
        for (int i = 0; i < c.Args.Count; i++)
        {
            var (type, qualifier, _) = f.Params[i];
            if (qualifier == "in") { EmitExpr(c.Args[i]); continue; }
            var lv = ResolveLValue(c.Args[i]);
            var temp = _il.DeclareLocal(type.Clr);
            if (qualifier == "inout") LoadLValue(lv); else EmitDefault(type);
            _il.Emit(OpCodes.Stloc, temp);
            _il.Emit(OpCodes.Ldloca, temp);
            copies.Add((lv, temp));
        }
        _il.Emit(OpCodes.Call, f.Method!);
        if (copies.Count == 0) return f.Ret;
        LocalBuilder? ret = f.Ret != GlslType.Void ? Spill(f.Ret) : null;
        foreach (var (target, temp) in copies) StoreLValue(target, temp);
        if (f.Ret != GlslType.Void) _il.Emit(OpCodes.Ldloc, ret);
        return f.Ret;
    }

    static MethodInfo MathMethod(string name, params Type[] types) =>
        typeof(GlslMath).GetMethod(name, types) ?? throw new InvalidOperationException($"GlslMath.{name}({string.Join(", ", types.Select(t => t.Name))})");

    /// <summary>
    /// A built-in applied per component: every argument is evaluated once, scalars are used for every component, vectors must all be the same size;
    /// <paramref name="pick"/> checks the argument types and returns the scalar method and the result's base type.
    /// </summary>
    GlslType Componentwise(CallExpr c, Func<GlslType[], (MethodInfo Method, BaseType Result)> pick)
    {
        var types = new GlslType[c.Args.Count];
        var locals = new LocalBuilder?[c.Args.Count];
        for (int i = 0; i < c.Args.Count; i++) { types[i] = EmitExpr(c.Args[i]); locals[i] = Spill(types[i]); }
        int n = types.Max(t => t.Size);
        if (types.Any(t => t.Size != 1 && t.Size != n) || types.Any(t => t.Base is BaseType.Void or BaseType.Sampler)) throw NoMatch(c, types);
        var (method, result) = pick(types);
        for (int k = 0; k < n; k++)
        {
            for (int i = 0; i < types.Length; i++) LoadComp(locals[i], types[i], k);
            _il.Emit(OpCodes.Call, method);
        }
        var type = new GlslType(result, n);
        Construct(type);
        return type;
    }

    static GlslException NoMatch(CallExpr c, IEnumerable<GlslType> types) =>
        new($"'{c.Name}' : no matching overloaded function found for ({string.Join(", ", types)})", c.Line);

    void ExpectArgs(CallExpr c, params int[] counts)
    {
        if (!counts.Contains(c.Args.Count)) throw new GlslException($"'{c.Name}' : wrong number of arguments", c.Line);
    }

    GlslType EmitBuiltin(CallExpr c)
    {
        var name = c.Name;
        switch (name)
        {
            case "radians" or "degrees" or "sin" or "cos" or "tan" or "asin" or "acos" or "exp" or "log" or "exp2" or "log2" or "sqrt" or "inversesqrt"
                or "floor" or "ceil" or "trunc" or "round" or "roundEven" or "fract":
            {
                ExpectArgs(c, 1);
                string method = name switch { "inversesqrt" => "InverseSqrt", "roundEven" => "RoundEven", _ => char.ToUpperInvariant(name[0]) + name[1..] };
                return Componentwise(c, t => t[0].Base == BaseType.Float ? (MathMethod(method, typeof(float)), BaseType.Float) : throw NoMatch(c, t));
            }
            case "atan":
                ExpectArgs(c, 1, 2);
                return Componentwise(c, t => t.All(x => x.Base == BaseType.Float) && (t.Length == 1 || t[0].Size == t[1].Size)
                    ? (MathMethod(t.Length == 1 ? "Atan" : "Atan2", [.. t.Select(_ => typeof(float))]), BaseType.Float) : throw NoMatch(c, t));
            case "pow":
                ExpectArgs(c, 2);
                return Componentwise(c, t => t[0] == t[1] && t[0].Base == BaseType.Float ? (MathMethod("Pow", typeof(float), typeof(float)), BaseType.Float) : throw NoMatch(c, t));
            case "abs" or "sign":
                ExpectArgs(c, 1);
                return Componentwise(c, t => t[0].Base is BaseType.Float or BaseType.Int
                    ? (MathMethod(name == "abs" ? "Abs" : "Sign", t[0].Component.Clr), t[0].Base) : throw NoMatch(c, t));
            case "mod":
                ExpectArgs(c, 2);
                return Componentwise(c, t => t.All(x => x.Base == BaseType.Float) && (t[1].Size == 1 || t[1] == t[0])
                    ? (MathMethod("Mod", typeof(float), typeof(float)), BaseType.Float) : throw NoMatch(c, t));
            case "min" or "max":
                ExpectArgs(c, 2);
                return Componentwise(c, t => t[0].IsNumeric && t[1].Base == t[0].Base && (t[1].Size == 1 || t[1] == t[0])
                    ? (MathMethod(name == "min" ? "Min" : "Max", t[0].Component.Clr, t[0].Component.Clr), t[0].Base) : throw NoMatch(c, t));
            case "clamp":
                ExpectArgs(c, 3);
                return Componentwise(c, t => t[0].IsNumeric && t.Skip(1).All(x => x.Base == t[0].Base && (x.Size == 1 || x == t[0]))
                    ? (MathMethod("Clamp", t[0].Component.Clr, t[0].Component.Clr, t[0].Component.Clr), t[0].Base) : throw NoMatch(c, t));
            case "mix":
                ExpectArgs(c, 3);
                return Componentwise(c, t =>
                {
                    if (t[0] != t[1] || t[0].Base != BaseType.Float) throw NoMatch(c, t);
                    if (t[2].Base == BaseType.Float && (t[2].Size == 1 || t[2].Size == t[0].Size)) return (MathMethod("Mix", typeof(float), typeof(float), typeof(float)), BaseType.Float);
                    if (t[2].Base == BaseType.Bool && t[2].Size == t[0].Size) return (MathMethod("MixBool", typeof(float), typeof(float), typeof(int)), BaseType.Float);
                    throw NoMatch(c, t);
                });
            case "step":
                ExpectArgs(c, 2);
                return Componentwise(c, t => t.All(x => x.Base == BaseType.Float) && (t[0].Size == 1 || t[0] == t[1])
                    ? (MathMethod("Step", typeof(float), typeof(float)), BaseType.Float) : throw NoMatch(c, t));
            case "smoothstep":
                ExpectArgs(c, 3);
                return Componentwise(c, t => t.All(x => x.Base == BaseType.Float) && t.Take(2).All(x => x.Size == 1 || x == t[2])
                    ? (MathMethod("SmoothStep", typeof(float), typeof(float), typeof(float)), BaseType.Float) : throw NoMatch(c, t));
            case "isnan" or "isinf":
                ExpectArgs(c, 1);
                return Componentwise(c, t => t[0].Base == BaseType.Float ? (MathMethod(name == "isnan" ? "IsNan" : "IsInf", typeof(float)), BaseType.Bool) : throw NoMatch(c, t));
            case "dot":
            {
                ExpectArgs(c, 2);
                var (a, at) = EmitSpilled(c.Args[0]);
                var (b, bt) = EmitSpilled(c.Args[1]);
                if (at != bt || at.Base != BaseType.Float) throw NoMatch(c, [at, bt]);
                EmitDot(a, b, at);
                return GlslType.Float;
            }
            case "length":
            {
                ExpectArgs(c, 1);
                var (a, at) = EmitSpilled(c.Args[0]);
                if (at.Base != BaseType.Float) throw NoMatch(c, [at]);
                EmitDot(a, a, at);
                _il.Emit(OpCodes.Call, MathMethod("Sqrt", typeof(float)));
                return GlslType.Float;
            }
            case "distance":
            {
                ExpectArgs(c, 2);
                var (a, at) = EmitSpilled(c.Args[0]);
                var (b, bt) = EmitSpilled(c.Args[1]);
                if (at != bt || at.Base != BaseType.Float) throw NoMatch(c, [at, bt]);
                for (int i = 0; i < at.Size; i++) { LoadComp(a, at, i); LoadComp(b, bt, i); _il.Emit(OpCodes.Sub); }
                Construct(at);
                var d = Spill(at);
                EmitDot(d, d, at);
                _il.Emit(OpCodes.Call, MathMethod("Sqrt", typeof(float)));
                return GlslType.Float;
            }
            case "normalize":
            {
                ExpectArgs(c, 1);
                var (a, at) = EmitSpilled(c.Args[0]);
                if (at.Base != BaseType.Float) throw NoMatch(c, [at]);
                EmitDot(a, a, at);
                _il.Emit(OpCodes.Call, MathMethod("InverseSqrt", typeof(float)));
                var s = Spill(GlslType.Float);
                for (int i = 0; i < at.Size; i++) { LoadComp(a, at, i); _il.Emit(OpCodes.Ldloc, s); _il.Emit(OpCodes.Mul); }
                Construct(at);
                return at;
            }
            case "cross":
            {
                ExpectArgs(c, 2);
                var v3 = new GlslType(BaseType.Float, 3);
                var (a, at) = EmitSpilled(c.Args[0]);
                var (b, bt) = EmitSpilled(c.Args[1]);
                if (at != v3 || bt != v3) throw NoMatch(c, [at, bt]);
                foreach (var (i, j) in new[] { (1, 2), (2, 0), (0, 1) })
                {
                    LoadComp(a, at, i); LoadComp(b, bt, j); _il.Emit(OpCodes.Mul);
                    LoadComp(a, at, j); LoadComp(b, bt, i); _il.Emit(OpCodes.Mul);
                    _il.Emit(OpCodes.Sub);
                }
                Construct(v3);
                return v3;
            }
            case "reflect":
            {
                ExpectArgs(c, 2);
                var (i0, it) = EmitSpilled(c.Args[0]);
                var (n0, nt) = EmitSpilled(c.Args[1]);
                if (it != nt || it.Base != BaseType.Float) throw NoMatch(c, [it, nt]);
                EmitDot(n0, i0, it);
                _il.Emit(OpCodes.Ldc_R4, 2f);
                _il.Emit(OpCodes.Mul);
                var k = Spill(GlslType.Float);
                for (int x = 0; x < it.Size; x++) { LoadComp(i0, it, x); _il.Emit(OpCodes.Ldloc, k); LoadComp(n0, nt, x); _il.Emit(OpCodes.Mul); _il.Emit(OpCodes.Sub); }
                Construct(it);
                return it;
            }
            case "lessThan" or "lessThanEqual" or "greaterThan" or "greaterThanEqual" or "equal" or "notEqual":
            {
                ExpectArgs(c, 2);
                var (a, at) = EmitSpilled(c.Args[0]);
                var (b, bt) = EmitSpilled(c.Args[1]);
                bool equality = name is "equal" or "notEqual";
                if (at != bt || !at.IsVector || (!equality && !at.IsNumeric)) throw NoMatch(c, [at, bt]);
                string op = name switch { "lessThan" => "<", "lessThanEqual" => "<=", "greaterThan" => ">", "greaterThanEqual" => ">=", "equal" => "==", _ => "!=" };
                var scalar = at.Component;
                for (int i = 0; i < at.Size; i++)
                {
                    LoadComp(a, at, i); var x = Spill(scalar);
                    LoadComp(b, bt, i); var y = Spill(scalar);
                    EmitArithmetic(op, x, scalar, y, scalar, c.Line);
                }
                var result = new GlslType(BaseType.Bool, at.Size);
                Construct(result);
                return result;
            }
            case "any" or "all" or "not":
            {
                ExpectArgs(c, 1);
                var (a, at) = EmitSpilled(c.Args[0]);
                if (at.Base != BaseType.Bool || !at.IsVector) throw NoMatch(c, [at]);
                if (name == "not")
                {
                    for (int i = 0; i < at.Size; i++) { LoadComp(a, at, i); _il.Emit(OpCodes.Ldc_I4, 0); _il.Emit(OpCodes.Ceq); }
                    Construct(at);
                    return at;
                }
                for (int i = 0; i < at.Size; i++)
                {
                    LoadComp(a, at, i);
                    if (i > 0) _il.Emit(name == "any" ? OpCodes.Or : OpCodes.And);
                }
                return GlslType.Bool;
            }
            case "texture" or "textureLod":
            {
                ExpectArgs(c, name == "texture" ? new[] { 2, 3 } : new[] { 3 });
                _il.EmitArg(OpCodes.Ldarg, 0);
                if (EmitExpr(c.Args[0]) != GlslType.Sampler || EmitExpr(c.Args[1]) != new GlslType(BaseType.Float, 2)) throw NoMatch(c, c.Args.Select(TypeOf));
                // The bias / level does not matter: textures have one level and are sampled with LINEAR, as ComfyUI sets them up.
                if (c.Args.Count == 3) { if (EmitExpr(c.Args[2]) != GlslType.Float) throw NoMatch(c, c.Args.Select(TypeOf)); _il.Emit(OpCodes.Pop); }
                _il.Emit(OpCodes.Call, Rt(nameof(GlslRuntime.Texture)));
                return new GlslType(BaseType.Float, 4);
            }
            case "texelFetch":
            {
                ExpectArgs(c, 3);
                _il.EmitArg(OpCodes.Ldarg, 0);
                if (EmitExpr(c.Args[0]) != GlslType.Sampler || EmitExpr(c.Args[1]) != new GlslType(BaseType.Int, 2) || EmitExpr(c.Args[2]) != GlslType.Int)
                    throw NoMatch(c, c.Args.Select(TypeOf));
                _il.Emit(OpCodes.Pop);
                _il.Emit(OpCodes.Call, Rt(nameof(GlslRuntime.TexelFetch)));
                return new GlslType(BaseType.Float, 4);
            }
            case "textureSize":
            {
                ExpectArgs(c, 2);
                _il.EmitArg(OpCodes.Ldarg, 0);
                if (EmitExpr(c.Args[0]) != GlslType.Sampler || EmitExpr(c.Args[1]) != GlslType.Int) throw NoMatch(c, c.Args.Select(TypeOf));
                _il.Emit(OpCodes.Pop);
                _il.Emit(OpCodes.Call, Rt(nameof(GlslRuntime.TextureSize)));
                return new GlslType(BaseType.Int, 2);
            }
        }
        throw new GlslException($"'{name}' : no matching overloaded function found", c.Line);
    }

    (LocalBuilder? Local, GlslType Type) EmitSpilled(Expr e)
    {
        var t = EmitExpr(e);
        return (Spill(t), t);
    }

    /// <summary>Pushes the dot product of two spilled float values of type <paramref name="t"/>.</summary>
    void EmitDot(LocalBuilder? a, LocalBuilder? b, GlslType t)
    {
        for (int i = 0; i < t.Size; i++)
        {
            LoadComp(a, t, i); LoadComp(b, t, i); _il.Emit(OpCodes.Mul);
            if (i > 0) _il.Emit(OpCodes.Add);
        }
    }
}
