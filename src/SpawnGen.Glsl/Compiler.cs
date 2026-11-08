using System.Reflection;
using System.Reflection.Emit;
using ILGPU;

namespace SpawnGen.Glsl;

/// <summary>
/// Type-checks a parsed shader and emits it as .NET IL: one static method per GLSL function (arguments: the <see cref="ShaderEnv"/>, a ref to the generated
/// Globals struct, then the GLSL parameters, by ref for out / inout), an initializer for mutable globals, and the kernel that runs main for one pixel.
/// </summary>
internal sealed partial class Compiler
{
    abstract record Symbol(GlslType Type);
    sealed record LocalSym(GlslType Type, LocalBuilder? Local) : Symbol(Type);
    sealed record ArgSym(GlslType Type, int Index, bool ByRef) : Symbol(Type);
    sealed record GlobalSym(GlslType Type, FieldInfo Field) : Symbol(Type);
    sealed record ConstSym(GlslType Type, Expr Init) : Symbol(Type);
    /// <summary>A uniform: Kind is float / int / bool / sampler / resolution / pass / zero (a uniform ComfyUI never sets: GL leaves it 0).</summary>
    sealed record UniformSym(GlslType Type, string Kind, int Index) : Symbol(Type);

    sealed class Func(FunctionDecl decl, GlslType ret, List<(GlslType Type, string Qualifier, string Name)> ps)
    {
        public FunctionDecl Decl { get; } = decl;
        public GlslType Ret { get; } = ret;
        public List<(GlslType Type, string Qualifier, string Name)> Params { get; } = ps;
        public MethodBuilder? Method { get; set; }
        public bool HasBody { get; set; }
    }

    readonly ShaderAst _ast;
    readonly TypeBuilder _kernels;
    readonly TypeBuilder _globals;
    readonly Dictionary<string, Symbol> _globalScope = new(StringComparer.Ordinal);
    readonly List<Func> _functions = [];
    readonly List<(GlslType Type, FieldInfo Field, Expr Init)> _globalInits = [];
    readonly SortedDictionary<int, FieldInfo> _outputs = [];
    FieldInfo _texCoord = null!, _fragCoord = null!;

    // Per-function state.
    Emitter _il = null!;
    Func _current = null!;
    readonly List<Dictionary<string, Symbol>> _scopes = [];
    readonly Stack<(Label Break, Label Continue)> _loops = new();

    Compiler(ShaderAst ast, ModuleBuilder module)
    {
        _ast = ast;
        _kernels = module.DefineType("GlslShader", TypeAttributes.Public | TypeAttributes.Abstract | TypeAttributes.Sealed);
        _globals = module.DefineType("Globals", TypeAttributes.Public | TypeAttributes.Sealed | TypeAttributes.SequentialLayout, typeof(ValueType));
    }

    /// <summary>Compiles the shader into <paramref name="module"/>: returns the output locations it writes (in order).</summary>
    public static int[] Compile(ShaderAst ast, ModuleBuilder module)
    {
        var c = new Compiler(ast, module);
        c.DeclareGlobals();
        c.DeclareFunctions();
        foreach (var f in c._functions.Where(f => f.HasBody)) c.EmitFunction(f);
        c.EmitInit();
        c.EmitKernel();
        c._globals.CreateType();
        c._kernels.CreateType();
        return [.. c._outputs.Keys];
    }

    // ---- declarations ----

    static GlslType TypeOfName(string name, int line) => GlslType.FromName(name) ?? throw new GlslException($"'{name}' : unknown type", line);

    void DeclareGlobals()
    {
        _texCoord = _globals.DefineField("v_texCoord", typeof(Vec2), FieldAttributes.Public);
        _fragCoord = _globals.DefineField("gl_FragCoord", typeof(Vec4), FieldAttributes.Public);
        _globalScope["gl_FragCoord"] = new GlobalSym(new GlslType(BaseType.Float, 4), _fragCoord);
        foreach (var g in _ast.Globals)
        {
            var type = TypeOfName(g.TypeName, g.Line);
            foreach (var v in g.Vars)
            {
                if (_globalScope.ContainsKey(v.Name)) throw new GlslException($"'{v.Name}' : redefinition", v.Line);
                switch (g.Storage)
                {
                    case "uniform":
                        _globalScope[v.Name] = Uniform(v.Name, type, v.Line);
                        break;
                    case "in":
                        if (v.Name != "v_texCoord" || type != new GlslType(BaseType.Float, 2))
                            throw new GlslException($"'{v.Name}' : the only input is 'in vec2 v_texCoord'", v.Line);
                        _globalScope[v.Name] = new GlobalSym(type, _texCoord);
                        break;
                    case "out":
                        if (type != new GlslType(BaseType.Float, 4)) throw new GlslException($"'{v.Name}' : outputs must be vec4", v.Line);
                        int location = g.Location ?? 0;
                        if (location is < 0 or > 3) throw new GlslException($"'{v.Name}' : output location must be 0 to 3", v.Line);
                        var field = _globals.DefineField(v.Name, typeof(Vec4), FieldAttributes.Public);
                        _outputs[location] = field;
                        _globalScope[v.Name] = new GlobalSym(type, field);
                        break;
                    case "const":
                        if (v.Init is null) throw new GlslException($"'{v.Name}' : a const variable needs an initializer", v.Line);
                        _globalScope[v.Name] = new ConstSym(type, v.Init);
                        break;
                    default:
                        var mutable = _globals.DefineField("g_" + v.Name, type.Clr, FieldAttributes.Public);
                        _globalScope[v.Name] = new GlobalSym(type, mutable);
                        if (v.Init is not null) _globalInits.Add((type, mutable, v.Init));
                        break;
                }
            }
        }
    }

    static Symbol Uniform(string name, GlslType type, int line)
    {
        static int? Index(string name, string prefix, int max) =>
            name.StartsWith(prefix, StringComparison.Ordinal) && int.TryParse(name.AsSpan(prefix.Length), out int k) && k >= 0 && k < max ? k : null;
        if (type.Base == BaseType.Sampler)
        {
            if (Index(name, "u_image", 5) is { } image) return new UniformSym(type, "sampler", image);
            if (Index(name, "u_curve", 4) is { } curve) return new UniformSym(type, "sampler", 5 + curve);
            return new UniformSym(type, "sampler", 9); // an empty slot
        }
        if (name == "u_resolution" && type == new GlslType(BaseType.Float, 2)) return new UniformSym(type, "resolution", 0);
        if (name == "u_pass" && type == GlslType.Int) return new UniformSym(type, "pass", 0);
        if (type == GlslType.Float && Index(name, "u_float", 20) is { } f) return new UniformSym(type, "float", f);
        if (type == GlslType.Int && Index(name, "u_int", 20) is { } i) return new UniformSym(type, "int", i);
        if (type == GlslType.Bool && Index(name, "u_bool", 10) is { } b) return new UniformSym(type, "bool", b);
        return new UniformSym(type, "zero", 0);
    }

    void DeclareFunctions()
    {
        foreach (var d in _ast.Functions)
        {
            var ret = TypeOfName(d.ReturnType, d.Line);
            var ps = d.Params.Select(p => (TypeOfName(p.TypeName, p.Line), p.Qualifier, p.Name)).ToList();
            var existing = _functions.FirstOrDefault(f => f.Decl.Name == d.Name && f.Params.Select(p => p.Type).SequenceEqual(ps.Select(p => p.Item1)));
            if (existing is null)
            {
                if (BuiltinNames.Contains(d.Name)) throw new GlslException($"'{d.Name}' : cannot redefine a built-in function", d.Line);
                existing = new Func(d, ret, ps);
                var paramTypes = new List<Type> { typeof(ShaderEnv), _globals.MakeByRefType() };
                paramTypes.AddRange(ps.Select(p => p.Qualifier == "in" ? p.Item1.Clr : p.Item1.Clr.MakeByRefType()));
                var name = d.Name + "__" + string.Join("_", ps.Select(p => p.Item1.ToString()));
                existing.Method = _kernels.DefineMethod(name, MethodAttributes.Public | MethodAttributes.Static, ret.Clr, [.. paramTypes]);
                _functions.Add(existing);
            }
            else if (existing.Ret != ret) throw new GlslException($"'{d.Name}' : overloaded functions must have the same return type", d.Line);
            if (d.Body is not null)
            {
                if (existing.HasBody) throw new GlslException($"'{d.Name}' : function already has a body", d.Line);
                existing.HasBody = true;
                existing = new Func(d, ret, ps) { Method = existing.Method, HasBody = true };
                _functions[_functions.FindIndex(f => f.Method == existing.Method)] = existing;
            }
        }
        var main = _functions.FirstOrDefault(f => f.Decl.Name == "main" && f.Params.Count == 0)
            ?? throw new GlslException("missing main() function", 0);
        if (main.Ret != GlslType.Void) throw new GlslException("'main' : must return void", main.Decl.Line);
        foreach (var f in _functions.Where(f => !f.HasBody))
            throw new GlslException($"'{f.Decl.Name}' : no definition for the declared function", f.Decl.Line);
    }

    // ---- functions ----

    void EmitFunction(Func f)
    {
        _current = f;
        _il = new Emitter(f.Method!.GetILGenerator());
        _scopes.Clear();
        _scopes.Add(new(StringComparer.Ordinal));
        for (int i = 0; i < f.Params.Count; i++)
            if (f.Params[i].Name.Length > 0) _scopes[0][f.Params[i].Name] = new ArgSym(f.Params[i].Type, i + 2, f.Params[i].Qualifier != "in");
        EmitBlock(f.Decl.Body!, newScope: false);
        // Falling off the end: void returns; anything else returns its default (GLSL leaves the value undefined).
        if (f.Ret != GlslType.Void) EmitDefault(f.Ret);
        _il.Emit(OpCodes.Ret);
    }

    /// <summary>Mutable globals' initializers, run once per pixel before main (they may read uniforms).</summary>
    MethodBuilder? _init;

    void EmitInit()
    {
        _init = _kernels.DefineMethod("__init", MethodAttributes.Public | MethodAttributes.Static, typeof(void), [typeof(ShaderEnv), _globals.MakeByRefType()]);
        _current = new Func(new FunctionDecl("void", "__init", [], null, 0), GlslType.Void, []);
        _il = new Emitter(_init.GetILGenerator());
        _scopes.Clear();
        _scopes.Add(new(StringComparer.Ordinal));
        foreach (var (type, field, init) in _globalInits)
        {
            _il.EmitArg(OpCodes.Ldarg, 1);
            ExpectType(EmitExpr(init), type, init.Line, "initializer");
            _il.Emit(OpCodes.Stfld, field);
        }
        _il.Emit(OpCodes.Ret);
    }

    /// <summary>Kernel(Index1D index, ShaderEnv env): one pixel (rows bottom up, as GL draws them); sets v_texCoord and gl_FragCoord, runs main, writes the outputs.</summary>
    void EmitKernel()
    {
        var kernel = _kernels.DefineMethod("Kernel", MethodAttributes.Public | MethodAttributes.Static, typeof(void), [typeof(Index1D), typeof(ShaderEnv)]);
        var il = kernel.GetILGenerator();
        var toInt = typeof(Index1D).GetMethods(BindingFlags.Public | BindingFlags.Static).First(m => m.Name == "op_Implicit" && m.ReturnType == typeof(int));
        var envWidth = typeof(ShaderEnv).GetField(nameof(ShaderEnv.Width))!;
        var envHeight = typeof(ShaderEnv).GetField(nameof(ShaderEnv.Height))!;
        var pixel = il.DeclareLocal(typeof(int));
        var x = il.DeclareLocal(typeof(int));
        var y = il.DeclareLocal(typeof(int));
        var g = il.DeclareLocal(_globals);

        il.Emit(OpCodes.Ldarg_0);
        il.Emit(OpCodes.Call, toInt);
        il.Emit(OpCodes.Stloc, pixel);
        il.Emit(OpCodes.Ldloc, pixel); il.Emit(OpCodes.Ldarga_S, (byte)1); il.Emit(OpCodes.Ldfld, envWidth); il.Emit(OpCodes.Rem); il.Emit(OpCodes.Stloc, x);
        il.Emit(OpCodes.Ldloc, pixel); il.Emit(OpCodes.Ldarga_S, (byte)1); il.Emit(OpCodes.Ldfld, envWidth); il.Emit(OpCodes.Div); il.Emit(OpCodes.Stloc, y);

        // v_texCoord = ((x + 0.5) / width, (y + 0.5) / height): the pixel centre, as the full-screen triangle interpolates it.
        il.Emit(OpCodes.Ldloca, g);
        il.Emit(OpCodes.Ldloc, x); il.Emit(OpCodes.Conv_R4); il.Emit(OpCodes.Ldc_R4, 0.5f); il.Emit(OpCodes.Add);
        il.Emit(OpCodes.Ldarga_S, (byte)1); il.Emit(OpCodes.Ldfld, envWidth); il.Emit(OpCodes.Conv_R4); il.Emit(OpCodes.Div);
        il.Emit(OpCodes.Ldloc, y); il.Emit(OpCodes.Conv_R4); il.Emit(OpCodes.Ldc_R4, 0.5f); il.Emit(OpCodes.Add);
        il.Emit(OpCodes.Ldarga_S, (byte)1); il.Emit(OpCodes.Ldfld, envHeight); il.Emit(OpCodes.Conv_R4); il.Emit(OpCodes.Div);
        il.Emit(OpCodes.Newobj, typeof(Vec2).GetConstructors()[0]);
        il.Emit(OpCodes.Stfld, _texCoord);
        // gl_FragCoord = (x + 0.5, y + 0.5, 0.5, 1).
        il.Emit(OpCodes.Ldloca, g);
        il.Emit(OpCodes.Ldloc, x); il.Emit(OpCodes.Conv_R4); il.Emit(OpCodes.Ldc_R4, 0.5f); il.Emit(OpCodes.Add);
        il.Emit(OpCodes.Ldloc, y); il.Emit(OpCodes.Conv_R4); il.Emit(OpCodes.Ldc_R4, 0.5f); il.Emit(OpCodes.Add);
        il.Emit(OpCodes.Ldc_R4, 0.5f); il.Emit(OpCodes.Ldc_R4, 1f);
        il.Emit(OpCodes.Newobj, typeof(Vec4).GetConstructors()[0]);
        il.Emit(OpCodes.Stfld, _fragCoord);

        il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldloca, g); il.Emit(OpCodes.Call, _init!);
        var main = _functions.Single(f => f.Decl.Name == "main" && f.Params.Count == 0).Method!;
        il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldloca, g); il.Emit(OpCodes.Call, main);
        var write = typeof(GlslRuntime).GetMethod(nameof(GlslRuntime.WriteOutput))!;
        foreach (var (location, field) in _outputs)
        {
            il.Emit(OpCodes.Ldarg_1); il.Emit(OpCodes.Ldloc, pixel); il.Emit(OpCodes.Ldc_I4, location);
            il.Emit(OpCodes.Ldloc, g); il.Emit(OpCodes.Ldfld, field);
            il.Emit(OpCodes.Call, write);
        }
        il.Emit(OpCodes.Ret);
    }

    // ---- statements ----

    void EmitBlock(BlockStmt block, bool newScope = true)
    {
        if (newScope) _scopes.Add(new(StringComparer.Ordinal));
        foreach (var s in block.Body) EmitStmt(s);
        if (newScope) _scopes.RemoveAt(_scopes.Count - 1);
    }

    void EmitStmt(Stmt s)
    {
        switch (s)
        {
            case BlockStmt b: EmitBlock(b); break;
            case EmptyStmt: break;
            case VarDeclStmt d: EmitVarDecl(d); break;
            case ExprStmt e:
                var t = EmitExpr(e.E, needValue: false);
                if (t != GlslType.Void) _il.Emit(OpCodes.Pop);
                break;
            case IfStmt i:
            {
                ExpectType(EmitExpr(i.Cond), GlslType.Bool, i.Line, "if condition");
                var els = _il.DefineLabel();
                var end = _il.DefineLabel();
                _il.Emit(OpCodes.Brfalse, els);
                EmitScoped(i.Then);
                _il.Emit(OpCodes.Br, end);
                _il.MarkLabel(els);
                if (i.Else is not null) EmitScoped(i.Else);
                _il.MarkLabel(end);
                break;
            }
            case ForStmt f:
            {
                _scopes.Add(new(StringComparer.Ordinal));
                if (f.Init is not null) EmitStmt(f.Init);
                var top = _il.DefineLabel();
                var cont = _il.DefineLabel();
                var brk = _il.DefineLabel();
                _il.MarkLabel(top);
                if (f.Cond is not null)
                {
                    ExpectType(EmitExpr(f.Cond), GlslType.Bool, f.Line, "for condition");
                    _il.Emit(OpCodes.Brfalse, brk);
                }
                _loops.Push((brk, cont));
                EmitScoped(f.Body);
                _loops.Pop();
                _il.MarkLabel(cont);
                if (f.Step is not null && EmitExpr(f.Step, needValue: false) != GlslType.Void) _il.Emit(OpCodes.Pop);
                _il.Emit(OpCodes.Br, top);
                _il.MarkLabel(brk);
                _scopes.RemoveAt(_scopes.Count - 1);
                break;
            }
            case WhileStmt w:
            {
                var top = _il.DefineLabel();
                var brk = _il.DefineLabel();
                _il.MarkLabel(top);
                ExpectType(EmitExpr(w.Cond), GlslType.Bool, w.Line, "while condition");
                _il.Emit(OpCodes.Brfalse, brk);
                _loops.Push((brk, top));
                EmitScoped(w.Body);
                _loops.Pop();
                _il.Emit(OpCodes.Br, top);
                _il.MarkLabel(brk);
                break;
            }
            case DoWhileStmt d:
            {
                var top = _il.DefineLabel();
                var cont = _il.DefineLabel();
                var brk = _il.DefineLabel();
                _il.MarkLabel(top);
                _loops.Push((brk, cont));
                EmitScoped(d.Body);
                _loops.Pop();
                _il.MarkLabel(cont);
                ExpectType(EmitExpr(d.Cond), GlslType.Bool, d.Line, "do-while condition");
                _il.Emit(OpCodes.Brtrue, top);
                _il.MarkLabel(brk);
                break;
            }
            case JumpStmt j:
                if (j.Kind == "discard") throw new GlslException("'discard' is not supported", j.Line);
                if (_loops.Count == 0) throw new GlslException($"'{j.Kind}' : statement only allowed in loops", j.Line);
                _il.Emit(OpCodes.Br, j.Kind == "break" ? _loops.Peek().Break : _loops.Peek().Continue);
                break;
            case ReturnStmt r:
                if (r.Value is null)
                {
                    if (_current.Ret != GlslType.Void) throw new GlslException("'return' : non-void function must return a value", r.Line);
                }
                else ExpectType(EmitExpr(r.Value), _current.Ret, r.Line, "return");
                _il.Emit(OpCodes.Ret);
                break;
            default:
                throw new GlslException("unsupported statement", s.Line);
        }
    }

    /// <summary>A statement in a scope of its own (an if / loop body that is not a block may declare a variable).</summary>
    void EmitScoped(Stmt s)
    {
        _scopes.Add(new(StringComparer.Ordinal));
        EmitStmt(s);
        _scopes.RemoveAt(_scopes.Count - 1);
    }

    void EmitVarDecl(VarDeclStmt d)
    {
        var type = TypeOfName(d.TypeName, d.Line);
        if (type == GlslType.Void || type.Base == BaseType.Sampler) throw new GlslException($"'{d.TypeName}' : not allowed for a local variable", d.Line);
        foreach (var v in d.Vars)
        {
            if (_scopes[^1].ContainsKey(v.Name)) throw new GlslException($"'{v.Name}' : redefinition", v.Line);
            var local = _il.DeclareLocal(type.Clr);
            if (v.Init is not null)
            {
                // The initializer cannot see the variable it initializes (GLSL scoping starts after the declarator).
                ExpectType(EmitExpr(v.Init), type, v.Line, "initializer");
                _il.Emit(OpCodes.Stloc, local);
            }
            else if (d.Const) throw new GlslException($"'{v.Name}' : a const variable needs an initializer", v.Line);
            else
            {
                // Locals start at zero (GLSL leaves them undefined); a loop must not see the last iteration's value.
                EmitDefault(type);
                _il.Emit(OpCodes.Stloc, local);
            }
            _scopes[^1][v.Name] = new LocalSym(type, local);
        }
    }

    // ---- helpers ----

    static void ExpectType(GlslType actual, GlslType expected, int line, string what)
    {
        if (actual != expected) throw new GlslException($"'{what}' : cannot convert from '{actual}' to '{expected}'", line);
    }

    Symbol? Lookup(string name)
    {
        for (int i = _scopes.Count - 1; i >= 0; i--)
            if (_scopes[i].TryGetValue(name, out var s)) return s;
        return _globalScope.GetValueOrDefault(name);
    }

    LocalBuilder? Spill(GlslType type)
    {
        var l = _il.DeclareLocal(type.Clr);
        _il.Emit(OpCodes.Stloc, l);
        return l;
    }

    /// <summary>Pushes component <paramref name="i"/> of a spilled value (a scalar gives itself for every component).</summary>
    void LoadComp(LocalBuilder? l, GlslType type, int i)
    {
        if (type.Size == 1) { _il.Emit(OpCodes.Ldloc, l); return; }
        _il.Emit(OpCodes.Ldloca, l);
        _il.Emit(OpCodes.Ldfld, type.Field(i));
    }

    /// <summary>Builds a vector from its components on the stack (nothing to do for a scalar).</summary>
    void Construct(GlslType type)
    {
        if (type.Size > 1) _il.Emit(OpCodes.Newobj, type.Constructor);
    }

    void EmitDefault(GlslType type)
    {
        if (type.Size == 1)
        {
            if (type.Base == BaseType.Float) _il.Emit(OpCodes.Ldc_R4, 0f);
            else _il.Emit(OpCodes.Ldc_I4, 0);
            return;
        }
        var l = _il.DeclareLocal(type.Clr);
        _il.Emit(OpCodes.Ldloca, l);
        _il.Emit(OpCodes.Initobj, type.Clr);
        _il.Emit(OpCodes.Ldloc, l);
    }

    /// <summary>Converts the scalar on the stack from one base type to another (GLSL's constructor conversions).</summary>
    void ConvertScalar(BaseType from, BaseType to)
    {
        if (from == to) return;
        switch (to)
        {
            case BaseType.Float:
                if (from == BaseType.UInt) _il.Emit(OpCodes.Conv_R_Un);
                _il.Emit(OpCodes.Conv_R4);
                break;
            case BaseType.Int:
                if (from == BaseType.Float) _il.Emit(OpCodes.Conv_I4);
                break;
            case BaseType.UInt:
                if (from == BaseType.Float) _il.Emit(OpCodes.Conv_U4);
                break;
            case BaseType.Bool:
                if (from == BaseType.Float) _il.Emit(OpCodes.Ldc_R4, 0f);
                else _il.Emit(OpCodes.Ldc_I4, 0);
                _il.Emit(OpCodes.Ceq);
                _il.Emit(OpCodes.Ldc_I4, 0);
                _il.Emit(OpCodes.Ceq);
                break;
        }
    }

    /// <summary>The type an expression has, without emitting it.</summary>
    GlslType TypeOf(Expr e)
    {
        _il.BeginDry();
        try { return EmitExpr(e); }
        finally { _il.EndDry(); }
    }
}
