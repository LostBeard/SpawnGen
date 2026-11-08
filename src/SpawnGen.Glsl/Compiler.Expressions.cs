using System.Reflection;
using System.Reflection.Emit;

namespace SpawnGen.Glsl;

internal sealed partial class Compiler
{
    /// <summary>Emits an expression's value (unless <paramref name="needValue"/> is false and it is an assignment, which then leaves nothing) and returns its type.</summary>
    GlslType EmitExpr(Expr e, bool needValue = true)
    {
        switch (e)
        {
            case LiteralExpr lit:
                switch (lit.Value)
                {
                    case float f: _il.Emit(OpCodes.Ldc_R4, f); break;
                    case int i: _il.Emit(OpCodes.Ldc_I4, i); break;
                    case uint u: _il.Emit(OpCodes.Ldc_I4, unchecked((int)u)); break;
                    case bool b: _il.Emit(OpCodes.Ldc_I4, b ? 1 : 0); break;
                }
                return lit.Type;
            case NameExpr n: return EmitName(n);
            case FieldExpr f: return EmitSwizzleRead(f);
            case IndexExpr ix: throw new GlslException("indexing is not supported", ix.Line);
            case UnaryExpr u: return EmitUnary(u, needValue);
            case PostfixExpr p: return EmitIncDec(p.Operand, p.Op, prefix: false, needValue, p.Line);
            case BinaryExpr b: return EmitBinary(b);
            case AssignExpr a: return EmitAssign(a, needValue);
            case TernaryExpr t: return EmitTernary(t);
            case CallExpr c: return EmitCall(c);
            default: throw new GlslException("unsupported expression", e.Line);
        }
    }

    GlslType EmitName(NameExpr n)
    {
        var sym = Lookup(n.Name) ?? throw new GlslException($"'{n.Name}' : undeclared identifier", n.Line);
        switch (sym)
        {
            case LocalSym l: _il.Emit(OpCodes.Ldloc, l.Local); break;
            case ArgSym a:
                _il.EmitArg(OpCodes.Ldarg, a.Index);
                if (a.ByRef) _il.Emit(OpCodes.Ldobj, a.Type.Clr);
                break;
            case GlobalSym g:
                _il.EmitArg(OpCodes.Ldarg, 1);
                _il.Emit(OpCodes.Ldfld, g.Field);
                break;
            case ConstSym c:
                ExpectType(EmitExpr(c.Init), c.Type, n.Line, n.Name);
                break;
            case UniformSym u:
                switch (u.Kind)
                {
                    case "sampler": _il.Emit(OpCodes.Ldc_I4, u.Index); break;
                    case "resolution": _il.EmitArg(OpCodes.Ldarg, 0); _il.Emit(OpCodes.Call, Rt(nameof(GlslRuntime.Resolution))); break;
                    case "pass": _il.EmitArg(OpCodes.Ldarg, 0); _il.Emit(OpCodes.Call, Rt(nameof(GlslRuntime.Pass))); break;
                    case "float": _il.EmitArg(OpCodes.Ldarg, 0); _il.Emit(OpCodes.Ldc_I4, u.Index); _il.Emit(OpCodes.Call, Rt(nameof(GlslRuntime.Float))); break;
                    case "int": _il.EmitArg(OpCodes.Ldarg, 0); _il.Emit(OpCodes.Ldc_I4, u.Index); _il.Emit(OpCodes.Call, Rt(nameof(GlslRuntime.Int))); break;
                    case "bool": _il.EmitArg(OpCodes.Ldarg, 0); _il.Emit(OpCodes.Ldc_I4, u.Index); _il.Emit(OpCodes.Call, Rt(nameof(GlslRuntime.Bool))); break;
                    default: EmitDefault(u.Type); break;
                }
                break;
        }
        return sym.Type;
    }

    static MethodInfo Rt(string name) => typeof(GlslRuntime).GetMethod(name)!;

    static int[] SwizzleComponents(string field, GlslType type, int line)
    {
        const string xyzw = "xyzw", rgba = "rgba", stpq = "stpq";
        string? set = xyzw.Contains(field[0]) ? xyzw : rgba.Contains(field[0]) ? rgba : stpq.Contains(field[0]) ? stpq : null;
        if (set is null || field.Length > 4 || type.Base is BaseType.Void or BaseType.Sampler) throw new GlslException($"'{field}' : illegal vector field selection", line);
        var comps = new int[field.Length];
        for (int i = 0; i < field.Length; i++)
        {
            int c = set.IndexOf(field[i]);
            if (c < 0 || c >= type.Size) throw new GlslException($"'{field}' : vector field selection out of range", line);
            comps[i] = c;
        }
        return comps;
    }

    GlslType EmitSwizzleRead(FieldExpr f)
    {
        var t = EmitExpr(f.Target);
        var comps = SwizzleComponents(f.Field, t, f.Line);
        var l = Spill(t);
        foreach (var c in comps) LoadComp(l, t, c);
        var result = t.WithSize(comps.Length);
        Construct(result);
        return result;
    }

    // ---- operators ----

    GlslType EmitUnary(UnaryExpr u, bool needValue)
    {
        if (u.Op is "++" or "--") return EmitIncDec(u.Operand, u.Op, prefix: true, needValue, u.Line);
        var t = EmitExpr(u.Operand);
        switch (u.Op)
        {
            case "+":
                if (!t.IsNumeric) throw new GlslException($"'+' : wrong operand type '{t}'", u.Line);
                return t;
            case "-":
            {
                if (!t.IsNumeric) throw new GlslException($"'-' : wrong operand type '{t}'", u.Line);
                var l = Spill(t);
                for (int i = 0; i < t.Size; i++) { LoadComp(l, t, i); _il.Emit(OpCodes.Neg); }
                Construct(t);
                return t;
            }
            case "!":
                if (t != GlslType.Bool) throw new GlslException($"'!' : wrong operand type '{t}'", u.Line);
                _il.Emit(OpCodes.Ldc_I4, 0);
                _il.Emit(OpCodes.Ceq);
                return t;
            case "~":
            {
                if (!t.IsIntegral) throw new GlslException($"'~' : wrong operand type '{t}'", u.Line);
                var l = Spill(t);
                for (int i = 0; i < t.Size; i++) { LoadComp(l, t, i); _il.Emit(OpCodes.Not); }
                Construct(t);
                return t;
            }
        }
        throw new GlslException($"'{u.Op}' : unsupported operator", u.Line);
    }

    GlslType EmitBinary(BinaryExpr b)
    {
        switch (b.Op)
        {
            case "&&" or "||":
            {
                var lt = EmitExpr(b.Left);
                if (lt != GlslType.Bool) throw new GlslException($"'{b.Op}' : wrong operand types", b.Line);
                var shortcut = _il.DefineLabel();
                var end = _il.DefineLabel();
                _il.Emit(b.Op == "&&" ? OpCodes.Brfalse : OpCodes.Brtrue, shortcut);
                if (EmitExpr(b.Right) != GlslType.Bool) throw new GlslException($"'{b.Op}' : wrong operand types", b.Line);
                _il.Emit(OpCodes.Br, end);
                _il.MarkLabel(shortcut);
                _il.Emit(OpCodes.Ldc_I4, b.Op == "&&" ? 0 : 1);
                _il.MarkLabel(end);
                return GlslType.Bool;
            }
            case "^^":
                if (EmitExpr(b.Left) != GlslType.Bool || EmitExpr(b.Right) != GlslType.Bool) throw new GlslException("'^^' : wrong operand types", b.Line);
                _il.Emit(OpCodes.Xor);
                return GlslType.Bool;
        }
        var left = EmitExpr(b.Left);
        var l = Spill(left);
        var right = EmitExpr(b.Right);
        var r = Spill(right);
        return EmitArithmetic(b.Op, l, left, r, right, b.Line);
    }

    /// <summary>The binary operators on two spilled operands (also used by compound assignment).</summary>
    GlslType EmitArithmetic(string op, LocalBuilder? l, GlslType left, LocalBuilder? r, GlslType right, int line)
    {
        GlslType Fail() => throw new GlslException($"'{op}' : wrong operand types - no operation '{op}' exists that takes a left-hand operand of type '{left}' and a right operand of type '{right}' (or there is no acceptable conversion)", line);
        switch (op)
        {
            case "+" or "-" or "*" or "/" or "%" or "&" or "|" or "^":
            {
                if (!left.IsNumeric || left.Base != right.Base || (left.Size != right.Size && left.Size != 1 && right.Size != 1)) return Fail();
                if (op is "%" or "&" or "|" or "^" && !left.IsIntegral) return Fail();
                var result = left.WithSize(Math.Max(left.Size, right.Size));
                bool unsigned = left.Base == BaseType.UInt;
                var code = op switch
                {
                    "+" => OpCodes.Add, "-" => OpCodes.Sub, "*" => OpCodes.Mul,
                    "/" => unsigned ? OpCodes.Div_Un : OpCodes.Div, "%" => unsigned ? OpCodes.Rem_Un : OpCodes.Rem,
                    "&" => OpCodes.And, "|" => OpCodes.Or, _ => OpCodes.Xor,
                };
                for (int i = 0; i < result.Size; i++) { LoadComp(l, left, i); LoadComp(r, right, i); _il.Emit(code); }
                Construct(result);
                return result;
            }
            case "<<" or ">>":
            {
                if (!left.IsIntegral || !right.IsIntegral || (right.Size != 1 && right.Size != left.Size)) return Fail();
                var code = op == "<<" ? OpCodes.Shl : left.Base == BaseType.UInt ? OpCodes.Shr_Un : OpCodes.Shr;
                for (int i = 0; i < left.Size; i++) { LoadComp(l, left, i); LoadComp(r, right, i); _il.Emit(code); }
                Construct(left);
                return left;
            }
            case "<" or ">" or "<=" or ">=":
            {
                if (left != right || !left.IsScalar || !left.IsNumeric) return Fail();
                LoadComp(l, left, 0); LoadComp(r, right, 0);
                bool f = left.Base == BaseType.Float, u = left.Base == BaseType.UInt;
                switch (op)
                {
                    case "<": _il.Emit(u ? OpCodes.Clt_Un : OpCodes.Clt); break;
                    case ">": _il.Emit(u ? OpCodes.Cgt_Un : OpCodes.Cgt); break;
                    // a <= b is !(a > b), with NaN false: cgt.un counts an unordered pair as greater.
                    case "<=": _il.Emit(f || u ? OpCodes.Cgt_Un : OpCodes.Cgt); _il.Emit(OpCodes.Ldc_I4, 0); _il.Emit(OpCodes.Ceq); break;
                    default: _il.Emit(f || u ? OpCodes.Clt_Un : OpCodes.Clt); _il.Emit(OpCodes.Ldc_I4, 0); _il.Emit(OpCodes.Ceq); break;
                }
                return GlslType.Bool;
            }
            case "==" or "!=":
            {
                if (left != right || left.Base is BaseType.Void or BaseType.Sampler) return Fail();
                for (int i = 0; i < left.Size; i++)
                {
                    LoadComp(l, left, i); LoadComp(r, right, i); _il.Emit(OpCodes.Ceq);
                    if (i > 0) _il.Emit(OpCodes.And);
                }
                if (op == "!=") { _il.Emit(OpCodes.Ldc_I4, 0); _il.Emit(OpCodes.Ceq); }
                return GlslType.Bool;
            }
        }
        return Fail();
    }

    GlslType EmitTernary(TernaryExpr t)
    {
        if (EmitExpr(t.Cond) != GlslType.Bool) throw new GlslException("'?:' : boolean expression expected", t.Line);
        var els = _il.DefineLabel();
        var end = _il.DefineLabel();
        _il.Emit(OpCodes.Brfalse, els);
        var a = EmitExpr(t.Then);
        _il.Emit(OpCodes.Br, end);
        _il.MarkLabel(els);
        var b = EmitExpr(t.Else);
        if (a != b) throw new GlslException($"'?:' : wrong operand types - no operation '?:' exists that takes '{a}' and '{b}'", t.Line);
        _il.MarkLabel(end);
        return a;
    }

    // ---- l-values and assignment ----

    /// <summary>Something that can be assigned: a variable (local, parameter, global) or a swizzle of one.</summary>
    sealed record LValue(Symbol Var, int[]? Components, GlslType Type);

    LValue ResolveLValue(Expr e)
    {
        switch (e)
        {
            case NameExpr n:
            {
                var sym = Lookup(n.Name) ?? throw new GlslException($"'{n.Name}' : undeclared identifier", n.Line);
                if (sym is not (LocalSym or ArgSym or GlobalSym)) throw new GlslException($"'{n.Name}' : l-value required (can't modify a const or uniform)", n.Line);
                return new LValue(sym, null, sym.Type);
            }
            case FieldExpr f:
            {
                var inner = ResolveLValue(f.Target);
                var comps = SwizzleComponents(f.Field, inner.Type, f.Line);
                if (comps.Distinct().Count() != comps.Length) throw new GlslException($"'{f.Field}' : l-value of swizzle cannot have duplicate components", f.Line);
                var mapped = inner.Components is null ? comps : comps.Select(c => inner.Components[c]).ToArray();
                return new LValue(inner.Var, mapped, inner.Type.WithSize(comps.Length));
            }
            default:
                throw new GlslException("l-value required", e.Line);
        }
    }

    void LoadLValue(LValue lv)
    {
        var whole = lv.Var.Type;
        switch (lv.Var)
        {
            case LocalSym l: _il.Emit(OpCodes.Ldloc, l.Local); break;
            case ArgSym a: _il.EmitArg(OpCodes.Ldarg, a.Index); if (a.ByRef) _il.Emit(OpCodes.Ldobj, a.Type.Clr); break;
            case GlobalSym g: _il.EmitArg(OpCodes.Ldarg, 1); _il.Emit(OpCodes.Ldfld, g.Field); break;
        }
        if (lv.Components is null) return;
        var t = Spill(whole);
        foreach (var c in lv.Components) LoadComp(t, whole, c);
        Construct(lv.Type);
    }

    /// <summary>Pushes the address of the l-value's variable (for a component store).</summary>
    void AddressOf(Symbol v)
    {
        switch (v)
        {
            case LocalSym l: _il.Emit(OpCodes.Ldloca, l.Local); break;
            case ArgSym a: _il.EmitArg(a.ByRef ? OpCodes.Ldarg : OpCodes.Ldarga, a.Index); break;
            case GlobalSym g: _il.EmitArg(OpCodes.Ldarg, 1); _il.Emit(OpCodes.Ldflda, g.Field); break;
        }
    }

    /// <summary>Stores the spilled value <paramref name="value"/> (of the l-value's type) into the l-value.</summary>
    void StoreLValue(LValue lv, LocalBuilder? value)
    {
        if (lv.Components is null)
        {
            switch (lv.Var)
            {
                case LocalSym l: _il.Emit(OpCodes.Ldloc, value); _il.Emit(OpCodes.Stloc, l.Local); break;
                case ArgSym { ByRef: false } a: _il.Emit(OpCodes.Ldloc, value); _il.EmitArg(OpCodes.Starg, a.Index); break;
                case ArgSym a: _il.EmitArg(OpCodes.Ldarg, a.Index); _il.Emit(OpCodes.Ldloc, value); _il.Emit(OpCodes.Stobj, a.Type.Clr); break;
                case GlobalSym g: _il.EmitArg(OpCodes.Ldarg, 1); _il.Emit(OpCodes.Ldloc, value); _il.Emit(OpCodes.Stfld, g.Field); break;
            }
            return;
        }
        var whole = lv.Var.Type;
        if (whole.Size == 1)
        {
            // A scalar's .x is the scalar itself.
            StoreLValue(lv with { Components = null, Type = whole }, value);
            return;
        }
        for (int k = 0; k < lv.Components.Length; k++)
        {
            AddressOf(lv.Var);
            LoadComp(value, lv.Type, k);
            _il.Emit(OpCodes.Stfld, whole.Field(lv.Components[k]));
        }
    }

    GlslType EmitAssign(AssignExpr a, bool needValue)
    {
        var lv = ResolveLValue(a.Target);
        LocalBuilder? result;
        if (a.Op == "=")
        {
            ExpectType(EmitExpr(a.Value), lv.Type, a.Line, "assign");
            result = Spill(lv.Type);
        }
        else
        {
            LoadLValue(lv);
            var cur = Spill(lv.Type);
            var rt = EmitExpr(a.Value);
            var r = Spill(rt);
            var t = EmitArithmetic(a.Op[..^1], cur, lv.Type, r, rt, a.Line);
            if (t != lv.Type) throw new GlslException($"'{a.Op}' : cannot convert from '{t}' to '{lv.Type}'", a.Line);
            result = Spill(t);
        }
        StoreLValue(lv, result);
        if (!needValue) return GlslType.Void;
        _il.Emit(OpCodes.Ldloc, result);
        return lv.Type;
    }

    GlslType EmitIncDec(Expr target, string op, bool prefix, bool needValue, int line)
    {
        var lv = ResolveLValue(target);
        if (!lv.Type.IsNumeric) throw new GlslException($"'{op}' : wrong operand type '{lv.Type}'", line);
        LoadLValue(lv);
        var old = Spill(lv.Type);
        for (int i = 0; i < lv.Type.Size; i++)
        {
            LoadComp(old, lv.Type, i);
            if (lv.Type.Base == BaseType.Float) _il.Emit(OpCodes.Ldc_R4, 1f); else _il.Emit(OpCodes.Ldc_I4, 1);
            _il.Emit(op == "++" ? OpCodes.Add : OpCodes.Sub);
        }
        Construct(lv.Type);
        var updated = Spill(lv.Type);
        StoreLValue(lv, updated);
        if (!needValue) return GlslType.Void;
        _il.Emit(OpCodes.Ldloc, prefix ? updated : old);
        return lv.Type;
    }
}
