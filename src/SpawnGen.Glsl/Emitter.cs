using System.Reflection;
using System.Reflection.Emit;

namespace SpawnGen.Glsl;

/// <summary>
/// An ILGenerator that can be switched off: in dry mode nothing is emitted and no locals or labels are made, so the compiler can walk an expression only to
/// learn its type (overload resolution) with the same code that emits it.
/// </summary>
internal sealed class Emitter(ILGenerator il)
{
    int _dry;

    public bool Dry => _dry > 0;
    public void BeginDry() => _dry++;
    public void EndDry() => _dry--;

    public void Emit(OpCode op) { if (_dry == 0) il.Emit(op); }
    public void Emit(OpCode op, int v) { if (_dry == 0) il.Emit(op, v); }
    public void Emit(OpCode op, float v) { if (_dry == 0) il.Emit(op, v); }
    public void Emit(OpCode op, MethodInfo m) { if (_dry == 0) il.Emit(op, m); }
    public void Emit(OpCode op, ConstructorInfo c) { if (_dry == 0) il.Emit(op, c); }
    public void Emit(OpCode op, FieldInfo f) { if (_dry == 0) il.Emit(op, f); }
    public void Emit(OpCode op, Type t) { if (_dry == 0) il.Emit(op, t); }
    public void Emit(OpCode op, LocalBuilder? l) { if (_dry == 0) il.Emit(op, l!); }
    public void Emit(OpCode op, Label l) { if (_dry == 0) il.Emit(op, l); }
    public void EmitArg(OpCode op, int index) { if (_dry == 0) il.Emit(op, (short)index); }

    public LocalBuilder? DeclareLocal(Type t) => _dry == 0 ? il.DeclareLocal(t) : null;
    public Label DefineLabel() => _dry == 0 ? il.DefineLabel() : default;
    public void MarkLabel(Label l) { if (_dry == 0) il.MarkLabel(l); }
}
