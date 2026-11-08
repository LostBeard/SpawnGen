namespace SpawnGen.Glsl;

internal abstract record Expr(int Line);
internal sealed record LiteralExpr(GlslType Type, object Value, int Line) : Expr(Line);
internal sealed record NameExpr(string Name, int Line) : Expr(Line);
/// <summary>A function call or a constructor (<c>vec3(...)</c>, <c>float(...)</c>).</summary>
internal sealed record CallExpr(string Name, List<Expr> Args, int Line) : Expr(Line);
/// <summary>A swizzle (<c>v.xy</c>, <c>c.rgb</c>).</summary>
internal sealed record FieldExpr(Expr Target, string Field, int Line) : Expr(Line);
internal sealed record IndexExpr(Expr Target, Expr Index, int Line) : Expr(Line);
/// <summary>Prefix operators: - + ! ~ and pre-increment / decrement (<c>++</c>, <c>--</c>).</summary>
internal sealed record UnaryExpr(string Op, Expr Operand, int Line) : Expr(Line);
internal sealed record PostfixExpr(string Op, Expr Operand, int Line) : Expr(Line);
internal sealed record BinaryExpr(string Op, Expr Left, Expr Right, int Line) : Expr(Line);
internal sealed record AssignExpr(string Op, Expr Target, Expr Value, int Line) : Expr(Line);
internal sealed record TernaryExpr(Expr Cond, Expr Then, Expr Else, int Line) : Expr(Line);

internal abstract record Stmt(int Line);
internal sealed record Declarator(string Name, Expr? Init, int Line);
internal sealed record VarDeclStmt(string TypeName, bool Const, List<Declarator> Vars, int Line) : Stmt(Line);
internal sealed record ExprStmt(Expr E, int Line) : Stmt(Line);
internal sealed record BlockStmt(List<Stmt> Body, int Line) : Stmt(Line);
internal sealed record IfStmt(Expr Cond, Stmt Then, Stmt? Else, int Line) : Stmt(Line);
internal sealed record ForStmt(Stmt? Init, Expr? Cond, Expr? Step, Stmt Body, int Line) : Stmt(Line);
internal sealed record WhileStmt(Expr Cond, Stmt Body, int Line) : Stmt(Line);
internal sealed record DoWhileStmt(Stmt Body, Expr Cond, int Line) : Stmt(Line);
internal sealed record JumpStmt(string Kind, int Line) : Stmt(Line); // break, continue, discard
internal sealed record ReturnStmt(Expr? Value, int Line) : Stmt(Line);
internal sealed record EmptyStmt(int Line) : Stmt(Line);

/// <summary>A global declaration: <c>uniform</c>, <c>in</c>, <c>out</c> (with its layout location), <c>const</c> or a plain mutable global.</summary>
internal sealed record GlobalDecl(string Storage, int? Location, string TypeName, List<Declarator> Vars, int Line);
internal sealed record ParamDecl(string Qualifier, string TypeName, string Name, int Line);
internal sealed record FunctionDecl(string ReturnType, string Name, List<ParamDecl> Params, BlockStmt? Body, int Line);
internal sealed record ShaderAst(List<GlobalDecl> Globals, List<FunctionDecl> Functions, List<(string Text, int Line)> Directives);
